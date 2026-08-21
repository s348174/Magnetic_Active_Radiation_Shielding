"""
Kernels.py
----------
Permutation-invariant kernel for arrays of `n_coils` identical, indistinguishable
coils, as derived in the thesis section "A Permutation-Invariant Kernel for Coil
Array Configurations".

Expected input layout (per configuration, last dimension):
    [x_1, y_1, z_1, nx_1, ny_1, nz_1,  x_2, y_2, z_2, nx_2, ny_2, nz_2,  ...]
i.e. `n_coils` contiguous blocks of size `per_coil_dim` (default 6 = 3 Cartesian
position + 3 normal-versor components). This is what `map_set` in BoUtils.py
must produce from the raw (r, theta, phi, alpha, beta) x 10 normalized input.

Two kernels are provided:
    - CoilSetKernel:    sum_{i,j} k_pos(p_i,p_j') * k_vMF(n_i,n_j')   (mean embedding)
    - CoilMomentKernel: RBF on the invariant (1st, 2nd) moment embedding Psi(X)

Both are permutation-invariant and PSD by construction (see thesis proofs).
Combine them with GPyTorch's kernel algebra, e.g.:

    covar_module = ScaleKernel(CoilSetKernel(n_coils=10)) \
                 + ScaleKernel(CoilMomentKernel(n_coils=10))
"""

import torch
import torch.nn.functional as F
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive

from input import K


class CoilSetKernel(Kernel):
    """
    Permutation-invariant "mean embedding" kernel:
        K(X,X') = sum_{i=1}^{n} sum_{j=1}^{n} k_pos(p_i,p_j') * k_vMF(n_i,n_j')
    with
        k_pos(p,p')  = exp(-||p-p'||^2 / (2 l^2))                  (Gaussian RBF)
        k_vMF(n,n')  = exp(kappa * (n . n' - 1))                    (von Mises-Fisher)

    IMPORTANT: lengthscale and kappa are SCALAR (not ARD) and shared across all
    coils. Using per-coil / per-dimension (ARD) lengthscales would break
    permutation invariance, since it would let the model tell coils apart by
    which "slot" they occupy in the input vector.
    """

    is_stationary = False  # not translation-invariant across coil-block boundaries in general

    def __init__(
        self,
        n_coils: int = K,
        per_coil_dim: int = 5,
        pos_dim: int = 3,
        normalize: bool = True,
        lengthscale_constraint=None,
        kappa_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_coils = n_coils
        self.per_coil_dim = per_coil_dim
        self.pos_dim = pos_dim
        self.normalize = normalize

        lengthscale_constraint = lengthscale_constraint or Positive()
        kappa_constraint = kappa_constraint or Positive()

        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )
        self.register_constraint("raw_lengthscale", lengthscale_constraint)

        self.register_parameter(
            name="raw_kappa",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )
        self.register_constraint("raw_kappa", kappa_constraint)

    # -- lengthscale property -------------------------------------------------
    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))

    # -- kappa property ---------------------------------------------------------
    @property
    def kappa(self):
        return self.raw_kappa_constraint.transform(self.raw_kappa)

    @kappa.setter
    def kappa(self, value):
        value = torch.as_tensor(value).to(self.raw_kappa)
        self.initialize(raw_kappa=self.raw_kappa_constraint.inverse_transform(value))

    # -- helpers ------------------------------------------------------------
    def _split(self, x):
        """(..., N, n_coils*per_coil_dim) -> pos (..., N, n_coils, pos_dim), n (..., N, n_coils, dir_dim)"""
        *batch, N, D = x.shape
        assert D == self.n_coils * self.per_coil_dim, (
            f"CoilSetKernel expected last dim {self.n_coils * self.per_coil_dim}, got {D}. "
            f"Did you pass map_set(X) rather than map_cart(X)?"
        )
        x = x.view(*batch, N, self.n_coils, self.per_coil_dim)
        pos = x[..., : self.pos_dim]
        nrm = F.normalize(x[..., self.pos_dim:], dim=-1, eps=1e-8)
        return pos, nrm

    def _raw_forward(self, x1, x2, diag=False):
        pos1, n1 = self._split(x1)  # (..., N1, n_coils, pos_dim/dir_dim)
        pos2, n2 = self._split(x2)  # (..., N2, n_coils, pos_dim/dir_dim)

        ell = self.lengthscale  # (..., 1, 1)
        kappa = self.kappa      # (..., 1, 1)

        if diag:
            # x1, x2 assumed paired (same N, index-aligned)
            diff = pos1.unsqueeze(-2) - pos2.unsqueeze(-3)          # (..., N, n_coils, n_coils, pos_dim)
            sq_dist = diff.pow(2).sum(-1)                            # (..., N, n_coils, n_coils)
            ell_b = ell.unsqueeze(-1)                                 # broadcast over coil x coil
            pos_k = torch.exp(-sq_dist / (2 * ell_b ** 2))

            cos_sim = (n1.unsqueeze(-2) * n2.unsqueeze(-3)).sum(-1)  # (..., N, n_coils, n_coils)
            kappa_b = kappa.unsqueeze(-1)
            vmf_k = torch.exp(kappa_b * (cos_sim - 1.0))

            K_diag = (pos_k * vmf_k).sum(dim=(-2, -1))               # (..., N)
            return K_diag

        # Full N1 x N2 covariance via flatten + cdist (avoids an explicit 4D loop)
        *B1, N1, nc, pd = pos1.shape
        *B2, N2, _, dd = n1.shape[:-1] + (n1.shape[-1],)

        pos1_flat = pos1.reshape(*B1, N1 * self.n_coils, pd)
        pos2_flat = pos2.reshape(*pos2.shape[:-3], pos2.shape[-3] * self.n_coils, pd)
        n1_flat = n1.reshape(*B1, N1 * self.n_coils, n1.shape[-1])
        n2_flat = n2.reshape(*n2.shape[:-3], n2.shape[-3] * self.n_coils, n2.shape[-1])

        sq_dist = torch.cdist(pos1_flat, pos2_flat, p=2).pow(2)       # (..., N1*n, N2*n)
        ell_flat = ell.view(*self.batch_shape, 1, 1)
        pos_k = torch.exp(-sq_dist / (2 * ell_flat ** 2))

        cos_sim = n1_flat @ n2_flat.transpose(-2, -1)                  # (..., N1*n, N2*n)
        kappa_flat = kappa.view(*self.batch_shape, 1, 1)
        vmf_k = torch.exp(kappa_flat * (cos_sim - 1.0))

        coil_k = pos_k * vmf_k
        N2_ = pos2.shape[-3]
        coil_k = coil_k.view(*B1, N1, self.n_coils, N2_, self.n_coils)
        K = coil_k.sum(dim=(-3, -1))                                   # (..., N1, N2)
        return K

    def forward(self, x1, x2, diag=False, **params):
        if not self.normalize:
            return self._raw_forward(x1, x2, diag=diag)

        if diag:
            # normalized diagonal is trivially 1 by construction
            return torch.ones(*x1.shape[:-1], device=x1.device, dtype=x1.dtype)

        K = self._raw_forward(x1, x2, diag=False)
        diag1 = self._raw_forward(x1, x1, diag=True)  # (..., N1)
        diag2 = self._raw_forward(x2, x2, diag=True)  # (..., N2)
        denom = torch.sqrt(diag1.unsqueeze(-1) * diag2.unsqueeze(-2)).clamp_min(1e-12)
        return K / denom


class CoilMomentKernel(Kernel):
    """
    Second-order-moment refinement kernel:
        Psi(X) = (mu_1(X), vec(mu_2(X)))
        K(X,X') = exp(-||Psi(X)-Psi(X')||^2 / (2 l_mom^2))
    PSD and permutation-invariant since Psi is a fixed, permutation-invariant
    map (unweighted mean over the coil axis) composed with an ordinary RBF.
    """

    is_stationary = False

    def __init__(self, n_coils: int = K, per_coil_dim: int = 5,
                 lengthscale_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.n_coils = n_coils
        self.per_coil_dim = per_coil_dim

        lengthscale_constraint = lengthscale_constraint or Positive()
        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )
        self.register_constraint("raw_lengthscale", lengthscale_constraint)

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))

    def _psi(self, x):
        *batch, N, D = x.shape
        x = x.view(*batch, N, self.n_coils, self.per_coil_dim)
        mu1 = x.mean(dim=-2)                                              # (..., N, d)
        outer = x.unsqueeze(-1) * x.unsqueeze(-2)                         # (..., N, n_coils, d, d)
        mu2 = outer.mean(dim=-3).reshape(*batch, N, self.per_coil_dim ** 2)  # (..., N, d^2)
        return torch.cat([mu1, mu2], dim=-1)                              # (..., N, d + d^2)

    def forward(self, x1, x2, diag=False, **params):
        psi1 = self._psi(x1)
        psi2 = self._psi(x2)
        ell = self.lengthscale

        if diag:
            sq_dist = (psi1 - psi2).pow(2).sum(-1)
            ell_b = ell.squeeze(-1)
            return torch.exp(-sq_dist / (2 * ell_b ** 2))

        sq_dist = torch.cdist(psi1, psi2, p=2).pow(2)
        ell2 = ell.view(*self.batch_shape, 1, 1)
        return torch.exp(-sq_dist / (2 * ell2 ** 2))