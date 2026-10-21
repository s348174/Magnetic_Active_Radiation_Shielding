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
        per_coil_dim: int = 6,
        pos_dim: int = 3,
        normalize: bool = True,
        chunk_size: int = 200,
        lengthscale_constraint=None,
        kappa_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_coils = n_coils
        self.per_coil_dim = per_coil_dim
        self.pos_dim = pos_dim
        self.normalize = normalize
        self.chunk_size = chunk_size

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

    def _diag_forward(self, x1, x2):
        """x1, x2 must have the SAME shape (..., N, D) -- paired, index-aligned."""
        pos1, n1 = self._split(x1)   # (..., N, n_coils, pos_dim/dir_dim)
        pos2, n2 = self._split(x2)

        ell = self.lengthscale       # (..., 1, 1)
        kappa = self.kappa           # (..., 1, 1)

        diff = pos1.unsqueeze(-2) - pos2.unsqueeze(-3)            # (..., N, n_coils, n_coils, pos_dim)
        sq_dist = diff.pow(2).sum(-1)                              # (..., N, n_coils, n_coils)
        ell_b = ell.unsqueeze(-1)                                   # -> (..., 1, 1, 1), broadcasts fine
        pos_k = torch.exp(-sq_dist / (2 * ell_b ** 2))

        cos_sim = (n1.unsqueeze(-2) * n2.unsqueeze(-3)).sum(-1)    # (..., N, n_coils, n_coils)
        kappa_b = kappa.unsqueeze(-1)
        vmf_k = torch.exp(kappa_b * (cos_sim - 1.0))

        K_diag = (pos_k * vmf_k).sum(dim=(-2, -1))                 # (..., N)
        return K_diag

    def _full_forward(self, x1, x2, chunk_size):
        """Returns (..., N1, N2). Chunks over N2 to bound peak memory."""
        pos1, n1 = self._split(x1)   # (..., N1, n_coils, pos_dim/dir_dim)
        pos2, n2 = self._split(x2)   # (..., N2, n_coils, pos_dim/dir_dim)

        # batch_prefix = everything before the (N, n_coils, dim) trailing dims
        batch_prefix = pos1.shape[:-3]
        N1 = pos1.shape[-3]
        N2 = pos2.shape[-3]

        pos1_flat = pos1.reshape(*batch_prefix, N1 * self.n_coils, pos1.shape[-1])
        n1_flat = n1.reshape(*batch_prefix, N1 * self.n_coils, n1.shape[-1])

        ell_flat = self.lengthscale.view(*self.batch_shape, 1, 1)
        kappa_flat = self.kappa.view(*self.batch_shape, 1, 1)

        out_chunks = []
        for start in range(0, N2, chunk_size):
            end = min(start + chunk_size, N2)
            width = end - start

            pos2_c = pos2[..., start:end, :, :].reshape(*batch_prefix, width * self.n_coils, pos2.shape[-1])
            n2_c = n2[..., start:end, :, :].reshape(*batch_prefix, width * self.n_coils, n2.shape[-1])

            sq_dist = torch.cdist(pos1_flat, pos2_c, p=2).pow(2)     # (..., N1*nc, width*nc)
            pos_k = torch.exp(-sq_dist / (2 * ell_flat ** 2))
            del sq_dist

            cos_sim = n1_flat @ n2_c.transpose(-2, -1)                # (..., N1*nc, width*nc)
            vmf_k = torch.exp(kappa_flat * (cos_sim - 1.0))
            del cos_sim

            coil_k = pos_k * vmf_k                                    # (..., N1*nc, width*nc)
            del pos_k, vmf_k

            coil_k = coil_k.reshape(*batch_prefix, N1, self.n_coils, width, self.n_coils)
            chunk_out = coil_k.sum(dim=(-3, -1))                       # (..., N1, width)
            del coil_k
            out_chunks.append(chunk_out)

        K = out_chunks[0] if len(out_chunks) == 1 else torch.cat(out_chunks, dim=-1)  # (..., N1, N2)

        expected_shape = (*batch_prefix, N1, N2)
        if tuple(K.shape) != expected_shape:
            raise RuntimeError(
                f"CoilSetKernel._full_forward produced shape {tuple(K.shape)}, "
                f"expected {expected_shape}. This indicates a reshape bug -- "
                f"check batch_prefix={batch_prefix}, N1={N1}, N2={N2}, n_coils={self.n_coils}."
            )
        return K

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            if self.normalize:
                return torch.ones(*x1.shape[:-1], device=x1.device, dtype=x1.dtype)
            return self._diag_forward(x1, x2)

        K = self._full_forward(x1, x2, chunk_size=self.chunk_size)
        if not self.normalize:
            return K

        diag1 = self._diag_forward(x1, x1)  # (..., N1)
        diag2 = self._diag_forward(x2, x2)  # (..., N2)
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

    def __init__(self, n_coils: int = 10, per_coil_dim: int = 6,
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