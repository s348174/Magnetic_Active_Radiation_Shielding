"""
Kernels.py  (optimized)
-----------------------
Permutation-invariant kernel for arrays of `n_coils` identical, indistinguishable
coils (see thesis section "A Permutation-Invariant Kernel for Coil Array
Configurations").

Expected input layout (per configuration, last dimension):
    [x_1, y_1, z_1, nx_1, ny_1, nz_1,  x_2, y_2, z_2, nx_2, ny_2, nz_2,  ...]
i.e. `n_coils` contiguous blocks of `per_coil_dim` (=6: Cartesian position +
unit normal). This is what `map_coilwise` in BoUtils.py produces.

Math (unchanged):
    K(X,X') = sum_i sum_j k_pos(p_i,p_j') * k_vMF(n_i,n_j')

Key identity used for speed [NEW]:
    For unit normals,  kappa*(n.n' - 1) = -kappa*||n - n'||^2 / 2, hence
        k_pos * k_vMF = exp(-||z - z'||^2 / 2),   z = [ p/l , sqrt(kappa)*n ]
    i.e. the per-coil kernel is a plain Gaussian RBF on a 6-dim scaled vector z.
    Everything below follows from that.

Two evaluation modes (selected by `n_features`):
    n_features=None  -> EXACT. Same values as before (to ~1e-12), but computed
                        with ONE matmul + ONE exp instead of cdist + 2 exps + products.
    n_features=F     -> Random Fourier Features. Approximate, but exactly PSD,
                        exactly permutation-invariant, exactly unit diagonal, and
                        ~O(N*K*F) instead of O(N^2*K^2). Much faster and far less
                        memory for fitting (autograd stores N x F, not (NK)^2).
"""

import math
import torch
import torch.nn.functional as F
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
from input import K


class CoilSetKernel(Kernel):
    """
    IMPORTANT: lengthscale and kappa are SCALAR (not ARD) and shared across all
    coils. Per-coil / per-dimension lengthscales would break permutation invariance.
    """

    is_stationary = False

    def __init__(
        self,
        n_coils: int = K,
        per_coil_dim: int = 6,
        pos_dim: int = 3,
        normalize: bool = True,
        chunk_size: int = 200,
        n_features=None,          # [NEW] None = exact; int = number of random Fourier features
        rff_seed: int = 0,        # [NEW] fixes the random feature map (must NOT change during a run)
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
        self.n_features = n_features

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

        # [NEW] Fixed random feature map for the RBF on z (d = pos_dim + dir_dim).
        # Buffers (not parameters): drawn once, never trained, moved with .to(device).
        if n_features is not None:
            d = per_coil_dim  # pos_dim + (per_coil_dim - pos_dim)
            g = torch.Generator().manual_seed(rff_seed)
            W = torch.randn(d, n_features, generator=g, dtype=torch.double)
            b = 2.0 * math.pi * torch.rand(n_features, generator=g, dtype=torch.double)
            self.register_buffer("rff_W", W)
            self.register_buffer("rff_b", b)

    # -- hyperparameter properties (unchanged) ---------------------------------
    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))

    @property
    def kappa(self):
        return self.raw_kappa_constraint.transform(self.raw_kappa)

    @kappa.setter
    def kappa(self, value):
        value = torch.as_tensor(value).to(self.raw_kappa)
        self.initialize(raw_kappa=self.raw_kappa_constraint.inverse_transform(value))

    # -- helpers ----------------------------------------------------------------
    def _split(self, x):
        """(..., N, n_coils*per_coil_dim) -> pos (..., N, n_coils, pos_dim), unit normals (..., N, n_coils, dir_dim)"""
        *batch, N, D = x.shape
        assert D == self.n_coils * self.per_coil_dim, (
            f"CoilSetKernel expected last dim {self.n_coils * self.per_coil_dim}, got {D}. "
            f"Did you pass map_coilwise(X)?"
        )
        x = x.reshape(*batch, N, self.n_coils, self.per_coil_dim)   # [CHANGED] reshape (view fails on non-contiguous input)
        pos = x[..., : self.pos_dim]
        nrm = F.normalize(x[..., self.pos_dim:], dim=-1, eps=1e-8)
        return pos, nrm

    def _coil_z(self, x):
        """[NEW] z = [p/l, sqrt(kappa)*n]  -> (..., N, n_coils, per_coil_dim). k_coil = exp(-|z-z'|^2/2)."""
        pos, nrm = self._split(x)
        inv_ell = 1.0 / self.lengthscale.unsqueeze(-1)
        sqrt_kappa = self.kappa.clamp_min(1e-12).sqrt().unsqueeze(-1)
        return torch.cat([pos * inv_ell, nrm * sqrt_kappa], dim=-1)

    @staticmethod
    def _augment(z_flat, first: bool):
        """[NEW] Fold the -|z|^2/2 terms into the matmul:  A @ B^T = z.z' - |z|^2/2 - |z'|^2/2 = -|z-z'|^2/2."""
        h = -0.5 * z_flat.pow(2).sum(-1, keepdim=True)
        one = torch.ones_like(h)
        return torch.cat([z_flat, h, one] if first else [z_flat, one, h], dim=-1)

    # -- exact path -------------------------------------------------------------
    def _full_exact(self, z1, z2):
        """[CHANGED] one matmul + one exp per chunk (was: cdist, 2x exp, cos-sim matmul, product)."""
        nc, d = self.n_coils, z1.shape[-1]
        N1, N2 = z1.shape[-3], z2.shape[-3]
        A = self._augment(z1.reshape(*z1.shape[:-3], N1 * nc, d), first=True)

        out = []
        for s in range(0, N2, self.chunk_size):
            e = min(s + self.chunk_size, N2)
            zc = z2[..., s:e, :, :]
            B = self._augment(zc.reshape(*zc.shape[:-3], (e - s) * nc, d), first=False)
            E = torch.exp(A @ B.transpose(-2, -1))                      # (..., N1*nc, w*nc)
            E = E.reshape(*E.shape[:-2], N1, nc, e - s, nc).sum(dim=(-3, -1))   # (..., N1, w)
            out.append(E)
        Kmat = out[0] if len(out) == 1 else torch.cat(out, dim=-1)
        assert Kmat.shape[-2:] == (N1, N2), f"shape bug: {tuple(Kmat.shape)} vs ({N1},{N2})"
        return Kmat

    def _diag_exact(self, z1, z2):
        """Paired (index-aligned) diagonal, (..., N). Uses the same Gram identity, no (N,K,K,d) tensor."""
        G = z1 @ z2.transpose(-1, -2)                                     # (..., N, nc, nc)
        h1 = -0.5 * z1.pow(2).sum(-1).unsqueeze(-1)
        h2 = -0.5 * z2.pow(2).sum(-1).unsqueeze(-2)
        return torch.exp(G + h1 + h2).sum(dim=(-2, -1))

    # -- random-Fourier-feature path [NEW] --------------------------------------
    def _phi(self, z):
        """Phi(X) = sum_i phi(z_i), phi(z)=sqrt(2/F) cos(W^T z + b).  (..., N, nc, d) -> (..., N, F)
        Sum over coils => exactly permutation-invariant. <Phi(X),Phi(X')> ~= K(X,X')."""
        W, b = self.rff_W.to(z), self.rff_b.to(z)
        scale = math.sqrt(2.0 / self.n_features)
        parts = []
        for s in range(0, z.shape[-3], self.chunk_size):
            zc = z[..., s:s + self.chunk_size, :, :]
            parts.append(scale * torch.cos(zc @ W + b).sum(dim=-2))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-2)

    # -- forward ----------------------------------------------------------------
    def forward(self, x1, x2, diag=False, **params):
        if diag and self.normalize:
            return torch.ones(*x1.shape[:-1], device=x1.device, dtype=x1.dtype)

        z1 = self._coil_z(x1)
        # [NEW] K(X,X) is the common case during fitting: reuse features instead of recomputing
        same = (x1 is x2) or (x1.shape == x2.shape and torch.equal(x1, x2))
        z2 = z1 if same else self._coil_z(x2)

        if self.n_features is not None:                                   # ---- RFF path
            p1 = self._phi(z1)
            p2 = p1 if same else self._phi(z2)
            if diag:
                return (p1 * p2).sum(-1)
            if self.normalize:                                            # cosine similarity of feature vectors
                p1 = F.normalize(p1, dim=-1, eps=1e-12)
                p2 = p1 if same else F.normalize(p2, dim=-1, eps=1e-12)
            return p1 @ p2.transpose(-2, -1)

        if diag:                                                          # ---- exact path
            return self._diag_exact(z1, z2)
        Kmat = self._full_exact(z1, z2)
        if not self.normalize:
            return Kmat
        d1 = self._diag_exact(z1, z1)
        d2 = d1 if same else self._diag_exact(z2, z2)
        denom = torch.sqrt(d1.unsqueeze(-1) * d2.unsqueeze(-2)).clamp_min(1e-12)
        return Kmat / denom


class CoilMomentKernel(Kernel):
    """
    Second-order-moment refinement kernel (unchanged except view -> reshape).
        Psi(X) = (mu_1(X), vec(mu_2(X)));  K = exp(-||Psi(X)-Psi(X')||^2 / (2 l^2))
    """

    is_stationary = False

    def __init__(self, n_coils: int = K, per_coil_dim: int = 6,
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
        x = x.reshape(*batch, N, self.n_coils, self.per_coil_dim)
        mu1 = x.mean(dim=-2)
        outer = x.unsqueeze(-1) * x.unsqueeze(-2)
        mu2 = outer.mean(dim=-3).reshape(*batch, N, self.per_coil_dim ** 2)
        return torch.cat([mu1, mu2], dim=-1)

    def forward(self, x1, x2, diag=False, **params):
        psi1 = self._psi(x1)
        psi2 = self._psi(x2)
        ell = self.lengthscale

        if diag:
            sq_dist = (psi1 - psi2).pow(2).sum(-1)
            return torch.exp(-sq_dist / (2 * ell.squeeze(-1) ** 2))

        sq_dist = torch.cdist(psi1, psi2, p=2).pow(2)
        return torch.exp(-sq_dist / (2 * ell.view(*self.batch_shape, 1, 1) ** 2))