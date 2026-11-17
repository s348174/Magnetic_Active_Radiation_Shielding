"""
Kernels_v3.py (stable exact distances and optional random features)
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

Scaled representation:
    For unit normals,  kappa*(n.n' - 1) = -kappa*||n - n'||^2 / 2, hence
        k_pos * k_vMF = exp(-||z - z'||^2 / 2),   z = [ p/l , sqrt(kappa)*n ]
    i.e. the per-coil kernel is a plain Gaussian RBF on a 6-dim scaled vector z.
    Direct coordinate differences are used to avoid cancellation when a fitted
    lengthscale is very small or the directional concentration is large.
    Fixed-geometry self-covariances are cached only when no gradients are needed.

Two evaluation modes (selected by `n_features`):
    n_features=None  -> EXACT. Chunked direct Euclidean distances and one exp.
    n_features=F     -> Random Fourier Features. Approximate, PSD and permutation
                        invariant. Normalization gives unit diagonal for nonzero
                        feature vectors. Feature construction costs O(N*K*F),
                        and the dense Gram product costs O(N^2*F). This reduces
                        intermediate storage; exact GP factorization is still cubic.
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
    coils. Independent per-coil scales would break permutation invariance.
    Coordinate-specific scales shared across coils could preserve it, but are not used.
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
        self._exact_cache = None  # One constant Gram matrix; excluded from state_dict.

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

    # -- exact path -------------------------------------------------------------
    @staticmethod
    def _gaussian(first, second):
        """Stable distances; reuse their storage only outside the autograd graph."""
        distances = torch.cdist(first, second, compute_mode='donot_use_mm_for_euclid_dist')
        if distances.requires_grad:
            return torch.exp(-.5 * distances.square())
        return distances.square_().mul_(-.5).exp_()

    def _full_exact(self, z1, z2):
        """Direct differences avoid cancellation for small learned lengthscales."""
        # For a few candidates against many observations, make one small cross
        # matrix instead of launching a separate operation per training chunk.
        # Only do this below 2 MiB per distance matrix, including batch copies.
        pairs = (math.prod(torch.broadcast_shapes(z1.shape[:-3], z2.shape[:-3]))
                 * z1.shape[-3] * z2.shape[-3] * self.n_coils**2)
        transpose = z1.shape[-3] < z2.shape[-3] and pairs*z1.element_size() <= 2*1024**2
        if transpose:
            z1, z2 = z2, z1
        nc, d = self.n_coils, z1.shape[-1]
        N1, N2 = z1.shape[-3], z2.shape[-3]
        first = z1.reshape(*z1.shape[:-3], N1 * nc, d)

        out = []
        for s in range(0, N2, self.chunk_size):
            e = min(s + self.chunk_size, N2)
            zc = z2[..., s:e, :, :]
            second = zc.reshape(*zc.shape[:-3], (e - s) * nc, d)
            E = self._gaussian(first, second)
            E = E.reshape(*E.shape[:-2], N1, nc, e - s, nc).sum(dim=(-3, -1))   # (..., N1, w)
            out.append(E)
        Kmat = out[0] if len(out) == 1 else torch.cat(out, dim=-1)
        assert Kmat.shape[-2:] == (N1, N2), f"shape bug: {tuple(Kmat.shape)} vs ({N1},{N2})"
        return Kmat.transpose(-2, -1) if transpose else Kmat

    def _diag_exact(self, z1, z2):
        """Paired cross-diagonals use the same stable distances as the full path."""
        return self._gaussian(z1, z2).sum(dim=(-2, -1))

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
        # [NEW] K(X,X) is the common case during fitting: reuse features instead of recomputing
        # Equal-valued independent inputs still need their own autograd paths.
        same = x1 is x2
        cacheable = (self.n_features is None and not diag and same
                     and not x1.requires_grad
                     and (not torch.is_grad_enabled() or
                          not (self.raw_lengthscale.requires_grad or self.raw_kappa.requires_grad)))
        if cacheable:
            scales = torch.cat((self.lengthscale.reshape(-1), self.kappa.reshape(-1))).detach()
            signature = (x1.dtype, x1.device, scales.dtype, scales.device,
                         self.raw_lengthscale.shape, self.raw_kappa.shape,
                         self.n_coils, self.per_coil_dim, self.pos_dim, self.normalize)
            cached = self._exact_cache
            # Compare saved VALUES: GPyTorch initialization may update .data
            # without incrementing PyTorch's tensor version counter.
            if (cached is not None and signature == cached[0]
                    and torch.equal(scales, cached[1]) and torch.equal(x1, cached[2])):
                return cached[3].clone()

        z1 = self._coil_z(x1)
        z2 = z1 if same else self._coil_z(x2)

        if self.n_features is not None:                                   # ---- RFF path
            p1 = self._phi(z1)
            p2 = p1 if same else self._phi(z2)
            if self.normalize:                                            # cosine similarity of feature vectors
                p1 = F.normalize(p1, dim=-1, eps=1e-12)
                p2 = p1 if same else F.normalize(p2, dim=-1, eps=1e-12)
            if diag:
                return (p1 * p2).sum(-1)
            return p1 @ p2.transpose(-2, -1)

        # Paired cross-diagonals need the same normalization as the full matrix.
        Kmat = self._diag_exact(z1, z2) if diag else self._full_exact(z1, z2)
        if self.normalize:
            # The self-pair sums are already present in a self-covariance.
            if same:
                d1 = Kmat if diag else Kmat.diagonal(dim1=-2, dim2=-1)
            else:
                d1 = self._diag_exact(z1, z1)
            d2 = d1 if same else self._diag_exact(z2, z2)
            denom = torch.sqrt(d1 * d2) if diag else torch.sqrt(d1.unsqueeze(-1) * d2.unsqueeze(-2))
            Kmat = Kmat / denom.clamp_min(1e-12)
        if cacheable:
            # Callers may modify returned tensors; never expose the stored matrix.
            self._exact_cache = (signature, scales.clone(), x1.detach().clone(), Kmat.detach().clone())
        return Kmat


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
