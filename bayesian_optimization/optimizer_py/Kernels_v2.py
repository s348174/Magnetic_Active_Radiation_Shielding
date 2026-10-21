"""
Kernels.py
----------
Permutation-invariant kernel for arrays of `n_coils` identical, indistinguishable
coils, optimized for CPU execution.
"""

import torch
import torch.nn.functional as F
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
from input import K


class CoilSetKernel(Kernel):
    """
    Permutation-invariant "mean embedding" kernel (CPU-optimized).
    """

    is_stationary = False

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

    def _split(self, x):
        *batch, N, D = x.shape
        assert D == self.n_coils * self.per_coil_dim, (
            f"CoilSetKernel expected last dim {self.n_coils * self.per_coil_dim}, got {D}."
        )
        x = x.view(*batch, N, self.n_coils, self.per_coil_dim)
        pos = x[..., : self.pos_dim]
        nrm = F.normalize(x[..., self.pos_dim:], dim=-1, eps=1e-8)
        return pos, nrm

    def _diag_forward(self, x1, x2):
        pos1, n1 = self._split(x1)   # (..., N, n_coils, pos_dim)
        pos2, n2 = self._split(x2)   # (..., N, n_coils, pos_dim)

        ell = self.lengthscale       # (..., 1, 1)
        kappa = self.kappa           # (..., 1, 1)

        # OPTIMIZATION: Expanded Euclidean distance using batched GEMM (@) instead of broadcasting diffs
        pos1_sq = pos1.pow(2).sum(dim=-1, keepdim=True)                   # (..., N, n_coils, 1)
        pos2_sq = pos2.pow(2).sum(dim=-1, keepdim=True).transpose(-1, -2)  # (..., N, 1, n_coils)
        pos_dot = pos1 @ pos2.transpose(-1, -2)                           # (..., N, n_coils, n_coils)
        sq_dist = (pos1_sq + pos2_sq - 2 * pos_dot).clamp_min(0.0)

        ell_b = ell.unsqueeze(-1)                                         # (..., 1, 1, 1)
        pos_k = torch.exp(-sq_dist / (2 * ell_b ** 2))

        # OPTIMIZATION: Dot product via matrix multiplication for vMF kernel
        cos_sim = n1 @ n2.transpose(-1, -2)                               # (..., N, n_coils, n_coils)
        kappa_b = kappa.unsqueeze(-1)
        vmf_k = torch.exp(kappa_b * (cos_sim - 1.0))

        K_diag = (pos_k * vmf_k).sum(dim=(-2, -1))                        # (..., N)
        return K_diag

    def _full_forward(self, x1, x2, chunk_size):
        pos1, n1 = self._split(x1)   # (..., N1, n_coils, pos_dim)
        pos2, n2 = self._split(x2)   # (..., N2, n_coils, pos_dim)

        batch_prefix = pos1.shape[:-3]
        N1 = pos1.shape[-3]
        N2 = pos2.shape[-3]

        pos1_flat = pos1.reshape(*batch_prefix, N1 * self.n_coils, pos1.shape[-1])
        n1_flat = n1.reshape(*batch_prefix, N1 * self.n_coils, n1.shape[-1])

        pos1_sq = pos1_flat.pow(2).sum(dim=-1, keepdim=True)              # (..., N1*nc, 1)
        ell_flat = self.lengthscale.view(*self.batch_shape, 1, 1)
        kappa_flat = self.kappa.view(*self.batch_shape, 1, 1)

        out_chunks = []
        for start in range(0, N2, chunk_size):
            end = min(start + chunk_size, N2)
            width = end - start

            pos2_c = pos2[..., start:end, :, :].reshape(*batch_prefix, width * self.n_coils, pos2.shape[-1])
            n2_c = n2[..., start:end, :, :].reshape(*batch_prefix, width * self.n_coils, n2.shape[-1])

            # OPTIMIZATION: Replaced torch.cdist with expanded GEMM distance calculation
            pos2_sq = pos2_c.pow(2).sum(dim=-1, keepdim=True).transpose(-1, -2)  # (..., 1, width*nc)
            pos_dot = pos1_flat @ pos2_c.transpose(-1, -2)                     # (..., N1*nc, width*nc)
            sq_dist = (pos1_sq + pos2_sq - 2 * pos_dot).clamp_min(0.0)
            del pos_dot, pos2_sq

            pos_k = torch.exp(-sq_dist / (2 * ell_flat ** 2))
            del sq_dist

            cos_sim = n1_flat @ n2_c.transpose(-2, -1)                         # (..., N1*nc, width*nc)
            vmf_k = torch.exp(kappa_flat * (cos_sim - 1.0))
            del cos_sim

            coil_k = pos_k * vmf_k                                             # (..., N1*nc, width*nc)
            del pos_k, vmf_k

            coil_k = coil_k.reshape(*batch_prefix, N1, self.n_coils, width, self.n_coils)
            chunk_out = coil_k.sum(dim=(-3, -1))                                # (..., N1, width)
            del coil_k
            out_chunks.append(chunk_out)

        K = out_chunks[0] if len(out_chunks) == 1 else torch.cat(out_chunks, dim=-1)

        expected_shape = (*batch_prefix, N1, N2)
        if tuple(K.shape) != expected_shape:
            raise RuntimeError(
                f"CoilSetKernel._full_forward produced shape {tuple(K.shape)}, expected {expected_shape}."
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

        diag1 = self._diag_forward(x1, x1)
        diag2 = self._diag_forward(x2, x2)
        denom = torch.sqrt(diag1.unsqueeze(-1) * diag2.unsqueeze(-2)).clamp_min(1e-12)
        return K / denom


class CoilMomentKernel(Kernel):
    """
    Second-order-moment refinement kernel (CPU-optimized with chunking).
    """

    is_stationary = False

    def __init__(self, n_coils: int = 10, per_coil_dim: int = 6,
                 chunk_size: int = 500, lengthscale_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.n_coils = n_coils
        self.per_coil_dim = per_coil_dim
        self.chunk_size = chunk_size

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
            ell_b = ell.squeeze(-1)
            return torch.exp(-sq_dist / (2 * ell_b ** 2))

        # OPTIMIZATION: Added chunk-wise computation for CoilMomentKernel to prevent 
        # memory spikes and CPU cache thrashing on large datasets (N1, N2).
        ell2 = ell.view(*self.batch_shape, 1, 1)
        N2 = psi2.shape[-2]
        psi1_sq = psi1.pow(2).sum(dim=-1, keepdim=True)

        out_chunks = []
        for start in range(0, N2, self.chunk_size):
            end = min(start + self.chunk_size, N2)
            psi2_c = psi2[..., start:end, :]
            psi2_sq = psi2_c.pow(2).sum(dim=-1, keepdim=True).transpose(-1, -2)
            psi_dot = psi1 @ psi2_c.transpose(-1, -2)
            
            sq_dist = (psi1_sq + psi2_sq - 2 * psi_dot).clamp_min(0.0)
            chunk_k = torch.exp(-sq_dist / (2 * ell2 ** 2))
            out_chunks.append(chunk_k)

        return out_chunks[0] if len(out_chunks) == 1 else torch.cat(out_chunks, dim=-1)