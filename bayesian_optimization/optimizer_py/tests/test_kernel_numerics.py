"""Exact set kernels and posterior checks must remain valid at narrow scales."""
from types import SimpleNamespace
import unittest

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel

from Kernels_v3 import CoilSetKernel
from PosteriorChecks import check_kernel, checked_posterior, raw_mean_variance


def configurations(count=6):
    generator = torch.Generator().manual_seed(23)
    values = torch.randn(count, 2, 6, generator=generator, dtype=torch.double)
    values[..., :3] += 2.
    values[..., 3:] = torch.nn.functional.normalize(values[..., 3:], dim=-1)
    return values.reshape(count, 12)


class PosteriorStub:
    def __init__(self, covariance, prior=1.):
        self.covariance = torch.as_tensor(covariance, dtype=torch.double)
        self.prior = prior

    def posterior(self, X):
        return SimpleNamespace(mean=X[..., :1]*0.,
                               distribution=SimpleNamespace(covariance_matrix=self.covariance))

    def covar_module(self, X1, X2, diag=False):
        return torch.full(X1.shape[:-1], self.prior, dtype=X1.dtype)


class KernelNumericsTests(unittest.TestCase):
    def test_tiny_lengthscale_and_large_concentration_remain_valid(self):
        X = configurations()
        for ell, kappa in [(1e-8, 1.), (1., 1e12), (1e-8, 1e12)]:
            kernel = CoilSetKernel(n_coils=2, chunk_size=2).double()
            kernel.lengthscale = ell
            kernel.kappa = kappa
            full = kernel(X, X).to_dense()
            torch.testing.assert_close(full.diag(), torch.ones(len(X),dtype=X.dtype), atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(full.diag(), kernel(X, X, diag=True), atol=1e-12, rtol=1e-12)
            self.assertGreaterEqual(torch.linalg.eigvalsh(full).min().item(), -1e-12)
            check_kernel(SimpleNamespace(covar_module=ScaleKernel(kernel).double()), X)

    def test_cross_diagonal_matches_full_and_permutations(self):
        X = configurations()
        Y = configurations()+.01
        kernel = CoilSetKernel(n_coils=2, chunk_size=2).double()
        torch.testing.assert_close(kernel(X,Y).to_dense().diag(), kernel(X,Y,diag=True))
        permuted = X.reshape(-1,2,6).flip(-2).reshape(-1,12)
        torch.testing.assert_close(kernel(X,Y).to_dense(),kernel(permuted,Y).to_dense())

    def test_equal_valued_independent_inputs_retain_both_gradients(self):
        # Reusing z1 merely because x1==x2 would erase all derivatives of x2.
        X = configurations(2).requires_grad_()
        Y = X.detach().clone().requires_grad_()
        kernel = CoilSetKernel(n_coils=2, chunk_size=1).double()
        torch.autograd.gradcheck(lambda a,b: kernel(a,b).to_dense(), (X,Y), eps=1e-6, atol=1e-5)
        matrix = kernel(X,Y).to_dense()
        a,b = torch.autograd.grad(matrix[0,1], (X,Y))
        self.assertGreater(b.abs().sum().item(), 0.)
        self.assertGreater(a.abs().sum().item(), 0.)

    def test_duplicates_and_batched_inputs(self):
        X = configurations(3)
        X[1] = X[0]
        kernel = CoilSetKernel(n_coils=2, chunk_size=1).double()
        batched = torch.stack([X,X])
        full = kernel(batched,batched).to_dense()
        torch.testing.assert_close(full.diagonal(dim1=-2,dim2=-1),kernel(batched,batched,diag=True))
        self.assertTrue(torch.isfinite(full).all())
        self.assertGreaterEqual(torch.linalg.eigvalsh(full).min().item(), -1e-12)

    def test_random_features_still_preserve_independent_input_gradients(self):
        X = configurations(2).requires_grad_()
        Y = X.detach().clone().requires_grad_()
        kernel = CoilSetKernel(n_coils=2,n_features=16).double()
        a,b = torch.autograd.grad(kernel(X,Y).to_dense()[0,1], (X,Y))
        self.assertGreater(a.abs().sum().item(), 0.)
        self.assertGreater(b.abs().sum().item(), 0.)

    def test_material_negative_and_excess_prior_variances_fail(self):
        X = torch.zeros(2,1,dtype=torch.double)
        for covariance in [torch.diag(torch.tensor([-.01,.1])),torch.diag(torch.tensor([1.01,.1]))]:
            with self.assertRaises(RuntimeError):
                raw_mean_variance(PosteriorStub(covariance), X)

    def test_tiny_positive_variance_is_not_floored(self):
        X = torch.zeros(2,1,dtype=torch.double)
        _, variance = raw_mean_variance(PosteriorStub(torch.diag(torch.tensor([1e-14,1e-12],dtype=torch.double))), X)
        self.assertEqual(variance[0].item(), 1e-14)

    def test_only_roundoff_negative_variance_is_clamped(self):
        X = torch.zeros(2,1,dtype=torch.double)
        _, variance = raw_mean_variance(PosteriorStub(torch.diag(torch.tensor([-1e-13,1e-12],dtype=torch.double))), X)
        self.assertEqual(variance[0].item(), 0.)
        # Tolerance scales with prior, rather than hiding errors in tiny kernels.
        with self.assertRaises(RuntimeError):
            raw_mean_variance(PosteriorStub(torch.eye(2,dtype=torch.double)*-1e-13, prior=1e-8), X)

    def test_non_psd_joint_posterior_fails(self):
        X = torch.zeros(2,1,dtype=torch.double)
        with self.assertRaises(RuntimeError):
            checked_posterior(PosteriorStub([[.1,.2],[.2,.1]]),X,full_covariance=True)

    def test_real_standardized_gp_passes_and_preserves_gradients(self):
        X = configurations()
        y = torch.arange(len(X),dtype=torch.double).unsqueeze(-1)/20-34
        gp = SingleTaskGP(X,y,train_Yvar=torch.ones_like(y)*.0003,
                          covar_module=ScaleKernel(CoilSetKernel(n_coils=2)).double(),
                          outcome_transform=Standardize(m=1))
        Z = configurations(2).detach().requires_grad_()
        mean,variance,covariance = checked_posterior(gp,Z,full_covariance=True)
        (mean.sum()+variance.sum()).backward()
        self.assertTrue(torch.isfinite(Z.grad).all())
        self.assertEqual(covariance.shape,(2,2))
        check_kernel(gp,X)

    def test_kernel_matrix_diagonal_mismatch_is_rejected(self):
        class BadKernel:
            def __call__(self,X,Y,diag=False):
                return torch.ones(len(X),dtype=X.dtype)*2 if diag else torch.eye(len(X),dtype=X.dtype)
        with self.assertRaises(RuntimeError):
            check_kernel(SimpleNamespace(covar_module=BadKernel()),torch.zeros(2,1,dtype=torch.double))


if __name__ == '__main__':
    unittest.main()
