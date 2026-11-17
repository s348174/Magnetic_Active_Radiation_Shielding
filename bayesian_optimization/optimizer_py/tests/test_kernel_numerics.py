"""Exact set kernels and posterior checks must remain valid at narrow scales."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.kernels import ScaleKernel

from Kernels_v3 import CoilSetKernel
from PosteriorChecks import check_kernel, checked_posterior, raw_mean_variance


def configurations(count=6):
    generator = torch.Generator().manual_seed(23)
    values = torch.randn(count, 2, 6, generator=generator, dtype=torch.double)
    values[..., :3] += 2.
    values[..., 3:] = torch.nn.functional.normalize(values[..., 3:], dim=-1)
    return values.reshape(count, 12)


def direct_kernel(kernel, X, Y):
    """Small independent coordinate-difference reference, including normalization."""
    def scaled(values):
        coils = values.reshape(*values.shape[:-1], kernel.n_coils, 6)
        positions = coils[..., :3] / kernel.lengthscale.unsqueeze(-1)
        normals = torch.nn.functional.normalize(coils[..., 3:], dim=-1, eps=1e-8)
        return torch.cat((positions, normals*kernel.kappa.sqrt().unsqueeze(-1)), dim=-1)

    def sums(a, b):
        differences = a[..., :, :, None, None, :] - b[..., None, None, :, :, :]
        return torch.exp(-.5*differences.square().sum(-1)).sum(dim=(-3,-1))

    a, b = scaled(X), scaled(Y)
    matrix = sums(a,b)
    if kernel.normalize:
        d1 = sums(a,a).diagonal(dim1=-2,dim2=-1)
        d2 = sums(b,b).diagonal(dim1=-2,dim2=-1)
        matrix = matrix/torch.sqrt(d1.unsqueeze(-1)*d2.unsqueeze(-2))
    return matrix


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
    def test_exact_values_and_gradients_match_direct_formula(self):
        # Exercise both cross-matrix orientations, chunking, and batch broadcasting.
        for normalize in (False, True):
            kernel = CoilSetKernel(n_coils=2, chunk_size=2, normalize=normalize).double()
            for n1, n2 in ((2,5),(5,2)):
                X = configurations(n1).unsqueeze(0).expand(2,-1,-1).clone().requires_grad_()
                Y = (configurations(n2)+.1).requires_grad_()
                actual = kernel(X,Y).to_dense()
                expected = direct_kernel(kernel,X,Y)
                torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
                inputs = (X,Y,kernel.raw_lengthscale,kernel.raw_kappa)
                actual_grad = torch.autograd.grad(actual.sum(),inputs)
                expected_grad = torch.autograd.grad(expected.sum(),inputs)
                for a,b in zip(actual_grad,expected_grad):
                    torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-10)

    def test_fixed_geometry_cache_invalidates_after_input_or_scale_changes(self):
        X = configurations()
        kernel = CoilSetKernel(n_coils=2).double().requires_grad_(False)
        with patch.object(kernel,'_full_exact',wraps=kernel._full_exact) as evaluate:
            first = kernel(X,X).to_dense()
            torch.testing.assert_close(kernel(X,X).to_dense(),first)
            self.assertEqual(evaluate.call_count,1)
            kernel(X,X).to_dense().zero_()
            torch.testing.assert_close(kernel(X,X).to_dense(),first)
            # Include .data mutation, which does not increment a version counter.
            X.data[0,0] += .4
            updated = kernel(X,X).to_dense()
            torch.testing.assert_close(updated,direct_kernel(kernel,X,X))
            self.assertFalse(torch.allclose(updated,first))
            self.assertEqual(evaluate.call_count,2)
            kernel.lengthscale = .3
            kernel.kappa = 4.
            torch.testing.assert_close(kernel(X,X).to_dense(),direct_kernel(kernel,X,X))
            self.assertEqual(evaluate.call_count,3)
            saved = {name:value.clone() for name,value in kernel.state_dict().items()}
            kernel.lengthscale = 2.
            kernel(X,X).to_dense()
            kernel.load_state_dict(saved)
            torch.testing.assert_close(kernel(X,X).to_dense(),direct_kernel(kernel,X,X))
            self.assertEqual(evaluate.call_count,5)

    def test_cache_respects_dtype_normalization_and_checkpoint_state(self):
        X = configurations()
        kernel = CoilSetKernel(n_coils=2).double().requires_grad_(False)
        kernel(X,X).to_dense()
        self.assertFalse(any('cache' in key for key in kernel.state_dict()))
        kernel.normalize = False
        torch.testing.assert_close(kernel(X,X).to_dense(),direct_kernel(kernel,X,X))
        # Keep the same input dtype: parameter conversion alone also invalidates.
        kernel.float()
        torch.testing.assert_close(kernel(X,X).to_dense(),direct_kernel(kernel,X,X))
        self.assertEqual(kernel(X.float(),X.float()).to_dense().dtype,torch.float)

    def test_cache_never_reuses_an_autograd_graph_or_detaches_needed_gradients(self):
        X = configurations()
        kernel = CoilSetKernel(n_coils=2).double().requires_grad_(False)
        kernel(X,X).to_dense()
        X.requires_grad_()
        actual = kernel(X,X).to_dense()
        expected = direct_kernel(kernel,X,X)
        a, = torch.autograd.grad(actual.sum(),X)
        b, = torch.autograd.grad(expected.sum(),X)
        torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-10)
        X = X.detach()
        kernel.requires_grad_(True)
        with torch.no_grad():
            kernel(X,X).to_dense()
        actual = kernel(X,X).to_dense()
        expected = direct_kernel(kernel,X,X)
        inputs = (kernel.raw_lengthscale,kernel.raw_kappa)
        for a,b in zip(torch.autograd.grad(actual.sum(),inputs),
                       torch.autograd.grad(expected.sum(),inputs)):
            torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-10)

    def test_cached_geometry_keeps_outputscale_trainable(self):
        X = configurations()
        base = CoilSetKernel(n_coils=2).double().requires_grad_(False)
        kernel = ScaleKernel(base).double()
        for value in (.5,2.):
            kernel.outputscale = value
            kernel.zero_grad()
            matrix = kernel(X,X).to_dense()
            torch.testing.assert_close(matrix,kernel.outputscale*direct_kernel(base,X,X))
            matrix.sum().backward()
            self.assertTrue(torch.isfinite(kernel.raw_outputscale.grad).all())
            self.assertGreater(kernel.raw_outputscale.grad.abs().sum().item(),0.)

    def test_fixed_geometry_model_batches_match_direct_formula(self):
        X = configurations().unsqueeze(0)
        kernel = CoilSetKernel(n_coils=2,batch_shape=torch.Size([2])).double().requires_grad_(False)
        kernel.lengthscale = torch.tensor([.3,1.],dtype=torch.double).reshape(2,1,1)
        kernel.kappa = torch.tensor([1.,4.],dtype=torch.double).reshape(2,1,1)
        for _ in range(2):
            actual = kernel(X,X).to_dense()
            self.assertEqual(actual.shape,(2,len(X[0]),len(X[0])))
            torch.testing.assert_close(actual,direct_kernel(kernel,X,X),atol=1e-12,rtol=1e-12)

    def test_cached_gp_and_noisy_ei_match_independent_exact_reference(self):
        class ReferenceKernel(CoilSetKernel):
            def forward(self,X,Y,diag=False,**kwargs):
                matrix = direct_kernel(self,X,Y)
                return matrix.diagonal(dim1=-2,dim2=-1) if diag else matrix

        X = configurations()
        y = torch.arange(len(X),dtype=X.dtype).unsqueeze(-1)/5
        Z = (configurations(2)+.2).unsqueeze(-2).requires_grad_()
        predictions, values, gradients = [], [], []
        for cls in (CoilSetKernel,ReferenceKernel):
            kernel = ScaleKernel(cls(n_coils=2).double().requires_grad_(False)).double()
            gp = SingleTaskGP(X,y,train_Yvar=torch.full_like(y,.02),covar_module=kernel,
                              outcome_transform=Standardize(m=1))
            kernel(X,X).to_dense()  # Populate constant geometry before predictions.
            posterior = gp.posterior(Z)
            predictions.append((posterior.mean,posterior.variance))
            acquisition = qLogNoisyExpectedImprovement(gp,X_baseline=X,cache_root=True,
                prune_baseline=False,sampler=SobolQMCNormalSampler(torch.Size([32]),seed=67))
            value = acquisition(Z)
            values.append(value)
            gradients.append(torch.autograd.grad(value.sum(),Z)[0])
        for a,b in zip(predictions[0],predictions[1]):
            torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)
        torch.testing.assert_close(values[0],values[1],atol=1e-10,rtol=1e-9)
        torch.testing.assert_close(gradients[0],gradients[1],atol=1e-10,rtol=1e-9)

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
