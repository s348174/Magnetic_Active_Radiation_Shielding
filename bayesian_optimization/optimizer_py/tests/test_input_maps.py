"""Physical coordinate agreement, acquisition gradients and all BO loop paths."""
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from botorch.acquisition import PosteriorMean, qLogNoisyExpectedImprovement, qKnowledgeGradient
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import BoUtils as utils
import BoLoop
import TurBo
from input import D, K, LOWER, UPPER
from Objective import spherical_to_cartesian
from Kernels_v3 import CoilSetKernel


class KernelDiagonalTests(unittest.TestCase):
    def test_cross_diagonals_and_gradients(self):
        for features in (None, 64):
            for normalize in (False, True):
                with self.subTest(features=features, normalize=normalize):
                    generator = torch.Generator().manual_seed(67)
                    X = torch.randn(2,4,18,dtype=torch.double,generator=generator).requires_grad_()
                    Y = torch.randn(2,4,18,dtype=torch.double,generator=generator).requires_grad_()
                    kernel = CoilSetKernel(n_coils=3, normalize=normalize,
                                           n_features=features, chunk_size=2).double()
                    paired = kernel.forward(X,Y,diag=True)
                    expected = kernel.forward(X,Y).diagonal(dim1=-2,dim2=-1)
                    torch.testing.assert_close(paired, expected, atol=1e-11, rtol=1e-10)
                    if normalize:
                        self.assertFalse(torch.allclose(paired, torch.ones_like(paired)))
                    inputs = (X,Y,kernel.raw_lengthscale,kernel.raw_kappa)
                    paired_grad = torch.autograd.grad(paired.sum(),inputs)
                    full_grad = torch.autograd.grad(expected.sum(),inputs)
                    for actual, reference in zip(paired_grad, full_grad):
                        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-8)

    def test_self_diagonals(self):
        X = torch.randn(2,4,18,dtype=torch.double,generator=torch.Generator().manual_seed(67))
        for features in (None,64):
            for normalize in (False,True):
                with self.subTest(features=features, normalize=normalize):
                    kernel = CoilSetKernel(n_coils=3, normalize=normalize, n_features=features).double()
                    paired = kernel.forward(X,X,diag=True)
                    expected = kernel.forward(X,X).diagonal(dim1=-2,dim2=-1)
                    torch.testing.assert_close(paired, expected)
                    if normalize:
                        torch.testing.assert_close(paired, torch.ones_like(paired))


class CoordinateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        self.X = .1 + .8*torch.rand(2, 3, D, dtype=torch.double)

    def test_roundtrip_and_gradients(self):
        X = self.X.clone().requires_grad_()
        physical = utils.denormalize(X)
        torch.testing.assert_close(utils.normalize(physical), X)
        physical.sum().backward()
        torch.testing.assert_close(X.grad, (UPPER-LOWER).expand_as(X))
        np.testing.assert_allclose(utils.normalize(physical.detach().numpy()).numpy(), self.X.numpy())

    def test_cartesian_matches_simulator(self):
        physical = utils.denormalize(self.X).numpy().reshape(-1, D)
        expected = []
        for row in physical:
            centers, normals = utils.unpack_configuration(row)
            c = spherical_to_cartesian(centers[:,0], centers[:,1], centers[:,2])
            n = spherical_to_cartesian(np.ones(K), normals[:,0], normals[:,1])
            expected.append(np.concatenate((c,n), axis=1))
        coils = utils.map_coilwise(self.X).reshape(-1,K,6).numpy()
        np.testing.assert_allclose(coils, expected, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(coils[:,:,3:], axis=-1), 1., atol=1e-14)
        component_layout = coils.transpose(0,2,1).reshape(2,3,6*K)
        np.testing.assert_allclose(utils.map_cart(self.X.numpy()), component_layout, atol=1e-14)

    def test_lower_radius_retains_angular_information(self):
        X = torch.zeros(2,D,dtype=torch.double)
        X[1,2*K:3*K] = .25
        coils = utils.map_coilwise(X).reshape(2,K,6)
        torch.testing.assert_close(coils[0,:,2], torch.ones(K,dtype=torch.double))
        torch.testing.assert_close(coils[1,:,0], torch.ones(K,dtype=torch.double))
        self.assertGreater((coils[0,:,:3]-coils[1,:,:3]).norm().item(), 1)

    def test_periodic_endpoints_and_permutations(self):
        X = self.X[0].clone()
        X[0,2*K:3*K] = 0
        X[1] = X[0]
        X[1,2*K:3*K] = 1
        torch.testing.assert_close(utils.map_cart(X[0]), utils.map_cart(X[1]), atol=1e-13, rtol=1e-13)
        permutation = torch.randperm(K)
        permuted = self.X.reshape(2,3,5,K)[...,permutation].reshape(2,3,D)
        torch.testing.assert_close(utils.map_set(self.X), utils.map_set(permuted))
        expected = utils.map_coilwise(self.X).reshape(2,3,K,6)[...,permutation,:]
        torch.testing.assert_close(utils.map_coilwise(permuted).reshape(2,3,K,6), expected)

    def test_cartesian_gradient_check(self):
        X = self.X[0,:1].clone().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(utils.map_cart, (X,)))
        for mapping in (utils.map_coilwise, utils.map_set):
            leaf = self.X.clone().requires_grad_()
            mapping(leaf).square().sum().backward()
            self.assertTrue(torch.isfinite(leaf.grad).all())
            self.assertGreater(leaf.grad.abs().sum().item(), 0)

    def test_single_coil_pooling(self):
        with patch.object(utils, 'K', 1), patch.object(utils, 'LOWER', LOWER[::K]), \
             patch.object(utils, 'UPPER', UPPER[::K]):
            pooled = utils.map_set(torch.full((2,5),.3,dtype=torch.double))
            self.assertTrue(torch.isfinite(pooled).all())
            torch.testing.assert_close(pooled[:,6:12], torch.zeros(2,6,dtype=torch.double))

    def test_duplicate_checks_use_normalized_rows(self):
        train = torch.stack([torch.zeros(D),torch.ones(D)]).double()
        self.assertTrue(utils.duplicate_safe_candidate(train, train[0].numpy()))
        candidate = torch.full((1,D),.5,dtype=torch.double)
        train[:,0] = 0
        candidate[:,0] = 0
        self.assertFalse(utils.duplicate_safe_candidate(train, candidate))

    def test_periodic_environment_toggle(self):
        code = ('import input, BoUtils, torch; '
                'assert input.USE_FEATURE_MAPPING; '
                'assert input.MAPPED_D == input.D + len(input.PERIODIC_IDXS); '
                'assert BoUtils.map_fn(torch.zeros(1,input.D)).shape[-1] == input.MAPPED_D')
        result = subprocess.run([sys.executable,'-c',code],
                                env=dict(os.environ, USE_FEATURE_MAPPING='1'),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        self.X = .1 + .8*torch.rand(4,D,dtype=torch.double)
        self.Y = -utils.denormalize(self.X)[:,:K].sum(-1,keepdim=True)

    def test_posteriors_and_acquisition_gradients(self):
        for mapping in (utils.identity_map, utils.periodic_feature_map, utils.map_cart,
                        utils.map_set, utils.map_coilwise):
            with self.subTest(mapping=mapping.__name__):
                gp = SingleTaskGP(mapping(self.X), self.Y)
                model = utils.InputMappedModel(gp, mapping)
                X = (.2 + .6*torch.rand(2,1,D,dtype=torch.double)).requires_grad_()
                torch.testing.assert_close(model.posterior(X).mean, gp.posterior(mapping(X)).mean)
                direct = utils.InputMappedAcquisition(PosteriorMean(gp), mapping)
                torch.testing.assert_close(direct(X), PosteriorMean(model)(X))
                nei = qLogNoisyExpectedImprovement(model, X_baseline=self.X, prune_baseline=False)
                value = nei(X).sum()
                gradient, = torch.autograd.grad(value, X)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)

    def test_knowledge_gradient_fantasies(self):
        gp = SingleTaskGP(utils.map_cart(self.X), self.Y)
        model = utils.InputMappedModel(gp, utils.map_cart)
        kg = qKnowledgeGradient(model, num_fantasies=2)
        X = (.2 + .6*torch.rand(1,3,D,dtype=torch.double)).requires_grad_()
        value = kg(X).sum()
        gradient, = torch.autograd.grad(value, X)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_loop_coordinate_contracts(self):
        # Execute actual GP fits and acquisition optimizers with small test budgets.
        # Only the expensive particle simulator is replaced by a known smooth response.
        def fit(mll):
            return fit_gpytorch_mll(mll, optimizer_kwargs={'options': {'maxiter': 3}})

        def fit_model(gp):
            return fit(ExactMarginalLogLikelihood(gp.likelihood, gp))

        def optimize(**kwargs):
            kwargs.update(num_restarts=2, raw_samples=8,
                          options={'maxiter': 3, 'batch_limit': 2},
                          retry_on_optimization_warning=False)
            return optimize_acqf(**kwargs)

        cases = [(BoLoop, BoLoop.bo_matern_kernel, {}, utils.identity_map),
                 (BoLoop, BoLoop.bo_matern_kernel, {}, utils.periodic_feature_map),
                 (BoLoop, BoLoop.bo_matern_kernel, {}, utils.map_cart),
                 (BoLoop, BoLoop.bo_rbf_kernel, {}, None),
                 (BoLoop, BoLoop.bo_rbf_old, {}, utils.identity_map),
                 (BoLoop, BoLoop.bo_rbf_old, {}, utils.periodic_feature_map),
                 (TurBo, TurBo.bo_turbo_matern, {'acqf':'ts'}, None),
                 (TurBo, TurBo.bo_turbo_matern, {'acqf':'ei'}, None)]
        for module, function, kwargs, mapping in cases:
            with self.subTest(function=function.__name__, kwargs=kwargs, mapping=mapping):
                observed = []
                def objective(seed, centers, normals):
                    centers, normals = np.asarray(centers), np.asarray(normals)
                    self.assertEqual(centers.shape, (K,3))
                    self.assertEqual(normals.shape, (K,2))
                    self.assertTrue(np.all((centers[:,0] >= 1) & (centers[:,0] <= 4)))
                    observed.append(centers.copy())
                    return float(centers[:,0].sum()), 1e-4

                initial = (self.X.clone(), self.Y.squeeze(-1).numpy().copy(), np.full(4,1e-4))
                with patch.object(module,'sobol_sample',return_value=initial), \
                     patch.object(module,'MAX_ITER',1), \
                     patch.object(module,'objective_function',side_effect=objective), \
                     patch.object(module,'fit_gp_with_noise_floor',side_effect=fit_model), \
                     patch.object(module,'fit_gpytorch_mll',side_effect=fit), \
                     patch.object(module,'optimize_acqf',side_effect=optimize), \
                     patch.object(module,'map_fn',mapping or utils.identity_map), \
                     patch.object(TurBo,'N_CANDIDATES',16):
                    result = function(**kwargs)
                self.assertEqual(len(observed),1)
                self.assertEqual(result[0].shape,(5,D))
                physical = utils.denormalize(result[0][-1]).numpy()
                centers, _ = utils.unpack_configuration(physical)
                np.testing.assert_allclose(observed[0], centers)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
