"""Exact coil-set kernel integration with vanilla BO and calibrated log noise."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.exceptions.errors import ModelFittingError
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.kernels import ScaleKernel

import BoLoop
import BoUtils
import RegretStopping
import bo_main
from Kernels_v3 import CoilSetKernel
from input import D, K


def short_fit(mll):
    return fit_gpytorch_mll(mll, optimizer_kwargs={'options': {'maxiter': 3}})


class CoilSetTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(67)
        self.X = .1+.8*torch.rand(5,D,dtype=torch.double,generator=generator)
        self.y = np.array([-34.00,-34.03,-34.01,-34.04,-34.02])
        self.noise = 2.5e-5
        self.calibration = {'variance':self.noise, 'X':self.X[:3].tolist(),
                            'mean':self.y[:3].tolist(), 'mean_variance':[self.noise/6]*3,
                            'names':['reference','near','far'], 'repeats':6,
                            'single_run_variance':[self.noise]*3, 'runs':[], 'settings':{}}
        self.initial = (self.X,self.y,np.full(len(self.y),.001))
        def direct_optimizer(**kwargs):
            for name in ('X','ranking','optimizer'):
                kwargs.pop(name,None)
            return BoLoop.optimize_acqf(**kwargs)
        for patcher in (
            patch.object(BoLoop,'optimize_acquisition',side_effect=direct_optimizer),
            patch.object(BoLoop,'select_coil_scales',return_value=(.5,1.,
                {'predictive_ok':True,'selection_skill':.1})),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def fit_gp(self):
        kernel = ScaleKernel(CoilSetKernel(n_coils=K,normalize=True,n_features=None))
        with patch.object(BoLoop,'fit_gpytorch_mll',side_effect=short_fit):
            return BoLoop._fit_model(self.X,self.y,
                torch.full((len(self.y),1),self.noise,dtype=torch.double),
                BoUtils.map_coilwise,kernel)

    def test_physical_features_and_kernel_ignore_coil_labels(self):
        permutation = torch.arange(K-1,-1,-1)
        permuted = self.X.reshape(-1,5,K)[...,permutation].reshape(-1,D)
        features = BoUtils.map_coilwise(self.X)
        transformed = BoUtils.map_coilwise(permuted)
        torch.testing.assert_close(transformed.reshape(-1,K,6),
                                   features.reshape(-1,K,6)[:,permutation])
        # Positions retain their physical radii; normals have unit length.
        physical = BoUtils.denormalize(self.X)
        torch.testing.assert_close(features.reshape(-1,K,6)[...,:3].norm(dim=-1),physical[:,:K])
        torch.testing.assert_close(features.reshape(-1,K,6)[...,3:].norm(dim=-1),
                                   torch.ones(len(self.X),K,dtype=torch.double))
        kernel = CoilSetKernel(n_coils=K,normalize=True,n_features=None).double()
        gram = kernel(features).to_dense()
        torch.testing.assert_close(kernel(transformed,features).to_dense(),gram)
        torch.testing.assert_close(kernel(features,transformed).to_dense(),gram)
        torch.testing.assert_close(gram.diagonal(),torch.ones(len(self.X),dtype=torch.double))
        self.assertGreaterEqual(torch.linalg.eigvalsh(gram).min().item(),-1e-10)

    def test_periodic_endpoints_have_identical_covariances(self):
        kernel = CoilSetKernel(n_coils=K,normalize=True,n_features=None).double()
        for coordinate in (2*K,4*K):
            with self.subTest(coordinate=coordinate):
                first = self.X[:1].clone()
                last = first.clone()
                first[:,coordinate] = 0.
                last[:,coordinate] = 1.
                a,b = BoUtils.map_coilwise(first),BoUtils.map_coilwise(last)
                torch.testing.assert_close(a,b,atol=1e-14,rtol=1e-14)
                torch.testing.assert_close(kernel(a,BoUtils.map_coilwise(self.X)).to_dense(),
                                           kernel(b,BoUtils.map_coilwise(self.X)).to_dense())

    def test_noisy_ei_cached_batches_and_gradients_match(self):
        gp = self.fit_gp()
        sampler = SobolQMCNormalSampler(torch.Size([32]),seed=67)
        base = qLogNoisyExpectedImprovement(gp,X_baseline=BoUtils.map_coilwise(self.X),
                                            sampler=sampler,cache_root=True,prune_baseline=False)
        self.assertTrue(base._cache_root)
        acquisition = BoUtils.InputMappedAcquisition(base,BoUtils.map_coilwise)
        candidate = (.1+.8*torch.rand(3,1,D,dtype=torch.double,
            generator=torch.Generator().manual_seed(103))).requires_grad_(True)
        full = acquisition(candidate)
        full_gradient, = torch.autograd.grad(full.sum(),candidate)
        split = torch.cat([acquisition(batch) for batch in candidate.split(1)])
        split_gradient, = torch.autograd.grad(split.sum(),candidate)
        torch.testing.assert_close(split,full)
        torch.testing.assert_close(split_gradient,full_gradient)
        self.assertTrue(torch.isfinite(full_gradient).all())
        self.assertGreater(full_gradient.abs().max().item(),0.)

    def test_fitted_posterior_is_invariant_to_coil_permutation(self):
        gp = self.fit_gp()
        permutation = torch.arange(K-1,-1,-1)
        permuted = self.X.reshape(-1,5,K)[...,permutation].reshape(-1,D)
        with torch.no_grad():
            original = gp.posterior(BoUtils.map_coilwise(self.X))
            reordered = gp.posterior(BoUtils.map_coilwise(permuted))
        torch.testing.assert_close(original.mean,reordered.mean)
        torch.testing.assert_close(original.variance,reordered.variance)

    def test_loop_keeps_raw_designs_calibrated_noise_and_model_stopping_guard(self):
        for model_ok in (True,False):
            stopping = RegretStopping.SimpleRegretStopping(
                enabled=True,min_evaluations=1,check_every=1,record_every=1,patience=1,grid_size=8)
            proposals = iter(torch.full((1,D),v,dtype=torch.double) for v in (.25,.75))
            snapshots, checkpoints, reports = [], [], []
            def fit(mll):
                gp = mll.model
                self.assertIsInstance(gp.covar_module,ScaleKernel)
                base = gp.covar_module.base_kernel
                self.assertIsInstance(base,CoilSetKernel)
                self.assertEqual(base.n_coils,K)
                self.assertIsNone(base.n_features)
                self.assertTrue(base.normalize)
                self.assertEqual(gp.train_inputs[0].shape[-1],6*K)
                raw_noise = gp.likelihood.noise*gp.outcome_transform.stdvs.square().squeeze()
                torch.testing.assert_close(raw_noise,torch.full_like(raw_noise,self.noise))
                snapshots.append(len(gp.train_targets))
                return short_fit(mll)
            def estimate(gp,mapping,X):
                self.assertIs(mapping,BoUtils.map_coilwise)
                self.assertEqual(X.shape[-1],D)
                return {'max_ucb':-34.,'incumbent_lcb':-34.,'incumbent_index':0,
                        'estimated_regret':0.}
            with self.subTest(model_ok=model_ok), \
                 patch.object(BoLoop,'map_fn',side_effect=AssertionError('Periodic map used')), \
                 patch.object(BoLoop,'map_set',side_effect=AssertionError('Pooled map used')), \
                 patch.object(BoLoop,'fit_gpytorch_mll',side_effect=fit), \
                 patch.object(BoLoop,'_best_posterior_mean',return_value=-34.), \
                 patch.object(BoLoop,'optimize_acqf',side_effect=lambda **kw:(next(proposals),torch.tensor(-7.))), \
                 patch.object(BoLoop,'objective_function',return_value=(34.,self.noise)) as objective, \
                 patch.object(BoLoop,'calibrate_noise') as calibration, \
                 patch.object(BoLoop,'assess_model',return_value={'model_ok':model_ok,
                     'reasons':[] if model_ok else ['Overconfident model']}), \
                 patch.object(BoLoop,'SimpleRegretStopping',return_value=stopping), \
                 patch.object(stopping,'estimate',side_effect=estimate):
                result = BoLoop.bo_coil_set_kernel(max_iter=2,initial_data=self.initial,
                    calibration=self.calibration,return_diagnostics=True,
                    checkpoint=lambda state,checks:checkpoints.append(state.copy()),
                    model_callback=lambda checks:reports.append(list(checks)))
            calibration.assert_not_called()
            expected = 1 if model_ok else 2
            self.assertEqual(snapshots,list(range(5,6+expected)))
            self.assertEqual(len(result[1]),5+expected)
            self.assertEqual(len(reports),1+expected)
            self.assertEqual(result[5]['stop_reason'],
                             'estimated_simple_regret' if model_ok else 'iteration_budget')
            self.assertEqual(checkpoints[-1]['status'].item(),'ready')
            self.assertEqual(checkpoints[-1]['X'].shape,(5+expected,D))
            np.testing.assert_array_equal(checkpoints[-1]['variance'],np.full(5+expected,self.noise))
            self.assertTrue(all(c.kwargs['log_variance'] == self.noise for c in objective.call_args_list))

    def test_failed_predictive_selection_halts_before_fitting_or_new_simulations(self):
        checkpoints = []
        with patch.object(BoLoop,'select_coil_scales',return_value=(.5,1.,
                {'predictive_ok':False,'selection_skill':-.2})), \
             patch.object(BoLoop,'fit_gpytorch_mll') as fit, \
             patch.object(BoLoop,'objective_function') as simulate, \
             patch.object(BoLoop,'optimize_acquisition') as acquire, \
             self.assertRaises(ModelFittingError):
            BoLoop.bo_coil_set_kernel(max_iter=2,initial_data=self.initial,
                calibration=self.calibration,checkpoint=lambda state,checks:checkpoints.append(state))
        fit.assert_not_called()
        simulate.assert_not_called()
        acquire.assert_not_called()
        self.assertEqual(checkpoints[-1]['status'].item(),'model_fitting_failed')
        np.testing.assert_array_equal(checkpoints[-1]['y'],self.y)
        self.assertEqual(checkpoints[-1]['evaluations'].item(),0)

    def test_cli_routes_coil_set_with_log_nei_metadata(self):
        diagnostics = {'settings':{},'checks':[],'incumbent_index':0,'stop_reason':'iteration_budget'}
        result = (self.X.numpy(),self.y,np.array([]),[],[],diagnostics)
        validation = [{'mean_response':1.,'ci95_lower':.9,'ci95_upper':1.1}]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(bo_main,'sobol_sample',return_value=self.initial), \
             patch.object(bo_main,'calibrate_noise',return_value=self.calibration), \
             patch.object(bo_main,'bo_coil_set_kernel',return_value=result) as run, \
             patch.object(bo_main,'bo_rbf_old') as rbf, \
             patch.object(bo_main,'bo_multi_turbo') as turbo, \
             patch.object(bo_main,'evaluate_finalists',return_value=validation):
            output = Path(folder)/'run'
            search,_ = bo_main.main(['--method','bo','--kernel','coil-set','--iterations','1',
                                    '--output-dir',str(output),'--no-plot'])
            run.assert_called_once()
            rbf.assert_not_called()
            turbo.assert_not_called()
            self.assertIs(run.call_args.kwargs['calibration'],self.calibration)
            self.assertEqual(search['selection']['kernel'],'coil-set')
            settings = json.loads((output/'bo_settings.json').read_text())
            self.assertEqual(settings['acquisition'],'log-nei')
            self.assertIsNone(settings['nu'])
        with patch.object(bo_main,'sobol_sample') as sample, self.assertRaises(SystemExit):
            bo_main.main(['--method','bo','--kernel','coil-set','--nu','2.5'])
        sample.assert_not_called()


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
