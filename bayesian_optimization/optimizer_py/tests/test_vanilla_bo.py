"""Noise scaling, rejected observations and the vanilla BO driver paths."""
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import RBFKernel, ScaleKernel

import BoLoop
import BoUtils
import bo_main
import RegretStopping
from input import D


def short_fit(mll):
    return fit_gpytorch_mll(mll, optimizer_kwargs={'options':{'maxiter':3}})


class VanillaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        self.X = .1 + .8*torch.rand(4,D,dtype=torch.double)
        self.y = np.array([-1.,-2.,-3.,-4.])
        self.variance = np.array([.01,.04,.09,.16])
        self.initial = (self.X,self.y,self.variance)

    def test_known_variances_are_standardized_with_targets(self):
        variance = torch.tensor(self.variance,dtype=torch.double).unsqueeze(-1)
        with patch.object(BoLoop,'fit_gpytorch_mll',side_effect=short_fit):
            gp = BoLoop._fit_model(self.X,self.y,variance,BoUtils.identity_map,
                                   ScaleKernel(RBFKernel(ard_num_dims=D)))
        expected = variance / gp.outcome_transform.stdvs.square()
        torch.testing.assert_close(gp.likelihood.noise,expected.squeeze(-1))
        with torch.no_grad():
            # BoTorch returns posterior uncertainties in the original target units.
            latent = gp.posterior(self.X).variance
            observed = gp.posterior(self.X,observation_noise=True).variance
            explicit = gp.posterior(self.X,observation_noise=expected).variance
        # Fixed-noise models use average training noise for an unspecified new point.
        torch.testing.assert_close(observed-latent,variance.mean().expand_as(variance))
        torch.testing.assert_close(explicit-latent,variance)

    def test_invalid_variance_never_enters_training_data(self):
        fitted = []
        def fit(mll):
            result = short_fit(mll)
            fitted.append(mll.model)
            return result
        proposed = iter([torch.full((1,D),v,dtype=torch.double,requires_grad=True)
                         for v in (.25,.5,.75)])
        def optimize(**kwargs):
            if isinstance(kwargs['acq_function'],BoUtils.InputMappedAcquisition):
                return torch.full((1,D),.5,dtype=torch.double),torch.tensor(-1.)
            return next(proposed),torch.tensor(-7.)
        with patch.object(BoLoop,'fit_gpytorch_mll',side_effect=fit), \
             patch.object(BoLoop,'optimize_acqf',side_effect=optimize), \
             patch.object(BoLoop,'objective_function',side_effect=[(1.,np.nan),(1.,.04),(1.,.09)]), \
             patch.object(BoLoop.SimpleRegretStopping,'check',return_value=False) as stopping:
            result = BoLoop.bo_rbf_old(max_iter=3,initial_data=self.initial)
        self.assertEqual([len(gp.train_targets) for gp in fitted],[4,5,6])
        self.assertEqual(result[0].shape,(6,D))
        self.assertFalse(fitted[-1].train_inputs[0].requires_grad)
        np.testing.assert_array_equal(result[2],[-7.,-7.]) # Keep log-NEI; do not exponentiate twice.
        self.assertTrue(all(isinstance(value,float) for value in result[3]))
        self.assertEqual([call.args[3] for call in stopping.call_args_list],[1,2])

    def test_invalid_initial_variance_is_removed(self):
        initial = (self.X,self.y,np.array([np.nan,.04,.09,-.1]))
        with patch.object(BoLoop,'fit_gpytorch_mll',side_effect=short_fit), \
             patch.object(BoLoop,'optimize_acqf',return_value=(self.X[1:2],torch.tensor(-7.))):
            result = BoLoop.bo_rbf_old(max_iter=1,initial_data=initial)
        np.testing.assert_array_equal(result[1],self.y[1:3])

    def test_single_point_contract_is_checked_before_sampling(self):
        with patch.object(BoLoop,'Q',2),patch.object(BoLoop,'sobol_sample') as sample:
            with self.assertRaises(ValueError):
                BoLoop.bo_rbf_old()
            sample.assert_not_called()

    def test_driver_routes_all_kernels_and_validates_finalists(self):
        extra = np.full((1,D),.5)
        diagnostics = {'settings':{}, 'checks':[], 'stop_reason':'iteration_budget',
                       'successful_evaluations':1, 'attempted_iterations':1, 'incumbent_index':4}
        result = (np.vstack((self.X.numpy(),extra)),np.r_[self.y,-.5],np.array([-7.]),[-.5],[.01],diagnostics)
        validation = [{'mean_response':1.,'ci95_lower':.8,'ci95_upper':1.2}]
        for kernel,function in [('matern','bo_matern_kernel'),('rbf','bo_rbf_old'),('rbf-set','bo_rbf_kernel')]:
            with self.subTest(kernel=kernel),tempfile.TemporaryDirectory() as root, \
                 patch.object(bo_main,'sobol_sample',return_value=self.initial) as sample, \
                 patch.object(bo_main,function,return_value=result) as run, \
                 patch.object(bo_main,'bo_multi_turbo') as turbo, \
                 patch.object(bo_main,'evaluate_finalists',return_value=validation) as evaluate:
                out = Path(root)/'run'
                args = ['--method','bo','--kernel',kernel,'--iterations','1','--output-dir',str(out),'--no-plot']
                if kernel == 'matern':
                    args += ['--nu','1.5']
                search,_ = bo_main.main(args)
                sample.assert_called_once()
                turbo.assert_not_called()
                self.assertIs(run.call_args.kwargs['initial_data'],self.initial)
                self.assertEqual(run.call_args.kwargs['max_iter'],1)
                self.assertTrue(run.call_args.kwargs['return_diagnostics'])
                if kernel == 'matern':
                    self.assertEqual(run.call_args.kwargs['nu'],1.5)
                self.assertEqual(evaluate.call_args.kwargs['n_particles'],100000)
                np.testing.assert_array_equal(search['finalists'][0]['x'],extra[0])
                self.assertTrue((out/'initial_design.npz').is_file())
                self.assertTrue((out/'regret_history.csv').is_file())
                history = np.load(out/'bo_history.npz')
                np.testing.assert_array_equal(history['y'],result[1])
                with open(out/'bo_settings.json') as file:
                    settings = json.load(file)
                self.assertEqual(settings['method'],'bo')
                self.assertEqual(settings['kernel'],kernel)


class LinearGP(torch.nn.Module):
    num_outputs = 1

    def __init__(self, offset=0., sigma=.05):
        super().__init__()
        self.offset, self.sigma = offset, sigma

    def posterior(self, X, **kwargs):
        mean = X[..., :1] + self.offset
        return SimpleNamespace(mean=mean, variance=torch.full_like(mean,self.sigma**2))


class RegretTests(unittest.TestCase):
    def setUp(self):
        self.X = torch.stack([torch.full((D,),value,dtype=torch.double) for value in (.1,.8)])

    def test_regret_formula_and_invariance_to_log_offset(self):
        stopping = RegretStopping.SimpleRegretStopping(grid_size=16)
        for offset in (0.,-34.5):
            result = stopping.estimate(LinearGP(offset),BoUtils.identity_map,self.X)
            self.assertEqual(result['incumbent_index'],1)
            self.assertAlmostEqual(result['max_ucb'],1.1+offset,places=6)
            self.assertAlmostEqual(result['incumbent_lcb'],.7+offset,places=6)
            self.assertAlmostEqual(result['estimated_regret'],.4,places=6)
        self.assertAlmostEqual(np.expm1(stopping.delta),stopping.relative_tolerance)

    def test_grid_and_observations_protect_against_a_poor_local_maximum(self):
        stopping = RegretStopping.SimpleRegretStopping(grid_size=16)
        X = torch.ones(2,D,dtype=torch.double)
        # The optimizer's reported value is deliberately wrong; evaluate its point afresh.
        with patch.object(RegretStopping,'optimize_acqf',return_value=(torch.zeros(1,D,dtype=torch.double),torch.tensor(-100.))):
            result = stopping.estimate(LinearGP(),BoUtils.identity_map,X)
        self.assertAlmostEqual(result['max_ucb'],1.1,places=6)
        self.assertAlmostEqual(result['estimated_regret'],.2,places=6)

    def test_incumbent_uses_uncertainty_not_just_the_largest_mean(self):
        class UnequalNoiseGP(LinearGP):
            def posterior(self, X, **kwargs):
                return SimpleNamespace(mean=X[..., :1],variance=(.1+X[..., :1]).square())
        index,lower = RegretStopping.lower_bound_incumbent(UnequalNoiseGP(),BoUtils.identity_map,self.X,2.)
        self.assertEqual(index,0)
        self.assertAlmostEqual(lower,-.3)

    def test_minimum_cadence_and_consecutive_passes(self):
        stopping = RegretStopping.SimpleRegretStopping(min_evaluations=2,check_every=2,patience=3,grid_size=8)
        good = {'max_ucb':1.,'incumbent_lcb':1.,'incumbent_index':0,'estimated_regret':0.}
        bad = dict(good,estimated_regret=.1)
        with patch.object(stopping,'estimate',side_effect=[good,good,bad,good,good,good]) as estimate:
            self.assertFalse(stopping.check(None,None,None,1,1))
            self.assertFalse(stopping.check(None,None,None,2,2))
            self.assertFalse(stopping.check(None,None,None,3,3))
            self.assertEqual(estimate.call_count,1)
            for count in (4,6,8,10):
                self.assertFalse(stopping.check(None,None,None,count,count))
            self.assertTrue(stopping.check(None,None,None,12,12))
        self.assertEqual([row['consecutive_passes'] for row in stopping.history],[1,2,0,1,2,3])

    def test_failed_search_resets_patience_and_cannot_stop(self):
        stopping = RegretStopping.SimpleRegretStopping(min_evaluations=1,check_every=1,patience=2,grid_size=8)
        good = {'max_ucb':1.,'incumbent_lcb':1.,'incumbent_index':0,'estimated_regret':0.}
        with patch.object(stopping,'estimate',side_effect=[good,RuntimeError('test failure')]):
            self.assertFalse(stopping.check(None,None,None,1,1))
            with self.assertWarns(UserWarning):
                self.assertFalse(stopping.check(None,None,None,2,2))
        self.assertEqual(stopping.consecutive,0)
        self.assertFalse(stopping.history[-1]['search_ok'])

    def test_stable_mean_with_unresolved_uncertainty_does_not_stop(self):
        stopping = RegretStopping.SimpleRegretStopping(min_evaluations=1,check_every=1,patience=1,grid_size=16)
        self.assertFalse(stopping.check(LinearGP(sigma=.01),BoUtils.identity_map,
                                        torch.ones(2,D,dtype=torch.double),1,1))
        self.assertGreater(stopping.history[-1]['estimated_regret'],stopping.delta)

    def test_actual_ucb_gradients_with_periodic_and_pooled_features(self):
        X = .1+.8*torch.rand(4,D,dtype=torch.double,generator=torch.Generator().manual_seed(67))
        y = np.array([-1.,-2.,-3.,-4.])
        variance = torch.full((4,1),.01,dtype=torch.double)
        stopping = RegretStopping.SimpleRegretStopping(grid_size=16)
        for mapping in (BoUtils.periodic_feature_map,BoUtils.map_set):
            with self.subTest(mapping=mapping.__name__),patch.object(BoLoop,'fit_gpytorch_mll',side_effect=short_fit):
                gp = BoLoop._fit_model(X,y,variance,mapping,
                        ScaleKernel(RBFKernel(ard_num_dims=mapping(X).shape[-1])))
                result = stopping.estimate(gp,mapping,X)
            self.assertTrue(np.isfinite(result['estimated_regret']))
            self.assertGreaterEqual(result['max_ucb'],result['incumbent_lcb'])

    def test_all_vanilla_loops_stop_and_return_diagnostics(self):
        initial = (self.X,np.array([-1.,-2.]),np.full(2,.01))
        for function in (BoLoop.bo_matern_kernel,BoLoop.bo_rbf_old,BoLoop.bo_rbf_kernel):
            stopping = RegretStopping.SimpleRegretStopping(min_evaluations=1,check_every=1,patience=2,grid_size=8)
            proposals = iter([torch.full((1,D),v,dtype=torch.double) for v in (.3,.6)])
            good = {'max_ucb':1.,'incumbent_lcb':1.,'incumbent_index':0,'estimated_regret':0.}
            with self.subTest(function=function.__name__), \
                 patch.object(BoLoop,'SimpleRegretStopping',return_value=stopping), \
                 patch.object(stopping,'estimate',return_value=good), \
                 patch.object(BoLoop,'fit_gpytorch_mll',side_effect=short_fit), \
                 patch.object(BoLoop,'_best_posterior_mean',return_value=-1.), \
                 patch.object(BoLoop,'optimize_acqf',side_effect=lambda **kwargs:(next(proposals),torch.tensor(-1.))), \
                 patch.object(BoLoop,'objective_function',return_value=(1.,.01)):
                result = function(max_iter=5,initial_data=initial,return_diagnostics=True)
            self.assertEqual(len(result[1]),4)
            self.assertEqual(result[5]['stop_reason'],'estimated_simple_regret')
            self.assertEqual(result[5]['successful_evaluations'],2)
            self.assertEqual(len(result[5]['checks']),2)

    def test_lcb_incumbent_is_validated_even_when_not_best_observed(self):
        initial = (self.X,np.array([-1.,-2.]),np.full(2,.01))
        diagnostics = {'settings':{},'checks':[],'incumbent_index':1,'stop_reason':'iteration_budget'}
        result = (self.X.numpy(),initial[1],np.array([]),[],[],diagnostics)
        with patch.object(bo_main,'sobol_sample',return_value=initial), \
             patch.object(bo_main,'bo_rbf_old',return_value=result),patch.object(bo_main,'FINALISTS',1):
            search = bo_main.run_vanilla(kernel='rbf',iterations=1)
        np.testing.assert_array_equal(search['finalists'][0]['x'],self.X[1].numpy())

    def test_disable_and_invalid_configuration(self):
        stopping = RegretStopping.SimpleRegretStopping(enabled=False,grid_size=8)
        self.assertFalse(stopping.check(None,None,None,10000,10000))
        self.assertEqual(stopping.history,[])
        for kwargs in ({'relative_tolerance':0.},{'confidence':float('nan')},{'patience':0}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                RegretStopping.SimpleRegretStopping(**kwargs)

    def test_incompatible_cli_options_fail_before_simulation(self):
        for args in (['--method','bo','--regions','2'],
                     ['--method','turbo','--kernel','rbf'],
                     ['--method','bo','--kernel','rbf','--nu','1.5'],
                     ['--method','bo','--iterations','0']):
            with self.subTest(args=args),patch.object(bo_main,'sobol_sample') as sample, \
                 patch.object(bo_main,'bo_multi_turbo') as turbo:
                with self.assertRaises(SystemExit):
                    bo_main.main(args)
                sample.assert_not_called()
                turbo.assert_not_called()


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
