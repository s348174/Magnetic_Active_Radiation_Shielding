"""Noise scaling, rejected observations and the vanilla BO driver paths."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import RBFKernel, ScaleKernel

import BoLoop
import BoUtils
import bo_main
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
             patch.object(BoLoop,'stopping_criterion',return_value=False) as stopping:
            result = BoLoop.bo_rbf_old(max_iter=3,initial_data=self.initial)
        self.assertEqual([len(gp.train_targets) for gp in fitted],[4,5,6])
        self.assertEqual(result[0].shape,(6,D))
        self.assertFalse(fitted[-1].train_inputs[0].requires_grad)
        np.testing.assert_array_equal(result[2],[-7.,-7.]) # Keep log-NEI; do not exponentiate twice.
        self.assertTrue(all(isinstance(value,float) for value in result[3]))
        self.assertAlmostEqual(stopping.call_args_list[0].args[1],np.r_[self.variance,.04].mean())
        self.assertAlmostEqual(stopping.call_args_list[1].args[1],np.r_[self.variance,.04,.09].mean())

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
        result = (np.vstack((self.X.numpy(),extra)),np.r_[self.y,-.5],np.array([-7.]),[-.5],[.01])
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
                if kernel == 'matern':
                    self.assertEqual(run.call_args.kwargs['nu'],1.5)
                self.assertEqual(evaluate.call_args.kwargs['n_particles'],100000)
                np.testing.assert_array_equal(search['finalists'][0]['x'],extra[0])
                self.assertTrue((out/'initial_design.npz').is_file())
                history = np.load(out/'bo_history.npz')
                np.testing.assert_array_equal(history['y'],result[1])
                with open(out/'bo_settings.json') as file:
                    settings = json.load(file)
                self.assertEqual(settings['method'],'bo')
                self.assertEqual(settings['kernel'],kernel)

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
