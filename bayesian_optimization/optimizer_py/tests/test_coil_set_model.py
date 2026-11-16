"""Fixed geometry scales, training-only candidates and predictive scale selection."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

import CoilSetModel as model
from RobustGPFitting import _fresh_parameters


class CoilModelTests(unittest.TestCase):
    def setUp(self):
        self.X = torch.zeros(30,6,dtype=torch.double)
        self.X[:,0] = torch.arange(30,dtype=torch.double)/10
        self.X[:,5] = 1.
        self.y = 1+self.X[:,0]
        self.variance = torch.full((30,1), .01, dtype=torch.double)

    def test_selected_geometry_parameters_stay_fixed_during_fresh_fit(self):
        kernel = model.make_coil_kernel(1, lengthscale=.25, kappa=4.)
        gp = SingleTaskGP(self.X[:4],self.y[:4,None],train_Yvar=self.variance[:4],
                          covar_module=kernel)
        _fresh_parameters(ExactMarginalLogLikelihood(gp.likelihood,gp))
        self.assertAlmostEqual(kernel.base_kernel.lengthscale.item(),.25)
        self.assertAlmostEqual(kernel.base_kernel.kappa.item(),4.)
        self.assertFalse(kernel.base_kernel.raw_lengthscale.requires_grad)
        self.assertFalse(kernel.base_kernel.raw_kappa.requires_grad)
        self.assertTrue(kernel.raw_outputscale.requires_grad)
        self.assertTrue(gp.mean_module.constant.requires_grad)

    def test_selection_uses_only_training_observations_and_preserves_rng(self):
        fits = []
        def fit(X,y,variance,n_coils,lengthscale,kappa):
            fits.append((X.clone(),y.clone(),variance.clone()))
            return SimpleNamespace(ell=lengthscale,kappa=kappa)
        def predict(gp,X):
            target = 1+X[:,:1]
            return (target if gp.ell == 1. and gp.kappa == 1. else target+1.), torch.ones_like(target)
        rng = torch.random.get_rng_state().clone()
        with patch.object(model,'_fit_candidate',side_effect=fit), \
             patch.object(model,'raw_mean_variance',side_effect=predict):
            ell,kappa,report = model.select_coil_scales(
                self.X,self.y,self.variance,1,lengthscales=(.5,1.),kappas=(.25,1.))
        torch.testing.assert_close(torch.random.get_rng_state(),rng)
        self.assertEqual((ell,kappa),(1.,1.))
        self.assertTrue(report['predictive_ok'])
        self.assertEqual(report['best_rmse'],0.)
        self.assertEqual(report['selection_skill'],1.)
        self.assertFalse(report['independent_validation'])
        training = report['train_indices']
        validation = report['validation_indices']
        self.assertFalse(set(training)&set(validation))
        self.assertEqual(set(training)|set(validation),set(range(30)))
        for X,y,v in fits:
            torch.testing.assert_close(X,self.X[training])
            torch.testing.assert_close(y,self.y[training,None])
            torch.testing.assert_close(v,self.variance[training])
        baseline = self.y[training].mean()
        expected = float((self.y[validation]-baseline).square().mean().sqrt())
        self.assertAlmostEqual(report['baseline_rmse'],expected)

    def test_a_finite_but_useless_predictor_does_not_pass(self):
        with patch.object(model,'_fit_candidate',return_value=SimpleNamespace()), \
             patch.object(model,'raw_mean_variance',side_effect=lambda gp,X:(
                 torch.zeros(len(X),1,dtype=torch.double),torch.ones(len(X),1,dtype=torch.double))):
            _,_,report = model.select_coil_scales(self.X,self.y,.01,1,
                                                  lengthscales=(.5,),kappas=(1.,))
        self.assertFalse(report['predictive_ok'])
        self.assertLess(report['selection_skill'],0.)

    def test_small_initial_design_is_explicitly_unvalidated(self):
        with patch.object(model,'_fit_candidate') as fit:
            ell,kappa,report = model.select_coil_scales(self.X[:4],self.y[:4],.01,1)
        self.assertEqual((ell,kappa),(.5,1.))
        self.assertIsNone(report['predictive_ok'])
        self.assertEqual(report['validation_count'],0)
        fit.assert_not_called()

    def test_invalid_or_failed_candidates_are_not_selected(self):
        with patch.object(model,'_fit_candidate',side_effect=ModelFittingError('failed')):
            with self.assertRaisesRegex(ModelFittingError,'All fixed-scale'):
                model.select_coil_scales(self.X,self.y,.01,1,lengthscales=(.5,),kappas=(1.,))
        for variance in [float('nan'),0.,-.01]:
            with self.assertRaises(ValueError):
                model.select_coil_scales(self.X,self.y,variance,1)

    def test_actual_fixed_scale_candidate_has_finite_checked_predictions(self):
        y = torch.sin(self.X[:,0])
        ell,kappa,report = model.select_coil_scales(self.X,y,.001,1,
                                                  lengthscales=(.5,),kappas=(1.,))
        self.assertEqual((ell,kappa),(.5,1.))
        self.assertTrue(report['candidates'][0]['valid'])
        self.assertTrue(np.isfinite(report['best_rmse']))
        self.assertTrue(report['predictive_ok'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
