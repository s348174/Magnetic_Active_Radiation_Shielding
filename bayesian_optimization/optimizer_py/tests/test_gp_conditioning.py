"""Deferred hyperparameter fitting still conditions exactly on every observation."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import warnings

import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.assorted import InputDataWarning
from gpytorch.kernels import RBFKernel, ScaleKernel

from GPConditioning import condition_on_observations


def make_gp(X,y,variance):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',InputDataWarning)
        gp = SingleTaskGP(X,y,train_Yvar=variance,
            covar_module=ScaleKernel(RBFKernel()).double(),
            outcome_transform=Standardize(m=1,batch_shape=X.shape[:-2]))
    gp.covar_module.initialize(outputscale=.8)
    gp.covar_module.base_kernel.initialize(lengthscale=.4)
    gp.mean_module.initialize(constant=.2)
    return gp.eval()


class GPConditioningTests(unittest.TestCase):
    def setUp(self):
        self.X = torch.linspace(0,1,8,dtype=torch.double).unsqueeze(-1)
        self.y = -34+torch.tensor([0.,.1,-.04,.05,.3,.6,-.5,1.],dtype=torch.double).unsqueeze(-1)
        self.variance = torch.linspace(.0002,.0005,8,dtype=torch.double).unsqueeze(-1)

    def test_preserves_units_parameters_and_matches_exact_reconstruction(self):
        gp = make_gp(self.X[:4],self.y[:4],self.variance[:4])
        mean,std = gp.outcome_transform.means.clone(),gp.outcome_transform.stdvs.clone()
        parameters = {name:value.clone() for name,value in gp.named_parameters()}
        gp.posterior(self.X[:2])
        self.assertIsNotNone(gp.prediction_strategy)
        result = condition_on_observations(gp,self.X,self.y,self.variance)
        self.assertIs(result,gp)
        self.assertIsNone(gp.prediction_strategy)
        torch.testing.assert_close(gp.outcome_transform.means,mean,atol=0,rtol=0)
        torch.testing.assert_close(gp.outcome_transform.stdvs,std,atol=0,rtol=0)
        torch.testing.assert_close(gp.train_targets,((self.y-mean)/std).squeeze(-1))
        torch.testing.assert_close(gp.likelihood.noise,self.variance.squeeze(-1)/std.squeeze().square())
        for name,value in gp.named_parameters():
            torch.testing.assert_close(value,parameters[name],atol=0,rtol=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',InputDataWarning)
            reference = SingleTaskGP(self.X,(self.y-mean)/std,
                train_Yvar=self.variance/std.square(),outcome_transform=None,
                covar_module=deepcopy(gp.covar_module),mean_module=deepcopy(gp.mean_module))
        Z = torch.tensor([[.08],[.37],[.83]],dtype=torch.double,requires_grad=True)
        posterior = gp.posterior(Z)
        expected = reference.posterior(Z)
        expected_mean = expected.mean*std+mean
        expected_variance = expected.variance*std.square()
        torch.testing.assert_close(posterior.mean,expected_mean,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(posterior.variance,expected_variance,atol=1e-12,rtol=1e-12)
        actual_grad, = torch.autograd.grad((posterior.mean+posterior.variance).sum(),Z)
        expected_grad, = torch.autograd.grad((expected_mean+expected_variance).sum(),Z)
        torch.testing.assert_close(actual_grad,expected_grad,atol=1e-11,rtol=1e-11)

    def test_repeated_updates_keep_all_data_and_original_standardization(self):
        gp = make_gp(self.X[:4],self.y[:4],self.variance[:4])
        transform_state = deepcopy(gp.outcome_transform.state_dict())
        for count in (5,6,8):
            condition_on_observations(gp,self.X[:count],self.y[:count,0],.0003)
            self.assertEqual(len(gp.train_targets),count)
            self.assertIsNone(gp.prediction_strategy)
            for name,value in gp.outcome_transform.state_dict().items():
                torch.testing.assert_close(value,transform_state[name],atol=0,rtol=0)
            self.assertTrue(torch.isfinite(gp.posterior(self.X[:2]).mean).all())

    def test_model_batch_shape_is_preserved(self):
        X = self.X.expand(2,-1,-1).clone()
        y = self.y.expand(2,-1,-1).clone()
        y[1] = 2*y[1]+10
        variance = self.variance.expand_as(y).clone()
        gp = make_gp(X[:,:4],y[:,:4],variance[:,:4])
        mean,std = gp.outcome_transform.means.clone(),gp.outcome_transform.stdvs.clone()
        condition_on_observations(gp,X,y,variance)
        self.assertEqual(gp.train_targets.shape,(2,8))
        torch.testing.assert_close(gp.train_targets,((y-mean)/std).squeeze(-1))
        torch.testing.assert_close(gp.likelihood.noise,(variance/std.square()).squeeze(-1))
        self.assertTrue(torch.isfinite(gp.posterior(X[:,:2]).variance).all())

    def test_checkpoint_parameters_and_saved_transform_restore_same_posterior(self):
        gp = make_gp(self.X[:4],self.y[:4],self.variance[:4])
        condition_on_observations(gp,self.X,self.y,self.variance)
        # Checkpoint contains raw data and state_dict arrays, not transformed targets.
        stored = {name:value.detach().cpu().numpy().copy() for name,value in gp.state_dict().items()}
        restored = make_gp(self.X,self.y,self.variance)
        restored.load_state_dict({name:torch.as_tensor(value) for name,value in stored.items()})
        condition_on_observations(restored,self.X,self.y,self.variance)
        original_posterior,restored_posterior = gp.posterior(self.X),restored.posterior(self.X)
        torch.testing.assert_close(original_posterior.mean,restored_posterior.mean,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(original_posterior.variance,restored_posterior.variance,atol=1e-12,rtol=1e-12)

    def test_invalid_data_leaves_model_and_cached_prediction_untouched(self):
        gp = make_gp(self.X[:4],self.y[:4],self.variance[:4])
        gp.posterior(self.X[:2])
        original = (gp.train_inputs,gp.train_targets,gp.likelihood.noise_covar.noise,gp.prediction_strategy)
        invalid_X = self.X.clone()
        invalid_X[0,0] = float('nan')
        for X,y,var in ((invalid_X,self.y,self.variance),(self.X,self.y,0.),
                         (self.X,self.y[:-1],self.variance),(self.X[:2],self.y[:2],.0003)):
            with self.assertRaises(ValueError):
                condition_on_observations(gp,X,y,var)
            self.assertIs(gp.train_inputs,original[0])
            self.assertIs(gp.train_targets,original[1])
            self.assertIs(gp.likelihood.noise_covar.noise,original[2])
            self.assertIs(gp.prediction_strategy,original[3])

    def test_update_failure_restores_previous_conditioning_state(self):
        gp = make_gp(self.X[:4],self.y[:4],self.variance[:4])
        gp.posterior(self.X[:2])
        original = (gp.train_inputs,gp.train_targets,gp.likelihood.noise_covar.noise,gp.prediction_strategy)
        with patch.object(gp.likelihood.noise_covar,'initialize',side_effect=RuntimeError('failed update')):
            with self.assertRaisesRegex(ModelFittingError,'previous state was restored'):
                condition_on_observations(gp,self.X,self.y,self.variance)
        self.assertIs(gp.train_inputs,original[0])
        self.assertIs(gp.train_targets,original[1])
        self.assertIs(gp.likelihood.noise_covar.noise,original[2])
        self.assertIs(gp.prediction_strategy,original[3])
        self.assertFalse(gp.training)
        self.assertTrue(torch.isfinite(gp.posterior(self.X[:2]).variance).all())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
