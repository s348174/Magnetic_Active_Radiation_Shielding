"""Raw posterior covariance and explicit regret components, without accessor floors."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from gpytorch.kernels import RBFKernel, ScaleKernel

import RegretStopping
from input import D


class RawGP(torch.nn.Module):
    num_outputs = 1

    def __init__(self,variance=1e-14,slope=0.):
        super().__init__()
        self.raw_variance = variance
        self.slope = slope
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=D)).double()
        self.covar_module.outputscale = .1

    def posterior(self,X):
        mean = self.slope*X[...,:1]
        variance = torch.full_like(mean,self.raw_variance)
        covariance = torch.diag_embed(variance.squeeze(-1))
        # An exaggerated accessor value must not enter the numerical check.
        return SimpleNamespace(mean=mean,variance=torch.full_like(mean,1.),
                               distribution=SimpleNamespace(covariance_matrix=covariance))


class RegretNumericalTests(unittest.TestCase):
    def setUp(self):
        self.X = torch.stack([torch.full((D,),v,dtype=torch.double) for v in (.1,.8)])
        self.candidate = torch.ones(1,D,dtype=torch.double)

    def test_small_positive_raw_variance_is_not_replaced_by_accessor_floor(self):
        stopping = RegretStopping.SimpleRegretStopping(grid_size=8)
        with patch.object(RegretStopping,'optimize_acqf',return_value=(self.candidate,torch.tensor(1.))):
            result = stopping.estimate(RawGP(),lambda X:X,self.X)
        self.assertAlmostEqual(result['estimated_regret'],4e-7,places=14)
        self.assertEqual(result['mean_gap'],0.)
        self.assertAlmostEqual(result['uncertainty_gap'],4e-7,places=14)

    def test_gap_decomposition_and_selected_incumbent(self):
        stopping = RegretStopping.SimpleRegretStopping(grid_size=8)
        with patch.object(RegretStopping,'optimize_acqf',return_value=(self.candidate,torch.tensor(-100.))):
            result = stopping.estimate(RawGP(variance=.0001,slope=1.),lambda X:X,self.X)
        self.assertEqual(result['incumbent_index'],1)
        self.assertAlmostEqual(result['upper_mean'],1.)
        self.assertAlmostEqual(result['incumbent_mean'],.8)
        self.assertAlmostEqual(result['upper_sigma'],.01)
        self.assertAlmostEqual(result['incumbent_sigma'],.01)
        self.assertAlmostEqual(result['mean_gap'],.2)
        self.assertAlmostEqual(result['uncertainty_gap'],.04)
        self.assertAlmostEqual(result['estimated_regret'],.24)
        self.assertAlmostEqual(result['max_ucb']-result['incumbent_lcb'],
                               result['mean_gap']+result['uncertainty_gap'])

    def test_invalid_covariance_cannot_count_as_stopping_evidence(self):
        for variance in (-1e-5,.2):
            with self.subTest(variance=variance):
                stopping = RegretStopping.SimpleRegretStopping(enabled=True,min_evaluations=1,
                    check_every=1,record_every=1,patience=1,grid_size=8)
                with self.assertWarns(UserWarning):
                    self.assertFalse(stopping.check(RawGP(variance),lambda X:X,self.X,1,1))
                self.assertFalse(stopping.history[-1]['search_ok'])
                self.assertEqual(stopping.consecutive,0)
                self.assertEqual(stopping.history[-1]['estimated_regret'],float('inf'))

    def test_zero_variance_ucb_keeps_finite_gradients(self):
        gp = RawGP(variance=0.,slope=1.)
        acquisition = RegretStopping.RawUpperConfidenceBound(gp,2.)
        X = self.X[:1].unsqueeze(-2).clone().requires_grad_(True)
        value = acquisition(X)
        gradient, = torch.autograd.grad(value.sum(),X)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertAlmostEqual(gradient[0,0,0].item(),1.)

    def test_fast_records_never_replace_optimized_stopping_decisions(self):
        stopping = RegretStopping.SimpleRegretStopping(enabled=True,fast_record=True,
            min_evaluations=2,check_every=2,record_every=1,patience=1,grid_size=8)
        gp = RawGP(variance=1e-14)
        with patch.object(RegretStopping,'optimize_acqf',return_value=(self.candidate,torch.tensor(1.))) as optimize:
            self.assertFalse(stopping.check(gp,lambda X:X,self.X,1,1))
            optimize.assert_not_called()
            self.assertEqual(stopping.consecutive,0)
            self.assertEqual(stopping.history[-1]['search_method'],'grid')
            self.assertFalse(stopping.history[-1]['stopping_check'])
            self.assertTrue(stopping.check(gp,lambda X:X,self.X,2,2))
            optimize.assert_called_once()
            self.assertEqual(stopping.history[-1]['search_method'],'optimized')
            self.assertTrue(stopping.history[-1]['stopping_check'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
