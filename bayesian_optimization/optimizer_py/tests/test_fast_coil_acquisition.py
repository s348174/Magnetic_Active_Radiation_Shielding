"""Fast EI keeps BoTorch's stable arithmetic and checks covariance before smoothing."""
from types import SimpleNamespace
import unittest

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize

from CoilSetModel import make_coil_kernel
from FastCoilAcquisition import CheckedLogExpectedImprovement


class RawVarianceModel(Model):
    def __init__(self, variance):
        super().__init__()
        self.variance = variance

    @property
    def num_outputs(self):
        return 1

    def covar_module(self, X, Y, diag=False):
        return torch.ones(X.shape[:-1],dtype=X.dtype,device=X.device)

    def posterior(self, X, **kwargs):
        variance = X[..., :1]*0+self.variance
        return SimpleNamespace(mean=X[..., :1],
            distribution=SimpleNamespace(covariance_matrix=variance))


class FastCoilAcquisitionTests(unittest.TestCase):
    def test_values_and_gradients_match_stock_log_ei(self):
        generator = torch.Generator().manual_seed(67)
        X = torch.randn(20,12,dtype=torch.double,generator=generator)
        y = X[:,:1].sin()
        gp = SingleTaskGP(X,y,train_Yvar=torch.full_like(y,.02),
            covar_module=make_coil_kernel(2),outcome_transform=Standardize(m=1))
        Z = torch.randn(4,1,12,dtype=torch.double,generator=generator).requires_grad_()
        for maximize in (True,False):
            expected = LogExpectedImprovement(gp,best_f=.3,maximize=maximize)(Z)
            actual = CheckedLogExpectedImprovement(gp,best_f=.3,maximize=maximize)(Z)
            torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-10)
            a, = torch.autograd.grad(actual.sum(),Z)
            b, = torch.autograd.grad(expected.sum(),Z)
            torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)

    def test_material_negative_raw_variance_is_rejected(self):
        X = torch.zeros(2,1,1,dtype=torch.double)
        acquisition = CheckedLogExpectedImprovement(RawVarianceModel(-.01),best_f=0.)
        with self.assertRaisesRegex(RuntimeError,'negative latent variance'):
            acquisition(X)

    def test_excess_prior_variance_is_rejected(self):
        X = torch.zeros(2,1,1,dtype=torch.double)
        acquisition = CheckedLogExpectedImprovement(RawVarianceModel(2.),best_f=0.)
        with self.assertRaisesRegex(RuntimeError,'exceeds its prior'):
            acquisition(X)

    def test_tiny_and_zero_raw_variances_have_finite_values_and_gradients(self):
        for variance in (0.,1e-300,1e-14):
            for location in (-1.,0.,1.):
                X = torch.full((2,1,1),location,dtype=torch.double,requires_grad=True)
                model = RawVarianceModel(variance)
                acquisition = CheckedLogExpectedImprovement(model,best_f=0.)
                values = acquisition(X)
                gradient, = torch.autograd.grad(values.sum(),X)
                self.assertTrue(torch.isfinite(values).all())
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertEqual(model.variance,variance)

    def test_q_greater_than_one_is_rejected(self):
        acquisition = CheckedLogExpectedImprovement(RawVarianceModel(.1),best_f=0.)
        with self.assertRaises(AssertionError):
            acquisition(torch.zeros(2,2,1,dtype=torch.double))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
