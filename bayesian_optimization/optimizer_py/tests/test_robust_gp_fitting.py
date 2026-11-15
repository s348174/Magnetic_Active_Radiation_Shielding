"""Different fitting starts, finite-result checks and bounded Adam recovery."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.models import SingleTaskGP
from botorch.optim.core import OptimizationStatus
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

import RobustGPFitting as fitting


class RobustFitTests(unittest.TestCase):
    def setUp(self):
        X = torch.linspace(0., 1., 8, dtype=torch.double).unsqueeze(-1)
        y = torch.sin(X*3.)
        self.gp = SingleTaskGP(X, y, train_Yvar=torch.full_like(y, .01),
                               covar_module=ScaleKernel(RBFKernel(ard_num_dims=1)))
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)

    def test_warm_start_has_larger_line_search_budget_and_one_attempt(self):
        with patch.object(fitting, 'fit_gpytorch_mll', return_value=self.mll) as fit:
            self.assertIs(fitting.fit_gpytorch_mll_robust(self.mll), self.mll)
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(fit.call_args.kwargs,
                         {'max_attempts':1, 'optimizer_kwargs':{'options':{'maxiter':300,'maxls':50}}})
        self.assertFalse(self.mll.training)

    def test_failed_warm_start_uses_fresh_parameters_without_replacing_kernel(self):
        self.gp.covar_module.initialize(outputscale=.001)
        self.gp.covar_module.base_kernel.initialize(lengthscale=8.)
        self.gp.mean_module.initialize(constant=-2.)
        kernel = self.gp.covar_module
        starts = []
        def fit(mll, **kwargs):
            starts.append((mll.model.covar_module.outputscale.item(),
                           mll.model.covar_module.base_kernel.lengthscale.item(),
                           mll.model.mean_module.constant.item()))
            if len(starts) == 1:
                mll.model.covar_module.initialize(outputscale=float('nan'))
                raise ModelFittingError('ABNORMAL')
            return mll.eval()
        with patch.object(fitting, 'fit_gpytorch_mll', side_effect=fit):
            fitting.fit_gpytorch_mll_robust(self.mll)
        self.assertIs(self.gp.covar_module, kernel)
        self.assertAlmostEqual(starts[0][0], .001)
        self.assertAlmostEqual(starts[0][1], 8.)
        self.assertEqual(starts[1], (1., 1., 0.))

    def test_nonfinite_parameters_cannot_be_accepted_as_success(self):
        calls = []
        def fit(mll, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                mll.model.mean_module.initialize(constant=float('nan'))
            return mll.eval()
        with patch.object(fitting, 'fit_gpytorch_mll', side_effect=fit):
            fitting.fit_gpytorch_mll_robust(self.mll)
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.isfinite(self.gp.mean_module.constant))

    def test_nonfinite_likelihood_cannot_be_accepted(self):
        with patch.object(fitting, 'fit_gpytorch_mll', return_value=self.mll), \
             patch.object(self.mll, 'forward', return_value=torch.tensor(float('nan'))):
            with self.assertRaises(ModelFittingError):
                fitting.fit_gpytorch_mll_robust(self.mll)

    def test_all_failures_restore_original_parameters(self):
        original = deepcopy(self.mll.state_dict())
        def fail(mll, **kwargs):
            mll.model.mean_module.initialize(constant=12.)
            raise ModelFittingError('synthetic failure')
        with patch.object(fitting, 'fit_gpytorch_mll', side_effect=fail) as fit:
            with self.assertRaisesRegex(ModelFittingError, 'No failed parameters were accepted'):
                fitting.fit_gpytorch_mll_robust(self.mll)
        self.assertEqual(fit.call_count, 3)
        for key, value in original.items():
            torch.testing.assert_close(self.mll.state_dict()[key], value)

    def test_adam_checks_optimizer_status_and_finite_result(self):
        for status, loss in [(OptimizationStatus.FAILURE, 1.),
                             (OptimizationStatus.STOPPED, float('nan'))]:
            result = SimpleNamespace(status=status, fval=loss, step=1, message='test')
            with self.subTest(status=status, loss=loss), \
                 patch.object(fitting, 'fit_gpytorch_mll_torch', return_value=result):
                with self.assertRaises(ModelFittingError):
                    fitting._fit_adam(self.mll)
        result = SimpleNamespace(status=OptimizationStatus.STOPPED,fval=1.,step=300,message='budget')
        with patch.object(fitting, 'fit_gpytorch_mll_torch', return_value=result) as fit:
            fitting._fit_adam(self.mll)
        self.assertEqual(fit.call_args.kwargs['step_limit'], 300)
        self.assertEqual(fit.call_args.kwargs['optimizer'].keywords['lr'], .05)

    def test_actual_adam_fallback_fits_finite_training_likelihood(self):
        original = fitting.fit_gpytorch_mll
        def fit(mll, **kwargs):
            if 'optimizer' not in kwargs:
                raise ModelFittingError('forced L-BFGS-B failure')
            return original(mll, **kwargs)
        with patch.object(fitting, 'fit_gpytorch_mll', side_effect=fit) as dispatcher:
            fitted = fitting.fit_gpytorch_mll_robust(self.mll)
        self.assertEqual(dispatcher.call_count, 3)
        self.assertFalse(fitted.training)
        self.assertTrue(torch.isfinite(self.gp.posterior(self.gp.train_inputs[0]).mean).all())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
