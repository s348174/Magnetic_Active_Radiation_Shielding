"""Repeat means must reject a flat, overconfident surrogate."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel

from ModelDiagnostics import assess_model


class FixedGP:
    def __init__(self, mean, covariance):
        self.train_inputs = (torch.zeros(len(mean), 1, dtype=torch.double),)
        self.mean = torch.tensor(mean, dtype=torch.double).unsqueeze(-1)
        self.covariance = torch.tensor(covariance, dtype=torch.double)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=1)).double()
        self.covar_module.outputscale = 1.

    def posterior(self, X):
        return SimpleNamespace(mean=self.mean, distribution=SimpleNamespace(covariance_matrix=self.covariance))


class ModelDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.calibration = {'variance': 2.5e-5, 'X': np.array([[0.], [1.]]),
                            'mean': np.array([-34., -34.05]),
                            'mean_variance': np.full(2, 2.5e-5/6),
                            'names': ['reference', 'far'], 'repeats': 6}

    def test_flat_overconfident_model_fails(self):
        result = assess_model(FixedGP([-34.025, -34.025], np.eye(2)*1e-10), lambda X: X, self.calibration)
        self.assertFalse(result['model_ok'])
        self.assertFalse(result['contrast_rows'][0]['passed'])
        self.assertTrue(result['contrast_rows'][0]['measured_contrast_significant'])

    def test_model_matching_repeats_passes(self):
        state = torch.random.get_rng_state().clone()
        result = assess_model(FixedGP([-34., -34.05], np.eye(2)*1e-10), lambda X: X, self.calibration)
        self.assertTrue(result['model_ok'])
        self.assertAlmostEqual(result['predicted_mean_range'], .05)
        torch.testing.assert_close(state, torch.random.get_rng_state())

    def test_uncertainty_uses_repeat_mean_variance_without_latent_floor(self):
        result = assess_model(FixedGP([-34., -34.05], np.eye(2)*1e-14), lambda X: X, self.calibration)
        row = result['calibration_rows'][0]
        self.assertEqual(row['latent_variance'], 1e-14)
        self.assertAlmostEqual(row['allowed_error'], result['critical_value']*np.sqrt(1e-14+2.5e-5/6))

    def test_contrast_uses_joint_covariance(self):
        # A common uncertain offset does not explain a missed configuration effect.
        covariance = np.array([[1e-3, .9999e-3], [.9999e-3, 1e-3]])
        result = assess_model(FixedGP([-34.01, -34.04], covariance), lambda X: X, self.calibration)
        self.assertTrue(all(row['passed'] for row in result['calibration_rows']))
        self.assertFalse(result['contrast_rows'][0]['passed'])
        self.assertFalse(result['model_ok'])

    def test_nonfinite_model_cannot_pass(self):
        result = assess_model(FixedGP([float('nan'), -34.05], np.eye(2)), lambda X: X, self.calibration)
        self.assertFalse(result['model_ok'])


if __name__ == '__main__':
    unittest.main()
