"""Measured scramble noise, geometry conventions and seed separation."""
import json
import unittest
from unittest.mock import patch

import numpy as np

import NoiseCalibration as calibration
import input as config


class NoiseCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.full((2, config.D), .5)
        self.X[1] = .6
        self.y = np.array([-1., -2.])

    def test_single_run_variance_is_pooled_within_cases_not_between_them(self):
        repeats = 6
        scores = [10+10*j+.01*(j+1)*np.arange(repeats) for j in range(3)]
        responses = [(float(np.exp(s)), 1e30) for group in scores for s in group]
        with patch.object(calibration, 'evaluate_configuration', side_effect=responses) as evaluate:
            result = calibration.calibrate_noise(self.X, self.y, repeats)
        expected = [np.var(group, ddof=1) for group in scores]
        np.testing.assert_allclose(result['single_run_variance'], expected)
        np.testing.assert_allclose(result['mean_variance'], np.array(expected)/repeats)
        np.testing.assert_allclose(result['mean'], [-group.mean() for group in scores])
        self.assertAlmostEqual(result['variance'], np.mean(expected))
        self.assertNotAlmostEqual(result['variance'], np.mean(expected)/repeats)
        np.testing.assert_array_equal(result['X'][0], self.X[0])
        self.assertEqual(result['names'], ['best_initial', 'near_radial', 'far_aligned'])
        self.assertEqual(evaluate.call_count, 3*repeats)
        seeds = [call.args[0] for call in evaluate.call_args_list]
        self.assertEqual(seeds, list(range(config.SEED+2_000_000, config.SEED+2_000_000+18)))
        self.assertTrue(all(call.kwargs == {'common_energy': True} for call in evaluate.call_args_list))
        self.assertEqual(result['settings']['particles_per_species'], config.N)
        self.assertEqual(result['settings']['energy_seed'], config.BO_ENERGY_SEED)
        json.dumps(result, allow_nan=False)

    def test_generated_designs_preserve_full_sphere_geometry_inside_bounds(self):
        near, far = calibration._radial_designs()
        self.assertTrue(np.all((near >= 0)&(near <= 1)))
        self.assertTrue(np.all((far >= 0)&(far <= 1)))
        index = np.arange(config.K)
        z = 1-2*(index+.5)/config.K
        angle = np.pi*(3-np.sqrt(5))*index
        directions = np.column_stack((np.sqrt(1-z*z)*np.cos(angle),
                                      np.sqrt(1-z*z)*np.sin(angle), z))
        _, _, centers_near, normals_near = calibration._geometry(near)
        _, _, centers_far, normals_far = calibration._geometry(far)
        np.testing.assert_allclose(centers_near, config.X_BOUNDS[0]*directions, atol=1e-14)
        np.testing.assert_allclose(normals_near, directions, atol=1e-14)
        np.testing.assert_allclose(centers_far, config.X_BOUNDS[1]*directions, atol=1e-14)
        np.testing.assert_allclose(normals_far, np.tile([0., 0., 1.], (config.K, 1)), atol=1e-14)

    def test_duplicate_geometries_are_excluded_and_only_numerical_floor_remains(self):
        self.X[0] = calibration._radial_designs()[0]
        with patch.object(calibration, 'evaluate_configuration', return_value=(10., 1e30)) as evaluate:
            result = calibration.calibrate_noise(self.X, self.y, repeats=2)
        self.assertEqual(result['names'], ['best_initial', 'far_aligned'])
        self.assertEqual(evaluate.call_count, 4)
        self.assertEqual(result['variance'], 1e-12)
        self.assertEqual(result['mean_variance'], [0., 0.])

    def test_invalid_inputs_fail_before_simulation(self):
        with patch.object(calibration, 'evaluate_configuration') as evaluate:
            for repeats in [1, 1.5]:
                with self.assertRaises(ValueError):
                    calibration.calibrate_noise(self.X, self.y, repeats)
            with self.assertRaises(ValueError):
                calibration.calibrate_noise(self.X+2, self.y)
            with self.assertRaises(ValueError):
                calibration.calibrate_noise(self.X, [np.nan, 1])
            evaluate.assert_not_called()

    def test_nonpositive_response_is_rejected(self):
        with patch.object(calibration, 'evaluate_configuration', return_value=(0., 0.)):
            with self.assertRaises(FloatingPointError):
                calibration.calibrate_noise(self.X, self.y)


if __name__ == '__main__':
    unittest.main()
