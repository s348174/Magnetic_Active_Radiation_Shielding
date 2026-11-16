"""Completed simulations survive initial and subsequent GP fitting failures."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError

import BoLoop
import Objective
import bo_main
from input import D


class FitCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.X = torch.stack([torch.full((D,), value, dtype=torch.double)
                              for value in (.1, .2, .3, .4)])
        self.y = np.array([-1., -2., -3., -4.])
        self.initial = (self.X, self.y, np.full(4, .5))
        self.calibration = dict(variance=.01, X=self.X.tolist(), mean=self.y.tolist(),
                                mean_variance=[.001]*4, names=['a', 'b', 'c', 'd'],
                                repeats=6, runs=[], settings={})
        self.records = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module in (BoLoop, bo_main):
            self.stack.enter_context(patch.object(module, 'calibrate_noise',
                                                 return_value=self.calibration))
        self.model_check = self.stack.enter_context(patch.object(
            BoLoop, 'assess_model', return_value={'model_ok': True, 'reasons': [],
                                                'outputscale': 1., 'predicted_mean_range': .1}))
        self.stopping = self.stack.enter_context(patch.object(
            BoLoop.SimpleRegretStopping, 'check', return_value=False))
        self.stack.enter_context(patch.object(BoLoop, '_best_posterior_mean', return_value=-.5))
        self.optimize = self.stack.enter_context(patch.object(
            BoLoop, 'optimize_acqf', return_value=(torch.full((1,D), .7, dtype=torch.double),
                                                torch.tensor(-7.))))
        def direct_optimizer(**kwargs):
            for name in ('X','ranking','optimizer'):
                kwargs.pop(name,None)
            return BoLoop.optimize_acqf(**kwargs)
        self.stack.enter_context(patch.object(BoLoop,'optimize_acquisition',side_effect=direct_optimizer))
        self.objective = self.stack.enter_context(patch.object(
            BoLoop, 'objective_function', return_value=(.5, self.calibration['variance'])))

    def save(self, history, checks):
        # Capture an independent snapshot, rather than references mutated later.
        self.records.append(({key: np.asarray(value).copy() for key, value in history.items()},
                             [dict(row) for row in checks]))

    def run_loop(self):
        return BoLoop.bo_rbf_old(max_iter=1, initial_data=self.initial,
                                 calibration=self.calibration, checkpoint=self.save)

    def assert_counts(self, history, evaluations, fitted):
        self.assertEqual(int(history['evaluations']), evaluations)
        self.assertEqual(int(history['fitted_evaluations']), fitted)
        self.assertEqual(int(history['initial_evaluations']), 4)
        self.assertEqual(len(history['y']), 4+evaluations)
        self.assertEqual(history['X'].shape, (4+evaluations,D))
        np.testing.assert_array_equal(history['variance'], np.full(4+evaluations,.01))
        for name in ('acquisition', 'posterior_mean', 'posterior_variance'):
            self.assertEqual(len(history[name]), evaluations)

    def test_initial_failure_saves_initial_data_before_attempting_fit(self):
        def fail(mll):
            self.assertEqual(str(self.records[-1][0]['status']), 'fit_pending')
            self.assert_counts(self.records[-1][0], 0, 0)
            raise ModelFittingError('forced initial fit failure')
        with patch.object(BoLoop, 'fit_gpytorch_mll', side_effect=fail), \
             self.assertRaises(ModelFittingError):
            self.run_loop()
        history, checks = self.records[-1]
        self.assertEqual(str(history['status']), 'model_fitting_failed')
        self.assert_counts(history, 0, 0)
        self.assertEqual(int(history['iteration']), 0)
        self.assertTrue(str(history['error']))
        np.testing.assert_array_equal(history['y'], self.y)
        self.assertEqual(checks, [])
        self.objective.assert_not_called()
        self.optimize.assert_not_called()
        self.model_check.assert_not_called()

    def test_updated_failure_saves_latest_observation_before_fit(self):
        calls = []
        def fit(mll):
            calls.append(mll)
            if len(calls) == 1:
                return mll.eval()
            history = self.records[-1][0]
            self.assertEqual(str(history['status']), 'fit_pending')
            self.assert_counts(history, 1, 0)
            self.assertEqual(history['y'][-1], -.5)
            self.assertEqual(int(history['next_spatial_seed']),Objective.optimization_seed_state())
            self.assertEqual(history['acquisition'][-1], -7.)
            self.assertTrue(np.isnan(history['posterior_mean'][-1]))
            self.assertTrue(np.isnan(history['posterior_variance'][-1]))
            raise ModelFittingError('forced updated fit failure')
        with patch.object(BoLoop, 'fit_gpytorch_mll', side_effect=fit), \
             self.assertRaises(ModelFittingError):
            self.run_loop()
        history, checks = self.records[-1]
        self.assertEqual(str(history['status']), 'model_fitting_failed')
        self.assert_counts(history, 1, 0)
        self.assertEqual(int(history['iteration']), 1)
        self.assertTrue(np.isnan(history['posterior_mean'][-1]))
        self.assertEqual(self.objective.call_count, 1)
        self.assertEqual(self.model_check.call_count, 1) # Only the successful initial fit.
        self.stopping.assert_not_called()
        self.assertEqual(checks, [])

    def test_success_replaces_pending_metrics_and_marks_fit_complete(self):
        with patch.object(BoLoop, 'fit_gpytorch_mll', side_effect=lambda mll:mll.eval()):
            result = self.run_loop()
        pending = [history for history, _ in self.records
                   if str(history['status']) == 'fit_pending' and int(history['evaluations']) == 1]
        self.assertTrue(pending)
        self.assertTrue(np.isnan(pending[0]['posterior_mean'][-1]))
        history, _ = self.records[-1]
        self.assertEqual(str(history['status']), 'ready')
        self.assert_counts(history, 1, 1)
        self.assertEqual(int(history['iteration']), 1)
        self.assertTrue(np.isfinite(history['posterior_mean']).all())
        self.assertTrue(np.isfinite(history['posterior_variance']).all())
        self.assertEqual(str(history['error']), '')
        np.testing.assert_array_equal(history['y'], result[1])

    def test_driver_failure_keeps_atomic_checkpoint_and_skips_final_validation(self):
        calls = []
        def fit(mll):
            calls.append(mll)
            if len(calls) == 1:
                return mll.eval()
            raise ModelFittingError('forced driver fit failure')
        with tempfile.TemporaryDirectory() as root, \
             patch.object(bo_main, 'sobol_sample', return_value=self.initial), \
             patch.object(bo_main, 'evaluate_finalists') as final_validation, \
             patch.object(BoLoop, 'fit_gpytorch_mll', side_effect=fit):
            directory = Path(root)/'run'
            with self.assertRaises(ModelFittingError) as raised:
                bo_main.main(['--method','bo','--kernel','rbf','--iterations','1',
                              '--output-dir',str(directory),'--no-plot'])
            self.assertIn(str(directory.resolve()), str(raised.exception))
            self.assertIn('bo_checkpoint.npz', str(raised.exception))
            final_validation.assert_not_called()
            with np.load(directory/'bo_checkpoint.npz', allow_pickle=False) as stored:
                history = {key:stored[key] for key in stored.files}
            self.assertEqual(str(history['status']), 'model_fitting_failed')
            self.assert_counts(history, 1, 0)
            self.assertEqual(history['y'][-1], -.5)
            self.assertTrue((directory/'initial_design.npz').is_file())
            self.assertTrue((directory/'regret_history.csv').is_file())
            self.assertFalse((directory/'bo_checkpoint.npz.tmp').exists())
            self.assertFalse((directory/'final_validation.csv').exists())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
