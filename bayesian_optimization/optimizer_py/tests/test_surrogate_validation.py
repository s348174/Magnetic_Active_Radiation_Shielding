"""Offline validation excludes outer test labels from geometry-scale selection."""
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import input as config
import validate_surrogate as validation


class SurrogateValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'run'
        self.run.mkdir()
        self.X = np.repeat(np.linspace(.05,.95,20)[:,None],config.D,axis=1)
        self.y = np.linspace(-34.5,-33.9,20)
        self.noise = .04
        self.settings = dict(K=config.K,particles_per_species=config.N,current_A=config.I,
                             coil_radius_m=config.R,energy_bins=config.ENERGY_BINS,
                             energy_seed=config.BO_ENERGY_SEED,
                             lower=config.LOWER.cpu().tolist(),upper=config.UPPER.cpu().tolist())
        self.save_calibration()
        np.savez(self.run/'bo_history.npz',X=self.X,y=self.y)
        order = np.random.default_rng(67).permutation(len(self.X))
        self.test,self.train = order[:4],order[4:]
        self.gp = SimpleNamespace(covar_module=SimpleNamespace(outputscale=torch.tensor(1.2)))

    def save_calibration(self):
        with open(self.run/'noise_calibration.json','w') as file:
            json.dump(dict(variance=self.noise,settings=self.settings),file)

    def mocks(self, prediction):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.select = stack.enter_context(patch.object(validation,'select_coil_scales',return_value=(
            .5,1.,dict(train_indices=list(range(12)),validation_indices=list(range(12,16)),
                       candidates=[dict(lengthscale=.5,kappa=1.,rmse=.1,valid=True)]))))
        self.fit = stack.enter_context(patch.object(validation,'_fit_model',return_value=self.gp))
        self.kernel = stack.enter_context(patch.object(validation,'check_kernel',return_value={
            'points':16,'minimum_eigenvalue':.001,'maximum_diagonal_error':0.}))
        self.raw = stack.enter_context(patch.object(validation,'raw_mean_variance',return_value=(
            torch.as_tensor(prediction,dtype=torch.double).reshape(-1,1),
            torch.full((4,1),1e-10,dtype=torch.double))))

    def test_outer_split_is_held_out_from_all_selection_and_fitting(self):
        self.mocks(self.y[self.test]+.1)
        report = validation.validate_surrogate(self.run,self.root/'report')
        self.assertEqual(report['train_indices'],self.train.tolist())
        self.assertEqual(report['test_indices'],self.test.tolist())
        self.assertTrue(report['outer_test_excluded_from_selection'])
        selected_y = np.asarray(self.select.call_args.args[1])
        np.testing.assert_array_equal(selected_y,self.y[self.train])
        self.assertEqual(self.select.call_args.args[0].shape,(16,6*config.K))
        self.assertTrue(set(selected_y).isdisjoint(self.y[self.test]))
        fitted_X,fitted_y,fitted_noise,mapping,_ = self.fit.call_args.args
        np.testing.assert_array_equal(fitted_X.cpu().numpy(),self.X[self.train])
        np.testing.assert_array_equal(fitted_y,self.y[self.train])
        np.testing.assert_array_equal(fitted_noise.cpu().numpy(),np.full((16,1),self.noise))
        self.assertIs(mapping,validation.map_coilwise)
        selection = report['selection']
        self.assertEqual(selection['global_train_indices'],self.train[:12].tolist())
        self.assertEqual(selection['global_validation_indices'],self.train[12:].tolist())
        self.assertTrue(set(selection['global_validation_indices']).isdisjoint(self.test))
        self.assertEqual(report['selected_lengthscale'],.5)
        self.assertEqual(report['selected_kappa'],1.)

    def test_baseline_metrics_and_coverage_include_single_observation_noise(self):
        self.mocks(self.y[self.test]+.1)
        directory = self.root/'report'
        report = validation.validate_surrogate(self.run,directory)
        baseline = np.mean(self.y[self.train])
        baseline_mse = np.mean((baseline-self.y[self.test])**2)
        self.assertAlmostEqual(report['log_rmse'],.1)
        self.assertAlmostEqual(report['constant_log_rmse'],np.sqrt(baseline_mse))
        self.assertAlmostEqual(report['skill'],1-.01/baseline_mse)
        self.assertAlmostEqual(report['correlation'],1.)
        self.assertEqual(report['coverage95'],1.)
        self.assertEqual(report['minimum_raw_variance'],1e-10)
        self.assertTrue(report['coverage_includes_observation_noise'])
        with open(directory/'held_out_predictions.csv') as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows),4)
        for row in rows:
            self.assertAlmostEqual(float(row['predictive_variance']),self.noise+1e-10)
            self.assertGreater(float(row['upper95'])-float(row['lower95']),.2)
        saved = json.loads((directory/'validation_report.json').read_text())
        self.assertEqual(saved['test_indices'],report['test_indices'])
        self.assertEqual(saved['kernel_check']['minimum_eigenvalue'],.001)

    def test_constant_prediction_has_zero_skill_and_no_undefined_json_correlation(self):
        self.mocks(np.full(4,self.y[self.train].mean()))
        report = validation.validate_surrogate(self.run,self.root/'report')
        self.assertAlmostEqual(report['skill'],0.)
        self.assertIsNone(report['correlation'])
        json.dumps(report,allow_nan=False)

    def test_metadata_mismatch_is_rejected_before_any_fit(self):
        for name in ('K','particles_per_species','current_A','coil_radius_m',
                     'energy_bins','energy_seed','lower','upper'):
            previous = self.settings[name]
            self.settings[name] = [0.] if isinstance(previous,list) else previous+1
            self.save_calibration()
            with self.subTest(name=name),patch.object(validation,'select_coil_scales') as select, \
                 patch.object(validation,'_fit_model') as fit,self.assertRaisesRegex(ValueError,name):
                validation.validate_surrogate(self.run,self.root/'rejected')
            select.assert_not_called()
            fit.assert_not_called()
            self.assertFalse((self.root/'rejected').exists())
            self.settings[name] = previous
        self.save_calibration()

    def test_outer_split_is_reproducible_without_consuming_numpy_global_rng(self):
        self.mocks(self.y[self.test])
        state = np.random.get_state()
        try:
            expected = np.random.random(3)
            np.random.set_state(state)
            first = validation.validate_surrogate(self.run,self.root/'first')
            np.testing.assert_array_equal(np.random.random(3),expected)
            second = validation.validate_surrogate(self.run,self.root/'second')
        finally:
            np.random.set_state(state)
        self.assertEqual(first['test_indices'],second['test_indices'])
        self.assertEqual(first['train_indices'],second['train_indices'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
