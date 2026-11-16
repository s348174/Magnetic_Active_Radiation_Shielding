"""Pilot reuses physical settings, preserves fits and independently compares designs."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError

import input as config
import Objective
from FinalEvaluation import save_configuration
import run_surrogate_pilot as pilot


class SurrogatePilotTests(unittest.TestCase):
    def setUp(self):
        previous_seed = Objective.optimization_seed_state()
        self.addCleanup(Objective.set_optimization_seed,previous_seed)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        self.X = np.repeat(np.linspace(.02,.62,20)[:,None],config.D,axis=1)
        self.y = np.linspace(-34.,-33.,20)
        self.noise = .02
        self.settings = dict(K=config.K,particles_per_species=config.N,current_A=config.I,
                             coil_radius_m=config.R,energy_bins=config.ENERGY_BINS,
                             energy_seed=config.BO_ENERGY_SEED,
                             lower=config.LOWER.cpu().tolist(),upper=config.UPPER.cpu().tolist())
        self.calibration = dict(variance=self.noise,settings=self.settings)
        self.write_calibration()
        np.savez(self.source/'bo_history.npz',X=self.X,y=self.y)
        save_configuration(self.source/'best_configuration.csv',self.X[2])
        self.report_dir = self.root/'pilot'
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.evaluate = self.stack.enter_context(patch.object(
            pilot,'evaluate_finalists',side_effect=self.validation))
        self.loop = self.stack.enter_context(patch.object(
            pilot,'bo_coil_set_kernel',side_effect=self.successful_loop))
        self.seeds = []

    def write_calibration(self):
        (self.source/'noise_calibration.json').write_text(json.dumps(self.calibration))

    def validation(self, finalists, **kwargs):
        # Return ranking order reversed to ensure labels, not row order, identify designs.
        return [dict(finalist=2,rank=1,x=finalists[1]['x'],mean_response=90.,
                     replicate_se=1.,ci95_lower=88.,ci95_upper=92.),
                dict(finalist=1,rank=2,x=finalists[0]['x'],mean_response=100.,
                     replicate_se=1.,ci95_lower=98.,ci95_upper=102.)]

    def successful_loop(self, **kwargs):
        self.seeds.append(Objective.next_optimization_seed())
        X,y,noise = kwargs['initial_data']
        self.assertEqual(len(y),20)
        self.assertEqual(config.N,self.settings['particles_per_species'])
        np.testing.assert_array_equal(X.cpu().numpy(),self.X)
        np.testing.assert_array_equal(noise,np.full(20,self.noise))
        self.assertEqual(kwargs['calibration'],self.calibration)
        initial_state = dict(X=self.X,y=self.y,variance=noise,status=np.asarray('ready'),
                             evaluations=np.asarray(0),
                             **{'gp_state__covar_module.outputscale':np.asarray(.8)})
        kwargs['checkpoint'](initial_state,[])
        next_X = np.vstack((self.X,np.full((1,config.D),.9),np.full((1,config.D),.95)))
        next_y = np.r_[self.y,-33.2,-32.8]
        checks = [dict(evaluations=2,model_ok=True,lengthscale=[.5],kappa=1.)]
        kwargs['model_callback'](checks)
        kwargs['checkpoint'](dict(initial_state,X=next_X,y=next_y,
                                  variance=np.full(22,self.noise),evaluations=np.asarray(2)),[])
        diagnostics = dict(settings={'enabled':False},stop_reason='iteration_budget',
                           model_checks=checks,checks=[])
        return next_X,next_y,np.array([-1.,-2.]),[-33.,-32.9],[.01,.01],diagnostics

    def test_pilot_reuses_all_data_and_validates_original_winner_against_best_new(self):
        Objective.set_optimization_seed(888)
        report = pilot.run_pilot(self.source,self.report_dir,evaluations=2)
        self.assertEqual(Objective.optimization_seed_state(),888)
        self.assertEqual(Objective.next_optimization_seed(),888)
        self.assertEqual(self.seeds,[config.SEED+500_000])
        self.assertEqual(report['completed_evaluations'],2)
        self.assertEqual(report['baseline_source_index'],2)
        self.assertEqual(report['source_best_observed_index'],19)
        self.assertEqual(report['best_new_index'],21)
        self.assertEqual(report['status'],'complete')
        self.assertAlmostEqual(report['validation_relative_new_minus_baseline'],-.1)
        finalists = self.evaluate.call_args.args[0]
        np.testing.assert_array_equal(finalists[0]['x'],self.X[2])
        np.testing.assert_array_equal(finalists[1]['x'],np.full(config.D,.95))
        self.assertEqual(self.evaluate.call_args.kwargs['n_particles'],config.FINAL_PARTICLES)
        self.assertEqual(self.evaluate.call_args.kwargs['repeats'],3)
        self.assertEqual(self.evaluate.call_args.kwargs['seed'],config.SEED+3_000_000)
        self.assertEqual(self.evaluate.call_args.kwargs['output_dir'],self.report_dir/'validation')
        saved = json.loads((self.report_dir/'pilot_report.json').read_text())
        self.assertEqual(saved['final_validation'][0]['finalist'],2)
        self.assertTrue((self.report_dir/'model_diagnostics.json').exists())
        self.assertTrue((self.report_dir/'regret_history.csv').exists())
        with np.load(self.report_dir/'initial_model_state.npz',allow_pickle=False) as initial:
            self.assertEqual(len(initial['y']),20)
            self.assertEqual(float(initial['gp_state__covar_module.outputscale']),.8)
        with np.load(self.report_dir/'bo_checkpoint.npz',allow_pickle=False) as final:
            self.assertEqual(len(final['y']),22)
        self.assertFalse((self.report_dir/'bo_checkpoint.npz.tmp').exists())

    def test_fitting_failure_preserves_checkpoint_and_skips_validation(self):
        def fail(**kwargs):
            kwargs['checkpoint'](dict(X=self.X,y=self.y,variance=np.full(20,self.noise),
                                      status=np.asarray('model_fitting_failed'),evaluations=np.asarray(0)),[])
            raise ModelFittingError('no predictive scale candidate')
        self.loop.side_effect = fail
        Objective.set_optimization_seed(888)
        with self.assertRaisesRegex(RuntimeError,'bo_checkpoint.npz'):
            pilot.run_pilot(self.source,self.report_dir,evaluations=2)
        self.assertEqual(Objective.optimization_seed_state(),888)
        self.evaluate.assert_not_called()
        self.assertTrue((self.report_dir/'bo_checkpoint.npz').exists())
        saved = json.loads((self.report_dir/'pilot_report.json').read_text())
        self.assertEqual(saved['status'],'pilot_failed')
        self.assertIn('no predictive scale candidate',saved['error'])

    def test_setting_mismatch_stops_before_running_or_creating_output(self):
        self.settings['particles_per_species'] += 1
        self.write_calibration()
        with self.assertRaisesRegex(ValueError,'particles_per_species'):
            pilot.run_pilot(self.source,self.report_dir,evaluations=2)
        self.loop.assert_not_called()
        self.evaluate.assert_not_called()
        self.assertFalse(self.report_dir.exists())

    def test_overlapping_seed_ranges_and_invalid_budgets_are_rejected(self):
        bad = [dict(evaluations=0),dict(repeats=1),dict(final_particles=1),
               dict(spatial_seed=config.SEED),dict(spatial_seed=config.SEED+1_000_000),
               dict(validation_seed=config.SEED+1_000_000)]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                pilot.run_pilot(self.source,self.report_dir,**kwargs)
        self.loop.assert_not_called()
        self.evaluate.assert_not_called()
        self.assertFalse(self.report_dir.exists())

    def test_pilot_uses_saved_next_seed_and_preserves_the_advanced_state(self):
        saved_next = config.SEED+500_033 # 500100 for the standard seed 67.
        np.savez(self.source/'bo_checkpoint.npz',next_spatial_seed=saved_next)
        report = pilot.run_pilot(self.source,self.report_dir,evaluations=2)
        self.assertEqual(self.seeds,[saved_next])
        self.assertEqual(report['pilot_spatial_seed_start'],saved_next)
        self.assertEqual(report['next_spatial_seed'],saved_next+1)
        with np.load(self.report_dir/'bo_history.npz',allow_pickle=False) as saved:
            self.assertEqual(int(saved['next_spatial_seed']),saved_next+1)

    def test_explicit_seed_below_saved_next_seed_is_rejected_before_work(self):
        np.savez(self.source/'bo_checkpoint.npz',next_spatial_seed=config.SEED+500_033)
        with self.assertRaisesRegex(ValueError,'spatial seeds'):
            pilot.run_pilot(self.source,self.report_dir,evaluations=2,
                            spatial_seed=config.SEED+500_000)
        self.loop.assert_not_called()
        self.evaluate.assert_not_called()
        self.assertFalse(self.report_dir.exists())

    def test_reading_optimization_seed_state_does_not_advance_the_counter(self):
        Objective.set_optimization_seed(7000)
        self.assertEqual(Objective.optimization_seed_state(),7000)
        self.assertEqual(Objective.optimization_seed_state(),7000)
        self.assertEqual(Objective.next_optimization_seed(),7000)
        self.assertEqual(Objective.optimization_seed_state(),7001)
        self.assertEqual(Objective.next_optimization_seed(),7001)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
