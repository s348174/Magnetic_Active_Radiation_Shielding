"""Resume reuses every saved observation, calibration and spatial-seed position."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import bo_main
import input as config
import Objective


class BOResumeTests(unittest.TestCase):
    def setUp(self):
        self.previous_seed = Objective.optimization_seed_state()
        self.addCleanup(Objective.set_optimization_seed,self.previous_seed)
        Objective.set_optimization_seed(config.SEED)
        generator = torch.Generator().manual_seed(67)
        self.X = torch.rand(150,config.D,dtype=torch.double,generator=generator).numpy()
        self.y = -34.+.01*np.arange(150)
        self.noise = .0003
        settings = dict(K=config.K,particles_per_species=config.N,current_A=config.I,
                        coil_radius_m=config.R,energy_bins=config.ENERGY_BINS,
                        energy_seed=config.BO_ENERGY_SEED,
                        lower=config.LOWER.cpu().tolist(),upper=config.UPPER.cpu().tolist())
        self.calibration = dict(variance=self.noise,settings=settings,repeats=6,
                                X=self.X[:3].tolist(),mean=self.y[:3].tolist(),
                                mean_variance=[self.noise/6]*3,names=['a','b','c'])
        self.checkpoint = dict(X=self.X,y=self.y,kernel='coil-set',next_spatial_seed=217,
                               initial_evaluations=100,evaluations=50,fitted_evaluations=49,
                               status='model_fitting_failed')

    def save_source(self,directory):
        directory.mkdir()
        (directory/'noise_calibration.json').write_text(json.dumps(self.calibration))
        np.savez(directory/'bo_checkpoint.npz',**self.checkpoint)

    def result(self,**kwargs):
        initial = kwargs['initial_data']
        np.testing.assert_array_equal(initial[0].cpu().numpy(),self.X)
        np.testing.assert_array_equal(initial[1],self.y)
        np.testing.assert_array_equal(initial[2],np.full(150,self.noise))
        self.assertEqual(kwargs['calibration'],self.calibration)
        self.assertTrue(kwargs['return_diagnostics'])
        # The unfinished final observation is included even though its old fit failed.
        self.assertEqual(len(initial[1]),150)
        self.assertEqual(initial[1][-1],self.y[-1])
        diagnostics = dict(settings={},checks=[],incumbent_index=0,
                           stop_reason='iteration_budget',acquisition='log-ei')
        return self.X,self.y,np.array([]),[],[],diagnostics

    def test_resume_reuses_data_and_noise_without_running_initial_simulations(self):
        with tempfile.TemporaryDirectory() as folder:
            source,output = Path(folder)/'source',Path(folder)/'output'
            self.save_source(source)
            output.mkdir()
            with patch.object(bo_main,'sobol_sample') as sample, \
                 patch.object(bo_main,'calibrate_noise') as calibrate, \
                 patch.object(bo_main,'bo_coil_set_kernel',side_effect=self.result) as loop:
                result = bo_main.run_vanilla(kernel='coil-set',iterations=450,
                                             checkpoint_dir=output,resume_from=source)
            sample.assert_not_called()
            calibrate.assert_not_called()
            self.assertEqual(loop.call_args.kwargs['max_iter'],450)
            self.assertEqual(Objective.optimization_seed_state(),217)
            self.assertEqual(result['selection']['resume_samples'],150)
            self.assertEqual(result['selection']['resume_from'],str(source.resolve()))
            with np.load(output/'initial_design.npz',allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['y'],self.y)
            with np.load(source/'bo_checkpoint.npz',allow_pickle=False) as saved:
                self.assertEqual(saved['fitted_evaluations'].item(),49)

    def test_resume_never_rewinds_a_seed_already_advanced_in_this_process(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source'
            self.save_source(source)
            Objective.set_optimization_seed(230)
            initial,calibration,metadata = bo_main._resume_coil_data(source,None)
            self.assertEqual(Objective.optimization_seed_state(),230)
            self.assertEqual(metadata['resume_spatial_seed'],230)

    def test_invalid_checkpoint_or_physics_does_not_change_seed_state(self):
        for kind in ('physics','seed_float','seed_reused','wrong_kernel','missing_seed'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                source = Path(folder)/'source'
                self.save_source(source)
                checkpoint = dict(self.checkpoint)
                if kind == 'physics':
                    calibration = json.loads((source/'noise_calibration.json').read_text())
                    calibration['settings']['particles_per_species'] += 1
                    (source/'noise_calibration.json').write_text(json.dumps(calibration))
                elif kind == 'seed_float':
                    checkpoint['next_spatial_seed'] = 217.
                elif kind == 'seed_reused':
                    checkpoint['next_spatial_seed'] = 216
                elif kind == 'wrong_kernel':
                    checkpoint['kernel'] = 'rbf'
                else:
                    checkpoint.pop('next_spatial_seed')
                np.savez(source/'bo_checkpoint.npz',**checkpoint)
                with self.assertRaises(ValueError):
                    bo_main._resume_coil_data(source,None)
                self.assertEqual(Objective.optimization_seed_state(),config.SEED)

    def test_source_cannot_be_overwritten_and_finished_history_is_not_silently_used(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source'
            self.save_source(source)
            with self.assertRaisesRegex(ValueError,'new directory'):
                bo_main._resume_coil_data(source,source)
            np.savez(source/'bo_history.npz',X=self.X[:-1],y=self.y[:-1])
            with self.assertRaisesRegex(ValueError,'unfinished run'):
                bo_main._resume_coil_data(source,None)

    def test_cli_routes_resume_to_coil_loop_and_creates_separate_output(self):
        validation = [dict(mean_response=1.,ci95_lower=.9,ci95_upper=1.1)]
        with tempfile.TemporaryDirectory() as folder:
            source,output = Path(folder)/'source',Path(folder)/'output'
            self.save_source(source)
            with patch.object(bo_main,'bo_coil_set_kernel',side_effect=self.result), \
                 patch.object(bo_main,'evaluate_finalists',return_value=validation):
                bo_main.main(['--method','bo','--kernel','coil-set','--resume-from',str(source),
                              '--iterations','450','--output-dir',str(output),'--no-plot'])
            settings = json.loads((output/'bo_settings.json').read_text())
            self.assertEqual(settings['resume_samples'],150)
            self.assertEqual(settings['initial_evaluations'],150)
            self.assertEqual(settings['total_refinement_budget'],450)
            self.assertTrue((source/'bo_checkpoint.npz').is_file())
        for arguments in (['--method','turbo'],['--method','bo','--kernel','rbf']):
            with self.subTest(arguments=arguments),self.assertRaises(SystemExit):
                bo_main.main(arguments+['--resume-from','unused'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
