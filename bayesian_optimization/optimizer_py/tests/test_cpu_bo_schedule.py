"""Deferred fitting keeps every observation in the coil-set posterior."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError

import BoLoop
import BoUtils
from FastCoilAcquisition import CheckedLogExpectedImprovement
from GPConditioning import condition_on_observations
from RegretStopping import SimpleRegretStopping
from input import D


class CPUCoilScheduleTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(67)
        designs = .1+.8*torch.rand(11,D,dtype=torch.double,generator=generator)
        self.X,self.proposals = designs[:6],designs[6:]
        self.y = np.array([-34.,-34.03,-34.01,-34.04,-34.02,-34.015])
        self.responses = 34.025+.003*np.arange(5)
        self.noise = .0003
        self.initial = (self.X,self.y,np.full(len(self.y),self.noise))
        self.fit_counts,self.observed_counts,self.records = [],[],[]

    def run_schedule(self,fail_conditioning=False):
        proposal_iter = iter(self.proposals)
        stopping = SimpleRegretStopping(enabled=False,record_every=0,grid_size=8)

        def fit(mll):
            self.fit_counts.append(len(mll.model.train_targets)-len(self.y))
            return mll.eval()

        def acquire(**kwargs):
            self.assertIsInstance(kwargs['acq_function'].base_acqf,CheckedLogExpectedImprovement)
            self.assertEqual(kwargs['num_restarts'],12)
            self.assertEqual(kwargs['raw_samples'],256)
            self.assertEqual(kwargs['options']['maxiter'],60)
            torch.testing.assert_close(kwargs['acq_function'].base_acqf.best_f,kwargs['ranking'].max())
            return next(proposal_iter).unsqueeze(0),torch.tensor(-7.,dtype=torch.double)

        def check_model(gp,mapping,calibration):
            count = len(gp.train_targets)-len(self.y)
            self.observed_counts.append(count)
            expected_X = torch.cat((self.X,self.proposals[:count]))
            expected_y = torch.as_tensor(np.r_[self.y,-self.responses[:count]],dtype=torch.double)
            torch.testing.assert_close(gp.train_inputs[0],BoUtils.map_coilwise(expected_X))
            transform = gp.outcome_transform
            expected_targets = (expected_y-transform.means.squeeze())/transform.stdvs.squeeze()
            torch.testing.assert_close(gp.train_targets,expected_targets)
            torch.testing.assert_close(gp.likelihood.noise,
                torch.full_like(expected_targets,self.noise)/transform.stdvs.squeeze().square())
            return {'model_ok':True,'reasons':[],'outputscale':1.}

        conditioning_calls = []
        def condition(gp,X,y,variance):
            conditioning_calls.append(len(y)-len(self.y))
            if fail_conditioning and len(conditioning_calls) == 1:
                raise ModelFittingError('forced conditioning failure')
            return condition_on_observations(gp,X,y,variance)

        with ExitStack() as stack:
            patches = {
                'BO_COIL_REFIT_EVERY':3,'BO_COIL_SELECT_EVERY':4,
                'BO_COIL_ACQUISITION':'log-ei','BO_COIL_RESTARTS':12,
                'BO_COIL_RAW_SAMPLES':256,'BO_COIL_OPT_MAXITER':60,
            }
            for name,value in patches.items():
                stack.enter_context(patch.object(BoLoop,name,value))
            stack.enter_context(patch.object(BoLoop,'fit_gpytorch_mll',side_effect=fit))
            stack.enter_context(patch.object(BoLoop,'optimize_acquisition',side_effect=acquire))
            stack.enter_context(patch.object(BoLoop,'condition_on_observations',side_effect=condition))
            stack.enter_context(patch.object(BoLoop,'assess_model',side_effect=check_model))
            stack.enter_context(patch.object(BoLoop,'SimpleRegretStopping',return_value=stopping))
            stack.enter_context(patch.object(BoLoop,'next_optimization_seed',return_value=67))
            stack.enter_context(patch.object(BoLoop,'objective_function',
                side_effect=[(float(value),self.noise) for value in self.responses]))
            select = stack.enter_context(patch.object(BoLoop,'select_coil_scales',
                return_value=(.5,1.,{'predictive_ok':True,'selection_skill':.1})))
            global_mean = stack.enter_context(patch.object(BoLoop,'_best_posterior_mean'))
            result = BoLoop.bo_coil_set_kernel(max_iter=5,initial_data=self.initial,
                calibration={'variance':self.noise},return_diagnostics=True,
                checkpoint=lambda state,checks:self.records.append(
                    {key:np.asarray(value).copy() for key,value in state.items()}))
            self.assertEqual(select.call_count,2) # Initial scales, then evaluation four.
            global_mean.assert_not_called()
        self.assertEqual(self.observed_counts,list(range(6)))
        self.assertEqual(len(result[1]),len(self.y)+5)
        self.assertEqual(len(result[-1]['iteration_timings']),5)
        return conditioning_calls

    def assert_checkpoint_counts(self,expected_hyper):
        ready = [row for row in self.records
                 if str(row['status']) == 'ready' and int(row['evaluations']) > 0]
        self.assertEqual(len(ready),5)
        self.assertEqual([int(row['fitted_evaluations']) for row in ready],list(range(1,6)))
        self.assertEqual([int(row['hyperparameter_evaluations']) for row in ready],expected_hyper)
        for count,row in enumerate(ready,1):
            self.assertEqual(len(row['y']),len(self.y)+count)
            np.testing.assert_allclose(row['variance'],self.noise)
            self.assertTrue(np.isfinite(row['posterior_mean']).all())
            self.assertTrue(np.isfinite(row['posterior_variance']).all())

    def test_scheduled_refits_and_exact_conditioning_include_every_observation(self):
        calls = self.run_schedule()
        self.assertEqual(self.fit_counts,[0,3,4])
        self.assertEqual(calls,[1,2,5])
        self.assert_checkpoint_counts([0,0,3,4,4])

    def test_conditioning_failure_falls_back_to_a_complete_refit(self):
        calls = self.run_schedule(fail_conditioning=True)
        self.assertEqual(self.fit_counts,[0,1,4])
        self.assertEqual(calls,[1,2,3,5])
        self.assert_checkpoint_counts([1,1,1,4,4])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
