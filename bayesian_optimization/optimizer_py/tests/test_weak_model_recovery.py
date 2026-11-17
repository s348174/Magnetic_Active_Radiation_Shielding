"""Weak finite predictive scores trigger recovery fits without losing observations."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

import BoLoop
import BoUtils
from GPConditioning import condition_on_observations
from RegretStopping import SimpleRegretStopping
from input import D


class WeakModelRecoveryTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(67)
        designs = .1+.8*torch.rand(13,D,dtype=torch.double,generator=generator)
        self.X,self.proposals = designs[:6],designs[6:]
        self.y = np.array([-34.,-34.03,-34.01,-34.04,-34.02,-34.015])
        self.responses = 34.025+.003*np.arange(len(self.proposals))
        self.noise = .0003

    def run_recovery(self,statuses,steps=5,normal_fit=10,normal_select=50,
                     recovery_fit=2,recovery_select=10,stopping_enabled=False):
        fits,conditions,selections,observed,checkpoints = [],[],[],[],[]
        stopping = SimpleRegretStopping(enabled=stopping_enabled,min_evaluations=1,
            check_every=1,record_every=1 if stopping_enabled else 0,patience=1,grid_size=8)
        proposals = iter(self.proposals[:steps])

        def fit(mll):
            fits.append(len(mll.model.train_targets)-len(self.y))
            return mll.eval()

        def select(X,y,*args,**kwargs):
            status = statuses[min(len(selections),len(statuses)-1)]
            selections.append(len(y)-len(self.y))
            # The new best finite geometry must be used even when its skill is negative.
            return .25,4.,{'predictive_ok':status,'selection_skill':.1 if status else -.1,
                           'best_rmse':.09 if status else .11,'baseline_rmse':.1}

        def condition(gp,X,y,variance):
            conditions.append(len(y)-len(self.y))
            return condition_on_observations(gp,X,y,variance)

        def assess(gp,mapping,calibration):
            count = len(gp.train_targets)-len(self.y)
            observed.append(count)
            expected_X = torch.cat((self.X,self.proposals[:count]))
            expected_y = torch.as_tensor(np.r_[self.y,-self.responses[:count]],dtype=torch.double)
            torch.testing.assert_close(gp.train_inputs[0],BoUtils.map_coilwise(expected_X))
            transform = gp.outcome_transform
            torch.testing.assert_close(gp.train_targets,
                (expected_y-transform.means.squeeze())/transform.stdvs.squeeze())
            torch.testing.assert_close(gp.likelihood.noise,
                torch.full_like(expected_y,self.noise)/transform.stdvs.squeeze().square())
            self.assertAlmostEqual(float(gp.covar_module.base_kernel.lengthscale),.25)
            self.assertAlmostEqual(float(gp.covar_module.base_kernel.kappa),4.)
            return {'model_ok':True,'reasons':[]}

        with ExitStack() as stack, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for name,value in {'BO_COIL_REFIT_EVERY':normal_fit,
                    'BO_COIL_SELECT_EVERY':normal_select,
                    'BO_COIL_RECOVERY_REFIT_EVERY':recovery_fit,
                    'BO_COIL_RECOVERY_SELECT_EVERY':recovery_select}.items():
                stack.enter_context(patch.object(BoLoop,name,value))
            stack.enter_context(patch.object(BoLoop,'fit_gpytorch_mll',side_effect=fit))
            stack.enter_context(patch.object(BoLoop,'select_coil_scales',side_effect=select))
            stack.enter_context(patch.object(BoLoop,'condition_on_observations',side_effect=condition))
            stack.enter_context(patch.object(BoLoop,'assess_model',side_effect=assess))
            stack.enter_context(patch.object(BoLoop,'optimize_acquisition',side_effect=lambda **kwargs:
                (next(proposals).unsqueeze(0),torch.tensor(-7.,dtype=torch.double))))
            stack.enter_context(patch.object(BoLoop,'next_optimization_seed',return_value=67))
            simulate = stack.enter_context(patch.object(BoLoop,'objective_function',
                side_effect=[(float(value),self.noise) for value in self.responses[:steps]]))
            stack.enter_context(patch.object(BoLoop,'SimpleRegretStopping',return_value=stopping))
            stack.enter_context(patch.object(stopping,'estimate',return_value={
                'max_ucb':0.,'incumbent_lcb':0.,'incumbent_index':0,'estimated_regret':0.}))
            result = BoLoop.bo_coil_set_kernel(max_iter=steps,
                initial_data=(self.X,self.y,np.full(len(self.y),self.noise)),
                calibration={'variance':self.noise},return_diagnostics=True,
                checkpoint=lambda state,checks:checkpoints.append(
                    {key:np.asarray(value).copy() for key,value in state.items()}))
            self.assertEqual(simulate.call_count,steps)
        self.assertEqual(observed,list(range(steps+1)))
        self.assertEqual(len(result[1]),len(self.y)+steps)
        self.assertEqual(result[-1]['stop_reason'],'iteration_budget')
        ready = [state for state in checkpoints if str(state['status']) == 'ready'
                 and int(state['evaluations']) > 0]
        self.assertEqual([int(state['fitted_evaluations']) for state in ready],list(range(1,steps+1)))
        for count,state in enumerate(ready,1):
            self.assertEqual(len(state['y']),len(self.y)+count)
            np.testing.assert_allclose(state['variance'],self.noise)
            self.assertTrue(np.isfinite(state['posterior_mean']).all())
            self.assertTrue(np.isfinite(state['posterior_variance']).all())
        self.assertTrue(any('Continuing with the best finite candidate' in str(item.message)
                            for item in caught))
        return fits,conditions,selections,result[-1],stopping,ready

    def test_weak_initial_selection_continues_and_blocks_false_convergence(self):
        fits,conditions,selections,diagnostics,stopping,ready = self.run_recovery(
            [False],stopping_enabled=True)
        self.assertEqual(fits,[0,2,4])
        self.assertEqual(conditions,[1,3,5])
        self.assertEqual(selections,[0])
        self.assertEqual([int(row['hyperparameter_evaluations']) for row in ready],[0,2,2,4,4])
        self.assertTrue(all(not row['model_ok'] for row in diagnostics['model_checks']))
        self.assertTrue(all(row['recovery_mode'] for row in diagnostics['model_checks']))
        self.assertTrue(all(row['refit_every'] == 2 and row['select_every'] == 10
                            for row in diagnostics['model_checks']))
        self.assertTrue(all('did not beat' in row['reasons'][-1]
                            for row in diagnostics['model_checks']))
        self.assertEqual(stopping.consecutive,0)
        self.assertEqual(len(stopping.history),5)
        self.assertTrue(all(not row['model_ok'] and row['consecutive_passes'] == 0
                            for row in stopping.history))

    def test_positive_reselection_restores_the_normal_schedule(self):
        fits,conditions,selections,diagnostics,_,ready = self.run_recovery(
            [False,True],recovery_select=4)
        self.assertEqual(fits,[0,2,4])
        self.assertEqual(conditions,[1,3,5])
        self.assertEqual(selections,[0,4])
        reports = diagnostics['model_checks']
        self.assertEqual([row['recovery_mode'] for row in reports],[True]*4+[False]*2)
        self.assertEqual([row['model_ok'] for row in reports],[False]*4+[True]*2)
        self.assertTrue(all(row['refit_every'] == 10 and row['select_every'] == 50
                            for row in reports[4:]))
        self.assertEqual([int(row['hyperparameter_evaluations']) for row in ready],[0,2,2,4,4])

    def test_failed_later_reselection_preserves_pending_data_and_recovers(self):
        fits,conditions,selections,diagnostics,_,ready = self.run_recovery(
            [True,False],steps=7,normal_select=4)
        self.assertEqual(fits,[0,4,6])
        self.assertEqual(conditions,[1,2,3,5,7])
        self.assertEqual(selections,[0,4])
        self.assertEqual([row['recovery_mode'] for row in diagnostics['model_checks']],
                         [False]*4+[True]*4)
        self.assertEqual([int(row['hyperparameter_evaluations']) for row in ready],
                         [0,0,0,4,4,6,6])

    def test_recovery_never_relaxes_an_already_tighter_user_schedule(self):
        fits,conditions,selections,diagnostics,_,_ = self.run_recovery(
            [False],normal_fit=1,normal_select=2)
        self.assertEqual(fits,list(range(6)))
        self.assertEqual(conditions,[])
        self.assertEqual(selections,[0,2,4])
        self.assertTrue(all(row['refit_every'] == 1 and row['select_every'] == 2
                            for row in diagnostics['model_checks']))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
