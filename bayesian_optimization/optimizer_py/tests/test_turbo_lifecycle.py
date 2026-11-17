"""Ten accepted nonimprovements shrink a region; a small region terminates."""
from types import SimpleNamespace
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import numpy as np
import torch

import TurBo
from input import D


class TurboLifecycleTests(unittest.TestCase):
    def test_halving_window_is_independent_of_dimension_and_ends_after_seven_halves(self):
        for dimension in (20,40):
            state = TurBo.TurboState(dim=dimension,batch_size=1,best_value=-34.)
            self.assertEqual(state.failure_tolerance,10)
            for count in range(1,71):
                TurBo.update_state(state,torch.tensor([[-34.1]]),new_index=count)
                self.assertAlmostEqual(state.length,.8/2**(count//10))
                self.assertEqual(state.restart_triggered,count == 70)
            self.assertLess(state.length,.01)
            self.assertEqual(state.best_value,-34.)

    def test_any_strict_improvement_resets_failure_window_even_for_log_objectives(self):
        state = TurBo.TurboState(dim=20,batch_size=1,best_value=-34.)
        for _ in range(9):
            TurBo.update_state(state,torch.tensor([[-34.1]],dtype=torch.double))
        TurBo.update_state(state,torch.tensor([[-33.99999]],dtype=torch.double),new_index=12)
        self.assertEqual(state.failure_counter,0)
        self.assertEqual(state.best_index,12)
        for _ in range(9):
            TurBo.update_state(state,torch.tensor([[-34.1]],dtype=torch.double))
        self.assertAlmostEqual(state.length,.8)
        TurBo.update_state(state,torch.tensor([[-34.1]],dtype=torch.double))
        self.assertAlmostEqual(state.length,.4)

    def test_exact_minimum_does_not_terminate_and_expansion_is_preserved(self):
        state = TurBo.TurboState(dim=20,batch_size=1,length=.02,length_min=.01,
                                 failure_tolerance=1,best_value=0.)
        TurBo.update_state(state,torch.tensor([[-1.]]))
        self.assertEqual(state.length,.01)
        self.assertFalse(state.restart_triggered)
        TurBo.update_state(state,torch.tensor([[-1.]]))
        self.assertTrue(state.restart_triggered)
        expanded = TurBo.TurboState(dim=20,batch_size=1,best_value=0.)
        for value in range(1,11):
            TurBo.update_state(expanded,torch.tensor([[float(value)]]))
        self.assertEqual(expanded.length,1.6)

    def test_invalid_observations_do_not_mutate_state(self):
        state = TurBo.TurboState(dim=20,batch_size=1,best_value=-34.)
        for values in (torch.tensor([[float('nan')]]),torch.tensor([[float('inf')]]),
                       torch.zeros(2,1)):
            with self.assertRaises(ValueError):
                TurBo.update_state(state,values)
            self.assertEqual(state.failure_counter,0)
            self.assertEqual(state.best_value,-34.)

    def run_loop(self,iterations,return_diagnostics,invalid_first=False):
        X = torch.stack((torch.full((D,),.5),torch.full((D,),.2))).double()
        initial = (X,np.array([-34.,-34.1]),np.full(2,.001))
        proposals = []

        def generate(**kwargs):
            state = kwargs['state']
            centre = kwargs['X_train'][state.best_index]
            candidate = centre.clone()
            candidate[0] += .25*state.length
            candidate[1] += state.length*.01*(len(proposals)%10)
            proposals.append(candidate.clone())
            return candidate.unsqueeze(0)

        def model(features,*args,**kwargs):
            def posterior(points):
                mean = torch.full(points.shape[:-1]+(1,),-34.,dtype=points.dtype)
                covariance = torch.eye(points.shape[-2],dtype=points.dtype)*.01
                return SimpleNamespace(mean=mean,distribution=SimpleNamespace(covariance_matrix=covariance))
            return SimpleNamespace(num_outputs=1,posterior=posterior,
                covar_module=lambda a,b,diag=False:torch.ones(a.shape[:-1],dtype=a.dtype))

        responses = [(float('nan'),.001)] if invalid_first else []
        responses += [(34.5,.001)]*iterations
        with patch.object(TurBo,'_generate_batch_ts',side_effect=generate), \
             patch.object(TurBo,'SingleTaskGP',side_effect=model), \
             patch.object(TurBo,'fit_gp_with_noise_floor') as fit, \
             patch.object(TurBo,'objective_function',side_effect=responses) as simulate, \
             patch.object(TurBo,'next_optimization_seed',return_value=67), \
             patch.object(TurBo,'optimize_acqf',return_value=(X[:1],torch.tensor(-34.))):
            with self.assertWarns(UserWarning) if invalid_first else nullcontext():
                result = TurBo.bo_turbo_matern(initial_data=initial,start_index=0,
                    max_iter=iterations,return_diagnostics=return_diagnostics)
        return result,fit.call_count,simulate.call_count

    def test_local_run_terminates_without_restarting_and_retains_terminal_observation(self):
        result,fits,simulations = self.run_loop(100,True)
        X,y,acquisition,means,variances,diagnostics = result
        self.assertEqual(simulations,70)
        self.assertEqual(fits,70) # Initial fit + 69 updates; no fit of the dead region.
        self.assertEqual(len(X),72)
        self.assertEqual(len(y),72)
        self.assertEqual(len(acquisition),70)
        self.assertEqual(len(means),70)
        self.assertEqual(len(variances),70)
        self.assertTrue(np.isnan(means[-1]))
        self.assertTrue(np.isnan(variances[-1]))
        self.assertEqual(diagnostics['stop_reason'],'length_min')
        self.assertEqual(diagnostics['successful_evaluations'],70)
        self.assertEqual(diagnostics['attempted_iterations'],70)
        self.assertEqual(diagnostics['final_model_evaluations'],69)
        self.assertAlmostEqual(diagnostics['final_length'],.00625)
        self.assertEqual(len(diagnostics['length_history']),70)
        self.assertEqual(diagnostics['best_index'],0)

    def test_legacy_return_and_invalid_simulation_does_not_count_toward_halving(self):
        legacy,_,_ = self.run_loop(1,False)
        self.assertEqual(len(legacy),5)
        result,_,simulations = self.run_loop(11,True,invalid_first=True)
        diagnostics = result[-1]
        self.assertEqual(simulations,11)
        self.assertEqual(diagnostics['successful_evaluations'],10)
        self.assertEqual(diagnostics['attempted_iterations'],11)
        self.assertAlmostEqual(diagnostics['final_length'],.4)
        self.assertEqual(diagnostics['stop_reason'],'iteration_budget')

if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
