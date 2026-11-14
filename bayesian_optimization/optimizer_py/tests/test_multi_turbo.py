"""Multi-start selection, region isolation, validation and seeded Monte Carlo."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, mock_open
from contextlib import redirect_stdout
import io
from itertools import count

import numpy as np
import torch

import MultiTurbo as multi
import FinalEvaluation as final
import TurBo
import Objective
import BoUtils
import bo_main
from input import D, K


class RegionTests(unittest.TestCase):
    def test_effective_rank_and_budget(self):
        quality = np.arange(20,dtype=float)
        selected,info = multi.choose_region_seeds(quality,np.eye(20),200,
                                                  min_region_iterations=50)
        self.assertEqual(len(selected),4)
        self.assertEqual(info['effective_rank'],10)
        self.assertEqual(selected[0],19)
        selected,info = multi.choose_region_seeds(quality,np.ones((20,20)),200)
        self.assertEqual(len(selected),1)
        self.assertEqual(info['effective_rank'],1)

    def test_correlated_groups(self):
        correlation = np.full((8,8),.05)
        for i in range(0,8,2):
            correlation[i:i+2,i:i+2] = 1
        selected,info = multi.choose_region_seeds(np.arange(8),correlation,200)
        self.assertEqual(info['regions'],4)
        self.assertEqual(len(set(selected//2)),4)

    def test_manual_region_count(self):
        C = .8*np.ones((10,10)) + .2*np.eye(10)
        selected,_ = multi.choose_region_seeds(np.arange(10),C,200,n_regions=5)
        self.assertEqual(len(selected),5)
        with self.assertRaises(ValueError):
            multi.choose_region_seeds(np.arange(10),C,2,n_regions=5)

    def test_independent_incumbent(self):
        X = torch.stack([torch.full((D,),.2),torch.full((D,),.6)]).double()
        y = np.array([10.,1.]) # Global best is deliberately another region's point.
        state = TurBo.TurboState(dim=D,batch_size=1,best_index=1,best_value=1.)
        bounds = TurBo._tr_bounds(state,X,y)
        torch.testing.assert_close(bounds[0],torch.full((D,),.2,dtype=torch.double))
        TurBo.update_state(state,torch.tensor([[2.]]),new_index=2)
        self.assertEqual(state.best_index,2) # Improve its own incumbent, even below global 10.
        TurBo.update_state(state,torch.tensor([[1.5]]),new_index=3)
        self.assertEqual(state.best_index,2)

    def test_initial_design_is_shared_once(self):
        X = torch.stack([torch.full((D,),v) for v in (.1,.2,.4,.8)]).double()
        y = np.array([-1.,-4.,-3.,-5.])
        initial = (X,y,np.full(4,1e-4))
        seen = []
        def run(**kwargs):
            index = kwargs['start_index']
            seen.append((index,kwargs['max_iter']))
            self.assertIs(kwargs['initial_data'],initial)
            point = np.full((1,D),.3 if index == 0 else .7)
            values = np.r_[y,.5 if index == 0 else -2.]
            return np.vstack((X.numpy(),point)),values,np.array([np.nan]),[],[]
        selection = {'effective_rank':2.,'pool_size':4,'regions':2}
        with patch.object(multi,'sobol_sample',return_value=initial) as sample, \
             patch.object(multi,'select_initial_regions',return_value=(np.array([0,2]),selection)), \
             patch.object(multi,'bo_turbo_matern',side_effect=run):
            result = multi.bo_multi_turbo(total_iterations=5,n_regions=2,max_regions=5)
        sample.assert_called_once()
        self.assertEqual(seen,[(0,3),(2,2)])
        own = [p for p in result['finalists'] if p['region'] == 2][0]
        np.testing.assert_allclose(own['x'],.7)

    def test_real_region_selection_fit(self):
        torch.manual_seed(67)
        X = .1+.8*torch.rand(6,D,dtype=torch.double)
        y = -BoUtils.denormalize(X)[:,:K].sum(-1).numpy()
        # Coordinate tests exercise actual fitting; here bound its iteration count.
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        def fit(gp):
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood,gp),
                             optimizer_kwargs={'options':{'maxiter':3}})
        with patch.object(multi,'fit_gp_with_noise_floor',side_effect=fit):
            indices,info = multi.select_initial_regions(X,y,250)
        self.assertGreaterEqual(len(indices),1)
        self.assertLessEqual(len(indices),5)
        self.assertGreaterEqual(info['effective_rank'],1-1e-12)


class FinalValidationTests(unittest.TestCase):
    def test_invalid_budget_stops_before_search(self):
        with patch.object(bo_main,'bo_multi_turbo') as search:
            with self.assertRaises(SystemExit):
                bo_main.main(['--validation-repeats','1'])
            with self.assertRaises(SystemExit):
                bo_main.main(['--final-particles','2'])
            search.assert_not_called()

    def test_high_n_and_replicate_statistics(self):
        proposals = [{'x':np.full(D,.3),'region':1}, {'x':np.full(D,.7),'region':2}]
        responses = [(10.,1e12),(12.,1e12),(14.,1e12),(8.,1e12),(9.,1e12),(10.,1e12)]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(final,'evaluate_configuration',side_effect=responses) as evaluate:
            results = final.evaluate_finalists(proposals,output_dir=directory)
            self.assertEqual(evaluate.call_count,6)
            seeds = [call.args[0] for call in evaluate.call_args_list]
            self.assertEqual(len(set(seeds)),6)
            self.assertTrue(all(seed >= 1000000 for seed in seeds))
            self.assertTrue(all(call.kwargs['n_particles'] == 100000 for call in evaluate.call_args_list))
            self.assertTrue(all(call.kwargs['common_energy'] is False for call in evaluate.call_args_list))
            self.assertEqual(results[0]['region'],2)
            self.assertEqual(results[0]['mean_response'],9)
            self.assertAlmostEqual(results[0]['replicate_se'],1/np.sqrt(3))
            self.assertEqual(results[1]['mean_response'],12) # Arithmetic raw mean, not geometric mean.
            with open(Path(directory)/'validation_runs.csv') as file:
                self.assertEqual(len(list(csv.DictReader(file))),6)
            self.assertTrue((Path(directory)/'best_configuration.csv').is_file())

    def test_objective_energy_seed(self):
        physical = BoUtils.denormalize(np.full(D,.5)).numpy()
        centers,normals = BoUtils.unpack_configuration(physical)
        with patch.object(Objective.simulator,'launch_simulation',return_value=(10.,1.)) as launch:
            Objective.evaluate_configuration(67,centers,normals,n_particles=16)
            first = launch.call_args.args[7]
            Objective.evaluate_configuration(67,centers,normals,n_particles=16)
            second = launch.call_args.args[7]
            Objective.evaluate_configuration(68,centers,normals,n_particles=16)
            other = launch.call_args.args[7]
        for name in first:
            np.testing.assert_array_equal(first[name],second[name])
        self.assertFalse(np.array_equal(first['proton'],other['proton']))

    def test_common_energies_fresh_spatial_seeds_and_unchanged_objective(self):
        physical = BoUtils.denormalize(np.full(D,.5)).numpy()
        centers,normals = BoUtils.unpack_configuration(physical)
        incoming = []
        def simulate(*args):
            energy = Objective.incident_energy(args[7],args[8])
            incoming.append(energy)
            return .25*energy,.01*energy**2
        output = io.StringIO()
        with patch.object(Objective,'N',32), \
             patch.object(Objective.simulator,'launch_simulation',side_effect=simulate) as launch, \
             redirect_stdout(output):
            first_result = Objective.objective_function(101,centers,normals)
            second_result = Objective.objective_function(102,centers,normals)
        first,second = [call.args for call in launch.call_args_list]
        self.assertEqual([first[0],second[0]],[101,102])
        for name in first[7]:
            np.testing.assert_array_equal(first[7][name],second[7][name])
            np.testing.assert_array_equal(first[8][name],second[8][name])
        self.assertEqual(incoming[0],incoming[1])
        self.assertAlmostEqual(first_result[0],np.log(.25*incoming[0]))
        self.assertEqual(first_result,second_result)
        self.assertIn('relative received=25.000000%',output.getvalue())
        self.assertIn('Owen spatial seed=102',output.getvalue())

    def test_validation_energy_stream_is_distinct_even_for_matching_seed_numbers(self):
        physical = BoUtils.denormalize(np.full(D,.5)).numpy()
        centers,normals = BoUtils.unpack_configuration(physical)
        with patch.object(Objective,'BO_ENERGY_SEED',67), \
             patch.object(Objective.simulator,'launch_simulation',return_value=(10.,1.)) as launch:
            Objective.evaluate_configuration(67,centers,normals,n_particles=32,common_energy=True)
            common = launch.call_args.args[7]
            Objective.evaluate_configuration(67,centers,normals,n_particles=32,common_energy=False)
            fresh = launch.call_args.args[7]
        self.assertFalse(np.array_equal(common['proton'],fresh['proton']))

    def test_incident_diagnostic_uses_flux_weights_and_total_nuclear_energy(self):
        particles = 'i,particleName,m,q\n1,proton,1,1\n2,Fe56,2*28,26\n'
        samples = {'proton':np.array([1.,3.,10.,20.]),'Fe56':np.array([2.,4.,15.,25.])}
        strata = {'proton':np.array([[2.,2.],[2.,3.]]),'Fe56':np.array([[2.,1.],[2.,4.]])}
        with patch.object(Objective,'open',mock_open(read_data=particles),create=True):
            incoming = Objective.incident_energy(samples,strata)
        self.assertEqual(incoming,1e6*(2*2+3*15+56*(1*3+4*20)))

    def test_spatial_seed_allocator_is_unique_and_separate_from_default_validation(self):
        with patch.object(Objective,'_optimization_seeds',count(Objective.SEED)):
            seeds = [Objective.next_optimization_seed() for _ in range(1000)]
        self.assertEqual(len(set(seeds)),1000)
        self.assertTrue(all(seed < Objective.SEED+1000000 for seed in seeds))

    def test_retry_retains_common_energies_and_consumes_a_fresh_scramble(self):
        with patch.object(Objective,'_optimization_seeds',count(Objective.SEED+1)), \
             patch.object(Objective,'evaluate_configuration',side_effect=[FloatingPointError(),(10.,1.)]) as evaluate:
            Objective.objective_function(Objective.SEED,None,None)
        self.assertEqual([call.args[0] for call in evaluate.call_args_list],
                         [Objective.SEED,Objective.SEED+1])
        self.assertTrue(all(call.kwargs['common_energy'] for call in evaluate.call_args_list))

    def test_driver_automatically_validates(self):
        X = torch.full((2,D),.5,dtype=torch.double)
        search = {'initial':(X,np.array([-1.,-2.]),np.array([.1,.1])),
                  'selection':{'regions':1},'seed_indices':np.array([0]),
                  'regions':[{'region':1,'budget':1,'X':X.numpy(),'y':np.array([-1.,-2.])}],
                  'finalists':[{'x':np.full(D,.5),'region':1}]}
        validation = [{'mean_response':1.,'ci95_lower':.8,'ci95_upper':1.2}]
        with tempfile.TemporaryDirectory() as root, \
             patch.object(bo_main,'bo_multi_turbo',return_value=search), \
             patch.object(bo_main,'evaluate_finalists',return_value=validation) as evaluate:
            bo_main.main(['--output-dir',str(Path(root)/'new-run'),'--no-plot'])
            self.assertEqual(evaluate.call_args.kwargs['n_particles'],100000)
            self.assertEqual(evaluate.call_args.kwargs['repeats'],3)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
