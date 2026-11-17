"""Many promising seeds share one strict budget and retain local incumbents."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import MultiTurbo as multi
import bo_main
from input import D


class ManyRegionSelectionTests(unittest.TestCase):
    def test_twelve_or_fifteen_distinct_seeds_come_from_the_promising_pool(self):
        quality = np.linspace(-35.,-34.,100)
        for count in (12,15):
            with self.subTest(regions=count):
                indices,report = multi.choose_region_seeds(
                    quality,np.eye(100),1050,n_regions=count,max_regions=15)
                self.assertEqual(len(indices),count)
                self.assertEqual(len(np.unique(indices)),count)
                self.assertEqual(indices[0],99)
                self.assertEqual(report['pool_size'],30)
                self.assertTrue(set(indices).issubset(set(range(70,100))))

    def test_real_physical_selection_ranks_observations_without_a_gp_fit(self):
        generator = torch.Generator().manual_seed(67)
        X = .1+.8*torch.rand(100,D,dtype=torch.double,generator=generator)
        quality = np.linspace(-35.,-34.,100)
        # Reject the old learned-GP ranking path while exercising real geometry.
        with patch.object(multi,'SingleTaskGP',side_effect=AssertionError('Unnecessary GP fit'),create=True), \
             patch.object(multi,'fit_gp_with_noise_floor',side_effect=AssertionError('Unnecessary GP fit'),create=True):
            indices,report = multi.select_initial_regions(
                X,quality,840,n_regions=12,max_regions=15)
        self.assertEqual(len(indices),12)
        self.assertEqual(indices[0],99)
        self.assertEqual(report['pool_size'],30)
        self.assertTrue(set(indices).issubset(set(range(70,100))))

    def test_identical_physical_geometries_do_not_launch_redundant_regions(self):
        X = torch.full((100,D),.3,dtype=torch.double)
        indices,report = multi.select_initial_regions(
            X,np.linspace(-35.,-34.,100),840,n_regions=12,max_regions=15)
        np.testing.assert_array_equal(indices,[99])
        self.assertEqual(report['regions'],1)


class SharedBudgetTests(unittest.TestCase):
    def test_driver_saves_total_budget_instead_of_summing_redistributed_grants(self):
        X = torch.full((2,D),.5,dtype=torch.double)
        y = np.array([-1.,-2.])
        selection = dict(regions=2,refinement_budget=200,used_iterations=140,unused_iterations=60)
        regions = [dict(region=index,budget=budget,X=X.numpy(),y=y,
                        acquisition=np.array([np.nan]),posterior_mean=np.array([0.]),
                        posterior_variance=np.array([1.]))
                   for index,budget in ((1,100),(2,130))]
        search = dict(initial=(X,y,np.full(2,1e-4)),selection=selection,
                      seed_indices=np.array([0,1]),regions=regions,
                      finalists=[dict(x=np.full(D,.5),region=1)])
        validation = [dict(mean_response=1.,ci95_lower=.9,ci95_upper=1.1)]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'run'
            with patch.object(bo_main,'bo_multi_turbo',return_value=search), \
                 patch.object(bo_main,'evaluate_finalists',return_value=validation):
                bo_main.main(['--method','turbo','--regions','2','--iterations','200',
                              '--output-dir',str(output),'--no-plot'])
            settings = json.loads((output/'region_selection.json').read_text())
            self.assertEqual(sum(row['budget'] for row in regions),230)
            self.assertEqual(settings['total_refinement_budget'],200)
            self.assertEqual(settings['used_iterations'],140)
            self.assertEqual(settings['unused_iterations'],60)

    def test_unused_region_budget_moves_to_remaining_seeds_without_changing_total(self):
        for last_attempts in (70,130):
            with self.subTest(last_attempts=last_attempts):
                X = torch.stack([torch.full((D,),v) for v in (.1,.2,.4,.8)]).double()
                y = np.array([-4.,-1.,-3.,-5.])
                initial = (X,y,np.full(4,1e-4))
                calls = []

                def run(**kwargs):
                    self.assertIs(kwargs['initial_data'],initial)
                    self.assertTrue(kwargs['return_diagnostics'])
                    index = kwargs['start_index']
                    calls.append((index,kwargs['max_iter']))
                    attempts = 70 if index == 0 else last_attempts
                    # The global initial winner is better than region two's
                    # own best. Its local finalist must still come from its run.
                    point,value = (.3,-.5) if index == 0 else (.7,-2.)
                    own_X = np.full((attempts,D),point)
                    own_y = np.full(attempts,value)
                    terminated = attempts < kwargs['max_iter']
                    length = .00625 if terminated else .1
                    diagnostics = dict(
                        stop_reason='length_min' if terminated else 'iteration_budget',
                        budget=kwargs['max_iter'],attempted_iterations=attempts,
                        successful_evaluations=attempts,final_length=length,
                        length_history=np.full(attempts,length),final_model_evaluations=attempts,
                    )
                    return (np.vstack((X.numpy(),own_X)),np.r_[y,own_y],
                            np.full(attempts,np.nan),np.zeros(attempts),np.ones(attempts),
                            diagnostics)

                selection = {'effective_rank':2.,'pool_size':4,'regions':2}
                with patch.object(multi,'sobol_sample',return_value=initial) as sample, \
                     patch.object(multi,'select_initial_regions',
                         return_value=(np.array([0,2]),selection)), \
                     patch.object(multi,'bo_turbo_matern',side_effect=run):
                    result = multi.bo_multi_turbo(
                        total_iterations=200,n_regions=2,max_regions=15,max_finalists=5)
                sample.assert_called_once()
                self.assertEqual(calls,[(0,100),(2,130)])
                self.assertEqual(len(result['regions']),2)
                self.assertEqual(result['selection']['refinement_budget'],200)
                self.assertEqual(result['selection']['used_iterations'],70+last_attempts)
                self.assertEqual(result['selection']['unused_iterations'],130-last_attempts)
                self.assertLessEqual(sum(r['attempted_iterations'] for r in result['regions']),200)
                self.assertEqual(sum(r['budget'] for r in result['regions']),230)
                self.assertEqual(result['regions'][0]['stop_reason'],'length_min')
                self.assertEqual(len(result['regions'][0]['length_history']),70)
                self.assertEqual(result['regions'][0]['final_length'],.00625)
                own = next(row for row in result['finalists'] if row['region'] == 2)
                np.testing.assert_allclose(own['x'],.7)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
