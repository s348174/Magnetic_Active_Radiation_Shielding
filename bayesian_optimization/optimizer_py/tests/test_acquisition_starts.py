"""Observed neighbourhoods must survive broad acquisition initialization."""
import unittest
from unittest.mock import patch

import torch
from botorch.acquisition import AcquisitionFunction, OneShotAcquisitionFunction

import AcquisitionStarts


class NarrowPeak(AcquisitionFunction):
    def __init__(self, center):
        super().__init__(model=None)
        self.center = center
        self.batches = []

    def forward(self, X):
        self.batches.append(X.shape[0])
        return -((X[...,0,:]-self.center)/.001).square().sum(-1)


class AcquisitionStartTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        self.bounds = torch.stack((torch.zeros(8,dtype=torch.double),
                                   torch.ones(8,dtype=torch.double)))
        self.X = torch.stack([torch.full((8,),v,dtype=torch.double) for v in (.25,.5,.75)])
        self.ranking = torch.tensor([3.,2.,1.],dtype=torch.double)

    def test_preserves_known_local_and_global_starts_in_small_scoring_batches(self):
        acq = NarrowPeak(self.X[0])
        def optimize(**kwargs):
            starts = kwargs['batch_initial_conditions'].squeeze(-2)
            self.assertEqual(starts.shape,(9,8))
            self.assertTrue((starts == self.X[0]).all(-1).any())
            distance = (starts-self.X[0]).norm(dim=-1)
            self.assertTrue(((distance > 0)&(distance < .3)).any())
            self.assertTrue((distance > .5).any())
            return torch.zeros(1,8,dtype=torch.double),torch.tensor(100.)
        with patch.object(AcquisitionStarts,'BO_OPT_INIT_BATCH_LIMIT',2):
            candidate,value = AcquisitionStarts.optimize_acquisition(acq_function=acq,
                bounds=self.bounds,X=self.X,ranking=self.ranking,num_restarts=9,raw_samples=64,
                optimizer=optimize,allow_observed=True)
        torch.testing.assert_close(candidate,self.X[:1])
        self.assertEqual(value.item(),0.)
        self.assertLessEqual(max(acq.batches),2)

    def test_returns_optimized_improvement_using_actual_score(self):
        center = torch.full((8,),.35,dtype=torch.double)
        acq = NarrowPeak(center)
        with patch.object(AcquisitionStarts,'optimize_acqf',return_value=(center[None],torch.tensor(-1e20))):
            candidate,value = AcquisitionStarts.optimize_acquisition(acq_function=acq,
                bounds=self.bounds,X=self.X,ranking=self.ranking,num_restarts=6,raw_samples=32)
        torch.testing.assert_close(candidate,center[None])
        self.assertEqual(value.item(),0.)

    def test_known_score_is_not_lost_when_optimizer_returns_nonfinite_or_infeasible_point(self):
        for bad in (float('nan'),2.):
            with self.subTest(bad=bad),patch.object(AcquisitionStarts,'optimize_acqf',
                    return_value=(torch.full((1,8),bad,dtype=torch.double),torch.tensor(100.))):
                candidate,value = AcquisitionStarts.optimize_acquisition(
                    acq_function=NarrowPeak(self.X[0]),bounds=self.bounds,X=self.X,
                    ranking=self.ranking,num_restarts=3,raw_samples=16,allow_observed=True)
            torch.testing.assert_close(candidate,self.X[:1])
            self.assertEqual(value.item(),0.)

    def test_bo_rejects_duplicate_optimizer_result_but_keeps_observed_starts(self):
        acquisition = NarrowPeak(self.X[0])
        def duplicate(**kwargs):
            starts = kwargs['batch_initial_conditions'].squeeze(-2)
            self.assertTrue((starts == self.X[0]).all(-1).any())
            return self.X[:1],torch.tensor(0.)
        candidate,value = AcquisitionStarts.optimize_acquisition(
            acq_function=acquisition,bounds=self.bounds,X=self.X,ranking=self.ranking,
            num_restarts=9,raw_samples=32,optimizer=duplicate)
        self.assertTrue(AcquisitionStarts.new_point_mask(candidate,self.X).item())
        torch.testing.assert_close(value,acquisition(candidate.unsqueeze(-2))[0])
        self.assertLess(value.item(),0.)

    def test_near_duplicates_use_the_same_per_coordinate_tolerance_as_bo_loop(self):
        # Every coordinate is within tolerance, despite a larger Euclidean norm.
        close = self.X[:1]+.9e-6
        self.assertFalse(AcquisitionStarts.new_point_mask(close,self.X).item())
        with patch.object(AcquisitionStarts,'optimize_acqf',return_value=(close,torch.tensor(100.))):
            candidate,_ = AcquisitionStarts.optimize_acquisition(
                acq_function=NarrowPeak(self.X[0]),bounds=self.bounds,X=self.X,
                ranking=self.ranking,num_restarts=6,raw_samples=32)
        self.assertTrue(AcquisitionStarts.new_point_mask(candidate,self.X).item())

    def test_no_finite_new_point_raises_instead_of_returning_a_duplicate(self):
        class OnlyObserved(AcquisitionFunction):
            def __init__(self, X):
                super().__init__(model=None)
                self.X = X
            def forward(self, X):
                observed = (X[...,0,:] == self.X[0]).all(dim=-1)
                return torch.where(observed,0.,float('nan'))
        with patch.object(AcquisitionStarts,'optimize_acqf',return_value=(self.X[:1],torch.tensor(0.))), \
             self.assertRaisesRegex(RuntimeError,'No finite unevaluated candidate'):
            AcquisitionStarts.optimize_acquisition(acq_function=OnlyObserved(self.X),
                bounds=self.bounds,X=self.X,ranking=self.ranking,num_restarts=3,raw_samples=16)

    def test_real_candidate_slots_are_feasible_and_include_top_observation(self):
        for count in (1,2,9):
            with self.subTest(count=count):
                points = AcquisitionStarts.observed_local_points(self.bounds,self.X,self.ranking,count)
                self.assertEqual(points.shape,(count,1,8))
                self.assertTrue(((points >= self.bounds[0])&(points <= self.bounds[1])).all())
                torch.testing.assert_close(points[0],self.X[:1])

    def test_does_not_seed_one_shot_fantasies_with_real_designs(self):
        class OneShot(OneShotAcquisitionFunction):
            def __init__(self):
                super().__init__(model=None)
            def forward(self,X):
                return X.sum(dim=(-1,-2))
            def get_augmented_q_batch_size(self,q):
                return q+2
            def extract_candidates(self,X):
                return X[...,:1,:]
        for acq,q in ((NarrowPeak(self.X[0]),2),(OneShot(),1)):
            with self.subTest(q=q,kind=type(acq).__name__),self.assertRaises(ValueError):
                AcquisitionStarts.optimize_acquisition(acq_function=acq,bounds=self.bounds,
                    X=self.X,ranking=self.ranking,num_restarts=3,raw_samples=16,q=q)


if __name__ == '__main__':
    unittest.main(verbosity=2)
