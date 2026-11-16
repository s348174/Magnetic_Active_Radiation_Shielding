"""KG initialization respects memory caps and preserves full one-shot trees."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from botorch.acquisition import qKnowledgeGradient
from botorch.models import SingleTaskGP

import AcquisitionOptimizer
import BoLoop
import KGInitialization
from input import D


class KGInitializationTests(unittest.TestCase):
    def setUp(self):
        self.bounds = torch.stack((torch.zeros(D,dtype=torch.double),torch.ones(D,dtype=torch.double)))
        self.starts = torch.full((4,3,D),.5,dtype=torch.double)
        model = torch.nn.Module()
        model.num_outputs = 1
        self.acquisition = SimpleNamespace(model=model,objective=None,posterior_transform=None,
                                           get_augmented_q_batch_size=lambda q:q+2)

    def test_lower_memory_caps_reach_outer_scoring_and_inner_optimizer(self):
        inner = torch.stack((torch.full((1,D),.2,dtype=torch.double),
                             torch.full((1,D),.8,dtype=torch.double)))
        def optimize(**kwargs):
            self.assertEqual(kwargs['options']['batch_limit'],1)
            self.assertEqual(kwargs['options']['init_batch_limit'],3)
            self.assertEqual(kwargs['num_restarts'],2)
            self.assertEqual(kwargs['raw_samples'],16)
            self.assertFalse(kwargs['return_best_only'])
            return inner,torch.tensor([1.,2.],dtype=torch.double)
        with patch.object(KGInitialization,'BO_OPT_BATCH_LIMIT',1), \
             patch.object(KGInitialization,'BO_OPT_INIT_BATCH_LIMIT',3), \
             patch.object(KGInitialization,'gen_batch_initial_conditions',return_value=self.starts.clone()) as generate, \
             patch.object(AcquisitionOptimizer,'botorch_optimize_acqf',side_effect=optimize) as fit:
            actual = KGInitialization.gen_one_shot_kg_initial_conditions(
                self.acquisition,self.bounds,1,4,8,options={'batch_limit':9,'init_batch_limit':20})
        self.assertEqual(fit.call_count,1)
        self.assertEqual(generate.call_args.kwargs['options']['batch_limit'],1)
        self.assertEqual(generate.call_args.kwargs['options']['init_batch_limit'],3)
        # Only the heuristic fantasy slot changes; actual and random fantasy slots remain intact.
        torch.testing.assert_close(actual[:,:2],self.starts[:,:2])
        self.assertTrue(((actual[:,2,0] == .2)|(actual[:,2,0] == .8)).all())

    def test_zero_heuristic_fantasy_count_does_not_overwrite_actual_candidates(self):
        self.acquisition.get_augmented_q_batch_size = lambda q:q+1
        starts = self.starts[:,:2].clone()
        with patch.object(KGInitialization,'gen_batch_initial_conditions',return_value=starts), \
             patch.object(KGInitialization,'optimize_acqf') as optimize:
            actual = KGInitialization.gen_one_shot_kg_initial_conditions(
                self.acquisition,self.bounds,1,4,8,options={'frac_random':.9})
        torch.testing.assert_close(actual,starts)
        optimize.assert_not_called()

    def test_nonpositive_memory_cap_fails_before_initialization(self):
        with patch.object(KGInitialization,'BO_OPT_INIT_BATCH_LIMIT',0), \
             patch.object(KGInitialization,'gen_batch_initial_conditions') as generate, \
             self.assertRaises(ValueError):
            KGInitialization.gen_one_shot_kg_initial_conditions(self.acquisition,self.bounds,1,4,8)
        generate.assert_not_called()

    def test_real_single_output_kg_initialization_with_small_memory_caps(self):
        X = torch.tensor([[.1,.2],[.3,.8],[.7,.4],[.9,.9]],dtype=torch.double)
        y = torch.sin(X.sum(dim=-1,keepdim=True))
        gp = SingleTaskGP(X,y,train_Yvar=torch.full_like(y,.01)).eval()
        acquisition = qKnowledgeGradient(gp,num_fantasies=2)
        bounds = torch.tensor([[0.,0.],[1.,1.]],dtype=torch.double)
        original = KGInitialization.optimize_acqf
        def small_inner(**kwargs):
            self.assertEqual(kwargs['options']['batch_limit'],1)
            self.assertEqual(kwargs['options']['init_batch_limit'],2)
            kwargs['options']['maxiter'] = 3
            return original(**kwargs)
        with patch.object(KGInitialization,'BO_OPT_BATCH_LIMIT',1), \
             patch.object(KGInitialization,'BO_OPT_INIT_BATCH_LIMIT',2), \
             patch.object(KGInitialization,'optimize_acqf',side_effect=small_inner):
            starts = KGInitialization.gen_one_shot_kg_initial_conditions(
                acquisition,bounds,1,3,8,options={'num_inner_restarts':2,'raw_inner_samples':8})
        self.assertEqual(starts.shape,(3,3,2))
        self.assertTrue(torch.isfinite(starts).all())
        self.assertTrue(((starts >= bounds[0])&(starts <= bounds[1])).all())


class KGProposalTests(unittest.TestCase):
    def setUp(self):
        self.X = torch.stack([torch.full((D,),v,dtype=torch.double) for v in (.25,.5)])
        self.starts = torch.full((40,3,D),.75,dtype=torch.double)
        self.starts[:,1,:] = .2
        self.starts[:,2,:] = .8

    def test_duplicate_kg_result_uses_distinct_full_tree_fallback_in_bounded_batches(self):
        sizes = []
        def acquisition(trees):
            self.assertEqual(trees.shape[-2],3)
            torch.testing.assert_close(trees[:,1:],self.starts[:len(trees),1:])
            sizes.append(len(trees))
            return trees[:,0,0]+10*trees[:,2,0]
        with patch.object(BoLoop,'BO_OPT_BATCH_LIMIT',1), \
             patch.object(BoLoop,'BO_OPT_INIT_BATCH_LIMIT',3), \
             patch.object(BoLoop,'gen_one_shot_kg_initial_conditions',return_value=self.starts.clone()) as generate, \
             patch.object(BoLoop,'_observed_means',return_value=torch.tensor([2.,1.])), \
             patch.object(BoLoop,'optimize_acqf',return_value=(self.X[:1],torch.tensor(100.))):
            candidate,value = BoLoop._kg_candidate(acquisition,None,None,self.X)
        self.assertFalse(BoLoop.duplicate_safe_candidate(self.X,candidate))
        self.assertAlmostEqual(value.item(),candidate[0,0].item()+8.)
        self.assertLessEqual(max(sizes),3)
        self.assertEqual(generate.call_args.kwargs['options']['batch_limit'],1)
        self.assertEqual(generate.call_args.kwargs['options']['init_batch_limit'],3)

    def test_kg_does_not_return_observed_point_if_all_restart_trees_duplicate(self):
        starts = self.X[:1,None,:].expand(40,3,D).clone()
        with patch.object(BoLoop,'gen_one_shot_kg_initial_conditions',return_value=starts), \
             patch.object(BoLoop,'_observed_means',return_value=torch.tensor([2.,1.])), \
             patch.object(BoLoop,'observed_local_points',return_value=starts[:20,:1].clone()), \
             patch.object(BoLoop,'optimize_acqf',return_value=(self.X[:1],torch.tensor(100.))), \
             self.assertRaisesRegex(RuntimeError,'no finite unevaluated candidate'):
            BoLoop._kg_candidate(lambda X:torch.zeros(len(X)),None,None,self.X)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
