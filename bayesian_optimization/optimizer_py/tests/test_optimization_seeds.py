"""Portable integer seed state, including safe checkpointing and retries."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import Objective


class OptimizationSeedTests(unittest.TestCase):
    def setUp(self):
        saved = Objective.optimization_seed_state()
        self.addCleanup(Objective.set_optimization_seed,saved)
        Objective.set_optimization_seed(Objective.SEED)

    def test_peeking_is_non_consuming_and_allocations_are_unique(self):
        self.assertIs(type(Objective.optimization_seed_state()),int)
        self.assertEqual(Objective.optimization_seed_state(),Objective.SEED)
        self.assertEqual(Objective.optimization_seed_state(),Objective.SEED)
        values = [Objective.next_optimization_seed() for _ in range(3)]
        self.assertEqual(values,list(range(Objective.SEED,Objective.SEED+3)))
        self.assertEqual(Objective.optimization_seed_state(),Objective.SEED+3)

    def test_checkpoint_roundtrip_needs_no_pickled_iterator(self):
        Objective.set_optimization_seed(np.int64(Objective.SEED+123))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            np.savez(path,next_spatial_seed=Objective.optimization_seed_state())
            Objective.next_optimization_seed()
            with np.load(path,allow_pickle=False) as state:
                Objective.set_optimization_seed(state['next_spatial_seed'].item())
        self.assertEqual(Objective.next_optimization_seed(),Objective.SEED+123)
        self.assertEqual(Objective.optimization_seed_state(),Objective.SEED+124)

    def test_exhaustion_does_not_advance_into_validation_seed_range(self):
        end = Objective.SEED+1_000_000
        Objective.set_optimization_seed(end-1)
        self.assertEqual(Objective.next_optimization_seed(),end-1)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError,'range exhausted'):
                Objective.next_optimization_seed()
            self.assertEqual(Objective.optimization_seed_state(),end)
        Objective.set_optimization_seed(end) # Restoring an exhausted checkpoint is valid.
        self.assertEqual(Objective.optimization_seed_state(),end)

    def test_invalid_state_is_rejected_without_changing_current_seed(self):
        for bad in [True,False,Objective.SEED-1,Objective.SEED+1_000_001,
                    Objective.SEED+.5,float('nan'),float('inf'),'67']:
            with self.subTest(state=bad),self.assertRaises(ValueError):
                Objective.set_optimization_seed(bad)
            self.assertEqual(Objective.optimization_seed_state(),Objective.SEED)

    def test_retry_uses_a_new_spatial_seed_and_keeps_common_energies(self):
        first = Objective.next_optimization_seed()
        with patch.object(Objective,'evaluate_configuration',
                side_effect=[FloatingPointError(),(10.,1.)]) as evaluate:
            Objective.objective_function(first,None,None)
        self.assertEqual([call.args[0] for call in evaluate.call_args_list],
                         [Objective.SEED,Objective.SEED+1])
        self.assertTrue(all(call.kwargs['common_energy'] for call in evaluate.call_args_list))
        self.assertEqual(Objective.optimization_seed_state(),Objective.SEED+2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
