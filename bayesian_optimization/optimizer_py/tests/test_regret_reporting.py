"""Separate diagnostic/stopping schedules and robust saved-run plotting."""
from contextlib import chdir
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from RegretStopping import SimpleRegretStopping
import bo_main
import plot_regret


GOOD = {'max_ucb':0.,'incumbent_lcb':0.,'incumbent_index':0,'estimated_regret':0.}


class RecordingScheduleTests(unittest.TestCase):
    def test_dense_records_do_not_advance_patience(self):
        stopping = SimpleRegretStopping(min_evaluations=10,check_every=10,record_every=2,patience=3,grid_size=8)
        with patch.object(stopping,'estimate',return_value=GOOD) as estimate:
            for evaluation in range(1,30):
                self.assertFalse(stopping.check(None,None,None,evaluation,evaluation))
            self.assertTrue(stopping.check(None,None,None,30,30))
        self.assertEqual(estimate.call_count,15) # Coincident recording/checking computes only once.
        self.assertEqual([row['evaluations'] for row in stopping.history],list(range(2,31,2)))
        self.assertEqual([row['evaluations'] for row in stopping.history if row['stopping_check']],[10,20,30])
        self.assertEqual(stopping.history[0]['consecutive_passes'],0)
        self.assertEqual(stopping.history[5]['consecutive_passes'],1)

    def test_stopping_checks_are_recorded_even_off_the_diagnostic_grid(self):
        stopping = SimpleRegretStopping(min_evaluations=5,check_every=10,record_every=2,patience=3,grid_size=8)
        with patch.object(stopping,'estimate',return_value=GOOD):
            for evaluation in range(1,25):
                self.assertFalse(stopping.check(None,None,None,evaluation,evaluation))
            self.assertTrue(stopping.check(None,None,None,25,25))
        self.assertEqual([row['evaluations'] for row in stopping.history if row['stopping_check']],[5,15,25])

    def test_recording_works_with_early_stopping_disabled(self):
        stopping = SimpleRegretStopping(enabled=False,record_every=1,grid_size=8)
        with patch.object(stopping,'estimate',return_value=GOOD):
            for evaluation in range(1,5):
                self.assertFalse(stopping.check(None,None,None,evaluation,evaluation))
        self.assertEqual(len(stopping.history),4)
        self.assertTrue(all(not row['stopping_check'] for row in stopping.history))
        self.assertEqual(stopping.consecutive,0)

    def test_diagnostic_failure_holds_patience_but_stopping_failure_resets_it(self):
        stopping = SimpleRegretStopping(min_evaluations=2,check_every=2,record_every=1,patience=3,grid_size=8)
        with patch.object(stopping,'estimate',side_effect=[GOOD,GOOD,RuntimeError('diagnostic failure'),RuntimeError('stopping failure')]):
            stopping.check(None,None,None,1,1)
            stopping.check(None,None,None,2,2)
            self.assertEqual(stopping.consecutive,1)
            with self.assertWarns(UserWarning):
                self.assertFalse(stopping.check(None,None,None,3,3))
            self.assertEqual(stopping.consecutive,1)
            with self.assertWarns(UserWarning):
                self.assertFalse(stopping.check(None,None,None,4,4))
        self.assertEqual(stopping.consecutive,0)

    def test_diagnostics_preserve_torch_random_state(self):
        stopping = SimpleRegretStopping(enabled=False,record_every=1,grid_size=8)
        def estimate(*args):
            torch.rand(20)
            return GOOD
        torch.manual_seed(67)
        state = torch.random.get_rng_state()
        expected = torch.rand(3)
        torch.random.set_rng_state(state)
        with patch.object(stopping,'estimate',side_effect=estimate):
            stopping.check(None,None,None,1,1)
        torch.testing.assert_close(torch.rand(3),expected)

    def test_recording_can_be_disabled_and_intervals_are_integers(self):
        stopping = SimpleRegretStopping(enabled=False,record_every=0,grid_size=8)
        with patch.object(stopping,'estimate') as estimate:
            self.assertFalse(stopping.check(None,None,None,1000,1000))
            estimate.assert_not_called()
        for interval in (-1,.5):
            with self.assertRaises(ValueError):
                SimpleRegretStopping(record_every=interval,grid_size=8)


class SavedReportTests(unittest.TestCase):
    def test_new_csv_flags_roundtrip_and_old_records_remain_compatible(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            old = dict(GOOD,evaluations=10,iteration=10,delta=.001,search_ok=True,
                       below_threshold=True,consecutive_passes=1)
            diagnostic = dict(old,evaluations=12,iteration=12,stopping_check=False)
            bo_main.save_regret_history(directory,[old,diagnostic])
            data = plot_regret.read_checks(directory)
            self.assertEqual(data['stopping_check'].tolist(),[True,False])
            data.drop(columns=['stopping_check','valid']).to_csv(directory/'regret_history.csv',index=False)
            self.assertEqual(plot_regret.read_checks(directory)['stopping_check'].tolist(),[True,True])

    def test_wrong_parent_prefix_and_bare_run_name_resolve_to_script_results(self):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder)/'optimizer_py'
            run = script/'results'/'bo_example'
            run.mkdir(parents=True)
            (run/'regret_history.csv').touch()
            with patch.object(plot_regret,'SCRIPT_DIR',script),chdir(script):
                self.assertEqual(plot_regret.resolve_run_dir(Path('../results/bo_example')),run)
                self.assertEqual(plot_regret.resolve_run_dir(Path('bo_example')),run)
                self.assertEqual(plot_regret.resolve_run_dir(run),run)

    def test_missing_report_has_helpful_error_and_absolute_paths_are_not_redirected(self):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder)/'optimizer_py'
            run = script/'results'/'bo_example'
            run.mkdir(parents=True)
            (run/'regret_history.csv').touch()
            with patch.object(plot_regret,'SCRIPT_DIR',script):
                with self.assertRaisesRegex(FileNotFoundError,'original regret history cannot be recovered'):
                    plot_regret.resolve_run_dir(Path(folder)/'wrong'/'bo_example')

    def test_ambiguous_fallback_requires_explicit_path(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            script = base/'optimizer_py'
            elsewhere = base/'elsewhere'
            for parent in (script,elsewhere):
                run = parent/'results'/'bo_example'
                run.mkdir(parents=True)
                (run/'regret_history.csv').touch()
            with patch.object(plot_regret,'SCRIPT_DIR',script),chdir(elsewhere):
                with self.assertRaisesRegex(ValueError,'Several matching runs'):
                    plot_regret.resolve_run_dir(Path('bo_example'))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
