"""Tests for the :class:`SessionData` intermediate container."""

import unittest

from camstim_behavior_processing import SessionData

from . import fixtures as F


class SessionDataTest(unittest.TestCase):
    """Tests for :class:`SessionData` and its derived views."""

    def setUp(self):
        """Build a SessionData from the shared fixtures."""
        self.session = F.make_session()

    def test_fields_present(self):
        """The four materialized intermediates are stored verbatim."""
        self.assertIsInstance(self.session, SessionData)
        self.assertEqual(len(self.session.events_df), len(F.make_events_df()))
        self.assertIn("interval_type", self.session.intervals_df.columns)
        self.assertIn("stim_vsync_fall", self.session.timestamp_data)
        self.assertEqual(self.session.task_parameters.name, "task_parameters")

    def test_trials_df_is_trial_rows(self):
        """trials_df is exactly the interval_type=='trial' rows, re-indexed."""
        trials = self.session.trials_df
        self.assertEqual(len(trials), 2)
        self.assertTrue((trials["interval_type"] == "trial").all())
        self.assertEqual(list(trials.index), [0, 1])

    def test_is_frozen(self):
        """SessionData is immutable (frozen dataclass)."""
        with self.assertRaises(AttributeError):
            self.session.events_df = None


if __name__ == "__main__":
    unittest.main()
