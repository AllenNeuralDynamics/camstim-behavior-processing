"""Tests for the trials table and flat intervals table."""

import unittest

import numpy as np

from camstim_behavior_processing.nwb.epochs import build_epoch_lookup
from camstim_behavior_processing.nwb.file import build_nwbfile
from camstim_behavior_processing.nwb.intervals import (
    _f,
    _i,
    _s,
    add_intervals,
    add_trials,
    build_intervals_table,
)

from . import fixtures as F


class CoerceHelpersTest(unittest.TestCase):
    """Tests for the null-coalescing helpers."""

    def test_present_and_missing(self):
        """Present values coerce; null values fall back to the default."""
        self.assertEqual(_f(2.5), 2.5)
        self.assertTrue(np.isnan(_f(np.nan)))
        self.assertEqual(_i(3.0), 3)
        self.assertEqual(_i(np.nan), -1)
        self.assertEqual(_s("x"), "x")
        self.assertEqual(_s(np.nan), "")


class AddTrialsTest(unittest.TestCase):
    """Tests for :func:`add_trials`."""

    def _trials_df(self, warm_up_n=1, with_epochs=True):
        """Build an NWBFile, add trials, return the trials dataframe."""
        iv = F.make_intervals_df()
        ev = F.make_events_df()
        nwb = build_nwbfile(F.make_pkl())
        epochs = build_epoch_lookup(iv, ev) if with_epochs else None
        add_trials(nwb, iv, warm_up_n, epochs)
        return nwb.trials.to_dataframe()

    def test_two_trials_with_windows(self):
        """Both trials are added; windows populate on the first trial."""
        df = self._trials_df()
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df.iloc[0]["change_window_start_time"], 1.9)
        self.assertAlmostEqual(df.iloc[0]["response_window_stop_time"], 2.75)

    def test_missing_windows_are_nan(self):
        """The catch trial with no windows gets NaN window times."""
        df = self._trials_df()
        self.assertTrue(np.isnan(df.iloc[1]["change_window_start_time"]))

    def test_warm_up_flag(self):
        """The first warm_up_n trials are flagged warm_up."""
        df = self._trials_df(warm_up_n=1)
        self.assertTrue(bool(df.iloc[0]["warm_up"]))
        self.assertFalse(bool(df.iloc[1]["warm_up"]))

    def test_epoch_default_without_list(self):
        """Without an epoch list, trials default to change_detection."""
        df = self._trials_df(with_epochs=False)
        self.assertTrue((df["epoch_name"] == "change_detection").all())


class BuildIntervalsTableTest(unittest.TestCase):
    """Tests for :func:`build_intervals_table`."""

    def setUp(self):
        """Build the flat intervals table from the shared fixtures."""
        self.df = build_intervals_table(
            F.make_intervals_df(), F.make_events_df()
        ).to_dataframe()

    def test_has_every_interval_type(self):
        """Every interval type is represented in the flat table."""
        self.assertEqual(
            set(self.df["interval_type"]),
            {
                "epoch",
                "trial",
                "change_window",
                "response_window",
                "stimulus_presentation",
                "movie_frame",
            },
        )

    def test_sorted_and_foreign_keys(self):
        """Rows are start-sorted and carry the right foreign keys."""
        self.assertTrue(self.df["start_time"].is_monotonic_increasing)
        stim = self.df[self.df["interval_type"] == "stimulus_presentation"]
        self.assertTrue((stim["stimulus_presentations_id"] >= 0).all())
        movie = self.df[self.df["interval_type"] == "movie_frame"]
        self.assertTrue(
            (movie["natural_movie_one_presentations_id"] >= 0).all()
        )

    def test_window_hed_from_hed_string(self):
        """change/response window rows carry their hed_string as HED."""
        cw = self.df[self.df["interval_type"] == "change_window"].iloc[0]
        self.assertEqual(cw["HED"], "Def/change_win")


class AddIntervalsTest(unittest.TestCase):
    """Tests for :func:`add_intervals`."""

    def test_adds_flat_table(self):
        """The flat intervals table lands in nwb.intervals."""
        nwb = build_nwbfile(F.make_pkl())
        add_intervals(nwb, F.make_intervals_df(), F.make_events_df())
        self.assertIn("intervals", nwb.intervals)


if __name__ == "__main__":
    unittest.main()
