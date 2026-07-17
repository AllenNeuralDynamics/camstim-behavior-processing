"""Tests for the stimulus-presentation builders."""

import unittest

import numpy as np
import pandas as pd

from camstim_behavior_processing.nwb.epochs import build_epoch_lookup
from camstim_behavior_processing.nwb.file import build_nwbfile
from camstim_behavior_processing.nwb.stimulus import (
    _lick_latency,
    add_stimulus_presentations,
    build_natural_movie_one_presentations,
    build_stimulus_presentations,
)

from . import fixtures as F


class StimulusPresentationsTest(unittest.TestCase):
    """Tests for :func:`build_stimulus_presentations`."""

    def test_rows_and_omitted(self):
        """Three flashes + one omitted slot, sorted by onset."""
        ti = build_stimulus_presentations(F.make_events_df())
        df = ti.to_dataframe().sort_values("start_time")
        self.assertEqual(len(df), 4)
        self.assertTrue(df["omitted"].any())
        self.assertTrue(df["start_time"].is_monotonic_increasing)

    def test_default_epoch_without_list(self):
        """Without an epoch list every row is tagged change_detection."""
        ti = build_stimulus_presentations(F.make_events_df())
        self.assertTrue(
            (ti.to_dataframe()["epoch_name"] == "change_detection").all()
        )

    def test_epoch_tagging_with_list(self):
        """With an epoch list rows are tagged by containing epoch."""
        events = F.make_events_df()
        epochs = build_epoch_lookup(F.make_intervals_df(), events)
        ti = build_stimulus_presentations(events, epochs)
        self.assertIn("change_detection", set(ti.to_dataframe()["epoch_name"]))

    def test_missing_offset_uses_default_duration(self):
        """An onset with no offset gets the synthetic 0.25 s duration."""
        ti = build_stimulus_presentations(F.make_events_df())
        df = ti.to_dataframe()
        row = df[df["stimulus_presentations_id"] == 1].iloc[0]
        self.assertAlmostEqual(row["stop_time"] - row["start_time"], 0.25)


class LickLatencyTest(unittest.TestCase):
    """Tests for :func:`_lick_latency`."""

    def test_latency_and_nan(self):
        """A lick before the next flash yields a latency; else NaN."""
        events = pd.DataFrame(
            {
                "event_type": ["lick"],
                "timestamp": [1.1],
            }
        )
        out = _lick_latency([1.0, 2.0], events)
        self.assertAlmostEqual(out[0], 0.1)
        self.assertTrue(np.isnan(out[1]))


class MoviePresentationsTest(unittest.TestCase):
    """Tests for :func:`build_natural_movie_one_presentations`."""

    def test_none_when_no_movie(self):
        """Sessions with no movie_onset events return None."""
        events = F.make_events_df()
        events = events[~events["event_type"].str.startswith("movie")]
        self.assertIsNone(build_natural_movie_one_presentations(events))

    def test_built_when_present(self):
        """Movie frames produce a two-row table."""
        ti = build_natural_movie_one_presentations(F.make_events_df())
        self.assertEqual(len(ti), 2)


class AddStimulusPresentationsTest(unittest.TestCase):
    """Tests for :func:`add_stimulus_presentations`."""

    def test_adds_both_tables(self):
        """Both the stim and movie tables land in nwb.intervals."""
        nwb = build_nwbfile(F.make_pkl())
        add_stimulus_presentations(nwb, F.make_events_df())
        self.assertIn("stimulus_presentations", nwb.intervals)
        self.assertIn("natural_movie_one_presentations", nwb.intervals)

    def test_no_movie_table_when_absent(self):
        """No movie table is added when the session has no movie."""
        nwb = build_nwbfile(F.make_pkl())
        events = F.make_events_df()
        events = events[~events["event_type"].str.startswith("movie")]
        add_stimulus_presentations(nwb, events)
        self.assertNotIn("natural_movie_one_presentations", nwb.intervals)


if __name__ == "__main__":
    unittest.main()
