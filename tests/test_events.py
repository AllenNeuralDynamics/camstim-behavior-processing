"""Tests for the pynwb EventsTable builder."""

import unittest

from pynwb.event import EventsTable

from camstim_behavior_processing.nwb.events import (
    add_events,
    build_events_table,
)
from camstim_behavior_processing.nwb.file import build_nwbfile

from . import fixtures as F


class BuildEventsTableTest(unittest.TestCase):
    """Tests for :func:`build_events_table`."""

    def setUp(self):
        """Build the events table from the shared fixture once."""
        self.et, self.meanings = build_events_table(F.make_events_df())
        self.df = self.et.to_dataframe()

    def test_drops_interval_and_miss_events(self):
        """Only discrete events survive; onsets/offsets/miss are dropped."""
        kept = set(self.df["event_type"])
        self.assertEqual(
            kept,
            {"lick", "reward", "image_change", "image_omission"},
        )

    def test_is_events_table_with_four_meanings(self):
        """The result is an EventsTable backed by four MeaningsTables."""
        self.assertIsInstance(self.et, EventsTable)
        self.assertEqual(len(self.meanings), 4)

    def test_lick_categoricals(self):
        """Classified bout-start lick vs unclassified within-bout lick."""
        licks = self.df[self.df["event_type"] == "lick"].sort_values(
            "timestamp"
        )
        first, second = licks.iloc[0], licks.iloc[1]
        self.assertEqual(first["lick_classification"], "hit")
        self.assertEqual(first["lick_bouts"], "bout_start")
        self.assertEqual(second["lick_classification"], "n/a")
        self.assertEqual(second["lick_bouts"], "within_bout")

    def test_reward_categoricals(self):
        """Reward type is carried, or n/a when absent."""
        rewards = self.df[self.df["event_type"] == "reward"].sort_values(
            "timestamp"
        )
        self.assertEqual(rewards.iloc[0]["reward_type"], "earned")
        self.assertEqual(rewards.iloc[1]["reward_type"], "n/a")

    def test_missing_orientation_column(self):
        """A frame without an orientation column still builds."""
        et, _ = build_events_table(F.make_events_df(with_orientation=False))
        self.assertIn("orientation", et.colnames)


class AddEventsTest(unittest.TestCase):
    """Tests for :func:`add_events`."""

    def test_adds_to_nwb(self):
        """The events table is attached to the NWBFile."""
        nwb = build_nwbfile(F.make_pkl())
        add_events(nwb, F.make_events_df())
        self.assertEqual(nwb.get_events_table("events").name, "events")


if __name__ == "__main__":
    unittest.main()
