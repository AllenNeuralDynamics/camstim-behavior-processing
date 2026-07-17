"""Tests for the epoch-lookup builders."""

import unittest

import pandas as pd

from camstim_behavior_processing.nwb.epochs import (
    build_epoch_lookup,
    epoch_name_at,
)

from . import fixtures as F


class BuildEpochLookupTest(unittest.TestCase):
    """Tests for :func:`build_epoch_lookup`."""

    def test_full_session(self):
        """Warm-up folds into change_detection; spontaneous fills gaps."""
        epochs = build_epoch_lookup(F.make_intervals_df(), F.make_events_df())
        names = [e["name"] for e in epochs]
        self.assertEqual(names[0], "change_detection")
        # Change detection starts at the warm_up start (0.0), not 0.5.
        self.assertEqual(epochs[0]["start"], 0.0)
        self.assertIn("natural_movie_one", names)
        # A spontaneous gap is inserted between 9.0 and 10.0.
        self.assertIn("spontaneous", names)

    def test_leading_gap(self):
        """A named epoch not starting at 0 gets a leading spontaneous row."""
        iv = pd.DataFrame(
            [
                {
                    "start_time": 2.0,
                    "stop_time": 5.0,
                    "interval_type": "epoch",
                    "label": "change_detection",
                }
            ]
        )
        ev = pd.DataFrame({"timestamp": [4.0]})
        epochs = build_epoch_lookup(iv, ev)
        self.assertEqual(epochs[0]["name"], "spontaneous")
        self.assertEqual(epochs[0]["start"], 0.0)
        self.assertEqual(epochs[0]["stop"], 2.0)

    def test_no_named_epochs(self):
        """With no named epochs the whole session is spontaneous."""
        iv = pd.DataFrame(
            [
                {
                    "start_time": 0.0,
                    "stop_time": 0.0,
                    "interval_type": "trial",
                    "label": "",
                }
            ]
        )
        ev = pd.DataFrame({"timestamp": [3.0]})
        epochs = build_epoch_lookup(iv, ev)
        self.assertEqual(len(epochs), 1)
        self.assertEqual(epochs[0]["name"], "spontaneous")
        self.assertEqual(epochs[0]["stop"], 3.0)

    def test_change_detection_without_warmup(self):
        """change_detection start is kept when there is no warm_up row."""
        iv = pd.DataFrame(
            [
                {
                    "start_time": 1.0,
                    "stop_time": 5.0,
                    "interval_type": "epoch",
                    "label": "change_detection",
                }
            ]
        )
        ev = pd.DataFrame({"timestamp": [5.0]})
        epochs = build_epoch_lookup(iv, ev)
        cd = [e for e in epochs if e["name"] == "change_detection"][0]
        self.assertEqual(cd["start"], 1.0)


class EpochNameAtTest(unittest.TestCase):
    """Tests for :func:`epoch_name_at`."""

    def test_inside_and_outside(self):
        """A time inside an epoch returns its name; outside -> spontaneous."""
        epochs = [{"name": "change_detection", "start": 0.0, "stop": 5.0}]
        self.assertEqual(epoch_name_at(2.0, epochs), "change_detection")
        self.assertEqual(epoch_name_at(9.0, epochs), "spontaneous")


if __name__ == "__main__":
    unittest.main()
