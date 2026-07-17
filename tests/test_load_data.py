"""Tests for the load_data helpers (loaders, bootstrap, trials_events)."""

import pickle
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from camstim_behavior_processing.load_data import _bootstrap, trials_events
from camstim_behavior_processing.load_data.loaders import load_stim_pkl


class LoadStimPklTest(unittest.TestCase):
    """Tests for :func:`load_stim_pkl`."""

    def test_round_trip(self):
        """A pickled dict is read back intact."""
        with TemporaryDirectory() as d:
            p = Path(d) / "session_stim.pkl"
            with open(p, "wb") as f:
                pickle.dump({"items": {"behavior": {}}}, f)
            self.assertIn("items", load_stim_pkl(p))


class BootstrapTest(unittest.TestCase):
    """Tests for :func:`add_legacy_to_path`."""

    def test_idempotent(self):
        """The legacy dir is added once and re-adding is a no-op."""
        p = _bootstrap.add_legacy_to_path()
        count = sys.path.count(str(p))
        # A second call must not add a duplicate entry.
        _bootstrap.add_legacy_to_path()
        self.assertEqual(sys.path.count(str(p)), count)

    def test_legacy_extractor_module_exists(self):
        """Resolved legacy dir includes ``build_events_and_intervals.py``."""
        p = _bootstrap.add_legacy_to_path()
        self.assertTrue((p / "build_events_and_intervals.py").is_file())


class BuildTrialsAndEventsTest(unittest.TestCase):
    """Tests for :func:`build_trials_and_events`."""

    def setUp(self):
        """Inject a fake legacy extractor module for the duration."""
        mod = types.ModuleType("build_events_and_intervals")
        self.calls = []

        def build_all(pkl, sync, output_dir=None):
            """Record the call and return a sentinel result dict."""
            self.calls.append((pkl, sync, output_dir))
            return {
                "events_df": "E",
                "intervals_df": "I",
                "timestamp_data": {"stim_vsync_fall": "T"},
                "task_parameters": "P",
            }

        mod.build_all = build_all
        sys.modules["build_events_and_intervals"] = mod

    def tearDown(self):
        """Remove the injected module."""
        sys.modules.pop("build_events_and_intervals", None)

    def test_delegates_to_build_all(self):
        """The wrapper forwards paths (as str) and wraps build_all()."""
        out = trials_events.build_trials_and_events(
            "a_stim.pkl", "a_sync.h5", output_dir="out"
        )
        self.assertIsInstance(out, trials_events.SessionData)
        self.assertEqual(out.events_df, "E")
        self.assertEqual(out.intervals_df, "I")
        self.assertEqual(out.task_parameters, "P")
        self.assertEqual(self.calls, [("a_stim.pkl", "a_sync.h5", "out")])

    def test_missing_extractor_raises_helpful_error(self):
        """A missing extractor is re-raised with a reinstall hint."""
        # A ``None`` entry in sys.modules makes the import raise
        # ModuleNotFoundError(name="build_events_and_intervals").
        with mock.patch.dict(
            sys.modules, {"build_events_and_intervals": None}
        ):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                trials_events.build_trials_and_events("a.pkl", "b.h5")
        self.assertIn("data_loading_and_formatting_code", str(ctx.exception))

    def test_unrelated_import_error_propagates(self):
        """A missing sibling dependency is re-raised unchanged."""
        sys.modules.pop("build_events_and_intervals", None)

        def fake_import(name, *args, **kwargs):
            """Simulate the extractor failing on a missing sibling."""
            raise ModuleNotFoundError(name="some_other_dep")

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                trials_events.build_trials_and_events("a.pkl", "b.h5")
        self.assertEqual(ctx.exception.name, "some_other_dep")


if __name__ == "__main__":
    unittest.main()
