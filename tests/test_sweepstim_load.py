"""Tests for the passive SweepStim load_data layer."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from camstim_behavior_processing.load_data import sweepstim as S

from . import fixtures as F


class ClassifyTest(unittest.TestCase):
    """Tests for :func:`classify_sweepstim_session`."""

    def test_behavior_session_rejected(self):
        """A non-empty behavior trial_log marks a non-sweepstim session."""
        pkl = {"items": {"behavior": {"trial_log": [{"x": 1}]}}}
        is_sweep, detail = S.classify_sweepstim_session(pkl)
        self.assertFalse(is_sweep)
        self.assertIn("behavior session", detail)

    def test_stimuli_list_accepted(self):
        """A top-level stimuli list marks a sweepstim session."""
        is_sweep, detail = S.classify_sweepstim_session({"stimuli": [{}]})
        self.assertTrue(is_sweep)
        self.assertIn("sweepstim-like", detail)

    def test_foraging_only_accepted(self):
        """An items.foraging payload alone marks a sweepstim session."""
        is_sweep, _ = S.classify_sweepstim_session({"items": {"foraging": {}}})
        self.assertTrue(is_sweep)

    def test_missing_signatures_rejected(self):
        """A pkl with neither signature is not a sweepstim session."""
        is_sweep, detail = S.classify_sweepstim_session({})
        self.assertFalse(is_sweep)
        self.assertIn("missing sweepstim", detail)


class ResolveFrameCountTest(unittest.TestCase):
    """Tests for :func:`resolve_frame_count`."""

    def test_behavior_intervalsms(self):
        """behavior.intervalsms takes priority (len + 1)."""
        pkl = {"items": {"behavior": {"intervalsms": [1, 2, 3]}}}
        self.assertEqual(S.resolve_frame_count(pkl), 4)

    def test_foraging_intervalsms(self):
        """foraging.intervalsms is used when behavior lacks it."""
        pkl = {"items": {"foraging": {"intervalsms": [1, 2]}}}
        self.assertEqual(S.resolve_frame_count(pkl), 3)

    def test_toplevel_intervalsms(self):
        """Top-level intervalsms is the third fallback."""
        self.assertEqual(S.resolve_frame_count({"intervalsms": [1]}), 2)

    def test_vsynccount(self):
        """vsynccount is used when no intervalsms is present."""
        self.assertEqual(S.resolve_frame_count({"vsynccount": 42}), 42)

    def test_total_frames(self):
        """total_frames is the last-resort fallback."""
        self.assertEqual(S.resolve_frame_count({"total_frames": 7}), 7)

    def test_missing_raises(self):
        """No frame-count source raises KeyError."""
        with self.assertRaises(KeyError):
            S.resolve_frame_count({})


class GetEdgesTest(unittest.TestCase):
    """Tests for :func:`_get_edges` and :func:`_resolve_bit_index`."""

    def setUp(self):
        """A short two-line sync buffer (bit0 vsync, bit1 photodiode)."""
        self.counters = np.arange(6)
        # bit0 pattern: 0,1,0,1,0,1 -> rising at 1,3,5 ; falling at 2,4
        self.bits = np.array([0, 1, 0, 1, 0, 1])
        self.labels = ["stim_vsync", "stim_photodiode"]

    def test_falling(self):
        """Falling edges are the 1->0 transitions."""
        edges = S._get_edges(
            self.bits,
            self.counters,
            self.labels,
            "stim_vsync",
            "falling",
            1.0,
        )
        self.assertEqual(edges.tolist(), [2.0, 4.0])

    def test_rising(self):
        """Rising edges are the 0->1 transitions."""
        edges = S._get_edges(
            self.bits,
            self.counters,
            self.labels,
            "stim_vsync",
            "rising",
            1.0,
        )
        self.assertEqual(edges.tolist(), [1.0, 3.0, 5.0])

    def test_both(self):
        """'both' returns every transition."""
        edges = S._get_edges(
            self.bits,
            self.counters,
            self.labels,
            "stim_vsync",
            "both",
            1.0,
        )
        self.assertEqual(edges.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_alias_lookup(self):
        """A known alias (vsync_stim) resolves to the labelled line."""
        idx = S._resolve_bit_index(["vsync_stim"], "stim_vsync")
        self.assertEqual(idx, 0)

    def test_missing_line_raises(self):
        """An unknown line raises ValueError."""
        with self.assertRaises(ValueError):
            S._resolve_bit_index(["something_else"], "stim_vsync")


class MonitorDelayTest(unittest.TestCase):
    """Tests for the photodiode monitor-delay estimator."""

    def test_too_few_regular_edges(self):
        """Fewer than 11 regular pulses yields no clean run (default)."""
        self.assertIsNone(S._clean_photodiode_edges(np.array([0.0, 1.0])))
        self.assertEqual(
            S._estimate_monitor_delay(np.array([0.0, 1.0]), np.array([0.0])),
            S._DEFAULT_MONITOR_DELAY,
        )

    def test_clean_removes_anomaly(self):
        """A sub-0.5 s anomaly edge is removed from the regular run."""
        edges = np.arange(0.0, 13.0, 1.0)  # 13 regular 1 Hz pulses
        edges = np.insert(edges, 5, 4.4)  # anomaly 0.4 s after edge 4
        clean = S._clean_photodiode_edges(edges)
        self.assertNotIn(4.4, clean.tolist())

    def test_empty_inputs_default(self):
        """Empty cleaned edges / transitions return the default."""
        self.assertEqual(
            S._delay_from_edges(np.array([]), np.array([1.0]), 0.03), 0.03
        )

    def test_mean_of_valid_delays(self):
        """More than 10 valid sub-70 ms lags average to the mean."""
        transitions = np.arange(0.0, 15.0, 1.0)
        clean = transitions + 0.03
        delay = S._delay_from_edges(clean, transitions, 0.0356)
        self.assertAlmostEqual(delay, 0.03, places=6)

    def test_median_in_range(self):
        """Few valid lags fall back to an in-range median."""
        transitions = np.arange(0.0, 5.0, 1.0)
        clean = transitions + 0.03
        delay = S._delay_from_edges(clean, transitions, 0.0356)
        self.assertAlmostEqual(delay, 0.03, places=6)

    def test_median_out_of_range_default(self):
        """An out-of-range median falls back to the default."""
        transitions = np.arange(0.0, 5.0, 1.0)
        clean = transitions - 0.1  # negative lags
        delay = S._delay_from_edges(clean, transitions, 0.0356)
        self.assertEqual(delay, 0.0356)

    def test_estimate_end_to_end(self):
        """A clean run of regular pulses yields a measured delay."""
        transitions = np.arange(0.0, 15.0, 1.0)
        edges = transitions + 0.03  # ~1 Hz regular pulses, 30 ms lag
        delay = S._estimate_monitor_delay(edges, transitions)
        self.assertAlmostEqual(delay, 0.03, places=6)


class SequenceHelpersTest(unittest.TestCase):
    """Tests for :func:`_as_sequence` and :func:`_clip_name`."""

    def test_as_sequence_variants(self):
        """None, ndarray, list/tuple, and scalar all normalize to a list."""
        self.assertEqual(S._as_sequence(None), [])
        self.assertEqual(S._as_sequence(np.array([1, 2])), [1, 2])
        self.assertEqual(S._as_sequence((1, 2)), [1, 2])
        self.assertEqual(S._as_sequence(5), [5])

    def test_clip_name_movie_path(self):
        """A Windows movie_path resolves to its extension-less basename."""
        self.assertEqual(
            S._clip_name({"movie_path": r"C:\a\clip.npy"}, 0), "clip"
        )

    def test_clip_name_stim_path_fallback(self):
        """stim_path is used when movie_path is absent."""
        self.assertEqual(
            S._clip_name({"stim_path": "/x/y/grating.stim"}, 0), "grating"
        )

    def test_clip_name_placeholder(self):
        """No path yields a zero-padded placeholder."""
        self.assertEqual(S._clip_name({}, 3), "stim_003")

    def test_clip_name_extension_only(self):
        """A dotfile-style name keeps the original basename."""
        self.assertEqual(S._clip_name({"movie_path": ".npy"}, 0), ".npy")


class BlockRowsTest(unittest.TestCase):
    """Tests for the per-block presentation row builders."""

    def test_frame_list_runs(self):
        """Constant runs >= 0 become presentations; -1 gaps are skipped."""
        ts = np.arange(12, dtype=float)
        stim = {"movie_path": "m.npy", "frame_list": [0, 0, -1, 1, 1, 1]}
        rows = S._block_rows_from_frame_list(stim, 0, ts)
        self.assertEqual([r["movie_frame_index"] for r in rows], [0, 1])
        self.assertEqual(rows[0]["movie_name"], "m")

    def test_frame_list_empty(self):
        """An empty frame_list yields no rows."""
        self.assertEqual(
            S._block_rows_from_frame_list(
                {"frame_list": []}, 0, np.arange(4.0)
            ),
            [],
        )

    def test_frame_list_out_of_range_and_last_index(self):
        """Runs past the timebase are skipped; a last-index run is padded."""
        ts = np.arange(4, dtype=float)  # only 4 frame times
        # values: run 0 at [0,2), gap, then a distinct value only at index 3,
        # plus a run that starts past the timebase (index 5).
        stim = {"frame_list": [0, 0, -1, 9, 7, 7]}
        rows = S._block_rows_from_frame_list(stim, 0, ts)
        # value 9 is a single-frame run at index 3 (== n_frames-1): padded.
        padded = [r for r in rows if r["movie_frame_index"] == 9][0]
        self.assertGreater(padded["stop_time"], padded["start_time"])
        # value 7 starts at index 4 (>= n_frames) -> skipped.
        self.assertNotIn(7, [r["movie_frame_index"] for r in rows])

    def test_sweep_frames_fallback(self):
        """Blocks without a frame_list fall back to sweep_frames."""
        ts = np.arange(12, dtype=float)
        stim = {
            "stim_path": "g.stim",
            "sweep_order": [0, None],
            "sweep_frames": [[0, 2], [4, 4]],
            "runs": 1,
        }
        rows = S._block_rows_from_sweep_frames(stim, 1, ts)
        # sweep_order None -> movie_frame_index -1; ef<=sf padded.
        self.assertEqual(rows[0]["movie_frame_index"], 0)
        self.assertEqual(rows[1]["movie_frame_index"], -1)
        self.assertEqual(rows[1]["stim_block"], 1)

    def test_sweep_frames_out_of_range(self):
        """A sweep starting past the timebase is skipped."""
        ts = np.arange(3, dtype=float)
        stim = {"sweep_order": [0], "sweep_frames": [[5, 6]]}
        self.assertEqual(S._block_rows_from_sweep_frames(stim, 0, ts), [])

    def test_sweep_frames_last_index_padded(self):
        """A sweep clamped to the last frame is padded to a frame duration."""
        ts = np.arange(5, dtype=float)  # last index is 4
        stim = {"sweep_order": [0], "sweep_frames": [[4, 4]]}
        rows = S._block_rows_from_sweep_frames(stim, 0, ts)
        self.assertGreater(rows[0]["stop_time"], rows[0]["start_time"])

    def test_iter_warns_on_missing_frame_list(self):
        """The fallback path logs a warning."""
        ts = np.arange(6, dtype=float)
        stim = {"sweep_order": [0], "sweep_frames": [[0, 2]]}
        with self.assertLogs(S.logger, level="WARNING"):
            rows = S._iter_sweep_rows([stim], ts)
        self.assertEqual(len(rows), 1)


class EpochBuildTest(unittest.TestCase):
    """Tests for the epoch-table builder."""

    def test_display_sequence_with_spontaneous(self):
        """display_sequence windows produce epochs padded with spontaneous."""
        ts = np.linspace(0.0, 1.0, 61)  # 60 fps, 1 s
        stim = {"movie_path": "clip.npy", "display_sequence": [[0.2, 0.5]]}
        with self.assertLogs(S.logger, level="WARNING"):
            pres = S.build_sweepstim_presentations([stim], ts)
        epochs = S.build_sweepstim_epochs([stim], pres, ts, 60.0)
        names = epochs["name"].tolist()
        self.assertIn("clip", names)
        self.assertIn("spontaneous", names)

    def test_bad_windows_skipped(self):
        """Malformed / zero-length windows are ignored."""
        ts = np.linspace(0.0, 1.0, 61)
        stim = {
            "movie_path": "c.npy",
            "display_sequence": [[0.5, 0.2], "bad", np.array([0.1, 0.4])],
        }
        epochs = S._raw_epochs_from_sequences(
            [stim], S._sec_to_time_fn(ts, 60.0)
        )
        # Only the valid ndarray window survives.
        self.assertEqual(len(epochs), 1)

    def test_fallback_epoch_from_presentations(self):
        """With no display_sequence, one passive_viewing epoch spans frames."""
        pres = F.make_sweepstim_presentations_df()
        epochs = S.build_sweepstim_epochs(
            [{"frame_list": [0]}], pres, np.linspace(0, 1, 30), 60.0
        )
        self.assertIn("passive_viewing", epochs["name"].tolist())

    def test_empty_when_no_epochs_no_presentations(self):
        """No sequences and no presentations yields an empty epoch table."""
        empty = S.build_sweepstim_presentations(
            [{"frame_list": []}], np.arange(4.0)
        )
        epochs = S.build_sweepstim_epochs(
            [{"frame_list": []}], empty, np.arange(4.0), 60.0
        )
        self.assertTrue(epochs.empty)


class StimuliAndFpsTest(unittest.TestCase):
    """Tests for :func:`_stimuli_list` and :func:`_resolve_fps`."""

    def test_stimuli_ndarray(self):
        """A numpy stimuli array is converted to a list."""
        out = S._stimuli_list({"stimuli": np.array([{"a": 1}], dtype=object)})
        self.assertIsInstance(out, list)

    def test_stimuli_empty_raises(self):
        """An empty / missing stimuli list raises ValueError."""
        with self.assertRaises(ValueError):
            S._stimuli_list({"stimuli": []})

    def test_fps_from_pkl(self):
        """The pkl-level fps wins."""
        self.assertEqual(S._resolve_fps({"fps": 30.0}, [{}]), 30.0)

    def test_fps_from_stimulus(self):
        """The first stimulus block supplies fps when the pkl lacks it."""
        self.assertEqual(S._resolve_fps({}, [{"fps": 15.0}]), 15.0)

    def test_fps_default(self):
        """fps defaults to 60 Hz."""
        self.assertEqual(S._resolve_fps({}, [{}]), 60.0)


class BuildSessionIntegrationTest(unittest.TestCase):
    """End-to-end tests over a synthetic pkl + sync pair."""

    def test_build_sweepstim_session(self):
        """A full session builds non-empty presentation/epoch tables."""
        with TemporaryDirectory() as d:
            pkl_path, sync_path = F.write_sweepstim_session_files(Path(d))
            import pickle

            with open(pkl_path, "rb") as f:
                pkl = pickle.load(f)
            sweep = S.build_sweepstim_session(pkl, sync_path)
        self.assertFalse(sweep.presentations_df.empty)
        self.assertIn("stim_vsync_fall", sweep.timestamp_data)
        self.assertEqual(
            sweep.timestamp_data["monitor_delay"], S._DEFAULT_MONITOR_DELAY
        )


if __name__ == "__main__":
    unittest.main()
