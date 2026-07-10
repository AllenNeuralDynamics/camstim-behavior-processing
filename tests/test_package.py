"""Tests for the package orchestration and NWB writers."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from camstim_behavior_processing import nwb as pkg
from camstim_behavior_processing.nwb import (
    _write_nwb,
    _write_sidecar,
    assemble_nwbfile,
    package_nwb,
)

from . import fixtures as F


def _assembled():
    """Assemble an NWBFile from the shared fixtures."""
    return assemble_nwbfile(
        F.make_pkl(),
        F.make_events_df(),
        F.make_intervals_df(),
        F.make_wheel_df(),
        F.make_task_parameters(),
        metadata={"institution": "AIND"},
    )


class AssembleNwbfileTest(unittest.TestCase):
    """Tests for :func:`assemble_nwbfile`."""

    def setUp(self):
        """Assemble once and reuse across assertions."""
        self.nwb = _assembled()

    def test_contains_all_containers(self):
        """Trials, intervals, events, running, and acquisition are present."""
        self.assertEqual(len(self.nwb.trials), 2)
        self.assertIn("intervals", self.nwb.intervals)
        self.assertIn("stimulus_presentations", self.nwb.intervals)
        self.assertEqual(len(self.nwb.get_events_table("events")), 6)
        self.assertIn("running", self.nwb.processing)
        self.assertIn("v_sig", self.nwb.acquisition)

    def test_lab_metadata(self):
        """task_parameters and hed_schema lab metadata are attached."""
        self.assertIn("task_parameters", self.nwb.lab_meta_data)


class WritersTest(unittest.TestCase):
    """Tests for the NWB / sidecar writers."""

    def test_write_zarr_and_hdf5(self):
        """Both output formats write without error."""
        nwb_zarr = _assembled()
        nwb_hdf5 = _assembled()
        with TemporaryDirectory() as d:
            _write_nwb(nwb_zarr, Path(d) / "behavior.nwb.zarr", "zarr")
            _write_nwb(nwb_hdf5, Path(d) / "behavior.nwb", "hdf5")
            self.assertTrue((Path(d) / "behavior.nwb.zarr").exists())
            self.assertTrue((Path(d) / "behavior.nwb").exists())

    def test_write_unknown_format(self):
        """An unknown format raises ValueError."""
        with self.assertRaises(ValueError):
            _write_nwb(_assembled(), Path("x"), "parquet")

    def test_sidecar_naming(self):
        """The sidecar stem is the output name up to the first dot."""
        with TemporaryDirectory() as d:
            path = _write_sidecar(Path(d) / "behavior.nwb.zarr")
            self.assertEqual(path.name, "behavior.events.json")
            self.assertTrue(path.exists())


class PackageNwbTest(unittest.TestCase):
    """Tests for :func:`package_nwb` (with the loaders mocked)."""

    def _patches(self):
        """Patch the raw-data loaders used inside package_nwb."""
        built = {
            "events_df": F.make_events_df(),
            "intervals_df": F.make_intervals_df(),
            "timestamp_data": {"stim_vsync_fall": np.linspace(0, 11, 660)},
            "task_parameters": F.make_task_parameters(),
        }
        return (
            mock.patch.object(
                pkg, "build_trials_and_events", return_value=built
            ),
            mock.patch.object(pkg, "load_stim_pkl", return_value=F.make_pkl()),
            mock.patch.object(
                pkg, "compute_running_speed", return_value=F.make_wheel_df()
            ),
        )

    def test_returns_without_writing(self):
        """With no output_path the NWB is returned and nothing is written."""
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            nwb = package_nwb("a_stim.pkl", "a_sync.h5")
        self.assertEqual(len(nwb.trials), 2)

    def test_writes_zarr_and_sidecar(self):
        """A zarr output plus its sidecar are written by default."""
        p1, p2, p3 = self._patches()
        with TemporaryDirectory() as d, p1, p2, p3:
            out = Path(d) / "behavior.nwb.zarr"
            package_nwb("a_stim.pkl", "a_sync.h5", output_path=out)
            self.assertTrue(out.exists())
            self.assertTrue((Path(d) / "behavior.events.json").exists())

    def test_hdf5_without_sidecar(self):
        """fmt='hdf5' and write_sidecar=False write only the NWB."""
        p1, p2, p3 = self._patches()
        with TemporaryDirectory() as d, p1, p2, p3:
            out = Path(d) / "behavior.nwb"
            package_nwb(
                "a_stim.pkl",
                "a_sync.h5",
                output_path=out,
                fmt="hdf5",
                write_sidecar=False,
            )
            self.assertTrue(out.exists())
            self.assertFalse((Path(d) / "behavior.events.json").exists())


if __name__ == "__main__":
    unittest.main()
