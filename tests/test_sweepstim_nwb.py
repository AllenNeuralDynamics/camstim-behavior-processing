"""Tests for the passive SweepStim NWB-packaging layer."""

import datetime
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from camstim_behavior_processing.nwb import sweepstim as pkg
from camstim_behavior_processing.nwb.sweepstim import (
    assemble_sweepstim_nwbfile,
    build_intervals_table_sweepstim,
    build_stimulus_presentations_sweepstim,
    build_sweepstim_nwbfile,
    build_sweepstim_sidecar,
    package_sweepstim_nwb,
)

from . import fixtures as F


class DatetimeHelpersTest(unittest.TestCase):
    """Tests for :func:`_to_datetime` and :func:`_normalize_dob`."""

    def test_to_datetime_aware(self):
        """An already tz-aware datetime is returned unchanged."""
        dt = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        self.assertEqual(pkg._to_datetime(dt), dt)

    def test_to_datetime_naive(self):
        """A naive datetime gains a UTC tzinfo."""
        dt = datetime.datetime(2024, 1, 1)
        self.assertIsNotNone(pkg._to_datetime(dt).tzinfo)

    def test_to_datetime_isoformat(self):
        """An ISO-8601 string is parsed."""
        out = pkg._to_datetime("2024-01-02T03:04:05")
        self.assertEqual(out.year, 2024)

    def test_to_datetime_epoch(self):
        """A Unix epoch number is parsed as UTC."""
        out = pkg._to_datetime(1_700_000_000.0)
        self.assertIsNotNone(out.tzinfo)

    def test_to_datetime_none(self):
        """An unrecognised value falls back to now()."""
        self.assertIsNotNone(pkg._to_datetime(None).tzinfo)

    def test_normalize_dob_variants(self):
        """dob strings/dates/datetimes normalize; bad strings -> None."""
        self.assertIsNone(pkg._normalize_dob("not-a-date"))
        self.assertIsNone(pkg._normalize_dob(None))
        out = pkg._normalize_dob("2024-05-06")
        self.assertEqual((out.year, out.month, out.day), (2024, 5, 6))
        as_date = pkg._normalize_dob(datetime.date(2023, 2, 3))
        self.assertIsInstance(as_date, datetime.datetime)
        aware = datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc)
        self.assertEqual(pkg._normalize_dob(aware), aware)


class SubjectIdTest(unittest.TestCase):
    """Tests for :func:`_subject_id_from_pkl`."""

    def test_behavior(self):
        """behavior.params.mouse_id wins."""
        pkl = {"items": {"behavior": {"params": {"mouse_id": 1}}}}
        self.assertEqual(pkg._subject_id_from_pkl(pkl), "1")

    def test_foraging(self):
        """foraging.params.mouse_id is used when behavior lacks it."""
        pkl = {"items": {"foraging": {"params": {"mouse_id": 2}}}}
        self.assertEqual(pkg._subject_id_from_pkl(pkl), "2")

    def test_top_level_mouseid(self):
        """Top-level mouseid is the next fallback."""
        self.assertEqual(pkg._subject_id_from_pkl({"mouseid": 3}), "3")

    def test_unknown(self):
        """A pkl with no id resolves to 'unknown'."""
        self.assertEqual(pkg._subject_id_from_pkl({}), "unknown")


class BuildNwbfileTest(unittest.TestCase):
    """Tests for :func:`build_sweepstim_nwbfile`."""

    def test_defaults_from_pkl(self):
        """Identity/subject default from the pkl (unknown sex, U)."""
        nwb = build_sweepstim_nwbfile(F.make_sweepstim_pkl())
        self.assertEqual(nwb.subject.subject_id, "999999")
        self.assertEqual(nwb.subject.sex, "U")

    def test_metadata_overrides(self):
        """Metadata overrides identity + subject fields."""
        nwb = build_sweepstim_nwbfile(
            F.make_sweepstim_pkl(),
            metadata={
                "session_description": "passive",
                "identifier": "id-1",
                "institution": "AIND",
                "sex": "F",
                "genotype": "wt",
                "age": "P90D",
                "date_of_birth": "2024-01-01",
            },
        )
        self.assertEqual(nwb.identifier, "id-1")
        self.assertEqual(nwb.subject.sex, "F")
        self.assertEqual(nwb.subject.genotype, "wt")


class BuildTablesTest(unittest.TestCase):
    """Tests for the SweepStim TimeIntervals builders + sidecar."""

    def test_stimulus_presentations(self):
        """Every presentation row gets a movie HED string."""
        tbl = build_stimulus_presentations_sweepstim(
            F.make_sweepstim_presentations_df()
        )
        self.assertEqual(len(tbl), 3)
        self.assertIn("Movie, Label/natural_movie_one", tbl["HED"][0])

    def test_intervals_table_flat(self):
        """Flat intervals mix epoch + stimulus_presentation rows, sorted."""
        tbl = build_intervals_table_sweepstim(
            F.make_sweepstim_epochs_df(),
            F.make_sweepstim_presentations_df(),
        )
        types = set(tbl["interval_type"][:])
        self.assertEqual(types, {"epoch", "stimulus_presentation"})
        starts = list(tbl["start_time"][:])
        self.assertEqual(starts, sorted(starts))
        # Epoch rows carry a -1 stimulus_presentations_id.
        epoch_fk = [
            fk
            for fk, t in zip(
                tbl["stimulus_presentations_id"][:], tbl["interval_type"][:]
            )
            if t == "epoch"
        ]
        self.assertTrue(all(fk == -1 for fk in epoch_fk))

    def test_intervals_epoch_hed(self):
        """Spontaneous and clip epochs get their respective HED strings."""
        tbl = build_intervals_table_sweepstim(
            F.make_sweepstim_epochs_df(),
            F.make_sweepstim_presentations_df(),
        )
        hed_by_label = dict(zip(tbl["label"][:], tbl["HED"][:]))
        self.assertIn("Label/spontaneous", hed_by_label["spontaneous"])

    def test_sidecar_keys(self):
        """The sidecar carries the passive-session columns + hed_defs."""
        sc = build_sweepstim_sidecar()
        for key in ("start_time", "movie_name", "interval_type", "hed_defs"):
            self.assertIn(key, sc)
        self.assertIn("epoch", sc["interval_type"]["Levels"])


class AssembleTest(unittest.TestCase):
    """Tests for :func:`assemble_sweepstim_nwbfile`."""

    def test_all_containers(self):
        """Assembled NWB has both interval tables, running, and acquisition."""
        nwb = assemble_sweepstim_nwbfile(
            F.make_sweepstim_pkl(),
            F.make_sweepstim_data(),
            F.make_wheel_df(n=30, span=0.5),
            metadata={"institution": "AIND"},
        )
        self.assertIn("stimulus_presentations", nwb.intervals)
        self.assertIn("intervals", nwb.intervals)
        self.assertIn("running", nwb.processing)
        self.assertIn("v_sig", nwb.acquisition)


class WritersTest(unittest.TestCase):
    """Tests for the SweepStim NWB / sidecar writers."""

    def _assembled(self):
        """Assemble an NWB from the SweepStim fixtures."""
        return assemble_sweepstim_nwbfile(
            F.make_sweepstim_pkl(),
            F.make_sweepstim_data(),
            F.make_wheel_df(n=30, span=0.5),
        )

    def test_write_zarr_and_hdf5(self):
        """Both output formats write without error."""
        with TemporaryDirectory() as d:
            pkg._write_nwb(
                self._assembled(), Path(d) / "behavior.nwb.zarr", "zarr"
            )
            pkg._write_nwb(self._assembled(), Path(d) / "behavior.nwb", "hdf5")
            self.assertTrue((Path(d) / "behavior.nwb.zarr").exists())
            self.assertTrue((Path(d) / "behavior.nwb").exists())

    def test_unknown_format(self):
        """An unknown format raises ValueError."""
        with self.assertRaises(ValueError):
            pkg._write_nwb(self._assembled(), Path("x"), "parquet")

    def test_sidecar_naming(self):
        """The sidecar stem is the output name up to the first dot."""
        with TemporaryDirectory() as d:
            path = pkg._write_sidecar(Path(d) / "behavior.nwb.zarr")
            self.assertEqual(path.name, "behavior.events.json")
            self.assertTrue(path.exists())


class PackageTest(unittest.TestCase):
    """Tests for :func:`package_sweepstim_nwb` (loaders mocked)."""

    def _patches(self):
        """Patch the raw-data loaders used inside package_sweepstim_nwb."""
        return (
            mock.patch.object(
                pkg, "load_stim_pkl", return_value=F.make_sweepstim_pkl()
            ),
            mock.patch.object(
                pkg,
                "build_sweepstim_session",
                return_value=F.make_sweepstim_data(),
            ),
            mock.patch.object(
                pkg,
                "compute_running_speed",
                return_value=F.make_wheel_df(n=30, span=0.5),
            ),
        )

    def test_not_sweepstim_raises(self):
        """A non-sweepstim pkl raises ValueError."""
        with mock.patch.object(pkg, "load_stim_pkl", return_value={}):
            with self.assertRaises(ValueError):
                package_sweepstim_nwb("a.pkl", "a_sync.h5")

    def test_returns_without_writing(self):
        """With no output_path the NWB is returned and nothing written."""
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            nwb = package_sweepstim_nwb("a.pkl", "a_sync.h5")
        self.assertIn("stimulus_presentations", nwb.intervals)

    def test_writes_zarr_and_sidecar(self):
        """A zarr output plus its sidecar are written by default."""
        p1, p2, p3 = self._patches()
        with TemporaryDirectory() as d, p1, p2, p3:
            out = Path(d) / "behavior.nwb.zarr"
            package_sweepstim_nwb("a.pkl", "a_sync.h5", output_path=out)
            self.assertTrue(out.exists())
            self.assertTrue((Path(d) / "behavior.events.json").exists())

    def test_hdf5_without_sidecar(self):
        """fmt='hdf5' and write_sidecar=False write only the NWB."""
        p1, p2, p3 = self._patches()
        with TemporaryDirectory() as d, p1, p2, p3:
            out = Path(d) / "behavior.nwb"
            package_sweepstim_nwb(
                "a.pkl",
                "a_sync.h5",
                output_path=out,
                fmt="hdf5",
                write_sidecar=False,
            )
            self.assertTrue(out.exists())
            self.assertFalse((Path(d) / "behavior.events.json").exists())


if __name__ == "__main__":
    unittest.main()
