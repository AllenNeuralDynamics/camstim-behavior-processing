"""Tests for the NWBFile / Subject builders."""

import datetime
import unittest

from camstim_behavior_processing.nwb.file import build_nwbfile, build_subject

from . import fixtures as F


class BuildSubjectTest(unittest.TestCase):
    """Tests for :func:`build_subject`."""

    def test_defaults_from_pkl(self):
        """subject_id comes from the pkl; species/sex default."""
        subj = build_subject(F.make_pkl(), {})
        self.assertEqual(subj.subject_id, "123456")
        self.assertEqual(subj.species, "Mus musculus")
        self.assertEqual(subj.sex, "U")

    def test_dob_iso_string(self):
        """A valid ISO date string becomes a tz-aware datetime."""
        subj = build_subject(F.make_pkl(), {"date_of_birth": "2023-01-15"})
        self.assertEqual(subj.date_of_birth.year, 2023)
        self.assertIsNotNone(subj.date_of_birth.tzinfo)

    def test_dob_invalid_string(self):
        """An unparseable date string yields no date_of_birth."""
        subj = build_subject(F.make_pkl(), {"date_of_birth": "nonsense"})
        self.assertIsNone(subj.date_of_birth)

    def test_dob_datetime_passthrough(self):
        """A datetime date_of_birth is passed through as given."""
        dob = datetime.datetime(2022, 5, 1, tzinfo=datetime.timezone.utc)
        subj = build_subject(F.make_pkl(), {"date_of_birth": dob})
        self.assertEqual(subj.date_of_birth, dob)

    def test_metadata_overrides(self):
        """Metadata overrides species/sex/genotype/strain."""
        subj = build_subject(
            F.make_pkl(),
            {"species": "Mus", "sex": "M", "genotype": "wt", "strain": "s"},
        )
        self.assertEqual(subj.sex, "M")
        self.assertEqual(subj.genotype, "wt")


class BuildNwbfileTest(unittest.TestCase):
    """Tests for :func:`build_nwbfile`."""

    def test_from_pkl(self):
        """Identity/subject/start-time are derived from the pkl."""
        nwb = build_nwbfile(F.make_pkl())
        self.assertEqual(nwb.subject.subject_id, "123456")
        self.assertEqual(nwb.session_description, "OPHYS_1_images_A")
        self.assertIsNotNone(nwb.session_start_time.tzinfo)

    def test_metadata_overrides(self):
        """Metadata overrides identity fields."""
        nwb = build_nwbfile(
            F.make_pkl(),
            {
                "session_description": "custom",
                "identifier": "id-1",
                "institution": "AIND",
            },
        )
        self.assertEqual(nwb.session_description, "custom")
        self.assertEqual(nwb.identifier, "id-1")
        self.assertEqual(nwb.institution, "AIND")

    def test_missing_start_time_defaults_to_now(self):
        """A pkl without start_time gets a tz-aware now()."""
        pkl = F.make_pkl()
        del pkl["start_time"]
        nwb = build_nwbfile(pkl)
        self.assertIsNotNone(nwb.session_start_time.tzinfo)


if __name__ == "__main__":
    unittest.main()
