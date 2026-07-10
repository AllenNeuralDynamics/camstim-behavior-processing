"""Tests for the events sidecar builder."""

import json
import unittest

from camstim_behavior_processing.nwb.sidecar import build_events_sidecar


class BuildEventsSidecarTest(unittest.TestCase):
    """Tests for :func:`build_events_sidecar`."""

    def setUp(self):
        """Build the sidecar once."""
        self.sidecar = build_events_sidecar()

    def test_json_serialisable(self):
        """The sidecar round-trips through JSON."""
        self.assertIsInstance(json.dumps(self.sidecar), str)

    def test_categorical_entry(self):
        """Categorical columns carry Description/Levels/HED."""
        entry = self.sidecar["event_type"]
        self.assertIn("Description", entry)
        self.assertIn("Levels", entry)
        self.assertIn("HED", entry)
        # Empty HED fragments (e.g. "n/a") are dropped.
        self.assertNotIn("n/a", entry.get("HED", {}))

    def test_interval_type_has_no_hed(self):
        """interval_type has Levels but no composed base HED block."""
        self.assertIn("Levels", self.sidecar["interval_type"])
        self.assertNotIn("HED", self.sidecar["interval_type"])

    def test_value_column_with_template(self):
        """A descriptive value column carries its HED # template."""
        self.assertEqual(self.sidecar["timestamp"]["HED"], "Time-value/# s")

    def test_value_column_without_template(self):
        """A value column with no HED template omits the HED key."""
        self.assertNotIn("HED", self.sidecar["initial_image_name"])

    def test_hed_defs_block(self):
        """The reserved hed_defs block is present."""
        self.assertIn("alldefs", self.sidecar["hed_defs"]["HED"])


if __name__ == "__main__":
    unittest.main()
