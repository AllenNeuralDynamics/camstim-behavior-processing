"""Tests for the running-speed NWB writer."""

import unittest

from pynwb import ProcessingModule

from camstim_behavior_processing.nwb.acquisition import add_running_speed
from camstim_behavior_processing.nwb.file import build_nwbfile

from . import fixtures as F


class AddRunningSpeedTest(unittest.TestCase):
    """Tests for :func:`add_running_speed`."""

    def test_creates_running_module(self):
        """A fresh NWBFile gets a new running module + raw acquisitions."""
        nwb = build_nwbfile(F.make_pkl())
        add_running_speed(nwb, F.make_wheel_df())
        self.assertIn("speed", nwb.processing["running"].data_interfaces)
        self.assertIn("v_sig", nwb.acquisition)
        self.assertIn("v_in", nwb.acquisition)

    def test_reuses_existing_running_module(self):
        """An existing running module is reused rather than recreated."""
        nwb = build_nwbfile(F.make_pkl())
        mod = ProcessingModule(name="running", description="pre-existing")
        nwb.add_processing_module(mod)
        add_running_speed(nwb, F.make_wheel_df())
        self.assertIs(nwb.processing["running"], mod)
        self.assertIn("speed", mod.data_interfaces)


if __name__ == "__main__":
    unittest.main()
