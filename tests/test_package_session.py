"""Tests for the type-agnostic :func:`package_session` dispatcher."""

import unittest
from unittest import mock

from camstim_behavior_processing import nwb as nwb_pkg
from camstim_behavior_processing.nwb import package_session

from . import fixtures as F

_SENTINEL_SWEEP = object()
_SENTINEL_CD = object()


class PackageSessionTest(unittest.TestCase):
    """Routing tests with the two packagers mocked out."""

    def _patches(self, pkl):
        """Patch the loader + both packagers on the nwb package."""
        return (
            mock.patch.object(nwb_pkg, "load_stim_pkl", return_value=pkl),
            mock.patch.object(
                nwb_pkg,
                "package_sweepstim_nwb",
                return_value=_SENTINEL_SWEEP,
            ),
            mock.patch.object(
                nwb_pkg, "package_nwb", return_value=_SENTINEL_CD
            ),
        )

    def test_routes_sweepstim(self):
        """A passive pkl dispatches to package_sweepstim_nwb."""
        p_load, p_sweep, p_cd = self._patches(F.make_sweepstim_pkl())
        with p_load, p_sweep as m_sweep, p_cd as m_cd:
            out = package_session("a.pkl", "a_sync.h5")
        self.assertIs(out, _SENTINEL_SWEEP)
        m_sweep.assert_called_once()
        m_cd.assert_not_called()

    def test_routes_change_detection(self):
        """A behavior pkl dispatches to package_nwb."""
        p_load, p_sweep, p_cd = self._patches(F.make_pkl())
        with p_load, p_sweep as m_sweep, p_cd as m_cd:
            out = package_session("a.pkl", "a_sync.h5")
        self.assertIs(out, _SENTINEL_CD)
        m_cd.assert_called_once()
        m_sweep.assert_not_called()

    def test_forwards_kwargs(self):
        """Every keyword argument is forwarded unchanged to the packager."""
        p_load, p_sweep, p_cd = self._patches(F.make_sweepstim_pkl())
        with p_load, p_sweep as m_sweep, p_cd:
            package_session(
                "a.pkl",
                "a_sync.h5",
                output_path="out.nwb.zarr",
                metadata={"institution": "AIND"},
                fmt="hdf5",
                write_sidecar=False,
            )
        m_sweep.assert_called_once_with(
            "a.pkl",
            "a_sync.h5",
            output_path="out.nwb.zarr",
            metadata={"institution": "AIND"},
            fmt="hdf5",
            write_sidecar=False,
        )


if __name__ == "__main__":
    unittest.main()
