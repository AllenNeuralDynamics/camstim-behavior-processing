"""Package a raw camstim change-detection session into NWB.

Pipeline, in two composable layers:

1. :mod:`camstim_behavior_processing.load_data` builds the intermediate
   DataFrames from a raw session (``*_stim.pkl`` + ``*_sync.h5``):
   ``build_trials_and_events`` (events / trials / intervals) and
   ``compute_running_speed`` (the wheel df). Any of these can be built on
   their own.
2. :mod:`camstim_behavior_processing.nwb` writes those DataFrames into the NWB
   ``acquisition`` (wheel), ``events``, and ``intervals`` (trials) containers.
   ``package_nwb`` is the one-call raw-files -> NWB path.

NWB-dependent names are imported lazily so that building only the DataFrames
does not require pynwb.
"""

from __future__ import annotations

from .load_data import (
    SessionData,
    SweepStimData,
    build_sweepstim_session,
    build_trials_and_events,
    classify_sweepstim_session,
)

__all__ = [
    "build_trials_and_events",
    "SessionData",
    "SweepStimData",
    "build_sweepstim_session",
    "classify_sweepstim_session",
    "compute_running_speed",
    "package_nwb",
    "build_nwbfile",
    "package_sweepstim_nwb",
    "package_session",
]

# pynwb-dependent names re-exported lazily (see __getattr__).
_LAZY_NWB_NAMES = (
    "package_nwb",
    "build_nwbfile",
    "package_sweepstim_nwb",
    "package_session",
)


def __getattr__(name):
    """Lazily re-export the scipy/pynwb-dependent helpers on first access."""
    if name == "compute_running_speed":
        from .load_data import compute_running_speed

        return compute_running_speed
    if name in _LAZY_NWB_NAMES:
        from . import nwb

        return getattr(nwb, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
