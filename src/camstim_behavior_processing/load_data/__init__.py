"""Intermediate-DataFrame builders for a change-detection session.

Pure data layer — no NWB packaging lives here. Composable building blocks turn
a raw camstim session (``*_stim.pkl`` + ``*_sync.h5``) into analysis-ready
pandas tables that the :mod:`camstim_behavior_processing.nwb` layer consumes:

- :func:`build_trials_and_events` (``trials_events``) — returns a
  :class:`SessionData` (events_df, intervals_df, timestamp_data,
  task_parameters, + the derived trials_df view). numpy / pandas / h5py only.
- :func:`compute_running_speed` (``running_speed``) — the wheel df. Adds scipy.
- :func:`load_stim_pkl` (``loaders``) — read the raw behavior pickle.

``compute_running_speed`` is imported lazily so that using only the
trials/events builder does not require scipy.
"""

from __future__ import annotations

from .loaders import load_stim_pkl
from .session_data import SessionData
from .sweepstim import (
    SweepStimData,
    build_sweepstim_session,
    classify_sweepstim_session,
)
from .trials_events import build_trials_and_events

__all__ = [
    "build_trials_and_events",
    "SessionData",
    "compute_running_speed",
    "load_stim_pkl",
    "SweepStimData",
    "build_sweepstim_session",
    "classify_sweepstim_session",
]


def __getattr__(name):
    """Lazily re-export the scipy-dependent helper on first access."""
    if name == "compute_running_speed":
        from . import running_speed

        return getattr(running_speed, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
