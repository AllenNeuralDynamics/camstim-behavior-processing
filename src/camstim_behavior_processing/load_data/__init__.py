"""Intermediate-DataFrame builders for a change-detection session.

Pure data layer — no NWB packaging lives here. Three composable building
blocks turn a raw camstim session (``*_stim.pkl`` + ``*_sync.h5``) into
analysis-ready pandas tables that the :mod:`camstim_behavior_processing.nwb`
layer consumes:

- :func:`build_trials_and_events` (``trials_events``) — events_df, trials_df,
  intervals_df. numpy / pandas / h5py only.
- :func:`compute_running_speed` (``running_speed``) — the wheel df. Adds scipy.
- :func:`load_stim_pkl` (``loaders``) — read the raw behavior pickle.

``compute_running_speed`` is imported lazily so that using only the
trials/events builder does not require scipy.
"""

from __future__ import annotations

from .loaders import load_stim_pkl
from .trials_events import build_trials_and_events

__all__ = [
    "build_trials_and_events",
    "compute_running_speed",
    "load_stim_pkl",
]


def __getattr__(name):
    """Lazily re-export the scipy-dependent helper on first access."""
    if name == "compute_running_speed":
        from . import running_speed

        return getattr(running_speed, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
