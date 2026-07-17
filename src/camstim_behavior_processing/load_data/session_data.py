"""Typed container for the materialized intermediates of one session.

:class:`SessionData` is the single intermediate structure produced by the
``load_data`` layer and consumed by the
:mod:`camstim_behavior_processing.nwb` layer. It replaces the loose
``dict[str, Any]`` that ``build_trials_and_events`` used to return, giving the
pipeline a typed hand-off point.

It holds only pandas / plain-Python objects (no scipy, no pynwb, no NWB
containers), so it stays on the eager, lightweight import path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class SessionData:
    """Materialized intermediates for one change-detection session.

    Produced by
    :func:`camstim_behavior_processing.load_data.trials_events.build_trials_and_events`
    and consumed by :func:`camstim_behavior_processing.nwb.assemble_nwbfile`.

    Attributes
    ----------
    events_df : point-events frame (licks, rewards, image/movie onsets &
        offsets, image_change, image_omission, miss), with HED strings and
        lick/reward classification columns.
    intervals_df : every interval type in one flat frame (epochs, trials, and
        per-trial change_window / response_window rows), discriminated by the
        ``interval_type`` column.
    timestamp_data : sync-derived timing arrays; ``stim_vsync_fall`` (frame
        times) drives the running-speed builder.
    task_parameters : the ``ChangeDetectionTaskParameters`` lab-metadata
        object built by the legacy extractor.
    """

    events_df: pd.DataFrame
    intervals_df: pd.DataFrame
    timestamp_data: Mapping[str, Any]
    task_parameters: Any

    @property
    def trials_df(self) -> pd.DataFrame:
        """The trial rows of :attr:`intervals_df` (one flat row per trial).

        A convenience view: the ``interval_type == 'trial'`` subset of
        :attr:`intervals_df`, re-indexed from 0. This is the same subset the
        NWB trials-table writer materializes.
        """
        mask = self.intervals_df["interval_type"] == "trial"
        return self.intervals_df[mask].reset_index(drop=True)
