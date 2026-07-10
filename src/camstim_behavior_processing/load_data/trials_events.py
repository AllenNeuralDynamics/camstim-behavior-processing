"""Build the trials and events tables for a change-detection session.

Thin wrapper over the live Stage-1 extractor
(``build_events_and_intervals.build_all``). It produces the three pandas
tables that downstream packaging consumes:

- ``events_df``    — point events (licks, rewards, image/movie onsets &
  offsets, image_change, image_omission, miss), with HED strings and
  lick/reward classification columns.
- ``trials_df``    — one flat row per trial (go/catch/hit/miss/... booleans,
  change/response timing, image names). Extracted from ``intervals_df``.
- ``intervals_df`` — every interval type in one frame (epochs, trials, and
  per-trial change_window / response_window rows).

Depends only on numpy / pandas / h5py — no NWB, no scipy — so it runs in the
base environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._bootstrap import add_legacy_to_path


def build_trials_and_events(
    pkl_path: str | Path,
    sync_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the events, trials, and intervals tables for one session.

    Parameters
    ----------
    pkl_path : path to the camstim ``*_stim.pkl``.
    sync_path : path to the ``*_sync.h5`` sync file.
    output_dir : optional directory; if given, ``build_all`` also writes
        ``events_table.csv`` / ``trials_table.csv`` / ``intervals_table.csv``
        (plus pickles and the timestamp ``.npz``) there.

    Returns
    -------
    dict with keys:
        ``events_df``, ``trials_df``, ``intervals_df``, ``timestamp_data``,
        ``task_parameters``.
    """
    # Imported here (not at module load) so the legacy extractor directory is
    # only required when actually building trials/events — importing the
    # package to build just the wheel df does not need it.
    add_legacy_to_path()
    from build_events_and_intervals import build_all

    return build_all(str(pkl_path), str(sync_path), output_dir=output_dir)
