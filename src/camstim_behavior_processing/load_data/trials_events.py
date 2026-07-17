"""Build the events and intervals tables for a change-detection session.

Thin wrapper over the live Stage-1 extractor
(``build_events_and_intervals.build_all``). It returns a typed
:class:`~camstim_behavior_processing.load_data.session_data.SessionData`
holding the intermediates downstream packaging consumes: ``events_df``,
``intervals_df``, ``timestamp_data``, and the ``task_parameters`` lab
metadata. The per-trial view is available as ``SessionData.trials_df``.

Depends only on numpy / pandas / h5py — no NWB, no scipy — so it runs in the
base environment.
"""

from __future__ import annotations

from pathlib import Path

from ._bootstrap import add_legacy_to_path
from .session_data import SessionData


def build_trials_and_events(
    pkl_path: str | Path,
    sync_path: str | Path,
    output_dir: str | Path | None = None,
) -> SessionData:
    """Build the events and intervals intermediates for one session.

    Parameters
    ----------
    pkl_path : path to the camstim ``*_stim.pkl``.
    sync_path : path to the ``*_sync.h5`` sync file.
    output_dir : optional directory; if given, ``build_all`` also writes
        ``events_table.csv`` / ``intervals_table.csv`` (plus pickles and the
        timestamp ``.npz``) there.

    Returns
    -------
    A :class:`~camstim_behavior_processing.load_data.session_data.SessionData`
    with ``events_df``, ``intervals_df``, ``timestamp_data``, and
    ``task_parameters`` (and the derived ``trials_df`` view).
    """
    # Imported here (not at module load) so the legacy extractor directory is
    # only required when actually building trials/events — importing the
    # package to build just the wheel df does not need it.
    legacy_dir = add_legacy_to_path()
    try:
        from build_events_and_intervals import build_all
    except ModuleNotFoundError as exc:
        if exc.name == "build_events_and_intervals":
            raise ModuleNotFoundError(
                "Could not import 'build_events_and_intervals'. "
                "Expected legacy extractor files under "
                f"'{legacy_dir}'. Reinstall this package from a wheel/sdist "
                "that includes 'data_loading_and_formatting_code', or "
                "install from source."
            ) from exc
        raise

    built = build_all(str(pkl_path), str(sync_path), output_dir=output_dir)
    return SessionData(
        events_df=built["events_df"],
        intervals_df=built["intervals_df"],
        timestamp_data=built["timestamp_data"],
        task_parameters=built["task_parameters"],
    )
