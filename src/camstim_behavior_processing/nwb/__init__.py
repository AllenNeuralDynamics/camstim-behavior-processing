"""Assemble a complete change-detection NWB file from the intermediates.

This subpackage is the NWB-packaging layer. It turns the ``load_data``
intermediates (events / trials / intervals DataFrames + the running-wheel df +
the task-parameter lab metadata) into a fully-populated
:class:`~ndx_events.NdxEventsNWBFile`:

    NWBFile
    |-- lab_meta_data (task_parameters + hed_schema)
    |-- subject
    |-- trials                      (intervals.add_trials)
    |-- intervals
    |   |-- intervals               (intervals.build_intervals_table)
    |   |-- stimulus_presentations  (stimulus.*)
    |   +-- natural_movie_one_presentations
    |-- events (ndx-events EventsTable)   (events.build_events_table)
    +-- acquisition + processing/running  (acquisition.add_running_speed)

:func:`assemble_nwbfile` builds the NWBFile from already-loaded DataFrames;
:func:`package_nwb` is the one-call path (raw ``*_stim.pkl`` / ``*_sync.h5`` ->
intermediates -> assembled NWB, written as NWB-Zarr by default) and also emits
the BIDS-style ``*.events.json`` sidecar. :func:`package_session` is the
type-agnostic entry point: it inspects the pkl and dispatches to
:func:`package_nwb` (change-detection) or :func:`package_sweepstim_nwb`
(passive SweepStim).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from hdmf_zarr.nwb import NWBZarrIO
from ndx_events import NdxEventsNWBFile
from ndx_hed import HedLabMetaData
from pynwb import NWBHDF5IO

from ..load_data.loaders import load_stim_pkl
from ..load_data.running_speed import compute_running_speed
from ..load_data.session_data import SessionData
from ..load_data.sweepstim import classify_sweepstim_session
from ..load_data.trials_events import build_trials_and_events
from .acquisition import add_running_speed
from .epochs import build_epoch_lookup
from .events import add_events, build_events_table
from .file import build_nwbfile, build_subject
from .hed_tags import HED_SCHEMA_VERSION
from .intervals import add_intervals, add_trials, build_intervals_table
from .sidecar import build_events_sidecar
from .stimulus import (
    add_stimulus_presentations,
    build_natural_movie_one_presentations,
    build_stimulus_presentations,
)
from .sweepstim import (
    assemble_sweepstim_nwbfile,
    build_sweepstim_nwbfile,
    build_sweepstim_sidecar,
    package_sweepstim_nwb,
)

__all__ = [
    "add_running_speed",
    "add_events",
    "add_trials",
    "add_intervals",
    "add_stimulus_presentations",
    "build_events_table",
    "build_intervals_table",
    "build_stimulus_presentations",
    "build_natural_movie_one_presentations",
    "build_epoch_lookup",
    "build_events_sidecar",
    "build_subject",
    "build_nwbfile",
    "assemble_nwbfile",
    "package_nwb",
    "package_session",
    "assemble_sweepstim_nwbfile",
    "build_sweepstim_nwbfile",
    "build_sweepstim_sidecar",
    "package_sweepstim_nwb",
    "HED_SCHEMA_VERSION",
]


def _warm_up_n(pkl: dict) -> int:
    """Number of warm-up trials from the behavior params (0 if absent)."""
    params = pkl["items"]["behavior"]["params"]
    return int(params.get("warm_up_trials", 0))


def assemble_nwbfile(
    pkl: dict,
    session: SessionData,
    wheel_df: pd.DataFrame,
    *,
    metadata: dict[str, Any] | None = None,
) -> NdxEventsNWBFile:
    """Assemble a complete NWBFile from the loaded intermediates.

    Parameters
    ----------
    pkl : loaded behavior pickle (identity/subject/warm-up source).
    session : the :class:`~.load_data.session_data.SessionData` from
        ``build_trials_and_events`` (events/intervals + task parameters).
    wheel_df : running-wheel df from ``compute_running_speed``.
    metadata : optional identity/subject override dict (see
        :func:`~camstim_behavior_processing.nwb.file.build_nwbfile`).

    Returns
    -------
    The fully-populated :class:`~ndx_events.NdxEventsNWBFile`.
    """
    events_df = session.events_df
    intervals_df = session.intervals_df

    nwb = build_nwbfile(pkl, metadata)
    nwb.add_lab_meta_data(session.task_parameters)
    nwb.add_lab_meta_data(
        HedLabMetaData(hed_schema_version=HED_SCHEMA_VERSION)
    )

    epoch_list = build_epoch_lookup(intervals_df, events_df)
    add_stimulus_presentations(nwb, events_df, epoch_list)
    add_trials(nwb, intervals_df, _warm_up_n(pkl), epoch_list)
    add_events(nwb, events_df)
    add_intervals(nwb, intervals_df, events_df)
    add_running_speed(nwb, wheel_df)
    return nwb


def _write_nwb(nwb: NdxEventsNWBFile, output_path: Path, fmt: str) -> None:
    """Write ``nwb`` to ``output_path`` as NWB-Zarr or HDF5.

    Parameters
    ----------
    nwb : the assembled NWBFile.
    output_path : destination path.
    fmt : ``"zarr"`` (NWB-Zarr) or ``"hdf5"`` (NWB-HDF5).
    """
    if fmt == "zarr":
        with NWBZarrIO(str(output_path), mode="w") as io:
            io.write(nwb)
    elif fmt == "hdf5":
        with NWBHDF5IO(str(output_path), mode="w") as io:
            io.write(nwb)
    else:
        raise ValueError(f"unknown fmt {fmt!r}; expected 'zarr' or 'hdf5'")


def _write_sidecar(output_path: Path) -> Path:
    """Write the events sidecar next to ``output_path`` and return its path.

    The sidecar stem is the output name up to the first dot, so a
    ``behavior.nwb.zarr`` output yields ``behavior.events.json``.
    """
    stem = output_path.name.split(".")[0]
    sidecar_path = output_path.parent / f"{stem}.events.json"
    with open(sidecar_path, "w") as f:
        json.dump(build_events_sidecar(), f, indent=2, ensure_ascii=False)
    return sidecar_path


def package_nwb(
    pkl_path: str | Path,
    sync_path: str | Path,
    *,
    output_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    fmt: str = "zarr",
    write_sidecar: bool = True,
) -> NdxEventsNWBFile:
    """Package one change-detection session into a complete NWB file.

    Loads the raw ``*_stim.pkl`` / ``*_sync.h5``, builds every intermediate,
    and assembles the full NWB (lab metadata, subject, trials, intervals,
    stimulus/movie presentations, events, running speed).

    Parameters
    ----------
    pkl_path : path to the camstim ``*_stim.pkl``.
    sync_path : path to the ``*_sync.h5`` sync file.
    output_path : if given, the assembled NWB is written here. When ``None``,
        nothing is written and no sidecar is emitted.
    metadata : optional identity/subject override dict.
    fmt : output format, ``"zarr"`` (default) or ``"hdf5"``.
    write_sidecar : if True (and ``output_path`` is given), also write the
        BIDS-style ``*.events.json`` sidecar next to the NWB.

    Returns
    -------
    The assembled :class:`~ndx_events.NdxEventsNWBFile` (always returned;
    written only when ``output_path`` is given).
    """
    session = build_trials_and_events(pkl_path, sync_path)
    pkl = load_stim_pkl(pkl_path)
    wheel_df = compute_running_speed(
        pkl, session.timestamp_data["stim_vsync_fall"]
    )

    nwb = assemble_nwbfile(pkl, session, wheel_df, metadata=metadata)

    if output_path is not None:
        output_path = Path(output_path)
        _write_nwb(nwb, output_path, fmt)
        if write_sidecar:
            _write_sidecar(output_path)
    return nwb


def package_session(
    pkl_path: str | Path,
    sync_path: str | Path,
    *,
    output_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    fmt: str = "zarr",
    write_sidecar: bool = True,
) -> NdxEventsNWBFile:
    """Package one session into NWB, auto-routing on its pkl structure.

    Inspects the raw ``*.pkl`` with
    :func:`~camstim_behavior_processing.load_data.classify_sweepstim_session`
    and delegates to :func:`package_sweepstim_nwb` for a passive SweepStim
    session or :func:`package_nwb` for an active change-detection session. All
    keyword arguments are forwarded unchanged, so a mixed batch can be packaged
    through this single entry point without knowing each session's type.

    Parameters
    ----------
    pkl_path : path to the camstim ``*.pkl``.
    sync_path : path to the ``*_sync.h5`` sync file.
    output_path : if given, the assembled NWB is written here. When ``None``,
        nothing is written and no sidecar is emitted.
    metadata : optional identity/subject override dict.
    fmt : output format, ``"zarr"`` (default) or ``"hdf5"``.
    write_sidecar : if True (and ``output_path`` is given), also write the
        BIDS-style ``*.events.json`` sidecar next to the NWB.

    Returns
    -------
    The assembled :class:`~ndx_events.NdxEventsNWBFile`.
    """
    pkl = load_stim_pkl(pkl_path)
    is_sweepstim, _ = classify_sweepstim_session(pkl)
    packager = package_sweepstim_nwb if is_sweepstim else package_nwb
    return packager(
        pkl_path,
        sync_path,
        output_path=output_path,
        metadata=metadata,
        fmt=fmt,
        write_sidecar=write_sidecar,
    )
