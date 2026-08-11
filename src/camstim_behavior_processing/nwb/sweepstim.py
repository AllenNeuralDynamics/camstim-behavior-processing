"""Package a passive SweepStim session into a complete NWB file.

The NWB-packaging layer for passive (movie / grating viewing) sessions,
mirroring the change-detection :mod:`camstim_behavior_processing.nwb` writers
but for the simpler passive structure (no trials / events tables — only
stimulus presentations and epochs):

    NWBFile
    |-- lab_meta_data (hed_schema)
    |-- subject
    |-- intervals
    |   |-- stimulus_presentations   (per movie/grating frame)
    |   +-- intervals                (flat: epochs + stimulus_presentation)
    +-- acquisition + processing/running  (acquisition.add_running_speed)

:func:`assemble_sweepstim_nwbfile` builds the NWBFile from already-loaded
intermediates; :func:`package_sweepstim_nwb` is the one-call raw-files -> NWB
path (written NWB-Zarr by default) and also emits the ``*.events.json``
sidecar.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from hdmf.common import VectorData
from hdmf_zarr.nwb import NWBZarrIO
from ndx_hed import HedLabMetaData, HedTags
from pynwb import NWBFile
from pynwb import NWBHDF5IO
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

from ..load_data.loaders import load_stim_pkl
from ..load_data.running_speed import compute_running_speed
from ..load_data.sweepstim import (
    SweepStimData,
    build_sweepstim_session,
    classify_sweepstim_session,
)
from . import hed_tags as H
from .acquisition import add_running_speed

# The SweepStim path reads the encoder from the foraging item first, then
# falls back to the behavior item.
_SWEEPSTIM_ENCODER_KEYS = ("foraging", "behavior")


def _to_datetime(value) -> datetime.datetime:
    """Coerce a camstim start-time value into a tz-aware UTC datetime."""
    if isinstance(value, datetime.datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.datetime.fromisoformat(value)
    elif isinstance(value, (int, float)):
        # camstim stores start_time as a Unix epoch timestamp.
        dt = datetime.datetime.fromtimestamp(
            float(value), datetime.timezone.utc
        )
    else:
        dt = datetime.datetime.now(datetime.timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def _normalize_dob(dob):
    """Coerce a date-of-birth value into a tz-aware datetime, or ``None``."""
    if isinstance(dob, str):
        try:
            dob = datetime.date.fromisoformat(dob)
        except ValueError:
            return None
    # pynwb Subject.date_of_birth requires a (tz-aware) datetime, not a date.
    if isinstance(dob, datetime.date) and not isinstance(
        dob, datetime.datetime
    ):
        return datetime.datetime(
            dob.year, dob.month, dob.day, tzinfo=datetime.timezone.utc
        )
    return dob


def _subject_id_from_pkl(pkl: dict) -> str:
    """Resolve the subject id from behavior/foraging params or ``mouseid``."""
    items = pkl.get("items") or {}
    for key in ("behavior", "foraging"):
        params = (items.get(key) or {}).get("params") or {}
        if params.get("mouse_id") is not None:
            return str(params.get("mouse_id"))
    if pkl.get("mouseid") is not None:
        return str(pkl.get("mouseid"))
    return "unknown"


def build_sweepstim_nwbfile(
    pkl: dict, metadata: dict[str, Any] | None = None
) -> NWBFile:
    """Create an NWBFile (identity + subject) for a passive session.

    Parameters
    ----------
    pkl : the loaded behavior pickle. ``session_start_time`` is taken from
        ``startdatetime`` / ``start_time``; the description defaults to the
        pkl ``stage``.
    metadata : optional override dict — keys ``session_description``,
        ``identifier``, ``experimenter``, ``lab``, ``institution``, ``notes``,
        and the subject keys (``species``, ``age``, ``sex``, ``genotype``,
        ``strain``, ``date_of_birth``, ``subject_description``).

    Returns
    -------
    A :class:`~pynwb.NWBFile` with no data yet.
    """
    metadata = metadata or {}
    nwb = NWBFile(
        session_description=metadata.get(
            "session_description",
            pkl.get("stage", "sweepstim_passive"),
        ),
        identifier=metadata.get("identifier", str(uuid4())),
        session_start_time=_to_datetime(
            pkl.get("startdatetime") or pkl.get("start_time")
        ),
        experimenter=metadata.get("experimenter"),
        lab=metadata.get("lab"),
        institution=metadata.get("institution"),
        notes=metadata.get("notes"),
    )
    nwb.subject = Subject(
        subject_id=_subject_id_from_pkl(pkl),
        species=metadata.get("species", "Mus musculus"),
        age=metadata.get("age"),
        sex=metadata.get("sex", "U"),
        genotype=metadata.get("genotype"),
        strain=metadata.get("strain"),
        date_of_birth=_normalize_dob(metadata.get("date_of_birth")),
        description=metadata.get("subject_description"),
    )
    return nwb


def build_stimulus_presentations_sweepstim(
    presentations_df: pd.DataFrame,
) -> TimeIntervals:
    """Build the passive ``stimulus_presentations`` TimeIntervals table.

    Parameters
    ----------
    presentations_df : the presentation table from
        :func:`~camstim_behavior_processing.load_data.sweepstim.build_sweepstim_session`.

    Returns
    -------
    A :class:`~pynwb.epoch.TimeIntervals` with one row per movie/grating frame.
    """
    df = presentations_df
    movie_name = df["movie_name"].tolist()
    hed = [H.sweepstim_movie_hed(name) for name in movie_name]
    return TimeIntervals(
        name="stimulus_presentations",
        description="Per-frame SweepStim movie presentations for passive "
        "sessions.",
        columns=[
            VectorData(
                name="start_time",
                description="Frame onset (s).",
                data=df["start_time"].tolist(),
            ),
            VectorData(
                name="stop_time",
                description="Frame offset (s).",
                data=df["stop_time"].tolist(),
            ),
            VectorData(
                name="movie_name",
                description=H.SWEEPSTIM_COLUMN_DESC["movie_name"],
                data=movie_name,
            ),
            VectorData(
                name="movie_frame_index",
                description=H.SWEEPSTIM_COLUMN_DESC["movie_frame_index"],
                data=df["movie_frame_index"].tolist(),
            ),
            VectorData(
                name="movie_repeat",
                description=H.SWEEPSTIM_COLUMN_DESC["movie_repeat"],
                data=df["movie_repeat"].tolist(),
            ),
            VectorData(
                name="stim_block",
                description=H.SWEEPSTIM_COLUMN_DESC["stim_block"],
                data=df["stim_block"].tolist(),
            ),
            VectorData(
                name="start_frame",
                description="Vsync frame index at onset.",
                data=df["start_frame"].tolist(),
            ),
            VectorData(
                name="stop_frame",
                description="Vsync frame index at offset.",
                data=df["stop_frame"].tolist(),
            ),
            VectorData(
                name="epoch_name",
                description="Canonical epoch label.",
                data=df["epoch_name"].tolist(),
            ),
            HedTags(
                name="HED",
                description="HED tag string for this movie frame.",
                data=hed,
            ),
        ],
        id=list(range(len(df))),
    )


def _flat_interval_rows(
    epochs_df: pd.DataFrame, presentations_df: pd.DataFrame
) -> list[dict]:
    """Build the flat epoch + stimulus_presentation rows, sorted by onset."""
    rows: list[dict] = []
    for name, start, stop in zip(
        epochs_df["name"],
        epochs_df["start_time"],
        epochs_df["stop_time"],
    ):
        rows.append(
            {
                "start_time": float(start),
                "stop_time": float(stop),
                "interval_type": "epoch",
                "label": name,
                "stimulus_presentations_id": -1,
                "HED": H.sweepstim_epoch_hed(name),
            }
        )
    for sid, (start, stop, name) in enumerate(
        zip(
            presentations_df["start_time"],
            presentations_df["stop_time"],
            presentations_df["movie_name"],
        )
    ):
        rows.append(
            {
                "start_time": float(start),
                "stop_time": float(stop),
                "interval_type": "stimulus_presentation",
                "label": name,
                "stimulus_presentations_id": sid,
                "HED": H.sweepstim_movie_hed(name),
            }
        )
    rows.sort(key=lambda r: r["start_time"])
    return rows


def build_intervals_table_sweepstim(
    epochs_df: pd.DataFrame, presentations_df: pd.DataFrame
) -> TimeIntervals:
    """Build the flat ``intervals`` table (epochs + stimulus presentations).

    Parameters
    ----------
    epochs_df : the epoch table.
    presentations_df : the presentation table (its row order defines the
        ``stimulus_presentations_id`` foreign key).

    Returns
    -------
    A :class:`~pynwb.epoch.TimeIntervals` with every epoch and presentation
    row, sorted by onset.
    """
    rows = _flat_interval_rows(epochs_df, presentations_df)
    return TimeIntervals(
        name="intervals",
        description="Flat intervals table for SweepStim sessions (epochs + "
        "stimulus frames).",
        columns=[
            VectorData(
                name="start_time",
                description="Interval start (s).",
                data=[r["start_time"] for r in rows],
            ),
            VectorData(
                name="stop_time",
                description="Interval stop (s).",
                data=[r["stop_time"] for r in rows],
            ),
            VectorData(
                name="interval_type",
                description="epoch or stimulus_presentation.",
                data=[r["interval_type"] for r in rows],
            ),
            VectorData(
                name="label",
                description="Epoch label or movie clip label.",
                data=[r["label"] for r in rows],
            ),
            VectorData(
                name="stimulus_presentations_id",
                description="Foreign key into stimulus_presentations (-1 if "
                "N/A).",
                data=[r["stimulus_presentations_id"] for r in rows],
            ),
            HedTags(
                name="HED",
                description="HED tag string for this interval.",
                data=[r["HED"] for r in rows],
            ),
        ],
        id=list(range(len(rows))),
    )


def build_sweepstim_sidecar() -> dict:
    """Build the compact BIDS-style sidecar for SweepStim-specific columns.

    Returns
    -------
    A JSON-serialisable dict. :func:`package_sweepstim_nwb` writes it to
    ``<output>.events.json`` alongside the NWB.
    """
    time_hed = H.VALUE_COLUMN_HED["start_time"]
    frame_hed = H.VALUE_COLUMN_HED["start_frame"]
    return {
        "start_time": {
            "Description": "Frame or interval start time (s).",
            "HED": time_hed,
        },
        "stop_time": {
            "Description": "Frame or interval stop time (s).",
            "HED": time_hed,
        },
        "movie_name": {"Description": H.SWEEPSTIM_COLUMN_DESC["movie_name"]},
        "movie_frame_index": {
            "Description": H.SWEEPSTIM_COLUMN_DESC["movie_frame_index"],
            "HED": H.SWEEPSTIM_COLUMN_HED["movie_frame_index"],
        },
        "movie_repeat": {
            "Description": H.SWEEPSTIM_COLUMN_DESC["movie_repeat"],
            "HED": H.SWEEPSTIM_COLUMN_HED["movie_repeat"],
        },
        "stim_block": {
            "Description": H.SWEEPSTIM_COLUMN_DESC["stim_block"],
            "HED": H.SWEEPSTIM_COLUMN_HED["stim_block"],
        },
        "start_frame": {
            "Description": "Vsync frame at onset.",
            "HED": frame_hed,
        },
        "stop_frame": {
            "Description": "Vsync frame at offset.",
            "HED": frame_hed,
        },
        "interval_type": {
            "Description": "Type of interval row.",
            "Levels": dict(H.SWEEPSTIM_INTERVAL_TYPE_DESC),
        },
        "epoch_name": {"Description": "Canonical epoch label."},
        "HED": {
            "Description": "Hierarchical Event Descriptor tags for each row."
        },
        "hed_defs": {"HED": {"alldefs": ""}},
    }


def assemble_sweepstim_nwbfile(
    pkl: dict,
    sweep: SweepStimData,
    wheel_df: pd.DataFrame,
    *,
    metadata: dict[str, Any] | None = None,
) -> NWBFile:
    """Assemble a complete passive-session NWBFile from the intermediates.

    Parameters
    ----------
    pkl : the loaded behavior pickle (identity / subject source).
    sweep : the ``SweepStimData`` from ``build_sweepstim_session``.
    wheel_df : running-wheel df from ``compute_running_speed``.
    metadata : optional identity/subject override dict.

    Returns
    -------
    The fully-populated :class:`~pynwb.NWBFile`.
    """
    nwb = build_sweepstim_nwbfile(pkl, metadata)
    nwb.add_lab_meta_data(
        HedLabMetaData(hed_schema_version=H.HED_SCHEMA_VERSION)
    )
    nwb.add_time_intervals(
        build_stimulus_presentations_sweepstim(sweep.presentations_df)
    )
    nwb.add_time_intervals(
        build_intervals_table_sweepstim(
            sweep.epochs_df, sweep.presentations_df
        )
    )
    add_running_speed(nwb, wheel_df)
    return nwb


def _write_nwb(nwb: NWBFile, output_path: Path, fmt: str) -> None:
    """Write ``nwb`` to ``output_path`` as NWB-Zarr or HDF5."""
    if fmt == "zarr":
        with NWBZarrIO(str(output_path), mode="w") as io:
            io.write(nwb)
    elif fmt == "hdf5":
        with NWBHDF5IO(str(output_path), mode="w") as io:
            io.write(nwb)
    else:
        raise ValueError(f"unknown fmt {fmt!r}; expected 'zarr' or 'hdf5'")


def _write_sidecar(output_path: Path) -> Path:
    """Write the SweepStim sidecar next to ``output_path`` and return it."""
    stem = output_path.name.split(".")[0]
    sidecar_path = output_path.parent / f"{stem}.events.json"
    with open(sidecar_path, "w") as f:
        json.dump(build_sweepstim_sidecar(), f, indent=2, ensure_ascii=False)
    return sidecar_path


def package_sweepstim_nwb(
    pkl_path: str | Path,
    sync_path: str | Path,
    *,
    output_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    fmt: str = "zarr",
    write_sidecar: bool = True,
) -> NWBFile:
    """Package one passive SweepStim session into a complete NWB file.

    Loads the raw ``*.pkl`` / ``*_sync.h5``, verifies the session is passive,
    builds every intermediate, and assembles the full NWB (subject, stimulus
    presentations, flat intervals, running speed).

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
    The assembled :class:`~pynwb.NWBFile`.

    Raises
    ------
    ValueError : if the pkl is not a passive SweepStim session.
    """
    pkl = load_stim_pkl(pkl_path)
    is_sweepstim, detail = classify_sweepstim_session(pkl)
    if not is_sweepstim:
        raise ValueError(f"Not a SweepStim session: {detail}")

    sweep = build_sweepstim_session(pkl, sync_path)
    wheel_df = compute_running_speed(
        pkl,
        sweep.timestamp_data["stim_vsync_fall"],
        encoder_keys=_SWEEPSTIM_ENCODER_KEYS,
    )
    nwb = assemble_sweepstim_nwbfile(pkl, sweep, wheel_df, metadata=metadata)

    if output_path is not None:
        output_path = Path(output_path)
        _write_nwb(nwb, output_path, fmt)
        if write_sidecar:
            _write_sidecar(output_path)
    return nwb
