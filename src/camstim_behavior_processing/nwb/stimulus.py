"""Build the stimulus-presentation ``TimeIntervals`` tables.

Two compositional interval tables, both derived from the ``events_df``:

- :func:`build_stimulus_presentations` — one row per active-task image flash
  (plus a synthetic row per omitted flash slot), carrying image identity,
  change/omission flags, orientation, frame indices, per-flash lick latency,
  and the containing epoch.
- :func:`build_natural_movie_one_presentations` — one row per frame of the
  post-task ``natural_movie_one`` fingerprint clip (``None`` if the session
  has no movie).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hdmf.common import VectorData
from ndx_hed import HedTags
from pynwb import NWBFile
from pynwb.epoch import TimeIntervals

from . import hed_tags as H
from .epochs import epoch_name_at

# Synthetic duration for an omitted flash slot (no offset event exists).
_OMITTED_DURATION_S = 0.25
_OMITTED_DURATION_FRAMES = 15


def _presentation_rows(events_df: pd.DataFrame) -> list[dict]:
    """Build the per-presentation row dicts, sorted by onset time.

    Parameters
    ----------
    events_df : the events frame.

    Returns
    -------
    list of row dicts (image flashes + omitted slots) sorted by start time.
    """
    onsets = events_df[events_df["event_type"] == "image_onset"]
    offsets = events_df[events_df["event_type"] == "image_offset"]
    omitted = events_df[events_df["event_type"] == "image_omission"]
    change_sids = set(
        events_df.loc[
            events_df["event_type"] == "image_change",
            "stimulus_presentations_id",
        ]
        .astype(int)
        .values
    )
    off_t_by_sid = dict(
        zip(
            offsets["stimulus_presentations_id"].astype(int),
            offsets["timestamp"],
        )
    )
    off_f_by_sid = dict(
        zip(
            offsets["stimulus_presentations_id"].astype(int),
            offsets["frame"].astype(int),
        )
    )

    rows: list[dict] = []
    for _, row in onsets.iterrows():
        sid = int(row["stimulus_presentations_id"])
        on_t = float(row["timestamp"])
        on_f = int(row["frame"])
        tid = int(row["trials_id"]) if pd.notna(row["trials_id"]) else -1
        ori = (
            float(row["orientation"])
            if pd.notna(row.get("orientation"))
            else np.nan
        )
        rows.append(
            {
                "start_time": on_t,
                "stop_time": float(
                    off_t_by_sid.get(sid, on_t + _OMITTED_DURATION_S)
                ),
                "image_name": str(row["image_name"]),
                "is_change": sid in change_sids,
                "omitted": False,
                "sid": sid,
                "tid": tid,
                "start_frame": on_f,
                "stop_frame": int(
                    off_f_by_sid.get(sid, on_f + _OMITTED_DURATION_FRAMES)
                ),
                "orientation": ori,
            }
        )
    for _, row in omitted.iterrows():
        sid = (
            int(row["stimulus_presentations_id"])
            if pd.notna(row["stimulus_presentations_id"])
            else -1
        )
        on_t = float(row["timestamp"])
        on_f = int(row["frame"])
        tid = int(row["trials_id"]) if pd.notna(row["trials_id"]) else -1
        rows.append(
            {
                "start_time": on_t,
                "stop_time": on_t + _OMITTED_DURATION_S,
                "image_name": "omitted",
                "is_change": False,
                "omitted": True,
                "sid": sid,
                "tid": tid,
                "start_frame": on_f,
                "stop_frame": on_f + _OMITTED_DURATION_FRAMES,
                "orientation": np.nan,
            }
        )
    rows.sort(key=lambda r: r["start_time"])
    return rows


def _lick_latency(start_time: list, events_df: pd.DataFrame) -> list:
    """Per-presentation latency to the first lick before the next flash.

    Parameters
    ----------
    start_time : the sorted presentation onset times.
    events_df : the events frame (for lick timestamps).

    Returns
    -------
    list of latencies (s), ``NaN`` where no lick falls before the next flash.
    """
    lick_times = np.asarray(
        events_df.loc[events_df["event_type"] == "lick", "timestamp"].values,
        dtype=float,
    )
    starts = np.asarray(start_time, dtype=float)
    next_start = np.concatenate([starts[1:], [np.inf]])
    out = []
    for s, nxt in zip(starts, next_start):
        idx = np.searchsorted(lick_times, s, side="left")
        if idx < len(lick_times) and lick_times[idx] < nxt:
            out.append(float(lick_times[idx] - s))
        else:
            out.append(np.nan)
    return out


def build_stimulus_presentations(
    events_df: pd.DataFrame,
    epoch_list: list[dict] | None = None,
) -> TimeIntervals:
    """Build the ``stimulus_presentations`` TimeIntervals table.

    Parameters
    ----------
    events_df : the events frame.
    epoch_list : optional canonical epoch list (from
        :func:`~camstim_behavior_processing.nwb.epochs.build_epoch_lookup`)
        used to tag each presentation with its epoch; defaults to
        ``change_detection`` for every row when omitted.

    Returns
    -------
    A :class:`~pynwb.epoch.TimeIntervals` with one row per image flash and one
    per omitted slot, sorted by onset.
    """
    rows = _presentation_rows(events_df)
    start_time = [r["start_time"] for r in rows]
    image_name = [r["image_name"] for r in rows]
    is_change = [r["is_change"] for r in rows]
    omitted_flag = [r["omitted"] for r in rows]
    hed = [
        H.stim_presentation_hed(n, c, o)
        for n, c, o in zip(image_name, is_change, omitted_flag)
    ]
    epoch_names = (
        [epoch_name_at(t, epoch_list) for t in start_time]
        if epoch_list is not None
        else ["change_detection"] * len(rows)
    )

    return TimeIntervals(
        name="stimulus_presentations",
        description="Per-flash stimulus presentations during the active "
        "change-detection task. Omitted flashes are represented with "
        'image_name="omitted" and omitted=True.',
        columns=[
            VectorData(
                name="start_time",
                description="Flash onset (s).",
                data=start_time,
            ),
            VectorData(
                name="stop_time",
                description="Flash offset (s).",
                data=[r["stop_time"] for r in rows],
            ),
            VectorData(
                name="image_name",
                description="Stimulus identity: image name for natural-image "
                'sessions, "gratings_<ori>" for gratings sessions, or '
                '"omitted".',
                data=image_name,
            ),
            VectorData(
                name="orientation",
                description="Grating orientation in degrees (NaN for "
                "natural-image and omitted presentations).",
                data=[r["orientation"] for r in rows],
            ),
            VectorData(
                name="is_change",
                description="True if image identity differs from the "
                "previous (non-omitted) flash.",
                data=is_change,
            ),
            VectorData(
                name="omitted",
                description="True for withheld flashes (no image shown).",
                data=omitted_flag,
            ),
            VectorData(
                name="stimulus_presentations_id",
                description="Sequential id matching the events table.",
                data=[r["sid"] for r in rows],
            ),
            VectorData(
                name="trials_id",
                description="Id of the trial this presentation belongs to "
                "(-1 if outside any trial).",
                data=[r["tid"] for r in rows],
            ),
            VectorData(
                name="start_frame",
                description="Vsync falling-edge frame index for the flash "
                "onset.",
                data=[r["start_frame"] for r in rows],
            ),
            VectorData(
                name="stop_frame",
                description="Vsync falling-edge frame index for the flash "
                "offset.",
                data=[r["stop_frame"] for r in rows],
            ),
            VectorData(
                name="lick_latency",
                description="Time from this stim onset to the first lick "
                "occurring before the next stim onset, in seconds. NaN if no "
                "lick in that window.",
                data=_lick_latency(start_time, events_df),
            ),
            VectorData(
                name="epoch_name",
                description="Canonical epoch containing this presentation.",
                data=epoch_names,
            ),
            HedTags(
                name="HED",
                description="HED tag string for this presentation.",
                data=hed,
            ),
        ],
        id=list(range(len(rows))),
    )


def build_natural_movie_one_presentations(
    events_df: pd.DataFrame,
) -> TimeIntervals | None:
    """Build the ``natural_movie_one_presentations`` table (one row/frame).

    Parameters
    ----------
    events_df : the events frame.

    Returns
    -------
    A :class:`~pynwb.epoch.TimeIntervals`, or ``None`` if the session has no
    ``movie_onset`` events.
    """
    onsets = events_df[events_df["event_type"] == "movie_onset"].sort_values(
        "timestamp"
    )
    offsets = events_df[events_df["event_type"] == "movie_offset"].sort_values(
        "timestamp"
    )
    if len(onsets) == 0:
        return None

    n = len(onsets)
    return TimeIntervals(
        name="natural_movie_one_presentations",
        description="Per-frame presentations of the natural_movie_one "
        "fingerprint clip shown after the active task.",
        columns=[
            VectorData(
                name="start_time",
                description="Frame onset (s).",
                data=onsets["timestamp"].astype(float).tolist(),
            ),
            VectorData(
                name="stop_time",
                description="Frame offset (s).",
                data=offsets["timestamp"].astype(float).tolist(),
            ),
            VectorData(
                name="movie_frame_index",
                description="Frame index within a single movie playback.",
                data=onsets["movie_frame_index"]
                .fillna(-1)
                .astype(int)
                .tolist(),
            ),
            VectorData(
                name="movie_repeat",
                description="Repetition of the movie this frame belongs to.",
                data=onsets["movie_repeat"].fillna(-1).astype(int).tolist(),
            ),
            VectorData(
                name="start_frame",
                description="Vsync falling-edge frame index for the "
                "movie-frame onset.",
                data=onsets["frame"].astype(int).tolist(),
            ),
            VectorData(
                name="stop_frame",
                description="Vsync falling-edge frame index for the "
                "movie-frame offset.",
                data=offsets["frame"].astype(int).tolist(),
            ),
            VectorData(
                name="epoch_name",
                description="Canonical epoch containing this frame.",
                data=["natural_movie_one"] * n,
            ),
            HedTags(
                name="HED",
                description="HED tag string for this movie frame.",
                data=[H.MOVIE_FRAME_HED] * n,
            ),
        ],
        id=list(range(n)),
    )


def add_stimulus_presentations(
    nwb: NWBFile,
    events_df: pd.DataFrame,
    epoch_list: list[dict] | None = None,
) -> NWBFile:
    """Add the stimulus_presentations (and movie, if any) tables to ``nwb``.

    Parameters
    ----------
    nwb : the NWBFile to add to (modified in place).
    events_df : the events frame.
    epoch_list : optional canonical epoch list for epoch tagging.

    Returns
    -------
    The same ``nwb`` instance, for chaining.
    """
    nwb.add_time_intervals(build_stimulus_presentations(events_df, epoch_list))
    movie = build_natural_movie_one_presentations(events_df)
    if movie is not None:
        nwb.add_time_intervals(movie)
    return nwb
