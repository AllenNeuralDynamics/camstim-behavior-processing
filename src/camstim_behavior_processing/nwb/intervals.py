"""Write the trials table and the canonical flat intervals table.

Two consumers of the ``intervals_df`` (and ``events_df``) from
:func:`camstim_behavior_processing.load_data.trials_events.build_trials_and_events`:

- :func:`add_trials` — the NWB trials table (a compositional view: timing plus
  go/catch/hit/miss/... booleans, change/response timing, image names,
  orientations, warm-up flag, epoch, and a HED outcome tag).
- :func:`build_intervals_table` — the canonical flat ``intervals``
  ``TimeIntervals`` holding every interval type (epoch, trial, change_window,
  response_window, stimulus_presentation, movie_frame) with timing, a type
  discriminator, a label, foreign keys, and a HED tag. Task-specific
  annotations live on the per-type physical tables, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hdmf.common import VectorData
from ndx_hed import HedTags
from pynwb import NWBFile
from pynwb.epoch import TimeIntervals

from . import hed_tags as H
from .epochs import build_epoch_lookup, epoch_name_at
from .stimulus import _OMITTED_DURATION_S


def _f(v, default=np.nan) -> float:
    """Coerce ``v`` to float, returning ``default`` when it is null."""
    return float(v) if pd.notna(v) else default


def _i(v, default=-1) -> int:
    """Coerce ``v`` to int, returning ``default`` when it is null."""
    return int(v) if pd.notna(v) else default


def _s(v, default="") -> str:
    """Coerce ``v`` to str, returning ``default`` when it is null."""
    return str(v) if pd.notna(v) else default


# ── Trials table ───────────────────────────────────────────────────────
_TRIAL_COLUMNS = [
    ("go", "Go trial (change presented)."),
    ("catch", "Catch trial (no change presented)."),
    ("auto_rewarded", "Trial with automatic reward delivery."),
    ("aborted", "Trial aborted by early lick."),
    ("hit", "Lick during response window after a change."),
    ("miss", "No lick during response window after a change."),
    (
        "false_alarm",
        "Lick during catch trial response window or change window.",
    ),
    ("correct_reject", "No lick on a catch trial."),
    ("warm_up", "True for the first N warm-up trials."),
    (
        "change_time",
        "Time of image change on this trial (NaN if no change).",
    ),
    ("change_frame", "Frame index of change (-1 if no change)."),
    (
        "initial_image_name",
        'Stimulus shown at trial start (image name, or "gratings_<ori>" '
        "for gratings sessions).",
    ),
    (
        "change_image_name",
        "Stimulus shown after change (same as initial if catch/abort).",
    ),
    (
        "initial_orientation",
        "Grating orientation (deg) at trial start (NaN for natural-image "
        "sessions).",
    ),
    (
        "change_orientation",
        "Grating orientation (deg) after change (NaN for natural-image "
        "sessions).",
    ),
    ("reward_time", "Time of reward delivery (NaN if none)."),
    ("reward_volume", "Volume of reward delivered (mL)."),
    ("response_time", "Time of first lick after change (NaN if none)."),
    (
        "response_latency",
        "Latency from change to first lick (NaN if none).",
    ),
    ("change_window_start_time", "Start of change window (NaN if none)."),
    ("change_window_stop_time", "Stop of change window (NaN if none)."),
    (
        "response_window_start_time",
        "Start of response window (NaN if none).",
    ),
    ("response_window_stop_time", "Stop of response window (NaN if none)."),
    ("epoch_name", "Canonical epoch containing this trial."),
]


def _declare_trial_columns(nwb: NWBFile) -> None:
    """Declare the custom trials columns (including the HED column)."""
    for name, desc in _TRIAL_COLUMNS:
        nwb.add_trial_column(name=name, description=desc)
    nwb.add_trial_column(
        name="HED",
        description="HED tag string for the trial outcome.",
        col_cls=HedTags,
    )


def _window_maps(intervals_df: pd.DataFrame) -> tuple[dict, dict]:
    """Map trial id -> (start, stop) for change and response windows."""
    cw = intervals_df[intervals_df["interval_type"] == "change_window"]
    rw = intervals_df[intervals_df["interval_type"] == "response_window"]
    cw_by_tid = {
        int(r["trials_id"]): (r["start_time"], r["stop_time"])
        for _, r in cw.iterrows()
    }
    rw_by_tid = {
        int(r["trials_id"]): (r["start_time"], r["stop_time"])
        for _, r in rw.iterrows()
    }
    return cw_by_tid, rw_by_tid


def _add_one_trial(
    nwb: NWBFile,
    row: pd.Series,
    cw_t: tuple,
    rw_t: tuple,
    warm_up: bool,
    epoch_list: list[dict] | None,
) -> None:
    """Add a single row to ``nwb.trials`` from a trial interval row."""
    epoch = (
        epoch_name_at(float(row["start_time"]), epoch_list)
        if epoch_list is not None
        else "change_detection"
    )
    nwb.add_trial(
        start_time=float(row["start_time"]),
        stop_time=float(row["stop_time"]),
        go=bool(row["go"]),
        catch=bool(row["catch"]),
        auto_rewarded=bool(row["auto_rewarded"]),
        aborted=bool(row["aborted"]),
        hit=bool(row["hit"]),
        miss=bool(row["miss"]),
        false_alarm=bool(row["false_alarm"]),
        correct_reject=bool(row["correct_reject"]),
        warm_up=warm_up,
        change_time=_f(row["change_time"]),
        change_frame=_i(row["change_frame"]),
        initial_image_name=_s(row["initial_image_name"]),
        change_image_name=_s(row["change_image_name"]),
        initial_orientation=_f(row.get("initial_orientation")),
        change_orientation=_f(row.get("change_orientation")),
        reward_time=_f(row["reward_time"]),
        reward_volume=_f(row["reward_volume"], 0.0),
        response_time=_f(row["response_time"]),
        response_latency=_f(row["response_latency"]),
        change_window_start_time=_f(cw_t[0]),
        change_window_stop_time=_f(cw_t[1]),
        response_window_start_time=_f(rw_t[0]),
        response_window_stop_time=_f(rw_t[1]),
        epoch_name=epoch,
        HED=H.trial_hed(row),
    )


def add_trials(
    nwb: NWBFile,
    intervals_df: pd.DataFrame,
    warm_up_n: int,
    epoch_list: list[dict] | None = None,
) -> NWBFile:
    """Populate ``nwb.trials`` from the trial rows of ``intervals_df``.

    Parameters
    ----------
    nwb : the NWBFile to add trials to (modified in place).
    intervals_df : the flat intervals frame; ``interval_type='trial'`` rows
        supply the trial annotations and the ``change_window`` /
        ``response_window`` rows supply the per-trial window times.
    warm_up_n : number of leading trials flagged ``warm_up=True``.
    epoch_list : optional canonical epoch list for per-trial epoch tagging.

    Returns
    -------
    The same ``nwb`` instance, for chaining.
    """
    trials = intervals_df[intervals_df["interval_type"] == "trial"].copy()
    cw_by_tid, rw_by_tid = _window_maps(intervals_df)

    _declare_trial_columns(nwb)
    for i, (_, row) in enumerate(trials.iterrows()):
        tid = int(row["trials_id"])
        _add_one_trial(
            nwb,
            row,
            cw_by_tid.get(tid, (np.nan, np.nan)),
            rw_by_tid.get(tid, (np.nan, np.nan)),
            i < warm_up_n,
            epoch_list,
        )
    return nwb


# ── Flat intervals table ───────────────────────────────────────────────
def _row(start, stop, itype, label, tid, sid, mid, hed) -> dict:
    """Build one flat-intervals row dict with all foreign-key columns."""
    return {
        "start_time": float(start),
        "stop_time": float(stop),
        "interval_type": itype,
        "label": label,
        "trials_id": tid,
        "stimulus_presentations_id": sid,
        "natural_movie_one_presentations_id": mid,
        "HED": hed,
    }


def _epoch_interval_rows(epoch_list: list[dict]) -> list[dict]:
    """Flat rows for the canonical session epochs."""
    return [
        _row(
            ep["start"],
            ep["stop"],
            "epoch",
            ep["name"],
            -1,
            -1,
            -1,
            H.EPOCH_HED.get(ep["name"], ""),
        )
        for ep in epoch_list
    ]


def _trial_window_rows(intervals_df: pd.DataFrame) -> list[dict]:
    """Flat rows for trials, change_windows, and response_windows."""
    iv = intervals_df.sort_values("start_time").reset_index(drop=True)
    rows: list[dict] = []
    for _, r in iv.iterrows():
        itype = r["interval_type"]
        if itype == "epoch":
            continue  # handled via the canonical epoch list
        if itype == "trial":
            hed = H.trial_hed(r)
        else:
            hed = _s(r.get("hed_string"))
        rows.append(
            _row(
                r["start_time"],
                r["stop_time"],
                itype,
                "",
                _i(r.get("trials_id")),
                -1,
                -1,
                hed,
            )
        )
    return rows


def _stim_interval_rows(events_df: pd.DataFrame) -> list[dict]:
    """Flat rows for image flashes + omitted slots (chronological sids)."""
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
    off_by_sid = dict(
        zip(
            offsets["stimulus_presentations_id"].astype(int),
            offsets["timestamp"],
        )
    )

    tmp = []
    for _, row in onsets.iterrows():
        sid = int(row["stimulus_presentations_id"])
        on_t = float(row["timestamp"])
        tmp.append(
            (
                on_t,
                float(off_by_sid.get(sid, on_t + _OMITTED_DURATION_S)),
                str(row["image_name"]),
                False,
                sid in change_sids,
            )
        )
    for _, row in omitted.iterrows():
        on_t = float(row["timestamp"])
        tmp.append((on_t, on_t + _OMITTED_DURATION_S, "omitted", True, False))
    tmp.sort(key=lambda r: r[0])

    # Reassign sequential sids in chronological order to match the physical
    # stimulus_presentations table row ids.
    return [
        _row(
            on_t,
            off_t,
            "stimulus_presentation",
            name,
            -1,
            new_sid,
            -1,
            H.stim_presentation_hed(name, is_change, is_omitted),
        )
        for new_sid, (on_t, off_t, name, is_omitted, is_change) in enumerate(
            tmp
        )
    ]


def _movie_interval_rows(events_df: pd.DataFrame) -> list[dict]:
    """Flat rows for natural_movie_one frames."""
    onsets = events_df[events_df["event_type"] == "movie_onset"].sort_values(
        "timestamp"
    )
    offsets = events_df[events_df["event_type"] == "movie_offset"].sort_values(
        "timestamp"
    )
    return [
        _row(
            on["timestamp"],
            off["timestamp"],
            "movie_frame",
            "",
            -1,
            -1,
            mid,
            H.MOVIE_FRAME_HED,
        )
        for mid, ((_, on), (_, off)) in enumerate(
            zip(onsets.iterrows(), offsets.iterrows())
        )
    ]


def build_intervals_table(
    intervals_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> TimeIntervals:
    """Build the canonical flat ``intervals`` TimeIntervals table.

    Parameters
    ----------
    intervals_df : the flat intervals frame (epochs / trials / windows).
    events_df : the events frame (source of the stimulus / movie rows).

    Returns
    -------
    A :class:`~pynwb.epoch.TimeIntervals` with one row per interval of every
    type, sorted by start time, carrying only timing, type, label, foreign
    keys, and HED.
    """
    epoch_list = build_epoch_lookup(intervals_df, events_df)
    rows = (
        _epoch_interval_rows(epoch_list)
        + _trial_window_rows(intervals_df)
        + _stim_interval_rows(events_df)
        + _movie_interval_rows(events_df)
    )
    rows.sort(key=lambda r: r["start_time"])

    def _col(name, desc):
        """Build a VectorData column pulling ``name`` from every row."""
        return VectorData(
            name=name, description=desc, data=[r[name] for r in rows]
        )

    return TimeIntervals(
        name="intervals",
        description="Canonical flat table of every session interval - "
        "epochs, trials, change_windows, response_windows, "
        "stimulus_presentations, and movie_frames. Contains only timing, "
        "type, label, foreign keys, and HED. Task-specific annotations live "
        "on the per-type physical tables (trials, stimulus_presentations, "
        "natural_movie_one_presentations).",
        columns=[
            _col("start_time", "Interval start (s)."),
            _col("stop_time", "Interval stop (s)."),
            _col(
                "interval_type",
                "One of: epoch, trial, change_window, response_window, "
                "stimulus_presentation, movie_frame.",
            ),
            _col(
                "label",
                "Descriptive label (epoch name; image_name for "
                "stimulus_presentation rows; empty otherwise).",
            ),
            _col(
                "trials_id",
                "Foreign key into the trials table (-1 if N/A).",
            ),
            _col(
                "stimulus_presentations_id",
                "Foreign key into the stimulus_presentations table (-1 if "
                "N/A).",
            ),
            _col(
                "natural_movie_one_presentations_id",
                "Foreign key into the natural_movie_one_presentations table "
                "(-1 if N/A).",
            ),
            HedTags(
                name="HED",
                description="HED tag string for this interval.",
                data=[r["HED"] for r in rows],
            ),
        ],
        id=list(range(len(rows))),
    )


def add_intervals(
    nwb: NWBFile,
    intervals_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> NWBFile:
    """Add the flat ``intervals`` table to ``nwb``.

    Parameters
    ----------
    nwb : the NWBFile to add to (modified in place).
    intervals_df : the flat intervals frame.
    events_df : the events frame.

    Returns
    -------
    The same ``nwb`` instance, for chaining.
    """
    nwb.add_time_intervals(build_intervals_table(intervals_df, events_df))
    return nwb
