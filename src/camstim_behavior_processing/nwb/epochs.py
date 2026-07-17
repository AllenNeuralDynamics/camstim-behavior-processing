"""Build the canonical session-epoch list from the intervals/events tables.

The epoch list is a small, plain list of ``{name, start, stop}`` dicts used to
tag every trial / stimulus presentation / interval row with the session epoch
(``change_detection``, ``natural_movie_one``, ``spontaneous``) that contains
it. ``nwb.epochs`` is intentionally not set by the packaging layer — epochs
live as ``interval_type='epoch'`` rows on the flat intervals table instead.
"""

from __future__ import annotations

import pandas as pd


def build_epoch_lookup(
    intervals_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> list[dict]:
    """Build the canonical session epoch list.

    - Folds ``warm_up`` rows into ``change_detection`` (start = warm_up.start).
    - Inserts ``spontaneous`` rows to fill any gap before the first epoch,
      between epochs, or after the last epoch (up to the latest timestamp in
      the events or intervals dataframes).

    Parameters
    ----------
    intervals_df : the flat intervals frame; ``interval_type='epoch'`` rows
        (with a ``label`` of ``warm_up`` / ``change_detection`` /
        ``natural_movie_one``) seed the named epochs.
    events_df : the events frame, used only for its max ``timestamp`` when
        computing the session end.

    Returns
    -------
    list of ``{name, start, stop}`` dicts, sorted by start.
    """
    raw = intervals_df[intervals_df["interval_type"] == "epoch"].sort_values(
        "start_time"
    )

    named = _named_epochs(raw)

    # Session end = max timestamp across events + intervals.
    session_end = max(
        float(events_df["timestamp"].max()),
        float(intervals_df["stop_time"].max()),
    )

    out: list[dict] = []
    prev = 0.0
    for ep in named:
        if ep["start"] - prev > 1e-6:
            out.append(
                {"name": "spontaneous", "start": prev, "stop": ep["start"]}
            )
        out.append(ep)
        prev = ep["stop"]
    if session_end - prev > 1e-6:
        out.append({"name": "spontaneous", "start": prev, "stop": session_end})
    return out


def _named_epochs(raw: pd.DataFrame) -> list[dict]:
    """Extract the named (non-spontaneous) epochs from the epoch rows.

    Parameters
    ----------
    raw : the ``interval_type='epoch'`` rows of the intervals frame.

    Returns
    -------
    list of ``{name, start, stop}`` dicts for ``change_detection`` (with any
    ``warm_up`` folded into its start) and ``natural_movie_one``, sorted by
    start.
    """
    labels = raw["label"].values
    named: list[dict] = []
    if "change_detection" in labels:
        cd = raw[raw["label"] == "change_detection"].iloc[0]
        cd_start = float(cd["start_time"])
        cd_stop = float(cd["stop_time"])
        # Fold warm_up in if present (it always precedes change_detection).
        if "warm_up" in labels:
            wu = raw[raw["label"] == "warm_up"].iloc[0]
            cd_start = float(wu["start_time"])
        named.append(
            {
                "name": "change_detection",
                "start": cd_start,
                "stop": cd_stop,
            }
        )
    if "natural_movie_one" in labels:
        nm = raw[raw["label"] == "natural_movie_one"].iloc[0]
        named.append(
            {
                "name": "natural_movie_one",
                "start": float(nm["start_time"]),
                "stop": float(nm["stop_time"]),
            }
        )
    named.sort(key=lambda e: e["start"])
    return named


def epoch_name_at(t: float, epoch_list: list[dict]) -> str:
    """Return the epoch name containing time ``t``.

    Parameters
    ----------
    t : a time in seconds.
    epoch_list : the list from :func:`build_epoch_lookup`.

    Returns
    -------
    The name of the containing epoch, or ``"spontaneous"`` if none contains
    ``t``.
    """
    for ep in epoch_list:
        if ep["start"] <= t < ep["stop"]:
            return ep["name"]
    return "spontaneous"
