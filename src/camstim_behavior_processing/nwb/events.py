"""Write the discrete point-events into a pynwb ``EventsTable``.

Consumes the ``events_df`` produced by
:func:`camstim_behavior_processing.load_data.trials_events.build_trials_and_events`
and builds a :class:`~pynwb.event.EventsTable` of the *discrete* events only —
``lick``, ``reward``, ``image_change``, ``image_omission``. Visual on/offsets
are intervals (they live on the intervals / stimulus_presentations tables)
and ``miss`` is a trial-level outcome (on ``nwb.trials``), so both are
dropped here.

Orthogonal categorical columns follow HED's PASS design: a base ``event_type``
plus independent ``lick_classification`` / ``reward_type`` / ``lick_bouts``
dimensions, each backed by its own :class:`~hdmf.common.MeaningsTable` whose
HED fragments compose into the full tag string. No DataFrame computation
happens here beyond selecting/sorting rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hdmf.common import MeaningsTable
from hdmf.common.table import VectorData
from pynwb import NWBFile
from pynwb.event import EventsTable, TimestampVectorData

from . import hed_tags as H


def _build_meanings(
    col: VectorData,
    table_desc: str,
    hed_dict: dict,
    desc_dict: dict,
) -> MeaningsTable:
    """Build one :class:`~hdmf.common.MeaningsTable` for a categorical column.

    Parameters
    ----------
    col : the VectorData column this table describes (already a member of
        the EventsTable; the MeaningsTable name is derived from it as
        ``"{col.name}_meanings"``).
    table_desc : table-level description.
    hed_dict : value -> HED fragment.
    desc_dict : value -> plain-English description.

    Returns
    -------
    The populated ``MeaningsTable`` with a ``value_description`` column.

    Notes
    -----
    The extra column is named ``value_description`` rather than ``description``
    because ``MeaningsTable`` already carries a table-level ``description``
    attribute; reusing the name collides at HDF5 write time.
    """
    mt = MeaningsTable(target=col, description=table_desc)
    mt.add_column(
        name="value_description",
        description="Plain-English semantic description of the value.",
    )
    for value, hed in hed_dict.items():
        mt.add_row(
            value=value,
            meaning=hed,
            value_description=desc_dict.get(value, ""),
        )
    return mt


def _categorical_columns(df: pd.DataFrame) -> tuple[list, list, list]:
    """Derive the per-row lick/reward categorical values.

    Parameters
    ----------
    df : the filtered, timestamp-sorted events frame.

    Returns
    -------
    ``(lick_classification, reward_type, lick_bouts)`` lists, one entry per
    row, with ``"n/a"`` for rows the dimension does not apply to.
    """
    n = len(df)
    is_lick = (df["event_type"] == "lick").values
    is_reward = (df["event_type"] == "reward").values

    lick_cls = ["n/a"] * n
    reward_tp = ["n/a"] * n
    bout = ["n/a"] * n
    for i, (_, row) in enumerate(df.iterrows()):
        if is_lick[i]:
            if pd.notna(row.get("lick_classification")):
                lick_cls[i] = str(row["lick_classification"])
            bout[i] = (
                "bout_start" if bool(row.get("bout_start")) else "within_bout"
            )
        elif is_reward[i] and pd.notna(row.get("reward_type")):
            reward_tp[i] = str(row["reward_type"])
    return lick_cls, reward_tp, bout


def _build_columns(
    df: pd.DataFrame,
    lick_cls: list,
    reward_tp: list,
    bout: list,
) -> list:
    """Build all VectorData column objects for the EventsTable.

    Parameters
    ----------
    df : the filtered, timestamp-sorted events frame.
    lick_cls : per-row lick-classification values.
    reward_tp : per-row reward-type values.
    bout : per-row lick-bout values.

    Returns
    -------
    Ordered list: ``[timestamp, event_type, lick_classification,
    reward_type, lick_bouts, trials_id, stimulus_presentations_id,
    image_name, orientation, frame, reward_volume]``.
    """
    n = len(df)
    ori = (
        [float(v) if pd.notna(v) else np.nan for v in df["orientation"]]
        if "orientation" in df.columns
        else [np.nan] * n
    )
    return [
        TimestampVectorData(
            name="timestamp",
            description=(
                "Event time in seconds from session start, aligned to the "
                "sync-file hardware clock."
            ),
            data=df["timestamp"].astype(float).tolist(),
        ),
        VectorData(
            name="event_type",
            description=(
                "Event type: lick, reward, image_change, image_omission."
            ),
            data=df["event_type"].tolist(),
        ),
        VectorData(
            name="lick_classification",
            description=(
                "Task context of a lick event (hit, false_alarm, abort, "
                'early, late, consumption, spontaneous). "n/a" for non-lick '
                "events."
            ),
            data=lick_cls,
        ),
        VectorData(
            name="reward_type",
            description=(
                'Reward type (earned or auto_reward). "n/a" for non-reward '
                "events."
            ),
            data=reward_tp,
        ),
        VectorData(
            name="lick_bouts",
            description=(
                "For lick events: bout_start vs within_bout. For non-lick "
                "events: n/a."
            ),
            data=bout,
        ),
        VectorData(
            name="trials_id",
            description="Id of trial this event belongs to (-1 if none).",
            data=[int(v) if pd.notna(v) else -1 for v in df["trials_id"]],
        ),
        VectorData(
            name="stimulus_presentations_id",
            description="Id of stimulus presentation (-1 if none).",
            data=[
                int(v) if pd.notna(v) else -1
                for v in df["stimulus_presentations_id"]
            ],
        ),
        VectorData(
            name="image_name",
            description=(
                "Stimulus identity for stimulus events: image name "
                '(natural-image sessions) or "gratings_<ori>" (gratings '
                "sessions); empty otherwise."
            ),
            data=[
                str(v) if pd.notna(v) else "" for v in df["image_name"]
            ],
        ),
        VectorData(
            name="orientation",
            description=(
                "Grating orientation in degrees for gratings stimulus events "
                "(NaN otherwise)."
            ),
            data=ori,
        ),
        VectorData(
            name="frame",
            description="Vsync falling-edge frame index (-1 if N/A).",
            data=[int(v) if pd.notna(v) else -1 for v in df["frame"]],
        ),
        VectorData(
            name="reward_volume",
            description="Volume of reward in mL (NaN if not a reward).",
            data=[
                float(v) if pd.notna(v) else np.nan
                for v in df["reward_volume"]
            ],
        ),
    ]


def _attach_meanings(
    et: EventsTable,
    evt_col: VectorData,
    lick_col: VectorData,
    reward_col: VectorData,
    bout_col: VectorData,
) -> list[MeaningsTable]:
    """Build the four MeaningsTables and attach them to ``et``.

    Parameters
    ----------
    et : the EventsTable to attach meanings to (modified in place).
    evt_col : the ``event_type`` VectorData column.
    lick_col : the ``lick_classification`` VectorData column.
    reward_col : the ``reward_type`` VectorData column.
    bout_col : the ``lick_bouts`` VectorData column.

    Returns
    -------
    ``[event_type, lick_classification, reward_type, lick_bouts]`` meanings
    tables, in declaration order.
    """
    specs = [
        (
            evt_col,
            "Base HED tag string and semantic description for each event "
            "type.",
            H.EVENT_TYPE_HED,
            H.EVENT_TYPE_DESC,
        ),
        (
            lick_col,
            "HED fragment and semantic description for the task context of a "
            "lick. Composes with the base lick tag from "
            "event_type_meanings.",
            H.LICK_CLASSIFICATION_HED,
            H.LICK_CLASSIFICATION_DESC,
        ),
        (
            reward_col,
            "HED fragment and semantic description for reward type (earned "
            "vs auto_reward). Composes with the base reward tag from "
            "event_type_meanings.",
            H.REWARD_TYPE_HED,
            H.REWARD_TYPE_DESC,
        ),
        (
            bout_col,
            "HED fragment and semantic description indicating whether a lick "
            "starts a bout (within-bout licks have no marker; bout-start "
            'licks carry a Temporal-marker tag). "n/a" for non-lick events.',
            H.LICK_BOUTS_HED,
            H.LICK_BOUTS_DESC,
        ),
    ]
    meanings = []
    for col, desc, hed_dict, desc_dict in specs:
        mt = _build_meanings(col, desc, hed_dict, desc_dict)
        et.add_meanings_table(mt)
        meanings.append(mt)
    return meanings


def build_events_table(
    events_df: pd.DataFrame,
) -> tuple[EventsTable, list[MeaningsTable]]:
    """Build the pynwb ``EventsTable`` of discrete (point) events.

    Parameters
    ----------
    events_df : the events frame from ``build_trials_and_events``. Rows whose
        ``event_type`` is in
        :data:`~camstim_behavior_processing.nwb.hed_tags.DROP_EVENT_TYPES`
        (visual on/offsets and ``miss``) are excluded.

    Returns
    -------
    ``(events_table, meanings_tables)`` — the populated ``EventsTable`` and
    the four backing ``MeaningsTable`` objects.
    """
    df = events_df[~events_df["event_type"].isin(H.DROP_EVENT_TYPES)]
    df = df.sort_values("timestamp").reset_index(drop=True)

    cats = _categorical_columns(df)
    cols = _build_columns(df, *cats)
    et = EventsTable(
        name="events",
        description=(
            "All behavior and task events with orthogonal categorical "
            "columns following HED PASS design. event_type gives the base "
            "action; lick_classification, reward_type, and lick_bouts "
            "provide independent dimensions that compose into the full HED "
            "string."
        ),
        columns=cols,
        id=list(range(len(df))),
    )
    meanings = _attach_meanings(et, cols[1], cols[2], cols[3], cols[4])
    return et, meanings


def add_events(
    nwb: NWBFile, events_df: pd.DataFrame
) -> NWBFile:
    """Build the events table and add it to ``nwb``.

    Parameters
    ----------
    nwb : the :class:`~pynwb.NWBFile` to add the table to (modified in
        place).
    events_df : the events frame from ``build_trials_and_events``.

    Returns
    -------
    The same ``nwb`` instance, for chaining.
    """
    et, _ = build_events_table(events_df)
    nwb.add_events_table(et)
    return nwb
