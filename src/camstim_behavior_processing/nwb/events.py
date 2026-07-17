"""Write the discrete point-events into an ndx-events ``EventsTable``.

Consumes the ``events_df`` produced by
:func:`camstim_behavior_processing.load_data.trials_events.build_trials_and_events`
and builds an :class:`~ndx_events.EventsTable` of the *discrete* events only —
``lick``, ``reward``, ``image_change``, ``image_omission``. Visual on/offsets
are intervals (they live on the intervals / stimulus_presentations tables)
and ``miss`` is a trial-level outcome (on ``nwb.trials``), so both are
dropped here.

Orthogonal categorical columns follow HED's PASS design: a base ``event_type``
plus independent ``lick_classification`` / ``reward_type`` / ``lick_bouts``
dimensions, each backed by its own :class:`~ndx_events.MeaningsTable` whose HED
fragments compose into the full tag string. No DataFrame computation happens
here beyond selecting/sorting rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ndx_events import (
    CategoricalVectorData,
    EventsTable,
    MeaningsTable,
    NdxEventsNWBFile,
)

from . import hed_tags as H


def _build_meanings(
    name: str,
    table_desc: str,
    hed_dict: dict,
    desc_dict: dict,
) -> MeaningsTable:
    """Build one :class:`~ndx_events.MeaningsTable` for a categorical column.

    Parameters
    ----------
    name : neurodata name for the table.
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
    mt = MeaningsTable(name=name, description=table_desc)
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


def _all_meanings() -> list[MeaningsTable]:
    """Build the four categorical MeaningsTables for the events table.

    Returns
    -------
    ``[event_type, lick_classification, reward_type, lick_bouts]`` meanings
    tables, in the order the events table declares its columns.
    """
    return [
        _build_meanings(
            "event_type_meanings",
            "Base HED tag string and semantic description for each event "
            "type.",
            H.EVENT_TYPE_HED,
            H.EVENT_TYPE_DESC,
        ),
        _build_meanings(
            "lick_classification_meanings",
            "HED fragment and semantic description for the task context of a "
            "lick. Composes with the base lick tag from event_type_meanings.",
            H.LICK_CLASSIFICATION_HED,
            H.LICK_CLASSIFICATION_DESC,
        ),
        _build_meanings(
            "reward_type_meanings",
            "HED fragment and semantic description for reward type (earned "
            "vs auto_reward). Composes with the base reward tag from "
            "event_type_meanings.",
            H.REWARD_TYPE_HED,
            H.REWARD_TYPE_DESC,
        ),
        _build_meanings(
            "lick_bouts_meanings",
            "HED fragment and semantic description indicating whether a lick "
            "starts a bout (within-bout licks have no marker; bout-start "
            'licks carry a Temporal-marker tag). "n/a" for non-lick events.',
            H.LICK_BOUTS_HED,
            H.LICK_BOUTS_DESC,
        ),
    ]


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


def _declare_columns(et: EventsTable, meanings: list[MeaningsTable]) -> None:
    """Declare every EventsTable column (schema only, no data).

    Parameters
    ----------
    et : the events table to add columns to.
    meanings : the four MeaningsTables, in ``_all_meanings`` order.
    """
    cat = [
        (
            "event_type",
            "Event type: lick, reward, image_change, image_omission.",
        ),
        (
            "lick_classification",
            "Task context of a lick event (hit, false_alarm, abort, early, "
            'late, consumption, spontaneous). "n/a" for non-lick events.',
        ),
        (
            "reward_type",
            'Reward type (earned or auto_reward). "n/a" for non-reward '
            "events.",
        ),
        (
            "lick_bouts",
            "For lick events: bout_start vs within_bout. For non-lick "
            "events: n/a.",
        ),
    ]
    for (name, desc), meaning in zip(cat, meanings):
        et.add_column(
            name=name,
            description=desc,
            col_cls=CategoricalVectorData,
            meanings=meaning,
        )
    plain = [
        ("trials_id", "Id of trial this event belongs to (-1 if none)."),
        (
            "stimulus_presentations_id",
            "Id of stimulus presentation (-1 if none).",
        ),
        (
            "image_name",
            "Stimulus identity for stimulus events: image name "
            '(natural-image sessions) or "gratings_<ori>" (gratings '
            "sessions); empty otherwise.",
        ),
        (
            "orientation",
            "Grating orientation in degrees for gratings stimulus events "
            "(NaN otherwise).",
        ),
        ("frame", "Vsync falling-edge frame index (-1 if N/A)."),
        (
            "reward_volume",
            "Volume of reward in mL (NaN if not a reward).",
        ),
    ]
    for name, desc in plain:
        et.add_column(name=name, description=desc)


def _populate_columns(et: EventsTable, df: pd.DataFrame, cats: tuple) -> None:
    """Bulk-populate every EventsTable column from the events frame.

    Parameters
    ----------
    et : the events table with columns already declared.
    df : the filtered, timestamp-sorted events frame.
    cats : the ``(lick_cls, reward_tp, bout)`` tuple from
        :func:`_categorical_columns`.
    """
    lick_cls, reward_tp, bout = cats
    n = len(df)
    et.timestamp.data.extend(df["timestamp"].astype(float).tolist())
    et.id.data.extend(list(range(n)))
    et["event_type"].data.extend(df["event_type"].tolist())
    et["lick_classification"].data.extend(lick_cls)
    et["reward_type"].data.extend(reward_tp)
    et["lick_bouts"].data.extend(bout)
    et["trials_id"].data.extend(
        [int(v) if pd.notna(v) else -1 for v in df["trials_id"]]
    )
    et["stimulus_presentations_id"].data.extend(
        [
            int(v) if pd.notna(v) else -1
            for v in df["stimulus_presentations_id"]
        ]
    )
    et["image_name"].data.extend(
        [str(v) if pd.notna(v) else "" for v in df["image_name"]]
    )
    et["orientation"].data.extend(
        [float(v) if pd.notna(v) else np.nan for v in df["orientation"]]
        if "orientation" in df.columns
        else [np.nan] * n
    )
    et["frame"].data.extend(
        [int(v) if pd.notna(v) else -1 for v in df["frame"]]
    )
    et["reward_volume"].data.extend(
        [float(v) if pd.notna(v) else np.nan for v in df["reward_volume"]]
    )


def build_events_table(
    events_df: pd.DataFrame,
) -> tuple[EventsTable, list[MeaningsTable]]:
    """Build the ndx-events ``EventsTable`` of discrete (point) events.

    Parameters
    ----------
    events_df : the events frame from ``build_trials_and_events``. Rows whose
        ``event_type`` is in
        :data:`~camstim_behavior_processing.nwb.hed_tags.DROP_EVENT_TYPES`
        (visual on/offsets and ``miss``) are excluded.

    Returns
    -------
    ``(events_table, meanings_tables)`` — the populated ``EventsTable`` and the
    four backing ``MeaningsTable`` objects.
    """
    df = events_df[~events_df["event_type"].isin(H.DROP_EVENT_TYPES)]
    df = df.sort_values("timestamp").reset_index(drop=True)

    meanings = _all_meanings()
    et = EventsTable(
        name="events",
        description="All behavior and task events with orthogonal "
        "categorical columns following HED PASS design. event_type gives "
        "the base action; lick_classification, reward_type, and lick_bouts "
        "provide independent dimensions that compose into the full HED "
        "string.",
        meanings_tables=meanings,
    )
    _declare_columns(et, meanings)
    _populate_columns(et, df, _categorical_columns(df))
    return et, meanings


def add_events(
    nwb: NdxEventsNWBFile, events_df: pd.DataFrame
) -> NdxEventsNWBFile:
    """Build the events table and add it to ``nwb``.

    Parameters
    ----------
    nwb : the :class:`~ndx_events.NdxEventsNWBFile` to add the table to
        (modified in place).
    events_df : the events frame from ``build_trials_and_events``.

    Returns
    -------
    The same ``nwb`` instance, for chaining.
    """
    et, _ = build_events_table(events_df)
    nwb.add_events_table(et)
    return nwb
