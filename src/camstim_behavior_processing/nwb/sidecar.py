"""Build the BIDS-style events sidecar JSON.

Emits a single JSON-serialisable dict describing every column used across the
events, intervals, trials, stimulus_presentations, and
natural_movie_one_presentations tables — categorical columns get
``Description`` / ``Levels`` / ``HED`` blocks, descriptive value/id columns get
a ``Description`` and (where defined) a ``HED`` template with a ``#``
placeholder. All content is sourced from
:mod:`camstim_behavior_processing.nwb.hed_tags`.
"""

from __future__ import annotations

from . import hed_tags as H


def _categorical(desc_dict: dict, hed_dict: dict, column_desc: str) -> dict:
    """Build a categorical column entry (Description + Levels + HED).

    Parameters
    ----------
    desc_dict : value -> plain-English description (the ``Levels`` block).
    hed_dict : value -> HED fragment (empty fragments are dropped).
    column_desc : the column-level ``Description``.

    Returns
    -------
    The sidecar entry dict for the column.
    """
    return {
        "Description": column_desc,
        "Levels": dict(desc_dict),
        "HED": {k: v for k, v in hed_dict.items() if v},
    }


def build_events_sidecar() -> dict:
    """Build the BIDS-style sidecar JSON for every table column.

    Returns
    -------
    A JSON-serialisable dict. ``package_nwb`` writes it to
    ``<output>.events.json`` alongside the NWB.
    """
    sidecar: dict = {}

    sidecar["event_type"] = _categorical(
        H.EVENT_TYPE_DESC,
        H.EVENT_TYPE_HED,
        H.VALUE_COLUMN_DESC["event_type"],
    )
    sidecar["lick_classification"] = _categorical(
        H.LICK_CLASSIFICATION_DESC,
        H.LICK_CLASSIFICATION_HED,
        H.VALUE_COLUMN_DESC["lick_classification"],
    )
    sidecar["reward_type"] = _categorical(
        H.REWARD_TYPE_DESC,
        H.REWARD_TYPE_HED,
        H.VALUE_COLUMN_DESC["reward_type"],
    )
    sidecar["lick_bouts"] = _categorical(
        H.LICK_BOUTS_DESC,
        H.LICK_BOUTS_HED,
        H.VALUE_COLUMN_DESC["lick_bouts"],
    )
    sidecar["interval_type"] = {
        "Description": H.VALUE_COLUMN_DESC["interval_type"],
        "Levels": dict(H.INTERVAL_TYPE_DESC),
        # No per-interval-type base HED - HED on intervals is composed from
        # the interval contents (epoch name, trial outcome, image_name, ...).
    }
    sidecar["epoch_name"] = _categorical(
        H.EPOCH_DESC, H.EPOCH_HED, H.VALUE_COLUMN_DESC["epoch_name"]
    )
    # Trial outcome is not a stored column but a derived discriminator; it is
    # included so the per-outcome HED on nwb.trials is documented.
    sidecar["trial_outcome"] = _categorical(
        H.TRIAL_OUTCOME_DESC,
        H.TRIAL_OUTCOME_HED,
        "Composite outcome of a trial, derived from the boolean go/catch/"
        "hit/miss/false_alarm/correct_reject/aborted/auto_rewarded columns. "
        "Used to compose the per-trial HED tag.",
    )

    # Descriptive value / id / metadata columns not already added above.
    for col, desc in H.VALUE_COLUMN_DESC.items():
        if col in sidecar:
            continue
        entry: dict = {"Description": desc}
        if col in H.VALUE_COLUMN_HED:
            entry["HED"] = H.VALUE_COLUMN_HED[col]
        sidecar[col] = entry

    # HED Definitions block. Reserved for any (Definition/...) macros
    # referenced above; none are needed since all tags use the base HED
    # v8.3.0 schema, but the block matches the reference sidecar format.
    sidecar["hed_defs"] = {"HED": {"alldefs": ""}}
    return sidecar
