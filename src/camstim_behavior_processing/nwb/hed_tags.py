"""HED tag fragments and semantic descriptions for the NWB writers.

Single source of truth for every HED (Hierarchical Event Descriptor) tag
string and plain-English description used by the NWB-packaging layer. The
:mod:`~camstim_behavior_processing.nwb` writers compose the full per-row HED
strings from these fragments following HED's PASS design (a base
``event_type`` tag plus orthogonal ``lick_classification`` / ``reward_type`` /
``lick_bouts`` fragments), and :mod:`~camstim_behavior_processing.nwb.sidecar`
exports them as the BIDS-style events sidecar.

All tags are validated against the HED v8.3.0 base schema
(:data:`HED_SCHEMA_VERSION`). No NWB or DataFrame dependency lives here — just
constants and two pure string helpers.
"""

from __future__ import annotations

import re

import pandas as pd

#: HED base schema version every tag in this module is written against.
HED_SCHEMA_VERSION = "8.3.0"

# ── HED fragments ──────────────────────────────────────────────────────
# Orthogonal design: a base event_type HED plus independent classification /
# reward_type / bout fragments that compose into the full tag string.

# The events table holds discrete (point) events only. Visual onsets/offsets
# (image_onset/image_offset/movie_onset/movie_offset) live in the intervals
# table as interval_type='stimulus_presentation' / 'movie_frame'. Miss is a
# trial-level outcome (annotated on nwb.trials), not a discrete event.
EVENT_TYPE_HED = {
    "lick": (
        "Agent-action, (Animal-agent, Move-face), "
        "Participant-response, Label/lick"
    ),
    "reward": (
        "Sensory-event, Gustatory-presentation, "
        "(Ingestible-object, Reward), Label/water"
    ),
    "image_change": (
        "Sensory-event, Visual-presentation, Target, Label/image_change"
    ),
    "image_omission": "Sensory-event, Unexpected, Label/omitted_flash",
}

# Event types from build_events_and_intervals.py that don't belong on the
# events table: intervals belong on the intervals table; 'miss' is a trial
# outcome (already on nwb.trials).
DROP_EVENT_TYPES = {
    "image_onset",
    "image_offset",
    "movie_onset",
    "movie_offset",
    "miss",
}

LICK_CLASSIFICATION_HED = {
    "hit": "Correct-action, Label/hit",
    "false_alarm": "Incorrect-action, Label/false_alarm",
    "abort": "Incorrect-action, Label/abort",
    "early": "Incorrect-action, Label/early",
    "late": "Incorrect-action, Label/late",
    "consumption": "Label/consumption",
    "spontaneous": "Label/spontaneous",
    "n/a": "",
}

REWARD_TYPE_HED = {
    "earned": "Label/earned",
    "auto_reward": "Label/auto_reward",
    "n/a": "",
}

LICK_BOUTS_HED = {
    "bout_start": "(Temporal-marker, Label/bout_start)",
    "within_bout": "",
    "n/a": "",
}

# Per-trial HED — composed from outcome flags.
TRIAL_OUTCOME_HED = {
    "hit": "Experimental-trial, Target, Correct-action, Label/hit_trial",
    "miss": "Experimental-trial, Target, Miss, Label/miss_trial",
    "false_alarm": (
        "Experimental-trial, Non-target, Incorrect-action, "
        "Label/false_alarm_trial"
    ),
    "correct_reject": (
        "Experimental-trial, Non-target, Correct-action, "
        "Label/correct_reject_trial"
    ),
    "aborted": ("Experimental-trial, Incorrect-action, Label/aborted_trial"),
    "auto_rewarded": "Experimental-trial, Reward, Label/auto_rewarded_trial",
    "no_outcome": "Experimental-trial",
}

EPOCH_HED = {
    "change_detection": (
        "Time-block, Experiment-procedure, Label/change_detection_task"
    ),
    "natural_movie_one": (
        "Time-block, Sensory-event, Visual-presentation, "
        "(Movie, Label/natural_movie_one), Label/fingerprint_epoch"
    ),
    "spontaneous": "Time-block, Pause, Label/spontaneous",
}

#: HED tag string for a single natural_movie_one frame presentation.
MOVIE_FRAME_HED = (
    "Sensory-event, Visual-presentation, (Movie, Label/natural_movie_one)"
)


# ── Semantic descriptions ──────────────────────────────────────────────
# Plain-English meanings for each categorical value. These get inserted into
# the MeaningsTables as a `value_description` column and exported as the
# `Levels` block of the sidecar JSON. Keys must match the *_HED dicts above.
EVENT_TYPE_DESC = {
    "lick": (
        "Time of a lick contact on the lick spout, detected by the lick "
        "sensor."
    ),
    "reward": "Time of a water-reward delivery to the lick spout.",
    "image_change": (
        "Time of a stimulus presentation where the identity of the "
        "presented image is distinct from the previously presented image; "
        "demarcates Go trials where the mouse can earn rewards for licks "
        "within the post-change reward window."
    ),
    "image_omission": (
        "Time of a scheduled image flash that was withheld (no image shown) "
        "in slots where a flash would otherwise have occurred."
    ),
}

LICK_CLASSIFICATION_DESC = {
    "hit": (
        "First lick within the response window after an image change on a "
        "go trial."
    ),
    "false_alarm": (
        "Incorrect lick occurring during the window of time where the image "
        "could have changed (based on the change trial distribution) but did "
        "not change; triggers reset of trial."
    ),
    "abort": (
        "Incorrect lick occurring in the 4 flash period after the start of a "
        "trial, prior to the change window onset; triggers reset of trial."
    ),
    "early": (
        "Unrewarded lick in the 150 ms window after the image change onset "
        "but prior to the reward window onset; considered as too early to be "
        "a valid response to the image change."
    ),
    "late": (
        "Lick after the response window has closed on a trial with a change "
        "but no reward (i.e., a missed change)."
    ),
    "consumption": (
        "Lick after a reward delivery on the same trial when the mouse is "
        "consuming the water reward."
    ),
    "spontaneous": (
        "Lick outside of any task-defined window (e.g., between trials, "
        "during warm-up, during inter-trial intervals)."
    ),
    "n/a": "Not a lick event; this column does not apply.",
}

REWARD_TYPE_DESC = {
    "earned": (
        "Reward delivered as a consequence of a correct lick (hit) on a go "
        "trial."
    ),
    "auto_reward": (
        "Reward delivered automatically by the task (e.g., during warm-up "
        "or auto-rewarded trials) regardless of the animal's response."
    ),
    "n/a": "Not a reward event; this column does not apply.",
}

LICK_BOUTS_DESC = {
    "bout_start": (
        "First lick of a bout: the inter-lick interval from the previous "
        "lick exceeds 500 ms (or this is the first lick of the session)."
    ),
    "within_bout": (
        "Lick that occurs within an ongoing bout (inter-lick interval from "
        "previous lick <= 500 ms)."
    ),
    "n/a": "Not a lick event; this column does not apply.",
}

TRIAL_OUTCOME_DESC = {
    "hit": (
        "Go trial in which the animal licked within the response window "
        "after the image change - a correct detection."
    ),
    "miss": (
        "Go trial in which the animal failed to lick within the response "
        "window after the image change."
    ),
    "false_alarm": (
        "Catch trial in which the animal licked within the response window "
        "despite no image change occurring."
    ),
    "correct_reject": (
        "Catch trial in which the animal correctly withheld licking during "
        "the response window."
    ),
    "aborted": (
        "Trial in which the animal licked before the change window opened, "
        "aborting the trial."
    ),
    "auto_rewarded": (
        "Trial on which the task automatically delivered a reward regardless "
        "of the animal's response (e.g., warm-up trials, reminder trials)."
    ),
    "no_outcome": (
        "Trial that did not produce any of the standard outcome categories "
        "(rare; typically a configuration/edge case)."
    ),
}

EPOCH_DESC = {
    "change_detection": (
        "Active change-detection task period during which mice can earn "
        "water rewards for licking after changes in image identity. Includes "
        "warm-up trials at the start of session, contingent reward trials "
        "throughout, and auto-reward reminders during disengaged periods."
    ),
    "natural_movie_one": (
        'Passive viewing of the "natural_movie_one" fingerprint movie clip, '
        "shown after the active task to identify recorded cells across "
        "sessions."
    ),
    "spontaneous": (
        "Any time outside of the named task epochs - gray-screen periods "
        "before, between, or after the change_detection and "
        "natural_movie_one epochs."
    ),
}

# Interval types in the canonical flat intervals table.
INTERVAL_TYPE_DESC = {
    "epoch": (
        "A session-level epoch (e.g., change_detection, natural_movie_one, "
        "spontaneous). The epoch name is carried in the `label` column."
    ),
    "trial": (
        "A single behavioral trial in the change-detection task, bounded by "
        "trial_start and trial_end, defined by the change time distribution "
        "parameter (typically geometric between 4-12 flashes after trial "
        "start). Trials contain multiple stimulus presentations. Trial start "
        "time is un-cued. Annotations live on `nwb.trials`."
    ),
    "change_window": (
        "Window of time after trial start during which an image change may "
        "occur (as defined by the change time distribution), starting at the "
        "change_flashes_min-th flash and ending at the change (or, for "
        "catch/aborted trials, at the change_flashes_max-th flash or trial "
        "end)."
    ),
    "response_window": (
        "Window after the image change during which a lick counts as a hit "
        "(go trial) or false alarm (catch trial). Defined by the "
        "response_window task parameter relative to change_time."
    ),
    "stimulus_presentation": (
        "A single image flash presentation (or omitted slot) during the "
        "active task. The image name is carried in the `label` column."
    ),
    "movie_frame": (
        "A single frame presentation of the natural_movie_one fingerprint "
        "movie."
    ),
}

# HED-tag templates for descriptive value columns (continuous / id columns).
# These mirror the reference sidecar's per-column HED templates with #
# placeholders.
VALUE_COLUMN_HED = {
    "timestamp": "Time-value/# s",
    "start_time": "Time-value/# s",
    "stop_time": "Time-value/# s",
    "reward_volume": "Volume/# mL",
    "frame": "Label/frame-#",
    "trials_id": "Label/trial-#",
    "stimulus_presentations_id": "Label/stimulus_presentation-#",
    "natural_movie_one_presentations_id": "Label/movie_frame-#",
    "movie_frame_index": "Label/movie_frame_index-#",
    "movie_repeat": "Label/movie_repeat-#",
    "change_frame": "Label/change_frame-#",
    "change_time": "Time-value/# s",
    "reward_time": "Time-value/# s",
    "response_time": "Time-value/# s",
    "response_latency": "Time-value/# s",
    "lick_latency": "Time-value/# s",
    "change_window_start_time": "Time-value/# s",
    "change_window_stop_time": "Time-value/# s",
    "response_window_start_time": "Time-value/# s",
    "response_window_stop_time": "Time-value/# s",
    "start_frame": "Label/frame-#",
    "stop_frame": "Label/frame-#",
}

# Descriptions for descriptive value / id / metadata columns that don't have
# a Levels-style enumeration.
VALUE_COLUMN_DESC = {
    "timestamp": (
        "Event time, in seconds from session start, aligned to the "
        "sync-file hardware clock. Visual events include the measured "
        "monitor delay; behavioral events (licks, rewards) do not."
    ),
    "start_time": "Interval start time, in seconds from session start.",
    "stop_time": "Interval stop time, in seconds from session start.",
    "reward_volume": (
        "Volume of water delivered for this reward, in millilitres."
    ),
    "frame": (
        "Vsync falling-edge frame index (into the sync-file frame array) "
        "corresponding to this event's timestamp. -1 if not applicable."
    ),
    "start_frame": "Vsync falling-edge frame index for the interval start.",
    "stop_frame": "Vsync falling-edge frame index for the interval stop.",
    "trials_id": (
        "Foreign key into the trials table - index of the trial this row "
        "belongs to. -1 if the row falls outside any trial."
    ),
    "stimulus_presentations_id": (
        "Foreign key into the stimulus_presentations table. -1 if not "
        "applicable."
    ),
    "natural_movie_one_presentations_id": (
        "Foreign key into the natural_movie_one_presentations table. -1 if "
        "not applicable."
    ),
    "movie_frame_index": (
        "Frame index within a single playback of natural_movie_one (0-based, "
        "0-899 for a 30-second clip at 30 Hz)."
    ),
    "movie_repeat": (
        "Repetition number of natural_movie_one (0-based) for this frame."
    ),
    "image_name": (
        'Identifier of the stimulus presented: an image name (e.g., "im065") '
        'for natural-image sessions, "gratings_<ori>" (e.g., "gratings_90") '
        'for gratings sessions, "natural_movie_one" for movie frames, or '
        '"omitted" for withheld flashes.'
    ),
    "orientation": (
        "Grating orientation in degrees for gratings-session stimulus rows. "
        "NaN for natural-image sessions, movie frames, and omitted flashes."
    ),
    "label": (
        "Descriptive label for this row. Carries the epoch name for "
        "interval_type=epoch rows and the image_name for "
        "interval_type=stimulus_presentation rows; empty otherwise."
    ),
    "change_time": (
        "Time of the image change on this trial, in seconds. NaN if no "
        "change occurred (catch / aborted trials)."
    ),
    "change_frame": (
        "Vsync falling-edge frame index of the image change. -1 if no "
        "change occurred."
    ),
    "reward_time": (
        "Time of reward delivery on this trial, in seconds. NaN if no reward "
        "was delivered."
    ),
    "response_time": (
        "Time of the first lick after the image change on this trial. NaN if "
        "the animal did not lick."
    ),
    "response_latency": (
        "Latency from change_time to response_time on this trial, in "
        "seconds. NaN if no response."
    ),
    "lick_latency": (
        "Time, in seconds, from the most recent image_onset to this lick "
        "(events table) or from this stimulus onset to the next lick before "
        "the following presentation (stimulus_presentations table)."
    ),
    "initial_image_name": (
        "Stimulus identity shown at the start of the trial (before any "
        'change): image name, or "gratings_<ori>" for gratings sessions.'
    ),
    "change_image_name": (
        "Stimulus identity shown after the change. Equal to "
        "initial_image_name for catch / aborted trials where no change "
        "occurred."
    ),
    "initial_orientation": (
        "Grating orientation in degrees at the start of the trial. NaN for "
        "natural-image sessions."
    ),
    "change_orientation": (
        "Grating orientation in degrees after the change. Equal to "
        "initial_orientation for catch / aborted trials. NaN for "
        "natural-image sessions."
    ),
    "change_window_start_time": (
        "Start time of the trial's change window, in seconds. NaN if "
        "undefined."
    ),
    "change_window_stop_time": (
        "Stop time of the trial's change window, in seconds. NaN if "
        "undefined."
    ),
    "response_window_start_time": (
        "Start time of the trial's response window, in seconds. NaN if no "
        "change occurred."
    ),
    "response_window_stop_time": (
        "Stop time of the trial's response window, in seconds. NaN if no "
        "change occurred."
    ),
    "epoch_name": (
        "Canonical name of the session epoch containing this row. One of "
        "change_detection, natural_movie_one, spontaneous."
    ),
    "is_change": (
        "True if this image flash differs in identity from the previous "
        "(non-omitted) flash - i.e., this is the target stimulus on a go "
        "trial."
    ),
    "omitted": (
        "True if this row is a withheld (omitted) flash slot rather than an "
        "actually-displayed image."
    ),
    "go": (
        "True if this trial was a go trial (image change scheduled, not "
        "auto-rewarded, not aborted)."
    ),
    "catch": (
        "True if this trial was a catch trial (no image change scheduled)."
    ),
    "auto_rewarded": (
        "True if this trial automatically delivered a reward (e.g., warm-up "
        "trials)."
    ),
    "aborted": (
        "True if this trial was aborted by a lick before the change window "
        "opened."
    ),
    "hit": (
        "True if the animal licked within the response window on a go trial."
    ),
    "miss": (
        "True if the animal failed to lick within the response window on a "
        "go trial."
    ),
    "false_alarm": (
        "True if the animal licked in the change or response window on a "
        "catch trial."
    ),
    "correct_reject": (
        "True if the animal correctly withheld licking on a catch trial."
    ),
    "warm_up": (
        "True for trials in the initial warm-up block (the first N "
        "auto-rewarded trials)."
    ),
    "interval_type": (
        "Discriminator for the kind of interval this row represents (epoch, "
        "trial, change_window, response_window, stimulus_presentation, "
        "movie_frame)."
    ),
    "event_type": (
        "Discriminator for the kind of point event this row represents "
        "(lick, reward, image_change, image_omission)."
    ),
    "lick_classification": (
        'Task-context classification of a lick event. "n/a" for non-lick '
        "events."
    ),
    "reward_type": (
        'Type of reward (earned vs auto_reward). "n/a" for non-reward '
        "events."
    ),
    "lick_bouts": (
        "Bout-structure marker for lick events (bout_start vs within_bout). "
        '"n/a" for non-lick events.'
    ),
    "HED": (
        "Hierarchical Event Descriptor (HED) tag string for this row, "
        "composed from the base event/interval tag and any orthogonal "
        "category fragments."
    ),
}


# ── HED composition helpers ────────────────────────────────────────────
def stim_presentation_hed(
    image_name: str, is_change: bool, omitted: bool
) -> str:
    """Compose the HED string for one stimulus presentation.

    Parameters
    ----------
    image_name : stimulus identity ("im065", "gratings_90", or "omitted").
    is_change : whether this flash is the change (target) flash.
    omitted : whether this is a withheld (omitted) flash slot.

    Returns
    -------
    The composed HED tag string. Gratings sessions (``gratings_*`` names)
    use the ``Grating`` visual-object tag; natural images use ``Image``.
    """
    if omitted:
        return "Sensory-event, Unexpected, Label/omitted_flash"
    obj_tag = "Grating" if str(image_name).startswith("gratings_") else "Image"
    base = (
        f"Sensory-event, Visual-presentation, "
        f"({obj_tag}, Label/{image_name})"
    )
    if is_change:
        base += ", Target"
    return base


def trial_hed(row: pd.Series) -> str:
    """Return the per-trial outcome HED string for a trial row.

    Parameters
    ----------
    row : a trial row with the boolean outcome columns (``hit``, ``miss``,
        ``false_alarm``, ``correct_reject``, ``aborted``, ``auto_rewarded``).

    Returns
    -------
    The HED string keyed off the first outcome flag that is set, falling
    back to the ``no_outcome`` tag.
    """
    if row.get("hit"):
        key = "hit"
    elif row.get("miss"):
        key = "miss"
    elif row.get("false_alarm"):
        key = "false_alarm"
    elif row.get("correct_reject"):
        key = "correct_reject"
    elif row.get("aborted"):
        key = "aborted"
    elif row.get("auto_rewarded"):
        key = "auto_rewarded"
    else:
        key = "no_outcome"
    return TRIAL_OUTCOME_HED[key]


# ── Passive SweepStim (movie / gratings) HED ───────────────────────────
# The passive path (sweepstim) presents movie/grating clips with no trial
# structure. Its interval types are only ``epoch`` and
# ``stimulus_presentation``; the epoch label is either a clip name, the
# ``passive_viewing`` fallback, or a ``spontaneous`` gap.
SWEEPSTIM_PASSIVE_TASK = "passive_viewing"

#: HED for a spontaneous (gray-screen) gap epoch in a passive session.
SWEEPSTIM_SPONTANEOUS_EPOCH_HED = (
    "Experimental-procedure, (Task, Label/spontaneous)"
)

#: HED for the whole-session ``passive_viewing`` fallback epoch (used when a
#: session declares no per-clip ``display_sequence`` windows).
SWEEPSTIM_PASSIVE_EPOCH_HED = (
    "Experimental-procedure, (Task, Label/passive_viewing)"
)

# Descriptions for the passive-session-specific columns, exported by the
# SweepStim sidecar builder.
SWEEPSTIM_COLUMN_DESC = {
    "movie_name": "Movie clip label.",
    "movie_frame_index": "Frame index within movie clip.",
    "movie_repeat": "Repeat index within clip.",
    "stim_block": "Stimulus block index from pkl top-level stimuli list.",
}

# HED templates (with ``#`` placeholders) for the passive-session columns.
SWEEPSTIM_COLUMN_HED = {
    "movie_frame_index": "Label/movie_frame_index-#",
    "movie_repeat": "Label/movie_repeat-#",
    "stim_block": "Label/stim_block-#",
}

# Levels for the passive-session ``interval_type`` discriminator.
SWEEPSTIM_INTERVAL_TYPE_DESC = {
    "epoch": "Session-level epoch row.",
    "stimulus_presentation": "Per-frame movie presentation.",
}


def hed_safe_label(name: str) -> str:
    """Return ``name`` with non-alphanumeric characters replaced by ``_``.

    HED ``Label/`` values may only contain word characters, so clip names
    (which come from Windows-style stimulus paths) are sanitised before use.

    Parameters
    ----------
    name : the raw label (e.g. a movie clip basename).

    Returns
    -------
    A HED-safe label string.
    """
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


def sweepstim_movie_hed(clip: str) -> str:
    """Compose the HED string for one passive movie/grating presentation.

    Parameters
    ----------
    clip : the movie clip label (carried in the ``movie_name`` column).

    Returns
    -------
    The composed HED tag string.
    """
    return (
        "Sensory-event, Visual-presentation, "
        f"(Movie, Label/{hed_safe_label(clip)})"
    )


def sweepstim_epoch_hed(name: str) -> str:
    """Return the epoch-row HED string for a passive-session epoch.

    Parameters
    ----------
    name : the epoch label — ``"spontaneous"``, ``"passive_viewing"``, or a
        movie clip name.

    Returns
    -------
    The composed HED tag string. Clip epochs nest the ``passive_viewing``
    task tag with a ``Movie`` label; the two reserved names map to their
    fixed fragments.
    """
    if name == "spontaneous":
        return SWEEPSTIM_SPONTANEOUS_EPOCH_HED
    if name == SWEEPSTIM_PASSIVE_TASK:
        return SWEEPSTIM_PASSIVE_EPOCH_HED
    return (
        "Experimental-procedure, (Task, Label/passive_viewing), "
        f"(Movie, Label/{hed_safe_label(name)})"
    )
