"""Synthetic session fixtures for the NWB-packaging tests.

Builds small, hand-authored ``events_df`` / ``intervals_df`` frames matching
the schema that ``build_events_and_intervals.build_all`` produces, plus a
minimal pkl, wheel df, and a task-parameters lab-metadata object — enough to
exercise every NWB writer without the legacy extractor or real recordings.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
from pynwb.file import LabMetaData

_EVENT_COLS = [
    "timestamp",
    "event_type",
    "trials_id",
    "stimulus_presentations_id",
    "image_name",
    "orientation",
    "frame",
    "reward_volume",
    "lick_classification",
    "bout_start",
    "reward_type",
    "movie_frame_index",
    "movie_repeat",
]

_TRIAL_ANNOT_COLS = [
    "go",
    "catch",
    "auto_rewarded",
    "aborted",
    "hit",
    "miss",
    "false_alarm",
    "correct_reject",
    "change_time",
    "change_frame",
    "initial_image_name",
    "change_image_name",
    "initial_orientation",
    "change_orientation",
    "reward_time",
    "reward_volume",
    "response_time",
    "response_latency",
]


def make_events_df(with_orientation: bool = True) -> pd.DataFrame:
    """Build a synthetic events frame (images session, with a movie)."""
    rows: list[dict] = []

    def add(**kw):
        """Append one event row, defaulting unset columns to NaN."""
        rows.append({c: kw.get(c, np.nan) for c in _EVENT_COLS})

    add(
        timestamp=1.0,
        event_type="image_onset",
        trials_id=0,
        stimulus_presentations_id=0,
        image_name="im000",
        frame=60,
    )
    add(
        timestamp=1.25,
        event_type="image_offset",
        trials_id=0,
        stimulus_presentations_id=0,
        image_name="im000",
        frame=75,
    )
    # An onset whose offset is missing (exercises the default-duration path)
    # and whose trials_id is NaN (outside any trial).
    add(
        timestamp=1.5,
        event_type="image_onset",
        stimulus_presentations_id=1,
        image_name="im000",
        frame=90,
    )
    add(
        timestamp=2.0,
        event_type="image_onset",
        trials_id=0,
        stimulus_presentations_id=2,
        image_name="im031",
        frame=120,
    )
    add(
        timestamp=2.0,
        event_type="image_change",
        trials_id=0,
        stimulus_presentations_id=2,
        image_name="im031",
        frame=120,
    )
    add(
        timestamp=2.25,
        event_type="image_offset",
        trials_id=0,
        stimulus_presentations_id=2,
        image_name="im031",
        frame=135,
    )
    add(
        timestamp=2.3,
        event_type="lick",
        trials_id=0,
        frame=138,
        lick_classification="hit",
        bout_start=True,
    )
    # A lick with no classification and not a bout start (within_bout, n/a).
    add(
        timestamp=2.35,
        event_type="lick",
        trials_id=0,
        frame=140,
        bout_start=False,
    )
    add(
        timestamp=2.4,
        event_type="reward",
        trials_id=0,
        frame=142,
        reward_volume=0.007,
        reward_type="earned",
    )
    # A reward with no reward_type (stays "n/a").
    add(timestamp=6.0, event_type="reward", trials_id=1, frame=360)
    # An omitted flash with a NaN stimulus_presentations_id.
    add(timestamp=3.0, event_type="image_omission", trials_id=0, frame=180)
    # Two movie frames.
    add(
        timestamp=10.0,
        event_type="movie_onset",
        frame=600,
        movie_frame_index=0,
        movie_repeat=0,
    )
    add(
        timestamp=10.03,
        event_type="movie_offset",
        frame=602,
        movie_frame_index=0,
        movie_repeat=0,
    )
    add(
        timestamp=10.03,
        event_type="movie_onset",
        frame=602,
        movie_frame_index=1,
        movie_repeat=0,
    )
    add(
        timestamp=10.06,
        event_type="movie_offset",
        frame=604,
        movie_frame_index=1,
        movie_repeat=0,
    )
    # A miss point event, which the events table drops.
    add(timestamp=10.06, event_type="miss")

    df = pd.DataFrame(rows)
    if not with_orientation:
        df = df.drop(columns=["orientation"])
    return df


def make_intervals_df() -> pd.DataFrame:
    """Build a synthetic flat intervals frame (epochs, trials, windows)."""
    base = {c: np.nan for c in _TRIAL_ANNOT_COLS}
    rows: list[dict] = []

    def add(
        start, stop, itype, label="", trials_id=np.nan, hed_string=np.nan, **kw
    ):
        """Append one interval row with the trial-annotation defaults."""
        r = dict(base)
        r.update(
            start_time=start,
            stop_time=stop,
            interval_type=itype,
            label=label,
            trials_id=trials_id,
            hed_string=hed_string,
        )
        r.update(kw)
        rows.append(r)

    add(0.0, 0.5, "epoch", label="warm_up")
    add(0.5, 9.0, "epoch", label="change_detection")
    add(10.0, 11.0, "epoch", label="natural_movie_one")
    add(
        1.0,
        3.5,
        "trial",
        trials_id=0,
        go=True,
        catch=False,
        auto_rewarded=False,
        aborted=False,
        hit=True,
        miss=False,
        false_alarm=False,
        correct_reject=False,
        change_time=2.0,
        change_frame=120,
        initial_image_name="im000",
        change_image_name="im031",
        reward_time=2.4,
        reward_volume=0.007,
        response_time=2.3,
        response_latency=0.3,
    )
    add(1.9, 2.1, "change_window", trials_id=0, hed_string="Def/change_win")
    add(2.0, 2.75, "response_window", trials_id=0, hed_string="Def/resp_win")
    # A second trial (catch / correct_reject) with no windows.
    add(
        4.0,
        6.5,
        "trial",
        trials_id=1,
        go=False,
        catch=True,
        auto_rewarded=False,
        aborted=False,
        hit=False,
        miss=False,
        false_alarm=False,
        correct_reject=True,
        initial_image_name="im000",
        change_image_name="im000",
        reward_volume=0.0,
    )
    return pd.DataFrame(rows)


def make_pkl(warm_up_trials: int = 1) -> dict:
    """Build a minimal behavior pkl (identity + params)."""
    return {
        "start_time": datetime.datetime(2024, 1, 2, 3, 4, 5),
        "items": {
            "behavior": {
                "params": {
                    "mouse_id": "123456",
                    "stage": "OPHYS_1_images_A",
                    "warm_up_trials": warm_up_trials,
                }
            }
        },
    }


def make_wheel_df(n: int = 660, span: float = 11.0) -> pd.DataFrame:
    """Build a trivial (all-zero speed) wheel df of length ``n``."""
    t = np.linspace(0, span, n)
    return pd.DataFrame(
        {
            "speed": np.zeros_like(t),
            "dx": np.zeros_like(t),
            "v_sig": np.ones_like(t),
            "v_in": np.full_like(t, 5.0),
        },
        index=pd.Index(t, name="timestamps"),
    )


def make_task_parameters() -> LabMetaData:
    """Build a stand-in task-parameters lab-metadata object."""
    return LabMetaData(name="task_parameters")
