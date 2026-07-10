"""Synthetic session fixtures for the NWB-packaging tests.

Builds small, hand-authored ``events_df`` / ``intervals_df`` frames matching
the schema that ``build_events_and_intervals.build_all`` produces, plus a
minimal pkl, wheel df, and a task-parameters lab-metadata object — enough to
exercise every NWB writer without the legacy extractor or real recordings.
"""

from __future__ import annotations

import datetime
import pickle

import h5py
import numpy as np
import pandas as pd
from pynwb.file import LabMetaData

from camstim_behavior_processing.load_data import SessionData, SweepStimData

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


def make_session(n: int = 660, span: float = 11.0) -> SessionData:
    """Bundle the synthetic frames into a :class:`SessionData`."""
    return SessionData(
        events_df=make_events_df(),
        intervals_df=make_intervals_df(),
        timestamp_data={"stim_vsync_fall": np.linspace(0, span, n)},
        task_parameters=make_task_parameters(),
    )


# ── Passive SweepStim fixtures ─────────────────────────────────────────
#: Number of display frames in the synthetic SweepStim session.
SWEEPSTIM_N_FRAMES = 30


def make_sweepstim_pkl(n_frames: int = SWEEPSTIM_N_FRAMES) -> dict:
    """Build a synthetic passive SweepStim pkl (one movie block).

    Has a top-level ``stimuli`` list with a ``frame_list`` (four runs, two
    blank ``-1`` gaps) and a ``display_sequence`` window, plus a foraging
    encoder and ``intervalsms`` for the frame-count resolver.
    """
    frame_list = [0] * 6 + [-1] * 2 + [1] * 6 + [-1] * 2 + [2] * 6 + [3] * 8
    stim = {
        "movie_path": r"C:\stim\natural_movie_one.npy",
        "frame_list": frame_list,
        "display_sequence": [[0.0, 0.3]],
        "fps": 60.0,
    }
    encoder = {
        "vsig": np.linspace(0.0, 5.0, n_frames),
        "vin": np.full(n_frames, 5.0),
        "dx": np.arange(n_frames, dtype=float),
    }
    return {
        "start_time": 1_700_000_000.0,
        "fps": 60.0,
        "stimuli": [stim],
        "items": {
            "foraging": {
                "params": {"mouse_id": "999999"},
                "encoders": [encoder],
                "intervalsms": [16.0] * (n_frames - 1),
            }
        },
    }


def write_sweepstim_sync(path, n_samples: int = 200) -> None:
    """Write a synthetic SweepStim sync ``.h5`` to ``path``.

    Bit 0 is a period-2 ``stim_vsync`` square wave (a falling edge every
    other sample); bit 1 (``stim_photodiode``) is held constant, so no
    monitor-delay estimate is made (the default is used).
    """
    counters = np.arange(n_samples, dtype=np.int64)
    bit0 = counters % 2
    bits = bit0  # photodiode bit stays 0
    data = np.stack([counters, bits], axis=1)
    meta = {
        "ni_daq": {"sample_rate": 1000},
        "line_labels": ["stim_vsync", "stim_photodiode"],
    }
    with h5py.File(path, "w") as f:
        f.create_dataset("meta", data=repr(meta).encode("utf-8"))
        f.create_dataset("data", data=data)


def write_sweepstim_session_files(directory) -> tuple:
    """Write a synthetic SweepStim pkl + sync pair; return their paths."""
    pkl_path = directory / "sweep.pkl"
    sync_path = directory / "sweep_sync.h5"
    with open(pkl_path, "wb") as f:
        pickle.dump(make_sweepstim_pkl(), f)
    write_sweepstim_sync(sync_path)
    return pkl_path, sync_path


def make_sweepstim_presentations_df() -> pd.DataFrame:
    """Build a synthetic SweepStim presentations frame (3 movie frames)."""
    return pd.DataFrame(
        {
            "start_time": [0.0, 0.1, 0.2],
            "stop_time": [0.1, 0.2, 0.3],
            "start_frame": [0, 6, 12],
            "stop_frame": [6, 12, 18],
            "movie_name": ["natural_movie_one"] * 3,
            "movie_frame_index": [0, 1, 2],
            "movie_repeat": [0, 0, 0],
            "stim_block": [0, 0, 0],
            "epoch_name": ["passive_viewing"] * 3,
        }
    )


def make_sweepstim_epochs_df() -> pd.DataFrame:
    """Build a synthetic SweepStim epoch frame (one clip + trailing gap)."""
    return pd.DataFrame(
        {
            "name": ["natural_movie_one", "spontaneous"],
            "start_time": [0.0, 0.3],
            "stop_time": [0.3, 0.5],
        }
    )


def make_sweepstim_data() -> SweepStimData:
    """Bundle the synthetic SweepStim frames into a :class:`SweepStimData`."""
    return SweepStimData(
        presentations_df=make_sweepstim_presentations_df(),
        epochs_df=make_sweepstim_epochs_df(),
        timestamp_data={"stim_vsync_fall": np.linspace(0.0, 0.5, 30)},
    )
