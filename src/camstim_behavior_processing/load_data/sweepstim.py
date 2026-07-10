"""Intermediate-DataFrame builders for a passive SweepStim session.

Passive SweepStim sessions (natural-movie / grating viewing, no behavioral
trial structure) are the second generation of camstim session this library
handles. Like the change-detection ``trials_events`` builder this is a pure
data layer — no NWB packaging, no scipy — turning a raw ``*.pkl`` +
``*_sync.h5`` pair into analysis-ready pandas tables (numpy/pandas/h5py only):

- :func:`classify_sweepstim_session` — is this pkl a passive SweepStim session?
- :func:`compute_sweepstim_timestamp_alignment` — sync-derived frame times.
- :func:`build_sweepstim_session` — a :class:`SweepStimData` holding the
  per-presentation and per-epoch tables plus the timing arrays.

HED annotation is intentionally *not* applied here; the
:mod:`camstim_behavior_processing.nwb.sweepstim` writer composes HED strings
from these columns (mirroring the change-detection split).
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Column schemas so the tables keep a stable shape even when a session yields
# no rows (an empty movie block, say).
_PRESENTATION_COLUMNS = [
    "start_time",
    "stop_time",
    "start_frame",
    "stop_frame",
    "movie_name",
    "movie_frame_index",
    "movie_repeat",
    "stim_block",
    "epoch_name",
]
_EPOCH_COLUMNS = ["name", "start_time", "stop_time"]

# Default per-display-frame duration (60 Hz) used when a run collapses to a
# single frame (stop_time would otherwise equal start_time).
_FRAME_DURATION_S = 1.0 / 60.0

# Photodiode-derived default monitor delay (s) used when the sync file has too
# few regular photodiode pulses to measure one.
_DEFAULT_MONITOR_DELAY = 0.0356


@dataclass(frozen=True)
class SweepStimData:
    """Materialized intermediates for one passive SweepStim session.

    Produced by :func:`build_sweepstim_session` and consumed by
    :func:`camstim_behavior_processing.nwb.sweepstim.assemble_sweepstim_nwbfile`.

    Attributes
    ----------
    presentations_df : one row per movie/grating frame presentation
        (timing, frame indices, clip label, repeat, block, epoch).
    epochs_df : one row per session epoch (``name`` + ``start_time`` /
        ``stop_time``), with ``spontaneous`` gaps filled in.
    timestamp_data : sync-derived timing arrays — ``stim_ts_visual`` (frame
        times including the monitor delay), ``stim_ts_behavioral`` /
        ``stim_vsync_fall`` (raw frame times, no delay), and ``monitor_delay``.
    """

    presentations_df: pd.DataFrame
    epochs_df: pd.DataFrame
    timestamp_data: Mapping[str, Any]


def classify_sweepstim_session(data: dict) -> tuple[bool, str]:
    """Return ``(is_sweepstim, detail)`` based on pkl structure.

    A passive SweepStim pkl is identified by one or both of a top-level
    ``stimuli`` list with entries and an ``items.foraging`` payload, while
    lacking a non-empty ``items.behavior.trial_log`` (which would mark an
    active change-detection session).

    Parameters
    ----------
    data : the loaded behavior pickle.

    Returns
    -------
    ``(is_sweepstim, detail)`` where ``detail`` explains the decision.
    """
    items = data.get("items") or {}
    behavior = items.get("behavior") or {}
    has_behavior_trials = bool((behavior.get("trial_log") or []))
    has_stim_list = (
        isinstance(data.get("stimuli"), list) and len(data["stimuli"]) > 0
    )
    has_foraging = "foraging" in items

    if has_behavior_trials:
        return False, "behavior session (non-empty behavior.trial_log)"
    if has_stim_list or has_foraging:
        return True, (
            f"sweepstim-like structure: top-level stimuli list="
            f"{has_stim_list}, items.foraging present={has_foraging}"
        )
    return False, "missing sweepstim signatures (no stimuli list/foraging)"


# ── Timestamp alignment ────────────────────────────────────────────────
_LINE_ALIASES = {
    "stim_vsync": ("stim_vsync", "vsync_stim"),
    "vsync_stim": ("stim_vsync", "vsync_stim"),
    "2p_vsync": ("2p_vsync", "vsync_2p"),
    "vsync_2p": ("2p_vsync", "vsync_2p"),
    "acq_trigger": ("acq_trigger", "stim_running"),
    "stim_running": ("acq_trigger", "stim_running"),
    "stim_photodiode": ("stim_photodiode", "photodiode"),
}


def _resolve_bit_index(line_labels: list, line_name: str) -> int:
    """Return the bit index for ``line_name`` (via known aliases)."""
    for cand in _LINE_ALIASES.get(line_name, (line_name,)):
        if cand in line_labels:
            return line_labels.index(cand)
    raise ValueError(
        f"No alias for line {line_name!r} found in sync labels: {line_labels}"
    )


def _get_edges(
    bits, counters, line_labels, line_name, edge_type, sample_rate
) -> np.ndarray:
    """Extract edge times (s) from sync data for a given line.

    Parameters
    ----------
    bits : per-sample packed line-state integers.
    counters : per-sample hardware counter values.
    line_labels : ordered list of sync line names (bit positions).
    line_name : the line to read (resolved through :data:`_LINE_ALIASES`).
    edge_type : ``"rising"``, ``"falling"``, or ``"both"``.
    sample_rate : counter frequency (Hz).

    Returns
    -------
    The edge times, in seconds.
    """
    bit_idx = _resolve_bit_index(line_labels, line_name)
    line_state = (bits >> bit_idx) & 1
    changes = np.diff(line_state.astype(np.int8))
    if edge_type == "falling":
        edge_indices = np.where(changes == -1)[0] + 1
    elif edge_type == "rising":
        edge_indices = np.where(changes == 1)[0] + 1
    else:
        edge_indices = np.where(changes != 0)[0] + 1
    return counters[edge_indices].astype(np.float64) / float(sample_rate)


def resolve_frame_count(pkl: dict) -> int:
    """Resolve the expected display-frame count from the pkl structure.

    Parameters
    ----------
    pkl : the loaded behavior pickle.

    Returns
    -------
    The number of stimulus frames, from the first available of
    ``items.behavior.intervalsms``, ``items.foraging.intervalsms``,
    top-level ``intervalsms`` (each ``len + 1``), ``vsynccount``, or
    ``total_frames``.

    Raises
    ------
    KeyError : if none of those fields are present.
    """
    items = pkl.get("items") or {}
    behavior = items.get("behavior") or {}
    foraging = items.get("foraging") or {}

    for source in (behavior, foraging, pkl):
        intervals = source.get("intervalsms")
        if isinstance(intervals, (list, tuple, np.ndarray)):
            return len(intervals) + 1

    if pkl.get("vsynccount"):
        return int(pkl["vsynccount"])
    if pkl.get("total_frames"):
        return int(pkl["total_frames"])
    raise KeyError("Could not resolve frame count from pkl")


def _clean_photodiode_edges(all_pd_edges: np.ndarray):
    """Return the cleaned run of ~1 Hz photodiode edges, or ``None``.

    Keeps the span between the first and last "regular" (0.8-1.2 s spaced)
    edges, then iteratively drops sub-0.5 s anomaly edges. Returns ``None``
    when there are too few regular pulses to trust a measurement.
    """
    pd_diffs = np.diff(all_pd_edges)
    regular = np.where((pd_diffs > 0.8) & (pd_diffs < 1.2))[0]
    if len(regular) <= 10:
        return None
    clean = all_pd_edges[regular[0] : regular[-1] + 2].copy()
    while True:
        anomalies = np.where(np.diff(clean) < 0.5)[0]
        if len(anomalies) == 0:
            break
        clean = np.delete(clean, anomalies[-1] + 1)
    return clean


def _delay_from_edges(
    clean_pd: np.ndarray, transitions: np.ndarray, default: float
) -> float:
    """Estimate the monitor delay (s) from cleaned photodiode edges.

    Aligns the cleaned photodiode pulses to every 60th vsync (``transitions``)
    and averages the positive sub-70 ms lags; falls back to their median (or
    ``default`` if that is out of range / unmeasurable).
    """
    if not len(clean_pd) or not len(transitions):
        return default
    nearest = int(np.argmin(np.abs(transitions - clean_pd[0])))
    # transitions is non-empty, so nearest < len(transitions) and n_match >= 1.
    n_match = min(len(clean_pd), len(transitions) - nearest)
    delays = clean_pd[:n_match] - transitions[nearest : nearest + n_match]
    valid = (delays > 0) & (delays < 0.07)
    if np.sum(valid) > 10:
        return float(np.mean(delays[valid]))
    median = float(np.median(delays))
    return median if 0 < median < 0.07 else default


def _estimate_monitor_delay(
    all_pd_edges: np.ndarray,
    transitions: np.ndarray,
    default: float = _DEFAULT_MONITOR_DELAY,
) -> float:
    """Estimate the photodiode monitor delay (s), or ``default`` if unmeasured.

    Parameters
    ----------
    all_pd_edges : all photodiode edge times (both directions), sorted.
    transitions : the ~1 Hz reference vsync times (``stim_vsync_fall[::60]``).
    default : delay returned when no reliable estimate can be made.

    Returns
    -------
    The monitor delay in seconds.
    """
    clean = _clean_photodiode_edges(all_pd_edges)
    if clean is None:
        return default
    return _delay_from_edges(clean, transitions, default)


def _read_sync(sync_path: str | Path):
    """Read counters, bits, labels, and sample rate from a sync ``.h5``."""
    with h5py.File(sync_path, "r") as sync_file:
        meta = ast.literal_eval(sync_file["meta"][()].decode("utf-8"))
        sample_rate = meta["ni_daq"].get(
            "sample_rate", meta["ni_daq"].get("counter_output_freq")
        )
        line_labels = meta["line_labels"]
        sync_data = sync_file["data"][:]
    return sync_data[:, 0], sync_data[:, 1], line_labels, sample_rate


def compute_sweepstim_timestamp_alignment(
    pkl: dict, sync_path: str | Path
) -> dict:
    """Compute visual / behavioral frame timestamps for a SweepStim session.

    Parameters
    ----------
    pkl : the loaded behavior pickle (for the expected frame count).
    sync_path : path to the ``*_sync.h5`` sync file.

    Returns
    -------
    A dict with ``stim_ts_visual`` (frame times + monitor delay),
    ``stim_ts_behavioral`` / ``stim_vsync_fall`` (raw frame times), and the
    scalar ``monitor_delay``.
    """
    counters, bits, line_labels, sample_rate = _read_sync(sync_path)

    stim_vsync_fall = _get_edges(
        bits, counters, line_labels, "stim_vsync", "falling", sample_rate
    )
    stim_vsync_fall = stim_vsync_fall[: resolve_frame_count(pkl)]

    all_pd_edges = np.sort(
        _get_edges(
            bits,
            counters,
            line_labels,
            "stim_photodiode",
            "both",
            sample_rate,
        )
    )
    monitor_delay = _estimate_monitor_delay(
        all_pd_edges, stim_vsync_fall[::60]
    )

    return {
        "stim_ts_visual": stim_vsync_fall + monitor_delay,
        "stim_ts_behavioral": stim_vsync_fall,
        "monitor_delay": monitor_delay,
        "stim_vsync_fall": stim_vsync_fall,
    }


# ── Presentation / epoch extraction ────────────────────────────────────
def _as_sequence(value) -> list:
    """Normalize a list/tuple/ndarray/scalar payload to a plain list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _clip_name(stim_obj: dict, default_index: int) -> str:
    """Return the movie/grating clip label for one stimulus block.

    Parameters
    ----------
    stim_obj : one entry of the pkl top-level ``stimuli`` list.
    default_index : block index, used when no path is present.

    Returns
    -------
    The basename (without extension) of ``movie_path`` / ``stim_path``, or a
    ``stim_<index>`` placeholder. camstim paths are Windows-style, so both
    separators are split.
    """
    raw = stim_obj.get("movie_path") or stim_obj.get("stim_path")
    if not raw:
        return f"stim_{default_index:03d}"
    base = str(raw).replace("\\", "/").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] or base


def _run_boundaries(frame_list: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(starts, stops)`` of constant runs in ``frame_list``."""
    change = np.flatnonzero(np.diff(frame_list)) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [frame_list.size]))
    return starts, stops


def _block_rows_from_frame_list(
    stim_obj: dict, block_idx: int, stim_ts_visual: np.ndarray
) -> list[dict]:
    """Build per-presentation rows for one block from camstim ``frame_list``.

    ``frame_list`` is indexed by *global* display frame (60 Hz from session
    start, aligned 1:1 with ``stim_ts_visual``) and holds the on-screen
    identifier (a grating condition or movie frame index), with ``-1`` where
    this block is off screen. Each maximal run of a constant value ``>= 0`` is
    one presentation.

    Parameters
    ----------
    stim_obj : one entry of the pkl top-level ``stimuli`` list.
    block_idx : this block's index in the ``stimuli`` list.
    stim_ts_visual : global per-frame visual timestamps (s).

    Returns
    -------
    list of per-presentation row dicts.
    """
    n_frames = len(stim_ts_visual)
    clip = _clip_name(stim_obj, block_idx)
    fl = np.asarray(_as_sequence(stim_obj.get("frame_list")))
    if fl.size == 0:
        return []

    starts, stops = _run_boundaries(fl)
    rows: list[dict] = []
    repeat_counter: dict[int, int] = {}
    for start_frame, stop_frame in zip(starts.tolist(), stops.tolist()):
        value = int(fl[start_frame])
        if value < 0 or start_frame >= n_frames:
            continue
        stop_frame = min(stop_frame, n_frames - 1)
        start_time = float(stim_ts_visual[start_frame])
        stop_time = float(stim_ts_visual[stop_frame])
        if stop_time <= start_time:
            stop_time = start_time + _FRAME_DURATION_S
        repeat = repeat_counter.get(value, 0)
        repeat_counter[value] = repeat + 1
        rows.append(
            {
                "start_time": start_time,
                "stop_time": stop_time,
                "start_frame": int(start_frame),
                "stop_frame": int(stop_frame),
                "movie_name": clip,
                "movie_frame_index": value,
                "movie_repeat": repeat,
                "stim_block": int(block_idx),
                "epoch_name": "passive_viewing",
            }
        )
    return rows


def _block_rows_from_sweep_frames(
    stim_obj: dict, block_idx: int, stim_ts_visual: np.ndarray
) -> list[dict]:
    """Fallback per-presentation rows for a block with no ``frame_list``.

    Treats ``sweep_frames`` as global vsync index pairs. Correct only for a
    single-block session starting at frame 0; the caller logs a warning.

    Parameters
    ----------
    stim_obj : one entry of the pkl top-level ``stimuli`` list.
    block_idx : this block's index in the ``stimuli`` list.
    stim_ts_visual : global per-frame visual timestamps (s).

    Returns
    -------
    list of per-presentation row dicts.
    """
    n_frames = len(stim_ts_visual)
    clip = _clip_name(stim_obj, block_idx)
    sweeps = _as_sequence(stim_obj.get("sweep_order"))
    sweep_frames = _as_sequence(stim_obj.get("sweep_frames"))
    n_sweeps = min(len(sweeps), len(sweep_frames))
    runs = int(stim_obj.get("runs") or 1)
    sweeps_per_run = max(1, n_sweeps // runs) if n_sweeps else 1

    rows: list[dict] = []
    for k in range(n_sweeps):
        sf, ef = sweep_frames[k]
        sf, ef = int(sf), int(ef)
        if sf >= n_frames:
            continue
        if ef <= sf:
            ef = sf + 1
        stop_frame = min(ef, n_frames - 1)
        start_time = float(stim_ts_visual[sf])
        stop_time = float(stim_ts_visual[stop_frame])
        if stop_time <= start_time:
            stop_time = start_time + _FRAME_DURATION_S
        rows.append(
            {
                "start_time": start_time,
                "stop_time": stop_time,
                "start_frame": sf,
                "stop_frame": stop_frame,
                "movie_name": clip,
                "movie_frame_index": (
                    int(sweeps[k]) if sweeps[k] is not None else -1
                ),
                "movie_repeat": int(k // sweeps_per_run),
                "stim_block": int(block_idx),
                "epoch_name": "passive_viewing",
            }
        )
    return rows


def _iter_sweep_rows(
    stimuli: list[dict], stim_ts_visual: np.ndarray
) -> list[dict]:
    """Build all per-presentation rows across every stimulus block."""
    rows: list[dict] = []
    for block_idx, stim_obj in enumerate(stimuli):
        frame_list = stim_obj.get("frame_list")
        if frame_list is not None and len(frame_list) > 0:
            rows.extend(
                _block_rows_from_frame_list(
                    stim_obj, block_idx, stim_ts_visual
                )
            )
        else:
            logger.warning(
                "Block %d (%s) has no frame_list; falling back to raw "
                "sweep_frames - timing may be wrong for multi-block sessions.",
                block_idx,
                _clip_name(stim_obj, block_idx),
            )
            rows.extend(
                _block_rows_from_sweep_frames(
                    stim_obj, block_idx, stim_ts_visual
                )
            )
    rows.sort(key=lambda r: r["start_time"])
    return rows


def build_sweepstim_presentations(
    stimuli: list[dict], stim_ts_visual: np.ndarray
) -> pd.DataFrame:
    """Build the per-presentation table for a SweepStim session.

    Parameters
    ----------
    stimuli : the pkl top-level ``stimuli`` list.
    stim_ts_visual : global per-frame visual timestamps (s).

    Returns
    -------
    A DataFrame with :data:`_PRESENTATION_COLUMNS`, sorted by onset.
    """
    rows = _iter_sweep_rows(stimuli, stim_ts_visual)
    return pd.DataFrame(rows, columns=_PRESENTATION_COLUMNS)


def _sec_to_time_fn(stim_ts_visual: np.ndarray, fps: float):
    """Return a ``seconds -> vsync-time`` mapper for display-clock windows.

    ``display_sequence`` windows are seconds on the stimulus clock (session
    start = 0); this converts them to the sync timebase via
    ``seconds -> global display frame (* fps) -> vsync time``.
    """
    n_frames = len(stim_ts_visual)

    def sec_to_time(sec) -> float:
        """Map a stimulus-clock time (s) to its vsync time (s)."""
        frame = int(round(float(sec) * fps))
        frame = max(0, min(frame, n_frames - 1))
        return float(stim_ts_visual[frame])

    return sec_to_time


def _raw_epochs_from_sequences(stimuli: list[dict], sec_to_time) -> list[dict]:
    """Build named epochs from each block's ``display_sequence`` windows."""
    epochs: list[dict] = []
    for block_idx, stim_obj in enumerate(stimuli):
        clip = _clip_name(stim_obj, block_idx)
        for window in _as_sequence(stim_obj.get("display_sequence")):
            if isinstance(window, np.ndarray):
                window = window.tolist()
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                continue
            start, stop = sec_to_time(window[0]), sec_to_time(window[1])
            if stop <= start:
                continue
            epochs.append(
                {"name": clip, "start_time": start, "stop_time": stop}
            )
    return epochs


def _fill_spontaneous(
    epochs: list[dict], session_start: float, session_end: float
) -> list[dict]:
    """Insert ``spontaneous`` gap epochs between/around the named epochs."""
    with_spont: list[dict] = []
    prev = session_start
    for ep in epochs:
        if ep["start_time"] > prev:
            with_spont.append(
                {
                    "name": "spontaneous",
                    "start_time": prev,
                    "stop_time": ep["start_time"],
                }
            )
        with_spont.append(ep)
        prev = max(prev, ep["stop_time"])
    if session_end > prev:
        with_spont.append(
            {
                "name": "spontaneous",
                "start_time": prev,
                "stop_time": session_end,
            }
        )
    return with_spont


def build_sweepstim_epochs(
    stimuli: list[dict],
    presentations_df: pd.DataFrame,
    stim_ts_visual: np.ndarray,
    fps: float,
) -> pd.DataFrame:
    """Build the epoch table (named epochs + filled spontaneous gaps).

    Parameters
    ----------
    stimuli : the pkl top-level ``stimuli`` list.
    presentations_df : the presentation table (for the fallback epoch span).
    stim_ts_visual : global per-frame visual timestamps (s).
    fps : stimulus display rate (Hz), for the display-sequence conversion.

    Returns
    -------
    A DataFrame with :data:`_EPOCH_COLUMNS`, sorted by start; empty if the
    session has neither display sequences nor presentations.
    """
    session_start = float(stim_ts_visual[0]) if len(stim_ts_visual) else 0.0
    epochs = _raw_epochs_from_sequences(
        stimuli, _sec_to_time_fn(stim_ts_visual, fps)
    )

    if not epochs:
        if presentations_df.empty:
            return pd.DataFrame([], columns=_EPOCH_COLUMNS)
        epochs = [
            {
                "name": "passive_viewing",
                "start_time": float(presentations_df["start_time"].iloc[0]),
                "stop_time": float(presentations_df["stop_time"].iloc[-1]),
            }
        ]

    epochs.sort(key=lambda e: e["start_time"])
    last_presentation = (
        float(presentations_df["stop_time"].iloc[-1])
        if not presentations_df.empty
        else 0.0
    )
    session_end = max(last_presentation, max(e["stop_time"] for e in epochs))
    filled = _fill_spontaneous(epochs, session_start, session_end)
    return pd.DataFrame(filled, columns=_EPOCH_COLUMNS)


def _stimuli_list(pkl: dict) -> list[dict]:
    """Return the pkl top-level ``stimuli`` list, or raise if empty."""
    stimuli = pkl.get("stimuli")
    if isinstance(stimuli, np.ndarray):
        stimuli = stimuli.tolist()
    if not isinstance(stimuli, list) or not stimuli:
        raise ValueError(
            "SweepStim packaging requires a non-empty top-level stimuli list"
        )
    return stimuli


def _resolve_fps(pkl: dict, stimuli: list[dict]) -> float:
    """Resolve the display rate (Hz) from the pkl or first stimulus block."""
    return float(
        pkl.get("fps") or (stimuli[0].get("fps") if stimuli else None) or 60.0
    )


def build_sweepstim_session(pkl: dict, sync_path: str | Path) -> SweepStimData:
    """Build the intermediate tables for one passive SweepStim session.

    Parameters
    ----------
    pkl : the loaded behavior pickle.
    sync_path : path to the ``*_sync.h5`` sync file.

    Returns
    -------
    A :class:`SweepStimData` with the presentation and epoch tables plus the
    timing arrays.

    Raises
    ------
    ValueError : if the pkl has no non-empty top-level ``stimuli`` list.
    """
    timestamp_data = compute_sweepstim_timestamp_alignment(pkl, sync_path)
    stimuli = _stimuli_list(pkl)
    fps = _resolve_fps(pkl, stimuli)
    stim_ts_visual = timestamp_data["stim_ts_visual"]

    presentations_df = build_sweepstim_presentations(stimuli, stim_ts_visual)
    epochs_df = build_sweepstim_epochs(
        stimuli, presentations_df, stim_ts_visual, fps
    )
    return SweepStimData(
        presentations_df=presentations_df,
        epochs_df=epochs_df,
        timestamp_data=timestamp_data,
    )
