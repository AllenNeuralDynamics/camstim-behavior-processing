"""Compute the running-wheel DataFrame (raw behavioral data).

Port of ``allensdk.brain_observatory.behavior.data_objects.running_speed.
running_processing.get_running_df``. Converts the rotary-encoder voltage
signal stored in the camstim pkl into linear running speed (cm/s), aligned to
the sync-file vsync falling-edge timestamps (no monitor delay — running is a
behavioral signal, not visual).

This module produces an intermediate DataFrame only — it does no NWB packaging
(that lives in :mod:`camstim_behavior_processing.nwb.acquisition`). It depends
only on numpy / pandas / scipy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.signal as _signal
from scipy.stats import zscore as _zscore

_WHEEL_DIAM_CM = 6.5 * 2.54  # 6.5" wheel
_WHEEL_RUNNING_RADIUS_CM = 0.5 * (2.0 * _WHEEL_DIAM_CM / 3.0)  # mouse at 2/3 R


def _shift(arr, periods=1, fill_value=np.nan):
    """Shift ``arr`` right by ``periods``; fill the gap with ``fill_value``."""
    shifted = np.roll(arr, periods).astype(float)
    shifted[:periods] = fill_value
    return shifted


def _identify_wraps(vsig, min_threshold=1.5, max_threshold=3.5):
    """Find 0V<->5V wrap indices in the encoder voltage signal."""
    shifted = _shift(np.asarray(vsig))
    vsig = np.asarray(vsig)
    with np.errstate(invalid="ignore"):
        pos = np.nonzero((vsig < min_threshold) & (shifted > max_threshold))[0]
        neg = np.nonzero((vsig > max_threshold) & (shifted < min_threshold))[0]
    return pos, neg


def _unwrap_voltage_signal(
    vsig, pos_wraps, neg_wraps, max_threshold=5.1, max_diff=1.0
):
    """Cumulatively unwrap the encoder voltage across 0V<->5V wraps."""
    vsig = np.asarray(vsig)
    vmax = vsig[vsig < max_threshold].max()
    diff = np.zeros(vsig.shape)
    vsig_prev = _shift(vsig)
    if len(pos_wraps):
        diff[pos_wraps] = (vsig[pos_wraps] + vmax) - vsig_prev[pos_wraps]
    if len(neg_wraps):
        diff[neg_wraps] = vsig[neg_wraps] - (vsig_prev[neg_wraps] + vmax)
    wrap_ix = np.concatenate((pos_wraps, neg_wraps))
    other_ix = np.array(sorted(set(range(len(vsig))) - set(wrap_ix.tolist())))
    diff[other_ix] = vsig[other_ix] - vsig_prev[other_ix]
    with np.errstate(invalid="ignore"):
        diff = np.where(np.abs(diff) <= max_diff, diff, np.nan)
    nan_ix = np.isnan(diff)
    summed = np.nancumsum(diff) + vsig[0]
    summed[nan_ix] = np.nan
    return summed


def _local_boundaries(time, index, span=0.25):
    """Return the first/last sample indices within ``span`` s of ``index``."""
    t_val = time[index]
    eligible = np.nonzero(
        (time <= t_val + abs(span)) & (time >= t_val - abs(span))
    )[0]
    return eligible.min(), eligible.max()


def _clip_speed_wraps(speed, time, wrap_indices, t_span=0.25):
    """Clip transient spikes at voltage wraps to the local min/max."""
    out = speed.copy()
    for w in wrap_indices:
        lo, hi = _local_boundaries(time, w, t_span)
        local = np.concatenate((speed[lo:w], speed[w + 1 : hi + 1]))
        out[w] = np.clip(speed[w], np.nanmin(local), np.nanmax(local))
    return out


def _zscore_threshold_1d(data, threshold=10.0):
    """Replace samples whose z-score exceeds ``threshold`` SDs with NaN."""
    out = data.copy().astype(float)
    scores = _zscore(data, nan_policy="omit")
    with np.errstate(invalid="ignore"):
        out[np.abs(scores) > threshold] = np.nan
    return out


def _encoder_from_pkl(pkl: dict, encoder_keys=("behavior",)) -> dict:
    """Return the first rotary encoder found under the given pkl item keys.

    Parameters
    ----------
    pkl : loaded behavior pickle.
    encoder_keys : the ``items`` sub-keys to search, in order. Change-detection
        sessions store the encoder under ``behavior``; passive SweepStim
        sessions store it under ``foraging`` (falling back to ``behavior``).

    Returns
    -------
    The encoder dict (``vsig`` / ``vin`` / ``dx``).

    Raises
    ------
    KeyError : if no encoder is found under any of ``encoder_keys``.
    """
    items = pkl.get("items") or {}
    for key in encoder_keys:
        encoders = (items.get(key) or {}).get("encoders") or []
        if encoders:
            return encoders[0]
    raise KeyError(
        f"No encoder found under items.{{{','.join(encoder_keys)}}}.encoders"
    )


def compute_running_speed(
    pkl: dict,
    time: np.ndarray,
    lowpass: bool = True,
    zscore_threshold: float = 10.0,
    encoder_keys=("behavior",),
) -> pd.DataFrame:
    """Compute linear running speed (cm/s) from the pkl encoder + sync times.

    Mirrors AllenSDK's ``get_running_df``: identifies voltage wraps,
    recomputes angular change from unwrapped voltage (more reliable than
    the pkl ``dx``), converts to linear speed via wheel geometry, clips
    wrap artifacts, removes z-score outliers, and (optionally) low-pass
    filters with a 3rd-order Butterworth at 4 Hz (60 Hz fs).

    Parameters
    ----------
    pkl : loaded behavior pickle.
    time : 1d sync-file vsync falling-edge times (s), one per frame.
    lowpass : whether to apply the 4 Hz Butterworth filter.
    zscore_threshold : outlier rejection threshold in SDs.
    encoder_keys : ``items`` sub-keys to search for the encoder, in order
        (default ``("behavior",)``; the SweepStim path passes
        ``("foraging", "behavior")``).

    Returns
    -------
    DataFrame indexed by ``timestamps`` with columns ``speed``, ``dx``,
    ``v_sig``, ``v_in`` (length-matched to ``time``).
    """
    enc = _encoder_from_pkl(pkl, encoder_keys)
    v_sig = np.asarray(enc["vsig"])
    v_in = np.asarray(enc["vin"])
    dx_raw = np.asarray(enc["dx"])

    # AllenSDK guard: encoder array can be 1 longer than the frame array.
    if len(v_in) == len(time) + 1:
        v_in = v_in[:-1]
        v_sig = v_sig[:-1]
    elif len(v_in) > len(time):
        v_in = v_in[: len(time)]
        v_sig = v_sig[: len(time)]
    elif len(v_in) < len(time):
        time = time[: len(v_in)]

    pos, neg = _identify_wraps(v_sig)
    unwrapped = _unwrap_voltage_signal(v_sig, pos, neg)
    delta_theta = np.diff(unwrapped, prepend=np.nan) / v_in * 2 * np.pi
    theta = np.nancumsum(delta_theta)
    theta[np.isnan(delta_theta)] = np.nan

    dt = np.diff(time, prepend=np.nan)
    angular_speed = np.diff(theta, prepend=np.nan) / dt  # rad/s
    linear_speed = angular_speed * _WHEEL_RUNNING_RADIUS_CM  # cm/s

    linear_speed = _clip_speed_wraps(
        linear_speed, time, np.concatenate([pos, neg]), t_span=0.25
    )
    linear_speed = _zscore_threshold_1d(
        linear_speed, threshold=zscore_threshold
    )

    if lowpass:
        b, a = _signal.butter(3, Wn=4, fs=60, btype="lowpass")
        linear_speed = _signal.filtfilt(b, a, np.nan_to_num(linear_speed))

    n = len(time)
    return pd.DataFrame(
        {
            "speed": linear_speed[:n],
            "dx": dx_raw[:n],
            "v_sig": v_sig[:n],
            "v_in": v_in[:n],
        },
        index=pd.Index(time, name="timestamps"),
    )
