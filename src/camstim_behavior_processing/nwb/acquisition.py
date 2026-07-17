"""Write the running-wheel data into an NWB file.

Consumes the wheel DataFrame produced by
:func:`camstim_behavior_processing.load_data.running_speed.compute_running_speed`
(columns ``speed``, ``dx``, ``v_sig``, ``v_in`` indexed by ``timestamps``) and
writes it into the NWB ``acquisition`` container plus a ``running`` processing
module. No DataFrame computation happens here (that lives in ``load_data``), so
this module only depends on pynwb.

NWB layout (matches AllenSDK):
    - ``processing/running/speed`` : low-pass-filtered cm/s
    - ``processing/running/dx``    : raw angular change from the pkl
    - ``acquisition/v_sig``        : raw encoder voltage (V)
    - ``acquisition/v_in``         : encoder supply voltage (V)
"""

from __future__ import annotations

import pandas as pd
from pynwb import NWBFile, ProcessingModule
from pynwb.base import TimeSeries


def add_running_speed(nwb: NWBFile, wheel_df: pd.DataFrame) -> NWBFile:
    """Add processed running speed + raw encoder traces to ``nwb``.

    Parameters
    ----------
    nwb : the :class:`~pynwb.NWBFile` to add the data to (modified in place).
    wheel_df : the wheel DataFrame from ``compute_running_speed`` — columns
        ``speed`` (cm/s), ``dx`` (cm), ``v_sig`` (V), ``v_in`` (V), indexed by
        ``timestamps`` (s).

    Returns
    -------
    The same ``nwb`` instance, for chaining.
    """
    timestamps = wheel_df.index.values

    speed_ts = TimeSeries(
        name="speed",
        data=wheel_df["speed"].values,
        timestamps=timestamps,
        unit="cm/s",
        description="Mouse running speed on the wheel, computed from the "
        "rotary-encoder voltage signal (AllenSDK pipeline: "
        "unwrap -> angular change -> linear speed via wheel "
        "geometry -> wrap-artifact clip -> 10 sigma z-score "
        "outlier rejection -> 3rd-order 4 Hz Butterworth "
        "low-pass). Timestamps are sync-file vsync falling edges "
        "(no monitor delay; running is a behavioral signal).",
    )
    dx_ts = TimeSeries(
        name="dx",
        data=wheel_df["dx"].values,
        timestamps=timestamps,
        unit="cm",
        description="Running-wheel angular change (raw pkl encoder dx).",
    )
    v_sig_ts = TimeSeries(
        name="v_sig",
        data=wheel_df["v_sig"].values,
        timestamps=timestamps,
        unit="V",
        description="Raw voltage signal from the running-wheel encoder.",
    )
    v_in_ts = TimeSeries(
        name="v_in",
        data=wheel_df["v_in"].values,
        timestamps=timestamps,
        unit="V",
        description="Theoretical max encoder voltage (nominally 5 V, varies "
        "in practice; used to normalise the wrap).",
    )

    if "running" in nwb.processing:
        mod = nwb.processing["running"]
    else:
        mod = ProcessingModule(
            name="running", description="Running speed processing module"
        )
        nwb.add_processing_module(mod)
    mod.add_data_interface(speed_ts)
    mod.add_data_interface(dx_ts)
    nwb.add_acquisition(v_sig_ts)
    nwb.add_acquisition(v_in_ts)
    return nwb
