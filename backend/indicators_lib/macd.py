"""MACD — own-pane: MACD line, signal line, and signed histogram.

Re-uses backend.indicators.macd for the math.
"""
from __future__ import annotations

import pandas as pd

from backend.indicators import macd as _macd
from backend.indicator_registry import (
    Indicator,
    IndicatorParam,
    IndicatorResult,
    PlotItem,
    line_points,
    register,
)


def _compute(df: pd.DataFrame, params: dict) -> IndicatorResult:
    fast = int(params.get("fast", 12))
    slow = int(params.get("slow", 26))
    signal_len = int(params.get("signal", 9))
    color_macd = params.get("color_macd", "#58a6ff")
    color_signal = params.get("color_signal", "#f1c40f")
    up_color = params.get("up_color", "#26a69a")
    down_color = params.get("down_color", "#ef5350")

    macd_line, signal_line, hist = _macd(df["close"], fast, slow, signal_len)
    times = df["timestamp"].dt.strftime("%Y-%m-%d").tolist()

    hist_data = [
        {"time": t, "value": float(v), "color": up_color if v >= 0 else down_color}
        for t, v in zip(times, hist)
        if pd.notna(v)
    ]

    return IndicatorResult(
        pane_title=f"MACD ({fast},{slow},{signal_len})",
        items=[
            PlotItem(
                kind="line", name="MACD", pane="own",
                data=line_points(times, macd_line),
                style={"color": color_macd, "lineWidth": 2, "lastValueVisible": True},
            ),
            PlotItem(
                kind="line", name="Signal", pane="own",
                data=line_points(times, signal_line),
                style={"color": color_signal, "lineWidth": 1.5, "lastValueVisible": True},
            ),
            PlotItem(
                kind="histogram", name="Histogram", pane="own",
                data=hist_data,
                style={"base": 0},
            ),
        ],
    )


register(
    Indicator(
        id="macd",
        name="MACD",
        category="Momentum",
        description="Moving Average Convergence/Divergence. Line (fast EMA - slow EMA), signal (EMA of line), histogram (line - signal).",
        params=[
            IndicatorParam(id="fast", label="Fast EMA", type="int", default=12, min=2, max=100, step=1),
            IndicatorParam(id="slow", label="Slow EMA", type="int", default=26, min=2, max=200, step=1),
            IndicatorParam(id="signal", label="Signal EMA", type="int", default=9, min=1, max=50, step=1),
            IndicatorParam(id="color_macd", label="MACD color", type="color", default="#58a6ff"),
            IndicatorParam(id="color_signal", label="Signal color", type="color", default="#f1c40f"),
            IndicatorParam(id="up_color", label="Hist up", type="color", default="#26a69a"),
            IndicatorParam(id="down_color", label="Hist down", type="color", default="#ef5350"),
        ],
        compute_fn=_compute,
        has_own_pane=True,
    )
)
