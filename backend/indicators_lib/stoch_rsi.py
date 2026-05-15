"""Stochastic RSI — own-pane oscillator with K/D lines and zone markers.

Re-uses backend.indicators.stoch_rsi for the math; just packages it as a
registered indicator so it can be enabled/disabled like the custom ones.
"""
from __future__ import annotations

import pandas as pd

from backend.indicators import stoch_rsi
from backend.indicator_registry import (
    Indicator,
    IndicatorParam,
    IndicatorResult,
    PlotItem,
    line_points,
    register,
)


def _compute(df: pd.DataFrame, params: dict) -> IndicatorResult:
    rsi_period = int(params.get("rsi_period", 14))
    stoch_period = int(params.get("stoch_period", 14))
    k_smooth = int(params.get("k_smooth", 3))
    d_smooth = int(params.get("d_smooth", 3))
    oversold = float(params.get("oversold", 20))
    overbought = float(params.get("overbought", 80))
    color_k = params.get("color_k", "#58a6ff")
    color_d = params.get("color_d", "#f1c40f")

    k, d = stoch_rsi(df["close"], rsi_period, stoch_period, k_smooth, d_smooth)
    times = df["timestamp"].dt.strftime("%Y-%m-%d").tolist()

    return IndicatorResult(
        pane_title=f"Stoch RSI ({rsi_period}, {stoch_period}, {k_smooth}, {d_smooth})",
        items=[
            PlotItem(
                kind="line", name="%K", pane="own",
                data=line_points(times, k),
                style={"color": color_k, "lineWidth": 2, "lastValueVisible": True},
            ),
            PlotItem(
                kind="line", name="%D", pane="own",
                data=line_points(times, d),
                style={"color": color_d, "lineWidth": 1.5, "lastValueVisible": True},
            ),
            PlotItem(
                kind="price_line", name="Zones", pane="own",
                data=[
                    {"price": overbought, "title": f"{overbought:g}", "color": "#ef5350",
                     "lineStyle": "dashed", "lineWidth": 1},
                    {"price": 50, "title": "50", "color": "#888888",
                     "lineStyle": "dotted", "lineWidth": 1},
                    {"price": oversold, "title": f"{oversold:g}", "color": "#26a69a",
                     "lineStyle": "dashed", "lineWidth": 1},
                ],
            ),
        ],
        pane_y_range=(0.0, 100.0),
    )


register(
    Indicator(
        id="stoch_rsi",
        name="Stoch RSI",
        category="Momentum",
        description="Stochastic oscillator applied to RSI. %K (smoothed RSI stochastic) + %D (signal).",
        params=[
            IndicatorParam(id="rsi_period", label="RSI period", type="int", default=14, min=2, max=100, step=1),
            IndicatorParam(id="stoch_period", label="Stoch period", type="int", default=14, min=2, max=100, step=1),
            IndicatorParam(id="k_smooth", label="%K smoothing", type="int", default=3, min=1, max=20, step=1),
            IndicatorParam(id="d_smooth", label="%D smoothing", type="int", default=3, min=1, max=20, step=1),
            IndicatorParam(id="oversold", label="Oversold", type="float", default=20, min=1, max=49, step=1),
            IndicatorParam(id="overbought", label="Overbought", type="float", default=80, min=51, max=99, step=1),
            IndicatorParam(id="color_k", label="%K color", type="color", default="#58a6ff"),
            IndicatorParam(id="color_d", label="%D color", type="color", default="#f1c40f"),
        ],
        compute_fn=_compute,
        has_own_pane=True,
    )
)
