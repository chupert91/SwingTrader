"""Supertrend — ATR-based trend follower.

Plots a single line that sits below price during uptrends (acts as a
trailing stop) and above price during downtrends. Color flips when
the trend reverses. Standard params: period=10, multiplier=3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.indicator_registry import (
    Indicator,
    IndicatorParam,
    IndicatorResult,
    PlotItem,
    line_points,
    register,
)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _compute(df: pd.DataFrame, params: dict) -> IndicatorResult:
    period = int(params.get("period", 10))
    mult = float(params.get("multiplier", 3.0))
    up_color = params.get("up_color", "#26a69a")
    down_color = params.get("down_color", "#ef5350")

    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    atr = _atr(df, period).to_numpy()
    mid = (high + low) / 2.0
    basic_upper = mid + mult * atr
    basic_lower = mid - mult * atr

    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    # Path-dependent recursion — must run sequentially.
    for i in range(n):
        if np.isnan(basic_upper[i]) or np.isnan(basic_lower[i]):
            direction[i] = 1
            continue
        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
            direction[i] = 1
            supertrend[i] = final_lower[i]
            continue
        c_prev = close[i - 1]
        final_upper[i] = (
            basic_upper[i]
            if basic_upper[i] < final_upper[i - 1] or c_prev > final_upper[i - 1]
            else final_upper[i - 1]
        )
        final_lower[i] = (
            basic_lower[i]
            if basic_lower[i] > final_lower[i - 1] or c_prev < final_lower[i - 1]
            else final_lower[i - 1]
        )
        c = close[i]
        if c > final_upper[i - 1]:
            direction[i] = 1
        elif c < final_lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
        supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    up_series = pd.Series(
        [v if direction[i] == 1 else np.nan for i, v in enumerate(supertrend)],
        index=df.index,
    )
    down_series = pd.Series(
        [v if direction[i] == -1 else np.nan for i, v in enumerate(supertrend)],
        index=df.index,
    )
    times = df["timestamp"].dt.strftime("%Y-%m-%d").tolist()

    return IndicatorResult(
        pane_title=f"Supertrend ({period}, {mult})",
        items=[
            PlotItem(
                kind="line",
                name="Supertrend (up)",
                pane="price",
                data=line_points(times, up_series),
                style={"color": up_color, "lineWidth": 2},
            ),
            PlotItem(
                kind="line",
                name="Supertrend (down)",
                pane="price",
                data=line_points(times, down_series),
                style={"color": down_color, "lineWidth": 2},
            ),
        ],
    )


register(
    Indicator(
        id="supertrend",
        name="Supertrend",
        category="Trend",
        description="ATR-based trend follower. Trails price during a trend and flips sides on reversal.",
        params=[
            IndicatorParam(id="period", label="ATR Period", type="int", default=10, min=2, max=200, step=1),
            IndicatorParam(id="multiplier", label="ATR Multiplier", type="float", default=3.0, min=0.5, max=10.0, step=0.1),
            IndicatorParam(id="up_color", label="Up color", type="color", default="#26a69a"),
            IndicatorParam(id="down_color", label="Down color", type="color", default="#ef5350"),
        ],
        compute_fn=_compute,
        has_own_pane=False,
    )
)
