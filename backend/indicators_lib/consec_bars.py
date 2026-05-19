"""Consecutive Up/Down Bars — own-pane signed run-length histogram.

One indicator, signed: when the current bar is up it plots the length of
the active up-run as a positive green bar; when down it plots the active
down-run as a negative red bar; a flat/undefined bar plots 0.

"Up" vs "down" is defined by `mode`:
  - "Prior close": close[i] vs close[i-1]  (TradingView-standard default)
  - "Open":        close[i] vs open[i]     (candle body sign)

A run resets to 0 on a flat bar (no change) and switches sides on a
reversal, so the bar magnitude is the number of consecutive same-side
bars ending at (and including) the current bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.indicator_registry import (
    Indicator,
    IndicatorParam,
    IndicatorResult,
    PlotItem,
    register,
)


def _signed_runs(df: pd.DataFrame, mode: str) -> np.ndarray:
    """Signed run-length per bar: +up_run when up, -down_run when down, 0 else.

    Index 0 is always 0 (no prior bar to compare under "Prior close"; under
    "Open" it is still well-defined, but we keep both modes consistent so the
    pane starts flat).
    """
    close = df["close"].astype(float).to_numpy()
    n = len(close)
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out

    if mode == "Open":
        ref = df["open"].astype(float).to_numpy()
        # close vs same-bar open; bar 0 is defined but kept 0 for parity.
        diff = np.full(n, np.nan)
        diff[1:] = close[1:] - ref[1:]
    else:  # "Prior close"
        diff = np.full(n, np.nan)
        diff[1:] = close[1:] - close[:-1]

    up_run = 0
    down_run = 0
    for i in range(n):
        d = diff[i]
        if np.isnan(d) or d == 0.0:
            up_run = 0
            down_run = 0
            out[i] = 0.0
        elif d > 0.0:
            up_run += 1
            down_run = 0
            out[i] = float(up_run)
        else:
            down_run += 1
            up_run = 0
            out[i] = float(-down_run)
    return out


def _compute(df: pd.DataFrame, params: dict) -> IndicatorResult:
    mode = params.get("mode", "Prior close")
    if mode not in ("Prior close", "Open"):
        mode = "Prior close"
    up_color = params.get("up_color", "#26a69a")
    down_color = params.get("down_color", "#ef5350")

    runs = _signed_runs(df, mode)
    times = df["timestamp"].dt.strftime("%Y-%m-%d").tolist()

    data = []
    for t, v in zip(times, runs):
        if not np.isfinite(v):
            continue
        if v > 0:
            color = up_color
        elif v < 0:
            color = down_color
        else:
            color = "#5c6370"
        data.append({"time": t, "value": float(v), "color": color})

    return IndicatorResult(
        pane_title=f"Consec Up/Down Bars ({mode})",
        items=[
            PlotItem(
                kind="histogram",
                name="Consecutive Up/Down",
                pane="own",
                data=data,
                style={"base": 0},
            ),
        ],
    )


register(
    Indicator(
        id="consec_bars",
        name="Consecutive Up/Down Bars",
        category="Momentum",
        description=(
            "Signed run-length of consecutive up/down bars. Up-run plots "
            "positive (green), down-run negative (red); flat bar resets to 0. "
            "Up/down defined by close vs prior close (default) or vs open."
        ),
        params=[
            IndicatorParam(
                id="mode",
                label="Bar direction",
                type="select",
                default="Prior close",
                options=["Prior close", "Open"],
                help="'Prior close': close vs previous close. 'Open': close vs same-bar open.",
            ),
            IndicatorParam(id="up_color", label="Up color", type="color", default="#26a69a"),
            IndicatorParam(id="down_color", label="Down color", type="color", default="#ef5350"),
        ],
        compute_fn=_compute,
        has_own_pane=True,
    )
)
