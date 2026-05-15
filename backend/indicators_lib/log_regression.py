"""Logarithmic regression channel.

Fits a linear regression on log(close) over the most recent N bars, then
plots the mean line plus ±1/2/3 sigma residual bands back in price space.
Useful for assets that trend exponentially (crypto, high-growth stocks)
where a *log*-space channel is the right reference frame.

Distinct from the existing 252-day linear regression channel — that one
runs in price space, this one in log space.
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


def _compute(df: pd.DataFrame, params: dict) -> IndicatorResult:
    window = int(params.get("window", 252))
    sd1 = bool(params.get("show_1sd", True))
    sd2 = bool(params.get("show_2sd", True))
    sd3 = bool(params.get("show_3sd", False))
    color = params.get("color", "#58a6ff")
    line_width = int(params.get("line_width", 2))

    closes = df["close"].astype(float).to_numpy()
    n = len(closes)
    if n < 5:
        return IndicatorResult(pane_title="Log Regression", items=[])

    # Use the last `window` bars (capped at n). Earlier bars stay NaN.
    fit_start = max(0, n - window)
    y = np.log(closes[fit_start:])
    x = np.arange(len(y), dtype=float)
    # Least-squares fit y = m*x + b
    m, b = np.polyfit(x, y, 1)
    fitted = m * x + b
    residuals = y - fitted
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    # Build full-length series (NaN before fit_start so the chart only draws
    # over the fit window).
    full_idx = np.arange(n, dtype=float)
    line = np.full(n, np.nan)
    line[fit_start:] = np.exp(fitted)
    bands = {}
    for k, on in [(1, sd1), (2, sd2), (3, sd3)]:
        if not on:
            continue
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        upper[fit_start:] = np.exp(fitted + k * sigma)
        lower[fit_start:] = np.exp(fitted - k * sigma)
        bands[k] = (upper, lower)
    _ = full_idx  # kept for clarity; not used downstream

    times = df["timestamp"].dt.strftime("%Y-%m-%d").tolist()
    items: list[PlotItem] = [
        PlotItem(
            kind="line",
            name="Log regression",
            pane="price",
            data=line_points(times, pd.Series(line, index=df.index)),
            style={"color": color, "lineWidth": line_width},
        ),
    ]
    style_for_band = {
        1: {"color": color, "lineWidth": 1, "lineStyle": "dotted"},
        2: {"color": color, "lineWidth": 1, "lineStyle": "dashed"},
        3: {"color": color, "lineWidth": 1.5, "lineStyle": "solid"},
    }
    for k, (upper, lower) in bands.items():
        items.append(PlotItem(
            kind="line", name=f"+{k}σ (log)", pane="price",
            data=line_points(times, pd.Series(upper, index=df.index)),
            style=style_for_band[k],
        ))
        items.append(PlotItem(
            kind="line", name=f"-{k}σ (log)", pane="price",
            data=line_points(times, pd.Series(lower, index=df.index)),
            style=style_for_band[k],
        ))

    return IndicatorResult(
        pane_title=f"Log Regression ({len(y)})",
        items=items,
    )


register(
    Indicator(
        id="log_regression",
        name="Logarithmic Regression Channel",
        category="Trend",
        description="Linear regression of log(close) with ±σ residual bands, plotted back in price space. Designed for assets that trend exponentially.",
        params=[
            IndicatorParam(id="window", label="Window (bars)", type="int", default=252, min=20, max=252, step=1),
            IndicatorParam(id="show_1sd", label="±1σ band", type="bool", default=True),
            IndicatorParam(id="show_2sd", label="±2σ band", type="bool", default=True),
            IndicatorParam(id="show_3sd", label="±3σ band", type="bool", default=False),
            IndicatorParam(id="line_width", label="Line width", type="int", default=2, min=1, max=4, step=1),
            IndicatorParam(id="color", label="Color", type="color", default="#58a6ff"),
        ],
        compute_fn=_compute,
        has_own_pane=False,
    )
)
