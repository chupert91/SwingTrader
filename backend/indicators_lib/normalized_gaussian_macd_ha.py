"""Normalised Gaussian MACD Heikin Ashi.

Port of reference/pine/normalized-gaussian-macd-ha.txt (AlgoAlpha).

Replaces standard MACD with a 1-pole Gaussian-filtered MACD normalized by
the gauss-smoothed high-low range. The result is HMA-smoothed, then folded
into Heikin-Ashi-style candles. A histogram (macd - signal) draws below.
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


def _gaussian_filter(series: pd.Series, length: int) -> pd.Series:
    # Pine: beta = (1 - cos(2π/length)) / (2^(1/1) - 1) = (1 - cos(2π/length)) / 1
    beta = (1 - np.cos(2 * np.pi / length)) / 1.0
    alpha = -beta + np.sqrt(beta * beta + 2 * beta)
    out = np.zeros(len(series), dtype=float)
    prev = 0.0
    for i, v in enumerate(series.to_numpy()):
        if np.isnan(v):
            out[i] = prev
            continue
        out[i] = alpha * v + (1 - alpha) * prev
        prev = out[i]
    return pd.Series(out, index=series.index)


def _wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=float)
    s = weights.sum()
    return series.rolling(length).apply(lambda x: float(np.dot(x, weights) / s), raw=True)


def _hma(series: pd.Series, length: int) -> pd.Series:
    half = max(1, int(length / 2))
    sqrt_len = max(1, int(np.sqrt(length)))
    return _wma(2 * _wma(series, half) - _wma(series, length), sqrt_len)


def _ha_from_macd(macd: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Turns the macd time series into Heikin-Ashi-style candles (per Pine).

    open_t = macd[t-1], high_t = max(macd[t], macd[t-1]), low_t = min(...), close_t = macd[t]
    Then standard HA recursion: ha_open = (prev_ha_open + prev_ha_close)/2
    """
    macd_v = macd.to_numpy()
    n = len(macd_v)
    open_v = np.roll(macd_v, 1)
    open_v[0] = macd_v[0] if not np.isnan(macd_v[0]) else 0.0
    high_v = np.fmax(macd_v, open_v)
    low_v = np.fmin(macd_v, open_v)
    close_v = macd_v

    ha_close = (open_v + high_v + low_v + close_v) / 4.0
    ha_open = np.zeros(n, dtype=float)
    for i in range(n):
        if i == 0 or np.isnan(ha_open[i - 1]):
            ha_open[i] = (open_v[i] + close_v[i]) / 2.0
        else:
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.fmax(high_v, np.fmax(ha_open, ha_close))
    ha_low = np.fmin(low_v, np.fmin(ha_open, ha_close))

    idx = macd.index
    return (pd.Series(ha_open, index=idx), pd.Series(ha_high, index=idx),
            pd.Series(ha_low, index=idx), pd.Series(ha_close, index=idx))


def _compute(df: pd.DataFrame, params: dict) -> IndicatorResult:
    fast = int(params.get("fast_length", 12))
    slow = int(params.get("slow_length", 26))
    smoothlen = int(params.get("smooth_length", 14))
    signal_len = int(params.get("signal_length", 9))
    upper = float(params.get("upper", 80))
    lower = float(params.get("lower", 60))
    show_lines = bool(params.get("show_lines", False))
    up_color = params.get("up_color", "#00ffbb")
    down_color = params.get("down_color", "#ff1100")
    up_hist = params.get("up_hist_color", "#79fcd9")
    down_hist = params.get("down_hist_color", "#fc7979")

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    gf_fast = _gaussian_filter(close, fast)
    gf_slow = _gaussian_filter(close, slow)
    gf_hl = _gaussian_filter(high - low, slow)
    raw_macd = (gf_fast - gf_slow) / gf_hl.replace(0.0, np.nan) * 100.0
    macd = _hma(raw_macd, smoothlen)
    signal = macd.ewm(span=signal_len, adjust=False).mean()
    hist = macd - signal

    ha_open, ha_high, ha_low, ha_close = _ha_from_macd(macd)

    times = df["timestamp"].dt.strftime("%Y-%m-%d").tolist()

    # HA candles for the MACD line. Colors follow ha_open vs ha_close per bar.
    candle_data = []
    for i, t in enumerate(times):
        if pd.isna(ha_close.iloc[i]) or pd.isna(ha_open.iloc[i]):
            continue
        candle_data.append({
            "time": t,
            "open": float(ha_open.iloc[i]),
            "high": float(ha_high.iloc[i]),
            "low": float(ha_low.iloc[i]),
            "close": float(ha_close.iloc[i]),
        })

    # Histogram with 4-tone coloring: rising-positive = bright up, falling-positive
    # = dim up, falling-negative = bright down, rising-negative = dim down.
    def _alpha(hex_color: str, opacity_pct: int) -> str:
        # Lightweight Charts accepts rgba() for histogram color; convert hex.
        h = hex_color.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
        return f"rgba({r},{g},{b},{opacity_pct / 100:.2f})"

    hist_data = []
    prev = None
    for i, (t, v) in enumerate(zip(times, hist)):
        if pd.isna(v):
            prev = v
            continue
        if v >= 0:
            color = _alpha(up_hist, 70) if (prev is None or pd.isna(prev) or prev < v) else _alpha(up_hist, 30)
        else:
            color = _alpha(down_hist, 70) if (prev is None or pd.isna(prev) or prev > v) else _alpha(down_hist, 30)
        hist_data.append({"time": t, "value": float(v), "color": color})
        prev = v

    items: list[PlotItem] = [
        PlotItem(
            kind="candle",
            name="GMACD Heikin Ashi",
            pane="own",
            data=candle_data,
            style={
                "upColor": up_color, "downColor": down_color,
                "borderUpColor": up_color, "borderDownColor": down_color,
                "wickUpColor": up_color, "wickDownColor": down_color,
            },
        ),
        PlotItem(
            kind="histogram",
            name="Histogram",
            pane="own",
            data=hist_data,
            style={"base": 0},
        ),
        PlotItem(
            kind="price_line", name="Zones", pane="own",
            data=[
                {"price": upper, "title": f"+{upper:g}", "color": "#ef5350",
                 "lineStyle": "dashed", "lineWidth": 1},
                {"price": lower, "title": f"+{lower:g}", "color": "#ef5350",
                 "lineStyle": "dotted", "lineWidth": 1},
                {"price": 0, "title": "0", "color": "#666666",
                 "lineStyle": "dotted", "lineWidth": 1},
                {"price": -lower, "title": f"-{lower:g}", "color": "#26a69a",
                 "lineStyle": "dotted", "lineWidth": 1},
                {"price": -upper, "title": f"-{upper:g}", "color": "#26a69a",
                 "lineStyle": "dashed", "lineWidth": 1},
            ],
        ),
    ]

    if show_lines:
        items.extend([
            PlotItem(
                kind="line", name="MACD", pane="own",
                data=[{"time": t, "value": float(v)} for t, v in zip(times, macd) if pd.notna(v)],
                style={"color": "#888888", "lineWidth": 1},
            ),
            PlotItem(
                kind="line", name="Signal", pane="own",
                data=[{"time": t, "value": float(v)} for t, v in zip(times, signal) if pd.notna(v)],
                style={"color": "#cccccc", "lineWidth": 1, "lineStyle": "dashed"},
            ),
        ])

    # Reversal arrows on the MACD pane: ha_close crosses ±upper.
    markers = []
    ha_c = ha_close.to_numpy()
    for i in range(1, len(ha_c)):
        if np.isnan(ha_c[i]) or np.isnan(ha_c[i - 1]):
            continue
        if ha_c[i - 1] <= -upper and ha_c[i] > -upper:
            markers.append({"time": times[i], "position": "belowBar",
                            "color": up_color, "shape": "arrowUp"})
        elif ha_c[i - 1] >= upper and ha_c[i] < upper:
            markers.append({"time": times[i], "position": "aboveBar",
                            "color": down_color, "shape": "arrowDown"})
    if markers:
        items.append(PlotItem(kind="marker", name="Reversals", pane="own", data=markers))

    return IndicatorResult(
        pane_title=f"GMACD HA ({fast},{slow},{signal_len})",
        items=items,
    )


register(
    Indicator(
        id="normalized_gaussian_macd_ha",
        name="Normalised Gaussian MACD HA",
        category="Momentum",
        description="MACD with 1-pole Gaussian filters, normalized by gauss(high-low), HMA-smoothed, and folded into Heikin-Ashi-style candles with histogram.",
        params=[
            IndicatorParam(id="fast_length", label="Fast Length", type="int", default=12, min=2, max=100, step=1),
            IndicatorParam(id="slow_length", label="Slow Length", type="int", default=26, min=2, max=200, step=1),
            IndicatorParam(id="smooth_length", label="MACD smoothing (HMA)", type="int", default=14, min=2, max=50, step=1),
            IndicatorParam(id="signal_length", label="Signal smoothing (EMA)", type="int", default=9, min=1, max=50, step=1),
            IndicatorParam(id="upper", label="OB level", type="float", default=80, min=10, max=200, step=5),
            IndicatorParam(id="lower", label="OB inner level", type="float", default=60, min=10, max=200, step=5),
            IndicatorParam(id="show_lines", label="Show MACD/Signal lines", type="bool", default=False),
            IndicatorParam(id="up_color", label="Up color", type="color", default="#00ffbb"),
            IndicatorParam(id="down_color", label="Down color", type="color", default="#ff1100"),
            IndicatorParam(id="up_hist_color", label="Histogram up", type="color", default="#79fcd9"),
            IndicatorParam(id="down_hist_color", label="Histogram down", type="color", default="#fc7979"),
        ],
        compute_fn=_compute,
        has_own_pane=True,
    )
)
