"""VuManChu Cipher B Divergences.

Port of the WaveTrend-based oscillator + divergences popularized by
VuManChu's TradingView indicator. Components:

  - WT1 / WT2 lines (WaveTrend oscillator, the core)
  - Overbought / Oversold horizontal zone lines
  - Buy/Sell cross circles at WT1/WT2 crossovers inside the OS/OB zones
  - MFI area (Money Flow Index scaled to fit on the WaveTrend pane)
  - Divergence markers: regular + hidden bullish/bearish, detected from
    pivots in WT1 vs. pivots in price

"Gold Buy" composite signal is omitted (it's a layered pattern combining
many of the above; users can read the same info off the components).
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


def _wavetrend(df: pd.DataFrame, n1: int, n2: int) -> tuple[pd.Series, pd.Series]:
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    esa = hlc3.ewm(span=n1, adjust=False).mean()
    d = (hlc3 - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * d.replace(0.0, np.nan))
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(4).mean()
    return wt1, wt2


def _mfi(df: pd.DataFrame, length: int, mult: float) -> pd.Series:
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    prev = hlc3.shift(2).replace(0.0, np.nan)
    chg = (hlc3 - hlc3.shift(2)) / prev
    raw = chg * 100.0 * df["volume"].astype(float)
    return raw.rolling(length).mean() / mult


def _find_pivots(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of local max (high) and local min (low) pivots, each k bars on
    both sides. NaN bars are skipped."""
    n = len(values)
    highs = np.zeros(n, dtype=bool)
    lows = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        v = values[i]
        if np.isnan(v):
            continue
        window = values[i - k:i + k + 1]
        if np.isnan(window).any():
            continue
        if v == window.max() and v > values[i - 1]:
            highs[i] = True
        if v == window.min() and v < values[i - 1]:
            lows[i] = True
    return highs, lows


def _find_divergences(wt: np.ndarray, price: np.ndarray, k: int) -> dict[str, list[int]]:
    """Returns indices where a divergence terminates (the current pivot).

    Each divergence type's list is the set of bar indices where that divergence
    was confirmed relative to the immediately preceding pivot on the same side.
    """
    wt_high, wt_low = _find_pivots(wt, k)
    low_idx = np.where(wt_low)[0]
    high_idx = np.where(wt_high)[0]
    bull_reg, bear_reg, bull_hid, bear_hid = [], [], [], []

    for i in range(1, len(low_idx)):
        prev, curr = int(low_idx[i - 1]), int(low_idx[i])
        if np.isnan(price[prev]) or np.isnan(price[curr]):
            continue
        # Regular bull: price LL, WT HL
        if price[curr] < price[prev] and wt[curr] > wt[prev]:
            bull_reg.append(curr)
        # Hidden bull: price HL, WT LL
        elif price[curr] > price[prev] and wt[curr] < wt[prev]:
            bull_hid.append(curr)

    for i in range(1, len(high_idx)):
        prev, curr = int(high_idx[i - 1]), int(high_idx[i])
        if np.isnan(price[prev]) or np.isnan(price[curr]):
            continue
        # Regular bear: price HH, WT LH
        if price[curr] > price[prev] and wt[curr] < wt[prev]:
            bear_reg.append(curr)
        # Hidden bear: price LH, WT HH
        elif price[curr] < price[prev] and wt[curr] > wt[prev]:
            bear_hid.append(curr)

    return {"bull_reg": bull_reg, "bear_reg": bear_reg,
            "bull_hid": bull_hid, "bear_hid": bear_hid}


def _crossovers(wt1: np.ndarray, wt2: np.ndarray) -> tuple[list[int], list[int]]:
    """Returns (buy_indices, sell_indices) — buy = WT1 crosses above WT2 in OS,
    sell = WT1 crosses below WT2 in OB. Threshold checks happen in the caller."""
    n = len(wt1)
    crosses_up, crosses_down = [], []
    for i in range(1, n):
        if np.isnan(wt1[i]) or np.isnan(wt2[i]) or np.isnan(wt1[i - 1]) or np.isnan(wt2[i - 1]):
            continue
        was_below = wt1[i - 1] <= wt2[i - 1]
        is_above = wt1[i] > wt2[i]
        if was_below and is_above:
            crosses_up.append(i)
        elif (not was_below) and wt1[i] < wt2[i]:
            crosses_down.append(i)
    return crosses_up, crosses_down


def _compute(df: pd.DataFrame, params: dict) -> IndicatorResult:
    n1 = int(params.get("n1", 9))
    n2 = int(params.get("n2", 12))
    ob1 = float(params.get("ob_level_1", 53))
    ob2 = float(params.get("ob_level_2", 60))
    os1 = -float(params.get("os_level_1", 53))
    os2 = -float(params.get("os_level_2", 60))
    mfi_len = int(params.get("mfi_length", 60))
    mfi_mult = float(params.get("mfi_mult", 150))
    show_mfi = bool(params.get("show_mfi", True))
    show_divs = bool(params.get("show_divergences", True))
    show_crosses = bool(params.get("show_crosses", True))
    pivot_k = int(params.get("pivot_lookback", 5))
    color_wt1 = params.get("color_wt1", "#00d1ff")
    color_wt2 = params.get("color_wt2", "#7fdcdc")
    color_mfi = params.get("color_mfi", "#f5d020")

    wt1, wt2 = _wavetrend(df, n1, n2)
    mfi = _mfi(df, mfi_len, mfi_mult) if show_mfi else None

    times = df["timestamp"].dt.strftime("%Y-%m-%d").tolist()

    items: list[PlotItem] = [
        PlotItem(
            kind="line", name="WT1", pane="own",
            data=line_points(times, wt1),
            style={"color": color_wt1, "lineWidth": 2, "lastValueVisible": True},
        ),
        PlotItem(
            kind="line", name="WT2", pane="own",
            data=line_points(times, wt2),
            style={"color": color_wt2, "lineWidth": 1, "lastValueVisible": False},
        ),
        # OB/OS zone lines on the WaveTrend pane.
        PlotItem(
            kind="price_line", name="WT zones", pane="own",
            data=[
                {"price": ob2, "title": f"OB {ob2:g}", "color": "#ef5350",
                 "lineStyle": "dashed", "lineWidth": 1},
                {"price": ob1, "title": f"OB {ob1:g}", "color": "#ef5350",
                 "lineStyle": "dotted", "lineWidth": 1},
                {"price": 0,   "title": "0", "color": "#666666",
                 "lineStyle": "dotted", "lineWidth": 1},
                {"price": os1, "title": f"OS {os1:g}", "color": "#26a69a",
                 "lineStyle": "dotted", "lineWidth": 1},
                {"price": os2, "title": f"OS {os2:g}", "color": "#26a69a",
                 "lineStyle": "dashed", "lineWidth": 1},
            ],
        ),
    ]

    if show_mfi and mfi is not None:
        items.append(PlotItem(
            kind="histogram", name="MFI area", pane="own",
            data=[
                {"time": t, "value": float(v), "color": (color_mfi if v >= 0 else "#888888")}
                for t, v in zip(times, mfi)
                if pd.notna(v)
            ],
            style={"color": color_mfi},
        ))

    # Crossover circles — small green/red dots at WT1xWT2 crosses inside OS/OB.
    markers: list[dict] = []
    if show_crosses:
        wt1v, wt2v = wt1.to_numpy(), wt2.to_numpy()
        ups, downs = _crossovers(wt1v, wt2v)
        for i in ups:
            if wt2v[i] < 0:  # cross inside oversold
                markers.append({"time": times[i], "position": "inBar",
                                "color": "#26a69a", "shape": "circle"})
        for i in downs:
            if wt2v[i] > 0:  # cross inside overbought
                markers.append({"time": times[i], "position": "inBar",
                                "color": "#ef5350", "shape": "circle"})

    if show_divs:
        wt1v = wt1.to_numpy()
        price = df["close"].astype(float).to_numpy()
        divs = _find_divergences(wt1v, price, pivot_k)
        for i in divs["bull_reg"]:
            markers.append({"time": times[i], "position": "belowBar",
                            "color": "#26a69a", "shape": "arrowUp", "text": "div"})
        for i in divs["bull_hid"]:
            markers.append({"time": times[i], "position": "belowBar",
                            "color": "#7fdcdc", "shape": "arrowUp", "text": "hid"})
        for i in divs["bear_reg"]:
            markers.append({"time": times[i], "position": "aboveBar",
                            "color": "#ef5350", "shape": "arrowDown", "text": "div"})
        for i in divs["bear_hid"]:
            markers.append({"time": times[i], "position": "aboveBar",
                            "color": "#f0a890", "shape": "arrowDown", "text": "hid"})

    if markers:
        # Markers must be time-sorted before setMarkers().
        markers.sort(key=lambda m: m["time"])
        items.append(PlotItem(
            kind="marker", name="WT signals", pane="own", data=markers,
        ))

    return IndicatorResult(
        pane_title=f"VuManChu Cipher B ({n1}, {n2})",
        items=items,
        pane_y_range=(-100.0, 100.0),
    )


register(
    Indicator(
        id="vumanchu_cipher_b",
        name="VuManChu Cipher B + Divergences",
        category="Momentum",
        description="WaveTrend oscillator with OB/OS zones, crossover dots, MFI area, and pivot-based regular/hidden bull/bear divergence markers.",
        params=[
            IndicatorParam(id="n1", label="Channel Length", type="int", default=9, min=2, max=100, step=1),
            IndicatorParam(id="n2", label="Average Length", type="int", default=12, min=2, max=100, step=1),
            IndicatorParam(id="ob_level_1", label="OB inner", type="float", default=53, min=10, max=99, step=1),
            IndicatorParam(id="ob_level_2", label="OB outer", type="float", default=60, min=10, max=99, step=1),
            IndicatorParam(id="os_level_1", label="OS inner", type="float", default=53, min=10, max=99, step=1),
            IndicatorParam(id="os_level_2", label="OS outer", type="float", default=60, min=10, max=99, step=1),
            IndicatorParam(id="show_mfi", label="Show MFI area", type="bool", default=True),
            IndicatorParam(id="mfi_length", label="MFI length", type="int", default=60, min=2, max=200, step=1),
            IndicatorParam(id="mfi_mult", label="MFI scale divisor", type="float", default=150, min=1, max=1000, step=10),
            IndicatorParam(id="show_crosses", label="Show buy/sell crosses", type="bool", default=True),
            IndicatorParam(id="show_divergences", label="Show divergences", type="bool", default=True),
            IndicatorParam(id="pivot_lookback", label="Pivot lookback bars", type="int", default=5, min=2, max=20, step=1),
            IndicatorParam(id="color_wt1", label="WT1 color", type="color", default="#00d1ff"),
            IndicatorParam(id="color_wt2", label="WT2 color", type="color", default="#7fdcdc"),
            IndicatorParam(id="color_mfi", label="MFI color", type="color", default="#f5d020"),
        ],
        compute_fn=_compute,
        has_own_pane=True,
    )
)
