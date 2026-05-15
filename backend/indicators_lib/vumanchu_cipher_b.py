"""VuManChu Cipher B + Divergences.

Faithful port of reference/pine/vumanchu-cipher-b-divergence.txt.

Components, top-to-bottom in the indicator pane:

  - WT1 wave (light-blue filled area, baseline 0)
  - WT2 wave (dark-blue filled area, baseline 0, drawn under WT1)
  - Fast WT (VWAP = WT1 - WT2) as a white filled area
  - RSI+MFI area (sma(((close-open)/(high-low)) * mult, len) - posY)
  - OB / OS horizontal level guides
  - Buy / sell / gold-buy / divergence dots — markers attached to WT series

Divergences use the same 5-bar fractal Pine uses (f_top_fractal /
f_bot_fractal in the source), evaluated on WT2 (not WT1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.indicators import rsi as _rsi
from backend.indicator_registry import (
    Indicator,
    IndicatorParam,
    IndicatorResult,
    PlotItem,
    line_points,
    register,
)


def _wavetrend(df: pd.DataFrame, chlen: int, avg: int, malen: int) -> tuple[pd.Series, pd.Series]:
    src = (df["high"] + df["low"] + df["close"]) / 3.0  # hlc3
    esa = src.ewm(span=chlen, adjust=False).mean()
    de = (src - esa).abs().ewm(span=chlen, adjust=False).mean()
    ci = (src - esa) / (0.015 * de.replace(0.0, np.nan))
    wt1 = ci.ewm(span=avg, adjust=False).mean()
    wt2 = wt1.rolling(malen).mean()
    return wt1, wt2


def _rsi_mfi(df: pd.DataFrame, period: int, mult: float, pos_y: float) -> pd.Series:
    # f_rsimfi: sma(((close - open) / (high - low)) * mult, period) - posY
    hl = (df["high"] - df["low"]).replace(0.0, np.nan)
    ratio = (df["close"] - df["open"]) / hl
    return (ratio * mult).rolling(period).mean() - pos_y


def _fractals(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pine's f_top_fractal / f_bot_fractal: 5-bar pattern centered on bar [2].

    Top fractal at bar i (Pine [2]): src[i-2] < src[i] AND src[i-1] < src[i]
                                     AND src[i] > src[i+1] AND src[i] > src[i+2]
    Bot fractal at bar i: same but with inequalities flipped.

    Returns boolean arrays of length len(values). Note: Pine plots the
    divergence with offset=-2, so the fractal at position `i` here corresponds
    to a confirmation 2 bars later — we just return the centered index.
    """
    n = len(values)
    tops = np.zeros(n, dtype=bool)
    bots = np.zeros(n, dtype=bool)
    for i in range(2, n - 2):
        v = values[i]
        if np.isnan(v):
            continue
        if (values[i - 2] < v and values[i - 1] < v and v > values[i + 1] and v > values[i + 2]):
            tops[i] = True
        if (values[i - 2] > v and values[i - 1] > v and v < values[i + 1] and v < values[i + 2]):
            bots[i] = True
    return tops, bots


def _find_divergences(
    src: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    top_limit: float,
    bot_limit: float,
    use_limits: bool,
) -> dict[str, list[int]]:
    """Pine's f_findDivs ported. Returns {bull_reg, bear_reg, bull_hid, bear_hid}
    each as a list of confirmation bar indices (the second pivot in the pair).
    """
    tops, bots = _fractals(src)
    if use_limits:
        tops = np.array([t and src[i] >= top_limit for i, t in enumerate(tops)])
        bots = np.array([b and src[i] <= bot_limit for i, b in enumerate(bots)])

    bull_reg, bear_reg, bull_hid, bear_hid = [], [], [], []

    last_top_src = None
    last_top_high = None
    for i in range(len(src)):
        if tops[i]:
            if last_top_src is not None:
                # bearSignal = high > last_high AND src < last_src
                if high[i] > last_top_high and src[i] < last_top_src:
                    bear_reg.append(i)
                # bearDivHidden = high < last_high AND src > last_src
                elif high[i] < last_top_high and src[i] > last_top_src:
                    bear_hid.append(i)
            last_top_src = src[i]
            last_top_high = high[i]

    last_bot_src = None
    last_bot_low = None
    for i in range(len(src)):
        if bots[i]:
            if last_bot_src is not None:
                # bullSignal = low < last_low AND src > last_src
                if low[i] < last_bot_low and src[i] > last_bot_src:
                    bull_reg.append(i)
                # bullDivHidden = low > last_low AND src < last_src
                elif low[i] > last_bot_low and src[i] < last_bot_src:
                    bull_hid.append(i)
            last_bot_src = src[i]
            last_bot_low = low[i]

    return {"bull_reg": bull_reg, "bear_reg": bear_reg,
            "bull_hid": bull_hid, "bear_hid": bear_hid}


def _compute(df: pd.DataFrame, params: dict) -> IndicatorResult:
    chlen = int(params.get("wt_channel_len", 9))
    avg = int(params.get("wt_average_len", 12))
    malen = int(params.get("wt_ma_len", 3))
    ob_level = float(params.get("ob_level", 53))
    ob_level2 = float(params.get("ob_level2", 60))
    os_level = -float(params.get("ob_level", 53))
    os_level2 = -float(params.get("ob_level2", 60))
    os_level3 = -float(params.get("os_level3", 75))
    mfi_period = int(params.get("mfi_period", 60))
    mfi_mult = float(params.get("mfi_multiplier", 150))
    mfi_pos_y = float(params.get("mfi_pos_y", 2.5))
    rsi_len = int(params.get("rsi_len", 14))
    div_ob_level = float(params.get("div_ob_level", 45))
    div_os_level = float(params.get("div_os_level", -65))
    show_mfi = bool(params.get("show_mfi", True))
    show_vwap = bool(params.get("show_vwap", True))
    show_buy_sell = bool(params.get("show_buy_sell", True))
    show_div = bool(params.get("show_divergences", True))
    show_div_hidden = bool(params.get("show_divergences_hidden", False))
    show_gold = bool(params.get("show_gold_buy", True))
    color_wt1 = params.get("color_wt1", "#4994ec")
    color_wt2 = params.get("color_wt2", "#1f1559")
    color_vwap = params.get("color_vwap", "#ffffff")
    color_mfi_pos = params.get("color_mfi_pos", "#3ee145")
    color_mfi_neg = params.get("color_mfi_neg", "#ff3d2e")

    wt1, wt2 = _wavetrend(df, chlen, avg, malen)
    vwap = wt1 - wt2
    times = df["timestamp"].dt.strftime("%Y-%m-%d").tolist()

    items: list[PlotItem] = []

    # WT2 first (drawn under WT1 visually). Both areas filled toward 0.
    items.append(PlotItem(
        kind="area", name="WT2", pane="own",
        data=line_points(times, wt2),
        style={"color": color_wt2, "fillOpacity": 0.55, "lineWidth": 1},
    ))
    items.append(PlotItem(
        kind="area", name="WT1", pane="own",
        data=line_points(times, wt1),
        style={"color": color_wt1, "fillOpacity": 0.32, "lineWidth": 1,
               "lastValueVisible": True},
    ))

    if show_vwap:
        items.append(PlotItem(
            kind="area", name="Fast WT (VWAP)", pane="own",
            data=line_points(times, vwap),
            style={"color": color_vwap, "fillOpacity": 0.18, "lineWidth": 1},
        ))

    # MFI area — colored histogram split by sign (positive green, negative red).
    if show_mfi:
        mfi = _rsi_mfi(df, mfi_period, mfi_mult, mfi_pos_y)
        mfi_data = [
            {"time": t, "value": float(v),
             "color": (color_mfi_pos if v >= 0 else color_mfi_neg)}
            for t, v in zip(times, mfi)
            if pd.notna(v)
        ]
        items.append(PlotItem(
            kind="histogram", name="MFI Area", pane="own",
            data=mfi_data,
            style={"base": 0},
        ))

    # OB/OS guide lines.
    items.append(PlotItem(
        kind="price_line", name="WT zones", pane="own",
        data=[
            {"price": ob_level2, "title": f"OB {ob_level2:g}", "color": "#ffffff",
             "lineStyle": "dotted", "lineWidth": 1},
            {"price": ob_level, "title": f"OB {ob_level:g}", "color": "#ef5350",
             "lineStyle": "dotted", "lineWidth": 1},
            {"price": 0, "title": "0", "color": "#888888",
             "lineStyle": "dotted", "lineWidth": 1},
            {"price": os_level, "title": f"OS {os_level:g}", "color": "#26a69a",
             "lineStyle": "dotted", "lineWidth": 1},
            {"price": os_level2, "title": f"OS {os_level2:g}", "color": "#ffffff",
             "lineStyle": "dotted", "lineWidth": 1},
        ],
    ))

    # --- Signals --------------------------------------------------------
    wt1v = wt1.to_numpy()
    wt2v = wt2.to_numpy()
    high_v = df["high"].astype(float).to_numpy()
    low_v = df["low"].astype(float).to_numpy()

    # wtCross = cross(wt1, wt2); wtCrossUp = wt2 - wt1 <= 0
    cross_up: list[int] = []
    cross_down: list[int] = []
    for i in range(1, len(wt1v)):
        if np.isnan(wt1v[i]) or np.isnan(wt2v[i]) or np.isnan(wt1v[i - 1]) or np.isnan(wt2v[i - 1]):
            continue
        prev_diff = wt1v[i - 1] - wt2v[i - 1]
        cur_diff = wt1v[i] - wt2v[i]
        if prev_diff <= 0 and cur_diff > 0:
            cross_up.append(i)
        elif prev_diff >= 0 and cur_diff < 0:
            cross_down.append(i)

    markers: list[dict] = []

    if show_buy_sell:
        # Buy circle: cross-up while wt2 oversold (wt2 <= os_level).
        for i in cross_up:
            if wt2v[i] <= os_level:
                markers.append({"time": times[i], "position": "inBar",
                                "color": "#00e676", "shape": "circle"})
        # Sell circle: cross-down while wt2 overbought.
        for i in cross_down:
            if wt2v[i] >= ob_level:
                markers.append({"time": times[i], "position": "inBar",
                                "color": "#ff5252", "shape": "circle"})

    # Divergences on wt2.
    if show_div or show_div_hidden:
        divs = _find_divergences(wt2v, high_v, low_v, div_ob_level, div_os_level, True)
        if show_div:
            for i in divs["bull_reg"]:
                markers.append({"time": times[i], "position": "belowBar",
                                "color": "#00e676", "shape": "arrowUp", "text": "div"})
            for i in divs["bear_reg"]:
                markers.append({"time": times[i], "position": "aboveBar",
                                "color": "#e60000", "shape": "arrowDown", "text": "div"})
        if show_div_hidden:
            for i in divs["bull_hid"]:
                markers.append({"time": times[i], "position": "belowBar",
                                "color": "#7fdcdc", "shape": "arrowUp", "text": "hid"})
            for i in divs["bear_hid"]:
                markers.append({"time": times[i], "position": "aboveBar",
                                "color": "#f0a890", "shape": "arrowDown", "text": "hid"})

    # Gold buy circle — Pine: wt bullish div + wtLow_prev <= osLevel3 + wt2 > osLevel3
    # + wtLow_prev - wt2 <= -5 + rsi at last bot fractal < 30.
    if show_gold:
        rsi_v = _rsi(df["close"], rsi_len).to_numpy()
        divs_all = _find_divergences(wt2v, high_v, low_v, div_ob_level, div_os_level, True)
        # Track most-recent bot fractal value and rsi at that fractal.
        bots = _fractals(wt2v)[1]
        last_bot_val = None
        last_bot_rsi = None
        for i in range(len(wt2v)):
            if bots[i]:
                last_bot_val = wt2v[i]
                last_bot_rsi = rsi_v[i] if i < len(rsi_v) else np.nan
            if i in divs_all["bull_reg"] and last_bot_val is not None:
                if (last_bot_val <= os_level3 and wt2v[i] > os_level3
                        and (last_bot_val - wt2v[i]) <= -5
                        and last_bot_rsi is not None and last_bot_rsi < 30):
                    markers.append({"time": times[i], "position": "inBar",
                                    "color": "#e2a400", "shape": "circle", "text": "GOLD"})

    if markers:
        markers.sort(key=lambda m: m["time"])
        items.append(PlotItem(kind="marker", name="Signals", pane="own", data=markers))

    return IndicatorResult(
        pane_title=f"VuManChu Cipher B ({chlen}, {avg})",
        items=items,
    )


register(
    Indicator(
        id="vumanchu_cipher_b",
        name="VuManChu Cipher B + Divergences",
        category="Momentum",
        description="WaveTrend wave fills, fast-WT (VWAP), MFI area, OB/OS zones, buy/sell/gold-buy circles, and fractal-based regular/hidden divergences on WT2.",
        params=[
            IndicatorParam(id="wt_channel_len", label="WT Channel Length", type="int", default=9, min=2, max=100, step=1),
            IndicatorParam(id="wt_average_len", label="WT Average Length", type="int", default=12, min=2, max=100, step=1),
            IndicatorParam(id="wt_ma_len", label="WT MA Length", type="int", default=3, min=1, max=20, step=1),
            IndicatorParam(id="ob_level", label="OB Level 1", type="float", default=53, min=10, max=99, step=1),
            IndicatorParam(id="ob_level2", label="OB Level 2", type="float", default=60, min=10, max=99, step=1),
            IndicatorParam(id="os_level3", label="OS Level 3 (Gold Buy)", type="float", default=75, min=10, max=99, step=1,
                           help="Used for gold-buy detection. Pine default: -75."),
            IndicatorParam(id="mfi_period", label="MFI period", type="int", default=60, min=2, max=200, step=1),
            IndicatorParam(id="mfi_multiplier", label="MFI multiplier", type="float", default=150, min=1, max=1000, step=10),
            IndicatorParam(id="mfi_pos_y", label="MFI Y offset", type="float", default=2.5, min=-10, max=10, step=0.5),
            IndicatorParam(id="rsi_len", label="RSI length (gold buy)", type="int", default=14, min=2, max=100, step=1),
            IndicatorParam(id="div_ob_level", label="Div Bearish min", type="float", default=45, min=0, max=99, step=1),
            IndicatorParam(id="div_os_level", label="Div Bullish min", type="float", default=-65, min=-99, max=0, step=1),
            IndicatorParam(id="show_mfi", label="Show MFI", type="bool", default=True),
            IndicatorParam(id="show_vwap", label="Show Fast WT (VWAP)", type="bool", default=True),
            IndicatorParam(id="show_buy_sell", label="Show buy/sell circles", type="bool", default=True),
            IndicatorParam(id="show_divergences", label="Show regular divergences", type="bool", default=True),
            IndicatorParam(id="show_divergences_hidden", label="Show hidden divergences", type="bool", default=False),
            IndicatorParam(id="show_gold_buy", label="Show gold-buy circles", type="bool", default=True),
            IndicatorParam(id="color_wt1", label="WT1 color", type="color", default="#4994ec"),
            IndicatorParam(id="color_wt2", label="WT2 color", type="color", default="#1f1559"),
            IndicatorParam(id="color_vwap", label="Fast WT color", type="color", default="#ffffff"),
            IndicatorParam(id="color_mfi_pos", label="MFI positive", type="color", default="#3ee145"),
            IndicatorParam(id="color_mfi_neg", label="MFI negative", type="color", default="#ff3d2e"),
        ],
        compute_fn=_compute,
        has_own_pane=True,
    )
)
