"""Reverse Cutler's RSI.

Ported from reference/pine/reverse-cutlers-rsi.txt (The_Caretaker, MPL-2.0).

Cutler's RSI is the simple-sum variant (no Wilder smoothing). The "reverse"
function inverts the RSI formula to compute the price the next bar would
need to reach a target RSI level. We plot it two ways:

  1. The RSI line + signal line in their own pane with fixed zone lines.
  2. The reverse-projected price (e.g. price needed for RSI=80, RSI=20,
     RSI=signal) as live moving lines on the *price* pane — they snake
     along bar-by-bar, terminating at today's actionable level.
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


def _cutlers_rsi(close: pd.Series, length: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (crsi, u_sum_today, d_sum_today).

    u_sum_today/d_sum_today are the (length-1)-bar rolling sums *ending today*
    — i.e. the inputs to the reverse-projection formula for tomorrow's bar.
    """
    delta = close.diff()
    u_move = delta.clip(lower=0)
    d_move = (-delta).clip(lower=0)
    # CRSI denominator/numerator window = `length` bars (= prev (length-1) bars
    # ending yesterday + today's move). Pine: ((uSum[1] + uMove) / (dSum[1] + dMove))
    u_sum_today = u_move.rolling(length - 1).sum()
    d_sum_today = d_move.rolling(length - 1).sum()
    u_sum_len = u_sum_today.shift(1) + u_move
    d_sum_len = d_sum_today.shift(1) + d_move
    crsi = 100.0 - 100.0 / (1.0 + u_sum_len / d_sum_len.replace(0.0, np.nan))
    return crsi, u_sum_today, d_sum_today


def _reverse_price(close: pd.Series, u_sum: pd.Series, d_sum: pd.Series, target_rsi: pd.Series) -> pd.Series:
    """Vectorized inverse of Cutler's RSI: price tomorrow such that RSI = target.

    target_rsi can be a scalar broadcast or a per-bar series (used for the
    signal-line projection where the target varies bar-by-bar).
    """
    v = pd.Series(target_rsi, index=close.index) if np.isscalar(target_rsi) else target_rsi
    p = close
    # Up-case: assume tomorrow's price >= today's price.
    # p' = p - uSum + dSum * v / (100 - v)
    up_p = p - u_sum + d_sum * v / (100.0 - v)
    # Down-case: tomorrow's price < today's.
    # p' = p + dSum - uSum * (100 - v) / v
    down_p = p + d_sum - u_sum * (100.0 - v) / v.replace(0.0, np.nan)
    # Per Pine, take whichever yields p' >= p (else fall back to down case).
    chosen = np.where(up_p >= p, up_p, down_p)
    chosen = np.clip(chosen, 0, None)
    return pd.Series(chosen, index=close.index)


def _compute(df: pd.DataFrame, params: dict) -> IndicatorResult:
    length = int(params.get("length", 14))
    sig_len = int(params.get("sig_length", 12))
    sig_type = params.get("sig_type", "SMA")
    crit_bull = float(params.get("crit_bull", 80))
    crit_bear = float(params.get("crit_bear", 20))
    alert_high = float(params.get("alert_high", 90))
    alert_low = float(params.get("alert_low", 15))
    show_alerts = bool(params.get("show_alerts", True))
    show_proj = bool(params.get("show_projections", False))
    color_rsi = params.get("color_rsi", "#00ffdd")
    color_signal = params.get("color_signal", "#ffffff")
    color_alert = params.get("color_alert", "#ffff00")
    color_proj_bull = params.get("color_proj_bull", "#26a69a")
    color_proj_bear = params.get("color_proj_bear", "#ef5350")
    color_proj_signal = params.get("color_proj_signal", "#f1c40f")

    close = df["close"].astype(float)
    crsi, u_sum_today, d_sum_today = _cutlers_rsi(close, length)
    if sig_type == "EMA":
        signal = crsi.ewm(span=sig_len, adjust=False).mean()
    else:
        signal = crsi.rolling(sig_len).mean()

    times = df["timestamp"].dt.strftime("%Y-%m-%d").tolist()

    items: list[PlotItem] = [
        PlotItem(
            kind="line",
            name="Cutler's RSI",
            pane="own",
            data=line_points(times, crsi),
            style={"color": color_rsi, "lineWidth": 2, "lastValueVisible": True},
        ),
        PlotItem(
            kind="line",
            name="Signal",
            pane="own",
            data=line_points(times, signal),
            style={"color": color_signal, "lineWidth": 1, "lastValueVisible": True},
        ),
        # Fixed zone lines on the RSI pane — attach to the most-recent series
        # in the pane (the signal line above). Includes the yellow user-alert
        # levels when enabled; the yellow band fills are drawn separately via
        # histogram series below.
        PlotItem(
            kind="price_line",
            name="RSI zones",
            pane="own",
            data=[
                *([{"price": alert_high, "title": f"{alert_high:g}",
                    "color": color_alert, "lineStyle": "dotted", "lineWidth": 1}]
                  if show_alerts else []),
                {"price": crit_bull, "title": f"{crit_bull:g}", "color": color_proj_bull,
                 "lineStyle": "dashed", "lineWidth": 1},
                {"price": 50.0, "title": "50", "color": "#888888",
                 "lineStyle": "dotted", "lineWidth": 1},
                {"price": crit_bear, "title": f"{crit_bear:g}", "color": color_proj_bear,
                 "lineStyle": "dashed", "lineWidth": 1},
                *([{"price": alert_low, "title": f"{alert_low:g}",
                    "color": color_alert, "lineStyle": "dotted", "lineWidth": 1}]
                  if show_alerts else []),
            ],
        ),
    ]

    # Yellow reversal highlight: per Pine, fill the band (alert_high..100) in
    # translucent yellow when CRSI > alert_high; same for (0..alert_low) when
    # CRSI < alert_low. We use histogram series with custom `base` so each
    # bar paints a vertical strip from base to value — consecutive bars look
    # like a continuous fill.
    if show_alerts:
        # Convert hex to rgba with low opacity for the fill.
        h = color_alert.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
        fill_color = f"rgba({r},{g},{b},0.30)"

        high_band = [
            {"time": t, "value": 100.0, "color": fill_color}
            for t, v in zip(times, crsi)
            if pd.notna(v) and v > alert_high
        ]
        low_band = [
            {"time": t, "value": 0.0, "color": fill_color}
            for t, v in zip(times, crsi)
            if pd.notna(v) and v < alert_low
        ]
        if high_band:
            items.append(PlotItem(
                kind="histogram", name="High alert", pane="own",
                data=high_band,
                style={"base": alert_high, "color": fill_color},
            ))
        if low_band:
            items.append(PlotItem(
                kind="histogram", name="Low alert", pane="own",
                data=low_band,
                style={"base": alert_low, "color": fill_color},
            ))

    if show_proj:
        # Reverse projections: each bar's value is the price NEXT bar would
        # need to reach the target RSI. Plotted on the price pane.
        proj_bull = _reverse_price(close, u_sum_today, d_sum_today, crit_bull)
        proj_bear = _reverse_price(close, u_sum_today, d_sum_today, crit_bear)
        proj_signal = _reverse_price(close, u_sum_today, d_sum_today, signal)
        items.extend([
            PlotItem(
                kind="line",
                name=f"Price for RSI {crit_bull:g}",
                pane="price",
                data=line_points(times, proj_bull),
                style={"color": color_proj_bull, "lineWidth": 1, "lineStyle": "dashed",
                       "lastValueVisible": True},
            ),
            PlotItem(
                kind="line",
                name=f"Price for RSI {crit_bear:g}",
                pane="price",
                data=line_points(times, proj_bear),
                style={"color": color_proj_bear, "lineWidth": 1, "lineStyle": "dashed",
                       "lastValueVisible": True},
            ),
            PlotItem(
                kind="line",
                name="Price for signal",
                pane="price",
                data=line_points(times, proj_signal),
                style={"color": color_proj_signal, "lineWidth": 1, "lineStyle": "dotted",
                       "lastValueVisible": True},
            ),
        ])

    return IndicatorResult(
        pane_title=f"Reverse Cutler's RSI ({length}, {sig_len} {sig_type})",
        items=items,
        pane_y_range=(0.0, 100.0),
    )


register(
    Indicator(
        id="reverse_cutlers_rsi",
        name="Reverse Cutler's RSI",
        category="Momentum",
        description="Cutler's RSI (simple-sum) plus inverse projection of the price needed to reach RSI 80/20/signal next bar.",
        params=[
            IndicatorParam(id="length", label="RSI period", type="int", default=14, min=2, max=200, step=1),
            IndicatorParam(id="sig_length", label="Signal MA period", type="int", default=12, min=1, max=100, step=1),
            IndicatorParam(id="sig_type", label="Signal MA type", type="select", default="SMA", options=["SMA", "EMA"]),
            IndicatorParam(id="crit_bull", label="Critical bull RSI", type="float", default=80, min=50, max=99, step=1),
            IndicatorParam(id="crit_bear", label="Critical bear RSI", type="float", default=20, min=1, max=50, step=1),
            IndicatorParam(id="show_alerts", label="Show alert highlights", type="bool", default=True,
                           help="Yellow band when CRSI breaches the alert levels."),
            IndicatorParam(id="alert_high", label="Alert high", type="float", default=90, min=51, max=100, step=1),
            IndicatorParam(id="alert_low", label="Alert low", type="float", default=15, min=0, max=49, step=1),
            IndicatorParam(id="show_projections", label="Show reverse projection on price pane", type="bool", default=False),
            IndicatorParam(id="color_rsi", label="RSI color", type="color", default="#00ffdd"),
            IndicatorParam(id="color_signal", label="Signal color", type="color", default="#ffffff"),
            IndicatorParam(id="color_alert", label="Alert highlight color", type="color", default="#ffff00"),
            IndicatorParam(id="color_proj_bull", label="Bull projection color", type="color", default="#26a69a"),
            IndicatorParam(id="color_proj_bear", label="Bear projection color", type="color", default="#ef5350"),
            IndicatorParam(id="color_proj_signal", label="Signal projection color", type="color", default="#f1c40f"),
        ],
        compute_fn=_compute,
        has_own_pane=True,
    )
)
