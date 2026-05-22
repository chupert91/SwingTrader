"""Signal strategy for the long-stock bot (stock_trader).

A faithful live implementation of the thesis-2 capitulation entry
(research/thesis2/THESIS.md):

  - 252-day LOG-price regression channel; z-score on the residuals.
  - Fire on a z <= entry_z FIRST touch - z crossed below the threshold
    since the prior bar (z_now <= entry_z AND z_prev > entry_z).
  - Require drawdown from the trailing 1-year high <= drawdown_max_pct.
  - Require RSI(14) >= rsi_min at the signal bar.

Pure stock signal - no options, no contracts. Reuses backend.data for bars.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backend.data import fetch_bars_bulk
from backend.sp500_universe import SP500_UNIVERSE

WINDOW = 252               # regression-channel lookback (trading days)
_MIN_BARS = WINDOW + 5     # need the window + a prior bar for first-touch
_FETCH_PERIOD = "2y"       # ~500 bars - ample for the 252-day window


def _ols_z(logp: np.ndarray) -> float:
    """z-score of the LAST point of `logp` vs an OLS line fit over it.

    `logp` is one trailing window of log-prices. Matches the thesis-2
    research definition (research/thesis2/channels.py)."""
    n = len(logp)
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    denom = float(np.sum(xc * xc))
    if denom <= 0:
        return float("nan")
    slope = float(np.sum(xc * (logp - logp.mean())) / denom)
    intercept = float(logp.mean() - slope * x.mean())
    fit = slope * x + intercept
    sigma = float(np.std(logp - fit, ddof=1))
    if sigma <= 0:
        return float("nan")
    return float((logp[-1] - fit[-1]) / sigma)


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """Wilder RSI (EMA smoothing) at the last bar."""
    s = pd.Series(closes)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return float((100.0 - 100.0 / (1.0 + rs)).iloc[-1])


def evaluate(closes: np.ndarray) -> dict | None:
    """Channel z (now and prior bar), 1y drawdown and RSI for one ticker.

    Returns None if the series is too short or non-positive. Pure function -
    the unit-testable core of the strategy."""
    if len(closes) < _MIN_BARS or not np.all(closes > 0):
        return None
    logp = np.log(closes)
    return {
        "z": _ols_z(logp[-WINDOW:]),
        "z_prev": _ols_z(logp[-WINDOW - 1:-1]),
        "drawdown_pct": (closes[-1] / float(np.max(closes[-WINDOW:])) - 1.0) * 100.0,
        "rsi": _rsi(closes),
    }


def scan_candidates(settings: dict) -> list[dict]:
    """Scan the S&P universe for entries that pass all three thesis-2 gates.

    Returns candidate dicts sorted deepest-z first."""
    entry_z = float(settings.get("entry_z", -3.0))
    dd_max = float(settings.get("drawdown_max_pct", -30.0))
    rsi_min = float(settings.get("rsi_min", 30.0))

    bars = fetch_bars_bulk(SP500_UNIVERSE, period=_FETCH_PERIOD)
    out: list[dict] = []
    for ticker, df in bars.items():
        closes = df["close"].to_numpy(dtype=float)
        m = evaluate(closes)
        if m is None or math.isnan(m["z"]) or math.isnan(m["z_prev"]):
            continue
        first_touch = m["z"] <= entry_z and m["z_prev"] > entry_z
        if not first_touch:
            continue
        if m["drawdown_pct"] > dd_max or m["rsi"] < rsi_min:
            continue
        out.append({
            "ticker": ticker.upper(),
            "price": round(float(closes[-1]), 2),
            "z": round(m["z"], 3),
            "drawdown_pct": round(m["drawdown_pct"], 1),
            "rsi": round(m["rsi"], 1),
            "touch_date": str(pd.to_datetime(df["timestamp"].iloc[-1]).date()),
        })
    out.sort(key=lambda c: c["z"])     # deepest (most negative) z first
    return out


__all__ = ["evaluate", "scan_candidates", "WINDOW"]
