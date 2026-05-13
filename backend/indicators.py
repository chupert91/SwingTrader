"""Technical indicators: RSI, Stochastic RSI, MACD."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing (EMA with alpha = 1/period)."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stoch_rsi(
    closes: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI: applies stochastic oscillator to RSI values.

    Returns (%K, %D) on a 0-100 scale. Standard params are 14/14/3/3.
    """
    r = rsi(closes, period=rsi_period)
    lowest = r.rolling(stoch_period).min()
    highest = r.rolling(stoch_period).max()
    raw_k = 100 * (r - lowest) / (highest - lowest).replace(0, np.nan)
    k = raw_k.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d


def macd(
    closes: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
