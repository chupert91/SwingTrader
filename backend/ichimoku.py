"""Ichimoku Cloud (Ichimoku Kinko Hyo) components."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class IchimokuComponents:
    """All five Ichimoku lines, indexed to df's original rows.

    Senkou A/B are NOT yet shifted forward — they're aligned with the bar they
    were computed from. Shift +26 at render time for visual cloud projection.
    """
    tenkan: pd.Series          # Conversion line, (9-high + 9-low) / 2
    kijun: pd.Series           # Base line, (26-high + 26-low) / 2
    senkou_a: pd.Series        # Leading Span A, (tenkan + kijun) / 2
    senkou_b: pd.Series        # Leading Span B, (52-high + 52-low) / 2
    chikou: pd.Series          # Lagging Span, close shifted -26


def compute(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    chikou_shift: int = 26,
) -> IchimokuComponents:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2
    chikou = close.shift(-chikou_shift)

    return IchimokuComponents(tenkan, kijun, senkou_a, senkou_b, chikou)


def cloud_position(close: float, senkou_a: float, senkou_b: float) -> str:
    """Where is price relative to the cloud at this bar?

    Note: at bar t, the "current cloud" is what was projected 26 bars ago,
    i.e. senkou_a and senkou_b values from t - 26. The caller is responsible
    for passing the right values.
    """
    if pd.isna(senkou_a) or pd.isna(senkou_b):
        return "unknown"
    top = max(senkou_a, senkou_b)
    bot = min(senkou_a, senkou_b)
    if close > top:
        return "above"
    if close < bot:
        return "below"
    return "inside"
