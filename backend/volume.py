"""Volume split into estimated buy/sell using close's position in day's range."""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_volume(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Estimate (buy_volume, sell_volume) per bar from OHLCV.

    buy_share = (close - low) / (high - low)
        close at high → 100% buy
        close at low  → 100% sell
        close mid     → 50/50

    When high == low (no intraday range, e.g. halted), split 50/50.
    """
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    buy_share = ((df["close"] - df["low"]) / rng).fillna(0.5).clip(0.0, 1.0)
    buy_vol = buy_share * df["volume"]
    sell_vol = df["volume"] - buy_vol
    return buy_vol, sell_vol
