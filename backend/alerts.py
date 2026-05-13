"""Alert detection + confidence scoring.

Signal logic (mean reversion):
- LONG  when close crosses BELOW -1σ (was >= -1σ at prior bar, < -1σ now)
- SHORT when close crosses ABOVE +1σ (was <=  1σ at prior bar, >  1σ now)

Confidence: 0-3 stars based on alignment of MACD, Stoch RSI, Ichimoku.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from backend import ichimoku
from backend.channels import REGRESSION_WINDOW, compute_channels
from backend.indicators import macd, stoch_rsi


Direction = Literal["long", "short"]


@dataclass
class Signal:
    ticker: str
    direction: Direction
    bar_date: str           # ISO date of the bar that triggered
    price: float
    sd_position: float
    confidence: int         # 0-3
    confirmations: list[str] = field(default_factory=list)


def detect(ticker: str, df: pd.DataFrame) -> Signal | None:
    """Return a Signal if the latest bar triggers a ±1σ cross; else None.

    Expects df with channels + indicators + ichimoku columns already computed.
    """
    if len(df) < 2:
        return None
    prev = df.iloc[-2]
    cur = df.iloc[-1]

    # Need ±1σ band values at both bars.
    if pd.isna(prev.get("upper_1sd")) or pd.isna(prev.get("lower_1sd")):
        return None

    direction: Direction | None = None
    if prev["close"] <= prev["upper_1sd"] and cur["close"] > cur["upper_1sd"]:
        direction = "short"
    elif prev["close"] >= prev["lower_1sd"] and cur["close"] < cur["lower_1sd"]:
        direction = "long"
    if direction is None:
        return None

    confidence, confirmations = _score(direction, cur, df)
    return Signal(
        ticker=ticker.upper(),
        direction=direction,
        bar_date=str(cur["timestamp"].date()) if hasattr(cur["timestamp"], "date") else str(cur["timestamp"])[:10],
        price=float(cur["close"]),
        sd_position=float(cur.get("sd_position", float("nan"))),
        confidence=confidence,
        confirmations=confirmations,
    )


def _score(direction: Direction, cur: pd.Series, df: pd.DataFrame) -> tuple[int, list[str]]:
    confirmations: list[str] = []
    score = 0

    # MACD: for LONG we want bullish momentum (hist rising / line > signal turning up).
    # For SHORT, the opposite.
    hist = cur.get("macd_hist")
    prev_hist = df["macd_hist"].iloc[-2] if "macd_hist" in df.columns else None
    if hist is not None and prev_hist is not None and pd.notna(hist) and pd.notna(prev_hist):
        if direction == "long" and hist > prev_hist:
            score += 1
            confirmations.append("MACD turning up")
        elif direction == "short" and hist < prev_hist:
            score += 1
            confirmations.append("MACD turning down")

    # Stoch RSI: for LONG want %K in oversold (<20) or crossing above %D from oversold.
    # For SHORT the opposite.
    k = cur.get("stoch_rsi_k")
    d = cur.get("stoch_rsi_d")
    if pd.notna(k) and pd.notna(d):
        if direction == "long" and (k < 25 or (k > d and k < 50)):
            score += 1
            confirmations.append(f"Stoch RSI oversold/turning ({k:.0f})")
        elif direction == "short" and (k > 75 or (k < d and k > 50)):
            score += 1
            confirmations.append(f"Stoch RSI overbought/turning ({k:.0f})")

    # Ichimoku: cloud at THIS bar reflects values from 26 bars ago.
    senkou_a_now = df["senkou_a"].iloc[-1 - 26] if len(df) > 26 and "senkou_a" in df.columns else None
    senkou_b_now = df["senkou_b"].iloc[-1 - 26] if len(df) > 26 and "senkou_b" in df.columns else None
    if senkou_a_now is not None and senkou_b_now is not None:
        pos = ichimoku.cloud_position(cur["close"], senkou_a_now, senkou_b_now)
        if direction == "long" and pos == "above":
            score += 1
            confirmations.append("Ichimoku: price above cloud (pullback in uptrend)")
        elif direction == "short" and pos == "below":
            score += 1
            confirmations.append("Ichimoku: price below cloud (bounce in downtrend)")

    return score, confirmations


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Attach channels, indicators, and Ichimoku columns to a raw OHLCV df."""
    df = compute_channels(df, window=REGRESSION_WINDOW)
    df["stoch_rsi_k"], df["stoch_rsi_d"] = stoch_rsi(df["close"])
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
    ichi = ichimoku.compute(df)
    df["tenkan"] = ichi.tenkan
    df["kijun"] = ichi.kijun
    df["senkou_a"] = ichi.senkou_a
    df["senkou_b"] = ichi.senkou_b
    df["chikou"] = ichi.chikou
    return df
