"""yfinance OHLCV fetcher with a small in-memory TTL cache."""
from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

_CACHE_TTL_SECONDS = 60


@dataclass
class _CacheEntry:
    df: pd.DataFrame
    fetched_at: float


_cache: dict[tuple[str, str], _CacheEntry] = {}


def fetch_bars(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch daily OHLCV bars for a ticker.

    Returns a DataFrame with columns: timestamp, open, high, low, close, volume.
    Empty DataFrame if the ticker is unknown.
    """
    ticker = ticker.upper().strip()
    key = (ticker, period)
    now = time.time()
    cached = _cache.get(key)
    if cached and now - cached.fetched_at < _CACHE_TTL_SECONDS:
        return cached.df

    raw = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = raw.reset_index().rename(
        columns={
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )[["timestamp", "open", "high", "low", "close", "volume"]]

    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(None)
    df["timestamp"] = ts

    _cache[key] = _CacheEntry(df=df, fetched_at=now)
    return df
