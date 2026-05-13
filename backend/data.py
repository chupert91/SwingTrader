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


def fetch_bars_bulk(tickers: list[str], period: str = "14mo") -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for many tickers in one batched call (uses yf.download's
    internal thread pool). Returns {TICKER: DataFrame} where each DataFrame
    has the same columns as fetch_bars(). Tickers with no data are omitted.

    Much faster than calling fetch_bars in a loop — typical 100-200 tickers
    complete in 5-15s vs. 3-5 min sequential.
    """
    tickers = [t.upper().strip() for t in tickers if t and t.strip()]
    if not tickers:
        return {}

    raw = yf.download(
        tickers=" ".join(tickers),
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if raw.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}
    # When only one ticker is passed, yf.download returns a flat-column DF
    # rather than a multi-index. Normalize both shapes.
    if len(tickers) == 1:
        tk = tickers[0]
        if not raw.empty:
            out[tk] = _shape_ohlcv(raw)
        return out

    for tk in tickers:
        try:
            tk_raw = raw[tk]
        except KeyError:
            continue
        if tk_raw.dropna(how="all").empty:
            continue
        out[tk] = _shape_ohlcv(tk_raw)
    return out


def _shape_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Re-shape a yfinance OHLCV slice into our standard column layout."""
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
    )
    keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna(subset=["close"])
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        df["timestamp"] = ts
    return df
