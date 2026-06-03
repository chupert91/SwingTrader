"""yfinance OHLCV fetcher with a small in-memory TTL cache.

Bulk daily bars fall back to Alpaca's free IEX feed when yfinance returns
nothing -- yfinance is regularly rate-limited / IP-blocked from datacenter
(e.g. Vercel serverless) IPs, which would otherwise leave the bot's scan with
an empty universe and silently place zero trades every run.
"""
from __future__ import annotations

import logging
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 60
_INFO_TTL_SECONDS = 24 * 3600  # company name barely changes


@dataclass
class _CacheEntry:
    df: pd.DataFrame
    fetched_at: float


_cache: dict[tuple[str, str], _CacheEntry] = {}
_info_cache: dict[str, tuple[float, str]] = {}  # ticker -> (fetched_at, name)


def fetch_ticker_name(ticker: str) -> str:
    """Returns the full company / fund name from yfinance, or "" on failure.

    Cached daily because .info makes a separate HTTP call that's pretty slow
    (often 1-2s) — fine for one warmup hit per ticker, but we shouldn't pay
    it on every chart load.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        return ""
    cached = _info_cache.get(ticker)
    now = time.time()
    if cached and now - cached[0] < _INFO_TTL_SECONDS:
        return cached[1]
    name = ""
    try:
        info = yf.Ticker(ticker).info or {}
        name = info.get("longName") or info.get("shortName") or ""
    except Exception as exc:
        logger.warning("yfinance .info failed for %s: %s", ticker, exc)
    _info_cache[ticker] = (now, name)
    return name


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
    """Fetch OHLCV for many tickers. Returns {TICKER: DataFrame} where each
    DataFrame has the same columns as fetch_bars(). Tickers with no data are
    omitted.

    Primary source is yfinance (one batched yf.download). Any ticker yfinance
    fails to return — including the common case where it returns nothing for
    the WHOLE list under Yahoo rate-limiting — is retried against Alpaca's free
    IEX daily-bar feed. Without this fallback a rate-limited yfinance leaves the
    scan with an empty universe (scan=0) every run, regardless of any signal.
    """
    tickers = [t.upper().strip() for t in tickers if t and t.strip()]
    if not tickers:
        return {}

    out = _yf_bars_bulk(tickers, period)
    missing = [t for t in tickers if t not in out]
    if missing:
        fb = _alpaca_bars_bulk(missing, period)
        if fb:
            logger.info(
                "alpaca bars fallback supplied %d/%d frames yfinance missed",
                len(fb), len(missing),
            )
        out.update(fb)
    return out


def _yf_bars_bulk(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """yfinance bulk path. Returns {} (never raises) on a total fetch failure
    so the caller can fall back to Alpaca."""
    try:
        raw = yf.download(
            tickers=" ".join(tickers),
            period=period,
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as exc:
        # A hard yfinance failure (network / library error) must not raise --
        # return {} so fetch_bars_bulk falls back to Alpaca for the whole list.
        logger.warning("yfinance bulk download failed: %s", exc)
        return {}
    if raw.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}
    # When only one ticker is passed, yf.download returns a flat-column DF
    # rather than a multi-index. Normalize both shapes.
    if len(tickers) == 1:
        tk = tickers[0]
        if not raw.empty:
            shaped = _shape_ohlcv(raw)
            if _is_usable(shaped):
                out[tk] = shaped
            else:
                logger.warning("dropping %s: shaped frame missing timestamp/close", tk)
        return out

    for tk in tickers:
        try:
            tk_raw = raw[tk]
        except KeyError:
            continue
        if tk_raw.dropna(how="all").empty:
            continue
        shaped = _shape_ohlcv(tk_raw)
        if not _is_usable(shaped):
            # One malformed yfinance slice used to raise KeyError('timestamp')
            # inside scan_candidates and kill the whole scan; now we just drop it.
            logger.warning("dropping %s: shaped frame missing timestamp/close", tk)
            continue
        out[tk] = shaped
    return out


# ---- Alpaca IEX daily-bar fallback ---------------------------------------

def _period_start_iso(period: str) -> str:
    """Convert a yfinance-style period ("18mo", "14mo", "1y", "400d") into an
    ISO start date for Alpaca's bars endpoint, with a buffer so the 252-bar
    regression window is always covered."""
    m = re.fullmatch(r"(\d+)(mo|y|wk|w|d)", period.strip().lower())
    if not m:
        days = 400
    else:
        n, unit = int(m.group(1)), m.group(2)
        days = {"mo": 31, "y": 366, "wk": 7, "w": 7, "d": 1}[unit] * n
    start = datetime.now(timezone.utc) - timedelta(days=days + 10)
    return start.strftime("%Y-%m-%d")


def _alpaca_bars_bulk(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """Fallback daily bars from Alpaca's free IEX feed. Returns {} (never
    raises) if Alpaca isn't configured or the request fails -- the bot must
    degrade to fewer names, not crash the scan."""
    try:
        from backend import alpaca_trading as at
    except Exception:
        return {}
    if not at.is_configured():
        return {}

    feed = os.environ.get("ALPACA_DATA_FEED", "iex")
    start = _period_start_iso(period)
    bars_by_sym: dict[str, list] = {}
    # Chunk symbols so the URL stays well under any length / count cap; each
    # chunk paginates independently via next_page_token.
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        page_token = None
        for _ in range(50):  # page cap (safety bound, not expected to hit)
            params = {
                "symbols": ",".join(chunk),
                "timeframe": "1Day",
                "start": start,
                # split (not raw) to match yfinance's split-adjusted close, so
                # the 252d log channel is continuous across splits.
                "adjustment": "split",
                "feed": feed,
                "limit": 10000,
            }
            if page_token:
                params["page_token"] = page_token
            try:
                res = at._request(at.DATA_BASE, f"/v2/stocks/bars?{urllib.parse.urlencode(params)}")
            except Exception as exc:
                logger.warning("alpaca bars fallback request failed: %s", exc)
                break
            page = (res or {}).get("bars") or {}
            for sym, arr in page.items():
                bars_by_sym.setdefault(sym, []).extend(arr)
            page_token = (res or {}).get("next_page_token")
            if not page_token:
                break

    out: dict[str, pd.DataFrame] = {}
    for sym, arr in bars_by_sym.items():
        if not arr:
            continue
        df = pd.DataFrame(arr).rename(columns={
            "t": "timestamp", "o": "open", "h": "high",
            "l": "low", "c": "close", "v": "volume",
        })
        keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].dropna(subset=["close"])
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert(None)
            df["timestamp"] = ts
        if _is_usable(df):
            # Flag that this frame's volume is IEX-only (a few % of
            # consolidated). Dollar-volume liquidity gates must scale it up;
            # see ai_strategy._avg_dollar_vol_m / ALPACA_IEX_VOLUME_SCALE.
            df.attrs["partial_volume"] = True
            out[sym] = df
    return out


def _is_usable(df: pd.DataFrame) -> bool:
    return (not df.empty) and "timestamp" in df.columns and "close" in df.columns


def _shape_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Re-shape a yfinance OHLCV slice into our standard column layout.

    Handles both flat-column frames (typical multi-ticker yf.download after
    raw[tk] selection) and MultiIndex column frames -- yfinance returns the
    latter for a SINGLE ticker passed via yf.download(group_by='ticker'),
    with columns like ('CCJ', 'Close'). Without this flatten, the rename
    map didn't match the tuple keys and 'close' was missing, surfacing as
    `KeyError(['close'])` in ai_strategy.current_z_for_tickers' z-fetch.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(-1)
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
