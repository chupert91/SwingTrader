"""Alpaca Market Data REST adapter — IEX-only free tier.

Used for live-quote polling on the chart page. The chart's historical
daily bars still come from yfinance via backend/data.py; this module only
provides the most-recent trade for the right-edge "live" price.

Auth via APCA_API_KEY_ID / APCA_API_SECRET_KEY env vars (paper or live
keys both work — Alpaca routes data access on either set of credentials).
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

DATA_BASE = "https://data.alpaca.markets"


def _credentials() -> tuple[str, str] | None:
    # Accept either Alpaca's official-SDK naming (APCA_API_KEY_ID /
    # APCA_API_SECRET_KEY) or the friendlier ALPACA_API_KEY / ALPACA_SECRET_KEY.
    key = (
        os.environ.get("APCA_API_KEY_ID")
        or os.environ.get("ALPACA_API_KEY")
    )
    secret = (
        os.environ.get("APCA_API_SECRET_KEY")
        or os.environ.get("ALPACA_SECRET_KEY")
    )
    if not key or not secret:
        return None
    return key, secret


def is_configured() -> bool:
    return _credentials() is not None


def latest_trade(symbol: str) -> dict | None:
    """Fetch the most recent IEX trade for a symbol.

    Returns {"price": float, "timestamp": str ISO8601} on success, None if
    credentials are missing OR the symbol has no trade data (e.g. illiquid
    or pre-market with no IEX prints).
    """
    creds = _credentials()
    if creds is None:
        return None
    key, secret = creds
    symbol = symbol.upper().strip()
    if not symbol:
        return None

    url = f"{DATA_BASE}/v2/stocks/{urllib.parse.quote(symbol)}/trades/latest?feed=iex"
    req = urllib.request.Request(url, method="GET")
    req.add_header("APCA-API-KEY-ID", key)
    req.add_header("APCA-API-SECRET-KEY", secret)

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        # 404 = symbol not on IEX or no trades yet today; 401 = bad keys.
        if exc.code == 404:
            return None
        logger.warning("alpaca latest_trade %s HTTP %s: %s", symbol, exc.code, exc.reason)
        raise
    except Exception as exc:
        logger.warning("alpaca latest_trade %s failed: %s", symbol, exc)
        raise

    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return None

    trade = data.get("trade") or {}
    price = trade.get("p")
    ts = trade.get("t")
    if price is None:
        return None
    return {"price": float(price), "timestamp": ts}


__all__ = ["latest_trade", "is_configured"]
