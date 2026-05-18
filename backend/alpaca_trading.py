"""Alpaca paper Trading + options-data REST adapter.

Separate from backend/alpaca.py (which is market-data-only for live quotes).
This module is what the autonomous AI options trader uses to read the paper
account, pick option contracts, and place/cancel orders.

Dependency-free (urllib) to match the rest of the codebase. Credentials use
the same convention as backend/alpaca.py: APCA_API_KEY_ID/APCA_API_SECRET_KEY
or ALPACA_API_KEY/ALPACA_SECRET_KEY. Trading base URL defaults to the paper
endpoint and can be overridden with ALPACA_TRADING_BASE.

Options market data is the free *indicative* feed (no OPRA subscription) —
good enough to price a marketable limit; paper fills are simulated by Alpaca.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

TRADING_BASE = os.environ.get(
    "ALPACA_TRADING_BASE", "https://paper-api.alpaca.markets"
).rstrip("/")
DATA_BASE = "https://data.alpaca.markets"
OPTIONS_FEED = "indicative"  # free tier; OPRA requires Algo Trader Plus


class AlpacaError(RuntimeError):
    """Raised when the Alpaca API returns an error or no credentials exist."""


def _credentials() -> tuple[str, str] | None:
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        return None
    return key, secret


def is_configured() -> bool:
    return _credentials() is not None


def _request(base: str, path: str, method: str = "GET", body: dict | None = None) -> dict | list:
    creds = _credentials()
    if creds is None:
        raise AlpacaError("Alpaca credentials not configured")
    key, secret = creds
    url = f"{base}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, method=method, data=data)
    req.add_header("APCA-API-KEY-ID", key)
    req.add_header("APCA-API-SECRET-KEY", secret)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            pass
        logger.warning("alpaca %s %s -> HTTP %s %s", method, path, exc.code, detail)
        raise AlpacaError(f"HTTP {exc.code}: {detail or exc.reason}") from exc
    except Exception as exc:
        logger.warning("alpaca %s %s failed: %s", method, path, exc)
        raise AlpacaError(str(exc)) from exc
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise AlpacaError(f"bad JSON from {path}") from exc


# ---- Account / clock ------------------------------------------------------

def get_account() -> dict:
    """Raw /v2/account. Key fields used: equity, cash, buying_power,
    options_buying_power, options_trading_level."""
    return _request(TRADING_BASE, "/v2/account")  # type: ignore[return-value]


def get_clock() -> dict:
    """/v2/clock — authoritative market open state (handles holidays/half
    days). Fields: is_open (bool), next_open, next_close, timestamp."""
    return _request(TRADING_BASE, "/v2/clock")  # type: ignore[return-value]


# ---- Positions / orders ---------------------------------------------------

def list_positions() -> list[dict]:
    res = _request(TRADING_BASE, "/v2/positions")
    return res if isinstance(res, list) else []


def list_option_positions() -> list[dict]:
    return [p for p in list_positions() if p.get("asset_class") == "us_option"]


def list_orders(status: str = "all", limit: int = 200, nested: bool = True) -> list[dict]:
    q = urllib.parse.urlencode(
        {"status": status, "limit": limit, "nested": str(nested).lower()}
    )
    res = _request(TRADING_BASE, f"/v2/orders?{q}")
    return res if isinstance(res, list) else []


def get_order(order_id: str) -> dict:
    return _request(TRADING_BASE, f"/v2/orders/{urllib.parse.quote(order_id)}")  # type: ignore[return-value]


def cancel_order(order_id: str) -> None:
    try:
        _request(TRADING_BASE, f"/v2/orders/{urllib.parse.quote(order_id)}", method="DELETE")
    except AlpacaError as exc:
        # 404 / already-filled / already-canceled are non-fatal for our flow.
        logger.info("cancel_order %s ignored: %s", order_id, exc)


def submit_order(
    symbol: str,
    qty: int,
    side: str,
    order_type: str,
    time_in_force: str = "day",
    limit_price: float | None = None,
    stop_price: float | None = None,
    client_order_id: str | None = None,
) -> dict:
    """Single-leg order. For options: type market|limit|stop|stop_limit,
    time_in_force day (we re-arm resting orders daily so day is sufficient)."""
    body: dict = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
    }
    if limit_price is not None:
        body["limit_price"] = f"{limit_price:.2f}"
    if stop_price is not None:
        body["stop_price"] = f"{stop_price:.2f}"
    if client_order_id:
        body["client_order_id"] = client_order_id
    return _request(TRADING_BASE, "/v2/orders", method="POST", body=body)  # type: ignore[return-value]


def close_position(symbol: str) -> dict:
    """Market-close an entire position (used for the time stop)."""
    return _request(  # type: ignore[return-value]
        TRADING_BASE, f"/v2/positions/{urllib.parse.quote(symbol)}", method="DELETE"
    )


# ---- Options chain + quotes ----------------------------------------------

def list_option_contracts(
    underlying: str,
    expiration_gte: str,
    expiration_lte: str,
    strike_gte: float,
    strike_lte: float,
    option_type: str = "call",
    limit: int = 1000,
) -> list[dict]:
    """/v2/options/contracts metadata (free; no quote subscription needed).

    Dates are 'YYYY-MM-DD'. Returns active, tradable contracts.
    """
    q = urllib.parse.urlencode({
        "underlying_symbols": underlying.upper(),
        "status": "active",
        "type": option_type,
        "expiration_date_gte": expiration_gte,
        "expiration_date_lte": expiration_lte,
        "strike_price_gte": f"{strike_gte:.2f}",
        "strike_price_lte": f"{strike_lte:.2f}",
        "limit": limit,
    })
    res = _request(TRADING_BASE, f"/v2/options/contracts?{q}")
    if isinstance(res, dict):
        return res.get("option_contracts") or []
    return []


def latest_option_quote(symbol: str) -> dict | None:
    """Indicative-feed latest quote for one OCC option symbol.

    Returns {"bid": float, "ask": float} or None if no quote.
    """
    q = urllib.parse.urlencode({"symbols": symbol, "feed": OPTIONS_FEED})
    try:
        res = _request(DATA_BASE, f"/v1beta1/options/quotes/latest?{q}")
    except AlpacaError as exc:
        logger.info("option quote %s failed: %s", symbol, exc)
        return None
    quotes = res.get("quotes") if isinstance(res, dict) else None
    if not quotes or symbol not in quotes:
        return None
    qd = quotes[symbol]
    bid = qd.get("bp")
    ask = qd.get("ap")
    if ask is None:
        return None
    return {"bid": float(bid) if bid is not None else None, "ask": float(ask)}


__all__ = [
    "AlpacaError",
    "is_configured",
    "get_account",
    "get_clock",
    "list_positions",
    "list_option_positions",
    "list_orders",
    "get_order",
    "cancel_order",
    "submit_order",
    "close_position",
    "list_option_contracts",
    "latest_option_quote",
]
