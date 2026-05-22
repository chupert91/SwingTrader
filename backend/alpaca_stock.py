"""Alpaca paper Trading adapter for the long-stock bot (stock_trader).

Deliberately separate from alpaca_trading.py: the stock bot trades a SECOND
Alpaca paper account, and isolating its broker client means a bug here can
never reach the options account. Stock orders only - no options.

Credentials are read ONLY from STOCK_ALPACA_* env vars, so this client
cannot see the options account's keys (and vice versa):
    STOCK_ALPACA_KEY_ID         second paper account API key id
    STOCK_ALPACA_SECRET_KEY     second paper account API secret
    STOCK_ALPACA_TRADING_BASE   trading base URL (default: paper endpoint)

Dependency-free (urllib), matching alpaca_trading.py.
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
    "STOCK_ALPACA_TRADING_BASE", "https://paper-api.alpaca.markets"
).rstrip("/")


class AlpacaError(RuntimeError):
    """Raised when the Alpaca API returns an error or no credentials exist."""


def _credentials() -> tuple[str, str] | None:
    key = os.environ.get("STOCK_ALPACA_KEY_ID")
    secret = os.environ.get("STOCK_ALPACA_SECRET_KEY")
    if not key or not secret:
        return None
    return key, secret


def is_configured() -> bool:
    return _credentials() is not None


def _request(path: str, method: str = "GET", body: dict | None = None) -> dict | list:
    creds = _credentials()
    if creds is None:
        raise AlpacaError("stock-bot Alpaca credentials not configured")
    key, secret = creds
    url = f"{TRADING_BASE}{path}"
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
        logger.warning("alpaca-stock %s %s -> HTTP %s %s", method, path, exc.code, detail)
        raise AlpacaError(f"HTTP {exc.code}: {detail or exc.reason}") from exc
    except Exception as exc:
        logger.warning("alpaca-stock %s %s failed: %s", method, path, exc)
        raise AlpacaError(str(exc)) from exc
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise AlpacaError(f"bad JSON from {path}") from exc


# ---- Account / clock ------------------------------------------------------

def get_account() -> dict:
    """Raw /v2/account. Fields used: equity, cash, buying_power."""
    return _request("/v2/account")  # type: ignore[return-value]


def get_clock() -> dict:
    """/v2/clock - authoritative market-open state. Fields: is_open, timestamp."""
    return _request("/v2/clock")  # type: ignore[return-value]


# ---- Positions / orders ---------------------------------------------------

def list_positions() -> list[dict]:
    res = _request("/v2/positions")
    return res if isinstance(res, list) else []


def list_orders(status: str = "all", limit: int = 200, nested: bool = True) -> list[dict]:
    q = urllib.parse.urlencode(
        {"status": status, "limit": limit, "nested": str(nested).lower()}
    )
    res = _request(f"/v2/orders?{q}")
    return res if isinstance(res, list) else []


def get_order(order_id: str) -> dict:
    return _request(f"/v2/orders/{urllib.parse.quote(order_id)}")  # type: ignore[return-value]


def cancel_order(order_id: str) -> None:
    try:
        _request(f"/v2/orders/{urllib.parse.quote(order_id)}", method="DELETE")
    except AlpacaError as exc:
        # 404 / already-filled / already-canceled are non-fatal for our flow.
        logger.info("cancel_order %s ignored: %s", order_id, exc)


def submit_order(
    symbol: str,
    qty: int,
    side: str,
    order_type: str = "limit",
    time_in_force: str = "day",
    limit_price: float | None = None,
    client_order_id: str | None = None,
) -> dict:
    """Single-leg stock order. side buy|sell; type market|limit;
    time_in_force day for the entry, gtc for the resting +target% sell."""
    body: dict = {
        "symbol": symbol.upper(),
        "qty": str(int(qty)),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
    }
    if limit_price is not None:
        body["limit_price"] = f"{limit_price:.2f}"
    if client_order_id:
        body["client_order_id"] = client_order_id
    return _request("/v2/orders", method="POST", body=body)  # type: ignore[return-value]


def close_position(symbol: str) -> dict:
    """Market-close an entire stock position (used for the time stop)."""
    return _request(
        f"/v2/positions/{urllib.parse.quote(symbol)}", method="DELETE"
    )  # type: ignore[return-value]


__all__ = [
    "AlpacaError", "is_configured",
    "get_account", "get_clock",
    "list_positions", "list_orders", "get_order", "cancel_order",
    "submit_order", "close_position",
]
