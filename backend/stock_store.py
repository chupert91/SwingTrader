"""KV-backed persistence for the long-stock bot (stock_trader).

A parallel of ai_store.py for the second bot. Fully separate KV namespace
(swt:stock:*) so the two bots' state never collides. Simpler than the
options bot: plain stock trades, no contract/DTE/IV fields.

Keys:
    swt:stock:settings   dict          - user-tunable config (UI source of truth)
    swt:stock:trades     list[dict]    - every trade, newest last (capped)
    swt:stock:equity     list[dict]    - paper-equity snapshots (capped)
    swt:stock:runlog     list[dict]    - cron run audit log, newest first (capped)
    swt:stock:state      dict          - runtime markers (entry_day, entries_today)
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from backend import kv

KEY_SETTINGS = "swt:stock:settings"
KEY_TRADES = "swt:stock:trades"
KEY_EQUITY = "swt:stock:equity"
KEY_RUNLOG = "swt:stock:runlog"
KEY_STATE = "swt:stock:state"

MAX_TRADES = 500
MAX_EQUITY = 1000
MAX_RUNLOG = 200

# Kill-switch defaults OFF: like the options bot, the autonomous stock bot
# must not fire orders until the user enables it in the UI and has watched
# the run log behave.
#
# Defaults are thesis-2-faithful (research/thesis2/THESIS.md): the validated
# long-stock implementation of the -3sigma capitulation strategy -
#   entry  : 252-day LOG-channel z <= -3.0 FIRST touch,
#            AND drawdown from the 1-year high <= -30%,
#            AND RSI(14) >= 30 at the signal bar;
#   exit   : +20% profit target (resting sell limit), else a 45-trading-day
#            time stop. No hard disaster stop - the thesis exit study used
#            target + time-stop only (worst 45d outcome was ~-14%).
DEFAULT_SETTINGS: dict = {
    "enabled": False,             # global kill switch
    "entry_z": -3.0,              # log-channel z first-touch trigger
    "drawdown_max_pct": -30.0,    # require drawdown from 1y high <= this (%)
    "rsi_min": 30.0,              # require RSI(14) >= this at the signal
    "profit_target_pct": 20.0,    # resting sell limit at fill*(1+target/100)
    "time_stop_days": 45,         # close if still open after N trading days
    "position_size_pct": 12.0,    # % of account equity per position
    "max_concurrent": 8,          # max simultaneous open stock positions
    "max_entries_per_day": 3,     # cap new entries opened in a single day
    "entry_limit_buffer_pct": 0.5,  # buy limit = ask*(1+buffer/100)
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- Settings -------------------------------------------------------------

def get_settings() -> dict:
    raw = kv.get_json(KEY_SETTINGS, default=None)
    merged = dict(DEFAULT_SETTINGS)
    if isinstance(raw, dict):
        merged.update({k: raw[k] for k in raw if k in DEFAULT_SETTINGS})
    return merged


def save_settings(patch: dict) -> dict:
    cur = get_settings()
    for k, v in (patch or {}).items():
        if k in DEFAULT_SETTINGS:
            cur[k] = v
    kv.set_json(KEY_SETTINGS, cur)
    return cur


# ---- Trades ---------------------------------------------------------------

def list_trades() -> list[dict]:
    return kv.get_json(KEY_TRADES, default=[]) or []


def _save_trades(trades: list[dict]) -> None:
    kv.set_json(KEY_TRADES, trades[-MAX_TRADES:])


def new_trade(ticker: str, signal: dict) -> dict:
    t = {
        "id": uuid.uuid4().hex[:12],
        "status": "pending_entry",
        "ticker": ticker.upper(),
        "signal": signal,        # {z, drawdown_pct, rsi, price_at_signal, touch_date}
        "entry": None,           # {order_id, limit_price, fill_price, qty, submitted_at, filled_at}
        "tp": None,              # {order_id, limit_price, status} - the +target% resting sell
        "exit": None,            # {reason, fill_price, filled_at, order_id, submitted_at}
        "pnl": None,             # {realized_usd, realized_pct}
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    trades = list_trades()
    trades.append(t)
    _save_trades(trades)
    return t


def update_trade(trade_id: str, patch: dict) -> dict | None:
    trades = list_trades()
    out = None
    for t in trades:
        if t.get("id") == trade_id:
            t.update(patch)
            t["updated_at"] = now_iso()
            out = t
            break
    if out is not None:
        _save_trades(trades)
    return out


def get_trade(trade_id: str) -> dict | None:
    for t in list_trades():
        if t.get("id") == trade_id:
            return t
    return None


def open_trades() -> list[dict]:
    return [t for t in list_trades() if t.get("status") in ("pending_entry", "open", "closing")]


def active_position_count() -> int:
    """Open + pending-entry + closing - all consume a concurrency slot."""
    return len(open_trades())


# ---- Equity snapshots -----------------------------------------------------

def append_equity(equity: float, cash: float, realized_cum: float) -> None:
    series = kv.get_json(KEY_EQUITY, default=[]) or []
    series.append({
        "t": now_iso(),
        "equity": round(float(equity), 2),
        "cash": round(float(cash), 2),
        "realized_cum": round(float(realized_cum), 2),
    })
    kv.set_json(KEY_EQUITY, series[-MAX_EQUITY:])


def equity_series() -> list[dict]:
    return kv.get_json(KEY_EQUITY, default=[]) or []


# ---- Runtime state (non-user markers, e.g. last entry-eval day) ----------

def get_state() -> dict:
    return kv.get_json(KEY_STATE, default={}) or {}


def set_state(patch: dict) -> dict:
    cur = get_state()
    cur.update(patch or {})
    kv.set_json(KEY_STATE, cur)
    return cur


# ---- Run log --------------------------------------------------------------

def log_run(entry: dict) -> None:
    entry = dict(entry)
    entry.setdefault("t", now_iso())
    entry.setdefault("ts", time.time())
    log = kv.get_json(KEY_RUNLOG, default=[]) or []
    log.insert(0, entry)
    kv.set_json(KEY_RUNLOG, log[:MAX_RUNLOG])


def run_log(limit: int = 100) -> list[dict]:
    return (kv.get_json(KEY_RUNLOG, default=[]) or [])[:limit]


__all__ = [
    "DEFAULT_SETTINGS",
    "get_settings", "save_settings",
    "list_trades", "new_trade", "update_trade", "get_trade",
    "open_trades", "active_position_count",
    "get_state", "set_state",
    "append_equity", "equity_series",
    "log_run", "run_log",
    "now_iso",
]
