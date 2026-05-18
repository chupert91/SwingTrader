"""KV-backed persistence for the autonomous AI options trader.

All state lives in Vercel KV (Upstash) so it survives the serverless,
read-only filesystem. Lists are capped to keep KV values bounded.

Keys:
    swt:ai:settings   dict          — user-tunable config (UI source of truth)
    swt:ai:trades     list[dict]    — every trade, newest last (capped)
    swt:ai:equity     list[dict]    — paper-equity snapshots (capped)
    swt:ai:runlog     list[dict]    — cron run audit log, newest first (capped)
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from backend import kv

KEY_SETTINGS = "swt:ai:settings"
KEY_TRADES = "swt:ai:trades"
KEY_EQUITY = "swt:ai:equity"
KEY_RUNLOG = "swt:ai:runlog"
KEY_STATE = "swt:ai:state"

MAX_TRADES = 500
MAX_EQUITY = 1000
MAX_RUNLOG = 200

# Kill-switch defaults OFF on purpose: even on paper, an autonomous trader
# must not start firing orders until the user explicitly enables it in the
# UI and has watched the run log prove it behaves.
DEFAULT_SETTINGS: dict = {
    "enabled": False,            # global kill switch
    "premium_cap_usd": 2000.0,   # max premium per single contract
    "max_concurrent": 2,         # max simultaneous open option positions
    "disaster_stop_enabled": True,
    "disaster_stop_pct": 50.0,   # native stop at entry*(1 - pct/100)
    "take_profit_pct": 30.0,     # resting sell limit at entry*(1 + pct/100)
    "time_stop_days": 10,        # close if still open after N trading days
    "otm_pct": 5.0,              # target strike ~ price*(1+otm/100)
    "dte_min": 45,
    "dte_max": 60,
    "entry_limit_buffer_pct": 1.0,  # pay up to ask*(1+buffer/100)
    # Optional Stoch RSI timing overlay. Untested vs. the validated edge
    # (see ai_strategy docstring) so it ships OFF. mode:
    #   "off"     - ignored
    #   "prefer"  - oversold names rank first within their tier
    #   "require" - drop names whose %K > stoch_oversold_max
    "stoch_rsi_mode": "off",
    "stoch_oversold_max": 30.0,  # %K <= this counts as oversold
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
        "signal": signal,
        "contract": None,
        "entry": None,
        "tp": None,
        "disaster_stop": None,
        "exit": None,
        "pnl": None,
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
    """Open + pending-entry — both consume a concurrency slot."""
    return len([t for t in list_trades() if t.get("status") in ("pending_entry", "open", "closing")])


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
