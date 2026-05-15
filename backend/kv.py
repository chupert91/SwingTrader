"""Storage abstraction for rules, dedup, and history.

Two backends with the same interface:
- Vercel KV (REST API) when KV_REST_API_URL + KV_REST_API_TOKEN are set
- Local JSON files (state/ dir) otherwise — used for dev and tests

Keys we use:
    swt:rules                  list[dict]        — all AlertRule dicts
    swt:lastfired              dict[str, str]    — dedup_key -> bar_date
    swt:history                list[dict]        — most-recent-first RuleSignal dicts (capped at MAX_HISTORY)
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

KEY_RULES = "swt:rules"
KEY_LAST_FIRED = "swt:lastfired"
KEY_HISTORY = "swt:history"
KEY_DRAWINGS = "swt:drawings"
KEY_INDICATORS_ACTIVE = "swt:indicators_active"
MAX_HISTORY = 200

_EMPTY_DRAWING = {"hlines": [], "trendlines": [], "fibs": [], "trades": []}

# Local-file fallback location
_LOCAL_DIR = Path(__file__).resolve().parent.parent / "state"
_LOCAL_KV_FILE = _LOCAL_DIR / "kv.json"
_local_lock = Lock()


def _kv_url() -> str | None:
    return os.environ.get("KV_REST_API_URL")


def _kv_token() -> str | None:
    return os.environ.get("KV_REST_API_TOKEN")


def using_vercel_kv() -> bool:
    return bool(_kv_url() and _kv_token())


def _is_serverless() -> bool:
    """True on Vercel/Lambda-like environments with a read-only filesystem."""
    return bool(os.environ.get("VERCEL"))


def get_json(key: str, default=None):
    if using_vercel_kv():
        return _kv_get_json(key, default)
    if _is_serverless():
        # No KV configured AND no local fs to fall back to. Treat as empty
        # so the app boots; the user must provision Vercel KV to enable
        # persistence (Storage tab in the dashboard).
        return default
    return _local_get_json(key, default)


def set_json(key: str, value) -> None:
    if using_vercel_kv():
        _kv_set_json(key, value)
        return
    if _is_serverless():
        logger.warning(
            "KV not configured (no KV_REST_API_URL/TOKEN env vars). "
            "Dropping write to %s — provision Vercel KV in the project's "
            "Storage tab to enable persistence.", key,
        )
        return
    _local_set_json(key, value)


# Vercel KV REST helpers ---------------------------------------------------

def _kv_request(path: str, method: str = "GET", body: bytes | None = None) -> dict:
    url = f"{_kv_url().rstrip('/')}/{path}"
    req = urllib.request.Request(url, method=method, data=body)
    req.add_header("Authorization", f"Bearer {_kv_token()}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        logger.exception("KV request failed (%s %s): %s", method, url, exc)
        raise


def _kv_get_json(key: str, default):
    # Vercel KV REST: GET /get/<key> -> {"result": "<json-string-or-null>"}
    encoded = urllib.parse.quote(key, safe="")
    resp = _kv_request(f"get/{encoded}")
    raw = resp.get("result")
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw  # already deserialized
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw


def _kv_set_json(key: str, value) -> None:
    # Vercel KV REST: POST /set/<key> with JSON body of the value
    encoded = urllib.parse.quote(key, safe="")
    payload = json.dumps(value).encode("utf-8")
    _kv_request(f"set/{encoded}", method="POST", body=payload)


# Local JSON fallback ------------------------------------------------------

def _local_load() -> dict:
    if not _LOCAL_KV_FILE.exists():
        return {}
    try:
        return json.loads(_LOCAL_KV_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _local_save(data: dict) -> None:
    _LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _LOCAL_KV_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(_LOCAL_KV_FILE)


def _local_get_json(key: str, default):
    with _local_lock:
        data = _local_load()
        return data.get(key, default)


def _local_set_json(key: str, value) -> None:
    with _local_lock:
        data = _local_load()
        data[key] = value
        _local_save(data)


# Domain helpers — used by both the scan endpoint and the rules API --------

def list_rules() -> list[dict]:
    return get_json(KEY_RULES, default=[]) or []


def save_rules(rules: list[dict]) -> None:
    set_json(KEY_RULES, rules)


def last_fired() -> dict[str, str]:
    return get_json(KEY_LAST_FIRED, default={}) or {}


def mark_fired(dedup_key: str, bar_date: str) -> None:
    cur = last_fired()
    cur[dedup_key] = bar_date
    set_json(KEY_LAST_FIRED, cur)


def history(limit: int = 50) -> list[dict]:
    h = get_json(KEY_HISTORY, default=[]) or []
    return h[:limit]


def append_history(entry: dict) -> None:
    h = get_json(KEY_HISTORY, default=[]) or []
    h.insert(0, entry)
    set_json(KEY_HISTORY, h[:MAX_HISTORY])


# Drawings — single blob keyed by ticker.
def _all_drawings() -> dict:
    return get_json(KEY_DRAWINGS, default={}) or {}


def get_ticker_drawings(ticker: str) -> dict:
    return _all_drawings().get(ticker.upper(), dict(_EMPTY_DRAWING))


def set_ticker_drawings(ticker: str, blob: dict) -> None:
    data = _all_drawings()
    data[ticker.upper()] = blob
    set_json(KEY_DRAWINGS, data)


# Active custom-indicator selection (global, not per-ticker). Each entry:
#   {"indicator_id": "supertrend", "params": {...}}
def get_active_indicators() -> list[dict]:
    raw = get_json(KEY_INDICATORS_ACTIVE, default=[]) or []
    return raw if isinstance(raw, list) else []


def set_active_indicators(active: list[dict]) -> None:
    set_json(KEY_INDICATORS_ACTIVE, active)


__all__ = [
    "using_vercel_kv",
    "list_rules", "save_rules",
    "last_fired", "mark_fired",
    "history", "append_history",
    "get_ticker_drawings", "set_ticker_drawings",
    "get_active_indicators", "set_active_indicators",
]
