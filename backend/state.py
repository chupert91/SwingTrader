"""JSON-file persistence for alert dedup + history.

Stored at state/alerts.json. Two top-level keys:
- "last_fired": map of "{TICKER}_{direction}" -> bar_date that triggered.
                Used to dedup: same ticker/direction won't fire twice on same bar.
- "history": list of {ticker, direction, bar_date, price, sd_position,
                      confidence, confirmations, fired_at} dicts.
                      Most recent first. Trimmed to MAX_HISTORY entries.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from backend.alerts import Signal

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
STATE_FILE = STATE_DIR / "alerts.json"
MAX_HISTORY = 200

_lock = Lock()


def _load() -> dict:
    if not STATE_FILE.exists():
        return {"last_fired": {}, "history": []}
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("last_fired", {})
        data.setdefault("history", [])
        return data
    except (json.JSONDecodeError, OSError):
        return {"last_fired": {}, "history": []}


def _save(data: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(STATE_FILE)


def should_fire(signal: Signal) -> bool:
    """Return True if this signal hasn't fired yet for this bar."""
    with _lock:
        data = _load()
        key = f"{signal.ticker}_{signal.direction}"
        last_date = data["last_fired"].get(key)
        return last_date != signal.bar_date


def record(signal: Signal) -> None:
    """Mark signal as fired and append to history."""
    with _lock:
        data = _load()
        key = f"{signal.ticker}_{signal.direction}"
        data["last_fired"][key] = signal.bar_date
        entry = asdict(signal)
        entry["fired_at"] = datetime.now(timezone.utc).isoformat()
        data["history"].insert(0, entry)
        data["history"] = data["history"][:MAX_HISTORY]
        _save(data)


def history(limit: int = 50) -> list[dict]:
    with _lock:
        return _load()["history"][:limit]
