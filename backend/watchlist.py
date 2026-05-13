"""Server-side mirror of the user's watchlist (used by the scheduler).

Frontend syncs its localStorage watchlist to /api/watchlist (PUT). Stored
in KV (Vercel KV in prod, local state/kv.json fallback in dev) so it
survives restarts AND is shared across serverless function invocations.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend import kv

KEY_WATCHLIST = "swt:watchlist"
_DEFAULT = ["AAPL", "MSFT", "NVDA", "PLTR", "AMD"]

# Legacy file path — read once, then migrate to KV.
_LEGACY_FILE = Path(__file__).resolve().parent.parent / "state" / "watchlist.json"


def _migrate_legacy_file() -> list[str] | None:
    """One-time: if the old state/watchlist.json exists and KV is empty,
    return its contents so the caller can seed KV. Returns None if nothing
    to migrate."""
    if not _LEGACY_FILE.exists():
        return None
    try:
        data = json.loads(_LEGACY_FILE.read_text(encoding="utf-8"))
        return [str(t).upper() for t in data if isinstance(t, str)]
    except (json.JSONDecodeError, OSError):
        return None


def get() -> list[str]:
    raw = kv.get_json(KEY_WATCHLIST, default=None)
    if raw is None:
        # KV empty — try a one-time migrate from the legacy file.
        legacy = _migrate_legacy_file()
        if legacy:
            kv.set_json(KEY_WATCHLIST, legacy)
            return legacy
        return list(_DEFAULT)
    return [str(t).upper() for t in raw if isinstance(t, str)]


def set_(tickers: list[str]) -> list[str]:
    cleaned = [t.upper().strip() for t in tickers if isinstance(t, str) and t.strip()]
    seen = set()
    deduped = []
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    kv.set_json(KEY_WATCHLIST, deduped)
    return deduped
