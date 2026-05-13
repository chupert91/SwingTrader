"""Rule-based alert scan — shared by local FastAPI and Vercel serverless.

One pass: load enabled rules from KV, fetch bars once per unique ticker,
evaluate every rule, dedup, send emails, persist to history.
"""
from __future__ import annotations

import logging

from backend import kv, watchlist
from backend.alert_engine import AlertRule, detect
from backend.data import fetch_bars
from backend.resend_email import send_signal

logger = logging.getLogger(__name__)


def run_scan() -> dict:
    """Synchronous scan pass. Returns a JSON-safe summary.

    The frontend syncs a canonical rule (id="ui-signal") on every
    Signal/Watchlist change, so we no longer seed a hard-coded default.
    No rules = no alerts; the user just opens the chart page to set up.
    """
    rules_raw = kv.list_rules()
    rules = [AlertRule.from_dict(r) for r in rules_raw if r.get("enabled", True)]
    if not rules:
        return {"ok": True, "scanned_tickers": 0, "fired": [], "note": "no enabled rules"}

    # Ticker universe = the user's live watchlist (synced from any device).
    # Rule.tickers is no longer used as the source of truth — that field is
    # kept in storage for compatibility with the frontend's rule-sync payload.
    all_tickers = [t.upper() for t in watchlist.get()]
    if not all_tickers:
        return {"ok": True, "scanned_tickers": 0, "fired": [], "note": "empty watchlist"}

    bar_cache: dict[str, object] = {}
    fetch_errors: dict[str, str] = {}
    for tk in all_tickers:
        try:
            df = fetch_bars(tk, period="14mo")
            if not df.empty:
                bar_cache[tk] = df
            else:
                fetch_errors[tk] = "empty"
        except Exception as exc:
            fetch_errors[tk] = repr(exc)
            logger.exception("fetch_bars failed for %s", tk)

    last = kv.last_fired()
    fired: list[dict] = []
    skipped_dups: list[str] = []

    for rule in rules:
        for tk in all_tickers:
            df = bar_cache.get(tk.upper())
            if df is None:
                continue
            try:
                sigs = detect(rule, tk, df)
            except Exception:
                logger.exception("detect() failed for rule=%s ticker=%s", rule.id, tk)
                continue
            for sig in sigs:
                key = sig.dedup_key()
                if last.get(key) == sig.bar_date:
                    skipped_dups.append(key)
                    continue
                email_ok = False
                if rule.notify_email:
                    email_ok = send_signal(sig, rule.notify_email)
                kv.mark_fired(key, sig.bar_date)
                entry = sig.to_dict()
                entry["email_sent"] = email_ok
                kv.append_history(entry)
                fired.append(entry)

    return {
        "ok": True,
        "scanned_tickers": len(bar_cache),
        "fired": fired,
        "skipped_dups": len(skipped_dups),
        "fetch_errors": fetch_errors,
        "rules_active": len(rules),
    }


__all__ = ["run_scan"]
