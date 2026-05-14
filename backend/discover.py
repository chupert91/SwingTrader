"""S&P 500 discovery scan — find tickers currently meeting the user's signal.

Unlike `scan.py` (which detects σ-band CROSSES on the latest bar and fires alert
emails), this module checks **current state** — is the ticker currently sitting
past -Nσ with a positive-slope trend and oversold stoch? It's the "what setups
exist right now that I haven't seen?" panel, not the "did anything just trip"
pipeline.

Storage: latest results blob lives at KV key `swt:discoveries:latest`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

import pandas as pd

from backend import kv, watchlist
from backend.alert_engine import AlertRule
from backend.channels import REGRESSION_WINDOW, compute_channels
from backend.data import fetch_bars_bulk
from backend.indicators import stoch_rsi
from backend.sp500_tickers import SP500_TICKERS

logger = logging.getLogger(__name__)

KEY_DISCOVERIES = "swt:discoveries:latest"


def _ui_signal_rule() -> AlertRule | None:
    """Return the user's auto-synced signal rule from KV, or None if missing."""
    rules = kv.list_rules()
    for r in rules:
        if r.get("id") == "ui-signal":
            return AlertRule.from_dict(r)
    return None


def _check_current_state(rule: AlertRule, ticker: str, df: pd.DataFrame) -> dict | None:
    """Return a match dict if `ticker` currently meets `rule`'s confluence
    criteria as a long OR short candidate."""
    if len(df) < REGRESSION_WINDOW + 1:
        return None
    prepared = compute_channels(df, window=REGRESSION_WINDOW)
    k, _ = stoch_rsi(prepared["close"])
    cur = prepared.iloc[-1]

    sd = cur.get("sd_position")
    slope = cur.get("slope")
    k_val = k.iloc[-1] if len(k) > 0 else None
    if sd is None or pd.isna(sd):
        return None

    # Liquidity gate. Compute average dollar volume over last 20 bars.
    avg_dv_m = 0.0
    if "volume" in prepared.columns and len(prepared) >= 20:
        tail = prepared.iloc[-20:]
        dv = (tail["close"] * tail["volume"]).mean()
        if not pd.isna(dv):
            avg_dv_m = float(dv) / 1_000_000.0
    if rule.min_avg_volume_m > 0 and avg_dv_m < rule.min_avg_volume_m:
        return None

    threshold = abs(float(rule.entry_sigma))
    direction: Literal["long", "short"] | None = None

    if rule.side in ("long", "both") and sd <= -threshold:
        if rule.require_trend and (slope is None or pd.isna(slope) or slope <= 0):
            pass  # filter out
        elif (rule.require_stoch_extreme
              and (k_val is None or pd.isna(k_val) or k_val > rule.stoch_oversold)):
            pass
        else:
            direction = "long"

    if direction is None and rule.side in ("short", "both") and sd >= threshold:
        if rule.require_trend and (slope is None or pd.isna(slope) or slope >= 0):
            pass
        elif (rule.require_stoch_extreme
              and (k_val is None or pd.isna(k_val) or k_val < rule.stoch_overbought)):
            pass
        else:
            direction = "short"

    if direction is None:
        return None

    # Annualized slope % for the UI
    price = float(cur["close"])
    slope_annual_pct = 0.0
    if slope is not None and not pd.isna(slope) and price > 0:
        slope_annual_pct = float(slope) * 252.0 / price * 100.0

    return {
        "ticker": ticker.upper(),
        "direction": direction,
        "current_price": price,
        "sd_position": float(sd),
        "slope_annual_pct": slope_annual_pct,
        "stoch_rsi_k": float(k_val) if k_val is not None and not pd.isna(k_val) else None,
        "avg_dollar_volume_m": avg_dv_m,
    }


def _interestingness(m: dict) -> float:
    """Sort key: how extreme is this setup? Larger |σ| and stronger trend rank higher."""
    sigma_extreme = abs(m.get("sd_position") or 0.0)
    trend_strength = abs(m.get("slope_annual_pct") or 0.0)
    return sigma_extreme * (1.0 + trend_strength / 100.0)


def run_discovery(top_n: int = 30) -> dict:
    """Scan the S&P 500 universe (excluding the user's watchlist), check
    current-state confluence against the ui-signal rule, persist top-N
    matches to KV. Returns the same blob it just wrote.
    """
    rule = _ui_signal_rule()
    if rule is None:
        return {"ok": False, "reason": "no ui-signal rule in KV; open the chart page to sync one"}

    on_watchlist = {t.upper() for t in watchlist.get()}
    candidates = [t for t in SP500_TICKERS if t.upper() not in on_watchlist]

    # One bulk yfinance call for the whole universe (threads internally).
    try:
        bars_by_ticker = fetch_bars_bulk(candidates, period="14mo")
    except Exception as exc:
        logger.exception("bulk fetch failed")
        return {"ok": False, "reason": f"yfinance bulk fetch failed: {exc!r}"}

    matches: list[dict] = []
    check_errors: dict[str, str] = {}
    for tk, df in bars_by_ticker.items():
        if df.empty:
            continue
        try:
            m = _check_current_state(rule, tk, df)
        except Exception as exc:
            logger.exception("discover check failed for %s", tk)
            check_errors[tk] = repr(exc)
            continue
        if m is not None:
            matches.append(m)

    matches.sort(key=_interestingness, reverse=True)
    matches = matches[:top_n]

    blob = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": _today_iso(),
        "universe": "sp500_subset",
        "universe_size": len(candidates),
        "config_used": {
            "entry_sigma_magnitude": abs(float(rule.entry_sigma)),
            "require_trend": rule.require_trend,
            "require_stoch_extreme": rule.require_stoch_extreme,
            "stoch_oversold": rule.stoch_oversold,
            "stoch_overbought": rule.stoch_overbought,
            "side": rule.side,
            "min_avg_volume_m": rule.min_avg_volume_m,
        },
        "matches": matches,
        "fetched_count": len(bars_by_ticker),
        "check_error_count": len(check_errors),
    }
    kv.set_json(KEY_DISCOVERIES, blob)
    return blob


def get_latest() -> dict:
    return kv.get_json(KEY_DISCOVERIES, default={"matches": []}) or {"matches": []}


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


__all__ = ["run_discovery", "get_latest", "KEY_DISCOVERIES"]
