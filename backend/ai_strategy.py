"""Signal + contract selection for the autonomous AI options trader.

The signal is a backend port of research/oversold_scanner.py (the validated
edge): LOG 252-day regression channel, fresh first-touch into the
z in [-2.5, -2.0] sweet spot, S&P universe, liquidity-gated, ranked by
same-day-breadth tier then freshness. Kept in-tree (not importing the
research script) so the production path has no dependency on research/.

Rationale lives in research/oversold_playbook.md and the user-trades-options
memory; do not widen the band or add a trend filter — both were tested and
make this specific OTM-2mo trade worse.
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd

from backend.data import fetch_bars_bulk
from backend.sp500_tickers import SP500_TICKERS

WINDOW = 252
SWEET_LO = -2.5
SWEET_HI = -2.0
RV_LOOKBACK = 20
MIN_AVG_DOLLAR_VOL_M = 50.0
STALE_BARS = 5

_TIER_RANK = {"PRIME": 0, "OK": 1, "PANIC": 2, "WEAK": 3, "?": 4}


def _log_channel_sd(closes: np.ndarray, window: int = WINDOW):
    n = len(closes)
    if n < window:
        return np.full(n, np.nan), 0.0, float("nan")
    y_all = np.log(closes)
    y = y_all[-window:]
    x = np.arange(window, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    sigma = float(np.std(y - fit, ddof=1))
    if sigma <= 0:
        return np.full(n, np.nan), 0.0, float("nan")
    full_x = np.arange(n, dtype=float) - (n - window)
    full_fit = slope * full_x + intercept
    sd = (y_all - full_fit) / sigma
    slope_ann_pct = (math.exp(slope * 252.0) - 1.0) * 100.0
    center_price_last = float(math.exp(full_fit[-1]))
    return sd, slope_ann_pct, center_price_last


def _realized_vol_pct(closes: np.ndarray, lookback: int = RV_LOOKBACK) -> float:
    if len(closes) < lookback + 1:
        return float("nan")
    logret = np.diff(np.log(closes[-(lookback + 1):]))
    return float(np.std(logret, ddof=1) * math.sqrt(252) * 100.0)


def _bars_in_zone(sd: np.ndarray, hi: float = SWEET_HI) -> int:
    cnt = 0
    for v in sd[::-1]:
        if np.isnan(v) or v > hi:
            break
        cnt += 1
    return cnt


def _avg_dollar_vol_m(df: pd.DataFrame, window: int = 20) -> float:
    if "volume" not in df.columns or len(df) < window:
        return 0.0
    tail = df.iloc[-window:]
    dv = (tail["close"] * tail["volume"]).mean()
    return 0.0 if pd.isna(dv) else float(dv) / 1_000_000.0


def tier_for(breadth: int) -> str:
    if breadth >= 20:
        return "PANIC"
    if breadth >= 8:
        return "PRIME"
    if breadth >= 3:
        return "OK"
    if breadth >= 1:
        return "WEAK"
    return "?"


def scan_candidates() -> list[dict]:
    """Ranked oversold-call candidates from the current S&P snapshot.

    WEAK tier is included in the output but the trader skips it (smallest
    size / skip per the playbook); ranking is tier-first then freshness.
    """
    bars = fetch_bars_bulk(SP500_TICKERS, period="14mo")
    breadth: Counter = Counter()
    pending: list[dict] = []

    for tk, df in bars.items():
        if df is None or df.empty or len(df) < WINDOW + 2:
            continue
        closes = df["close"].to_numpy(dtype=float)
        sd, slope_ann, center_px = _log_channel_sd(closes)
        if np.isnan(sd[-1]):
            continue
        ts_dates = pd.to_datetime(df["timestamp"]).dt.date.to_numpy()
        for t in range(1, len(sd)):
            if np.isnan(sd[t]) or np.isnan(sd[t - 1]):
                continue
            if sd[t] <= SWEET_HI and sd[t - 1] > SWEET_HI:
                breadth[ts_dates[t]] += 1

        z = float(sd[-1])
        if not (SWEET_LO <= z <= SWEET_HI):
            continue
        adv_m = _avg_dollar_vol_m(df)
        if adv_m < MIN_AVG_DOLLAR_VOL_M:
            continue
        zone = _bars_in_zone(sd)
        n = len(sd)
        touch_idx = n - zone
        touch_date = ts_dates[touch_idx] if 0 <= touch_idx < n else ts_dates[-1]
        first_touch = bool(sd[-2] > SWEET_HI and sd[-1] <= SWEET_HI)
        price = float(closes[-1])
        reversion_pct = (center_px / price - 1.0) * 100.0
        pending.append({
            "ticker": tk,
            "price": round(price, 2),
            "z_log": round(z, 2),
            "touch_date": str(touch_date),
            "_touch_date": touch_date,
            "first_touch": first_touch,
            "bars_in_zone": zone,
            "status": "FRESH" if zone <= 1 else ("recent" if zone <= STALE_BARS else "STALE"),
            "reversion_to_mean_pct": round(reversion_pct, 1),
            "realized_vol_pct": round(_realized_vol_pct(closes), 1),
            "log_slope_ann_pct": round(slope_ann, 1),
            "avg_dollar_vol_m": round(adv_m, 1),
        })

    for c in pending:
        b = int(breadth.get(c["_touch_date"], 0))
        c["same_day_breadth"] = b
        c["tier"] = tier_for(b)
        del c["_touch_date"]

    pending.sort(key=lambda c: (
        _TIER_RANK.get(c["tier"], 9),
        0 if c["first_touch"] else 1,
        c["bars_in_zone"],
        -c["realized_vol_pct"],
    ))
    return pending


# ---- Contract selection ---------------------------------------------------

def pick_contract(
    underlying: str,
    price: float,
    *,
    otm_pct: float,
    dte_min: int,
    dte_max: int,
    today: date | None = None,
) -> dict | None:
    """Choose ~otm_pct OTM call expiring within [dte_min, dte_max] DTE.

    Strategy: take the earliest expiration in the DTE band (least theta
    over-pay per the playbook), then the strike closest to the OTM target.
    Returns {symbol, strike, expiration, dte} or None if nothing fits.
    Requires Alpaca credentials (queries the contract list).
    """
    from backend import alpaca_trading as at

    today = today or date.today()
    exp_lo = (today + timedelta(days=dte_min)).isoformat()
    exp_hi = (today + timedelta(days=dte_max)).isoformat()
    target_strike = price * (1.0 + otm_pct / 100.0)
    # Widen the strike window so we still find something if the exact 5%
    # strike isn't listed; we re-pick the closest below.
    strike_lo = target_strike * 0.95
    strike_hi = target_strike * 1.15

    contracts = at.list_option_contracts(
        underlying,
        expiration_gte=exp_lo,
        expiration_lte=exp_hi,
        strike_gte=strike_lo,
        strike_lte=strike_hi,
        option_type="call",
    )
    if not contracts:
        return None

    def dte_of(c: dict) -> int:
        try:
            y, m, d = (int(x) for x in c["expiration_date"].split("-"))
            return (date(y, m, d) - today).days
        except Exception:
            return 10_000

    valid = [c for c in contracts if c.get("strike_price") and c.get("expiration_date")]
    if not valid:
        return None
    earliest_exp = min(c["expiration_date"] for c in valid)
    same_exp = [c for c in valid if c["expiration_date"] == earliest_exp]
    best = min(same_exp, key=lambda c: abs(float(c["strike_price"]) - target_strike))
    return {
        "symbol": best["symbol"],
        "strike": float(best["strike_price"]),
        "expiration": best["expiration_date"],
        "dte": dte_of(best),
    }


__all__ = ["scan_candidates", "pick_contract", "tier_for"]
