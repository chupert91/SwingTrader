"""Standalone scanner for the validated oversold fast-exit edge.

Screens the S&P universe on the LOG-price 252d regression channel for names
currently in the research-validated sweet spot:

    oversold, z in [-2.5, -2.0]   (log channel)

Rationale (see research/ arc + memory user-trades-options):
  - Oversold OTM-call mean-reversion is the user's repeatable edge.
  - The MODERATE extreme (2.0-2.5 sigma) has the best fast-win base rate;
    deeper/parabolic deviations are WORSE, so we cap the band, not just floor it.
  - Log channel is the theoretically-correct frame for equities and gave more
    events at comparable quality vs. linear.
  - No trend filter: the slope>30%/yr (trend-continuation) filter points at the
    WEAKEST regime for this fast-exit option trade, so it is deliberately omitted.

Historical reference for this band (183 S&P names x 5y; 5% OTM, 60 DTE):
  P(option +30% within 5 td) ~ 25% , within 10 td ~ 35% , ever <=60d ~ 60%
  median ~5 td to the +30% , median heat ~ -8% before the win.
Treat fresh FIRST-TOUCH names as prime (median ~5 td to target from the touch);
names that have sat in-zone for many bars have likely spent the fast window.

Each row also gets a same-day-breadth TIER (breadth = how many universe
oversold first-touches occurred on that name's touch date). From
research/cluster_breadth.py, P(+30% within 10td) by tier:
  WEAK  (1-2 lone)        ~29%  -> smallest size or skip
  OK    (3-7 small)       ~32%  -> normal size
  PRIME (8-19 moderate)   ~44%  -> best setup, size up
  PANIC (20+ mass flush)  ~34%  -> tradeable but ~-12% heat, scale in
Rows are ranked by tier first, then freshness.

This is a read-only screen. It does NOT touch the live KV rule / alert crons.

Run:
    python research/oversold_scanner.py
Outputs:
    research/out/oversold_candidates.csv
    (and a ranked table to stdout)
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.sp500_tickers import SP500_TICKERS  # noqa: E402

WINDOW = 252
SWEET_LO = -2.5          # most-negative edge of the sweet spot
SWEET_HI = -2.0          # least-negative edge (the -2 sigma trigger)
RV_LOOKBACK = 20
MIN_AVG_DOLLAR_VOL_M = 50.0   # matches the project's default liquidity gate
STALE_BARS = 5                # in-zone longer than this => fast window likely spent
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def log_channel_sd(closes: np.ndarray, window: int = WINDOW) -> tuple[np.ndarray, float, float]:
    """Mirror backend/channels.py but in LOG space.

    Fit OLS on the last `window` log-closes, project the line across all bars,
    sigma = stdev of in-window residuals. sd_position = (log px - fitted)/sigma.

    Returns (sd_position_series, annualized_log_slope_pct, centerline_price_at_last_bar).
    """
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


def realized_vol_pct(closes: np.ndarray, lookback: int = RV_LOOKBACK) -> float:
    if len(closes) < lookback + 1:
        return float("nan")
    logret = np.diff(np.log(closes[-(lookback + 1):]))
    return float(np.std(logret, ddof=1) * math.sqrt(252) * 100.0)


def bars_in_zone(sd: np.ndarray, hi: float = SWEET_HI) -> int:
    """How many consecutive recent bars have been at/below `hi` (i.e. z<=-2).
    0 would mean 'not in zone'; 1 means it just crossed in on the latest bar."""
    cnt = 0
    for v in sd[::-1]:
        if np.isnan(v) or v > hi:
            break
        cnt += 1
    return cnt


def avg_dollar_vol_m(df: pd.DataFrame, window: int = 20) -> float:
    if "volume" not in df.columns or len(df) < window:
        return 0.0
    tail = df.iloc[-window:]
    dv = (tail["close"] * tail["volume"]).mean()
    return 0.0 if pd.isna(dv) else float(dv) / 1_000_000.0


# Same-day breadth tiers from research/cluster_breadth.py (P(+30% within 10td)):
#   1-2 lone ~29%  |  3-7 small ~32%  |  8-19 moderate ~44%  |  20+ panic ~34%
# Rank order favors the calmer OK tier over PANIC when hit rates are ~equal
# (panic days take ~-12% heat vs ~-8% otherwise).
_TIER_RANK = {"PRIME": 0, "OK": 1, "PANIC": 2, "WEAK": 3, "?": 4}


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


def scan() -> pd.DataFrame:
    bars = fetch_bars_bulk(SP500_TICKERS, period="14mo")

    # Pass A: one log-channel pass per ticker. Accumulate universe-wide
    # oversold first-touch counts per date (breadth = market-flush proxy,
    # ALL z<=-2 first-touches, same definition as cluster_breadth.py but
    # computed with this scanner's projected-fit channel for consistency),
    # and collect current sweet-spot candidates pending their tier.
    breadth: Counter = Counter()
    pending: list[dict] = []
    for tk, df in bars.items():
        if df.empty or len(df) < WINDOW + 2:
            continue
        closes = df["close"].to_numpy(dtype=float)
        sd, slope_ann, center_px = log_channel_sd(closes)
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
        adv_m = avg_dollar_vol_m(df)
        if adv_m < MIN_AVG_DOLLAR_VOL_M:
            continue
        zone = bars_in_zone(sd)
        n = len(sd)
        touch_idx = n - zone  # first bar of the current in-zone streak
        touch_date = ts_dates[touch_idx] if 0 <= touch_idx < n else ts_dates[-1]
        first_touch = bool(sd[-2] > SWEET_HI and sd[-1] <= SWEET_HI)
        price = float(closes[-1])
        reversion_pct = (center_px / price - 1.0) * 100.0  # upside to centerline
        pending.append({
            "ticker": tk,
            "price": round(price, 2),
            "z_log": round(z, 2),
            "touch_date": str(touch_date),
            "_touch_date": touch_date,
            "first_touch": "YES" if first_touch else "",
            "bars_in_zone": zone,
            "status": "FRESH" if zone <= 1 else ("recent" if zone <= STALE_BARS else "STALE"),
            "reversion_to_mean_%": round(reversion_pct, 1),
            "realized_vol_%": round(realized_vol_pct(closes), 1),
            "log_slope_ann_%": round(slope_ann, 1),
            "avg_$vol_M": round(adv_m, 1),
        })

    if not pending:
        return pd.DataFrame()

    # Pass B: attach same-day breadth + tier (breadth as of each name's
    # own touch date, so the tier varies row to row).
    for c in pending:
        b = int(breadth.get(c["_touch_date"], 0))
        c["same_day_breadth"] = b
        c["tier"] = tier_for(b)
        del c["_touch_date"]

    out = pd.DataFrame(pending)
    out["_tier"] = out["tier"].map(_TIER_RANK).fillna(9).astype(int)
    out["_ft"] = (out["first_touch"] == "YES").astype(int)
    # Rank: tier (PRIME>OK>PANIC>WEAK), then fresh first-touch, then fewest
    # bars in zone, then higher realized vol.
    out = out.sort_values(
        by=["_tier", "_ft", "bars_in_zone", "realized_vol_%"],
        ascending=[True, False, True, False],
    ).drop(columns=["_tier", "_ft"]).reset_index(drop=True)
    # Readable column order.
    cols = ["ticker", "price", "z_log", "touch_date", "same_day_breadth", "tier",
            "first_touch", "bars_in_zone", "status", "reversion_to_mean_%",
            "realized_vol_%", "log_slope_ann_%", "avg_$vol_M"]
    return out[cols]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Scanning {len(SP500_TICKERS)} S&P names on the LOG 252d channel...")
    print(f"Sweet spot: {SWEET_LO} <= z <= {SWEET_HI}  |  liquidity >= ${MIN_AVG_DOLLAR_VOL_M}M ADV  |  oversold/call only\n")
    df = scan()
    if df.empty:
        print("No names in the sweet spot today. (Empty days are expected for a high-precision band.)")
        return
    asof = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"As of {asof} - {len(df)} candidate(s):\n")
    print(df.to_string(index=False))
    print("\nTier (same-day breadth, hist. P(+30% in 10td)):  "
          "PRIME 8-19 ~44%  >  OK 3-7 ~32%  >  PANIC 20+ ~34% (more heat)  >  WEAK 1-2 ~29%")
    print("Within tier: fresh first-touch first. STALE (in-zone > "
          f"{STALE_BARS} bars) likely past the ~5td fast-reversal window.")
    out_csv = os.path.join(OUT_DIR, "oversold_candidates.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  -> {out_csv}")


if __name__ == "__main__":
    main()
