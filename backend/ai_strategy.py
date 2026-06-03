"""Signal + contract selection for the autonomous AI options trader.

The signal is a backend port of research/oversold_scanner.py (the validated
edge): LOG 252-day regression channel, fresh first-touch into the
z in [-2.5, -2.0] sweet spot, liquidity-gated, ranked by same-day-breadth
tier then freshness. Universe is the high-IV thematic basket in
backend/volatile_universe (AI / quantum / crypto / critical-minerals /
EV / fintech / biotech / consumer apps / tech megacap) — NOT SP500.
The universe swap was driven by research/out/volatile_universe_sweep.txt:
same engine, same exits, only the universe changed, PF went 1.41 -> 2.32,
matching the user's real 2025 PF of 2.09 (research/out/personal_trade_audit.txt).

Rationale lives in research/oversold_playbook.md. Tested-and-rejected (see
the playbook "What to avoid" table): more-extreme z bands and the
slope/trend-direction filter both make this specific OTM-2mo trade worse —
do not re-add those.

The optional Stoch RSI overlay (stoch_mode) is a *timing/momentum*
refinement that was NOT part of that research. It is unproven, not
contradicted-by-research, and ships off by default. Keep it opt-in until
the AI-page closed-trade stats prove it earns its keep.
"""
from __future__ import annotations

import logging
import math
import os
from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd

from backend.data import fetch_bars_bulk
from backend.indicators import rsi, stoch_rsi
from backend.volatile_universe import universe as _scan_universe

logger = logging.getLogger(__name__)

WINDOW = 252
# When daily bars come from the Alpaca IEX fallback (yfinance rate-limited),
# the volume column is IEX-only -- roughly a few % of consolidated tape. The
# dollar-volume liquidity gate is calibrated for yfinance's consolidated
# volume, so IEX-sourced frames get scaled up by this factor before the gate.
# Deliberately conservative (~1/20 IEX share assumption): large liquid names
# clear the $50M gate, genuine microcaps still fail. Override per deployment.
IEX_VOLUME_SCALE = float(os.environ.get("ALPACA_IEX_VOLUME_SCALE", "20"))
# Calls scanner fetch period. Bumped from 14mo to 18mo so we always have
# WINDOW (252) + DIV_LOOKBACK (60) + 2 = 314 trading days for the
# bullish-divergence prior-pivot search. 18mo ~ 378 td.
CALLS_FETCH_PERIOD = "18mo"
# Primary "sweet spot" band — the validated mean-reversion edge.
SWEET_LO = -2.5
SWEET_HI = -2.0
# Deep capitulation threshold. Names that cross BELOW this get fired as
# a secondary "deep" signal even if they're outside the primary band.
# The (DEEP_THRESHOLD, SWEET_LO) range — [-3.5, -2.5] — is the DEAD ZONE
# (research/out/sigma_band_sweep.txt: PF 1.04, CAGR -18% standalone),
# and is explicitly NOT eligible. The hybrid 2.5+3.5 sweep
# (research/out/calls_hybrid_band_sweep.txt) showed +5pts CAGR and
# -13pts maxDD vs the primary-only baseline with same trade count.
DEEP_THRESHOLD = -3.5
RV_LOOKBACK = 20
MIN_AVG_DOLLAR_VOL_M = 50.0
STALE_BARS = 5

# Bounce-confirm tier-promotion. The user's chart-eye pattern: a name
# already in the primary band [-2.5,-2.0] that has bounced off structural
# support with a turning Stoch RSI -- broad-market breadth irrelevant.
# Five features, all must be true to promote a WEAK candidate to BOUNCE:
#   F1 in primary band               (caller already checked z in band)
#   F2 z reclaim                     z_today >= 20b z-min + BOUNCE_RECLAIM_SIGMA
#   F3 higher low (252d)             min(low, last 20b) > min(low, prior 232b)
#   F4 Stoch RSI bottoming           %K <= 30 AND %K > 20b min(%K) + STOCH_TURN_MIN
#   F5 support confluence            >= 5 prior bars with low within +/- 2%
#                                    of the recent 20b low
# Calibrated by research/bounce_confirm_sweep.py on the volatile universe
# with R6 exits (5y): the combined "baseline UNION bounce reclaim>=0.20σ"
# config returned CAGR +15.5% vs baseline +13.0%, Sharpe 0.71 vs 0.61,
# MaxDD -23.8% vs -31.9%, MAR 0.65 vs 0.41 (PF 1.71 vs 2.32 -- the
# bounce trades pull at a lower ratio but lift risk-adjusted return).
BOUNCE_LOOKBACK = 20
BOUNCE_RECLAIM_SIGMA = 0.20
BOUNCE_SUPPORT_PCT = 0.02
BOUNCE_SUPPORT_HITS_MIN = 5
BOUNCE_STOCH_OS_MAX = 30.0
BOUNCE_STOCH_TURN_MIN = 1.0

# BOUNCE / DIV rank below PANIC (which still wins capital) but above WEAK
# so the bot will trade them. DIV slots right after BOUNCE -- both are
# chart-pattern signals without broad-market confirmation, comparable
# standalone PF (~1.7); BOUNCE keeps precedence on collision since the
# 5-feature pattern is the older, more validated of the two.
# Sort order: PRIME > OK > PANIC > BOUNCE > DIV > WEAK > ?
_TIER_RANK = {"PRIME": 0, "OK": 1, "PANIC": 2, "BOUNCE": 3, "DIV": 4, "WEAK": 5, "?": 6}

# ---- DIV LEG (DIV+R6 union, lb=60) ---------------------------------------
# Bullish RSI(14) regular divergence, gated to GENUINELY oversold price.
# Validated by research/divergence_sweep.py + research/divergence_window_sweep.py
# on the volatile universe (5y); the DIV+R6 union beat R6-only there. The
# DIV-only standalone (no R6 baseline) was weak (PF 1.71 / -2.2% CAGR), so
# DIV is shipped strictly as an UNION addition.
#
# DIV_BAND_HI was TIGHTENED -1.0 -> -2.0 (research/div_guardrail_sweep.py):
# the original [-3.5, -1.0] band let DIV fire on barely-oversold price --
# 205 of 337 DIV fires sat at z > -2.0, INSIDE the +/-2 sigma bands (the
# LCID 2026-05-21 call, z -1.37, was one). That shallow sub-band was a net
# drag. Raising the floor to -2.0 lifted the shipped calls config from
# CAGR +19.2% / PF 3.02 / MaxDD -22.5% to +30.9% / 3.37 / -14.7%. With
# band_hi -2.0 the DIV leg's unique contribution is divergence-gated entries
# in the (-3.5, -2.5) DEAD ZONE (primary/deep exclude it; collisions inside
# the primary band resolve in favour of primary). A slope/trend floor was
# also tested and made it WORSE -- do not add one.
# Shorter regression windows blow up; 252d is the sweet spot
# (see research_252d_window_sweet_spot memory).
DIV_BAND_LO = -3.5
DIV_BAND_HI = -2.0               # tightened from -1.0 (see div_guardrail_sweep)
DIV_LOOKBACK = 60                # bars back to search for the prior pivot
DIV_PIVOT_GAP = 3                # min bars between current bar and prior pivot
DIV_RSI_DIVERG_MIN = 3.0         # RSI[now] >= RSI[prior_pivot] + this
DIV_RSI_PERIOD = 14

# ---- PUTS SLEEVE ----------------------------------------------------------
# Mirror of the call bounce_confirm pattern for the overbought side, with
# one tuned difference: ob_floor=85 instead of the symmetric 70.
# Calibrated by research/bounce_confirm_puts_tune.py on the volatile
# universe (R6-mirror exits, 5y). The tuned-PF config returned
#   PF 2.53, CAGR +4.5%, Sharpe 0.42, MaxDD -17.1%, n=32 trades
# which is the FIRST positive puts edge in the project -- breaks the 0.91
# PF ceiling that the prior 9 puts sweeps couldn't crack (see
# [[put_leg_falsified]]). Counter-correlated to the calls sleeve: fires
# in overbought regimes when the calls bot has nothing to do.
PUTS_BAND_LO = 2.0                  # F1m: z in [+2.0, +2.5]
PUTS_BAND_HI = 2.5
PUTS_BOUNCE_RECLAIM_SIGMA = 0.10    # F2m pullback >= 0.10 sigma from 20b z-max
PUTS_BOUNCE_OB_FLOOR = 85.0         # F4m: %K >= this (tuned; sym would be 70)
PUTS_BOUNCE_STOCH_TURN_MIN = 1.0    # F4m: %K must roll over >= this from 20b max
PUTS_BOUNCE_SUPPORT_PCT = 0.02      # F5m: +/- 2% of recent 20b high
PUTS_BOUNCE_SUPPORT_HITS_MIN = 5


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
    # Read the source tag BEFORE slicing (DataFrame.attrs is not guaranteed to
    # survive .iloc). IEX-fallback frames carry partial (exchange-only) volume.
    scale = IEX_VOLUME_SCALE if df.attrs.get("partial_volume") else 1.0
    tail = df.iloc[-window:]
    dv = (tail["close"] * tail["volume"]).mean()
    return 0.0 if pd.isna(dv) else float(dv) * scale / 1_000_000.0


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


def _bounce_features(sd: np.ndarray, lows: np.ndarray,
                     k_arr: np.ndarray) -> dict:
    """Compute the 5 bounce-confirm features at the LAST bar of the
    arrays. Returns the feature booleans + the underlying numbers for
    transparency in the AI UI. The caller has already verified F1 (z is
    in the primary band) by selecting the candidate in scan_candidates."""
    out = {
        "f1_in_band": True,                # caller verified
        "f2_reclaim_sigma": False,
        "f3_higher_low_252d": False,
        "f4_stoch_turning": False,
        "f5_support_hits": 0,
        "f5_support_confluence": False,
        "recent_z_min_20b": None,
        "reclaim_from_min": None,
        "recent_low_20b": None,
        "prior_low_252b": None,
        "stoch_k_min_20b": None,
        "bounce_confirm": False,
    }
    n = len(sd)
    LB = BOUNCE_LOOKBACK
    if n < WINDOW + LB + 2:
        return out

    z_today = float(sd[-1])
    recent_z_window = sd[-LB:]
    if np.all(np.isnan(recent_z_window)):
        return out
    recent_z_min = float(np.nanmin(recent_z_window))
    out["recent_z_min_20b"] = round(recent_z_min, 2)
    reclaim = z_today - recent_z_min
    out["reclaim_from_min"] = round(reclaim, 2)
    out["f2_reclaim_sigma"] = reclaim >= BOUNCE_RECLAIM_SIGMA

    recent_low = float(np.nanmin(lows[-LB:]))
    prior_lows = lows[-WINDOW:-LB]
    if len(prior_lows) > 0:
        prior_low = float(np.nanmin(prior_lows))
        out["recent_low_20b"] = round(recent_low, 2)
        out["prior_low_252b"] = round(prior_low, 2)
        out["f3_higher_low_252d"] = bool(recent_low > prior_low)
        band_lo = recent_low * (1.0 - BOUNCE_SUPPORT_PCT)
        band_hi = recent_low * (1.0 + BOUNCE_SUPPORT_PCT)
        hits = int(((prior_lows >= band_lo) & (prior_lows <= band_hi)).sum())
        out["f5_support_hits"] = hits
        out["f5_support_confluence"] = hits >= BOUNCE_SUPPORT_HITS_MIN

    if k_arr is not None and len(k_arr) >= LB:
        k_today = k_arr[-1]
        k_window = k_arr[-LB:]
        if np.isfinite(k_today) and not np.all(np.isnan(k_window)):
            k_min = float(np.nanmin(k_window))
            out["stoch_k_min_20b"] = round(k_min, 1)
            out["f4_stoch_turning"] = bool(
                k_today <= BOUNCE_STOCH_OS_MAX
                and k_today > k_min + BOUNCE_STOCH_TURN_MIN
            )

    out["bounce_confirm"] = bool(
        out["f1_in_band"]
        and out["f2_reclaim_sigma"]
        and out["f3_higher_low_252d"]
        and out["f4_stoch_turning"]
        and out["f5_support_confluence"]
    )
    return out


def _bullish_divergence(sd: np.ndarray, lows: np.ndarray,
                        rsi_arr: np.ndarray) -> dict:
    """Compute bullish RSI(14) regular-divergence features at the LAST bar.

    Pattern:
      G1  z[-1] in [DIV_BAND_LO, DIV_BAND_HI] = [-3.5, -2.0]   (caller-checked)
      G2  low[-1] is the lowest low in the last DIV_LOOKBACK bars
      G3  prior pivot low = argmin(low) over the slice
            [-DIV_LOOKBACK : -DIV_PIVOT_GAP]
          AND low[-1] < low[prior_pivot]               (PRICE lower-low)
      G4  RSI[-1] >= RSI[prior_pivot] + DIV_RSI_DIVERG_MIN
                                                       (RSI HIGHER-low = div)

    Returns the feature dict (always populated for UI transparency)
    plus the boolean `div_fires`.
    """
    out = {
        "div_fires": False,
        "g2_new_low_60b": False,
        "g3_lower_low_vs_pivot": False,
        "g4_rsi_higher_low": False,
        "recent_low_60b": None,
        "prior_pivot_idx_offset": None,    # bars back from -1
        "prior_pivot_low": None,
        "prior_pivot_rsi": None,
        "current_rsi": None,
        "rsi_delta": None,                 # current - prior; positive = bullish div
    }
    n = len(sd)
    LB = DIV_LOOKBACK
    GAP = DIV_PIVOT_GAP
    if n < WINDOW + LB + 2:
        return out
    if rsi_arr is None or len(rsi_arr) < n:
        return out

    # G2: new 60-bar low (ties allowed)
    win_lows = lows[-LB:]
    if np.all(np.isnan(win_lows)):
        return out
    recent_lo = float(np.nanmin(win_lows))
    out["recent_low_60b"] = round(recent_lo, 2)
    if not np.isfinite(lows[-1]) or lows[-1] > recent_lo + 1e-12:
        return out
    out["g2_new_low_60b"] = True

    # G3: find the prior pivot low in [n-LB, n-GAP)
    prior_start = n - LB
    prior_end = n - GAP        # exclusive
    if prior_end - prior_start < 2:
        return out
    prior_slice = lows[prior_start:prior_end]
    if not np.any(np.isfinite(prior_slice)):
        return out
    prior_arg = int(np.nanargmin(prior_slice))
    prior_idx = prior_start + prior_arg
    prior_low = float(lows[prior_idx])
    out["prior_pivot_idx_offset"] = (n - 1) - prior_idx
    out["prior_pivot_low"] = round(prior_low, 2)
    if not np.isfinite(prior_low) or lows[-1] >= prior_low:
        return out
    out["g3_lower_low_vs_pivot"] = True

    # G4: RSI higher-low (bullish divergence proper)
    ri = float(rsi_arr[-1])
    rp = float(rsi_arr[prior_idx])
    if not (np.isfinite(ri) and np.isfinite(rp)):
        return out
    out["current_rsi"] = round(ri, 1)
    out["prior_pivot_rsi"] = round(rp, 1)
    out["rsi_delta"] = round(ri - rp, 1)
    if ri < rp + DIV_RSI_DIVERG_MIN:
        return out
    out["g4_rsi_higher_low"] = True

    out["div_fires"] = True
    return out


def scan_candidates(
    *,
    stoch_mode: str = "off",
    stoch_oversold_max: float = 30.0,
    deep_threshold: float | None = DEEP_THRESHOLD,
    divergence_enabled: bool = True,
) -> list[dict]:
    """Ranked oversold-call candidates (HYBRID-band: primary + deep + div).

    Three signal sources, all eligible (the third is opt-out via
    divergence_enabled=False):
      PRIMARY band  z in [SWEET_LO, SWEET_HI] = [-2.5, -2.0]  — the
                    validated mean-reversion edge (PF 2.32 standalone).
      DEEP cross    z <= deep_threshold (default -3.5) AND the cross
                    just occurred — captures rare violent capitulations
                    (PF 6.91 standalone, n=19/5y). Pass deep_threshold=
                    None to disable the deep leg.
      DIV          z in [DIV_BAND_LO, DIV_BAND_HI] = [-3.5, -2.0] AND
                    bullish RSI(14) regular divergence at lb=60 (price
                    lower-low vs. prior pivot, RSI higher-low by
                    >= DIV_RSI_DIVERG_MIN). DIV is a UNION addition to the
                    primary+deep baseline; collisions are resolved in
                    favor of primary/deep (higher tier), so DIV's unique
                    contribution is divergence-gated entries in the
                    (-3.5, -2.5) dead zone. The upper band was tightened
                    -1.0 -> -2.0 (research/div_guardrail_sweep.py) to drop
                    barely-oversold fires; see the DIV LEG note above.
                    Set divergence_enabled=False to disable.

    The DEAD ZONE (deep_threshold, SWEET_LO) = (-3.5, -2.5) is explicitly
    NOT eligible to PRIMARY/DEEP (PF 1.04 / CAGR -18% in standalone
    testing). DIV CAN fire in the dead zone -- but only if the divergence
    pattern is present, which is the whole point of the gate.

    Each candidate carries a `source` field ("primary" | "deep" | "div").
    They share the same capital frame in ai_trader; the engine's tier
    rank handles competition for slots so deep / primary signals on
    broad-capitulation days replace lower-tier DIV candidates.

    Breadth is the broad-market count of crossings into z <= SWEET_HI
    (-2.0) on the candidate's touch date — band-independent, so deep
    candidates need broad-market confirmation just like primary ones.

    Stoch RSI overlay (every candidate carries stoch_rsi_k / stoch_oversold
    regardless of mode, for AI-page transparency):
      - "off":     computed and attached, no effect on filter or rank.
      - "prefer":  oversold names sort first *within their tier* (does not
                   override the breadth tier, the strongest validated lever).
      - "require":  names with %K > stoch_oversold_max (or unknown %K) are
                   dropped entirely.
    """
    uni = _scan_universe()
    bars = fetch_bars_bulk(uni, period=CALLS_FETCH_PERIOD)
    # Expose fetch coverage so the entry funnel log can tell a DATA failure
    # (bars=0/N -> yfinance+Alpaca both empty) apart from a real no-signal day
    # (bars=N/N, scan=0). Both otherwise collapse to the same "scan=0" line.
    scan_candidates.last_universe = len(uni)
    scan_candidates.last_bars = len(bars)
    breadth: Counter = Counter()
    pending: list[dict] = []

    for tk, df in bars.items():
        # Per-ticker try/except: one malformed frame (e.g. yfinance returns
        # a slice without a 'timestamp' column for one ticker) must not kill
        # the entire scan and brick the morning's call-side entries.
        try:
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
            # Band check (in precedence order):
            #   PRIMARY  z in [-2.5, -2.0]
            #   DEEP     z <= deep_threshold (default -3.5)
            #   DIV      z in [-3.5, -2.0] AND bullish-divergence pattern fires
            # PRIMARY/DEEP take precedence, so DIV's eligible zone is the dead
            # zone (-3.5, -2.5): NOT tradeable to PRIMARY/DEEP, but tradeable to
            # DIV when the divergence pattern is present. The DIV upper band was
            # tightened -1.0 -> -2.0 (research/div_guardrail_sweep.py) to drop
            # barely-oversold fires inside the +/-2 sigma bands.
            in_primary = SWEET_LO <= z <= SWEET_HI
            in_deep = deep_threshold is not None and z <= deep_threshold
            div_eligible_zone = (
                divergence_enabled
                and not in_primary and not in_deep
                and DIV_BAND_LO <= z <= DIV_BAND_HI
            )
            if not (in_primary or in_deep or div_eligible_zone):
                continue

            adv_m = _avg_dollar_vol_m(df)
            if adv_m < MIN_AVG_DOLLAR_VOL_M:
                continue

            # Divergence features computed only for DIV-eligible candidates
            # (rsi() is a few ms per ticker; gated to keep the scan fast).
            div = None
            lows = df["low"].to_numpy(dtype=float)
            if div_eligible_zone:
                rsi_arr = rsi(df["close"], period=DIV_RSI_PERIOD).to_numpy(dtype=float)
                div = _bullish_divergence(sd, lows, rsi_arr)
                if not div["div_fires"]:
                    continue

            if in_primary or in_deep:
                source = "deep" if in_deep else "primary"
                # band_hi controls the freshness/touch computation so a name
                # that drops from -2.5 to -3.7 has bars_in_zone=1 in the deep
                # band even though it crossed -2 weeks ago.
                band_hi = deep_threshold if in_deep else SWEET_HI
            else:
                source = "div"
                # For DIV, freshness is measured against the DIV band's upper
                # bound (DIV_BAND_HI = -2.0) -- how many bars since z entered
                # genuinely-oversold territory.
                band_hi = DIV_BAND_HI

            k_ser, _ = stoch_rsi(df["close"])
            k_last = float(k_ser.iloc[-1]) if len(k_ser) and not pd.isna(k_ser.iloc[-1]) else None
            is_os = k_last is not None and k_last <= stoch_oversold_max
            if stoch_mode == "require" and not is_os:
                continue
            zone = _bars_in_zone(sd, hi=band_hi)
            n = len(sd)
            touch_idx = n - zone
            touch_date = ts_dates[touch_idx] if 0 <= touch_idx < n else ts_dates[-1]
            first_touch = bool(sd[-2] > band_hi and sd[-1] <= band_hi)
            price = float(closes[-1])
            reversion_pct = (center_px / price - 1.0) * 100.0
            # Bounce-confirm features (always computed, attached for UI). The
            # tier-promotion (WEAK -> BOUNCE) happens after breadth is known.
            # DIV candidates also get bounce features for transparency, but
            # they're tier-stamped DIV regardless (no promotion needed).
            k_arr = k_ser.to_numpy(dtype=float) if len(k_ser) else np.array([])
            bounce = _bounce_features(sd, lows, k_arr)
            pending.append({
                "ticker": tk,
                "price": round(price, 2),
                "z_log": round(z, 2),
                "source": source,             # "primary" | "deep" | "div"
                "touch_date": str(touch_date),
                "_touch_date": touch_date,
                "first_touch": first_touch,
                "bars_in_zone": zone,
                "status": "FRESH" if zone <= 1 else ("recent" if zone <= STALE_BARS else "STALE"),
                "reversion_to_mean_pct": round(reversion_pct, 1),
                "realized_vol_pct": round(_realized_vol_pct(closes), 1),
                "log_slope_ann_pct": round(slope_ann, 1),
                "avg_dollar_vol_m": round(adv_m, 1),
                "stoch_rsi_k": round(k_last, 1) if k_last is not None else None,
                "stoch_oversold": is_os,
                "bounce": bounce,
                "divergence": div,            # None for primary/deep; dict for div
            })
        except Exception:
            logger.exception("scan_candidates: skipping %s", tk)
            continue

    for c in pending:
        b = int(breadth.get(c["_touch_date"], 0))
        c["same_day_breadth"] = b
        if c["source"] == "div":
            # DIV candidates are tier-stamped DIV regardless of breadth.
            # The pattern (bullish RSI divergence at lb=60) is the gate;
            # broad-market breadth is informational only for these.
            c["tier"] = "DIV"
        else:
            base_tier = tier_for(b)
            # Tier-promote WEAK / ? PRIMARY/DEEP candidates to BOUNCE when
            # the 5-feature chart-eye bounce pattern fires. Validated by
            # research/bounce_confirm_sweep.py (combined-0.20 beats
            # baseline on CAGR / Sharpe / MaxDD / MAR; PF drops).
            # PRIME/OK/PANIC are already eligible and keep their
            # (higher-ranking) tier so the broad-market breadth edge still
            # wins same-day capital competition.
            if base_tier in ("WEAK", "?") and c["bounce"]["bounce_confirm"]:
                c["tier"] = "BOUNCE"
                c["tier_promoted_from"] = base_tier
            else:
                c["tier"] = base_tier
        del c["_touch_date"]

    # "prefer" inserts an oversold-first key directly under the tier rank so
    # it reorders within a tier without overriding it. For "off"/"require"
    # the term is constant across kept rows -> no effect on the ordering.
    pending.sort(key=lambda c: (
        _TIER_RANK.get(c["tier"], 9),
        0 if (stoch_mode == "prefer" and c["stoch_oversold"]) else 1,
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


def current_z_for_tickers(tickers: list[str]) -> dict[str, float]:
    """Latest 252d LOG-channel z for each ticker. Returns only tickers with
    enough history. Used by ai_trader's sigma-target exit so an open trade
    can be closed when z reverts to a configured threshold (e.g. 0)."""
    if not tickers:
        return {}
    bars = fetch_bars_bulk(tickers, period="14mo")
    out: dict[str, float] = {}
    for tk, df in bars.items():
        try:
            if df is None or df.empty or len(df) < WINDOW + 2:
                continue
            closes = df["close"].to_numpy(dtype=float)
            sd, _, _ = _log_channel_sd(closes)
            if len(sd) and not np.isnan(sd[-1]):
                out[tk] = float(sd[-1])
        except Exception:
            logger.exception("current_z_for_tickers: skipping %s", tk)
            continue
    return out


# ---- PUTS SLEEVE ----------------------------------------------------------

def _bounce_features_put(sd: np.ndarray, highs: np.ndarray,
                         k_arr: np.ndarray) -> dict:
    """Compute the 5 mirror features for the puts sleeve at the LAST bar.
    F1m (z in band) is verified by the caller via selection in
    scan_put_candidates. Returns booleans + the underlying numbers."""
    out = {
        "f1_in_band": True,                # caller verified
        "f2_pullback_sigma": False,
        "f3_lower_high_252d": False,
        "f4_stoch_rolling_over": False,
        "f5_resistance_hits": 0,
        "f5_resistance_confluence": False,
        "recent_z_max_20b": None,
        "pullback_from_max": None,
        "recent_high_20b": None,
        "prior_high_252b": None,
        "stoch_k_max_20b": None,
        "bounce_confirm_put": False,
    }
    n = len(sd)
    LB = BOUNCE_LOOKBACK
    if n < WINDOW + LB + 2:
        return out

    z_today = float(sd[-1])
    recent_z_window = sd[-LB:]
    if np.all(np.isnan(recent_z_window)):
        return out
    recent_z_max = float(np.nanmax(recent_z_window))
    out["recent_z_max_20b"] = round(recent_z_max, 2)
    pullback = recent_z_max - z_today
    out["pullback_from_max"] = round(pullback, 2)
    out["f2_pullback_sigma"] = pullback >= PUTS_BOUNCE_RECLAIM_SIGMA

    recent_hi = float(np.nanmax(highs[-LB:]))
    prior_highs = highs[-WINDOW:-LB]
    if len(prior_highs) > 0:
        prior_hi = float(np.nanmax(prior_highs))
        out["recent_high_20b"] = round(recent_hi, 2)
        out["prior_high_252b"] = round(prior_hi, 2)
        out["f3_lower_high_252d"] = bool(recent_hi < prior_hi)
        band_lo = recent_hi * (1.0 - PUTS_BOUNCE_SUPPORT_PCT)
        band_hi = recent_hi * (1.0 + PUTS_BOUNCE_SUPPORT_PCT)
        hits = int(((prior_highs >= band_lo) & (prior_highs <= band_hi)).sum())
        out["f5_resistance_hits"] = hits
        out["f5_resistance_confluence"] = hits >= PUTS_BOUNCE_SUPPORT_HITS_MIN

    if k_arr is not None and len(k_arr) >= LB:
        k_today = k_arr[-1]
        k_window = k_arr[-LB:]
        if np.isfinite(k_today) and not np.all(np.isnan(k_window)):
            k_max = float(np.nanmax(k_window))
            out["stoch_k_max_20b"] = round(k_max, 1)
            out["f4_stoch_rolling_over"] = bool(
                k_today >= PUTS_BOUNCE_OB_FLOOR
                and k_today < k_max - PUTS_BOUNCE_STOCH_TURN_MIN
            )

    out["bounce_confirm_put"] = bool(
        out["f1_in_band"]
        and out["f2_pullback_sigma"]
        and out["f3_lower_high_252d"]
        and out["f4_stoch_rolling_over"]
        and out["f5_resistance_confluence"]
    )
    return out


def scan_put_candidates() -> list[dict]:
    """Ranked PUTS-side bounce_confirm-mirror candidates.

    Only fires the tuned 5-feature pattern (no broad-market breadth gate --
    the prior 9 puts sweeps falsified that path). A candidate must have:
      F1m  z in [+2.0, +2.5]
      F2m  z pullback >= 0.10 sigma from the 20b z-max
      F3m  LOWER high (252d frame)
      F4m  Stoch RSI %K >= 85 AND rolling down >= 1.0 from 20b max
      F5m  >= 5 prior bars with high within +/- 2% of the recent 20b high
    AND ADV >= MIN_AVG_DOLLAR_VOL_M.

    All passing names share the single tier 'BOUNCE_PUT' and are ranked
    PRIMARILY by support_hits descending (stronger structural resistance =
    higher conviction), then by Stoch RSI %K descending (more overbought =
    further to revert).
    """
    uni = _scan_universe()
    bars = fetch_bars_bulk(uni, period="14mo")
    scan_put_candidates.last_universe = len(uni)
    scan_put_candidates.last_bars = len(bars)
    out: list[dict] = []
    for tk, df in bars.items():
        try:
            if df is None or df.empty or len(df) < WINDOW + 2:
                continue
            closes = df["close"].to_numpy(dtype=float)
            highs = df["high"].to_numpy(dtype=float)
            sd, slope_ann, center_px = _log_channel_sd(closes)
            if np.isnan(sd[-1]):
                continue
            z = float(sd[-1])
            if not (PUTS_BAND_LO <= z <= PUTS_BAND_HI):
                continue
            adv_m = _avg_dollar_vol_m(df)
            if adv_m < MIN_AVG_DOLLAR_VOL_M:
                continue
            k_ser, _ = stoch_rsi(df["close"])
            k_arr = k_ser.to_numpy(dtype=float) if len(k_ser) else np.array([])
            k_last = float(k_ser.iloc[-1]) if len(k_ser) and not pd.isna(k_ser.iloc[-1]) else None
            feat = _bounce_features_put(sd, highs, k_arr)
            if not feat["bounce_confirm_put"]:
                continue
            price = float(closes[-1])
            # Mean-reversion target: distance from price back to the channel
            # center (negative for puts because price is above the line).
            reversion_pct = (center_px / price - 1.0) * 100.0
            out.append({
                "ticker": tk,
                "price": round(price, 2),
                "z_log": round(z, 2),
                "source": "bounce_put",
                "tier": "BOUNCE_PUT",
                "reversion_to_mean_pct": round(reversion_pct, 1),
                "realized_vol_pct": round(_realized_vol_pct(closes), 1),
                "log_slope_ann_pct": round(slope_ann, 1),
                "avg_dollar_vol_m": round(adv_m, 1),
                "stoch_rsi_k": round(k_last, 1) if k_last is not None else None,
                "bounce": feat,
            })
        except Exception:
            logger.exception("scan_put_candidates: skipping %s", tk)
            continue

    out.sort(key=lambda c: (
        -c["bounce"]["f5_resistance_hits"],          # stronger resistance first
        -(c["stoch_rsi_k"] if c["stoch_rsi_k"] else 0),  # more overbought first
    ))
    return out


def pick_put_contract(
    underlying: str,
    price: float,
    *,
    otm_pct: float,
    dte_min: int,
    dte_max: int,
    today: date | None = None,
) -> dict | None:
    """Choose ~otm_pct OTM put expiring within [dte_min, dte_max] DTE.

    Mirror of pick_contract: strike = price * (1 - otm_pct/100); fetch
    option_type='put'; widen the strike window; pick the earliest exp in
    band and the strike closest to the OTM target.
    """
    from backend import alpaca_trading as at

    today = today or date.today()
    exp_lo = (today + timedelta(days=dte_min)).isoformat()
    exp_hi = (today + timedelta(days=dte_max)).isoformat()
    target_strike = price * (1.0 - otm_pct / 100.0)
    # Widen so we still find SOMETHING; we re-pick closest to the target.
    strike_lo = target_strike * 0.85
    strike_hi = target_strike * 1.05

    contracts = at.list_option_contracts(
        underlying,
        expiration_gte=exp_lo,
        expiration_lte=exp_hi,
        strike_gte=strike_lo,
        strike_lte=strike_hi,
        option_type="put",
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


__all__ = [
    "scan_candidates", "pick_contract",
    "scan_put_candidates", "pick_put_contract",
    "tier_for", "current_z_for_tickers",
]
