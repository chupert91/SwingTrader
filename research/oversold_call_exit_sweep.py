"""Capital-constrained, theta-aware exit sweep on the LIVE oversold-call bot.

The entry, instrument, and capital frame are FROZEN to mirror the production
bot (backend/ai_strategy.py + backend/ai_store.DEFAULT_SETTINGS); only the
EXITS vary. Each axis is swept while the other exits stay at the live
baseline, so we read which knob actually moves CAGR/Sharpe/MAR.

Frozen:
  ENTRY        252d LOG-channel z, first touch into [-2.5, -2.0]; ADV>=$50M;
               same-day-breadth tier in {PRIME, OK, PANIC} (skip WEAK / ?);
               within tier, higher breadth ranks first.
  INSTRUMENT   5% OTM call, 45 DTE (earliest in the live 45-60 band; least
               theta over-pay). IV = RV20 * 1.25 clamped [0.15, 1.20]; IV
               crushes to RV floor over 10 trading days. r = 4%.
  CAPITAL      $10k start; 1 contract/trade; premium cap $2000; max 2
               concurrent / 2 per day; 3% half-spread + 1% entry buffer.

Exit baseline (live production):
  TP         +30% option mark (intraday touch on the bar high — resting limit)
  DISASTER   -50% option mark on CLOSE (matches ai_trader._reconcile's
             mark-based software stop; NOT intraday-low, which overstates
             disaster hits and which I flagged in the SMA option sim)
  TIME       10 trading days held
  EXPIRY     settle intrinsic on/after the expiration date
  + no sigma-target, no trailing-peak

Sweeps (each with all OTHER exits at the live baseline):
  TP             {15, 20, 25, 30, 40, 50, none}
  DISASTER       {-30, -50, -70, none}
  TIME           {5, 10, 15, 20, 30, 40, none}
  SIGMA-TARGET   {none, -1.0, 0.0, +1.0}  -- close when z reverts to target
  TRAIL-PEAK     {none, (20%,30%), (20%,50%), (30%,30%), (30%,50%)}
                 -- once peak option P&L >= activation, exit on giveback%
  + TUNED        best-per-axis stacked

Run:
    python research/oversold_call_exit_sweep.py
Outputs:
    research/out/oversold_call_exit_sweep.png
    research/out/oversold_call_exit_sweep.txt
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.sp500_tickers import SP500_TICKERS  # noqa: E402
from research.theta_overlay import bs_call, IV_ELEVATION, CRUSH_TD, R, RV_LOOKBACK  # noqa: E402
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, _stats, _fmt_row, HEADER, STARTING_CAPITAL,
    MIN_ADV_M, OPTION_MULT,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Entry signal
LOG_WINDOW = 252
SWEET_LO, SWEET_HI = -2.5, -2.0     # first touch into this band
TIER_THRESH = {"PANIC": 20, "PRIME": 8, "OK": 3, "WEAK": 1}  # mirrors ai_strategy.tier_for
TIER_RANK = {"PRIME": 0, "OK": 1, "PANIC": 2, "WEAK": 3, "?": 4}
ACCEPT_TIERS = {"PRIME", "OK", "PANIC"}      # bot skips WEAK / ?

# Instrument
OTM_PCT = 5.0
DTE = 45

# Capital (live bot)
MAX_CONCURRENT = 2
MAX_PER_DAY = 2
PREMIUM_CAP = 2000.0
# Module-level constants are no longer used by simulate() — kept only for
# the "pre-fix" cost-layer comparison row in main(). Per-config costs live
# in ExitCfg.half_spread and ExitCfg.entry_buffer below.
PREFIX_HALF_SPREAD = 0.03    # the original (pre-fix) modeled half-spread
PREFIX_ENTRY_BUFFER = 0.01   # the original (pre-fix) entry buffer

# Exit baseline (live bot)
BASE_TP = 30.0
BASE_DISASTER = 50.0
BASE_TIME = 10

# Default cost layer — realistic for liquid SP500 OTM options with patient
# mid-target limit fills (the post-fix bot, also closer to manual reality).
# 1.5% half-spread, no entry buffer (mid fill). Compare against the pre-fix
# bot (3% half-spread + 1% buffer) and the user's manual reality (~0.5%
# round-trip, mid fills routinely) in the cost-layer block.
DEFAULT_HALF_SPREAD = 0.015
DEFAULT_ENTRY_BUFFER = 0.0


@dataclass(frozen=True)
class ExitCfg:
    tp_pct: float | None = BASE_TP             # None = no TP
    disaster_pct: float | None = BASE_DISASTER  # mark-on-close, % of premium
    disaster_usd: float | None = None           # absolute-$ loss cap (user's actual rule)
    time_stop: int | None = BASE_TIME           # trading days
    sigma_target: float | None = None           # exit when z >= this
    trail_activate: float | None = None         # peak option P&L threshold (%)
    trail_giveback: float | None = None         # giveback fraction (e.g. 0.5)
    half_spread: float = DEFAULT_HALF_SPREAD
    entry_buffer: float = DEFAULT_ENTRY_BUFFER
    # IV model overrides (defaults match research/theta_overlay.py)
    iv_elevation: float = IV_ELEVATION          # entry IV = RV * this, clamped
    crush_td: int = CRUSH_TD                    # trading days for IV to decay to floor
    # Instrument overrides (defaults match live bot). Per personal_trade_audit:
    # real-trade median DTE was 94d (not 45), and strikes were ATM-ish (not 5% OTM).
    otm_pct: float = OTM_PCT
    dte: int = DTE
    # Underlying-pct directional target. For calls: close when underlying has
    # RISEN by this % from entry close (positive). For puts: close when
    # underlying has FALLEN by this % from entry close (positive value, sign
    # interpreted by the simulator). None = disabled.
    underlying_target_pct: float | None = None


def _tier_for(b: int) -> str:
    if b >= TIER_THRESH["PANIC"]: return "PANIC"
    if b >= TIER_THRESH["PRIME"]: return "PRIME"
    if b >= TIER_THRESH["OK"]:    return "OK"
    if b >= TIER_THRESH["WEAK"]:  return "WEAK"
    return "?"


def _log_channel_z(close: np.ndarray) -> np.ndarray:
    """Rolling 252-bar LOG-channel z per bar (mirrors ai_strategy)."""
    n = len(close)
    z = np.full(n, np.nan)
    if n < LOG_WINDOW:
        return z
    y_all = np.log(close)
    x = np.arange(LOG_WINDOW, dtype=float)
    x_mean = x.mean()
    xc = x - x_mean
    denom = float(np.sum(xc ** 2))
    for t in range(LOG_WINDOW - 1, n):
        chunk = y_all[t - LOG_WINDOW + 1: t + 1]
        y_mean = chunk.mean()
        slope = float(np.sum(xc * (chunk - y_mean)) / denom)
        intercept = y_mean - slope * x_mean
        fit_end = slope * (LOG_WINDOW - 1) + intercept
        sigma = float(np.std(chunk - (slope * x + intercept), ddof=1))
        if sigma > 0:
            z[t] = (y_all[t] - fit_end) / sigma
    return z


@dataclass
class Sig:
    tk: str
    i: int
    z: float
    date: object
    tier: str = "?"
    breadth: int = 0
    adv: float = 0.0


def _signals(series: dict[str, Series], z_by_tk: dict[str, np.ndarray]) -> dict:
    """date -> list[Sig]. Matches ai_strategy.scan_candidates exactly:
    breadth = count of ALL crossings into z<=SWEET_HI on that date (even
    names that fell past SWEET_LO contribute), then a signal is eligible
    iff z lands in [SWEET_LO, SWEET_HI] AND ADV>=MIN_ADV_M. Tier from
    breadth; drop WEAK/?."""
    breadth_by_day: Counter = Counter()
    eligible: list[Sig] = []
    for tk, s in series.items():
        z = z_by_tk[tk]
        for i in range(1, len(s.close)):
            if np.isnan(z[i]) or np.isnan(z[i - 1]):
                continue
            crossed = z[i] <= SWEET_HI and z[i - 1] > SWEET_HI
            if crossed:
                breadth_by_day[s.dates[i]] += 1
            if not crossed:
                continue
            if not (SWEET_LO <= z[i] <= SWEET_HI):
                continue
            if np.isnan(s.rv[i]) or not (s.adv_m[i] >= MIN_ADV_M):
                continue
            eligible.append(Sig(tk=tk, i=i, z=float(z[i]), date=s.dates[i],
                                adv=float(s.adv_m[i])))
    out: dict = {}
    for sg in eligible:
        breadth = int(breadth_by_day.get(sg.date, 0))
        tier = _tier_for(breadth)
        if tier not in ACCEPT_TIERS:
            continue
        sg.tier = tier
        sg.breadth = breadth
        out.setdefault(sg.date, []).append(sg)
    return out


def _iv_at(entry_iv: float, iv_floor: float, held: int, crush_td: int) -> float:
    crush = min(held / crush_td, 1.0) if crush_td > 0 else 1.0
    return entry_iv - (entry_iv - iv_floor) * crush


def _price(S: float, K: float, exp_date, bar_date, iv: float) -> float:
    T = max((exp_date - bar_date).days, 0) / 365.0
    if T <= 0.0:
        return max(S - K, 0.0)
    return bs_call(S, K, T, R, iv)


@dataclass
class Pos:
    tk: str
    entry_idx: int
    entry_date: object
    exp_date: object
    strike: float
    entry_cost: float       # per-contract premium paid (incl. spread+buffer)
    entry_iv: float
    iv_floor: float
    qty: int
    peak_pnl: float = 0.0   # peak option P&L fraction, close-mark
    trail_activated: bool = False


def simulate(series: dict[str, Series], z_by_tk: dict[str, np.ndarray],
             sig_by_day: dict, cfg: ExitCfg) -> dict:
    all_dates = sorted({d for s in series.values() for d in s.dates})
    cash = STARTING_CAPITAL
    open_pos: dict[str, Pos] = {}
    trades: list[dict] = []
    equity_curve: list[tuple] = []

    def mtm_close(pos: Pos, s: Series, idx: int) -> float:
        iv = _iv_at(pos.entry_iv, pos.iv_floor, idx - pos.entry_idx, cfg.crush_td)
        mid = _price(s.close[idx], pos.strike, pos.exp_date, s.dates[idx], iv)
        return mid * (1 - cfg.half_spread) * OPTION_MULT * pos.qty

    for d in all_dates:
        # ---- exits ----
        for tk in sorted(list(open_pos.keys())):
            pos = open_pos[tk]
            s = series[tk]
            idx = s.by_date.get(d)
            if idx is None or idx <= pos.entry_idx:
                continue
            held = idx - pos.entry_idx
            iv = _iv_at(pos.entry_iv, pos.iv_floor, held, cfg.crush_td)
            total_cost_usd = pos.entry_cost * OPTION_MULT * pos.qty

            def realized_ret(S: float) -> float:
                proc = (_price(S, pos.strike, pos.exp_date, s.dates[idx], iv)
                        * (1 - cfg.half_spread) * OPTION_MULT * pos.qty)
                return proc / (pos.entry_cost * OPTION_MULT * pos.qty) - 1.0

            exit_ret, reason = None, None
            expired = s.dates[idx] >= pos.exp_date

            if expired:
                exit_ret, reason = realized_ret(s.close[idx]), "expiry"
            else:
                # Intraday TP touch (resting limit on the bar high).
                if cfg.tp_pct is not None and realized_ret(s.high[idx]) >= cfg.tp_pct / 100.0:
                    exit_ret, reason = cfg.tp_pct / 100.0, "take_profit"
                else:
                    cr = realized_ret(s.close[idx])
                    # Update peak (close-mark) BEFORE evaluating trail.
                    if cr > pos.peak_pnl:
                        pos.peak_pnl = cr
                    if (cfg.trail_activate is not None
                            and pos.peak_pnl >= cfg.trail_activate / 100.0):
                        pos.trail_activated = True

                    # Close-based exits, priority: disaster -> sigma-target ->
                    # trailing-peak -> time stop. Disaster fires on whichever
                    # of disaster_pct (% of premium) or disaster_usd ($ loss)
                    # is triggered first; either or both can be set.
                    hit_pct = (cfg.disaster_pct is not None
                               and cr <= -cfg.disaster_pct / 100.0)
                    hit_usd = (cfg.disaster_usd is not None
                               and -cr * total_cost_usd >= cfg.disaster_usd)
                    if hit_pct or hit_usd:
                        exit_ret, reason = cr, "disaster"
                    elif (cfg.sigma_target is not None
                          and not np.isnan(z_by_tk[tk][idx])
                          and z_by_tk[tk][idx] >= cfg.sigma_target):
                        exit_ret, reason = cr, "sigma_target"
                    elif (pos.trail_activated and cfg.trail_giveback is not None
                          and cr <= pos.peak_pnl * (1.0 - cfg.trail_giveback)):
                        exit_ret, reason = cr, "trail"
                    elif cfg.time_stop is not None and held >= cfg.time_stop:
                        exit_ret, reason = cr, "time_stop"

            if exit_ret is not None:
                exit_ret = max(exit_ret, -1.0)
                proceeds = pos.entry_cost * OPTION_MULT * pos.qty * (1.0 + exit_ret)
                cash += proceeds
                trades.append({
                    "ticker": tk,
                    "entry_date": str(pos.entry_date),
                    "exit_date": str(s.dates[idx]),
                    "bars_held": held,
                    "opt_ret": exit_ret,
                    "exit_reason": reason,
                    "cost": pos.entry_cost * OPTION_MULT * pos.qty,
                })
                del open_pos[tk]

        # ---- entries ----
        sigs = sig_by_day.get(d, [])
        if sigs and len(open_pos) < MAX_CONCURRENT:
            placed_today = 0
            # PRIME -> OK -> PANIC, higher breadth first, ticker tie-break.
            for sg in sorted(sigs, key=lambda x: (TIER_RANK.get(x.tier, 9),
                                                   -x.breadth, x.tk)):
                if len(open_pos) >= MAX_CONCURRENT or placed_today >= MAX_PER_DAY:
                    break
                tk, i = sg.tk, sg.i
                if tk in open_pos:
                    continue
                s = series[tk]
                S0 = s.close[i]
                K = S0 * (1.0 + cfg.otm_pct / 100.0)
                rv = s.rv[i]
                if np.isnan(rv):
                    continue
                entry_iv = min(max(rv * cfg.iv_elevation, 0.15), 1.20)
                iv_floor = min(max(rv, 0.10), 1.10)
                exp_date = s.dates[i] + timedelta(days=cfg.dte)
                mid0 = _price(S0, K, exp_date, s.dates[i], entry_iv)
                if mid0 <= 1e-6:
                    continue
                entry_cost = mid0 * (1.0 + cfg.half_spread) * (1.0 + cfg.entry_buffer)
                cost_usd = entry_cost * OPTION_MULT
                if cost_usd > PREMIUM_CAP or cost_usd > cash:
                    continue
                cash -= cost_usd
                open_pos[tk] = Pos(tk, i, s.dates[i], exp_date, K, entry_cost,
                                   entry_iv, iv_floor, 1)
                placed_today += 1

        # ---- equity snapshot ----
        held_val = 0.0
        for tk, pos in open_pos.items():
            s = series[tk]
            idx = s.by_date.get(d)
            if idx is not None and idx >= pos.entry_idx:
                held_val += mtm_close(pos, s, idx)
        equity_curve.append((d, cash + held_val))

    return _stats(trades, equity_curve)


def _fmt_with_n(tag: str, st: dict) -> str:
    row = _fmt_row(tag, st)
    if st.get("n"):
        reasons = st.get("reasons") or {}
        top = ",".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        return f"{row}  | {top}"
    return row


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Fetching {len(SP500_TICKERS)} tickers x 5y...")
    bars = fetch_bars_bulk(SP500_TICKERS, period="5y")
    print(f"  got {len(bars)} usable tickers")

    series: dict[str, Series] = {}
    z_by_tk: dict[str, np.ndarray] = {}
    for tk, df in bars.items():
        s = build_series(tk, df)
        if s is not None:
            series[tk] = s
            z_by_tk[tk] = _log_channel_z(s.close)
    print(f"  {len(series)} tickers with full history")
    sig_by_day = _signals(series, z_by_tk)
    n_sig = sum(len(v) for v in sig_by_day.values())
    print(f"  {n_sig} entry signals (tier PRIME/OK/PANIC, ADV-gated, first-touch)\n")

    baseline = ExitCfg()
    base_st = simulate(series, z_by_tk, sig_by_day, baseline)

    lines: list[str] = []
    lines.append("Capital-constrained, theta-aware exit sweep on the LIVE oversold-call bot.")
    lines.append("Entry, instrument, capital all frozen to production; only exits vary.")
    lines.append(f"Baseline exits: TP +{BASE_TP:.0f}% | disaster -{BASE_DISASTER:.0f}% mark-on-close "
                 f"| time {BASE_TIME}td | no sigma-target | no trailing.")
    lines.append(f"Default cost layer: {DEFAULT_HALF_SPREAD*100:.1f}% half-spread, "
                 f"{DEFAULT_ENTRY_BUFFER*100:.1f}% entry buffer (mid-fill, realistic for $50M+ ADV).")
    lines.append(f"Signals: {n_sig}  Tickers: {len(series)}  "
                 f"Capital ${STARTING_CAPITAL:,.0f}  Max {MAX_CONCURRENT} concurrent / "
                 f"{MAX_PER_DAY}/day.\n")

    EHEADER = HEADER + "  | exit-reason mix"

    # Cost-layer impact (baseline exits, vary only the spread/buffer).
    lines.append("[COST-LAYER COMPARISON]  baseline exits, vary only spread/buffer")
    lines.append(EHEADER)
    cost_layers = [
        ("pre-fix (3.0%, 1.0%)", PREFIX_HALF_SPREAD, PREFIX_ENTRY_BUFFER),
        ("realistic (1.5%, 0%)", 0.015, 0.0),
        ("manual-ish (0.5%, 0%)", 0.005, 0.0),
    ]
    for tag, hs, eb in cost_layers:
        cfg = replace(baseline, half_spread=hs, entry_buffer=eb)
        st = simulate(series, z_by_tk, sig_by_day, cfg)
        lines.append(_fmt_with_n(tag, st))
    lines.append("")

    lines.append("[BASELINE]  (default cost layer used for all sweeps below)")
    lines.append(EHEADER)
    lines.append(_fmt_with_n("live default", base_st))
    lines.append("")

    def block(title: str, rows: list[tuple]) -> dict[str, dict]:
        lines.append(f"[{title}]  (other exits = baseline)")
        lines.append(EHEADER)
        out: dict[str, dict] = {}
        for tag, cfg in rows:
            st = simulate(series, z_by_tk, sig_by_day, cfg)
            out[tag] = st
            lines.append(_fmt_with_n(tag, st))
        lines.append("")
        return out

    tp_rows = [(f"TP={'none' if t is None else f'{t:g}%'}", replace(baseline, tp_pct=t))
               for t in (15, 20, 25, 30, 40, 50, None)]
    tp_res = block("TP SWEEP", tp_rows)

    dis_rows = [(f"dis={'none' if d is None else f'-{d:g}%'}",
                 replace(baseline, disaster_pct=d))
                for d in (30, 50, 70, None)]
    dis_res = block("DISASTER SWEEP", dis_rows)

    time_rows = [(f"time={'none' if t is None else f'{t}td'}",
                  replace(baseline, time_stop=t))
                 for t in (5, 10, 15, 20, 30, 40, None)]
    time_res = block("TIME SWEEP", time_rows)

    sigma_rows = [(f"sigma>=-1.0", replace(baseline, sigma_target=-1.0)),
                  (f"sigma>=0.0",  replace(baseline, sigma_target=0.0)),
                  (f"sigma>=+1.0", replace(baseline, sigma_target=1.0))]
    sigma_res = block("SIGMA-TARGET SWEEP (NEW EXIT)", sigma_rows)

    trail_rows = [
        ("trail 20/30", replace(baseline, trail_activate=20.0, trail_giveback=0.30)),
        ("trail 20/50", replace(baseline, trail_activate=20.0, trail_giveback=0.50)),
        ("trail 30/30", replace(baseline, trail_activate=30.0, trail_giveback=0.30)),
        ("trail 30/50", replace(baseline, trail_activate=30.0, trail_giveback=0.50)),
    ]
    trail_res = block("TRAIL-PEAK SWEEP (NEW EXIT)", trail_rows)

    # ---- patient multi-axis configs ----------------------------------------
    # The single-axis sweep can't surface multi-knob interactions. Per
    # research/out/otm_reversion.txt, capitulation 5% OTM 45 DTE peaks at
    # +47.4% mean option PnL at day 33; the live 10td time stop + -50%
    # disaster cuts trades long before that. These configs RELAX both at
    # once so trades survive to reach the regime's natural peak.
    patient_rows = [
        ("P0 hold-to-peak",
         ExitCfg(tp_pct=None, disaster_pct=None, time_stop=45)),
        ("P1 TP50,no-dis,45td",
         ExitCfg(tp_pct=50.0, disaster_pct=None, time_stop=45)),
        ("P2 sigma-revert",
         ExitCfg(tp_pct=None, disaster_pct=None, time_stop=45, sigma_target=0.0)),
        ("P3 live-TP,no-dis,45td",
         ExitCfg(tp_pct=30.0, disaster_pct=None, time_stop=45)),
        ("P4 live-TP,dis-90,30td",
         ExitCfg(tp_pct=30.0, disaster_pct=90.0, time_stop=30)),
        ("P5 wide-cap aligned",
         ExitCfg(tp_pct=40.0, disaster_pct=90.0, time_stop=45, sigma_target=0.0)),
    ]
    patient_res = block("PATIENT MULTI-AXIS CONFIGS", patient_rows)

    # ---- $-cap disaster sweep (user's actual rule shape) -------------------
    # The user's manual rule is "cap each losing trade at ~$200" — an
    # ABSOLUTE dollar cap, not a % of premium. On a $1.6k premium that's
    # ~12% (much tighter than the bot's -50%); on a $300 premium it's >60%
    # (effectively no stop). Sweeps the $-cap, replacing the %-disaster.
    usd_rows = [
        ("dis_usd=$100", replace(baseline, disaster_pct=None, disaster_usd=100.0)),
        ("dis_usd=$200", replace(baseline, disaster_pct=None, disaster_usd=200.0)),
        ("dis_usd=$300", replace(baseline, disaster_pct=None, disaster_usd=300.0)),
        ("dis_usd=$500", replace(baseline, disaster_pct=None, disaster_usd=500.0)),
    ]
    usd_res = block("DOLLAR-CAP DISASTER SWEEP (user's rule shape)", usd_rows)

    # ---- IV relaxation sweep (less aggressive entry IV + slower crush) ----
    # Default model: entry_iv = RV * 1.25, crush over 10td to RV floor.
    # Real IV often stays elevated longer when underlying is choppy. These
    # configs test cheaper entries and/or longer IV-hold; both should
    # reduce the day-1-5 mark damage that triggers disasters in the model.
    iv_rows = [
        ("iv 1.25, crush 10 (default)",
         replace(baseline, iv_elevation=1.25, crush_td=10)),
        ("iv 1.25, crush 20",
         replace(baseline, iv_elevation=1.25, crush_td=20)),
        ("iv 1.15, crush 10",
         replace(baseline, iv_elevation=1.15, crush_td=10)),
        ("iv 1.10, crush 20",
         replace(baseline, iv_elevation=1.10, crush_td=20)),
        ("iv 1.00, crush 30",
         replace(baseline, iv_elevation=1.00, crush_td=30)),
    ]
    iv_res = block("IV RELAXATION SWEEP", iv_rows)

    # ---- combined user-style emulation -------------------------------------
    # $200 absolute loss cap (your rule) + relaxed IV (less day-1 mark
    # damage) + baseline TP+30, time 10td.
    user_rows = [
        ("user dis$200, iv 1.15/20",
         ExitCfg(disaster_pct=None, disaster_usd=200.0,
                 iv_elevation=1.15, crush_td=20)),
        ("user dis$200, iv 1.10/20, time 20",
         ExitCfg(disaster_pct=None, disaster_usd=200.0, time_stop=20,
                 iv_elevation=1.10, crush_td=20)),
        ("user dis$200, iv 1.00/30, time 30",
         ExitCfg(disaster_pct=None, disaster_usd=200.0, time_stop=30,
                 iv_elevation=1.00, crush_td=30)),
        ("user dis$200, iv 1.00/30, TP50, time 30",
         ExitCfg(tp_pct=50.0, disaster_pct=None, disaster_usd=200.0,
                 time_stop=30, iv_elevation=1.00, crush_td=30)),
        ("user dis$200, iv relaxed, sigma-revert",
         ExitCfg(tp_pct=None, disaster_pct=None, disaster_usd=200.0,
                 time_stop=45, sigma_target=0.0,
                 iv_elevation=1.00, crush_td=30)),
    ]
    user_res = block("USER-STYLE COMBINED EMULATION", user_rows)

    # ---- DTE sweep (real-trade audit said median 94d, not 45) -------------
    # The 45-60d band held only 7% of real trades. Test 60/90/120/150.
    dte_rows = [
        ("dte=45 (live)",  replace(baseline, dte=45)),
        ("dte=60",         replace(baseline, dte=60)),
        ("dte=90",         replace(baseline, dte=90)),
        ("dte=120",        replace(baseline, dte=120)),
        ("dte=150",        replace(baseline, dte=150)),
    ]
    dte_res = block("DTE SWEEP (real median was 94d)", dte_rows)

    # ---- OTM% sweep (real strikes look closer to ATM than 5% OTM) ---------
    otm_rows = [
        ("otm=0% (ATM)",   replace(baseline, otm_pct=0.0)),
        ("otm=2.5%",       replace(baseline, otm_pct=2.5)),
        ("otm=5% (live)",  replace(baseline, otm_pct=5.0)),
        ("otm=7.5%",       replace(baseline, otm_pct=7.5)),
        ("otm=10%",        replace(baseline, otm_pct=10.0)),
    ]
    otm_res = block("OTM%% SWEEP (real entries look ATM-ish)", otm_rows)

    # ---- REAL-TRADE-CALIBRATED CONFIGS ------------------------------------
    # Derived from research/out/personal_trade_audit.txt (44 paired option
    # trades, 2025): median DTE 94d, median exit +17%, p75 +30%, worst loss
    # -44%, median hold 3d, 24% of winners held past 10d, NO trades hit -50%.
    # Test the standardization rules I proposed in the analysis.
    real_rows = [
        # R0: just the DTE+OTM fixes, keep live exits.
        ("R0 dte90 ATM, live exits",
         replace(baseline, dte=90, otm_pct=0.0)),
        # R1: + TP +20% (real median exit). Disaster still -50%.
        ("R1 dte90 ATM TP20",
         replace(baseline, dte=90, otm_pct=0.0, tp_pct=20.0)),
        # R2: + tighten disaster to -30% (real p10 was -27%).
        ("R2 dte90 ATM TP20 dis30",
         replace(baseline, dte=90, otm_pct=0.0, tp_pct=20.0, disaster_pct=30.0)),
        # R3: + extend time stop to 30d (real winners hold 11-40d).
        ("R3 dte90 ATM TP20 dis30 time30",
         replace(baseline, dte=90, otm_pct=0.0, tp_pct=20.0,
                 disaster_pct=30.0, time_stop=30)),
        # R4: + relax IV (we found 1.00/30 was the dominant fix).
        ("R4 dte90 ATM TP20 dis30 t30 iv1.00/30",
         replace(baseline, dte=90, otm_pct=0.0, tp_pct=20.0,
                 disaster_pct=30.0, time_stop=30,
                 iv_elevation=1.00, crush_td=30)),
        # R5: full real-trade emulation — TP 20, dis $200 (user's $-cap rule),
        # no % disaster, time 30, relaxed IV.
        ("R5 dte90 ATM TP20 dis$200 t30 iv1.00/30",
         replace(baseline, dte=90, otm_pct=0.0, tp_pct=20.0,
                 disaster_pct=None, disaster_usd=200.0, time_stop=30,
                 iv_elevation=1.00, crush_td=30)),
        # R6: hold-to-peak variant — no TP, just $-cap + sigma-revert + time.
        ("R6 dte90 ATM no-TP dis$200 sigma-revert iv1.00/30",
         replace(baseline, dte=90, otm_pct=0.0, tp_pct=None,
                 disaster_pct=None, disaster_usd=200.0, time_stop=45,
                 sigma_target=0.0, iv_elevation=1.00, crush_td=30)),
    ]
    real_res = block("REAL-TRADE-CALIBRATED CONFIGS (from personal_trade_audit)", real_rows)

    # Tuned: best per axis by CAGR, stacked.
    def best(rs: dict[str, dict]) -> tuple[str, dict]:
        return max(rs.items(), key=lambda kv: kv[1].get("cagr", -1e9))

    bt_tag, bt_st = best(tp_res); bd_tag, bd_st = best(dis_res)
    btime_tag, btime_st = best(time_res); bsig_tag, bsig_st = best(sigma_res)
    btr_tag, btr_st = best(trail_res)

    # Extract numeric params for tuning from the stat dicts via the original rows.
    def cfg_of(rows: list[tuple], tag: str) -> ExitCfg:
        for t, c in rows:
            if t == tag:
                return c
        return baseline

    tuned = ExitCfg(
        tp_pct=cfg_of(tp_rows, bt_tag).tp_pct,
        disaster_pct=cfg_of(dis_rows, bd_tag).disaster_pct,
        time_stop=cfg_of(time_rows, btime_tag).time_stop,
        sigma_target=cfg_of(sigma_rows, bsig_tag).sigma_target,
        trail_activate=cfg_of(trail_rows, btr_tag).trail_activate,
        trail_giveback=cfg_of(trail_rows, btr_tag).trail_giveback,
    )
    tuned_st = simulate(series, z_by_tk, sig_by_day, tuned)

    lines.append("[BEST PER AXIS (by CAGR)]")
    lines.append(f"  TP        : {bt_tag}   (CAGR {bt_st.get('cagr', 0)*100:+.1f}%)")
    lines.append(f"  Disaster  : {bd_tag}   (CAGR {bd_st.get('cagr', 0)*100:+.1f}%)")
    lines.append(f"  Time      : {btime_tag}   (CAGR {btime_st.get('cagr', 0)*100:+.1f}%)")
    lines.append(f"  Sigma-tgt : {bsig_tag}   (CAGR {bsig_st.get('cagr', 0)*100:+.1f}%)")
    lines.append(f"  Trail     : {btr_tag}   (CAGR {btr_st.get('cagr', 0)*100:+.1f}%)")
    lines.append("")
    lines.append("[TUNED COMBINED]")
    lines.append(EHEADER)
    lines.append(_fmt_with_n("tuned", tuned_st))
    lines.append(_fmt_with_n("baseline", base_st))
    lines.append("")

    fig, ax = plt.subplots(figsize=(14, 7))
    for tag, st in (("baseline (live)", base_st), ("tuned", tuned_st)):
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.6,
                label=f"{tag} (CAGR {st['cagr']*100:+.1f}%, "
                      f"maxDD {st['max_dd']*100:.0f}%, Sharpe {st['sharpe']:.2f}, "
                      f"n={st['n']})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title("Oversold-call bot — baseline vs tuned exits "
                 f"(N={MAX_CONCURRENT}, OTM {OTM_PCT:.0f}%, {DTE} DTE)")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "oversold_call_exit_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "oversold_call_exit_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
