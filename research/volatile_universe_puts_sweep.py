"""Put-leg exit sweep on the volatile thematic universe.

Direct mirror of research/volatile_universe_sweep.py with three direction
flips:

  ENTRY    z >= +2.0 first-touch (vs. z <= -2.0 for calls); band
           [+2.0, +2.5]. Breadth = same-day count of crossings UP into
           z >= +2.0.
  STRIKE   K = price * (1 - otm_pct/100)  (OTM put, below spot).
  PRICE    bs_put via put-call parity: P = C - S + K*exp(-r*T).
  TP CHK   intraday touch on bar LOW (not high) — put pops on
           underlying DOWN.
  SIGMA-T  fire when cur_z <= -sigma_target (vs >= +sigma_target).

Everything else — cost layer, IV model, capital frame, sizing,
tier ranking — is identical to the calls engine so the rows are
directly comparable to research/out/volatile_universe_sweep.txt.

Go / no-go gate per the put-leg spec: ship the leg if the
LIVE-OTM-sigma-revert config produces PF >= 1.5 here. Calls hit 2.32.
User's real 2025 put PF was 3.80 (research/out/personal_trade_audit.txt,
20 paired puts) but that was hand-picked; this run measures whether the
SAME automation that works for calls also works for puts.
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.volatile_universe import universe, theme_of  # noqa: E402
from research.theta_overlay import bs_call, R  # noqa: E402
from research.oversold_call_exit_sweep import (  # noqa: E402
    ExitCfg, _log_channel_z, _iv_at, _tier_for, _fmt_with_n,
    BASE_TP, BASE_DISASTER, BASE_TIME, DEFAULT_HALF_SPREAD,
    DEFAULT_ENTRY_BUFFER, OTM_PCT, DTE, MAX_CONCURRENT, MAX_PER_DAY,
    PREMIUM_CAP, ACCEPT_TIERS, TIER_RANK,
    LOG_WINDOW,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, _stats, HEADER, STARTING_CAPITAL,
    MIN_ADV_M, OPTION_MULT,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Put-leg band — direct mirror of the call leg's [-2.5, -2.0].
SWEET_LO_PUT = 2.0
SWEET_HI_PUT = 2.5


def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """OTM put price via put-call parity. Same r/iv/T inputs as bs_call so
    the IV model in _iv_at applies symmetrically."""
    if T <= 0.0:
        return max(K - S, 0.0)
    c = bs_call(S, K, T, r, sigma)
    return c - S + K * math.exp(-r * T)


@dataclass
class Sig:
    tk: str
    i: int
    z: float
    date: object
    tier: str = "?"
    breadth: int = 0
    adv: float = 0.0


def _signals_put(series: dict[str, Series], z_by_tk: dict[str, np.ndarray]) -> dict:
    """date -> list[Sig] for the PUT leg. Mirrors scan_candidates/_signals
    from the calls sweep with reversed cross direction:

      breadth: count of all crossings UP into z >= SWEET_LO_PUT today
      eligible: z lands in [SWEET_LO_PUT, SWEET_HI_PUT] AND ADV >= MIN_ADV_M
    """
    breadth_by_day: Counter = Counter()
    eligible: list[Sig] = []
    for tk, s in series.items():
        z = z_by_tk[tk]
        for i in range(1, len(s.close)):
            if np.isnan(z[i]) or np.isnan(z[i - 1]):
                continue
            crossed = z[i] >= SWEET_LO_PUT and z[i - 1] < SWEET_LO_PUT
            if crossed:
                breadth_by_day[s.dates[i]] += 1
            if not crossed:
                continue
            if not (SWEET_LO_PUT <= z[i] <= SWEET_HI_PUT):
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


def _price_put(S: float, K: float, exp_date, bar_date, iv: float) -> float:
    T = max((exp_date - bar_date).days, 0) / 365.0
    if T <= 0.0:
        return max(K - S, 0.0)
    return _bs_put(S, K, T, R, iv)


@dataclass
class Pos:
    tk: str
    entry_idx: int
    entry_date: object
    exp_date: object
    strike: float
    entry_cost: float
    entry_iv: float
    iv_floor: float
    qty: int
    peak_pnl: float = 0.0
    trail_activated: bool = False


def simulate_put(series: dict[str, Series], z_by_tk: dict[str, np.ndarray],
                 sig_by_day: dict, cfg: ExitCfg) -> dict:
    """Capital-constrained put-leg sim. Direction-flipped: strike below
    spot, TP intraday-touch on bar LOW, sigma-target fires on z <= -tgt."""
    all_dates = sorted({d for s in series.values() for d in s.dates})
    cash = STARTING_CAPITAL
    open_pos: dict[str, Pos] = {}
    trades: list[dict] = []
    equity_curve: list[tuple] = []

    def mtm_close(pos: Pos, s: Series, idx: int) -> float:
        iv = _iv_at(pos.entry_iv, pos.iv_floor, idx - pos.entry_idx, cfg.crush_td)
        mid = _price_put(s.close[idx], pos.strike, pos.exp_date, s.dates[idx], iv)
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
                proc = (_price_put(S, pos.strike, pos.exp_date, s.dates[idx], iv)
                        * (1 - cfg.half_spread) * OPTION_MULT * pos.qty)
                return proc / (pos.entry_cost * OPTION_MULT * pos.qty) - 1.0

            exit_ret, reason = None, None
            expired = s.dates[idx] >= pos.exp_date

            if expired:
                exit_ret, reason = realized_ret(s.close[idx]), "expiry"
            else:
                # TP intraday: a put's best mark on the day comes from the
                # bar LOW (underlying drop -> put pop).
                if cfg.tp_pct is not None and realized_ret(s.low[idx]) >= cfg.tp_pct / 100.0:
                    exit_ret, reason = cfg.tp_pct / 100.0, "take_profit"
                else:
                    cr = realized_ret(s.close[idx])
                    if cr > pos.peak_pnl:
                        pos.peak_pnl = cr
                    if (cfg.trail_activate is not None
                            and pos.peak_pnl >= cfg.trail_activate / 100.0):
                        pos.trail_activated = True

                    hit_pct = (cfg.disaster_pct is not None
                               and cr <= -cfg.disaster_pct / 100.0)
                    hit_usd = (cfg.disaster_usd is not None
                               and -cr * total_cost_usd >= cfg.disaster_usd)
                    if hit_pct or hit_usd:
                        exit_ret, reason = cr, "disaster"
                    elif (getattr(cfg, "underlying_target_pct", None) is not None
                          # PUT: profit when underlying FALLS by target%. Compare
                          # current close to entry close on the underlying.
                          and (s.close[idx] - s.close[pos.entry_idx]) / s.close[pos.entry_idx]
                              <= -cfg.underlying_target_pct / 100.0):
                        exit_ret, reason = cr, "underlying_target"
                    elif (cfg.sigma_target is not None
                          and not np.isnan(z_by_tk[tk][idx])
                          # Put sigma-target: close when z reverts DOWN to
                          # -sigma_target (mirror of calls' z >= sigma_target).
                          and z_by_tk[tk][idx] <= -cfg.sigma_target):
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
            for sg in sorted(sigs, key=lambda x: (TIER_RANK.get(x.tier, 9),
                                                   -x.breadth, x.tk)):
                if len(open_pos) >= MAX_CONCURRENT or placed_today >= MAX_PER_DAY:
                    break
                tk, i = sg.tk, sg.i
                if tk in open_pos:
                    continue
                s = series[tk]
                S0 = s.close[i]
                # OTM PUT: strike BELOW spot.
                K = S0 * (1.0 - cfg.otm_pct / 100.0)
                rv = s.rv[i]
                if np.isnan(rv):
                    continue
                entry_iv = min(max(rv * cfg.iv_elevation, 0.15), 1.20)
                iv_floor = min(max(rv, 0.10), 1.10)
                exp_date = s.dates[i] + timedelta(days=cfg.dte)
                mid0 = _price_put(S0, K, exp_date, s.dates[i], entry_iv)
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


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tks = universe()
    print(f"Fetching {len(tks)} volatile-universe tickers x 5y...")
    bars = fetch_bars_bulk(tks, period="5y")
    print(f"  got {len(bars)} usable tickers")

    series: dict[str, Series] = {}
    z_by_tk: dict[str, np.ndarray] = {}
    for tk, df in bars.items():
        s = build_series(tk, df)
        if s is not None:
            series[tk] = s
            z_by_tk[tk] = _log_channel_z(s.close)
    print(f"  {len(series)} tickers with full 252d history")
    sig_by_day = _signals_put(series, z_by_tk)
    n_sig = sum(len(v) for v in sig_by_day.values())
    print(f"  {n_sig} PUT entry signals (z>=+2.0 first-touch, tier-gated)\n")

    sig_by_theme: dict[str, int] = defaultdict(int)
    tickers_with_sigs: dict[str, int] = defaultdict(int)
    for d, sigs in sig_by_day.items():
        for sg in sigs:
            sig_by_theme[theme_of(sg.tk)] += 1
            tickers_with_sigs[sg.tk] += 1

    baseline = ExitCfg()
    EHEADER = HEADER + "  | exit-reason mix"

    lines: list[str] = []
    lines.append("PUT-LEG exit sweep on the volatile thematic universe (mirror of")
    lines.append("research/out/volatile_universe_sweep.txt — same engine, same configs,")
    lines.append("flipped to overbought first-touch + OTM put pricing + sigma <= -tgt).")
    lines.append(f"Tickers with full 252d history: {len(series)}.   Total PUT signals: {n_sig}.")
    lines.append(f"Capital ${STARTING_CAPITAL:,.0f}   Max {MAX_CONCURRENT} concurrent / "
                 f"{MAX_PER_DAY}/day.\n")

    lines.append("[PUT SIGNALS PER THEME]")
    for theme in sorted(sig_by_theme, key=lambda k: -sig_by_theme[k]):
        lines.append(f"  {theme:18s} signals={sig_by_theme[theme]:4d}")
    lines.append("")
    lines.append("[TOP TICKERS WITH PUT SIGNALS] (top 25)")
    for tk in sorted(tickers_with_sigs, key=lambda k: -tickers_with_sigs[k])[:25]:
        lines.append(f"  {tk:7s} ({theme_of(tk):14s}) signals={tickers_with_sigs[tk]:3d}")
    lines.append("")

    configs = [
        ("BASELINE (live-equivalent defaults)",
         baseline),
        ("R0 dte90 ATM, live exits",
         replace(baseline, dte=90, otm_pct=0.0)),
        ("R1 dte90 ATM TP20",
         replace(baseline, dte=90, otm_pct=0.0, tp_pct=20.0)),
        ("R2 dte90 ATM TP20 dis30",
         replace(baseline, dte=90, otm_pct=0.0, tp_pct=20.0, disaster_pct=30.0)),
        ("R3 dte90 ATM TP20 dis30 time30",
         replace(baseline, dte=90, otm_pct=0.0, tp_pct=20.0,
                 disaster_pct=30.0, time_stop=30)),
        ("R4 dte90 ATM TP20 dis30 t30 iv1.00/30",
         replace(baseline, dte=90, otm_pct=0.0, tp_pct=20.0,
                 disaster_pct=30.0, time_stop=30,
                 iv_elevation=1.00, crush_td=30)),
        ("R5 dte90 ATM TP20 dis$200 t30 iv1.00/30",
         replace(baseline, dte=90, otm_pct=0.0, tp_pct=20.0,
                 disaster_pct=None, disaster_usd=200.0, time_stop=30,
                 iv_elevation=1.00, crush_td=30)),
        ("R6 dte90 ATM no-TP dis$200 sigma-revert iv1.00/30",
         replace(baseline, dte=90, otm_pct=0.0, tp_pct=None,
                 disaster_pct=None, disaster_usd=200.0, time_stop=45,
                 sigma_target=0.0, iv_elevation=1.00, crush_td=30)),
        # The decision row: SAME config that hit PF 2.32 on calls.
        ("LIVE-OTM sigma-revert (decision row)",
         ExitCfg(tp_pct=None, disaster_pct=None, disaster_usd=200.0,
                 time_stop=45, sigma_target=0.0,
                 iv_elevation=1.00, crush_td=30)),
    ]

    lines.append("[CONFIGS ON VOLATILE UNIVERSE — PUT LEG]")
    lines.append(EHEADER)
    results: list[tuple[str, dict]] = []
    for tag, cfg in configs:
        st = simulate_put(series, z_by_tk, sig_by_day, cfg)
        results.append((tag, st))
        lines.append(_fmt_with_n(tag, st))
    lines.append("")

    by_pf = max(results, key=lambda kv: kv[1].get("profit_factor", 0) or 0)
    by_cagr = max(results, key=lambda kv: kv[1].get("cagr", -1e9))
    lines.append("[HEADLINE]")
    lines.append(f"  best by CAGR  : {by_cagr[0]}   CAGR {by_cagr[1]['cagr']*100:+.1f}%  "
                 f"PF {by_cagr[1].get('profit_factor', 0):.2f}  Sharpe {by_cagr[1].get('sharpe', 0):.2f}")
    lines.append(f"  best by PF    : {by_pf[0]}   CAGR {by_pf[1]['cagr']*100:+.1f}%  "
                 f"PF {by_pf[1].get('profit_factor', 0):.2f}  Sharpe {by_pf[1].get('sharpe', 0):.2f}")
    lines.append("")
    # Reference row from the call leg for direct compare.
    lines.append("[REFERENCE — call leg LIVE-OTM sigma-revert on same universe]")
    lines.append("  (from research/out/volatile_universe_sweep.txt)")
    lines.append("  call leg: CAGR +13.0% | PF 2.32 | Sharpe +0.61 | n=50 | maxDD -32%")
    lines.append("  GO/NO-GO: ship put leg if puts PF >= 1.5 on the decision row.")

    fig, ax = plt.subplots(figsize=(14, 7))
    for tag, st in results:
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.5,
                label=f"{tag} (CAGR {st['cagr']*100:+.1f}% | PF "
                      f"{st.get('profit_factor', 0):.2f} | n={st['n']})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title(f"PUT-leg volatile-universe sweep (n={len(series)} tickers, {n_sig} signals)")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "volatile_universe_puts_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "volatile_universe_puts_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
