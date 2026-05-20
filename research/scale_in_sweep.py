"""Scale-in (averaging-down) sweep on the volatile-universe calls leg.

User question: what does the P&L curve look like if instead of a stop loss
we ADD another contract when the trade goes against us? Spec:
  - Trigger: underlying z drops Z_STEP sigma further into the band from
    the prior lot's entry z (e.g. lot1 at z=-2.2, lot2 at z=-2.7 with
    step 0.5, lot3 at z=-3.2, etc.)
  - Max adds per trade: sweep {1, 2, 3}
  - Disaster stop: DISABLED entirely. Trade rides to expiry / sigma-target
    / time stop regardless of drawdown depth. Pure mean-reversion conviction.
  - Each add is 1 contract at the current ask (same option symbol),
    subject to available cash; skipped if BP is exhausted.

This isn't a free lunch -- adds use BP that could have funded a NEW
signal. We let the sim's existing cash/concurrency frame enforce that:
adds consume cash before new entries get to claim slots.

Baseline rows for compare:
  R6 LIVE  -- shipped config ($200 disaster cap, 1 contract, no scale-in)
  R6 NO-STOP -- same minus disaster ($cap=None, no scale-in) -- isolates
                the value of the cap removal from the value of adds.

Sweep:
  Z_STEP    in {0.25, 0.5, 0.75, 1.0}
  MAX_ADDS  in {1, 2, 3}    (16 cells)

Outputs:
  research/out/scale_in_sweep.png
  research/out/scale_in_sweep.txt
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from research.volatile_universe import universe  # noqa: E402
from research.theta_overlay import bs_call, R  # noqa: E402
from research.oversold_call_exit_sweep import (  # noqa: E402
    ExitCfg, _log_channel_z, _signals, _fmt_with_n, _iv_at, _price,
    MAX_CONCURRENT, MAX_PER_DAY, PREMIUM_CAP,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, _stats, HEADER, STARTING_CAPITAL,
    OPTION_MULT,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Anchor exits (R6 minus disaster -- per user's "remove disaster entirely")
EXITS_NO_STOP = ExitCfg(
    tp_pct=None,
    disaster_pct=None, disaster_usd=None,    # explicitly OFF
    time_stop=45,
    sigma_target=0.0,
    iv_elevation=1.00, crush_td=30,
    otm_pct=5.0, dte=45,
)

# For comparison: the actual shipped config (with $200 disaster).
EXITS_LIVE_R6 = ExitCfg(
    tp_pct=None,
    disaster_pct=None, disaster_usd=200.0,
    time_stop=45,
    sigma_target=0.0,
    iv_elevation=1.00, crush_td=30,
    otm_pct=5.0, dte=45,
)

MIN_BARS_BETWEEN_ADDS = 3   # avoid 3-add-in-a-day on a single capitulation bar


@dataclass
class Lot:
    entry_idx: int
    entry_date: object
    entry_z: float
    entry_cost: float    # per-contract premium paid
    entry_iv: float
    iv_floor: float


@dataclass
class Pos:
    tk: str
    exp_date: object
    strike: float
    lots: list = field(default_factory=list)
    last_add_idx: int = -10_000      # bar index of most recent add


def _mtm_lot_close(lot: Lot, S: float, strike: float, exp_date, bar_date,
                   crush_td: int, half_spread: float) -> float:
    iv = _iv_at(lot.entry_iv, lot.iv_floor, _holding(lot, bar_date), crush_td)
    mid = _price(S, strike, exp_date, bar_date, iv)
    return mid * (1 - half_spread) * OPTION_MULT


def _holding(lot: Lot, bar_date) -> int:
    """Bars held -- approximate via day delta since the sim is daily."""
    try:
        return max(0, (bar_date - lot.entry_date).days)
    except Exception:
        return 0


def _pos_total_cost(pos: Pos) -> float:
    return sum(l.entry_cost * OPTION_MULT for l in pos.lots)


def _pos_qty(pos: Pos) -> int:
    return len(pos.lots)


def _pos_mtm_close(pos: Pos, s: Series, idx: int, cfg: ExitCfg) -> float:
    """MTM all lots at the close mark."""
    total = 0.0
    for lot in pos.lots:
        total += _mtm_lot_close(lot, s.close[idx], pos.strike, pos.exp_date,
                                s.dates[idx], cfg.crush_td, cfg.half_spread)
    return total


def _pos_realized_ret(pos: Pos, S: float, bar_date, cfg: ExitCfg) -> float:
    """Aggregate option-return (mark/cost - 1) across all lots at price S."""
    total_value = 0.0
    for lot in pos.lots:
        iv = _iv_at(lot.entry_iv, lot.iv_floor, _holding(lot, bar_date), cfg.crush_td)
        mid = _price(S, pos.strike, pos.exp_date, bar_date, iv)
        total_value += mid * (1 - cfg.half_spread) * OPTION_MULT
    total_cost = _pos_total_cost(pos)
    return (total_value / total_cost - 1.0) if total_cost > 0 else 0.0


def simulate_scalein(
    series: dict[str, Series],
    z_by_tk: dict[str, np.ndarray],
    sig_by_day: dict,
    cfg: ExitCfg,
    z_step: float | None,    # None disables adds (baseline)
    max_adds: int,
) -> dict:
    """Scale-in capable sim. Multi-lot positions; adds triggered when
    underlying z drops z_step further into the band from the prior lot's
    entry z (so add1 at entry_z - z_step, add2 at entry_z - 2*z_step, ...).
    Each add is 1 contract priced at the current bar's IV-modeled mid+spread."""
    all_dates = sorted({d for s in series.values() for d in s.dates})
    cash = STARTING_CAPITAL
    open_pos: dict[str, Pos] = {}
    trades: list[dict] = []
    equity_curve: list[tuple] = []

    for d in all_dates:
        # ---- exits ----
        for tk in sorted(list(open_pos.keys())):
            pos = open_pos[tk]
            s = series[tk]
            idx = s.by_date.get(d)
            if idx is None or idx <= pos.lots[0].entry_idx:
                continue
            held = idx - pos.lots[0].entry_idx
            total_cost = _pos_total_cost(pos)

            exit_ret, reason = None, None
            expired = s.dates[idx] >= pos.exp_date

            if expired:
                exit_ret, reason = _pos_realized_ret(pos, s.close[idx], s.dates[idx], cfg), "expiry"
            else:
                # Intraday TP touch (high). Computed against AVERAGE basis.
                if cfg.tp_pct is not None and _pos_realized_ret(pos, s.high[idx], s.dates[idx], cfg) >= cfg.tp_pct / 100.0:
                    exit_ret, reason = cfg.tp_pct / 100.0, "take_profit"
                else:
                    cr = _pos_realized_ret(pos, s.close[idx], s.dates[idx], cfg)
                    # Disaster (% or $) -- typically OFF in scale-in configs.
                    total_cost_usd = total_cost
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
                    elif cfg.time_stop is not None and held >= cfg.time_stop:
                        exit_ret, reason = cr, "time_stop"

            if exit_ret is not None:
                exit_ret = max(exit_ret, -1.0)
                proceeds = total_cost * (1.0 + exit_ret)
                cash += proceeds
                trades.append({
                    "ticker": tk,
                    "entry_date": str(pos.lots[0].entry_date),
                    "exit_date": str(s.dates[idx]),
                    "bars_held": held,
                    "opt_ret": exit_ret,
                    "exit_reason": reason,
                    "cost": total_cost,
                    "n_lots": _pos_qty(pos),
                })
                del open_pos[tk]
                continue

            # ---- ADD trigger ----
            if z_step is None or len(pos.lots) > max_adds:
                continue
            if idx - pos.last_add_idx < MIN_BARS_BETWEEN_ADDS:
                continue
            z_now = z_by_tk[tk][idx]
            if np.isnan(z_now):
                continue
            entry_z = pos.lots[0].entry_z
            # Trigger when z has dropped >= len(lots)*z_step below entry_z.
            trigger_z = entry_z - z_step * len(pos.lots)
            if z_now > trigger_z:
                continue
            # Price the add at current IV (RV at this bar), same OTM strike.
            rv_idx = s.rv[idx]
            if np.isnan(rv_idx):
                continue
            entry_iv = min(max(rv_idx * cfg.iv_elevation, 0.15), 1.20)
            iv_floor = min(max(rv_idx, 0.10), 1.10)
            mid = _price(s.close[idx], pos.strike, pos.exp_date, s.dates[idx], entry_iv)
            if mid <= 1e-6:
                continue
            add_cost = mid * (1.0 + cfg.half_spread) * (1.0 + cfg.entry_buffer)
            add_usd = add_cost * OPTION_MULT
            if add_usd > PREMIUM_CAP or add_usd > cash:
                continue
            cash -= add_usd
            pos.lots.append(Lot(
                entry_idx=idx, entry_date=s.dates[idx], entry_z=float(z_now),
                entry_cost=add_cost, entry_iv=entry_iv, iv_floor=iv_floor,
            ))
            pos.last_add_idx = idx

        # ---- entries (new positions only -- adds handled above) ----
        sigs = sig_by_day.get(d, [])
        if sigs and len(open_pos) < MAX_CONCURRENT:
            placed_today = 0
            from research.oversold_call_exit_sweep import TIER_RANK
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
                open_pos[tk] = Pos(
                    tk=tk, exp_date=exp_date, strike=K,
                    lots=[Lot(entry_idx=i, entry_date=s.dates[i],
                              entry_z=float(z_by_tk[tk][i]),
                              entry_cost=entry_cost, entry_iv=entry_iv,
                              iv_floor=iv_floor)],
                )
                placed_today += 1

        # ---- equity snapshot ----
        held_val = 0.0
        for tk, pos in open_pos.items():
            s = series[tk]
            idx = s.by_date.get(d)
            if idx is not None and idx >= pos.lots[0].entry_idx:
                held_val += _pos_mtm_close(pos, s, idx, cfg)
        equity_curve.append((d, cash + held_val))

    return _stats(trades, equity_curve)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tks = universe()
    print(f"Fetching {len(tks)} volatile-universe tickers x 5y...")
    bars = fetch_bars_bulk(tks, period="5y")
    series, z_by_tk = {}, {}
    for tk, df in bars.items():
        s = build_series(tk, df)
        if s is not None:
            series[tk] = s
            z_by_tk[tk] = _log_channel_z(s.close)
    print(f"  {len(series)} tickers with full 252d history")
    sig_by_day = _signals(series, z_by_tk)
    n_sig = sum(len(v) for v in sig_by_day.values())
    print(f"  {n_sig} entry signals\n")

    lines = []
    lines.append("Scale-in (averaging-down) sweep on the volatile-universe calls leg.")
    lines.append("Trigger: underlying z drops Z_STEP sigma further into the band from the")
    lines.append(f"prior lot's z. Min {MIN_BARS_BETWEEN_ADDS}b between adds; each add 1 contract.")
    lines.append("All scale-in rows have disaster DISABLED (no stop). Time stop 45td, sigma")
    lines.append("target 0, dte 45, 5% OTM (R6 minus disaster).\n")

    EHEADER = HEADER + "  | exit-reason mix"

    # Baselines for compare
    base_live = simulate_scalein(series, z_by_tk, sig_by_day,
                                  EXITS_LIVE_R6, z_step=None, max_adds=0)
    base_nostop = simulate_scalein(series, z_by_tk, sig_by_day,
                                    EXITS_NO_STOP, z_step=None, max_adds=0)
    lines.append("[BASELINES]")
    lines.append(EHEADER)
    lines.append(_fmt_with_n("R6 LIVE   ($200 cap, 0 adds)", base_live))
    lines.append(_fmt_with_n("R6 NO-STOP (no disaster, 0 adds)", base_nostop))
    lines.append("")

    # Sweep
    lines.append("[SCALE-IN SWEEP]  (no TP, no disaster, max_adds x z_step)")
    lines.append(EHEADER)
    results: list[tuple[str, dict, dict]] = []   # (tag, st, params)
    for max_adds in (1, 2, 3):
        for z_step in (0.25, 0.5, 0.75, 1.0):
            st = simulate_scalein(series, z_by_tk, sig_by_day,
                                   EXITS_NO_STOP, z_step=z_step, max_adds=max_adds)
            tag = f"adds<={max_adds}  z_step={z_step:.2f}"
            results.append((tag, st, {"max_adds": max_adds, "z_step": z_step}))
            lines.append(_fmt_with_n(tag, st))
    lines.append("")

    # TP=+30% layered on top. This is the only exit shape where scale-in
    # could theoretically help -- with a TP triggered against the AVERAGE
    # cost basis, a deeper add lowers the basis and makes the TP easier
    # to clear. Tests whether the basis-averaging actually flips losers
    # to winners when there's a P&L-based exit in the mix.
    EXITS_TP30_NO_STOP = ExitCfg(
        tp_pct=30.0,
        disaster_pct=None, disaster_usd=None,
        time_stop=45,
        sigma_target=0.0,
        iv_elevation=1.00, crush_td=30,
        otm_pct=5.0, dte=45,
    )
    base_tp30 = simulate_scalein(series, z_by_tk, sig_by_day,
                                  EXITS_TP30_NO_STOP, z_step=None, max_adds=0)
    lines.append("[TP=+30% BASELINE + SCALE-IN SWEEP]  (TP on average basis, no disaster)")
    lines.append(EHEADER)
    lines.append(_fmt_with_n("R6 NO-STOP + TP30 (0 adds)", base_tp30))
    results_tp: list[tuple[str, dict, dict]] = []
    for max_adds in (1, 2, 3):
        for z_step in (0.25, 0.5, 0.75, 1.0):
            st = simulate_scalein(series, z_by_tk, sig_by_day,
                                   EXITS_TP30_NO_STOP, z_step=z_step, max_adds=max_adds)
            tag = f"TP30 adds<={max_adds}  z_step={z_step:.2f}"
            results_tp.append((tag, st, {"max_adds": max_adds, "z_step": z_step}))
            lines.append(_fmt_with_n(tag, st))
    lines.append("")

    # Headline picks
    def pf(st): return st.get("profit_factor", 0) or 0
    def cagr(st): return st.get("cagr", -1e9)
    all_rows = ([("R6 LIVE", base_live), ("R6 NO-STOP", base_nostop),
                 ("R6 NO-STOP + TP30", base_tp30)]
                + [(t, st) for t, st, _ in results]
                + [(t, st) for t, st, _ in results_tp])
    best_pf = max(all_rows, key=lambda kv: pf(kv[1]))
    best_cagr = max(all_rows, key=lambda kv: cagr(kv[1]))
    best_mar = max(all_rows, key=lambda kv: kv[1].get("mar", -1e9))
    lines.append("[HEADLINE]")
    lines.append(f"  best PF   : {best_pf[0]}    PF {pf(best_pf[1]):.2f}  "
                 f"CAGR {cagr(best_pf[1])*100:+.1f}%  MaxDD {best_pf[1].get('max_dd', 0)*100:.1f}%")
    lines.append(f"  best CAGR : {best_cagr[0]}    CAGR {cagr(best_cagr[1])*100:+.1f}%  "
                 f"PF {pf(best_cagr[1]):.2f}  MaxDD {best_cagr[1].get('max_dd', 0)*100:.1f}%")
    lines.append(f"  best MAR  : {best_mar[0]}    MAR {best_mar[1].get('mar', 0):.2f}  "
                 f"CAGR {cagr(best_mar[1])*100:+.1f}%  PF {pf(best_mar[1]):.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    # Always plot baselines
    for tag, st in (("R6 LIVE ($200 cap)", base_live),
                    ("R6 NO-STOP", base_nostop),
                    ("R6 NO-STOP + TP30", base_tp30)):
        ec = st.get("equity_curve") or []
        if ec:
            ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=2.0,
                    label=f"{tag} (CAGR {st['cagr']*100:+.1f}%, PF {pf(st):.2f})")
    # Plot scale-in winners (top 3 from each block for clarity)
    for tag, st, _ in sorted(results, key=lambda x: -cagr(x[1]))[:3]:
        ec = st.get("equity_curve") or []
        if ec:
            ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.0, alpha=0.7,
                    label=f"{tag} (CAGR {st['cagr']*100:+.1f}%, PF {pf(st):.2f})")
    for tag, st, _ in sorted(results_tp, key=lambda x: -cagr(x[1]))[:3]:
        ec = st.get("equity_curve") or []
        if ec:
            ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.2, alpha=0.85,
                    label=f"{tag} (CAGR {st['cagr']*100:+.1f}%, PF {pf(st):.2f})", linestyle="--")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.5)
    ax.set_title("Scale-in vs stop loss -- volatile universe, calls leg, R6-style exits")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "scale_in_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "scale_in_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
