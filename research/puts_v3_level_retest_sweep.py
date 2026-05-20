"""v3 puts with LEVEL-RETEST exit (the actual mental model).

At entry, we identify the broken_level (highest swing-high BELOW BTO close)
exactly as the v3 signal generator does. We then attach that PRICE level to
the position. During the trade, we close when the underlying close re-touches
that level within +/- band%.

This is what the user described:
  > "the price has a high probability of retesting levels it recently moved
  >  through, so I played on all these factors and took profits during the
  >  retest"

It's NOT a generic %-drop from entry. It's a ticker-specific PRICE the bot
knew about at the moment it opened the trade.

To support this, this script ships its own `simulate_put_with_level_exit`
(forks volatile_universe_puts_sweep.simulate_put with two changes):
  - Position carries `broken_level: float | None` captured at entry
  - New exit branch: close if close[idx] <= broken_level * (1 + band)
                     AND close[idx] >= broken_level * (1 - band)
    (i.e. underlying came back down INTO the +/- band around the level)

Sweep bands: {1, 2, 3, 5}% with time stops {7, 10, 21, 45}.

Ship gate: PF >= 1.5  AND  CAGR > 0  AND  n >= 10
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, replace
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.volatile_universe import universe  # noqa: E402
from research.theta_overlay import R  # noqa: E402
from research.oversold_call_exit_sweep import (  # noqa: E402
    ExitCfg, _log_channel_z, _iv_at, _fmt_with_n,
    TIER_RANK, MAX_CONCURRENT, MAX_PER_DAY, PREMIUM_CAP,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, _stats, HEADER, STARTING_CAPITAL, OPTION_MULT,
)
from research.volatile_universe_puts_sweep import _bs_put, _price_put  # noqa: E402
from research.puts_v3_level_sweep import (  # noqa: E402
    _signals_v3, SH20_LOOKBACK, SH50_LOOKBACK,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def _broken_level_at(s: Series, i: int) -> float | None:
    """Same computation _signals_v3 uses: highest of {sh20, sh50} that's below
    today's close. None if neither qualifies."""
    bto = float(s.close[i])
    lo20, hi20 = max(0, i - SH20_LOOKBACK[0]), max(0, i - SH20_LOOKBACK[1])
    sh20 = float(s.high[lo20:hi20].max()) if hi20 > lo20 else None
    lo50, hi50 = max(0, i - SH50_LOOKBACK[0]), max(0, i - SH50_LOOKBACK[1])
    sh50 = float(s.high[lo50:hi50].max()) if hi50 > lo50 else None
    broken = None
    for v in [sh20, sh50]:
        if v is not None and v < bto:
            broken = max(broken, v) if broken is not None else v
    return broken


@dataclass
class PosLR:
    tk: str
    entry_idx: int
    entry_date: object
    exp_date: object
    strike: float
    entry_cost: float
    entry_iv: float
    iv_floor: float
    qty: int
    broken_level: float | None = None
    peak_pnl: float = 0.0
    trail_activated: bool = False


def simulate_put_with_level_exit(
    series: dict[str, Series],
    z_by_tk: dict[str, np.ndarray],
    sig_by_day: dict,
    cfg: ExitCfg,
    level_retest_band_pct: float,
) -> dict:
    """Forked simulate_put that supports a PRICE-LEVEL retest exit.

    At entry, compute broken_level (highest sh20/sh50 below entry close) and
    attach to the position. During the trade, close when the underlying close
    re-enters the +/- band around that level (i.e. retests it from above).
    """
    band = level_retest_band_pct / 100.0
    all_dates = sorted({d for s in series.values() for d in s.dates})
    cash = STARTING_CAPITAL
    open_pos: dict[str, PosLR] = {}
    trades: list[dict] = []
    equity_curve: list[tuple] = []

    def mtm_close(pos: PosLR, s: Series, idx: int) -> float:
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
                if cfg.tp_pct is not None and realized_ret(s.low[idx]) >= cfg.tp_pct / 100.0:
                    exit_ret, reason = cfg.tp_pct / 100.0, "take_profit"
                else:
                    cr = realized_ret(s.close[idx])
                    if cr > pos.peak_pnl:
                        pos.peak_pnl = cr

                    hit_pct = (cfg.disaster_pct is not None
                               and cr <= -cfg.disaster_pct / 100.0)
                    hit_usd = (cfg.disaster_usd is not None
                               and -cr * total_cost_usd >= cfg.disaster_usd)
                    if hit_pct or hit_usd:
                        exit_ret, reason = cr, "disaster"
                    # LEVEL-RETEST EXIT: did underlying close come back down into
                    # the +/- band around the broken_level for this position?
                    elif (pos.broken_level is not None
                          and s.close[idx] <= pos.broken_level * (1.0 + band)
                          and s.close[idx] >= pos.broken_level * (1.0 - band)):
                        exit_ret, reason = cr, "level_retest"
                    elif (cfg.sigma_target is not None
                          and not np.isnan(z_by_tk[tk][idx])
                          and z_by_tk[tk][idx] <= -cfg.sigma_target):
                        exit_ret, reason = cr, "sigma_target"
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
                broken = _broken_level_at(s, i)
                open_pos[tk] = PosLR(tk, i, s.dates[i], exp_date, K, entry_cost,
                                      entry_iv, iv_floor, 1,
                                      broken_level=broken)
                placed_today += 1

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
    series: dict[str, Series] = {}
    z_by_tk: dict[str, np.ndarray] = {}
    for tk, df in bars.items():
        s = build_series(tk, df)
        if s is not None:
            series[tk] = s
            z_by_tk[tk] = _log_channel_z(s.close)
    print(f"  {len(series)} tickers")

    sig_by_day = _signals_v3(series, z_by_tk)
    n_sig = sum(len(v) for v in sig_by_day.values())
    print(f"  {n_sig} v3 signals\n")

    base = ExitCfg(
        tp_pct=None, disaster_pct=None, disaster_usd=300.0,
        time_stop=21, sigma_target=0.0,
        iv_elevation=1.00, crush_td=30,
        otm_pct=5.0, dte=90,
    )

    # Sweep retest band x time stop. The level retest IS the primary exit;
    # sigma + time + disaster are safety nets behind it.
    bands = [1.0, 2.0, 3.0, 5.0]
    time_stops = [7, 10, 21, 45]
    configs = []
    for ts in time_stops:
        for band in bands:
            cfg = replace(base, time_stop=ts)
            configs.append((f"band +/-{band:.0f}% + t={ts:>2d}d", cfg, band))
    # Reference: same v3 + sigma-revert WITHOUT level retest (for comparison)
    configs.append(("REF v3 + sigma + t=10 (no level retest)",
                    replace(base, time_stop=10), None))
    configs.append(("REF v3 + sigma + t=21 (no level retest)",
                    replace(base, time_stop=21), None))

    EHEADER = HEADER + "  | exit-reason mix"
    lines: list[str] = []
    lines.append("v3 PUTS + TICKER-SPECIFIC LEVEL-RETEST EXIT")
    lines.append("")
    lines.append(f"  signal  : v3 level-based (z>=+1, Pattern A or B, cooldown 5d)")
    lines.append(f"  exit    : close when UNDERLYING retests the broken_level captured")
    lines.append(f"            AT ENTRY (the swing high price the entry filter saw)")
    lines.append(f"  safety  : $300 disaster cap, sigma=0 (z back to centerline), time stop")
    lines.append(f"  vary    : retest band in {{1,2,3,5}}% x time stop {{7,10,21,45}}d")
    lines.append("")
    lines.append("Hypothesis: closing at the level the entry filter named (not at a")
    lines.append("generic %-drop) should capture the user's actual exit behavior.")
    lines.append("")
    lines.append(EHEADER)

    from research.volatile_universe_puts_sweep import simulate_put
    results: list[tuple[str, dict]] = []
    for tag, cfg, band in configs:
        if band is None:
            st = simulate_put(series, z_by_tk, sig_by_day, cfg)
        else:
            st = simulate_put_with_level_exit(series, z_by_tk, sig_by_day, cfg, band)
        results.append((tag, st))
        lines.append(_fmt_with_n(tag, st))
    lines.append("")

    by_pf = max(results, key=lambda kv: kv[1].get("profit_factor", 0) or 0)
    by_cagr = max(results, key=lambda kv: kv[1].get("cagr", -1e9))
    by_sharpe = max(results, key=lambda kv: kv[1].get("sharpe", -1e9))

    lines.append("[HEADLINE]")
    lines.append(f"  best by CAGR  : {by_cagr[0]}")
    lines.append(f"    CAGR {by_cagr[1]['cagr']*100:+.1f}%  PF {by_cagr[1].get('profit_factor', 0):.2f}  "
                 f"Sharpe {by_cagr[1].get('sharpe', 0):.2f}  maxDD {by_cagr[1].get('max_dd', 0)*100:.0f}%  "
                 f"n={by_cagr[1]['n']}")
    lines.append(f"  best by PF    : {by_pf[0]}")
    lines.append(f"    CAGR {by_pf[1]['cagr']*100:+.1f}%  PF {by_pf[1].get('profit_factor', 0):.2f}  "
                 f"Sharpe {by_pf[1].get('sharpe', 0):.2f}  maxDD {by_pf[1].get('max_dd', 0)*100:.0f}%  "
                 f"n={by_pf[1]['n']}")
    lines.append(f"  best by Sharpe: {by_sharpe[0]}")
    lines.append(f"    CAGR {by_sharpe[1]['cagr']*100:+.1f}%  PF {by_sharpe[1].get('profit_factor', 0):.2f}  "
                 f"Sharpe {by_sharpe[1].get('sharpe', 0):.2f}  maxDD {by_sharpe[1].get('max_dd', 0)*100:.0f}%  "
                 f"n={by_sharpe[1]['n']}")
    lines.append("")

    lines.append("[SHIP GATE]  PF >= 1.5  AND  CAGR > 0  AND  n >= 10")
    any_passed = False
    for tag, st in results:
        pf = st.get("profit_factor", 0) or 0
        cagr = st.get("cagr", 0)
        n = st.get("n", 0)
        passed = pf >= 1.5 and cagr > 0 and n >= 10
        any_passed = any_passed or passed
        verdict = "PASS" if passed else "fail"
        lines.append(f"  [{verdict}] {tag}")
        lines.append(f"           PF {pf:>5.2f}  CAGR {cagr*100:>+6.1f}%  n {n:>4d}  "
                     f"maxDD {st.get('max_dd',0)*100:>+4.0f}%")
    lines.append("")
    lines.append(f"[VERDICT] {'SHIP-WORTHY config found' if any_passed else 'NO config passes ship gate'}")

    fig, ax = plt.subplots(figsize=(14, 7))
    for tag, st in results:
        ec = st.get("equity_curve") or []
        if not ec: continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.4,
                label=f"{tag}  (CAGR {st['cagr']*100:+.1f}% | "
                      f"PF {st.get('profit_factor', 0):.2f} | n={st['n']})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title("v3 puts + ticker-specific level-retest exit")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "puts_v3_level_retest_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "puts_v3_level_retest_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
