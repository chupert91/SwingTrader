"""Put-leg extreme-band sweep with fast-exit configs.

Symmetric [+2.0, +2.5] put leg failed the ship gate (PF 0.82 at the R6
decision config). User hypothesis: overbought reversals are different
from oversold capitulations — only EXTREME overbought (z >= +3.5) gives
a real reversion edge, and the exit profile should differ (sharp, fast
reversals, not slow mean-reversion).

Method (band × exit grid):

  Bands (call-leg [-3.5,-3.0] dead-zone mirror suggests [+3.0,+3.5]
  might be a dead zone too; we test it):
    [+2.0, +2.5]   mirror of original call sweet — KNOWN to fail
    [+2.5, +3.0]   moderately overbought
    [+3.0, +3.5]   high overbought (mirror of dead zone?)
    [+3.5, +4.0]   extreme (mirror of [-4.0,-3.5] gold)
    open >= +3.5   open-ended deep
    open >= +4.0   ultra deep

  Exit profiles (fast-reversal capture, NOT mean-reversion):
    R6        (baseline, mirror of calls' winner) — no TP, sigma-revert,
              $200 cap, 45d time stop, IV 1.00/30
    FAST20    TP +20%, $200 cap, 5d time stop, IV 1.00/30 — quick win take
    FAST30    TP +30%, $200 cap, 5d time stop, IV 1.00/30 — bigger
              quick-win
    REVERT-1  no TP, sigma-target +1.0 (close earlier), $200 cap, 10d
              time, IV 1.00/30 — capture early reversal without waiting
              to z=0
    REVERT+0  same as R6 but time stop 10d (sharp/fail-fast variant)

That's 6 bands x 5 exit configs = 30 rows. Per-row: simulate_put from
volatile_universe_puts_sweep with parameterized band.

Ship gate (per the puts spec): any cell with PF >= 1.5 AND positive
CAGR is a candidate to ship.

Run:
    python research/puts_extreme_band_sweep.py
Outputs:
    research/out/puts_extreme_band_sweep.png
    research/out/puts_extreme_band_sweep.txt
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
from research.oversold_call_exit_sweep import (  # noqa: E402
    ExitCfg, _log_channel_z, _tier_for, _fmt_with_n,
    ACCEPT_TIERS, TIER_RANK,
)
from research.volatile_universe_puts_sweep import (  # noqa: E402
    simulate_put, Sig as _PutSig,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, _stats, HEADER, STARTING_CAPITAL, MIN_ADV_M,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Breadth still computed at the CANONICAL broad threshold (z >= +2.0
# crossings) — answers "is the broad market also overbought today?"
# regardless of how extreme the individual name is.
BREADTH_LO = 2.0


@dataclass
class Sig:
    tk: str
    i: int
    z: float
    date: object
    tier: str = "?"
    breadth: int = 0
    adv: float = 0.0


def _signals_put_band(series: dict[str, Series], z_by_tk: dict[str, np.ndarray],
                      entry_lo: float, entry_hi: float) -> dict:
    """Same scan semantics as volatile_universe_puts_sweep._signals_put
    but the band [entry_lo, entry_hi] is parametric. First-touch defined
    as crossing into z >= entry_lo (lower boundary on the way UP).
    Breadth counts crossings into z >= BREADTH_LO."""
    breadth_by_day: Counter = Counter()
    eligible: list[Sig] = []
    for tk, s in series.items():
        z = z_by_tk[tk]
        for i in range(1, len(s.close)):
            if np.isnan(z[i]) or np.isnan(z[i - 1]):
                continue
            if z[i] >= BREADTH_LO and z[i - 1] < BREADTH_LO:
                breadth_by_day[s.dates[i]] += 1
            crossed_entry = z[i] >= entry_lo and z[i - 1] < entry_lo
            if not crossed_entry:
                continue
            if not (entry_lo <= z[i] <= entry_hi):
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
        # simulate_put expects the _PutSig dataclass from
        # volatile_universe_puts_sweep — re-pack with same field names.
        out.setdefault(sg.date, []).append(_PutSig(
            tk=sg.tk, i=sg.i, z=sg.z, date=sg.date,
            tier=sg.tier, breadth=sg.breadth, adv=sg.adv))
    return out


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
    print(f"  {len(series)} tickers with full 252d history\n")

    POS_INF = 1e9
    bands = [
        ("[+2.0,+2.5] mirror",  2.0,  2.5),
        ("[+2.5,+3.0]",         2.5,  3.0),
        ("[+3.0,+3.5]",         3.0,  3.5),
        ("[+3.5,+4.0] extreme", 3.5,  4.0),
        ("open >= +3.5",        3.5,  POS_INF),
        ("open >= +4.0",        4.0,  POS_INF),
    ]

    exits = [
        ("R6 (no-TP, sigma=0, t45)",
         ExitCfg(tp_pct=None, disaster_pct=None, disaster_usd=200.0,
                 time_stop=45, sigma_target=0.0,
                 iv_elevation=1.00, crush_td=30)),
        ("FAST20 (TP20, t5, $200)",
         ExitCfg(tp_pct=20.0, disaster_pct=None, disaster_usd=200.0,
                 time_stop=5, sigma_target=None,
                 iv_elevation=1.00, crush_td=30)),
        ("FAST30 (TP30, t5, $200)",
         ExitCfg(tp_pct=30.0, disaster_pct=None, disaster_usd=200.0,
                 time_stop=5, sigma_target=None,
                 iv_elevation=1.00, crush_td=30)),
        ("REVERT+1 (no-TP, sigma=1, t10)",
         ExitCfg(tp_pct=None, disaster_pct=None, disaster_usd=200.0,
                 time_stop=10, sigma_target=1.0,
                 iv_elevation=1.00, crush_td=30)),
        ("FAIL-FAST (no-TP, sigma=0, t10)",
         ExitCfg(tp_pct=None, disaster_pct=None, disaster_usd=200.0,
                 time_stop=10, sigma_target=0.0,
                 iv_elevation=1.00, crush_td=30)),
    ]

    EHEADER = HEADER + "  | exit-reason mix"
    lines: list[str] = []
    lines.append("PUT-LEG band x exit-config grid on the volatile universe.")
    lines.append("All configs use OTM 5%, DTE 45 (matching live bot OTM_PCT/DTE).")
    lines.append("Ship gate: any cell with PF >= 1.5 AND positive CAGR.")
    lines.append("")

    all_results: list[tuple[str, str, dict, int]] = []  # (band, exit, st, n_sig)

    for band_label, lo, hi in bands:
        sig_by_day = _signals_put_band(series, z_by_tk, lo, hi)
        n_sig = sum(len(v) for v in sig_by_day.values())
        lines.append(f"[BAND {band_label}]   raw_signals={n_sig}")
        lines.append(EHEADER)
        for ex_label, cfg in exits:
            st = simulate_put(series, z_by_tk, sig_by_day, cfg)
            all_results.append((band_label, ex_label, st, n_sig))
            lines.append(_fmt_with_n(ex_label, st))
        lines.append("")

    # Find ship-gate passers and the overall best.
    passers = [(b, e, st) for b, e, st, _ in all_results
               if (st.get("profit_factor", 0) or 0) >= 1.5
               and (st.get("cagr", 0) or 0) > 0
               and st.get("n", 0) >= 5]
    by_pf = max(all_results, key=lambda r: r[2].get("profit_factor", 0) or 0)
    by_cagr = max(all_results, key=lambda r: r[2].get("cagr", -1e9))

    lines.append("[SHIP-GATE PASSERS]  (PF >= 1.5 AND CAGR > 0 AND n >= 5)")
    if not passers:
        lines.append("  (none)")
    else:
        for b, e, st in sorted(passers, key=lambda x: -(x[2].get("profit_factor", 0) or 0)):
            lines.append(f"  {b} + {e} : CAGR {st['cagr']*100:+.1f}% "
                         f"PF {st['profit_factor']:.2f} Sharpe {st['sharpe']:.2f} n={st['n']}")
    lines.append("")
    lines.append("[OVERALL BEST]")
    lines.append(f"  by CAGR : {by_cagr[0]} + {by_cagr[1]} -> "
                 f"CAGR {by_cagr[2]['cagr']*100:+.1f}% PF {by_cagr[2].get('profit_factor', 0):.2f} "
                 f"Sharpe {by_cagr[2].get('sharpe', 0):.2f} n={by_cagr[2].get('n', 0)}")
    lines.append(f"  by PF   : {by_pf[0]} + {by_pf[1]} -> "
                 f"CAGR {by_pf[2]['cagr']*100:+.1f}% PF {by_pf[2].get('profit_factor', 0):.2f} "
                 f"Sharpe {by_pf[2].get('sharpe', 0):.2f} n={by_pf[2].get('n', 0)}")

    # Plot the winning equity curves from each band.
    fig, ax = plt.subplots(figsize=(14, 7))
    # Pick best exit per band by PF.
    best_per_band: dict[str, tuple] = {}
    for b, e, st, n in all_results:
        cur = best_per_band.get(b)
        if cur is None or (st.get("profit_factor", 0) or 0) > (cur[1].get("profit_factor", 0) or 0):
            best_per_band[b] = (e, st)
    for b, (e, st) in best_per_band.items():
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.5,
                label=f"{b} | best: {e} "
                      f"(PF {st.get('profit_factor', 0):.2f}, CAGR {st['cagr']*100:+.1f}%, n={st['n']})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title("PUT-leg band x exit grid: best-PF exit per band, volatile universe")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "puts_extreme_band_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "puts_extreme_band_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
