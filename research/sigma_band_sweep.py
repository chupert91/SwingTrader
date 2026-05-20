"""Sigma-band entry sweep on the volatile universe (call leg only).

The live bot fires on first-touch into z ∈ [-2.5, -2.0] (the "sweet spot"
band from research/oversold_playbook.md). User recalls past-3σ working
very well historically; this sweep tests whether a deeper band — same
engine, same R6 exits, same universe — beats the [-2.5, -2.0] default.

Method:
  - Pin the R6 / LIVE-OTM-sigma-revert exit config (PF 2.32 on the call
    leg, the live default).
  - Pin everything else (universe, IV model, capital, tier ranking).
  - Vary the ENTRY BAND only. Breadth tier is still computed at z<=-2.0
    crossings (the "is the broad market capitulating?" question is
    band-independent — a deep -3σ name in a market-wide capitulation is
    fundamentally different from one in isolation).

Bands tested (call leg; z is the negative side):
  shallow      [-2.0, -1.5]   wider net, weaker edge
  CURRENT      [-2.5, -2.0]   the live default
  deeper       [-3.0, -2.5]   past 2.5σ but not yet 3
  past-3       [-3.5, -3.0]   user's claim
  very deep    [-4.0, -3.5]   tail extremes
  open >=-2    [-inf, -2.0]   anything past 2σ
  open >=-2.5  [-inf, -2.5]   anything past 2.5σ
  open >=-3    [-inf, -3.0]   anything past 3σ
  open >=-3.5  [-inf, -3.5]   anything past 3.5σ

Reads from the shipped engine (simulate, _log_channel_z, ExitCfg) so
this is a pure entry-side experiment.

Run:
    python research/sigma_band_sweep.py
Outputs:
    research/out/sigma_band_sweep.png
    research/out/sigma_band_sweep.txt
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.volatile_universe import universe, theme_of  # noqa: E402
from research.oversold_call_exit_sweep import (  # noqa: E402
    ExitCfg, simulate, _log_channel_z, _tier_for, _fmt_with_n,
    ACCEPT_TIERS, TIER_RANK,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, _stats, HEADER, STARTING_CAPITAL, MIN_ADV_M,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Pinned exit config: R6 / LIVE-OTM-sigma-revert. The configuration that
# hit PF 2.32 on the call leg in volatile_universe_sweep.
PINNED_EXITS = ExitCfg(
    tp_pct=None,
    disaster_pct=None,
    disaster_usd=200.0,
    time_stop=45,
    sigma_target=0.0,
    iv_elevation=1.00,
    crush_td=30,
)

# Breadth is always computed at the canonical broad-market threshold (z
# <= -2.0). Deeper-band rows still benefit from / are gated by broad
# market capitulation.
BREADTH_HI = -2.0


@dataclass
class Sig:
    tk: str
    i: int
    z: float
    date: object
    tier: str = "?"
    breadth: int = 0
    adv: float = 0.0


def _signals_band(series: dict[str, Series], z_by_tk: dict[str, np.ndarray],
                  entry_lo: float, entry_hi: float) -> dict:
    """Same scan semantics as oversold_call_exit_sweep._signals but the
    band [entry_lo, entry_hi] is parametric. First-touch is defined as
    crossing into z <= entry_hi (the upper boundary of the band on the
    way DOWN). Breadth still counts crossings into z <= BREADTH_HI for
    direct comparability across rows."""
    breadth_by_day: Counter = Counter()
    eligible: list[Sig] = []
    for tk, s in series.items():
        z = z_by_tk[tk]
        for i in range(1, len(s.close)):
            if np.isnan(z[i]) or np.isnan(z[i - 1]):
                continue
            # Broad-market breadth: crossings into z <= -2.0.
            if z[i] <= BREADTH_HI and z[i - 1] > BREADTH_HI:
                breadth_by_day[s.dates[i]] += 1
            # Entry: crossing into the configured band on the way down.
            crossed_entry = z[i] <= entry_hi and z[i - 1] > entry_hi
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
        out.setdefault(sg.date, []).append(sg)
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
    print(f"  {len(series)} tickers with full 252d history")

    # Each row: (label, entry_lo, entry_hi). entry_lo can be -inf for
    # "open" bands that include arbitrarily deep z.
    NEG_INF = -1e9
    bands = [
        ("shallow [-2.0, -1.5]",      -2.0, -1.5),
        ("CURRENT [-2.5, -2.0]",      -2.5, -2.0),
        ("deeper  [-3.0, -2.5]",      -3.0, -2.5),
        ("past-3  [-3.5, -3.0]",      -3.5, -3.0),
        ("very deep [-4.0, -3.5]",    -4.0, -3.5),
        ("open >= -2.0  [-inf, -2.0]", NEG_INF, -2.0),
        ("open >= -2.5  [-inf, -2.5]", NEG_INF, -2.5),
        ("open >= -3.0  [-inf, -3.0]", NEG_INF, -3.0),
        ("open >= -3.5  [-inf, -3.5]", NEG_INF, -3.5),
    ]

    EHEADER = HEADER + "  | exit-reason mix"
    lines: list[str] = []
    lines.append("Sigma-band entry sweep on the call leg (volatile universe).")
    lines.append("Exit config PINNED to R6 / LIVE-OTM-sigma-revert (PF 2.32 baseline).")
    lines.append(f"Tickers: {len(series)}.   Breadth always counted at z<=-2.0 crossings.")
    lines.append("")
    lines.append("[BAND SWEEP — R6 exits, signals from CALL side]")
    lines.append(EHEADER)

    results: list[tuple[str, dict, int]] = []
    for label, lo, hi in bands:
        sig_by_day = _signals_band(series, z_by_tk, lo, hi)
        n_sig = sum(len(v) for v in sig_by_day.values())
        st = simulate(series, z_by_tk, sig_by_day, PINNED_EXITS)
        results.append((label, st, n_sig))
        lines.append(_fmt_with_n(label, st) + f"  | raw_sigs={n_sig}")
    lines.append("")

    # Per-band quick summary of WHERE signals came from (themes/tickers)
    # for the user-most-relevant rows.
    lines.append("[SIGNAL SOURCE — past-3 bands]")
    for label, lo, hi in bands:
        if "past-3" not in label and "open >= -3" not in label:
            continue
        sig_by_day = _signals_band(series, z_by_tk, lo, hi)
        themes: dict[str, int] = defaultdict(int)
        tks_in: dict[str, int] = defaultdict(int)
        for d, sigs in sig_by_day.items():
            for sg in sigs:
                themes[theme_of(sg.tk)] += 1
                tks_in[sg.tk] += 1
        if not tks_in:
            lines.append(f"  {label}: (no signals)")
            continue
        top_tks = sorted(tks_in, key=lambda k: -tks_in[k])[:10]
        lines.append(f"  {label}: signals={sum(tks_in.values())}, "
                     f"themes={dict(themes)}")
        lines.append(f"    top tickers: " + ", ".join(
            f"{tk}({theme_of(tk)},n={tks_in[tk]})" for tk in top_tks))
    lines.append("")

    # Headline.
    by_pf = max(results, key=lambda kv: kv[1].get("profit_factor", 0) or 0)
    by_cagr = max(results, key=lambda kv: kv[1].get("cagr", -1e9))
    by_sharpe = max(results, key=lambda kv: kv[1].get("sharpe", -1e9))
    lines.append("[HEADLINE]")
    lines.append(f"  best by CAGR   : {by_cagr[0]}   CAGR {by_cagr[1]['cagr']*100:+.1f}%  "
                 f"PF {by_cagr[1].get('profit_factor', 0):.2f}  Sharpe {by_cagr[1].get('sharpe', 0):.2f}  n={by_cagr[1].get('n', 0)}")
    lines.append(f"  best by PF     : {by_pf[0]}   CAGR {by_pf[1]['cagr']*100:+.1f}%  "
                 f"PF {by_pf[1].get('profit_factor', 0):.2f}  Sharpe {by_pf[1].get('sharpe', 0):.2f}  n={by_pf[1].get('n', 0)}")
    lines.append(f"  best by Sharpe : {by_sharpe[0]}   CAGR {by_sharpe[1]['cagr']*100:+.1f}%  "
                 f"PF {by_sharpe[1].get('profit_factor', 0):.2f}  Sharpe {by_sharpe[1].get('sharpe', 0):.2f}  n={by_sharpe[1].get('n', 0)}")

    fig, ax = plt.subplots(figsize=(14, 7))
    for label, st, n_sig in results:
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.5,
                label=f"{label}  (CAGR {st['cagr']*100:+.1f}% | "
                      f"PF {st.get('profit_factor', 0):.2f} | n={st['n']})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title("Sigma-band entry sweep, call leg, R6 exits, volatile universe")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "sigma_band_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "sigma_band_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
