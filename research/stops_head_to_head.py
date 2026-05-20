"""Disaster-stop head-to-head: 30% vs $200 cap (vs other variants).

User question: "should I run the 30% stop loss or the $200 stop?"
Going to answer with the data, not with intuition.

Setup matches the LIVE config so the comparison is honest:
  Universe       — volatile thematic basket (102 names, 5y)
  Signal         — HYBRID band: [-2.5,-2.0] OR z<=-3.5 (live default)
  Exits          — R6 base: no TP, sigma-revert at z=0, 45d time stop,
                   IV 1.00/30 crush. ONLY the disaster stop varies.

Stop variants:
  no-stop        — disaster_pct=None, disaster_usd=None  (control)
  30% only       — disaster_pct=30,   disaster_usd=None
  50% only       — disaster_pct=50,   disaster_usd=None  (legacy default)
  $100 only      — disaster_pct=None, disaster_usd=100
  $200 only      — disaster_pct=None, disaster_usd=200   (current default)
  $300 only      — disaster_pct=None, disaster_usd=300
  30% + $200     — both set, whichever fires first
  20% + $200     — both set, tighter %-leg

Run:
    python research/stops_head_to_head.py
Outputs:
    research/out/stops_head_to_head.png
    research/out/stops_head_to_head.txt
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.volatile_universe import universe  # noqa: E402
from research.oversold_call_exit_sweep import (  # noqa: E402
    ExitCfg, simulate, _log_channel_z, _fmt_with_n,
)
from research.calls_hybrid_band_sweep import (  # noqa: E402
    _signals_hybrid, PINNED_EXITS,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, HEADER, STARTING_CAPITAL,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


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
    print(f"  {len(series)} tickers with full 252d history")

    # HYBRID band signals (the live default).
    sig_by_day = _signals_hybrid(
        series, z_by_tk,
        primary_lo=-2.5, primary_hi=-2.0, deep_threshold=-3.5,
    )
    n_sig = sum(len(v) for v in sig_by_day.values())
    print(f"  {n_sig} entry signals (hybrid: primary + deep z<=-3.5)\n")

    # PINNED_EXITS is R6: tp_pct=None, disaster_pct=None,
    # disaster_usd=200.0, time_stop=45, sigma_target=0.0, iv 1.00/30.
    # We override JUST the disaster fields per variant.
    variants = [
        ("no stop (control)",
         replace(PINNED_EXITS, disaster_pct=None, disaster_usd=None)),
        ("30% only",
         replace(PINNED_EXITS, disaster_pct=30.0, disaster_usd=None)),
        ("50% only (legacy)",
         replace(PINNED_EXITS, disaster_pct=50.0, disaster_usd=None)),
        ("$100 only",
         replace(PINNED_EXITS, disaster_pct=None, disaster_usd=100.0)),
        ("$200 only (live default)",
         replace(PINNED_EXITS, disaster_pct=None, disaster_usd=200.0)),
        ("$300 only",
         replace(PINNED_EXITS, disaster_pct=None, disaster_usd=300.0)),
        ("30% + $200 (whichever first)",
         replace(PINNED_EXITS, disaster_pct=30.0, disaster_usd=200.0)),
        ("20% + $200 (tighter %)",
         replace(PINNED_EXITS, disaster_pct=20.0, disaster_usd=200.0)),
    ]

    EHEADER = HEADER + "  | exit-reason mix"
    lines: list[str] = []
    lines.append("DISASTER-STOP HEAD-TO-HEAD on the live config")
    lines.append(f"  universe : volatile thematic basket ({len(series)} tickers, 5y)")
    lines.append(f"  signal   : HYBRID band ([-2.5,-2.0] OR z<=-3.5)")
    lines.append(f"  exits    : R6 base (no TP, sigma=0, time 45d, IV 1.00/30)")
    lines.append(f"  vary     : disaster stop ONLY")
    lines.append(f"  signals  : {n_sig} raw, capital-constrained from there")
    lines.append("")
    lines.append(EHEADER)

    results: list[tuple[str, dict]] = []
    for label, cfg in variants:
        st = simulate(series, z_by_tk, sig_by_day, cfg)
        results.append((label, st))
        lines.append(_fmt_with_n(label, st))
    lines.append("")

    by_pf = max(results, key=lambda r: r[1].get("profit_factor", 0) or 0)
    by_cagr = max(results, key=lambda r: r[1].get("cagr", -1e9))
    by_sharpe = max(results, key=lambda r: r[1].get("sharpe", -1e9))
    by_mar = max(results, key=lambda r: r[1].get("mar", -1e9))

    lines.append("[HEADLINE]")
    lines.append(f"  best by CAGR   : {by_cagr[0]}   CAGR {by_cagr[1]['cagr']*100:+.1f}%  "
                 f"PF {by_cagr[1].get('profit_factor', 0):.2f}  Sharpe {by_cagr[1].get('sharpe', 0):.2f}")
    lines.append(f"  best by PF     : {by_pf[0]}   CAGR {by_pf[1]['cagr']*100:+.1f}%  "
                 f"PF {by_pf[1].get('profit_factor', 0):.2f}  Sharpe {by_pf[1].get('sharpe', 0):.2f}")
    lines.append(f"  best by Sharpe : {by_sharpe[0]}   CAGR {by_sharpe[1]['cagr']*100:+.1f}%  "
                 f"PF {by_sharpe[1].get('profit_factor', 0):.2f}  Sharpe {by_sharpe[1].get('sharpe', 0):.2f}")
    lines.append(f"  best by MAR    : {by_mar[0]}   CAGR {by_mar[1]['cagr']*100:+.1f}%  "
                 f"maxDD {by_mar[1].get('max_dd', 0)*100:.0f}%  MAR {by_mar[1].get('mar', 0):.2f}")
    lines.append("")

    # Pairwise comparison: 30% vs $200 — the question that motivated this.
    cfg_30 = next(st for l, st in results if l == "30% only")
    cfg_200 = next(st for l, st in results if l == "$200 only (live default)")
    lines.append("[30% vs $200 DIRECT COMPARISON]")
    for k, label in (("cagr", "CAGR"), ("profit_factor", "PF"),
                     ("sharpe", "Sharpe"), ("max_dd", "maxDD"),
                     ("n", "trades")):
        v30 = cfg_30.get(k, 0) or 0
        v200 = cfg_200.get(k, 0) or 0
        if k in ("cagr", "max_dd"):
            v30, v200 = v30 * 100, v200 * 100
        if k == "n":
            lines.append(f"  {label:>10s}  :  30% = {v30:>6.0f}  |  $200 = {v200:>6.0f}  |  diff = {v200 - v30:+.0f}")
        else:
            lines.append(f"  {label:>10s}  :  30% = {v30:>+6.2f}  |  $200 = {v200:>+6.2f}  |  diff = {v200 - v30:+.2f}")
    lines.append("")

    # Plot equity curves.
    fig, ax = plt.subplots(figsize=(14, 7))
    for label, st in results:
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.5,
                label=f"{label}  (CAGR {st['cagr']*100:+.1f}% | "
                      f"PF {st.get('profit_factor', 0):.2f} | "
                      f"DD {st.get('max_dd', 0)*100:.0f}% | n={st['n']})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title("Disaster-stop head-to-head (R6 exits, hybrid signal, volatile universe)")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "stops_head_to_head.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "stops_head_to_head.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
