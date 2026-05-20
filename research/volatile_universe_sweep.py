"""Exit sweep on the VOLATILE THEMATIC universe (research/volatile_universe).

Same engine as oversold_call_exit_sweep but the universe is restricted to
the 103 high-IV thematic names the user actually trades adjacent to (AI /
quantum / crypto-miners / critical-minerals-and-nuclear / EV / fintech /
biotech / etc.) instead of SP500.

If the hypothesis is right — that mean-reversion edge is structurally
bigger on volatile thematic names than on S&P large-caps — the SAME
configs that were CAGR -2..+2% on SP500 should be materially positive
here. If that holds, the live bot's universe is its ceiling.

Runs a focused config matrix (no full sweep grid) — just the configurations
that mattered in the SP500 run:
  - baseline (live defaults)
  - the seven REAL-TRADE-CALIBRATED configs (R0..R6)
  - the strongest user-style sigma-revert combo

Outputs:
    research/out/volatile_universe_sweep.png
    research/out/volatile_universe_sweep.txt
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from research.volatile_universe import universe, theme_of  # noqa: E402
from research.oversold_call_exit_sweep import (  # noqa: E402
    ExitCfg, simulate, _log_channel_z, _signals, _fmt_with_n,
    BASE_TP, BASE_DISASTER, BASE_TIME,
    DEFAULT_HALF_SPREAD, DEFAULT_ENTRY_BUFFER,
    OTM_PCT, DTE, MAX_CONCURRENT, MAX_PER_DAY,
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
    print(f"  got {len(bars)} usable tickers")

    series: dict[str, Series] = {}
    z_by_tk: dict[str, np.ndarray] = {}
    for tk, df in bars.items():
        s = build_series(tk, df)
        if s is not None:
            series[tk] = s
            z_by_tk[tk] = _log_channel_z(s.close)
    print(f"  {len(series)} tickers with full 252d-of-history available")
    sig_by_day = _signals(series, z_by_tk)
    n_sig = sum(len(v) for v in sig_by_day.values())
    print(f"  {n_sig} entry signals (tier PRIME/OK/PANIC, ADV-gated)\n")

    # Sanity: list which themes contributed signals
    sig_by_theme: dict[str, int] = defaultdict(int)
    tickers_with_sigs: dict[str, int] = defaultdict(int)
    for d, sigs in sig_by_day.items():
        for sg in sigs:
            sig_by_theme[theme_of(sg.tk)] += 1
            tickers_with_sigs[sg.tk] += 1

    baseline = ExitCfg()
    EHEADER = HEADER + "  | exit-reason mix"

    lines: list[str] = []
    lines.append("Exit sweep on the VOLATILE THEMATIC universe (103 names, AI/quantum/")
    lines.append("crypto/critical-minerals/EV/fintech/biotech) vs SP500 in oversold_call_exit_sweep.")
    lines.append(f"Tickers with full 252d history: {len(series)}.   Total entry signals: {n_sig}.")
    lines.append(f"Capital ${STARTING_CAPITAL:,.0f}   Max {MAX_CONCURRENT} concurrent / "
                 f"{MAX_PER_DAY}/day   Cost {DEFAULT_HALF_SPREAD*100:.1f}% half-spread.\n")

    lines.append("[SIGNALS PER THEME]")
    for theme in sorted(sig_by_theme, key=lambda k: -sig_by_theme[k]):
        lines.append(f"  {theme:18s} signals={sig_by_theme[theme]:4d}")
    lines.append("")
    lines.append("[TICKERS WITH SIGNALS] (top 25)")
    for tk in sorted(tickers_with_sigs, key=lambda k: -tickers_with_sigs[k])[:25]:
        lines.append(f"  {tk:7s} ({theme_of(tk):14s}) signals={tickers_with_sigs[tk]:3d}")
    lines.append("")

    # Config matrix — same as the REAL-TRADE-CALIBRATED block from the
    # SP500 sweep, so the rows are directly comparable.
    configs = [
        ("BASELINE (live defaults)",
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
        # Best PF config from the SP500 sweep (5% OTM sigma-revert)
        ("LIVE-OTM sigma-revert (SP500-best)",
         ExitCfg(tp_pct=None, disaster_pct=None, disaster_usd=200.0,
                 time_stop=45, sigma_target=0.0,
                 iv_elevation=1.00, crush_td=30)),
    ]

    lines.append("[CONFIGS ON VOLATILE UNIVERSE]")
    lines.append(EHEADER)
    results: list[tuple[str, dict]] = []
    for tag, cfg in configs:
        st = simulate(series, z_by_tk, sig_by_day, cfg)
        results.append((tag, st))
        lines.append(_fmt_with_n(tag, st))
    lines.append("")

    # Headline summary — pick the best by PF and best by CAGR
    by_pf = max(results, key=lambda kv: kv[1].get("profit_factor", 0) or 0)
    by_cagr = max(results, key=lambda kv: kv[1].get("cagr", -1e9))
    lines.append("[HEADLINE]")
    lines.append(f"  best by CAGR  : {by_cagr[0]}   CAGR {by_cagr[1]['cagr']*100:+.1f}%  "
                 f"PF {by_cagr[1].get('pf', 0):.2f}  Sharpe {by_cagr[1].get('sharpe', 0):.2f}")
    lines.append(f"  best by PF    : {by_pf[0]}   CAGR {by_pf[1]['cagr']*100:+.1f}%  "
                 f"PF {by_pf[1].get('pf', 0):.2f}  Sharpe {by_pf[1].get('sharpe', 0):.2f}")
    lines.append("")

    # Plot equity curves for top configs.
    fig, ax = plt.subplots(figsize=(14, 7))
    for tag, st in results:
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.5,
                label=f"{tag} (CAGR {st['cagr']*100:+.1f}% | PF "
                      f"{st.get('profit_factor', 0):.2f} | n={st['n']})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title(f"Volatile-universe exit sweep (n={len(series)} tickers, {n_sig} signals)")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "volatile_universe_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "volatile_universe_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
