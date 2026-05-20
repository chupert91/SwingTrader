"""v3 puts + underlying-drop-target exit (the user's actual mental model).

User's real put exits, on the UNDERLYING:
  GOOG  -4.7% in 4d  -> +35% option
  APLD -18.9% in 7d  -> +34% option
  IONQ  -8.9% in 2d  -> +30% option

The user exits based on UNDERLYING price action (level retest, % drop, intraday
read) — NOT on the option's P&L. The closest algorithmic proxy we can wire in:
'close when the underlying has dropped X% from entry close.'

This sweep tests underlying-drop targets in {-2, -3, -5, -7, -10}% combined
with the v3 entry signal and a tight time stop. Engineering change: added
`underlying_target_pct` to ExitCfg and patched simulate_put to honor it.

Ship gate: PF >= 1.5  AND  CAGR > 0  AND  n >= 10

Run:
    python research/puts_v3_underlying_target_sweep.py
Outputs:
    research/out/puts_v3_underlying_target_sweep.txt
    research/out/puts_v3_underlying_target_sweep.png
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
    ExitCfg, _log_channel_z, _fmt_with_n,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, HEADER, STARTING_CAPITAL,
)
from research.volatile_universe_puts_sweep import simulate_put  # noqa: E402
from research.puts_v3_level_sweep import _signals_v3  # noqa: E402

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

    sig_by_day = _signals_v3(series, z_by_tk)
    n_sig = sum(len(v) for v in sig_by_day.values())
    print(f"  {n_sig} v3 signals\n")

    base = ExitCfg(
        tp_pct=None, disaster_pct=None, disaster_usd=300.0,
        time_stop=10, sigma_target=0.0,
        iv_elevation=1.00, crush_td=30,
        otm_pct=5.0, dte=90,
    )

    configs = [
        # No underlying target (baseline at t=10)
        ("baseline t=10 (no underlying target)",  base),
        # Underlying drop targets — close when underlying down X%
        ("UND drop -2% + t=10",  replace(base, underlying_target_pct=2.0)),
        ("UND drop -3% + t=10",  replace(base, underlying_target_pct=3.0)),
        ("UND drop -5% + t=10",  replace(base, underlying_target_pct=5.0)),
        ("UND drop -7% + t=10",  replace(base, underlying_target_pct=7.0)),
        ("UND drop -10% + t=10", replace(base, underlying_target_pct=10.0)),
        # Combine with even tighter time stops
        ("UND drop -5% + t=5",   replace(base, underlying_target_pct=5.0, time_stop=5)),
        ("UND drop -5% + t=7",   replace(base, underlying_target_pct=5.0, time_stop=7)),
        ("UND drop -3% + t=5",   replace(base, underlying_target_pct=3.0, time_stop=5)),
        ("UND drop -3% + t=7",   replace(base, underlying_target_pct=3.0, time_stop=7)),
    ]

    EHEADER = HEADER + "  | exit-reason mix"
    lines: list[str] = []
    lines.append("v3 PUTS + UNDERLYING-DROP-TARGET EXIT on 5y volatile universe")
    lines.append("")
    lines.append(f"  signal  : v3 level-based (z>=+1, Pattern A or B, cooldown 5d)")
    lines.append(f"  pinned  : no TP, sigma=0, $300 cap, IV 1.00/30, dte 90, otm 5%")
    lines.append(f"  vary    : underlying-drop target in {{2,3,5,7,10}}% +/- time stop")
    lines.append(f"  univ    : volatile basket ({len(series)} tickers, 5y)")
    lines.append(f"  signals : {n_sig} raw")
    lines.append("")
    lines.append("Reference - user's REAL 2025 winning puts:")
    lines.append("  GOOG  -4.7% in 4d  -> +35% option")
    lines.append("  APLD -18.9% in 7d  -> +34% option")
    lines.append("  IONQ  -8.9% in 2d  -> +30% option")
    lines.append("  median: ~-5% underlying drop / ~2-5d hold")
    lines.append("")
    lines.append(EHEADER)

    results: list[tuple[str, dict]] = []
    for tag, cfg in configs:
        st = simulate_put(series, z_by_tk, sig_by_day, cfg)
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
        lines.append(f"           PF {pf:>5.2f}  CAGR {cagr*100:>+6.1f}%  "
                     f"n {n:>4d}  maxDD {st.get('max_dd',0)*100:>+4.0f}%")
    lines.append("")
    lines.append(f"[VERDICT] {'SHIP-WORTHY config found' if any_passed else 'NO config passes ship gate'}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    for tag, st in results:
        ec = st.get("equity_curve") or []
        if not ec: continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.4,
                label=f"{tag}  (CAGR {st['cagr']*100:+.1f}% | "
                      f"PF {st.get('profit_factor', 0):.2f} | n={st['n']})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title("v3 puts + underlying-drop-target exit (5y volatile)")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "puts_v3_underlying_target_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "puts_v3_underlying_target_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
