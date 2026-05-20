"""Forward test of the level-based put entry rule (v3) on the 5y volatile
universe. Derived from the personal-trade diagnostic on 20 paired puts.

ENTRY RULE v3:
  Each bar, signal fires when ALL of:
    1. z >= +1.0 sigma in 252d LOG channel
    2. ADV >= $50M
    3. Tier (same-day breadth = count of z>=+1.0 crossings) in PRIME/OK/PANIC
    4. One of:
       A) BTO close >= 1% above a 20d or 50d swing high (extension breakout)
       B) BTO close within +/-3% of the 20d swing high (test/rejection)
  Cooldown: 5 bars per ticker (prevents same setup re-firing).

Pinned exits to test (4 variants, R6-mirror put exits + composite TP):

  P0 SIGMA-REVERT     no TP, sigma=0, $300 disaster, t=45  (mirror of calls R6)
  P1 TP25 + SIGMA     +25% TP, sigma=0, $300 disaster, t=45
  P2 TP25 + TIME5     +25% TP, no sigma, $300 disaster, t=5  (fast scalp)
  P3 TP25 + SIGMA + T10  +25% TP, sigma=0, $300 disaster, t=10

Ship gate (5y volatile, 102-name universe):
  PF >= 1.5  AND  CAGR > 0  AND  n >= 10

Compared against baseline volatile_universe_puts_sweep PF 0.82 (z>=+2.0).

Run:
    python research/puts_v3_level_sweep.py
Outputs:
    research/out/puts_v3_level_sweep.txt
    research/out/puts_v3_level_sweep.png
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from dataclasses import replace
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
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, HEADER, STARTING_CAPITAL, MIN_ADV_M,
)
from research.volatile_universe_puts_sweep import (  # noqa: E402
    Sig, simulate_put,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# v3 entry parameters (from personal_puts_level_diagnostic.py)
Z_MIN = 1.0
PATTERN_A_BREAKOUT_PCT = 0.01    # >= 1% above broken swing high
PATTERN_B_REJECT_BAND = 0.03     # +/-3% of sh_20
SH20_LOOKBACK = (25, 3)          # rolling max(high) over [i-25 .. i-3]
SH50_LOOKBACK = (55, 3)
COOLDOWN_BARS = 5                # don't re-fire on same ticker for N bars


def _signals_v3(series: dict[str, Series], z_by_tk: dict[str, np.ndarray]) -> dict:
    """Generate v3 put-entry signals across the universe.

    Returns {date: list[Sig]} with same Sig shape as volatile_universe_puts_sweep.
    """
    breadth_by_day: Counter = Counter()
    eligible: list[Sig] = []

    for tk, s in series.items():
        z = z_by_tk[tk]
        last_fire_idx = -10_000
        high = s.high if hasattr(s, "high") else None
        if high is None:
            # Series doesn't carry high — abort
            continue

        for i in range(1, len(s.close)):
            zi = z[i]
            if np.isnan(zi):
                continue

            # Breadth: count of ALL crossings UP into z >= +1.0 today.
            if zi >= 1.0 and not np.isnan(z[i - 1]) and z[i - 1] < 1.0:
                breadth_by_day[s.dates[i]] += 1

            # Eligibility checks
            if zi < Z_MIN:
                continue
            if i - last_fire_idx < COOLDOWN_BARS:
                continue
            if np.isnan(s.rv[i]) or not (s.adv_m[i] >= MIN_ADV_M):
                continue

            bto = float(s.close[i])
            # 20d swing high over [i-25 .. i-3]
            lo20, hi20 = max(0, i - SH20_LOOKBACK[0]), max(0, i - SH20_LOOKBACK[1])
            sh20 = float(high[lo20:hi20].max()) if hi20 > lo20 else None
            # 50d swing high
            lo50, hi50 = max(0, i - SH50_LOOKBACK[0]), max(0, i - SH50_LOOKBACK[1])
            sh50 = float(high[lo50:hi50].max()) if hi50 > lo50 else None

            # Broken level = highest of {sh20, sh50} that's BELOW bto
            broken = None
            for v in [sh20, sh50]:
                if v is not None and v < bto:
                    broken = max(broken, v) if broken is not None else v

            # Pattern A: BTO >= 1% above broken level
            pattern_a = broken is not None and bto >= broken * (1.0 + PATTERN_A_BREAKOUT_PCT)
            # Pattern B: BTO within +/-3% of sh_20
            pattern_b = (sh20 is not None and sh20 > 0
                         and abs(bto / sh20 - 1.0) <= PATTERN_B_REJECT_BAND)

            if not (pattern_a or pattern_b):
                continue

            eligible.append(Sig(tk=tk, i=i, z=float(zi), date=s.dates[i],
                                adv=float(s.adv_m[i])))
            last_fire_idx = i

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

    sig_by_day = _signals_v3(series, z_by_tk)
    n_sig = sum(len(v) for v in sig_by_day.values())
    print(f"  {n_sig} PUT entry signals (v3: z>=+1 AND (Pattern A OR B), "
          f"tier-gated, cooldown {COOLDOWN_BARS} bars)\n")

    # Per-theme + per-ticker tally
    sig_by_theme: dict[str, int] = defaultdict(int)
    tickers_with_sigs: dict[str, int] = defaultdict(int)
    for _, sigs in sig_by_day.items():
        for sg in sigs:
            sig_by_theme[theme_of(sg.tk)] += 1
            tickers_with_sigs[sg.tk] += 1

    # Pinned exit configs to test
    # Base: R6-style put exits (otm 5%, dte 90, iv 1.00/30, no buffer)
    base = ExitCfg(
        tp_pct=None, disaster_pct=None, disaster_usd=300.0,
        time_stop=45, sigma_target=0.0,
        iv_elevation=1.00, crush_td=30,
        otm_pct=5.0, dte=90,
    )
    configs = [
        ("P0 sigma-revert (R6-mirror, no TP, $300 cap, t=45)",
         base),
        ("P1 +TP25 + sigma-revert + $300 cap + t=45",
         replace(base, tp_pct=25.0)),
        ("P2 +TP25 + no-sigma + $300 cap + t=5  (fast scalp)",
         replace(base, tp_pct=25.0, sigma_target=None, time_stop=5)),
        ("P3 +TP25 + sigma-revert + $300 cap + t=10",
         replace(base, tp_pct=25.0, time_stop=10)),
        ("P4 +TP30 + sigma-revert + $300 cap + t=10",
         replace(base, tp_pct=30.0, time_stop=10)),
    ]

    EHEADER = HEADER + "  | exit-reason mix"
    lines: list[str] = []
    lines.append("PUT-LEG v3 (LEVEL-BASED ENTRY) on volatile universe — 5y forward test")
    lines.append("")
    lines.append(f"  ENTRY  : z >= +1.0 AND")
    lines.append(f"          A) BTO >= 1% above sh_20/sh_50 (extension)  OR")
    lines.append(f"          B) BTO within +/-3% of sh_20 (test/rejection)")
    lines.append(f"  filter : ADV >= ${MIN_ADV_M}M, tier in PRIME/OK/PANIC, cooldown {COOLDOWN_BARS}d")
    lines.append(f"  exits  : varied (see configs)")
    lines.append(f"  univ   : volatile thematic basket ({len(series)} tickers, 5y)")
    lines.append(f"  signal : {n_sig} raw, capital-constrained from there")
    lines.append("")
    lines.append(f"  Baseline (from volatile_universe_puts_sweep): z>=+2.0, PF 0.82 best.")
    lines.append(f"  Personal-trade diagnostic v3: 10/14 winners caught (71%), 2/6 losers (33%).")
    lines.append("")

    lines.append("[SIGNALS PER THEME]")
    for theme in sorted(sig_by_theme, key=lambda k: -sig_by_theme[k]):
        lines.append(f"  {theme:18s} signals={sig_by_theme[theme]:4d}")
    lines.append("")
    lines.append("[TOP 15 TICKERS WITH SIGNALS]")
    for tk in sorted(tickers_with_sigs, key=lambda k: -tickers_with_sigs[k])[:15]:
        lines.append(f"  {tk:7s} ({theme_of(tk):14s}) signals={tickers_with_sigs[tk]:3d}")
    lines.append("")

    lines.append("[CONFIGS ON VOLATILE UNIVERSE — PUT LEG v3]")
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
    lines.append(f"     CAGR {by_cagr[1]['cagr']*100:+.1f}%  PF {by_cagr[1].get('profit_factor', 0):.2f}  "
                 f"Sharpe {by_cagr[1].get('sharpe', 0):.2f}  maxDD {by_cagr[1].get('max_dd', 0)*100:.0f}%  "
                 f"n={by_cagr[1]['n']}")
    lines.append(f"  best by PF    : {by_pf[0]}")
    lines.append(f"     CAGR {by_pf[1]['cagr']*100:+.1f}%  PF {by_pf[1].get('profit_factor', 0):.2f}  "
                 f"Sharpe {by_pf[1].get('sharpe', 0):.2f}  maxDD {by_pf[1].get('max_dd', 0)*100:.0f}%  "
                 f"n={by_pf[1]['n']}")
    lines.append(f"  best by Sharpe: {by_sharpe[0]}")
    lines.append(f"     CAGR {by_sharpe[1]['cagr']*100:+.1f}%  PF {by_sharpe[1].get('profit_factor', 0):.2f}  "
                 f"Sharpe {by_sharpe[1].get('sharpe', 0):.2f}  maxDD {by_sharpe[1].get('max_dd', 0)*100:.0f}%  "
                 f"n={by_sharpe[1]['n']}")
    lines.append("")

    # SHIP GATE
    lines.append("[SHIP GATE]   PF >= 1.5  AND  CAGR > 0  AND  n >= 10")
    any_passed = False
    for tag, st in results:
        pf = st.get("profit_factor", 0) or 0
        cagr = st.get("cagr", 0)
        n = st.get("n", 0)
        passed = pf >= 1.5 and cagr > 0 and n >= 10
        any_passed = any_passed or passed
        verdict = "PASS" if passed else "fail"
        lines.append(f"  [{verdict}] {tag}")
        lines.append(f"          PF {pf:>5.2f}  CAGR {cagr*100:>+6.1f}%  n {n:>4d}")
    lines.append("")
    lines.append(f"[VERDICT] {'SHIP-WORTHY config found' if any_passed else 'NO config passes ship gate'}")
    lines.append("")

    # Compare to baseline put sweep
    lines.append("[COMPARE TO BASELINE z>=+2 puts sweep]")
    lines.append("  Previous best (volatile_universe_puts_sweep): PF 0.82, CAGR -5%")
    pf_v3 = by_pf[1].get('profit_factor', 0) or 0
    cagr_v3 = by_pf[1].get('cagr', 0)
    lines.append(f"  v3 (level-based) best:                       PF {pf_v3:.2f}, CAGR {cagr_v3*100:+.1f}%")
    delta_pf = pf_v3 - 0.82
    lines.append(f"  delta PF: {delta_pf:+.2f}")

    # Plot equity curves
    fig, ax = plt.subplots(figsize=(14, 7))
    for tag, st in results:
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.5,
                label=f"{tag}  (CAGR {st['cagr']*100:+.1f}% | "
                      f"PF {st.get('profit_factor', 0):.2f} | "
                      f"DD {st.get('max_dd', 0)*100:.0f}% | n={st['n']})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title("v3 LEVEL-BASED PUT ENTRY on volatile universe (5y forward test)")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "puts_v3_level_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "puts_v3_level_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
