"""Threshold sensitivity sweep on the user's 20 paired 2025 put trades.

`personal_puts_pattern_analysis.py` surfaced that the user's puts are
parabolic-exhaustion shorts, not symmetric z-band reversions:
  WINNERS' median 20d return: +37.7%   LOSERS': +12.5%
  WINNERS' median dist_sma20: +17.6%   LOSERS':  +6.4%
  WINNERS' median RSI(14):    78.5     LOSERS':  70.7

This script sweeps thresholds on those discriminating features to find
the cleanest WIN-vs-LOSS separator. Output:
  - per single-feature threshold: WR / wins-kept / losers-excluded
  - top pairwise AND filters
  - "best practical" filter that keeps >= 75% of winners while excluding
    >= 75% of losers

OVERFITTING CAVEAT — flagged loudly because this matters:
  n=20 trades is a TINY sample. Any filter that maximally separates
  20 trades is overfit by construction. The output here is a HYPOTHESIS
  for what filter to test next on the 5y volatile universe — not a
  filter to ship directly.

Run:
    python research/puts_threshold_sensitivity.py
Outputs:
    research/out/puts_threshold_sensitivity.txt
"""
from __future__ import annotations

import math
import os
import sys
from itertools import combinations
from statistics import median

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from research.personal_trade_audit import load_rows, pair_options  # noqa: E402
from research.personal_puts_pattern_analysis import (  # noqa: E402
    _compute_features, PutFeatures,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Single-feature thresholds to test. (name, getter, [(threshold, direction)])
# direction: ">=" means "keep entry if feature >= threshold"
SINGLE_SWEEPS = [
    ("ret_20d",      lambda f: f.ret_20d,      [10, 15, 20, 25, 30, 35]),
    ("ret_10d",      lambda f: f.ret_10d,      [0, 5, 10, 15, 20, 25]),
    ("ret_5d",       lambda f: f.ret_5d,       [0, 5, 10, 15, 20]),
    ("rsi_14",       lambda f: f.rsi_14,       [60, 65, 70, 75, 80]),
    ("stoch_k",      lambda f: f.stoch_k,      [50, 70, 85, 95, 100]),
    ("dist_sma20",   lambda f: f.dist_sma20,   [0, 5, 10, 15, 20]),
    ("dist_sma50",   lambda f: f.dist_sma50,   [0, 10, 20, 30, 40]),
    ("rv_20",        lambda f: f.rv_20,        [30, 40, 50, 60, 70]),
    ("vol_spike",    lambda f: f.vol_spike,    [0.8, 1.0, 1.2, 1.5, 2.0]),
    ("z_log",        lambda f: f.z_log,        [0, 0.5, 1.0, 1.5, 2.0]),
]


def _apply_threshold(features: list[PutFeatures], getter, thr: float):
    """Returns (kept, dropped) lists where kept passes feature >= thr."""
    kept, dropped = [], []
    for f in features:
        v = getter(f)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            dropped.append(f)
            continue
        if v >= thr:
            kept.append(f)
        else:
            dropped.append(f)
    return kept, dropped


def _summary(group: list[PutFeatures]):
    n = len(group)
    wins = sum(1 for f in group if f.is_win)
    losses = n - wins
    wr = (wins / n * 100) if n else 0.0
    total_pl = sum(f.t.gross_pl for f in group)
    return n, wins, losses, wr, total_pl


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows(os.path.join(ROOT, "reference", "DownloadTxnHistory.csv"))
    puts = [t for t in pair_options(rows) if t.right == "PUT"]
    print(f"Loaded {len(puts)} paired PUT trades.")
    tickers = sorted({t.ticker for t in puts})
    bars = fetch_bars_bulk(tickers, period="5y")

    features: list[PutFeatures] = []
    for t in sorted(puts, key=lambda x: x.entry_date):
        df = bars.get(t.ticker)
        if df is None or df.empty:
            continue
        f = _compute_features(t, df)
        if f is None:
            continue
        features.append(f)
    print(f"Got features for {len(features)} trades.\n")

    total_n, total_w, total_l, total_wr, total_pl = _summary(features)
    out: list[str] = []
    out.append("PUTS THRESHOLD SENSITIVITY SWEEP")
    out.append(f"  baseline: {total_n} trades  ({total_w}W / {total_l}L, WR {total_wr:.1f}%, "
               f"net ${total_pl:+,.0f})")
    out.append("  OVERFITTING CAVEAT: n=20 is tiny. Output is a HYPOTHESIS for")
    out.append("  what to backtest on the 5y volatile universe — NOT a ship-ready filter.")
    out.append("")

    # ---- single-feature sweep ----
    out.append("[SINGLE-FEATURE THRESHOLDS, KEEP IF feature >= thr]")
    out.append(f"  {'feature':>12s} {'thr':>7s}  {'kept':>6s} {'WR%':>6s} {'wins':>5s}/"
               f"{'orig':>4s}  {'losses':>6s}/{'orig':>4s}  {'$PL':>10s}")
    rankings: list[tuple] = []
    for name, getter, thrs in SINGLE_SWEEPS:
        for thr in thrs:
            kept, dropped = _apply_threshold(features, getter, thr)
            kn, kw, kl, kwr, kpl = _summary(kept)
            dn, dw, dl, _, _ = _summary(dropped)
            if kn < 3:
                continue
            # Score: kept-WR * fraction-of-winners-kept * fraction-of-losers-excluded
            wins_kept = (kw / total_w) if total_w else 0
            losers_excluded = (dl / total_l) if total_l else 0
            score = (kwr / 100.0) * wins_kept * losers_excluded
            rankings.append((score, name, thr, kn, kw, kl, kwr, kpl,
                             wins_kept, losers_excluded))
            out.append(
                f"  {name:>12s} >={thr:>5g}  {kn:>6d} {kwr:>5.1f}% "
                f"{kw:>4d}/{total_w:>3d}  {kl:>5d}/{total_l:>3d}  "
                f"${kpl:>+9,.0f}"
            )
    out.append("")

    # ---- top single-feature filters by composite score ----
    rankings.sort(key=lambda x: -x[0])
    out.append("[TOP 10 SINGLE-FEATURE FILTERS (by score=WR x %-wins-kept x %-losers-excluded)]")
    out.append(f"  {'feature':>12s} {'thr':>7s}  {'kept':>6s} {'WR%':>6s}  "
               f"{'%wins-kept':>11s} {'%losers-cut':>11s}  {'score':>7s}")
    for r in rankings[:10]:
        score, name, thr, kn, kw, kl, kwr, kpl, wk, le = r
        out.append(
            f"  {name:>12s} >={thr:>5g}  {kn:>6d} {kwr:>5.1f}%  "
            f"{wk * 100:>10.0f}% {le * 100:>10.0f}%  {score:>7.3f}"
        )
    out.append("")

    # ---- pairwise AND filters ----
    out.append("[TOP 15 PAIRWISE AND-FILTERS (best single per feature, then combined)]")
    out.append("  Goal: keep >=75% of original wins AND exclude >=75% of original losses.")
    # Use the top single per feature.
    by_feature: dict = {}
    for r in rankings:
        name = r[1]
        if name not in by_feature:
            by_feature[name] = r
    best_singles = list(by_feature.values())[:8]  # top 8 unique features

    pairs: list[tuple] = []
    for r1, r2 in combinations(best_singles, 2):
        name1, thr1, _g1 = r1[1], r1[2], None
        name2, thr2, _g2 = r2[1], r2[2], None
        getter1 = next(g for n, g, _ in SINGLE_SWEEPS if n == name1)
        getter2 = next(g for n, g, _ in SINGLE_SWEEPS if n == name2)
        kept = []
        for f in features:
            v1, v2 = getter1(f), getter2(f)
            if v1 is None or v2 is None: continue
            if isinstance(v1, float) and math.isnan(v1): continue
            if isinstance(v2, float) and math.isnan(v2): continue
            if v1 >= thr1 and v2 >= thr2:
                kept.append(f)
        kn, kw, kl, kwr, kpl = _summary(kept)
        if kn < 3:
            continue
        wk = (kw / total_w) if total_w else 0
        le = ((total_l - kl) / total_l) if total_l else 0
        score = (kwr / 100.0) * wk * le
        pairs.append((score, name1, thr1, name2, thr2, kn, kw, kl, kwr, kpl, wk, le))

    pairs.sort(key=lambda x: -x[0])
    out.append(f"  {'filter':>50s}  {'kept':>5s} {'WR%':>6s}  {'%w-kept':>8s} {'%l-cut':>7s}  {'score':>7s}")
    for p in pairs[:15]:
        score, n1, t1, n2, t2, kn, kw, kl, kwr, kpl, wk, le = p
        label = f"{n1}>={t1:g} AND {n2}>={t2:g}"
        out.append(
            f"  {label:>50s}  {kn:>5d} {kwr:>5.1f}%  "
            f"{wk * 100:>7.0f}% {le * 100:>6.0f}%  {score:>7.3f}"
        )
    out.append("")

    # ---- "best practical" ----
    practical = [p for p in pairs
                 if p[10] >= 0.75 and p[11] >= 0.75 and p[8] >= 80.0]
    out.append("[BEST PRACTICAL FILTERS]  (>=75% wins kept, >=75% losers cut, >=80% WR)")
    if not practical:
        out.append("  (none — relax constraints below)")
        lo = [p for p in pairs if p[10] >= 0.65 and p[11] >= 0.65 and p[8] >= 75.0][:5]
        out.append("  Falling back: >=65% wins kept, >=65% losers cut, >=75% WR:")
        for p in lo:
            label = f"{p[1]}>={p[2]:g} AND {p[3]}>={p[4]:g}"
            out.append(f"    {label:>50s}  kept={p[5]} WR={p[8]:.0f}% "
                       f"wk={p[10] * 100:.0f}% lc={p[11] * 100:.0f}% PL=${p[9]:+,.0f}")
    else:
        for p in practical[:5]:
            label = f"{p[1]}>={p[2]:g} AND {p[3]}>={p[4]:g}"
            out.append(f"    {label:>50s}  kept={p[5]} WR={p[8]:.0f}% "
                       f"wk={p[10] * 100:.0f}% lc={p[11] * 100:.0f}% PL=${p[9]:+,.0f}")
    out.append("")

    # ---- final hypothesis ----
    out.append("[HYPOTHESIS FOR FORWARD BACKTEST]")
    out.append("  Take the top pairwise filter above and test it on the full 5y")
    out.append("  volatile universe via a new puts_parabolic_exhaustion_sweep.py.")
    out.append("  Ship gate: PF >= 1.5 AND CAGR > 0 on the 5y sim.")

    text = "\n".join(out)
    print(text)
    out_txt = os.path.join(OUT_DIR, "puts_threshold_sensitivity.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  -> {out_txt}")


if __name__ == "__main__":
    main()
