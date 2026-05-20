"""Per-trade drill-down on the v3 level-based put sim.

Re-runs the best v3 config (P0 sigma-revert), captures every executed trade
with full entry context, then segments winners vs losers to find the failure
modes.

Features attached to each trade:
  - z_bto, z_stc
  - 20d swing high, 50d swing high at BTO
  - broken_level (highest of those BELOW BTO close), pct_above_broken
  - pattern label: 'A' (extension) or 'B' (rejection)
  - theme (from volatile_universe.theme_of)
  - exit_reason (from sim)

Outputs:
    research/out/puts_v3_trade_drilldown.txt   (segmented tables + summaries)
    research/out/puts_v3_trade_drilldown.csv   (all trades, all features)
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date as date_type

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.volatile_universe import universe, theme_of  # noqa: E402
from research.oversold_call_exit_sweep import (  # noqa: E402
    ExitCfg, _log_channel_z,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series,
)
from research.volatile_universe_puts_sweep import simulate_put  # noqa: E402
from research.puts_v3_level_sweep import (  # noqa: E402
    _signals_v3, Z_MIN, PATTERN_A_BREAKOUT_PCT, PATTERN_B_REJECT_BAND,
    SH20_LOOKBACK, SH50_LOOKBACK,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def _enrich_trade(tr: dict, series: dict, z_by_tk: dict) -> dict:
    tk = tr["ticker"]
    s = series.get(tk)
    if s is None:
        return tr
    z = z_by_tk[tk]
    # entry_date / exit_date are stringified ISO from simulate_put
    ent_d = date_type.fromisoformat(tr["entry_date"])
    ext_d = date_type.fromisoformat(tr["exit_date"])
    i_ent = s.by_date.get(ent_d)
    i_ext = s.by_date.get(ext_d)
    if i_ent is None:
        return tr

    bto = float(s.close[i_ent])
    # 20d / 50d swing highs at entry
    lo20, hi20 = max(0, i_ent - SH20_LOOKBACK[0]), max(0, i_ent - SH20_LOOKBACK[1])
    sh20 = float(s.high[lo20:hi20].max()) if hi20 > lo20 else None
    lo50, hi50 = max(0, i_ent - SH50_LOOKBACK[0]), max(0, i_ent - SH50_LOOKBACK[1])
    sh50 = float(s.high[lo50:hi50].max()) if hi50 > lo50 else None
    broken = None
    for v in [sh20, sh50]:
        if v is not None and v < bto:
            broken = max(broken, v) if broken is not None else v
    pct_above = (bto / broken - 1.0) if broken else None

    pattern_a = broken is not None and bto >= broken * (1.0 + PATTERN_A_BREAKOUT_PCT)
    pattern_b = (sh20 is not None and sh20 > 0
                 and abs(bto / sh20 - 1.0) <= PATTERN_B_REJECT_BAND)
    if pattern_a and not pattern_b:
        pat = "A"
    elif pattern_b and not pattern_a:
        pat = "B"
    elif pattern_a and pattern_b:
        pat = "AB"
    else:
        pat = "?"

    out = dict(tr)
    out["bto_close"] = bto
    out["stc_close"] = float(s.close[i_ext]) if i_ext is not None else None
    out["z_bto"] = float(z[i_ent]) if not np.isnan(z[i_ent]) else None
    out["z_stc"] = (float(z[i_ext]) if i_ext is not None
                                    and not np.isnan(z[i_ext]) else None)
    out["sh20"] = sh20
    out["sh50"] = sh50
    out["broken"] = broken
    out["pct_above_broken"] = pct_above
    out["pattern"] = pat
    out["theme"] = theme_of(tk)
    return out


def _pct(num: int, denom: int) -> str:
    if denom == 0: return "0/0"
    return f"{num}/{denom} ({num/denom*100:.0f}%)"


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
    print(f"  {sum(len(v) for v in sig_by_day.values())} v3 signals\n")

    # Best v3 config (P0 sigma-revert)
    cfg = ExitCfg(
        tp_pct=None, disaster_pct=None, disaster_usd=300.0,
        time_stop=45, sigma_target=0.0,
        iv_elevation=1.00, crush_td=30,
        otm_pct=5.0, dte=90,
    )
    print("Running sim (P0 sigma-revert, best config)...")
    st = simulate_put(series, z_by_tk, sig_by_day, cfg)
    raw_trades = st.get("trades", [])
    print(f"  {len(raw_trades)} executed trades")

    enriched = [_enrich_trade(t, series, z_by_tk) for t in raw_trades]
    winners = [t for t in enriched if t["opt_ret"] > 0]
    losers = [t for t in enriched if t["opt_ret"] <= 0]
    print(f"  winners {len(winners)}  losers {len(losers)}\n")

    # ----------- per-trade segmentation -----------
    lines: list[str] = []
    lines.append("PUTS v3 DRILL-DOWN — every trade from the 5y sim (best config P0)")
    lines.append("")
    lines.append(f"  config : no TP, sigma=0, $300 disaster, t=45, IV 1.00/30")
    lines.append(f"  n      : {len(enriched)} trades   "
                 f"W {len(winners)}   L {len(losers)}   "
                 f"WR {len(winners)/max(len(enriched),1)*100:.1f}%")
    pf = st.get("profit_factor", 0) or 0
    lines.append(f"  PF     : {pf:.2f}    CAGR {st.get('cagr',0)*100:+.1f}%    "
                 f"maxDD {st.get('max_dd',0)*100:.0f}%    "
                 f"Sharpe {st.get('sharpe',0):.2f}")
    lines.append("")

    # --- by pattern ---
    by_pat = defaultdict(lambda: {"n": 0, "w": 0, "ret_sum": 0.0})
    for t in enriched:
        b = by_pat[t.get("pattern", "?")]
        b["n"] += 1
        if t["opt_ret"] > 0: b["w"] += 1
        b["ret_sum"] += t["opt_ret"]
    lines.append("[BY ENTRY PATTERN]")
    lines.append(f"  {'pat':<5} {'n':>5} {'wins':>5} {'WR%':>6} {'mean_ret%':>10}")
    for pat in ("A", "B", "AB", "?"):
        b = by_pat.get(pat, {"n": 0, "w": 0, "ret_sum": 0.0})
        if b["n"] == 0: continue
        lines.append(f"  {pat:<5} {b['n']:>5d} {b['w']:>5d} "
                     f"{b['w']/b['n']*100:>5.1f}% {b['ret_sum']/b['n']*100:>+9.1f}%")
    lines.append("")

    # --- by z bucket ---
    z_buckets = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 5.0)]
    lines.append("[BY z AT BTO]")
    lines.append(f"  {'z range':<14} {'n':>5} {'wins':>5} {'WR%':>6} {'mean_ret%':>10}")
    for lo, hi in z_buckets:
        ts = [t for t in enriched
              if t.get("z_bto") is not None and lo <= t["z_bto"] < hi]
        if not ts: continue
        wins = sum(1 for t in ts if t["opt_ret"] > 0)
        mr = sum(t["opt_ret"] for t in ts) / len(ts) * 100
        lines.append(f"  z in [{lo:.1f},{hi:.1f}) {len(ts):>5d} {wins:>5d} "
                     f"{wins/len(ts)*100:>5.1f}% {mr:>+9.1f}%")
    lines.append("")

    # --- by %-above-broken bucket ---
    pct_buckets = [(-0.05, 0.0), (0.0, 0.05), (0.05, 0.10),
                   (0.10, 0.20), (0.20, 0.50)]
    lines.append("[BY % BTO ABOVE BROKEN LEVEL]")
    lines.append(f"  {'pct range':<14} {'n':>5} {'wins':>5} {'WR%':>6} {'mean_ret%':>10}")
    for lo, hi in pct_buckets:
        ts = [t for t in enriched
              if t.get("pct_above_broken") is not None
              and lo <= t["pct_above_broken"] < hi]
        if not ts: continue
        wins = sum(1 for t in ts if t["opt_ret"] > 0)
        mr = sum(t["opt_ret"] for t in ts) / len(ts) * 100
        lines.append(f"  [{lo*100:+4.0f},{hi*100:+4.0f}%)  {len(ts):>5d} {wins:>5d} "
                     f"{wins/len(ts)*100:>5.1f}% {mr:>+9.1f}%")
    pat_b_only = [t for t in enriched if t.get("pattern") == "B"]
    if pat_b_only:
        wins = sum(1 for t in pat_b_only if t["opt_ret"] > 0)
        mr = sum(t["opt_ret"] for t in pat_b_only) / len(pat_b_only) * 100
        lines.append(f"  Pattern B only  {len(pat_b_only):>5d} {wins:>5d} "
                     f"{wins/len(pat_b_only)*100:>5.1f}% {mr:>+9.1f}%")
    lines.append("")

    # --- by exit reason ---
    by_reason = defaultdict(lambda: {"n": 0, "w": 0, "ret_sum": 0.0})
    for t in enriched:
        b = by_reason[t.get("exit_reason", "?")]
        b["n"] += 1
        if t["opt_ret"] > 0: b["w"] += 1
        b["ret_sum"] += t["opt_ret"]
    lines.append("[BY EXIT REASON]")
    lines.append(f"  {'reason':<14} {'n':>5} {'wins':>5} {'WR%':>6} {'mean_ret%':>10}")
    for r in sorted(by_reason, key=lambda k: -by_reason[k]["n"]):
        b = by_reason[r]
        lines.append(f"  {r:<14} {b['n']:>5d} {b['w']:>5d} "
                     f"{b['w']/b['n']*100:>5.1f}% {b['ret_sum']/b['n']*100:>+9.1f}%")
    lines.append("")

    # --- by theme ---
    by_theme = defaultdict(lambda: {"n": 0, "w": 0, "ret_sum": 0.0})
    for t in enriched:
        b = by_theme[t.get("theme", "?")]
        b["n"] += 1
        if t["opt_ret"] > 0: b["w"] += 1
        b["ret_sum"] += t["opt_ret"]
    lines.append("[BY THEME]")
    lines.append(f"  {'theme':<18} {'n':>5} {'wins':>5} {'WR%':>6} {'mean_ret%':>10}")
    for th in sorted(by_theme, key=lambda k: -by_theme[k]["n"]):
        b = by_theme[th]
        lines.append(f"  {th:<18} {b['n']:>5d} {b['w']:>5d} "
                     f"{b['w']/b['n']*100:>5.1f}% {b['ret_sum']/b['n']*100:>+9.1f}%")
    lines.append("")

    # --- top 10 worst trades (anatomy of the worst losses) ---
    losers_sorted = sorted(losers, key=lambda t: t["opt_ret"])[:10]
    lines.append("[10 WORST LOSSES — anatomy]")
    lines.append(f"  {'tkr':<6} {'entry':<11} {'exit':<11} "
                 f"{'ret%':>7} {'hold':>4} {'z_bto':>6} {'pat':>3} "
                 f"{'+broken%':>9} {'reason':<14} {'theme':<14}")
    for t in losers_sorted:
        lines.append(
            f"  {t['ticker']:<6} {t['entry_date']:<11} {t['exit_date']:<11} "
            f"{t['opt_ret']*100:>+6.1f}% {t['bars_held']:>3d}d "
            f"{t.get('z_bto', 0) or 0:>+5.2f} "
            f"{t.get('pattern','?'):>3} "
            f"{(t.get('pct_above_broken') or 0)*100:>+8.1f}% "
            f"{t.get('exit_reason','?'):<14} {t.get('theme','?'):<14}"
        )
    lines.append("")

    # --- top 10 best trades ---
    winners_sorted = sorted(winners, key=lambda t: -t["opt_ret"])[:10]
    lines.append("[10 BEST WINS — anatomy]")
    lines.append(f"  {'tkr':<6} {'entry':<11} {'exit':<11} "
                 f"{'ret%':>7} {'hold':>4} {'z_bto':>6} {'pat':>3} "
                 f"{'+broken%':>9} {'reason':<14} {'theme':<14}")
    for t in winners_sorted:
        lines.append(
            f"  {t['ticker']:<6} {t['entry_date']:<11} {t['exit_date']:<11} "
            f"{t['opt_ret']*100:>+6.1f}% {t['bars_held']:>3d}d "
            f"{t.get('z_bto', 0) or 0:>+5.2f} "
            f"{t.get('pattern','?'):>3} "
            f"{(t.get('pct_above_broken') or 0)*100:>+8.1f}% "
            f"{t.get('exit_reason','?'):<14} {t.get('theme','?'):<14}"
        )
    lines.append("")

    # --- per-ticker stats (where do we lose the most?) ---
    by_tk = defaultdict(lambda: {"n": 0, "w": 0, "ret_sum": 0.0, "dollar_pnl": 0.0})
    for t in enriched:
        b = by_tk[t["ticker"]]
        b["n"] += 1
        if t["opt_ret"] > 0: b["w"] += 1
        b["ret_sum"] += t["opt_ret"]
        b["dollar_pnl"] += t.get("cost", 0) * t["opt_ret"]
    lines.append("[PER-TICKER stats (top 15 by # trades)]")
    lines.append(f"  {'tkr':<7} {'n':>3} {'WR%':>6} {'mean_ret%':>10} {'sum$pnl':>10}")
    for tk in sorted(by_tk, key=lambda k: -by_tk[k]["n"])[:15]:
        b = by_tk[tk]
        lines.append(f"  {tk:<7} {b['n']:>3d} {b['w']/b['n']*100:>5.1f}% "
                     f"{b['ret_sum']/b['n']*100:>+9.1f}% "
                     f"{b['dollar_pnl']:>+9.0f}")
    lines.append("")

    # --- hold-time profile ---
    lines.append("[HOLD-TIME PROFILE]")
    lines.append(f"  {'hold range':<12} {'n':>5} {'wins':>5} {'WR%':>6} {'mean_ret%':>10}")
    for lo, hi in [(0, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 45)]:
        ts = [t for t in enriched if lo <= t["bars_held"] <= hi]
        if not ts: continue
        wins = sum(1 for t in ts if t["opt_ret"] > 0)
        mr = sum(t["opt_ret"] for t in ts) / len(ts) * 100
        lines.append(f"  {lo:>3d}-{hi:<3d} bars  {len(ts):>5d} {wins:>5d} "
                     f"{wins/len(ts)*100:>5.1f}% {mr:>+9.1f}%")
    lines.append("")

    # --- interpretation block ---
    lines.append("=" * 72)
    lines.append("[INTERPRETATION]")
    a_count = by_pat.get("A", {"n": 0, "w": 0})["n"]
    a_wins = by_pat.get("A", {"n": 0, "w": 0})["w"]
    b_count = by_pat.get("B", {"n": 0, "w": 0})["n"]
    b_wins = by_pat.get("B", {"n": 0, "w": 0})["w"]
    lines.append(f"  Pattern A (extension):  WR {a_wins}/{a_count} "
                 f"= {a_wins/max(a_count,1)*100:.1f}%")
    lines.append(f"  Pattern B (rejection):  WR {b_wins}/{b_count} "
                 f"= {b_wins/max(b_count,1)*100:.1f}%")
    lines.append(f"  Personal book Pattern A: WR 100%   (perfect)")
    lines.append(f"  Personal book Pattern B: WR 67%    (mixed)")
    lines.append(f"  Forward-test Pattern A WR drop vs personal: "
                 f"{a_wins/max(a_count,1)*100 - 100:+.1f}pts")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    out_txt = os.path.join(OUT_DIR, "puts_v3_trade_drilldown.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

    # CSV
    out_csv = os.path.join(OUT_DIR, "puts_v3_trade_drilldown.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ticker", "theme", "entry_date", "exit_date", "bars_held",
            "opt_ret", "exit_reason", "cost", "bto_close", "stc_close",
            "z_bto", "z_stc", "sh20", "sh50", "broken_level",
            "pct_above_broken", "pattern",
        ])
        for t in enriched:
            w.writerow([
                t.get("ticker"), t.get("theme"), t.get("entry_date"),
                t.get("exit_date"), t.get("bars_held"),
                round(t.get("opt_ret", 0), 4), t.get("exit_reason"),
                round(t.get("cost", 0), 2),
                round(t.get("bto_close", 0), 2) if t.get("bto_close") else "",
                round(t.get("stc_close", 0), 2) if t.get("stc_close") else "",
                round(t["z_bto"], 3) if t.get("z_bto") is not None else "",
                round(t["z_stc"], 3) if t.get("z_stc") is not None else "",
                round(t["sh20"], 2) if t.get("sh20") else "",
                round(t["sh50"], 2) if t.get("sh50") else "",
                round(t["broken"], 2) if t.get("broken") else "",
                round(t["pct_above_broken"], 4) if t.get("pct_above_broken") is not None else "",
                t.get("pattern"),
            ])

    print(f"\n  -> {out_txt}")
    print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()
