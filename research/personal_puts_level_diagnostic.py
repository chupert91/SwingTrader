"""Per-trade level diagnostic on all 20 paired puts (14 winners + 6 losers).

For each trade, measure at BTO:
  - z-score in the 252d LOG channel (entry-signal candidate)
  - Whether price has just broken above a recent swing-high resistance
  - Levels: 20d swing high, 50d swing high, recent broken-resistance ladder,
    auto-anchored Fibonacci retracements (38.2 / 50 / 61.8)
At STC:
  - Distance from STC close to each level (%)
  - Exit-type classification: LEVEL_RETEST | FAST_TP | OTHER

Hypothesis under test:
  ENTRY  : z >= +1.0 sigma AND price >= 5% above a recently-broken swing high
  EXIT   : retest of nearest broken level (within +/-2-3%) OR option +25% TP

Ship gate (informal — this is diagnostic, not a backtest):
  >= 10/14 winners (71%) match the entry conditions
  >= 10/14 winners classified as {LEVEL_RETEST | FAST_TP}
  < 3/6  losers match the same entry conditions

If all three hold, the next step is a forward backtest on the 5y volatile
universe with these rules wired in.

Outputs:
    research/out/personal_puts_level_diagnostic.txt   (per-trade table + summary)
    research/out/personal_puts_level_diagnostic.csv   (per-trade rows)
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars  # noqa: E402
from research.personal_trade_audit import CSV_PATH, load_rows, pair_options, Trade  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
LOG_WINDOW = 252

# Retest bands. We report all three; the summary uses RETEST_TIGHT as the
# primary verdict to stay strict.
RETEST_TIGHT = 0.02   # +/-2%
RETEST_MED = 0.03     # +/-3%
RETEST_LOOSE = 0.05   # +/-5%

# Profit-target threshold: STC option return >= this = FAST_TP candidate.
FAST_TP_THRESH = 0.25


@dataclass
class TradeLevels:
    trade: Trade
    bto_close: float
    stc_close: float
    z_bto: float | None
    z_stc: float | None
    swing_high_20d: float | None   # 20d high excluding last 3 bars
    swing_high_50d: float | None   # 50d high excluding last 3 bars
    broken_level: float | None     # highest swing-high BELOW bto_close
    swing_low_60d: float | None    # pre-rally floor for fib anchor
    fib_382: float | None
    fib_500: float | None
    fib_618: float | None
    pct_above_broken_at_bto: float | None  # bto / broken_level - 1


def _log_channel_z(close: np.ndarray) -> np.ndarray:
    """Rolling 252-bar LOG-channel z per bar (mirrors ai_strategy)."""
    n = len(close)
    z = np.full(n, np.nan)
    if n < LOG_WINDOW:
        return z
    y_all = np.log(close)
    x = np.arange(LOG_WINDOW, dtype=float)
    x_mean = x.mean()
    xc = x - x_mean
    denom = float(np.sum(xc ** 2))
    for t in range(LOG_WINDOW - 1, n):
        chunk = y_all[t - LOG_WINDOW + 1: t + 1]
        y_mean = chunk.mean()
        slope = float(np.sum(xc * (chunk - y_mean)) / denom)
        intercept = y_mean - slope * x_mean
        fit_end = slope * (LOG_WINDOW - 1) + intercept
        sigma = float(np.std(chunk - (slope * x + intercept), ddof=1))
        if sigma > 0:
            z[t] = (y_all[t] - fit_end) / sigma
    return z


def _measure(trade: Trade, df: pd.DataFrame, z: np.ndarray) -> TradeLevels | None:
    ts = pd.to_datetime(df["timestamp"])
    entry_pos = int(ts.searchsorted(pd.Timestamp(trade.entry_date), side="left"))
    exit_pos = int(ts.searchsorted(pd.Timestamp(trade.exit_date), side="left"))
    if entry_pos >= len(df) or exit_pos >= len(df):
        return None

    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    bto_close = float(close[entry_pos])
    stc_close = float(close[exit_pos])

    # 20d swing high: max(high) over [entry-25 .. entry-3]
    lo_20 = max(0, entry_pos - 25)
    hi_20 = max(0, entry_pos - 3)
    sh_20 = float(high[lo_20:hi_20].max()) if hi_20 > lo_20 else None

    # 50d swing high: max(high) over [entry-55 .. entry-3]
    lo_50 = max(0, entry_pos - 55)
    hi_50 = max(0, entry_pos - 3)
    sh_50 = float(high[lo_50:hi_50].max()) if hi_50 > lo_50 else None

    # Most-recently-broken level = highest of the swing-highs that is STILL
    # BELOW bto_close. (If sh_20 already above bto, the rally hasn't "broken"
    # anything yet — skip it.)
    candidates = [v for v in [sh_20, sh_50] if v is not None and v < bto_close]
    broken = max(candidates) if candidates else None

    # Fib anchors: swing-low in [entry-90 .. entry-15] (the pre-rally zone)
    lo_60 = max(0, entry_pos - 90)
    hi_60 = max(0, entry_pos - 15)
    sl_60 = float(low[lo_60:hi_60].min()) if hi_60 > lo_60 else None

    # Standard retracements from leg [sl_60 .. bto_close]
    fib_382 = fib_500 = fib_618 = None
    if sl_60 is not None and bto_close > sl_60:
        leg = bto_close - sl_60
        fib_382 = bto_close - 0.382 * leg
        fib_500 = bto_close - 0.500 * leg
        fib_618 = bto_close - 0.618 * leg

    pct_above = None
    if broken is not None and broken > 0:
        pct_above = (bto_close / broken) - 1.0

    return TradeLevels(
        trade=trade, bto_close=bto_close, stc_close=stc_close,
        z_bto=float(z[entry_pos]) if not np.isnan(z[entry_pos]) else None,
        z_stc=float(z[exit_pos]) if not np.isnan(z[exit_pos]) else None,
        swing_high_20d=sh_20, swing_high_50d=sh_50, broken_level=broken,
        swing_low_60d=sl_60, fib_382=fib_382, fib_500=fib_500, fib_618=fib_618,
        pct_above_broken_at_bto=pct_above,
    )


def _retest_flag(stc: float, level: float | None, band: float) -> str:
    if level is None or level <= 0:
        return "-"
    dist = abs(stc / level - 1.0)
    return "Y" if dist <= band else "n"


def _classify_exit(tl: TradeLevels) -> str:
    """LEVEL_RETEST | FAST_TP | OTHER. Tight retest band, conservative TP gate."""
    stc = tl.stc_close
    levels = [
        ("broken", tl.broken_level),
        ("fib382", tl.fib_382),
        ("fib500", tl.fib_500),
        ("fib618", tl.fib_618),
    ]
    for _, lv in levels:
        if lv is not None and lv > 0 and abs(stc / lv - 1.0) <= RETEST_MED:
            return "LEVEL_RETEST"
    if tl.trade.ret_pct >= FAST_TP_THRESH:
        return "FAST_TP"
    return "OTHER"


REJECT_BAND = 0.03   # +/-3% from swing high counts as "testing/rejection"


def _classify_entry(tl: TradeLevels, z_min: float = 1.0, pct_min: float = 0.05) -> str:
    """Returns 'A' (extension above broken), 'B' (rejection at resistance),
    or 'n' (no match). Combined rule passes if A or B fires and z >= z_min."""
    if tl.z_bto is None or tl.z_bto < z_min:
        return "n"
    # Pattern A: extension above a broken swing high by >= pct_min
    pct_above = tl.pct_above_broken_at_bto or 0
    if tl.broken_level is not None and pct_above >= pct_min:
        return "A"
    # Pattern B: BTO within +/-REJECT_BAND of the 20d swing high
    sh = tl.swing_high_20d
    if sh is not None and sh > 0 and abs(tl.bto_close / sh - 1.0) <= REJECT_BAND:
        return "B"
    return "n"


def _entry_pass(tl: TradeLevels) -> bool:
    return _classify_entry(tl) in ("A", "B")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = load_rows(CSV_PATH)
    puts = [t for t in pair_options(rows) if t.right == "PUT"]
    print(f"paired puts: {len(puts)}")

    # Bars per ticker (3y window is plenty for 252d-before-entry + post-exit).
    tickers = sorted({t.ticker for t in puts})
    print(f"fetching {len(tickers)} tickers x 3y...")

    series: dict[str, pd.DataFrame] = {}
    z_by_tk: dict[str, np.ndarray] = {}
    for tk in tickers:
        df = fetch_bars(tk, period="3y")
        if df is None or df.empty:
            print(f"  no bars for {tk}")
            continue
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        series[tk] = df
        z_by_tk[tk] = _log_channel_z(df["close"].values.astype(float))
    print(f"  {len(series)} tickers loaded")

    measurements: list[TradeLevels] = []
    for t in puts:
        if t.ticker not in series:
            continue
        m = _measure(t, series[t.ticker], z_by_tk[t.ticker])
        if m is not None:
            measurements.append(m)
    print(f"  {len(measurements)} trades measured")

    winners = [m for m in measurements if m.trade.gross_pl > 0]
    losers = [m for m in measurements if m.trade.gross_pl <= 0]
    print(f"  winners: {len(winners)}   losers: {len(losers)}\n")

    # ---- Per-trade table ----
    lines: list[str] = []
    lines.append("PUTS LEVEL DIAGNOSTIC — all 20 paired puts")
    lines.append("")
    lines.append(f"Hypothesis tested (v2 — two entry patterns):")
    lines.append(f"  ENTRY  : z >= +1.0 sigma AND one of:")
    lines.append(f"           A) price >= 5% above a recently-broken swing high")
    lines.append(f"           B) price within +/-3% of the 20d swing high (rejection/test)")
    lines.append(f"  EXIT   : retest of nearest broken level (+/-3%) OR option +{FAST_TP_THRESH*100:.0f}% TP")
    lines.append("")
    lines.append("=== WINNERS (n={}) ===".format(len(winners)))
    lines.append("")
    header = (f"{'TKR':<5} {'BTO':<10} {'STC':<10} "
              f"{'ret%':>6} {'hold':>4} "
              f"{'BTO$':>8} {'STC$':>8} "
              f"{'z_BTO':>6} {'z_STC':>6} "
              f"{'sh20':>7} {'sh50':>7} {'broken':>7} {'+pct':>6} "
              f"{'fib38':>7} {'fib50':>7} {'fib62':>7} "
              f"{'ENT?':>5} {'retest':>10} {'class':<14}")
    lines.append(header)
    lines.append("-" * len(header))

    def _row(m: TradeLevels) -> str:
        t = m.trade
        ent_ok = _classify_entry(m)  # 'A' | 'B' | 'n'
        # Best retest distance (closest level)
        best_lv = None
        best_dist = 999.0
        for name, lv in [("broken", m.broken_level), ("fib38", m.fib_382),
                          ("fib50", m.fib_500), ("fib62", m.fib_618)]:
            if lv is None or lv <= 0:
                continue
            d = abs(m.stc_close / lv - 1.0)
            if d < best_dist:
                best_dist = d
                best_lv = name
        retest_txt = f"{best_lv or '-'}({best_dist*100:.1f}%)" if best_lv else "-"
        cls = _classify_exit(m)
        return (f"{t.ticker:<5} {str(t.entry_date):<10} {str(t.exit_date):<10} "
                f"{t.ret_pct*100:>+5.0f}% {t.days_held:>3d}d "
                f"{m.bto_close:>8.2f} {m.stc_close:>8.2f} "
                f"{m.z_bto if m.z_bto is not None else float('nan'):>+6.2f} "
                f"{m.z_stc if m.z_stc is not None else float('nan'):>+6.2f} "
                f"{m.swing_high_20d or 0:>7.2f} {m.swing_high_50d or 0:>7.2f} "
                f"{m.broken_level or 0:>7.2f} "
                f"{(m.pct_above_broken_at_bto or 0)*100:>+5.1f}% "
                f"{m.fib_382 or 0:>7.2f} {m.fib_500 or 0:>7.2f} {m.fib_618 or 0:>7.2f} "
                f"{ent_ok:>5} {retest_txt:>10} {cls:<14}")

    for m in sorted(winners, key=lambda x: -x.trade.ret_pct):
        lines.append(_row(m))
    lines.append("")
    lines.append("=== LOSERS (n={}) ===".format(len(losers)))
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    for m in sorted(losers, key=lambda x: x.trade.ret_pct):
        lines.append(_row(m))
    lines.append("")

    # ---- Summary ----
    def _summary(group: list[TradeLevels], label: str) -> list[str]:
        n = len(group)
        if n == 0:
            return [f"{label}: empty"]
        # Entry conditions
        zs = [m.z_bto for m in group if m.z_bto is not None]
        z_ge_1 = sum(1 for v in zs if v >= 1.0)
        z_ge_2 = sum(1 for v in zs if v >= 2.0)
        z_ge_25 = sum(1 for v in zs if v >= 2.5)
        # pct-above-broken
        pcts = [(m.pct_above_broken_at_bto or 0) for m in group if m.broken_level is not None]
        pct_ge_5 = sum(1 for v in pcts if v >= 0.05)
        pct_ge_10 = sum(1 for v in pcts if v >= 0.10)
        entry_a = sum(1 for m in group if _classify_entry(m) == "A")
        entry_b = sum(1 for m in group if _classify_entry(m) == "B")
        entry_pass = entry_a + entry_b
        # Exit classifications
        cls_counts = Counter(_classify_exit(m) for m in group)
        # Best retest level (tight band)
        retest_broken = sum(1 for m in group
                            if m.broken_level and abs(m.stc_close / m.broken_level - 1) <= RETEST_TIGHT)
        retest_fib382 = sum(1 for m in group
                            if m.fib_382 and abs(m.stc_close / m.fib_382 - 1) <= RETEST_TIGHT)
        retest_fib500 = sum(1 for m in group
                            if m.fib_500 and abs(m.stc_close / m.fib_500 - 1) <= RETEST_TIGHT)
        retest_fib618 = sum(1 for m in group
                            if m.fib_618 and abs(m.stc_close / m.fib_618 - 1) <= RETEST_TIGHT)

        def _pct(num: int) -> str:
            return f"{num}/{n} ({num/n*100:.0f}%)"

        out = [f"--- {label} (n={n}) ---"]
        out.append(f"  z @ BTO  >=+1.0      : {_pct(z_ge_1)}")
        out.append(f"  z @ BTO  >=+2.0      : {_pct(z_ge_2)}")
        out.append(f"  z @ BTO  >=+2.5      : {_pct(z_ge_25)}")
        out.append(f"  BTO >= +5% above broken : {_pct(pct_ge_5)}")
        out.append(f"  BTO >= +10% above broken: {_pct(pct_ge_10)}")
        out.append(f"  ENTRY RULE pass total : {_pct(entry_pass)}")
        out.append(f"    Pattern A (z>=1 AND >=5% above broken)        : {_pct(entry_a)}")
        out.append(f"    Pattern B (z>=1 AND BTO within +/-3% of sh20) : {_pct(entry_b)}")
        out.append(f"")
        out.append(f"  Exit classification:")
        for k in ("LEVEL_RETEST", "FAST_TP", "OTHER"):
            out.append(f"    {k:<14} : {_pct(cls_counts.get(k, 0))}")
        out.append(f"")
        out.append(f"  STC within +/-{RETEST_TIGHT*100:.0f}% of:")
        out.append(f"    broken level : {_pct(retest_broken)}")
        out.append(f"    fib 38.2%    : {_pct(retest_fib382)}")
        out.append(f"    fib 50%      : {_pct(retest_fib500)}")
        out.append(f"    fib 61.8%    : {_pct(retest_fib618)}")
        return out

    lines.append("=" * 80)
    lines.extend(_summary(winners, "WINNERS"))
    lines.append("")
    lines.extend(_summary(losers, "LOSERS"))
    lines.append("")

    # Verdict
    n_win = len(winners); n_los = len(losers)
    entry_win = sum(1 for m in winners if _entry_pass(m))
    cls_win = Counter(_classify_exit(m) for m in winners)
    catch_win = cls_win.get("LEVEL_RETEST", 0) + cls_win.get("FAST_TP", 0)
    entry_los = sum(1 for m in losers if _entry_pass(m))
    cls_los = Counter(_classify_exit(m) for m in losers)
    catch_los = cls_los.get("LEVEL_RETEST", 0) + cls_los.get("FAST_TP", 0)

    lines.append("=" * 80)
    lines.append("[VERDICT]")
    lines.append(f"  ENTRY RULE catches  : winners {entry_win}/{n_win} ({entry_win/max(n_win,1)*100:.0f}%)  "
                 f"vs losers {entry_los}/{n_los} ({entry_los/max(n_los,1)*100:.0f}%)")
    lines.append(f"  EXIT  TYPE matches  : winners {catch_win}/{n_win} ({catch_win/max(n_win,1)*100:.0f}%)  "
                 f"vs losers {catch_los}/{n_los} ({catch_los/max(n_los,1)*100:.0f}%)")
    gate1 = entry_win >= 10
    gate2 = catch_win >= 10
    gate3 = entry_los < 3
    lines.append(f"  Ship gate informal:")
    lines.append(f"    >=10/{n_win} winners entry-rule       : {'PASS' if gate1 else 'FAIL'}")
    lines.append(f"    >=10/{n_win} winners exit-classified  : {'PASS' if gate2 else 'FAIL'}")
    lines.append(f"    < 3/{n_los} losers entry-rule         : {'PASS' if gate3 else 'FAIL'}")
    overall = "PASS" if (gate1 and gate2 and gate3) else "FAIL"
    lines.append(f"  Overall: {overall}")

    text = "\n".join(lines)
    print(text)
    out_txt = os.path.join(OUT_DIR, "personal_puts_level_diagnostic.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

    # CSV per-trade
    out_csv = os.path.join(OUT_DIR, "personal_puts_level_diagnostic.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "win_loss", "ticker", "entry_date", "exit_date",
            "ret_pct", "days_held", "bto_close", "stc_close",
            "z_bto", "z_stc",
            "swing_high_20d", "swing_high_50d", "broken_level",
            "pct_above_broken_at_bto",
            "swing_low_60d", "fib_382", "fib_500", "fib_618",
            "stc_dist_broken_pct", "stc_dist_fib382_pct",
            "stc_dist_fib500_pct", "stc_dist_fib618_pct",
            "entry_rule_pass", "exit_class",
        ])
        for m in measurements:
            t = m.trade
            wl = "W" if t.gross_pl > 0 else "L"
            def _d(lv): return None if lv is None or lv <= 0 else (m.stc_close / lv - 1.0)
            w.writerow([
                wl, t.ticker, t.entry_date, t.exit_date,
                round(t.ret_pct, 4), t.days_held,
                round(m.bto_close, 2), round(m.stc_close, 2),
                round(m.z_bto, 3) if m.z_bto is not None else "",
                round(m.z_stc, 3) if m.z_stc is not None else "",
                round(m.swing_high_20d, 2) if m.swing_high_20d else "",
                round(m.swing_high_50d, 2) if m.swing_high_50d else "",
                round(m.broken_level, 2) if m.broken_level else "",
                round(m.pct_above_broken_at_bto, 4) if m.pct_above_broken_at_bto is not None else "",
                round(m.swing_low_60d, 2) if m.swing_low_60d else "",
                round(m.fib_382, 2) if m.fib_382 else "",
                round(m.fib_500, 2) if m.fib_500 else "",
                round(m.fib_618, 2) if m.fib_618 else "",
                round(_d(m.broken_level), 4) if _d(m.broken_level) is not None else "",
                round(_d(m.fib_382), 4) if _d(m.fib_382) is not None else "",
                round(_d(m.fib_500), 4) if _d(m.fib_500) is not None else "",
                round(_d(m.fib_618), 4) if _d(m.fib_618) is not None else "",
                _classify_entry(m),
                _classify_exit(m),
            ])
    print(f"\n  -> {out_txt}")
    print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()
