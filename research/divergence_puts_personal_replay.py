"""Replay the winning puts variant (rsi>=2 gap=8, lb=25, Variant-C shell)
against the user's real 2025 put trades.

Question: how many of the user's actual puts would the bot have fired on?
And do the matched trades do better than the unmatched?

Two passes per personal put trade:
  STRICT      signal must fire on the exact entry day (or last trading day
              at-or-before entry_date if entry_date is a non-trading day).
  LOOSE       signal must fire any time in [entry_date - 5, entry_date + 1]
              (allows for the discretionary trader being a few days early
              or late vs. textbook).

Frozen signal (from divergence_puts_tune.py, qualifier cell):
  band              z in [+2.0, +2.5]
  lookback          25 bars
  RSI_DIVERG_MIN    2.0
  PIVOT_GAP         8
  resistance        >= 5 prior 252d bars within +/-2% of recent 20b high
  cooldown          20 bars (within ticker)

Output: research/out/divergence_puts_personal_replay.txt
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from statistics import mean, median

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.indicators import rsi as rsi_series  # noqa: E402
from research.oversold_call_exit_sweep import _log_channel_z, LOG_WINDOW  # noqa: E402
from research.volatile_universe_puts_sweep import SWEET_LO_PUT, SWEET_HI_PUT  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
AUDIT_CSV = os.path.join(OUT_DIR, "personal_trade_audit.csv")

# Frozen winning cell
LOOKBACK = 25
RSI_PERIOD = 14
RSI_DIVERG_MIN = 2.0
PIVOT_GAP = 8
SUPPORT_PCT = 0.02
SUPPORT_HITS_MIN = 5
COOLDOWN_BARS = 20

LOOSE_BACK_DAYS = 5      # how many trading days before entry the signal may fire
LOOSE_FWD_DAYS = 1       # ... and after


@dataclass
class PutTrade:
    ticker: str
    entry_date: date
    gross_pl: float
    ret_pct: float
    days_held: int


def _load_put_trades() -> list[PutTrade]:
    """Dedupe to one row per (ticker, entry_date) by summing FIFO sub-lots."""
    by_key: dict[tuple[str, date], dict] = {}
    with open(AUDIT_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["right"] != "PUT":
                continue
            ed = datetime.strptime(row["entry_date"], "%Y-%m-%d").date()
            key = (row["ticker"], ed)
            gp = float(row["gross_pl"])
            ec = abs(float(row["entry_cost"]))
            slot = by_key.setdefault(key, {
                "ticker": row["ticker"],
                "entry_date": ed,
                "gross_pl": 0.0,
                "entry_cost": 0.0,
                "days_held": int(row["days_held"]),
            })
            slot["gross_pl"] += gp
            slot["entry_cost"] += ec
            # days_held should agree on a true single trade; take the max
            slot["days_held"] = max(slot["days_held"], int(row["days_held"]))
    out = []
    for slot in by_key.values():
        ret = slot["gross_pl"] / slot["entry_cost"] if slot["entry_cost"] > 0 else 0.0
        out.append(PutTrade(
            ticker=slot["ticker"],
            entry_date=slot["entry_date"],
            gross_pl=slot["gross_pl"],
            ret_pct=ret,
            days_held=slot["days_held"],
        ))
    out.sort(key=lambda t: t.entry_date)
    return out


def _rsi_arr(closes: np.ndarray) -> np.ndarray:
    return rsi_series(pd.Series(closes), period=RSI_PERIOD).to_numpy(dtype=float)


def _check_signal_at(
    df: pd.DataFrame, idx: int, z: np.ndarray, rsi_a: np.ndarray,
    *, band_lo: float = SWEET_LO_PUT, band_hi: float = SWEET_HI_PUT,
) -> tuple[bool, str | None]:
    """Returns (fired, first_failing_gate). first_failing_gate is None on fire."""
    if idx - LOG_WINDOW + 1 < 0:
        return False, "history"
    zi = z[idx]
    if not np.isfinite(zi):
        return False, "z_nan"
    if not (band_lo <= zi <= band_hi):
        return False, "band"

    highs = df["high"].to_numpy(dtype=float)
    win_highs = highs[idx - LOOKBACK + 1: idx + 1]
    recent_hi = float(np.nanmax(win_highs))
    if not np.isfinite(highs[idx]) or highs[idx] < recent_hi - 1e-12:
        return False, "new_high"

    prior_end = idx - PIVOT_GAP + 1
    prior_start = idx - LOOKBACK + 1
    if prior_end - prior_start < 2:
        return False, "gap_window"
    prior_slice = highs[prior_start: prior_end]
    if not np.any(np.isfinite(prior_slice)):
        return False, "nan_pivot"
    prior_arg = int(np.nanargmax(prior_slice))
    prior_idx = prior_start + prior_arg
    prior_high = float(highs[prior_idx])
    if not np.isfinite(prior_high) or highs[idx] <= prior_high:
        return False, "lower_high"

    ri = rsi_a[idx]
    rp = rsi_a[prior_idx]
    if not (np.isfinite(ri) and np.isfinite(rp)):
        return False, "rsi_nan"
    if ri > rp - RSI_DIVERG_MIN:
        return False, "rsi_div"

    band_lo_p = recent_hi * (1.0 - SUPPORT_PCT)
    band_hi_p = recent_hi * (1.0 + SUPPORT_PCT)
    prior_highs = highs[idx - LOG_WINDOW + 1: idx - LOOKBACK + 1]
    hits = int(((prior_highs >= band_lo_p) & (prior_highs <= band_hi_p)).sum())
    if hits < SUPPORT_HITS_MIN:
        return False, "resistance"

    return True, None


def _index_for_date(dates: list[pd.Timestamp], target: date) -> int | None:
    """Last trading-day index at-or-before target. None if before history."""
    arr = np.array([d.date() for d in dates])
    mask = arr <= target
    if not mask.any():
        return None
    return int(np.where(mask)[0][-1])


def _stats(label: str, pls: list[float], rets: list[float], hold: list[int]) -> list[str]:
    if not pls:
        return [f"  {label:25} n=0"]
    wins = [p for p in pls if p > 0]
    losses = [p for p in pls if p <= 0]
    gw = sum(wins); gl = abs(sum(losses))
    pf = gw / gl if gl > 0 else float("inf")
    wr = len(wins) / len(pls) * 100
    return [
        f"  {label:25} n={len(pls):>2}  WR {wr:5.1f}%  total ${sum(pls):+,.0f}  "
        f"PF {pf:.2f}  exp ${mean(pls):+,.0f}/trade  median ret {median(rets)*100:+.1f}%  med-hold {median(hold):.0f}d"
    ]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    trades = _load_put_trades()
    print(f"Loaded {len(trades)} unique-(ticker, entry_date) put trades")

    tks = sorted({t.ticker for t in trades})
    print(f"Tickers: {tks}")

    # Need enough history before each trade: 252d log window + 25 bar lookback.
    # Trades span 2025-09 to 2025-12, fetch 3y to cover.
    print(f"Fetching {len(tks)} tickers x 3y...")
    bars = fetch_bars_bulk(tks, period="3y")
    print(f"  got {len(bars)} tickers with bars")

    # Precompute z and RSI per ticker
    z_by_tk: dict[str, np.ndarray] = {}
    rsi_by_tk: dict[str, np.ndarray] = {}
    df_by_tk: dict[str, pd.DataFrame] = {}
    dates_by_tk: dict[str, list[pd.Timestamp]] = {}
    for tk, df in bars.items():
        if df is None or df.empty:
            continue
        df = df.dropna(subset=["close", "high", "low"]).copy()
        if len(df) < LOG_WINDOW + LOOKBACK + 10:
            continue
        df_by_tk[tk] = df
        closes = df["close"].to_numpy(dtype=float)
        z_by_tk[tk] = _log_channel_z(closes)
        rsi_by_tk[tk] = _rsi_arr(closes)
        ts_col = df["timestamp"]
        dates_by_tk[tk] = [pd.Timestamp(d).to_pydatetime() for d in ts_col]

    lines: list[str] = []
    lines.append("DIV-PUTS personal-audit replay -- rsi>=2 gap=8 lb=25, Variant-C shell")
    lines.append(f"  band [+{SWEET_LO_PUT}, +{SWEET_HI_PUT}], resistance >={SUPPORT_HITS_MIN} prior bars within +/-{SUPPORT_PCT*100:.0f}%,")
    lines.append(f"  RSI_DIVERG_MIN={RSI_DIVERG_MIN} pts, PIVOT_GAP={PIVOT_GAP}, cooldown {COOLDOWN_BARS}b.")
    lines.append(f"  STRICT window: signal must fire on entry_date itself.")
    lines.append(f"  LOOSE  window: signal must fire in [entry-{LOOSE_BACK_DAYS}d, entry+{LOOSE_FWD_DAYS}d] trading days.")
    lines.append(f"  Real put trades (deduped): {len(trades)}.")
    lines.append("")

    # Per-trade table -- run BOTH the strict band (shipped cell) AND a
    # wide-band variant (z in [+1.0, +3.5]) so we can see how much of the
    # 0% match is due to the band vs. the divergence rule itself.
    lines.append("PER-TRADE TABLE")
    lines.append(f"  {'entry':10} {'tkr':5} {'z':>6} {'rsi':>5}  S L Sw Lw | strict first-fail | wide first-fail")
    lines.append(f"  S=STRICT band [+{SWEET_LO_PUT},+{SWEET_HI_PUT}]   Sw=WIDE band [+1.0,+3.5]   L/Lw = +/- td window")
    strict_hits: list[bool] = []
    loose_hits: list[bool] = []
    wide_strict_hits: list[bool] = []
    wide_loose_hits: list[bool] = []
    strict_first_fail = defaultdict(int)
    wide_first_fail = defaultdict(int)
    z_at_entry: list[float] = []

    for t in trades:
        if t.ticker not in df_by_tk:
            lines.append(f"  {t.entry_date} {t.ticker:5}    --     --   . . .  .  | no bars for ticker | --")
            for arr in (strict_hits, loose_hits, wide_strict_hits, wide_loose_hits):
                arr.append(False)
            z_at_entry.append(float("nan"))
            continue
        df = df_by_tk[t.ticker]
        z = z_by_tk[t.ticker]
        rsi_a = rsi_by_tk[t.ticker]
        dates = dates_by_tk[t.ticker]

        idx = _index_for_date(dates, t.entry_date)
        if idx is None:
            for arr in (strict_hits, loose_hits, wide_strict_hits, wide_loose_hits):
                arr.append(False)
            z_at_entry.append(float("nan"))
            lines.append(f"  {t.entry_date} {t.ticker:5}    --     --   . . .  .  | history starts after entry | --")
            continue

        # STRICT
        fired_s, gate_s = _check_signal_at(df, idx, z, rsi_a)
        strict_hits.append(fired_s)
        if not fired_s and gate_s:
            strict_first_fail[gate_s] += 1

        # STRICT, wide band
        fired_sw, gate_sw = _check_signal_at(
            df, idx, z, rsi_a, band_lo=1.0, band_hi=3.5,
        )
        wide_strict_hits.append(fired_sw)
        if not fired_sw and gate_sw:
            wide_first_fail[gate_sw] += 1

        # LOOSE (strict band)
        lo_idx = max(0, idx - LOOSE_BACK_DAYS)
        hi_idx = min(len(df) - 1, idx + LOOSE_FWD_DAYS)
        fired_l = any(_check_signal_at(df, j, z, rsi_a)[0]
                      for j in range(lo_idx, hi_idx + 1))
        loose_hits.append(fired_l)

        # LOOSE (wide band)
        fired_lw = any(_check_signal_at(df, j, z, rsi_a, band_lo=1.0, band_hi=3.5)[0]
                       for j in range(lo_idx, hi_idx + 1))
        wide_loose_hits.append(fired_lw)

        zv = z[idx] if np.isfinite(z[idx]) else float("nan")
        rv = rsi_a[idx] if np.isfinite(rsi_a[idx]) else float("nan")
        z_at_entry.append(zv)
        marks = "".join("Y" if h else "." for h in (fired_s, fired_l, fired_sw, fired_lw))
        lines.append(
            f"  {t.entry_date} {t.ticker:5} {zv:>+6.2f} {rv:>5.1f}   {marks[0]} {marks[1]} {marks[2]}  {marks[3]}  | "
            f"{(gate_s or '-'):<14} | {(gate_sw or '-')}"
        )

    lines.append("")
    n = len(trades)
    s_hit = sum(strict_hits); l_hit = sum(loose_hits)
    sw_hit = sum(wide_strict_hits); lw_hit = sum(wide_loose_hits)
    lines.append(f"MATCH RATES")
    lines.append(f"  STRICT band [+{SWEET_LO_PUT},+{SWEET_HI_PUT}], exact entry day      : {s_hit:>2}/{n} = {s_hit/n*100:5.1f}%")
    lines.append(f"  STRICT band, [-{LOOSE_BACK_DAYS},+{LOOSE_FWD_DAYS}] td window               : {l_hit:>2}/{n} = {l_hit/n*100:5.1f}%")
    lines.append(f"  WIDE   band [+1.0,+3.5], exact entry day      : {sw_hit:>2}/{n} = {sw_hit/n*100:5.1f}%")
    lines.append(f"  WIDE   band, [-{LOOSE_BACK_DAYS},+{LOOSE_FWD_DAYS}] td window               : {lw_hit:>2}/{n} = {lw_hit/n*100:5.1f}%")
    lines.append("")

    lines.append("FIRST-FAILING GATE COUNTS (n trades that didn't fire, by reason)")
    lines.append(f"  STRICT band fail counts (n={n - s_hit}):")
    for k in ("history", "z_nan", "band", "new_high", "gap_window", "nan_pivot",
              "lower_high", "rsi_nan", "rsi_div", "resistance"):
        if strict_first_fail.get(k, 0):
            lines.append(f"    {k:12}: {strict_first_fail[k]}")
    lines.append(f"  WIDE band fail counts (n={n - sw_hit}):")
    for k in ("history", "z_nan", "band", "new_high", "gap_window", "nan_pivot",
              "lower_high", "rsi_nan", "rsi_div", "resistance"):
        if wide_first_fail.get(k, 0):
            lines.append(f"    {k:12}: {wide_first_fail[k]}")
    lines.append("")

    # Z DISTRIBUTION at user's actual entry day
    valid_z = [z for z in z_at_entry if np.isfinite(z)]
    if valid_z:
        lines.append("Z AT USER'S ENTRY DAY -- distribution")
        bins = [(-99, -1, "z < -1"), (-1, 0, "-1 .. 0"), (0, 0.5, "0 .. +0.5"),
                (0.5, 1.0, "+0.5 .. +1.0"), (1.0, 1.5, "+1.0 .. +1.5"),
                (1.5, 2.0, "+1.5 .. +2.0"), (2.0, 2.5, "+2.0 .. +2.5  <- shipped band"),
                (2.5, 99, "+2.5+")]
        for lo, hi, lbl in bins:
            ct = sum(1 for z in valid_z if lo <= z < hi)
            bar = "#" * ct
            lines.append(f"  {lbl:30} {ct:>2}  {bar}")
        lines.append(f"  median z at entry: {sorted(valid_z)[len(valid_z)//2]:+.2f}")
        lines.append(f"  trades w/ z >= +2.0: {sum(1 for z in valid_z if z >= 2.0)} / {len(valid_z)}")
        lines.append("")

    # Outcomes split
    lines.append("OUTCOMES BY MATCH BUCKET")
    all_pl = [t.gross_pl for t in trades]
    all_ret = [t.ret_pct for t in trades]
    all_hold = [t.days_held for t in trades]
    lines.extend(_stats("ALL PUT TRADES", all_pl, all_ret, all_hold))
    lines.append("")
    lines.append("  -- STRICT split --")
    s_pl_m = [trades[i].gross_pl for i in range(n) if strict_hits[i]]
    s_ret_m = [trades[i].ret_pct for i in range(n) if strict_hits[i]]
    s_hold_m = [trades[i].days_held for i in range(n) if strict_hits[i]]
    s_pl_u = [trades[i].gross_pl for i in range(n) if not strict_hits[i]]
    s_ret_u = [trades[i].ret_pct for i in range(n) if not strict_hits[i]]
    s_hold_u = [trades[i].days_held for i in range(n) if not strict_hits[i]]
    lines.extend(_stats("MATCHED (signal fired)", s_pl_m, s_ret_m, s_hold_m))
    lines.extend(_stats("UNMATCHED (no signal)", s_pl_u, s_ret_u, s_hold_u))
    lines.append("")
    lines.append("  -- LOOSE split --")
    l_pl_m = [trades[i].gross_pl for i in range(n) if loose_hits[i]]
    l_ret_m = [trades[i].ret_pct for i in range(n) if loose_hits[i]]
    l_hold_m = [trades[i].days_held for i in range(n) if loose_hits[i]]
    l_pl_u = [trades[i].gross_pl for i in range(n) if not loose_hits[i]]
    l_ret_u = [trades[i].ret_pct for i in range(n) if not loose_hits[i]]
    l_hold_u = [trades[i].days_held for i in range(n) if not loose_hits[i]]
    lines.extend(_stats("MATCHED (signal fired)", l_pl_m, l_ret_m, l_hold_m))
    lines.extend(_stats("UNMATCHED (no signal)", l_pl_u, l_ret_u, l_hold_u))
    lines.append("")

    # Headline interpretation hooks (read in the response, not the file)
    lines.append("READING")
    lines.append("  high MATCHED PF/WR vs UNMATCHED -> divergence selects the user's better trades")
    lines.append("  similar PF/WR                   -> divergence is orthogonal (catches different setups)")
    lines.append("  low MATCH rate                  -> bot would have skipped most of the user's puts")
    lines.append("    (informative either way: user trades a DIFFERENT regime than this signal)")

    out_txt = os.path.join(OUT_DIR, "divergence_puts_personal_replay.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n  -> {out_txt}")


if __name__ == "__main__":
    main()
