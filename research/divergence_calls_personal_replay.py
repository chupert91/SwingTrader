"""Replay the winning calls variant (DIV+R6 union, lb=60) against the
user's real 2025 call trades.

The union has TWO firing legs; a personal trade is "matched" if EITHER
leg fires at entry:

  LEG A  R6 baseline    same logic as scan_candidates in production:
                        z[i-1] < -2.0, z[i] in [-2.5, -2.0] (fresh cross),
                        same-day breadth tier in {OK, PRIME, PANIC},
                        ADV >= MIN_ADV_M. Breadth is computed across the
                        FULL volatile-universe (102 tickers) so the tier
                        gate matches live behavior.
  LEG B  Bullish RSI    z[i] in [-3.5, -1.0] ('outside 1 sigma'),
         divergence     new local low in 60-bar window,
                        prior pivot at 8+ bar gap = lower-low in price,
                        RSI[i] >= RSI[prior_pivot] + 3.0,
                        cooldown 20 bars.

Two windows per leg, same as the puts replay:
  STRICT  fire on exact entry day (last td at-or-before entry_date)
  LOOSE   any bar in [entry - 5td, entry + 1td]

Also reports the z distribution at user's actual call-entry day,
analogous to the puts replay diagnostic.

Output: research/out/divergence_calls_personal_replay.txt
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from statistics import mean, median

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.indicators import rsi as rsi_series  # noqa: E402
from research.volatile_universe import universe  # noqa: E402
from research.oversold_call_exit_sweep import (  # noqa: E402
    _log_channel_z, LOG_WINDOW, SWEET_LO, SWEET_HI,
    TIER_THRESH, ACCEPT_TIERS,
)
from research.sma_reversion_option_sim import MIN_ADV_M  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
AUDIT_CSV = os.path.join(OUT_DIR, "personal_trade_audit.csv")

# Bullish-divergence params (lb=60 -- the winning lookback in DIV+R6 union)
LOOKBACK = 60
RSI_PERIOD = 14
RSI_DIVERG_MIN = 3.0
PIVOT_GAP = 8
COOLDOWN_BARS = 20

# Wide call band ('outside 1 sigma' = z <= -1, capped at -3.5)
CALL_BAND_LO = -3.5
CALL_BAND_HI = -1.0

LOOSE_BACK_DAYS = 5
LOOSE_FWD_DAYS = 1


@dataclass
class CallTrade:
    ticker: str
    entry_date: date
    gross_pl: float
    ret_pct: float
    days_held: int


def _load_call_trades() -> list[CallTrade]:
    by_key: dict[tuple[str, date], dict] = {}
    with open(AUDIT_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["right"] != "CALL":
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
            slot["days_held"] = max(slot["days_held"], int(row["days_held"]))
    out = []
    for slot in by_key.values():
        ret = slot["gross_pl"] / slot["entry_cost"] if slot["entry_cost"] > 0 else 0.0
        out.append(CallTrade(
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


def _tier_for(breadth: int) -> str:
    for t in ("PANIC", "PRIME", "OK", "WEAK"):
        if breadth >= TIER_THRESH[t]:
            return t
    return "?"


def _build_baseline_signal_set(
    dfs_by_tk: dict, z_by_tk: dict, dates_by_tk: dict,
) -> set[tuple[str, date]]:
    """Reproduce the R6 baseline call signal across the FULL universe and
    return the set of (ticker, fire_date) pairs that survived the tier
    gate. Mirrors _signals() from oversold_call_exit_sweep but doesn't
    need the Series wrapper."""
    # First pass: count fresh crosses into z in [-2.5, -2.0] per date,
    # collect eligible candidates.
    breadth_by_day: dict[date, int] = defaultdict(int)
    candidates: list[tuple[str, date, float, float]] = []  # (tk, date, z, adv_proxy)

    for tk, df in dfs_by_tk.items():
        z = z_by_tk[tk]
        dates = dates_by_tk[tk]
        close = df["close"].to_numpy(dtype=float)
        vol = df["volume"].to_numpy(dtype=float)
        # 20d ADV in $millions
        adv_m = pd.Series(close * vol).rolling(20).mean().to_numpy() / 1_000_000.0
        for i in range(1, len(close)):
            if not np.isfinite(z[i]) or not np.isfinite(z[i - 1]):
                continue
            crossed = z[i] <= SWEET_HI and z[i - 1] > SWEET_HI
            if crossed:
                breadth_by_day[dates[i]] += 1
            if not crossed:
                continue
            if not (SWEET_LO <= z[i] <= SWEET_HI):
                continue
            adv = adv_m[i]
            if not (np.isfinite(adv) and adv >= MIN_ADV_M):
                continue
            candidates.append((tk, dates[i], float(z[i]), float(adv)))

    fires: set[tuple[str, date]] = set()
    for tk, d, _, _ in candidates:
        tier = _tier_for(breadth_by_day.get(d, 0))
        if tier in ACCEPT_TIERS:
            fires.add((tk, d))
    return fires


def _check_divergence_at(
    df: pd.DataFrame, idx: int, z: np.ndarray, rsi_a: np.ndarray,
    *, band_lo: float, band_hi: float,
) -> tuple[bool, str | None]:
    """Bullish RSI divergence variant. Returns (fired, first_failing_gate)."""
    if idx - LOG_WINDOW + 1 < 0:
        return False, "history"
    zi = z[idx]
    if not np.isfinite(zi):
        return False, "z_nan"
    if not (band_lo <= zi <= band_hi):
        return False, "band"

    lows = df["low"].to_numpy(dtype=float)
    win_lows = lows[idx - LOOKBACK + 1: idx + 1]
    recent_lo = float(np.nanmin(win_lows))
    if not np.isfinite(lows[idx]) or lows[idx] > recent_lo + 1e-12:
        return False, "new_low"

    prior_end = idx - PIVOT_GAP + 1
    prior_start = idx - LOOKBACK + 1
    if prior_end - prior_start < 2:
        return False, "gap_window"
    prior_slice = lows[prior_start: prior_end]
    if not np.any(np.isfinite(prior_slice)):
        return False, "nan_pivot"
    prior_arg = int(np.nanargmin(prior_slice))
    prior_idx = prior_start + prior_arg
    prior_low = float(lows[prior_idx])
    if not np.isfinite(prior_low) or lows[idx] >= prior_low:
        return False, "higher_low"  # need PRICE lower-low for bullish div

    ri = rsi_a[idx]
    rp = rsi_a[prior_idx]
    if not (np.isfinite(ri) and np.isfinite(rp)):
        return False, "rsi_nan"
    if ri < rp + RSI_DIVERG_MIN:
        return False, "rsi_div"

    return True, None


def _index_for_date(dates: list, target: date) -> int | None:
    arr = np.array([d if isinstance(d, date) else pd.Timestamp(d).date() for d in dates])
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
    trades = _load_call_trades()
    print(f"Loaded {len(trades)} unique-(ticker, entry_date) call trades")
    user_tks = sorted({t.ticker for t in trades})
    print(f"User tickers: {user_tks}")

    # Need full volatile universe for the BREADTH calc in the R6 baseline.
    # Union the user's tickers in case any are missing from the universe.
    uni = list(set(universe()) | set(user_tks))
    print(f"Fetching {len(uni)} tickers (volatile universe + user) x 3y...")
    bars = fetch_bars_bulk(uni, period="3y")
    print(f"  got {len(bars)} tickers with bars")

    z_by_tk: dict[str, np.ndarray] = {}
    rsi_by_tk: dict[str, np.ndarray] = {}
    dfs_by_tk: dict[str, pd.DataFrame] = {}
    dates_by_tk: dict[str, list[date]] = {}
    for tk, df in bars.items():
        if df is None or df.empty:
            continue
        df = df.dropna(subset=["close", "high", "low"]).copy()
        if len(df) < LOG_WINDOW + LOOKBACK + 10:
            continue
        dfs_by_tk[tk] = df
        closes = df["close"].to_numpy(dtype=float)
        z_by_tk[tk] = _log_channel_z(closes)
        rsi_by_tk[tk] = _rsi_arr(closes)
        ts_col = df["timestamp"]
        dates_by_tk[tk] = [pd.Timestamp(d).date() for d in ts_col]

    # Build R6 baseline fires across the full universe (uses breadth)
    print("Computing R6 baseline signal across full universe...")
    baseline_fires = _build_baseline_signal_set(dfs_by_tk, z_by_tk, dates_by_tk)
    print(f"  baseline produced {len(baseline_fires)} (ticker, date) firings universe-wide")

    lines: list[str] = []
    lines.append("DIV+R6 union CALLS personal-audit replay -- lb=60, RSI_DIV>=3.0, gap=8")
    lines.append(f"  LEG A: R6 baseline   z fresh cross into [{SWEET_LO},{SWEET_HI}], breadth tier in {sorted(ACCEPT_TIERS)},")
    lines.append(f"         ADV>={MIN_ADV_M}M$. Breadth computed over {len(dfs_by_tk)}-ticker universe.")
    lines.append(f"  LEG B: bullish div   z in [{CALL_BAND_LO},{CALL_BAND_HI}], new low (60b),")
    lines.append(f"         lower-low in price vs prior pivot (gap {PIVOT_GAP}), RSI higher-low by >= {RSI_DIVERG_MIN}.")
    lines.append(f"  Match if EITHER leg fires. STRICT = entry day; LOOSE = [-{LOOSE_BACK_DAYS}, +{LOOSE_FWD_DAYS}] td window.")
    lines.append(f"  Real call trades (deduped): {len(trades)}.")
    lines.append("")

    lines.append("PER-TRADE TABLE")
    lines.append(f"  {'entry':10} {'tkr':5} {'z':>6} {'rsi':>5}  Ra Rl Da Dl Ua Ul | DIV first-fail (strict)")
    lines.append(f"  R = R6 leg, D = DIV leg, U = UNION.  a=STRICT (entry day), l=LOOSE ([-5,+1]td)")

    r6_strict: list[bool] = []
    r6_loose: list[bool] = []
    div_strict: list[bool] = []
    div_loose: list[bool] = []
    z_at_entry: list[float] = []
    div_fail = defaultdict(int)

    for t in trades:
        if t.ticker not in dfs_by_tk:
            for arr in (r6_strict, r6_loose, div_strict, div_loose):
                arr.append(False)
            z_at_entry.append(float("nan"))
            lines.append(f"  {t.entry_date} {t.ticker:5}    --     --   . . . . .  .  | no bars for ticker")
            continue

        df = dfs_by_tk[t.ticker]
        z = z_by_tk[t.ticker]
        rsi_a = rsi_by_tk[t.ticker]
        dates = dates_by_tk[t.ticker]
        idx = _index_for_date(dates, t.entry_date)
        if idx is None:
            for arr in (r6_strict, r6_loose, div_strict, div_loose):
                arr.append(False)
            z_at_entry.append(float("nan"))
            lines.append(f"  {t.entry_date} {t.ticker:5}    --     --   . . . . .  .  | history starts after entry")
            continue

        # LEG A: R6 baseline
        d_strict = dates[idx]
        ra = (t.ticker, d_strict) in baseline_fires
        rl = False
        lo_idx = max(0, idx - LOOSE_BACK_DAYS)
        hi_idx = min(len(df) - 1, idx + LOOSE_FWD_DAYS)
        for j in range(lo_idx, hi_idx + 1):
            if (t.ticker, dates[j]) in baseline_fires:
                rl = True
                break
        r6_strict.append(ra); r6_loose.append(rl)

        # LEG B: Bullish divergence
        fired_da, gate_da = _check_divergence_at(
            df, idx, z, rsi_a, band_lo=CALL_BAND_LO, band_hi=CALL_BAND_HI,
        )
        if not fired_da and gate_da:
            div_fail[gate_da] += 1
        fired_dl = False
        for j in range(lo_idx, hi_idx + 1):
            f, _ = _check_divergence_at(df, j, z, rsi_a, band_lo=CALL_BAND_LO, band_hi=CALL_BAND_HI)
            if f:
                fired_dl = True
                break
        div_strict.append(fired_da); div_loose.append(fired_dl)

        ua = ra or fired_da
        ul = rl or fired_dl

        zv = z[idx] if np.isfinite(z[idx]) else float("nan")
        rv = rsi_a[idx] if np.isfinite(rsi_a[idx]) else float("nan")
        z_at_entry.append(zv)
        marks = [ra, rl, fired_da, fired_dl, ua, ul]
        m = "".join("Y" if x else "." for x in marks)
        lines.append(
            f"  {t.entry_date} {t.ticker:5} {zv:>+6.2f} {rv:>5.1f}   {m[0]} {m[1]} {m[2]} {m[3]} {m[4]}  {m[5]}  | "
            f"{gate_da or '-'}"
        )

    lines.append("")
    n = len(trades)
    union_strict = [r6_strict[i] or div_strict[i] for i in range(n)]
    union_loose = [r6_loose[i] or div_loose[i] for i in range(n)]

    def pct(hits): return f"{sum(hits):>2}/{n} = {sum(hits)/n*100:5.1f}%"
    lines.append("MATCH RATES")
    lines.append(f"  R6 baseline,     STRICT (exact entry day) : {pct(r6_strict)}")
    lines.append(f"  R6 baseline,     LOOSE  [-{LOOSE_BACK_DAYS},+{LOOSE_FWD_DAYS}] td window : {pct(r6_loose)}")
    lines.append(f"  Bullish DIV lb60, STRICT                  : {pct(div_strict)}")
    lines.append(f"  Bullish DIV lb60, LOOSE                   : {pct(div_loose)}")
    lines.append(f"  UNION (either),   STRICT                  : {pct(union_strict)}")
    lines.append(f"  UNION (either),   LOOSE                   : {pct(union_loose)}")
    lines.append("")

    lines.append("DIV-leg FIRST-FAILING GATE COUNTS (n trades div didn't fire, by reason)")
    for k in ("history", "z_nan", "band", "new_low", "gap_window", "nan_pivot",
              "higher_low", "rsi_nan", "rsi_div"):
        if div_fail.get(k, 0):
            lines.append(f"  {k:12}: {div_fail[k]}")
    lines.append("")

    valid_z = [z for z in z_at_entry if np.isfinite(z)]
    if valid_z:
        lines.append("Z AT USER'S CALL-ENTRY DAY -- distribution")
        bins = [(-99, -3.5, "z < -3.5"), (-3.5, -2.5, "-3.5 .. -2.5"),
                (-2.5, -2.0, "-2.5 .. -2.0  <- R6 baseline band"),
                (-2.0, -1.5, "-2.0 .. -1.5"), (-1.5, -1.0, "-1.5 .. -1.0  (still in 1-sigma DIV band)"),
                (-1.0, -0.5, "-1.0 .. -0.5"), (-0.5, 0, "-0.5 .. 0"),
                (0, 0.5, "0 .. +0.5"), (0.5, 99, "z > +0.5")]
        for lo, hi, lbl in bins:
            ct = sum(1 for z in valid_z if lo <= z < hi)
            bar = "#" * ct
            lines.append(f"  {lbl:48} {ct:>2}  {bar}")
        lines.append(f"  median z at call entry: {sorted(valid_z)[len(valid_z)//2]:+.2f}")
        lines.append(f"  trades w/ z <= -1.0 (in DIV wide band): {sum(1 for z in valid_z if z <= -1.0)} / {len(valid_z)}")
        lines.append(f"  trades w/ z <= -2.0 (in R6 band)      : {sum(1 for z in valid_z if z <= -2.0)} / {len(valid_z)}")
        lines.append("")

    lines.append("OUTCOMES BY MATCH BUCKET (STRICT, union)")
    all_pl = [t.gross_pl for t in trades]
    all_ret = [t.ret_pct for t in trades]
    all_hold = [t.days_held for t in trades]
    lines.extend(_stats("ALL CALL TRADES", all_pl, all_ret, all_hold))
    m_pl = [trades[i].gross_pl for i in range(n) if union_strict[i]]
    m_ret = [trades[i].ret_pct for i in range(n) if union_strict[i]]
    m_hold = [trades[i].days_held for i in range(n) if union_strict[i]]
    u_pl = [trades[i].gross_pl for i in range(n) if not union_strict[i]]
    u_ret = [trades[i].ret_pct for i in range(n) if not union_strict[i]]
    u_hold = [trades[i].days_held for i in range(n) if not union_strict[i]]
    lines.extend(_stats("MATCHED (union fired)", m_pl, m_ret, m_hold))
    lines.extend(_stats("UNMATCHED (no signal)", u_pl, u_ret, u_hold))
    lines.append("")
    lines.append("OUTCOMES BY MATCH BUCKET (LOOSE, union)")
    m_pl_l = [trades[i].gross_pl for i in range(n) if union_loose[i]]
    m_ret_l = [trades[i].ret_pct for i in range(n) if union_loose[i]]
    m_hold_l = [trades[i].days_held for i in range(n) if union_loose[i]]
    u_pl_l = [trades[i].gross_pl for i in range(n) if not union_loose[i]]
    u_ret_l = [trades[i].ret_pct for i in range(n) if not union_loose[i]]
    u_hold_l = [trades[i].days_held for i in range(n) if not union_loose[i]]
    lines.extend(_stats("MATCHED (union fired)", m_pl_l, m_ret_l, m_hold_l))
    lines.extend(_stats("UNMATCHED (no signal)", u_pl_l, u_ret_l, u_hold_l))

    out_txt = os.path.join(OUT_DIR, "divergence_calls_personal_replay.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n  -> {out_txt}")


if __name__ == "__main__":
    main()
