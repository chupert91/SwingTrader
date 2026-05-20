"""Reverse-engineer the user's 2025 winning put trades.

PF 3.80 on 20 hand-picked puts vs PF 0.50-0.85 on every automated put
signal we tested (research/out/volatile_universe_puts_sweep.txt,
research/out/puts_extreme_band_sweep.txt). The user is seeing SOMETHING
the simple `z >= +X first-touch` signal misses. This script tries to
find it.

Method:
  1. Pair the user's 2025 put trades from reference/DownloadTxnHistory.csv
     (reuse the pairing logic from personal_trade_audit.py).
  2. For each put, fetch ~14 months of yfinance bars ending at the entry
     date and compute a feature vector AT ENTRY:
         z_log         252d log-channel z (how stretched on entry?)
         z_log_prev    z one bar before entry (where it was)
         z_log_max20   max z over the prior 20 bars (recency of peak)
         days_since_zmax  bars since z hit its 20-bar high
         ret_5d, ret_10d, ret_20d  cumulative %returns into entry
         rsi_14        Wilder's RSI
         stoch_k       fast stoch %K
         rv_20         annualized realized vol of log returns, 20d
         vol_spike     today's volume / 20-bar avg volume
         dist_sma20    (close - sma20) / sma20 * 100
         dist_sma50    (close - sma50) / sma50 * 100
         gap_pct       today's open - yesterday's close, %
         body_pct      today's close - today's open, % (intraday tone)
         z_slope_5d    OLS slope of z over the last 5 bars (rolling-over?)
  3. Also capture what happened AFTER entry (the realized outcome):
         max_drop_pct  largest underlying drop within (hold_days) of entry
         days_to_drop  trading days from entry to that drop's bottom
  4. Compare features distribution: WINNERS vs LOSERS. Print a table of
     medians, plus per-trade detail.
  5. Look for any single feature (or pair) where wins cluster.

This is exploratory — the goal is to find a HYPOTHESIS for a feature
that, added to the z-signal, makes the put leg viable. The hypothesis
then needs to be backtested.

Run:
    python research/personal_puts_pattern_analysis.py
Outputs:
    research/out/personal_puts_pattern_analysis.txt
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median, mean

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from research.personal_trade_audit import (  # noqa: E402
    load_rows, pair_options, Trade,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

LOG_WINDOW = 252


# ---------- feature computations ----------------------------------------

def _log_channel_z(closes: np.ndarray, idx: int, window: int = LOG_WINDOW) -> float:
    """z at bar idx using the 252-bar log-OLS channel ending at idx."""
    if idx < window:
        return float("nan")
    y = np.log(closes[idx - window + 1: idx + 1])
    x = np.arange(window, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    sigma = float(np.std(y - fit, ddof=1))
    if sigma <= 0:
        return float("nan")
    return float((y[-1] - fit[-1]) / sigma)


def _rsi_wilder(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return float("nan")
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _stoch_k(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period:
        return float("nan")
    win = closes[-period:]
    lo, hi = win.min(), win.max()
    if hi == lo:
        return 50.0
    return float(100.0 * (closes[-1] - lo) / (hi - lo))


def _realized_vol(closes: np.ndarray, lookback: int = 20) -> float:
    if len(closes) < lookback + 1:
        return float("nan")
    logret = np.diff(np.log(closes[-(lookback + 1):]))
    return float(np.std(logret, ddof=1) * math.sqrt(252) * 100.0)


def _z_slope_n(z_series: np.ndarray, n: int = 5) -> float:
    if len(z_series) < n:
        return float("nan")
    tail = z_series[-n:]
    if np.any(np.isnan(tail)):
        return float("nan")
    x = np.arange(n, dtype=float)
    s, _ = np.polyfit(x, tail, 1)
    return float(s)


@dataclass
class PutFeatures:
    t: Trade
    z_log: float
    z_log_prev: float
    z_max_20: float
    days_since_zmax: int
    ret_5d: float
    ret_10d: float
    ret_20d: float
    rsi_14: float
    stoch_k: float
    rv_20: float
    vol_spike: float
    dist_sma20: float
    dist_sma50: float
    gap_pct: float
    body_pct: float
    z_slope_5d: float
    # post-entry
    max_drop_pct: float        # most-negative cumulative return on close, win-window
    days_to_drop: int
    final_close_change_pct: float  # actual %drop from entry to exit

    @property
    def is_win(self) -> bool:
        return self.t.gross_pl > 0


def _compute_features(t: Trade, df: pd.DataFrame) -> PutFeatures | None:
    """df is the underlying's daily bars indexed by date (oldest first).
    We locate the bar matching t.entry_date and compute features as of
    that bar."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df = df.reset_index(drop=True)
    try:
        entry_i = df.index[df["date"] == t.entry_date].tolist()
    except KeyError:
        return None
    if not entry_i:
        # The user may have entered on a date yfinance doesn't show
        # (rare). Find the closest bar on or before entry_date.
        before = df[df["date"] <= t.entry_date]
        if before.empty:
            return None
        entry_i = [before.index[-1]]
    i = entry_i[0]
    if i < LOG_WINDOW + 50:
        return None

    closes = df["close"].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float)
    vols = df["volume"].to_numpy(dtype=float)

    # z-channel at entry and 20 bars back.
    z_now = _log_channel_z(closes, i)
    z_prev = _log_channel_z(closes, i - 1)
    z_hist = np.array([_log_channel_z(closes, j) for j in range(i - 19, i + 1)])
    z_max_20 = float(np.nanmax(z_hist)) if not np.all(np.isnan(z_hist)) else float("nan")
    if np.isnan(z_max_20):
        days_since_zmax = -1
    else:
        zmax_idx = int(np.nanargmax(z_hist))
        days_since_zmax = (len(z_hist) - 1) - zmax_idx
    z_slope_5d = _z_slope_n(z_hist, n=5)

    # Returns into entry.
    def _ret(n: int) -> float:
        if i - n < 0:
            return float("nan")
        return (closes[i] / closes[i - n] - 1.0) * 100.0
    ret_5d = _ret(5)
    ret_10d = _ret(10)
    ret_20d = _ret(20)

    rsi_14 = _rsi_wilder(closes[: i + 1])
    stoch_k = _stoch_k(closes[: i + 1])
    rv_20 = _realized_vol(closes[: i + 1])

    # Volume vs 20-bar avg (use prior 20 not including today).
    vol_avg = float(np.mean(vols[i - 20: i])) if i >= 20 else float("nan")
    vol_spike = (vols[i] / vol_avg) if (vol_avg and vol_avg > 0) else float("nan")

    # SMA distance.
    def _sma(n: int) -> float:
        if i < n: return float("nan")
        s = float(np.mean(closes[i - n + 1: i + 1]))
        return ((closes[i] / s) - 1.0) * 100.0
    dist_sma20 = _sma(20)
    dist_sma50 = _sma(50)

    # Gap (today open vs yesterday close) and intraday body.
    gap_pct = ((opens[i] / closes[i - 1]) - 1.0) * 100.0 if i >= 1 else float("nan")
    body_pct = ((closes[i] / opens[i]) - 1.0) * 100.0 if opens[i] > 0 else float("nan")

    # Post-entry: max close-mark drop within the hold window.
    hold_end = min(i + max(t.days_held, 1) + 1, len(df))
    if hold_end > i + 1:
        future_closes = closes[i + 1: hold_end]
        future_returns = (future_closes / closes[i] - 1.0) * 100.0
        min_idx = int(np.argmin(future_returns))
        max_drop_pct = float(future_returns[min_idx])
        days_to_drop = min_idx + 1
    else:
        max_drop_pct = float("nan")
        days_to_drop = -1
    # Actual %change of underlying from entry close to exit-bar close (or today).
    final_i = min(i + t.days_held, len(df) - 1)
    final_close_change_pct = (closes[final_i] / closes[i] - 1.0) * 100.0 if final_i > i else 0.0

    return PutFeatures(
        t=t, z_log=z_now, z_log_prev=z_prev, z_max_20=z_max_20,
        days_since_zmax=days_since_zmax,
        ret_5d=ret_5d, ret_10d=ret_10d, ret_20d=ret_20d,
        rsi_14=rsi_14, stoch_k=stoch_k, rv_20=rv_20,
        vol_spike=vol_spike, dist_sma20=dist_sma20, dist_sma50=dist_sma50,
        gap_pct=gap_pct, body_pct=body_pct, z_slope_5d=z_slope_5d,
        max_drop_pct=max_drop_pct, days_to_drop=days_to_drop,
        final_close_change_pct=final_close_change_pct,
    )


# ---------- reporting ----------------------------------------------------

def _fmt(x: float, n: int = 1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "  —  "
    return f"{x:+{n + 6}.{n}f}" if isinstance(x, float) else str(x)


def _pctile(xs: list[float], p: float) -> float:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows(os.path.join(ROOT, "reference", "DownloadTxnHistory.csv"))
    all_trades = pair_options(rows)
    puts = [t for t in all_trades if t.right == "PUT"]
    print(f"Loaded {len(puts)} paired PUT trades from the user's 2025 history.")
    if not puts:
        return

    # Fetch underlyings (each ticker once — yfinance is rate-friendly that way).
    tickers = sorted({t.ticker for t in puts})
    print(f"Fetching ~14mo bars for {len(tickers)} underlyings...")
    bars = fetch_bars_bulk(tickers, period="14mo")
    # For some tickers we may need older data — re-fetch 5y if 14mo is too short.
    # The earliest 2025 entry is May; 14mo of bars from today would NOT cover
    # the 252-day window needed to compute z at that entry. Use 5y to be safe.
    print(f"Re-fetching 5y for full coverage at entry dates...")
    bars = fetch_bars_bulk(tickers, period="5y")
    print(f"  got bars for {len(bars)} tickers")

    features: list[PutFeatures] = []
    skipped = []
    for t in sorted(puts, key=lambda x: x.entry_date):
        df = bars.get(t.ticker)
        if df is None or df.empty:
            skipped.append(f"{t.ticker} {t.entry_date}: no bars")
            continue
        f = _compute_features(t, df)
        if f is None:
            skipped.append(f"{t.ticker} {t.entry_date}: insufficient history")
            continue
        features.append(f)

    if not features:
        print("No features could be computed — check yfinance coverage.")
        return

    out: list[str] = []
    out.append("REVERSE-ENGINEERING USER'S 2025 WINNING PUTS")
    out.append(f"  paired puts:  {len(puts)}")
    out.append(f"  with features: {len(features)}   (skipped: {len(skipped)})")
    if skipped:
        for s in skipped:
            out.append(f"    skip {s}")
    out.append("")

    # ---- per-trade table ----
    out.append("PER-PUT-TRADE FEATURE TABLE")
    cols = ("entry", "tkr", "W/L", "ret%", "hold", "z", "z-prev",
            "zmax20", "dz5", "ret5d", "ret10d", "ret20d", "rsi", "%K",
            "rv", "vol/avg", "dSMA20", "dSMA50", "gap%", "body%",
            "max_drop", "days2drop")
    out.append("  " + " ".join(f"{c:>8s}" for c in cols))
    for f in features:
        out.append("  " + " ".join((
            f"{str(f.t.entry_date):>10s}",
            f"{f.t.ticker:>5s}",
            f"{'W' if f.is_win else 'L':>3s}",
            f"{f.t.ret_pct * 100:>+7.1f}",
            f"{f.t.days_held:>5d}d",
            f"{f.z_log:>+8.2f}",
            f"{f.z_log_prev:>+8.2f}",
            f"{f.z_max_20:>+8.2f}",
            f"{f.z_slope_5d:>+8.3f}",
            f"{f.ret_5d:>+8.1f}",
            f"{f.ret_10d:>+8.1f}",
            f"{f.ret_20d:>+8.1f}",
            f"{f.rsi_14:>8.1f}",
            f"{f.stoch_k:>8.1f}",
            f"{f.rv_20:>8.1f}",
            f"{f.vol_spike:>8.2f}",
            f"{f.dist_sma20:>+8.1f}",
            f"{f.dist_sma50:>+8.1f}",
            f"{f.gap_pct:>+8.2f}",
            f"{f.body_pct:>+8.2f}",
            f"{f.max_drop_pct:>+8.1f}",
            f"{f.days_to_drop:>8d}",
        )))
    out.append("")

    # ---- aggregate: wins vs losses median comparison ----
    wins = [f for f in features if f.is_win]
    losses = [f for f in features if not f.is_win]
    feature_names = [
        ("z_log", lambda f: f.z_log),
        ("z_log_prev", lambda f: f.z_log_prev),
        ("z_max_20", lambda f: f.z_max_20),
        ("days_since_zmax", lambda f: float(f.days_since_zmax) if f.days_since_zmax >= 0 else float("nan")),
        ("z_slope_5d", lambda f: f.z_slope_5d),
        ("ret_5d", lambda f: f.ret_5d),
        ("ret_10d", lambda f: f.ret_10d),
        ("ret_20d", lambda f: f.ret_20d),
        ("rsi_14", lambda f: f.rsi_14),
        ("stoch_k", lambda f: f.stoch_k),
        ("rv_20", lambda f: f.rv_20),
        ("vol_spike", lambda f: f.vol_spike),
        ("dist_sma20", lambda f: f.dist_sma20),
        ("dist_sma50", lambda f: f.dist_sma50),
        ("gap_pct", lambda f: f.gap_pct),
        ("body_pct", lambda f: f.body_pct),
        ("max_drop_pct", lambda f: f.max_drop_pct),
    ]

    out.append("FEATURE COMPARISON  (wins vs losses, median + p25/p75)")
    hdr = f"  {'feature':>18s}  {'WIN med':>10s} {'WIN p25':>10s} {'WIN p75':>10s}  | "
    hdr += f"{'LOSS med':>10s} {'LOSS p25':>10s} {'LOSS p75':>10s}  | {'spread':>8s}"
    out.append(hdr)
    discriminators: list[tuple[str, float]] = []
    for name, fn in feature_names:
        w_vals = [fn(f) for f in wins]
        l_vals = [fn(f) for f in losses]
        w_med = median([x for x in w_vals if not math.isnan(x)]) if any(not math.isnan(x) for x in w_vals) else float("nan")
        l_med = median([x for x in l_vals if not math.isnan(x)]) if any(not math.isnan(x) for x in l_vals) else float("nan")
        spread = (w_med - l_med) if not (math.isnan(w_med) or math.isnan(l_med)) else float("nan")
        out.append(
            f"  {name:>18s}  "
            f"{_fmt(w_med, 2)} {_fmt(_pctile(w_vals, 25), 2)} {_fmt(_pctile(w_vals, 75), 2)}  | "
            f"{_fmt(l_med, 2)} {_fmt(_pctile(l_vals, 25), 2)} {_fmt(_pctile(l_vals, 75), 2)}  | "
            f"{_fmt(spread, 2)}"
        )
        if not math.isnan(spread):
            discriminators.append((name, abs(spread)))
    out.append("")

    # Rank discriminators by absolute median spread.
    discriminators.sort(key=lambda x: -x[1])
    out.append("FEATURES RANKED BY ABS MEDIAN SPREAD (wins vs losses)")
    for name, sp in discriminators[:8]:
        out.append(f"  {name:>18s}   |spread| = {sp:.2f}")
    out.append("")

    # ---- best single-feature classifier ----
    # For each feature, find a threshold that maximizes (win-rate above
    # threshold) - (win-rate below threshold). Crude but useful filter test.
    out.append("BEST SINGLE-FEATURE THRESHOLDS")
    out.append("  (try gating future puts on these — does it separate wins from losses?)")
    out.append(f"  {'feature':>18s}  {'threshold':>10s}  {'side':>6s}  "
               f"{'wins/n above':>14s}  {'wins/n below':>14s}  {'lift':>8s}")
    for name, fn in feature_names:
        rows = sorted([(fn(f), f.is_win) for f in features
                       if not math.isnan(fn(f))], key=lambda r: r[0])
        if len(rows) < 6:
            continue
        best = None
        for i in range(2, len(rows) - 2):
            thr = (rows[i - 1][0] + rows[i][0]) / 2.0
            below = rows[:i]
            above = rows[i:]
            wb = sum(1 for _, w in below if w) / len(below) if below else 0
            wa = sum(1 for _, w in above if w) / len(above) if above else 0
            lift = wa - wb
            for side, ww, nn, other_ww, other_nn in (
                ("above", wa, len(above), wb, len(below)),
                ("below", wb, len(below), wa, len(above)),
            ):
                # We want the GATED group to have higher WR than the
                # excluded group.
                if ww > other_ww and ww >= 0.75 and nn >= 3:
                    cand = (name, thr, side, ww, nn, other_ww, other_nn,
                            ww - other_ww)
                    if best is None or cand[7] > best[7]:
                        best = cand
        if best is not None:
            _, thr, side, ww, nn, ow, on, lift = best
            out.append(f"  {name:>18s}  {thr:>+10.2f}  {side:>6s}  "
                       f"{f'{int(ww * nn)}/{nn} ({ww * 100:.0f}%)':>14s}  "
                       f"{f'{int(ow * on)}/{on} ({ow * 100:.0f}%)':>14s}  "
                       f"{lift * 100:>+7.0f}%")
    out.append("")

    # ---- winners' price-action narrative ----
    out.append("WINNERS' STORY")
    out.append("  median features:")
    if wins:
        out.append(f"    z_log at entry         : {median([f.z_log for f in wins]):+.2f}")
        out.append(f"    z_log 20bar high       : {median([f.z_max_20 for f in wins]):+.2f}")
        out.append(f"    days since zmax        : {median([f.days_since_zmax for f in wins if f.days_since_zmax >= 0]):.0f}")
        out.append(f"    5d return into entry   : {median([f.ret_5d for f in wins if not math.isnan(f.ret_5d)]):+.1f}%")
        out.append(f"    20d return into entry  : {median([f.ret_20d for f in wins if not math.isnan(f.ret_20d)]):+.1f}%")
        out.append(f"    RSI(14) at entry       : {median([f.rsi_14 for f in wins if not math.isnan(f.rsi_14)]):.1f}")
        out.append(f"    realized vol at entry  : {median([f.rv_20 for f in wins if not math.isnan(f.rv_20)]):.1f}%")
        out.append(f"    distance from SMA20    : {median([f.dist_sma20 for f in wins if not math.isnan(f.dist_sma20)]):+.1f}%")
    out.append("  median outcomes:")
    if wins:
        out.append(f"    max drop in hold window: {median([f.max_drop_pct for f in wins if not math.isnan(f.max_drop_pct)]):+.1f}%")
        out.append(f"    days to that drop      : {median([f.days_to_drop for f in wins if f.days_to_drop > 0]):.0f}")
    out.append("")

    text = "\n".join(out)
    print(text)
    out_txt = os.path.join(OUT_DIR, "personal_puts_pattern_analysis.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  -> {out_txt}")


if __name__ == "__main__":
    main()
