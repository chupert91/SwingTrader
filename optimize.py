"""Find the best regression-channel parameter set across a basket of tickers.

Risk frame is FIXED to the user's rules:
    - 25% of equity per trade
    - 2% underlying stop  -> 10% option P&L stop @ 5x leverage
    - 20% option P&L profit target
    - 5x option leverage

Variables explored: entry sigma, min_confidence (indicator confirmation),
time_stop_bars, and optional trailing-stop variants.

Sharpe is averaged across tickers (each ticker contributes one sample). The
top combos by mean Sharpe are printed, along with per-ticker breakdowns.
"""
from __future__ import annotations

import dataclasses
import itertools
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Force stdout to UTF-8 so we can print sigma chars on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CACHE_PATH = Path(__file__).resolve().parent / ".opt_cache.pkl"

from backend import backtest
from backend.data import fetch_bars

TICKERS = ["TSLA", "AMD", "NVDA", "PLTR", "IONQ", "MP"]
PERIOD = "5y"

# Locked risk frame
BASE = backtest.BacktestConfig(
    long_enabled=True,
    short_enabled=True,
    leverage=5.0,
    stop_loss_pct=10.0,          # 2% underlying x 5x leverage
    profit_target_pct=20.0,
    starting_capital=10_000.0,
    allocation_pct=25.0,
)

# Search grid — keep modest so we can iterate
GRID = {
    "long_entry_sigma": [-1.5, -2.0, -2.5, -3.0],
    "min_confidence": [0, 1, 2, 3],
    "time_stop_bars": [10, 14, 20, 30],
    # Trail variants: (activation_pct, distance_pct) on UNDERLYING %.
    # None = no trail. Activation 2% with distance 1% lets winners breathe
    # but cuts them once they roll over.
    "trail": [
        (None, None),
        (2.0, 1.0),
        (3.0, 1.5),
    ],
}

MIN_TRADES = 5  # don't trust configs that barely trade


def main() -> None:
    print(f"Optimizing across {len(TICKERS)} tickers, lookback={PERIOD}")
    print(f"Locked: alloc={BASE.allocation_pct}%, stop={BASE.stop_loss_pct}% opt "
          f"(~2% underlying @ {BASE.leverage}x), target={BASE.profit_target_pct}% opt")
    print()

    # Pre-fetch + prepare each ticker once
    prepared: dict[str, pd.DataFrame] = {}
    for tk in TICKERS:
        raw = fetch_bars(tk, period=PERIOD)
        if raw.empty:
            print(f"  !! {tk}: no data, skipping")
            continue
        df = backtest.prepare(raw, window=BASE.window)
        prepared[tk] = df
        bars = len(df)
        valid = df["sd_position"].notna().sum()
        print(f"  {tk:5s}: {bars} bars, {valid} with valid sd_position")
    print()

    combos = list(itertools.product(
        GRID["long_entry_sigma"],
        GRID["min_confidence"],
        GRID["time_stop_bars"],
        GRID["trail"],
    ))
    cache_key = (tuple(sorted(prepared.keys())), PERIOD, str(GRID))
    rows: list[dict] | None = None
    if CACHE_PATH.exists():
        try:
            payload = pickle.loads(CACHE_PATH.read_bytes())
            if payload.get("key") == cache_key:
                rows = payload["rows"]
                print(f"Loaded {len(rows)} combos from cache ({CACHE_PATH.name})\n")
        except (pickle.UnpicklingError, KeyError, EOFError):
            pass

    if rows is None:
        print(f"Evaluating {len(combos)} combos x {len(prepared)} tickers = "
              f"{len(combos) * len(prepared)} sims")
        t0 = time.time()

        rows = []
        for (sig, conf, tstop, (trail_act, trail_dist)) in combos:
            cfg = dataclasses.replace(
                BASE,
                long_entry_sigma=sig,
                short_entry_sigma=-sig,
                min_confidence=conf,
                time_stop_bars=tstop,
                trail_activation_pct=trail_act,
                trail_distance_pct=trail_dist,
            )
            per_ticker: dict[str, dict] = {}
            for tk, df in prepared.items():
                sim = backtest._simulate_prepared(df, cfg)
                per_ticker[tk] = sim["stats"]
            rows.append({
                "entry_sigma": sig,
                "min_conf": conf,
                "time_stop": tstop,
                "trail": f"{trail_act}/{trail_dist}" if trail_act else "off",
                "per_ticker": per_ticker,
            })

        dt = time.time() - t0
        print(f"Done in {dt:.1f}s\n")
        CACHE_PATH.write_bytes(pickle.dumps({"key": cache_key, "rows": rows}))

    # Score each combo: mean Sharpe across tickers, but only count tickers
    # with >= MIN_TRADES (else they're not really tested).
    scored: list[dict] = []
    for r in rows:
        sharpes: list[float] = []
        total_returns: list[float] = []
        trade_counts: list[int] = []
        win_rates: list[float] = []
        max_dds: list[float] = []
        skipped = 0
        for tk, stats in r["per_ticker"].items():
            if stats["trade_count"] < MIN_TRADES:
                skipped += 1
                continue
            sh = stats["sharpe"]
            if np.isfinite(sh):
                sharpes.append(sh)
                total_returns.append(stats["total_return_pct"])
                trade_counts.append(stats["trade_count"])
                win_rates.append(stats["win_rate_pct"])
                max_dds.append(stats["max_drawdown_pct"])
        if len(sharpes) < 3:  # require at least half the tickers to qualify
            continue
        scored.append({
            **{k: r[k] for k in ("entry_sigma", "min_conf", "time_stop", "trail")},
            "mean_sharpe": float(np.mean(sharpes)),
            "median_sharpe": float(np.median(sharpes)),
            "worst_sharpe": float(np.min(sharpes)),
            "mean_total_return": float(np.mean(total_returns)),
            "mean_trades": float(np.mean(trade_counts)),
            "mean_winrate": float(np.mean(win_rates)),
            "mean_maxdd": float(np.mean(max_dds)),
            "tickers_qualified": len(sharpes),
            "per_ticker": r["per_ticker"],
        })

    scored.sort(key=lambda r: r["mean_sharpe"], reverse=True)

    print("=" * 110)
    print("TOP 15 CONFIGS BY MEAN SHARPE ACROSS TICKERS")
    print("=" * 110)
    header = (f"{'rank':>4} {'sig_in':>6} {'conf':>4} {'tstop':>5} {'trail':>8} "
              f"{'mean_Sh':>8} {'med_Sh':>7} {'worst':>7} "
              f"{'tot_ret%':>9} {'trades':>7} {'win%':>6} {'maxDD%':>7} {'n_tk':>4}")
    print(header)
    print("-" * 110)
    for i, r in enumerate(scored[:15], 1):
        print(f"{i:>4} {r['entry_sigma']:>6.1f} {r['min_conf']:>4d} {r['time_stop']:>5d} "
              f"{r['trail']:>8} {r['mean_sharpe']:>8.3f} {r['median_sharpe']:>7.3f} "
              f"{r['worst_sharpe']:>7.3f} {r['mean_total_return']:>9.1f} "
              f"{r['mean_trades']:>7.1f} {r['mean_winrate']:>6.1f} "
              f"{r['mean_maxdd']:>7.1f} {r['tickers_qualified']:>4d}")

    # Show per-ticker breakdown of the #1 config
    if scored:
        best = scored[0]
        print()
        print("=" * 110)
        print(f"PER-TICKER DETAIL -- BEST CONFIG: sigma={best['entry_sigma']}, "
              f"conf={best['min_conf']}, tstop={best['time_stop']}, trail={best['trail']}")
        print("=" * 110)
        ph = (f"{'ticker':>7} {'sharpe':>7} {'tot_ret%':>9} {'B&H%':>7} "
              f"{'trades':>7} {'win%':>6} {'avg_W%':>7} {'avg_L%':>7} {'maxDD%':>7}")
        print(ph)
        print("-" * 110)
        for tk, s in best["per_ticker"].items():
            print(f"{tk:>7} {s['sharpe']:>7.3f} {s['total_return_pct']:>9.1f} "
                  f"{(s['buy_and_hold_pct'] or 0):>7.1f} "
                  f"{s['trade_count']:>7d} {s['win_rate_pct']:>6.1f} "
                  f"{s['avg_win_pct']:>7.1f} {s['avg_loss_pct']:>7.1f} "
                  f"{s['max_drawdown_pct']:>7.1f}")

        # Exit reasons for the best config across all tickers
        all_reasons: dict[str, int] = {}
        for tk, s in best["per_ticker"].items():
            for reason, n in s.get("exit_reason_counts", {}).items():
                all_reasons[reason] = all_reasons.get(reason, 0) + n
        print()
        print("Exit reasons (best config, all tickers combined):")
        for reason, n in sorted(all_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:>10s}: {n}")


if __name__ == "__main__":
    main()
