"""Second-pass sweep: vary the stop to see where the 2% is costing us.

Locks: 25% alloc, 5x leverage, 20% option target, min_confidence=0
        (already proven dominant in pass 1).

Sweeps: stop_loss_pct (10-30% option = 2-6% underlying @ 5x),
        long_entry_sigma, time_stop_bars, trail variants.
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

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend import backtest
from backend.data import fetch_bars

CACHE_PATH = Path(__file__).resolve().parent / ".opt_stops_cache.pkl"

TICKERS = ["TSLA", "AMD", "NVDA", "PLTR", "IONQ", "MP"]
PERIOD = "5y"

BASE = backtest.BacktestConfig(
    long_enabled=True,
    short_enabled=True,
    leverage=5.0,
    profit_target_pct=20.0,
    starting_capital=10_000.0,
    allocation_pct=25.0,
    min_confidence=0,
)

GRID = {
    # 10% option = 2% underlying, 30% option = 6% underlying (@ 5x)
    "stop_loss_pct": [10.0, 15.0, 20.0, 25.0, 30.0],
    "long_entry_sigma": [-1.5, -2.0, -2.5, -3.0],
    "time_stop_bars": [10, 20],
    "trail": [
        (None, None),
        (3.0, 1.5),
        (5.0, 2.0),
    ],
}

MIN_TRADES = 5


def main() -> None:
    print(f"Stop-loss sweep across {len(TICKERS)} tickers, lookback={PERIOD}")
    print(f"Locked: alloc=25%, leverage=5x, target=20% opt, conf=0")
    print()

    prepared: dict[str, pd.DataFrame] = {}
    for tk in TICKERS:
        raw = fetch_bars(tk, period=PERIOD)
        if raw.empty:
            print(f"  !! {tk}: no data, skipping")
            continue
        df = backtest.prepare(raw, window=BASE.window)
        prepared[tk] = df
        print(f"  {tk:5s}: {len(df)} bars")
    print()

    combos = list(itertools.product(
        GRID["stop_loss_pct"],
        GRID["long_entry_sigma"],
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
                print(f"Loaded {len(rows)} combos from cache\n")
        except (pickle.UnpicklingError, KeyError, EOFError):
            pass

    if rows is None:
        print(f"Evaluating {len(combos)} combos x {len(prepared)} tickers = "
              f"{len(combos) * len(prepared)} sims")
        t0 = time.time()
        rows = []
        for (stop, sig, tstop, (trail_act, trail_dist)) in combos:
            cfg = dataclasses.replace(
                BASE,
                stop_loss_pct=stop,
                long_entry_sigma=sig,
                short_entry_sigma=-sig,
                time_stop_bars=tstop,
                trail_activation_pct=trail_act,
                trail_distance_pct=trail_dist,
            )
            per_ticker = {tk: backtest._simulate_prepared(df, cfg)["stats"]
                          for tk, df in prepared.items()}
            rows.append({
                "stop_opt": stop,
                "stop_underlying": stop / 5.0,
                "entry_sigma": sig,
                "time_stop": tstop,
                "trail": f"{trail_act}/{trail_dist}" if trail_act else "off",
                "per_ticker": per_ticker,
            })
        print(f"Done in {time.time() - t0:.1f}s\n")
        CACHE_PATH.write_bytes(pickle.dumps({"key": cache_key, "rows": rows}))

    scored: list[dict] = []
    for r in rows:
        sharpes, total_returns, trade_counts, win_rates, max_dds = [], [], [], [], []
        for tk, stats in r["per_ticker"].items():
            if stats["trade_count"] < MIN_TRADES:
                continue
            sh = stats["sharpe"]
            if np.isfinite(sh):
                sharpes.append(sh)
                total_returns.append(stats["total_return_pct"])
                trade_counts.append(stats["trade_count"])
                win_rates.append(stats["win_rate_pct"])
                max_dds.append(stats["max_drawdown_pct"])
        if len(sharpes) < 3:
            continue
        scored.append({
            **{k: r[k] for k in ("stop_opt", "stop_underlying", "entry_sigma",
                                  "time_stop", "trail")},
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

    print("=" * 120)
    print("TOP 15 CONFIGS BY MEAN SHARPE")
    print("=" * 120)
    header = (f"{'rank':>4} {'stop_o%':>7} {'stop_u%':>7} {'sig_in':>6} {'tstop':>5} "
              f"{'trail':>8} {'mean_Sh':>8} {'med_Sh':>7} {'worst':>7} "
              f"{'tot_ret%':>9} {'trades':>7} {'win%':>6} {'maxDD%':>7} {'n_tk':>4}")
    print(header)
    print("-" * 120)
    for i, r in enumerate(scored[:15], 1):
        print(f"{i:>4} {r['stop_opt']:>7.0f} {r['stop_underlying']:>7.1f} "
              f"{r['entry_sigma']:>6.1f} {r['time_stop']:>5d} {r['trail']:>8} "
              f"{r['mean_sharpe']:>8.3f} {r['median_sharpe']:>7.3f} "
              f"{r['worst_sharpe']:>7.3f} {r['mean_total_return']:>9.1f} "
              f"{r['mean_trades']:>7.1f} {r['mean_winrate']:>6.1f} "
              f"{r['mean_maxdd']:>7.1f} {r['tickers_qualified']:>4d}")

    if scored:
        best = scored[0]
        print()
        print("=" * 120)
        print(f"PER-TICKER DETAIL -- BEST: stop={best['stop_opt']:.0f}% opt "
              f"({best['stop_underlying']:.1f}% underlying), sigma={best['entry_sigma']}, "
              f"tstop={best['time_stop']}, trail={best['trail']}")
        print("=" * 120)
        ph = (f"{'ticker':>7} {'sharpe':>7} {'tot_ret%':>9} {'B&H%':>8} "
              f"{'trades':>7} {'win%':>6} {'avg_W%':>7} {'avg_L%':>7} {'maxDD%':>7}")
        print(ph)
        print("-" * 120)
        for tk, s in best["per_ticker"].items():
            print(f"{tk:>7} {s['sharpe']:>7.3f} {s['total_return_pct']:>9.1f} "
                  f"{(s['buy_and_hold_pct'] or 0):>8.1f} "
                  f"{s['trade_count']:>7d} {s['win_rate_pct']:>6.1f} "
                  f"{s['avg_win_pct']:>7.1f} {s['avg_loss_pct']:>7.1f} "
                  f"{s['max_drawdown_pct']:>7.1f}")

        all_reasons: dict[str, int] = {}
        for tk, s in best["per_ticker"].items():
            for reason, n in s.get("exit_reason_counts", {}).items():
                all_reasons[reason] = all_reasons.get(reason, 0) + n
        print("\nExit reasons (best, all tickers combined):")
        for reason, n in sorted(all_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:>10s}: {n}")

        # Stop-level summary: mean Sharpe by stop level (best config per stop)
        print("\n" + "=" * 60)
        print("MEAN SHARPE BY STOP LEVEL (best subconfig at each stop)")
        print("=" * 60)
        print(f"{'stop_opt%':>10} {'stop_und%':>10} {'best_Sh':>9} {'tot_ret%':>10}")
        by_stop: dict[float, dict] = {}
        for r in scored:
            cur = by_stop.get(r["stop_opt"])
            if cur is None or r["mean_sharpe"] > cur["mean_sharpe"]:
                by_stop[r["stop_opt"]] = r
        for stop in sorted(by_stop.keys()):
            r = by_stop[stop]
            print(f"{stop:>10.0f} {r['stop_underlying']:>10.1f} "
                  f"{r['mean_sharpe']:>9.3f} {r['mean_total_return']:>10.1f}")


if __name__ == "__main__":
    main()
