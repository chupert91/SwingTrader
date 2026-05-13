"""Third-pass sweep: does a regression-slope trend filter help?

Compares: filter off vs. filter on (sign only) vs. filter on with min |trend|.
All other parameters locked at the previous optimum.
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

CACHE_PATH = Path(__file__).resolve().parent / ".opt_trend_cache.pkl"

TICKERS = ["TSLA", "AMD", "NVDA", "PLTR", "IONQ", "MP"]
PERIOD = "5y"

BASE = backtest.BacktestConfig(
    long_enabled=True,
    short_enabled=True,
    leverage=5.0,
    stop_loss_pct=10.0,
    profit_target_pct=20.0,
    starting_capital=10_000.0,
    allocation_pct=25.0,
    min_confidence=0,
    long_entry_sigma=-2.0,
    short_entry_sigma=2.0,
    time_stop_bars=10,
)

# Trend filter variants: (require_alignment, min_trend_pct annualized)
TREND_VARIANTS = [
    (False, 0.0),     # baseline: no trend filter
    (True, 0.0),      # sign only
    (True, 10.0),     # at least 10%/yr trend in trade direction
    (True, 20.0),     # at least 20%/yr trend
    (True, 30.0),     # at least 30%/yr trend (strong trend only)
]

# Also explore: long-only with uptrend, short-only with downtrend, both
SIDE_VARIANTS = [
    ("both", True, True),
    ("long_only", True, False),
    ("short_only", False, True),
]

MIN_TRADES = 5


def main() -> None:
    print(f"Trend-filter sweep across {len(TICKERS)} tickers, lookback={PERIOD}")
    print(f"Locked: alloc=25%, leverage=5x, stop=10% opt, target=20% opt, "
          f"sigma=-2.0, conf=0, tstop=10\n")

    prepared: dict[str, pd.DataFrame] = {}
    for tk in TICKERS:
        raw = fetch_bars(tk, period=PERIOD)
        if raw.empty:
            continue
        prepared[tk] = backtest.prepare(raw, window=BASE.window)
        print(f"  {tk:5s}: {len(prepared[tk])} bars")
    print()

    combos = list(itertools.product(TREND_VARIANTS, SIDE_VARIANTS))

    cache_key = (tuple(sorted(prepared.keys())), PERIOD,
                 str(TREND_VARIANTS), str(SIDE_VARIANTS))
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
        for ((require_align, min_tr), (side_lbl, long_en, short_en)) in combos:
            cfg = dataclasses.replace(
                BASE,
                long_enabled=long_en,
                short_enabled=short_en,
                require_trend_alignment=require_align,
                min_trend_pct=min_tr,
            )
            per_ticker = {tk: backtest._simulate_prepared(df, cfg)["stats"]
                          for tk, df in prepared.items()}
            rows.append({
                "trend_filter": f"sign+>{min_tr:.0f}%/yr" if require_align and min_tr > 0
                                else ("sign_only" if require_align else "off"),
                "min_trend_pct": min_tr,
                "side": side_lbl,
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
        if len(sharpes) < 2:  # relax: short_only may have fewer qualifying tickers
            continue
        scored.append({
            "trend_filter": r["trend_filter"],
            "min_trend_pct": r["min_trend_pct"],
            "side": r["side"],
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
    print("ALL CONFIGS BY MEAN SHARPE (trend filter x sides)")
    print("=" * 120)
    header = (f"{'rank':>4} {'side':>10} {'trend_filter':>16} {'mean_Sh':>8} "
              f"{'med_Sh':>7} {'worst':>7} {'tot_ret%':>9} {'trades':>7} "
              f"{'win%':>6} {'maxDD%':>7} {'n_tk':>4}")
    print(header)
    print("-" * 120)
    for i, r in enumerate(scored, 1):
        print(f"{i:>4} {r['side']:>10} {r['trend_filter']:>16} "
              f"{r['mean_sharpe']:>8.3f} {r['median_sharpe']:>7.3f} "
              f"{r['worst_sharpe']:>7.3f} {r['mean_total_return']:>9.1f} "
              f"{r['mean_trades']:>7.1f} {r['mean_winrate']:>6.1f} "
              f"{r['mean_maxdd']:>7.1f} {r['tickers_qualified']:>4d}")

    if scored:
        best = scored[0]
        print()
        print("=" * 120)
        print(f"PER-TICKER DETAIL -- BEST: side={best['side']}, filter={best['trend_filter']}")
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


if __name__ == "__main__":
    main()
