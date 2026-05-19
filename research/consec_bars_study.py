"""Research: does a consecutive-down-bars filter improve the -2sigma entry?

Companion to capitulation_backtest.py. It reuses the *exact* event
definition from that script -- rolling 252d OLS linear channel on raw
closes, z = (close - c252)/sig252, first touch into z <= -2.0, 60-day
forward close path -- so this is the same event set, just re-partitioned.

Instead of bucketing by the 63/252 gap regime, we bucket each -2sigma
first-touch event by the length of the consecutive DOWN-bar run ending at
(and including) the touch bar, with "down" = close < prior close (the
Consecutive Up/Down Bars indicator's default definition):

  0-1 down days   (touch with little/no preceding down-streak)
  2-3 down days
  4-5 down days
  6+  down days    (deep capitulation streak)

For each bucket we compare PnL shape, not just a point estimate:
  - mean cumulative return path with SE bands
  - win rate by hold day
  - intra-trade max adverse excursion (MAE) distribution
  - time-to-peak distribution

Run:
    python research/consec_bars_study.py
Outputs:
    research/out/consec_bars_study.png
    research/out/consec_bars_study.txt
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.sp500_tickers import SP500_TICKERS  # noqa: E402

LONG_WINDOW = 252
FORWARD_DAYS = 60
SIGMA_THRESHOLD = -2.0
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# (name, predicate on down-run length at touch, color)
BUCKETS = [
    ("0-1 down days", lambda d: d <= 1,          "#7f7f7f"),
    ("2-3 down days", lambda d: 2 <= d <= 3,     "#1f77b4"),
    ("4-5 down days", lambda d: 4 <= d <= 5,     "#ff7f0e"),
    ("6+ down days",  lambda d: d >= 6,          "#2ca02c"),
]


def rolling_ols_components(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Endpoint of the fit and residual sigma at each bar (no lookahead).

    Identical math to capitulation_backtest.py so the event set matches.
    """
    n = len(y)
    endpoint = np.full(n, np.nan)
    sigmas = np.full(n, np.nan)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = float(np.sum(x_centered ** 2))
    for t in range(window - 1, n):
        chunk = y[t - window + 1 : t + 1]
        y_mean = chunk.mean()
        slope = float(np.sum(x_centered * (chunk - y_mean)) / denom)
        intercept = y_mean - slope * x_mean
        fit = slope * x + intercept
        endpoint[t] = fit[-1]
        sigmas[t] = float(np.std(chunk - fit, ddof=1))
    return endpoint, sigmas


def down_run_lengths(closes: np.ndarray) -> np.ndarray:
    """Length of the consecutive down-bar run ending at each bar.

    down bar = close < prior close. A flat or up bar resets the run to 0.
    Bar 0 has no prior bar -> 0.
    """
    n = len(closes)
    run = np.zeros(n, dtype=int)
    for i in range(1, n):
        run[i] = run[i - 1] + 1 if closes[i] < closes[i - 1] else 0
    return run


def collect_events(ticker: str, df: pd.DataFrame) -> list[dict]:
    if len(df) < LONG_WINDOW + FORWARD_DAYS + 5:
        return []
    closes = df["close"].to_numpy(dtype=float)
    c252, sig252 = rolling_ols_components(closes, LONG_WINDOW)
    z = (closes - c252) / sig252
    dr = down_run_lengths(closes)
    out: list[dict] = []
    prev_z = np.nan
    n = len(closes)
    for t in range(n):
        if np.isnan(z[t]) or np.isnan(sig252[t]):
            prev_z = z[t]
            continue
        first_touch = z[t] <= SIGMA_THRESHOLD and (np.isnan(prev_z) or prev_z > SIGMA_THRESHOLD)
        prev_z = z[t]
        if not first_touch:
            continue
        if t + FORWARD_DAYS >= n:
            continue
        path = closes[t : t + FORWARD_DAYS + 1] / closes[t] - 1.0  # starts at 0
        out.append({"ticker": ticker, "down_run": int(dr[t]), "path": path})
    return out


def partition(events: list[dict]) -> dict[str, np.ndarray]:
    by_bucket: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in BUCKETS}
    for ev in events:
        d = ev["down_run"]
        for name, predicate, _ in BUCKETS:
            if predicate(d):
                by_bucket[name].append(ev["path"])
                break
    return {k: np.vstack(v) if v else np.empty((0, FORWARD_DAYS + 1)) for k, v in by_bucket.items()}


def plot_results(paths_by_bucket: dict[str, np.ndarray], out_path: str) -> dict:
    days = np.arange(FORWARD_DAYS + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax_path, ax_winrate, ax_mae, ax_ttp = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    summary: dict[str, dict] = {}

    for name, _, color in BUCKETS:
        paths = paths_by_bucket[name]
        if paths.size == 0:
            continue
        n = paths.shape[0]
        mean = paths.mean(axis=0)
        sem = paths.std(axis=0, ddof=1) / np.sqrt(n)

        ax_path.plot(days, mean * 100, color=color, linewidth=2, label=f"{name} (n={n})")
        ax_path.fill_between(days, (mean - sem) * 100, (mean + sem) * 100,
                             color=color, alpha=0.18)

        win_by_day = (paths[:, 1:] > 0).mean(axis=0)
        ax_winrate.plot(days[1:], win_by_day * 100, color=color, linewidth=2,
                        label=f"{name} (n={n})")

        mae = paths[:, 1:].min(axis=1)
        mfe = paths[:, 1:].max(axis=1)
        ttp = paths[:, 1:].argmax(axis=1) + 1

        summary[name] = {
            "n": int(n),
            "ret_20d_mean": float(paths[:, 20].mean()),
            "ret_60d_mean": float(paths[:, -1].mean()),
            "win_20d": float((paths[:, 20] > 0).mean()),
            "win_60d": float((paths[:, -1] > 0).mean()),
            "mae_mean": float(mae.mean()),
            "mae_median": float(np.median(mae)),
            "mae_p10": float(np.quantile(mae, 0.10)),
            "mfe_mean": float(mfe.mean()),
            "mfe_median": float(np.median(mfe)),
            "ttp_median": int(np.median(ttp)),
            "ttp_mean": float(ttp.mean()),
        }

        ax_mae.hist(mae * 100, bins=40, alpha=0.45, color=color, label=name, density=True)
        ax_ttp.hist(ttp, bins=range(1, FORWARD_DAYS + 2, 2), alpha=0.45, color=color,
                    label=name, density=True)

    ax_path.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
    ax_path.set_xlabel("days since -2sigma touch")
    ax_path.set_ylabel("mean cumulative return (%)")
    ax_path.set_title("PnL shape by down-streak at entry (mean +/- SE)")
    ax_path.legend(loc="lower right")
    ax_path.grid(True, alpha=0.3)

    ax_winrate.axhline(50, color="gray", linewidth=0.8, alpha=0.6, linestyle="--")
    ax_winrate.set_xlabel("days held")
    ax_winrate.set_ylabel("win rate (%)  (cumret > 0)")
    ax_winrate.set_title("Win rate vs. hold horizon")
    ax_winrate.legend(loc="lower right")
    ax_winrate.grid(True, alpha=0.3)

    ax_mae.set_xlabel("intra-trade MAE (worst cumret within 60 days, %)")
    ax_mae.set_ylabel("density")
    ax_mae.set_title("Max adverse excursion -- how deep does it dig?")
    ax_mae.legend(loc="upper left")
    ax_mae.grid(True, alpha=0.3)

    ax_ttp.set_xlabel("day of peak cumret (1..60)")
    ax_ttp.set_ylabel("density")
    ax_ttp.set_title("Time-to-peak -- when does the bounce play out?")
    ax_ttp.legend(loc="upper right")
    ax_ttp.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return summary


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Fetching {len(SP500_TICKERS)} tickers x 5y...")
    bars = fetch_bars_bulk(SP500_TICKERS, period="5y")
    print(f"  got {len(bars)} usable tickers")

    all_events: list[dict] = []
    for ticker, df in bars.items():
        all_events.extend(collect_events(ticker, df))
    print(f"  {len(all_events)} -2sigma first-touch events with 60d forward path")

    # Raw down-run distribution across all touches (context for the buckets).
    runs = np.array([e["down_run"] for e in all_events], dtype=int) if all_events else np.array([], dtype=int)

    by_bucket = partition(all_events)
    for name, _, _ in BUCKETS:
        print(f"  {name}: n={by_bucket[name].shape[0]}")

    out_png = os.path.join(OUT_DIR, "consec_bars_study.png")
    out_txt = os.path.join(OUT_DIR, "consec_bars_study.txt")
    summary = plot_results(by_bucket, out_png)

    lines = []
    lines.append(f"Total -2sigma first-touch events: {len(all_events)}")
    lines.append(f"Forward horizon: {FORWARD_DAYS} trading days")
    lines.append('Down bar = close < prior close; run = consecutive down bars ending at the touch.\n')
    if runs.size:
        lines.append("Down-run length distribution at touch:")
        for k in range(0, 8):
            c = int((runs == k).sum())
            lines.append(f"  ={k}: {c:5d}  ({c / runs.size * 100:4.1f}%)")
        c = int((runs >= 8).sum())
        lines.append(f"  >=8: {c:5d}  ({c / runs.size * 100:4.1f}%)")
        lines.append("")
    for name, _, _ in BUCKETS:
        if name not in summary:
            continue
        s = summary[name]
        lines.append(f"[{name}]  n={s['n']}")
        lines.append(f"  mean 20d ret: {s['ret_20d_mean']*100:+.2f}%  (win {s['win_20d']*100:.1f}%)   "
                     f"mean 60d ret: {s['ret_60d_mean']*100:+.2f}%  (win {s['win_60d']*100:.1f}%)")
        lines.append(f"  MAE  mean: {s['mae_mean']*100:+.2f}%   "
                     f"median: {s['mae_median']*100:+.2f}%   "
                     f"p10: {s['mae_p10']*100:+.2f}%")
        lines.append(f"  MFE  mean: {s['mfe_mean']*100:+.2f}%   "
                     f"median: {s['mfe_median']*100:+.2f}%")
        lines.append(f"  Time-to-peak  mean: {s['ttp_mean']:.1f}d   median: {s['ttp_median']}d")
        lines.append("")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
