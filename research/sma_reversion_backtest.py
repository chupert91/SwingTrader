"""Research: 4-down-bars-below-SMA20 entry, exit on close back above SMA20.

Strategy (exactly as specified):
  ENTRY  at the close of the bar where the consecutive DOWN-bar run first
         reaches DOWN_BARS (default 4) AND that close is below the
         SMA(SMA_PERIOD) (default 20). One position per ticker at a time.
  EXIT   at the close of the first subsequent bar whose close is back
         ABOVE the SMA; OR, once the trade has been held >= TIME_LOSS_EXIT
         bars (default 11), at the close of any bar where it is still in a
         loss (a profitable trade keeps riding to the SMA cross -- the
         stop only fires when underwater). A trade still open at the end
         of the data is closed at the last bar, reason "eod".

"down bar" = close < prior close (same definition as the Consecutive
Up/Down Bars indicator and research/consec_bars_study.py).

This is a fast equity-timing trade, NOT the -2sigma OTM-2mo structure, so
the primary metric is the underlying (stock) return. A 5x option-style
approximation (max(r*5, -1), matching backend/backtest.py's leverage
model) is reported alongside for comparability with the other studies --
treat it as an upper bound, real options have theta/delta.

A compact sensitivity table sweeps DOWN_BARS in {3,4,5}, "first reach ==N"
vs ">=N", and the below-SMA entry filter on/off, so it is clear whether
the 4th-bar and SMA conditions are each pulling weight.

Run:
    python research/sma_reversion_backtest.py
Outputs:
    research/out/sma_reversion_backtest.png
    research/out/sma_reversion_backtest.txt
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.sp500_tickers import SP500_TICKERS  # noqa: E402

DOWN_BARS = 4
SMA_PERIOD = 20
TIME_LOSS_EXIT = 11  # once held >= this many bars, bail any bar still in a loss
STOP_SWEEP = [None, 8, 11, 15, 20]  # None = no time-loss stop (SMA-cross only)
LEVERAGE = 5.0  # option-style approximation only
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


@dataclass(frozen=True)
class Variant:
    down_bars: int
    at_least: bool       # True: run >= down_bars; False: run == down_bars
    require_below_sma: bool

    def label(self) -> str:
        op = ">=" if self.at_least else "=="
        flt = "below-SMA" if self.require_below_sma else "no-SMA-filter"
        return f"run{op}{self.down_bars}, {flt}"


def down_run_lengths(closes: np.ndarray) -> np.ndarray:
    n = len(closes)
    run = np.zeros(n, dtype=int)
    for i in range(1, n):
        run[i] = run[i - 1] + 1 if closes[i] < closes[i - 1] else 0
    return run


def run_strategy(df: pd.DataFrame, v: Variant, time_loss_exit: int | None) -> list[dict]:
    """Return list of trades for one ticker under variant v.

    time_loss_exit: if set, once a trade has been held >= this many bars,
    exit at the close of any bar where it is still in a loss (a profitable
    trade keeps riding to the SMA cross). None disables the stop (A/B).
    """
    if len(df) < SMA_PERIOD + 5:
        return []
    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    dates = pd.to_datetime(df["timestamp"]).dt.date.to_numpy()
    sma = pd.Series(closes).rolling(SMA_PERIOD).mean().to_numpy()
    dr = down_run_lengths(closes)

    trades: list[dict] = []
    pos: dict | None = None
    n = len(closes)
    for t in range(n):
        if np.isnan(sma[t]):
            continue
        if pos is None:
            triggered = (dr[t] >= v.down_bars) if v.at_least else (dr[t] == v.down_bars)
            if triggered and (not v.require_below_sma or closes[t] < sma[t]):
                pos = {
                    "entry_idx": t,
                    "entry_price": closes[t],
                    "entry_date": dates[t],
                    "min_low": lows[t],
                    "max_high": highs[t],
                }
            continue
        # In a position: track excursions, exit on first close above SMA.
        pos["min_low"] = min(pos["min_low"], lows[t])
        pos["max_high"] = max(pos["max_high"], highs[t])
        is_last = t == n - 1
        held = t - pos["entry_idx"]
        ep = pos["entry_price"]
        ret = closes[t] / ep - 1.0
        crossed_up = closes[t] > sma[t]
        time_stopped = (
            not crossed_up
            and time_loss_exit is not None
            and held >= time_loss_exit
            and ret < 0.0
        )
        if crossed_up or time_stopped or is_last:
            if crossed_up:
                reason = "sma_cross"
            elif time_stopped:
                reason = "time_loss"
            else:
                reason = "eod"
            trades.append({
                "entry_date": str(pos["entry_date"]),
                "exit_date": str(dates[t]),
                "bars_held": held,
                "ret": ret,
                "opt_ret": max(ret * LEVERAGE, -1.0),
                "mae": pos["min_low"] / ep - 1.0,
                "mfe": pos["max_high"] / ep - 1.0,
                "exit_reason": reason,
            })
            pos = None
    return trades


def aggregate(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0}
    r = np.array([t["ret"] for t in trades], dtype=float)
    o = np.array([t["opt_ret"] for t in trades], dtype=float)
    mae = np.array([t["mae"] for t in trades], dtype=float)
    held = np.array([t["bars_held"] for t in trades], dtype=float)
    wins = r[r > 0]
    losses = r[r <= 0]
    gains_sum = float(r[r > 0].sum())
    loss_sum = float(-r[r < 0].sum())
    eod = sum(1 for t in trades if t["exit_reason"] == "eod")
    n_sma = sum(1 for t in trades if t["exit_reason"] == "sma_cross")
    n_tl = sum(1 for t in trades if t["exit_reason"] == "time_loss")
    tl_rets = np.array([t["ret"] for t in trades if t["exit_reason"] == "time_loss"], dtype=float)
    return {
        "n": len(trades),
        "n_sma_cross": n_sma,
        "n_time_loss": n_tl,
        "time_loss_pct": n_tl / len(trades),
        "time_loss_avg_ret": float(tl_rets.mean()) if tl_rets.size else 0.0,
        "win_rate": float((r > 0).mean()),
        "mean_ret": float(r.mean()),
        "median_ret": float(np.median(r)),
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "profit_factor": (gains_sum / loss_sum) if loss_sum > 0 else float("inf"),
        "expectancy": float(r.mean()),
        "mean_hold": float(held.mean()),
        "median_hold": float(np.median(held)),
        "mae_mean": float(mae.mean()),
        "mae_median": float(np.median(mae)),
        "mae_p10": float(np.quantile(mae, 0.10)),
        "opt_mean_ret": float(o.mean()),
        "opt_win_rate": float((o > 0).mean()),
        "eod_open_pct": eod / len(trades),
    }


def collect_all(bars: dict[str, pd.DataFrame], v: Variant,
                time_loss_exit: int | None) -> list[dict]:
    out: list[dict] = []
    for tk, df in bars.items():
        if df is None or df.empty:
            continue
        for tr in run_strategy(df, v, time_loss_exit):
            tr["ticker"] = tk
            out.append(tr)
    return out


def plot_primary(trades: list[dict], out_path: str) -> None:
    r = np.array([t["ret"] for t in trades], dtype=float) * 100
    held = np.array([t["bars_held"] for t in trades], dtype=float)
    mae = np.array([t["mae"] for t in trades], dtype=float) * 100
    by_year: dict[int, list[float]] = {}
    for t in trades:
        y = int(t["exit_date"][:4])
        by_year.setdefault(y, []).append(t["ret"] * 100)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    a0, a1, a2, a3 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    a0.hist(r, bins=60, color="#1f77b4", alpha=0.8)
    a0.axvline(0, color="gray", lw=0.8)
    a0.axvline(r.mean(), color="#d62728", lw=1.5, label=f"mean {r.mean():+.2f}%")
    a0.set_title("Per-trade return distribution (underlying %)")
    a0.set_xlabel("trade return (%)"); a0.set_ylabel("count")
    a0.legend(); a0.grid(True, alpha=0.3)

    a1.hist(held, bins=range(0, int(held.max()) + 2), color="#2ca02c", alpha=0.8)
    a1.axvline(np.median(held), color="#d62728", lw=1.5,
               label=f"median {np.median(held):.0f}d")
    a1.set_title("Holding period (bars from entry to SMA cross-up)")
    a1.set_xlabel("bars held"); a1.set_ylabel("count")
    a1.legend(); a1.grid(True, alpha=0.3)

    a2.hist(mae, bins=60, color="#ff7f0e", alpha=0.8)
    a2.axvline(np.median(mae), color="#d62728", lw=1.5,
               label=f"median {np.median(mae):+.2f}%")
    a2.set_title("Max adverse excursion (intra-trade low vs entry)")
    a2.set_xlabel("MAE (%)"); a2.set_ylabel("count")
    a2.legend(); a2.grid(True, alpha=0.3)

    years = sorted(by_year)
    means = [float(np.mean(by_year[y])) for y in years]
    a3.bar([str(y) for y in years], means, color="#9467bd", alpha=0.85)
    a3.axhline(0, color="gray", lw=0.8)
    a3.set_title("Mean trade return by exit year (%)")
    a3.set_xlabel("year"); a3.set_ylabel("mean return (%)")
    a3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Fetching {len(SP500_TICKERS)} tickers x 5y...")
    bars = fetch_bars_bulk(SP500_TICKERS, period="5y")
    print(f"  got {len(bars)} usable tickers")

    spec = Variant(down_bars=DOWN_BARS, at_least=False, require_below_sma=True)
    primary = collect_all(bars, spec, TIME_LOSS_EXIT)
    pstats = aggregate(primary)
    print(f"  primary spec ({spec.label()}, {TIME_LOSS_EXIT}d-loss stop): "
          f"{pstats.get('n', 0)} trades")

    out_png = os.path.join(OUT_DIR, "sma_reversion_backtest.png")
    out_txt = os.path.join(OUT_DIR, "sma_reversion_backtest.txt")
    if primary:
        plot_primary(primary, out_png)

    lines: list[str] = []
    lines.append(f"Strategy: enter at close of the {DOWN_BARS}th consecutive down bar "
                 f"while close < SMA{SMA_PERIOD}; exit at first close back above SMA,")
    lines.append(f"OR once held >= {TIME_LOSS_EXIT} bars exit any bar still in a loss "
                 f"(profitable trades keep riding to the SMA cross).")
    lines.append("Down bar = close < prior close. One position per ticker.\n")
    lines.append(f"[PRIMARY SPEC]  {spec.label()}, exit=SMA-cross or {TIME_LOSS_EXIT}d-loss stop")
    if pstats.get("n"):
        s = pstats
        lines.append(f"  trades: {s['n']}   win rate: {s['win_rate']*100:.1f}%   "
                     f"profit factor: {s['profit_factor']:.2f}")
        lines.append(f"  mean ret: {s['mean_ret']*100:+.2f}%   median: {s['median_ret']*100:+.2f}%   "
                     f"avg win: {s['avg_win']*100:+.2f}%   avg loss: {s['avg_loss']*100:+.2f}%")
        lines.append(f"  hold: mean {s['mean_hold']:.1f}d  median {s['median_hold']:.0f}d   "
                     f"(eod-open at data end: {s['eod_open_pct']*100:.1f}%)")
        lines.append(f"  MAE: mean {s['mae_mean']*100:+.2f}%  median {s['mae_median']*100:+.2f}%  "
                     f"p10 {s['mae_p10']*100:+.2f}%")
        lines.append(f"  exits: sma_cross {s['n_sma_cross']} | "
                     f"time_loss {s['n_time_loss']} ({s['time_loss_pct']*100:.1f}%, "
                     f"avg {s['time_loss_avg_ret']*100:+.2f}%)")
        lines.append(f"  5x option approx: mean {s['opt_mean_ret']*100:+.2f}%  "
                     f"win {s['opt_win_rate']*100:.1f}%  (upper bound, no theta)")
    else:
        lines.append("  no trades")
    lines.append("")

    # Stop sweep: same entries (primary spec), vary only the time-loss exit.
    lines.append("[STOP SWEEP]  primary entries, exit rule only")
    lines.append("  stop       | trades | win% | mean% | median% |   PF | meanHold | meanMAE% | p10MAE% | TLoss% | optMean%")
    for sd in STOP_SWEEP:
        st = aggregate(collect_all(bars, spec, sd))
        if not st.get("n"):
            continue
        pf = st["profit_factor"]
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        tag = "none" if sd is None else f"{sd}d-loss"
        tl = st.get("time_loss_pct", 0.0) * 100.0
        lines.append(f"  {tag:<10} | {st['n']:5d}  | {st['win_rate']*100:4.1f} | "
                     f"{st['mean_ret']*100:+5.2f} | {st['median_ret']*100:+6.2f} | "
                     f"{pf_s:>4} | {st['mean_hold']:6.1f}d  | {st['mae_mean']*100:+7.2f} | "
                     f"{st['mae_p10']*100:+7.2f} | {tl:5.1f} | {st['opt_mean_ret']*100:+6.2f}")
    lines.append("")
    lines.append("  Read: 'none' is the SMA-cross-only baseline. A tighter (smaller) stop")
    lines.append("  cuts the MAE tail (p10MAE less negative) but realizes more would-be")
    lines.append("  reverters as losses (win%/PF/mean fall). Best stop = least PF give-up")
    lines.append("  per unit of tail removed.")
    lines.append("")

    # Sensitivity (all with the time-loss stop applied): entry-knob weight.
    lines.append(f"[SENSITIVITY]  (all with {TIME_LOSS_EXIT}d-loss stop)  "
                 f"trades | win% | mean% | median% | PF | mean hold")
    variants = []
    for db in (3, 4, 5):
        for at_least in (False, True):
            for below in (True, False):
                variants.append(Variant(db, at_least, below))
    for v in variants:
        st = aggregate(collect_all(bars, v, TIME_LOSS_EXIT))
        if not st.get("n"):
            lines.append(f"  {v.label():<26}  (no trades)")
            continue
        pf = st["profit_factor"]
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        lines.append(f"  {v.label():<26}  {st['n']:5d} | {st['win_rate']*100:5.1f} | "
                     f"{st['mean_ret']*100:+6.2f} | {st['median_ret']*100:+6.2f} | "
                     f"{pf_s:>5} | {st['mean_hold']:5.1f}d")
    lines.append("")

    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    if primary:
        print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
