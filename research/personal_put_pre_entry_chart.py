"""Render a 252-trading-day candlestick chart of the underlying for ONE
representative winning put trade, ending on (and including) the BTO day.

Used for the user to draw their own support / Fibonacci levels on the chart,
then return it so we can analyze what levels actually matter for the trade.

Minimal overlays on purpose:
  - bare candles (green up / red down)
  - light price grid
  - vertical guide on the BTO day
  - title with the trade spec; NO indicators, NO SMAs, NO bands

Output:
    research/out/personal_put_pre_entry_<TICKER>_<entry-date>.png
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars  # noqa: E402
from research.personal_trade_audit import CSV_PATH, load_rows, pair_options  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
LOOKBACK_TD = 252


def _pick_representative_winner(puts):
    """Pick a winning put with decent hold (>=2d), strong %-return, and a
    well-known liquid ticker. Sorts by a composite score."""
    cands = [t for t in puts if t.gross_pl > 0 and t.days_held >= 2]
    if not cands:
        cands = [t for t in puts if t.gross_pl > 0]
    # Score: ret_pct + small bonus for hold-days in [2..7] (visible action)
    def score(t):
        hold_bonus = 0.05 if 2 <= t.days_held <= 7 else 0.0
        return t.ret_pct + hold_bonus
    cands.sort(key=score, reverse=True)
    return cands[0] if cands else None


def _draw_candles(ax, df: pd.DataFrame, entry_idx: int) -> None:
    """Hand-roll candles from OHLC; categorical x so weekend gaps disappear."""
    n = len(df)
    width = 0.7
    up = "#1f9d55"      # green
    down = "#c83232"    # red
    wick_w = 0.9
    body_w = 0.7

    for i in range(n):
        o = float(df["open"].iloc[i])
        h = float(df["high"].iloc[i])
        l = float(df["low"].iloc[i])
        c = float(df["close"].iloc[i])
        color = up if c >= o else down
        # wick
        ax.vlines(i, l, h, color=color, linewidth=wick_w, alpha=0.9)
        # body (always at least 1 pixel tall)
        body_top = max(o, c)
        body_bot = min(o, c)
        if body_top - body_bot < 1e-6:
            body_top = body_bot + (h - l) * 0.005 if h > l else body_bot * 0.001
        ax.add_patch(plt.Rectangle(
            (i - body_w / 2, body_bot), body_w, body_top - body_bot,
            facecolor=color, edgecolor=color, linewidth=0,
        ))

    # Entry-day vertical guide (the rightmost candle, by definition)
    ax.axvline(entry_idx, color="#1f6feb", alpha=0.45, linewidth=1.2,
               linestyle="--", zorder=4)


def _fmt_axes(ax, df: pd.DataFrame, ticker: str) -> None:
    n = len(df)
    ax.set_xlim(-1, n)
    # X-axis: month labels at the first bar of each month
    dates = pd.DatetimeIndex(df["timestamp"])
    month_starts = []
    seen_months = set()
    for i, d in enumerate(dates):
        ym = (d.year, d.month)
        if ym not in seen_months:
            seen_months.add(ym)
            month_starts.append(i)
    ax.set_xticks(month_starts)
    ax.set_xticklabels([dates[i].strftime("%b %y") for i in month_starts],
                       rotation=0, fontsize=9)

    # Y-axis: price grid with sensible major ticks
    pmin = float(df["low"].min())
    pmax = float(df["high"].max())
    span = pmax - pmin
    # major tick step ~ 8 ticks across the range, rounded to a nice number
    raw_step = span / 8
    nice = [0.1, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100]
    step = next(s for s in nice if s >= raw_step) if raw_step < nice[-1] else 100
    y_lo = (pmin // step) * step
    y_hi = ((pmax // step) + 2) * step
    ax.set_yticks(np.arange(y_lo, y_hi + step / 2, step))
    ax.set_ylim(y_lo - step * 0.1, y_hi)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.10, linewidth=0.4)
    ax.set_ylabel(f"{ticker} close ($)", fontsize=10)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", help="Specific ticker (otherwise auto-pick the top winner)")
    parser.add_argument("--entry", help="Specific entry date YYYY-MM-DD (only with --ticker)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    rows = load_rows(CSV_PATH)
    opt_trades = pair_options(rows)
    puts = [t for t in opt_trades if t.right == "PUT"]
    print(f"paired puts: {len(puts)}  winners: {sum(1 for t in puts if t.gross_pl > 0)}")

    if args.ticker:
        matches = [t for t in puts if t.ticker == args.ticker.upper()]
        if args.entry:
            from datetime import date
            tgt = date.fromisoformat(args.entry)
            matches = [t for t in matches if t.entry_date == tgt]
        if not matches:
            print(f"no matching put found for ticker={args.ticker} entry={args.entry}")
            return
        pick = matches[0]
    else:
        cands = sorted([t for t in puts if t.gross_pl > 0 and t.days_held >= 2],
                       key=lambda t: -t.ret_pct)[:5]
        print("\ntop 5 winning puts (>=2d hold) ranked by ret%:")
        for t in cands:
            print(f"  {t.ticker:5} BTO {t.entry_date}  STC {t.exit_date}  "
                  f"K=${t.strike:.0f}  ret {t.ret_pct*100:+.0f}%  hold {t.days_held}d")
        pick = _pick_representative_winner(puts)
        if pick is None:
            print("no winners found; aborting"); return

    print(f"\npicked: {pick.ticker} PUT ${pick.strike:.0f} "
          f"BTO {pick.entry_date} STC {pick.exit_date} "
          f"ret {pick.ret_pct*100:+.0f}% hold {pick.days_held}d")

    # Fetch a generous window so we have 252 td up to entry.
    # Entry is ~mid-2025; "3y" gives us May 2023+ which is plenty.
    df = fetch_bars(pick.ticker, period="3y")
    if df is None or df.empty:
        print(f"no bars for {pick.ticker}; aborting"); return

    df = df.dropna(subset=["close"]).reset_index(drop=True)
    # Find entry index (first bar on/after entry_date)
    ts = pd.to_datetime(df["timestamp"])
    entry_pos = int(ts.searchsorted(pd.Timestamp(pick.entry_date), side="left"))
    if entry_pos >= len(df):
        print(f"entry date {pick.entry_date} past last bar; aborting"); return

    # Window: 252 trading days ENDING ON the entry day (inclusive).
    lo = max(0, entry_pos - LOOKBACK_TD + 1)
    hi = entry_pos + 1
    window = df.iloc[lo:hi].reset_index(drop=True)
    entry_idx_in_window = len(window) - 1

    n_bars = len(window)
    first_date = pd.Timestamp(window["timestamp"].iloc[0]).date()
    last_date = pd.Timestamp(window["timestamp"].iloc[-1]).date()
    print(f"window: {n_bars} bars from {first_date} to {last_date}")

    fig, ax = plt.subplots(figsize=(18, 8))
    _draw_candles(ax, window, entry_idx_in_window)
    _fmt_axes(ax, window, pick.ticker)

    title = (f"{pick.ticker} PUT  ${pick.strike:.2f}  "
             f"BTO {pick.entry_date}  ({pick.dte_at_entry}d DTE)")
    subtitle = (f"252 trading days ending on BTO day  -  "
                f"trade outcome: ret {pick.ret_pct*100:+.0f}%, hold "
                f"{pick.days_held}d, STC {pick.exit_date}  -  "
                f"draw your levels and return")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    ax.set_title(subtitle, fontsize=10, color="#444", pad=8)

    # Annotate entry day with a small tag
    last_high = float(window["high"].iloc[-1])
    last_low = float(window["low"].iloc[-1])
    ax.annotate("BTO", (entry_idx_in_window, last_high),
                xytext=(8, 8), textcoords="offset points",
                ha="left", va="bottom", fontsize=11, fontweight="bold",
                color="#1f6feb")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_name = f"personal_put_pre_entry_{pick.ticker}_{pick.entry_date}.png"
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
