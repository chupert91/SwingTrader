"""Per-trade price-action snapshots for the user's real 2025 option trades.

For each of the 44 paired option trades from `personal_trade_audit.py`, plot
the underlying's price action over the window [entry - 10td, exit + 10td].
Mark the entry/exit days. Annotate with P/L%, hold, DTE, and the 252d LOG-
channel z at entry and exit so we can see where on the band the trade sat.

Two output PNGs, one per book:
    research/out/personal_trade_snapshots_calls.png   (6 cols x 4 rows)
    research/out/personal_trade_snapshots_puts.png    (5 cols x 4 rows)

Per-thumbnail layout:
    Top:    underlying price line, entry marker (green dot), exit marker
            (red/green dot by P/L), light grid.
    Title:  TKR  CALL/PUT  Kstrike  +PnL%  hold Nd  z=entryZ->exitZ
            Green title for wins, red for losses.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from research.personal_trade_audit import (  # noqa: E402
    CSV_PATH, load_rows, pair_options, Trade,
)
from research.oversold_call_exit_sweep import _log_channel_z  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# How many trading-day bars to show before entry and after exit.
PRE_BARS = 10
POST_BARS = 10


def _find_idx(dates: pd.DatetimeIndex, target) -> int | None:
    """Index of the first bar on or after `target` (a Python date). None if past end."""
    ts = pd.Timestamp(target)
    pos = dates.searchsorted(ts, side="left")
    if pos >= len(dates):
        return None
    return int(pos)


def _z_at(z: np.ndarray, i: int | None) -> float | None:
    if i is None or i < 0 or i >= len(z):
        return None
    v = z[i]
    return None if np.isnan(v) else float(v)


def _plot_thumbnail(ax: plt.Axes, t: Trade, df: pd.DataFrame, z: np.ndarray) -> None:
    dates = df.index
    i_ent = _find_idx(dates, t.entry_date)
    i_ext = _find_idx(dates, t.exit_date)
    if i_ent is None or i_ext is None:
        ax.text(0.5, 0.5, f"{t.ticker} no-bars", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        return

    lo = max(0, i_ent - PRE_BARS)
    hi = min(len(dates), i_ext + POST_BARS + 1)
    sub = df.iloc[lo:hi]
    sub_dates = sub.index
    sub_close = sub["close"].values.astype(float)
    if len(sub_close) == 0:
        ax.text(0.5, 0.5, f"{t.ticker} no-bars", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        return

    win = t.gross_pl > 0
    win_color = "#1f9d55"   # green
    loss_color = "#c83232"  # red
    title_color = win_color if win else loss_color

    # Entry/exit ALWAYS use distinct colors and shapes so winners don't blend.
    # Entry  = BLUE upward triangle ("in")
    # Exit   = ORANGE downward triangle ("out"); border encodes win/loss.
    entry_color = "#1f6feb"   # blue
    exit_color = "#ff8c1a"    # orange
    exit_edge = win_color if win else loss_color

    # Price line
    ax.plot(sub_dates, sub_close, color="#1a3d5a", linewidth=1.3)
    ax.fill_between(sub_dates, sub_close, sub_close.min(),
                    color="#1a3d5a", alpha=0.06)

    # Markers
    px_ent = float(df["close"].iloc[i_ent])
    px_ext = float(df["close"].iloc[i_ext])

    # Entry marker drawn first; exit on top. For same-day trades (i_ent ==
    # i_ext) we draw the exit triangle slightly above the entry so both stay
    # visible.
    ax.scatter([dates[i_ent]], [px_ent], color=entry_color, s=70,
               marker="^", zorder=5, edgecolors="white", linewidths=1.0,
               label="BTO")
    same_day = (i_ent == i_ext)
    px_ext_plot = px_ext * (1.015 if same_day else 1.0)  # nudge up 1.5% if overlap
    ax.scatter([dates[i_ext]], [px_ext_plot], color=exit_color, s=70,
               marker="v", zorder=6, edgecolors=exit_edge, linewidths=1.2,
               label="STC")

    # Small letter labels above each marker so they're identifiable even at
    # thumbnail size.
    ax.annotate("BTO", (dates[i_ent], px_ent),
                xytext=(0, 9), textcoords="offset points",
                ha="center", va="bottom", fontsize=6,
                color=entry_color, fontweight="bold")
    ax.annotate("STC", (dates[i_ext], px_ext_plot),
                xytext=(0, -10), textcoords="offset points",
                ha="center", va="top", fontsize=6,
                color=exit_color, fontweight="bold")

    # Vertical guides
    ax.axvline(dates[i_ent], color=entry_color, alpha=0.25, linewidth=0.7)
    ax.axvline(dates[i_ext], color=exit_color, alpha=0.25, linewidth=0.7)

    # Annotate underlying %-move during the trade
    und_pct = (px_ext - px_ent) / px_ent * 100.0
    ax.text(0.02, 0.96, f"und {und_pct:+.1f}%",
            transform=ax.transAxes, fontsize=7, color="#333",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # Z-readings
    z_ent = _z_at(z, i_ent)
    z_ext = _z_at(z, i_ext)
    z_txt = ""
    if z_ent is not None and z_ext is not None:
        z_txt = f" z={z_ent:+.1f}>{z_ext:+.1f}"
    elif z_ent is not None:
        z_txt = f" z={z_ent:+.1f}>?"

    # Title
    strike_txt = f"${t.strike:.0f}" if t.strike is not None else ""
    title = (f"{t.ticker} {t.right} {strike_txt}  "
             f"{t.ret_pct*100:+.0f}%  {t.days_held}d{z_txt}")
    ax.set_title(title, color=title_color, fontsize=8, pad=2)

    # Spartan ticks
    ax.tick_params(axis="both", which="both", labelsize=6, length=2)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    # Format x-axis lightly: just first/last dates as labels
    ax.set_xticks([sub_dates[0], sub_dates[-1]])
    ax.set_xticklabels([sub_dates[0].strftime("%b-%d"),
                         sub_dates[-1].strftime("%b-%d")],
                        fontsize=6)


def _plot_grid(trades: list[Trade], series: dict, z_by_tk: dict,
               book_name: str, ncols: int) -> str:
    n = len(trades)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.6, nrows * 2.0),
                             squeeze=False)
    # Sort trades by entry date so the grid reads left-to-right, top-to-bottom
    # in chronological order.
    trades = sorted(trades, key=lambda t: (t.entry_date, t.ticker))
    for k, t in enumerate(trades):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        df = series.get(t.ticker)
        z = z_by_tk.get(t.ticker)
        if df is None or z is None:
            ax.text(0.5, 0.5, f"{t.ticker}\nno data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="#888")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        _plot_thumbnail(ax, t, df, z)
    # Hide unused axes
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")

    wins = sum(1 for t in trades if t.gross_pl > 0)
    suptitle = (f"Personal 2025 {book_name} trades  -  "
                f"n={n}  wins={wins} ({wins/max(n,1)*100:.0f}%)  "
                f"net=${sum(t.gross_pl for t in trades):+,.0f}  "
                f"[-10td .. exit .. +10td]   "
                f"blue triangle = BTO entry, orange triangle = STC exit, "
                f"title color + exit border = win/loss")
    fig.suptitle(suptitle, fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    out = os.path.join(OUT_DIR,
                       f"personal_trade_snapshots_{book_name.lower()}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = load_rows(CSV_PATH)
    opt_trades = pair_options(rows)
    if not opt_trades:
        print("no paired option trades; aborting"); return

    calls = [t for t in opt_trades if t.right == "CALL"]
    puts = [t for t in opt_trades if t.right == "PUT"]
    print(f"paired option trades: {len(opt_trades)}  "
          f"(calls={len(calls)} / puts={len(puts)})")

    # Tickers we need bars for, plus the time range. Pad 260 trading days
    # before the earliest entry so the z calc has its 252-bar window.
    tickers = sorted({t.ticker for t in opt_trades})
    earliest_entry = min(t.entry_date for t in opt_trades)
    latest_exit = max(t.exit_date for t in opt_trades)
    # yfinance period strings only — pick "2y" if range fits, else "5y".
    # Earliest entry is ~May 2025; need 252 bars before -> Apr 2024. "2y" from
    # today (May 2026) covers May 2024 onward, which is too late for the z
    # warmup. Use "3y" for safety.
    period = "3y"
    print(f"fetching {len(tickers)} tickers x {period} "
          f"(entries {earliest_entry}..{latest_exit})")
    bars = fetch_bars_bulk(tickers, period=period)

    series: dict[str, pd.DataFrame] = {}
    z_by_tk: dict[str, np.ndarray] = {}
    for tk, df in bars.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        df = df.dropna(subset=["close"]).copy()
        if df.empty:
            continue
        # Index by timestamp so searchsorted works against Python dates.
        df = df.set_index(pd.DatetimeIndex(df["timestamp"]))
        series[tk] = df
        z_by_tk[tk] = _log_channel_z(df["close"].values.astype(float))

    missing = [tk for tk in tickers if tk not in series]
    if missing:
        print(f"  WARN: no bars for {len(missing)} tickers: {missing}")

    out_calls = _plot_grid(calls, series, z_by_tk, "Calls", ncols=6)
    print(f"  -> {out_calls}")
    out_puts = _plot_grid(puts, series, z_by_tk, "Puts", ncols=5)
    print(f"  -> {out_puts}")


if __name__ == "__main__":
    main()
