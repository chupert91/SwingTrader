"""Underlying price paths around the user's 20 paired 2025 put trades.

User's hypothesis (after seeing the failed forward-tests):
  "I didn't always time the top. I think I caught the BREATHER before
   continuations or sideways trading."

Translation: the alpha isn't entry-side (calling the top), it's
exit-side (closing during the brief pullback before the trend resumes).
If true:
  - Winners' underlying drops modestly (-3 to -8%) within 1-5 days
  - User exits DURING that drop / pause
  - After exit, the underlying often resumes the uptrend
  - So the "right exit" is short-horizon, opportunistic, not patient

Method:
  1. For each put trade, fetch the underlying's daily bars from
     entry_date - 5 bars to entry_date + 20 bars.
  2. Normalize to entry_close = 1.0.
  3. Plot all winner paths together, all loser paths together. Overlay
     the median path for each group. Mark each trade's exit_date.
  4. Tabulate underlying %change at days {1, 2, 3, 5, 10, 15} for
     winners vs losers — plus the price AFTER exit.
  5. Specifically test: does the price recover after the user exits a
     winner? If yes -> user is catching breathers, not tops.

Run:
    python research/personal_puts_post_entry_paths.py
Outputs:
    research/out/personal_puts_post_entry_paths.png  (small-multiples)
    research/out/personal_puts_post_entry_paths_combined.png  (overlay)
    research/out/personal_puts_post_entry_paths.txt
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from research.personal_trade_audit import load_rows, pair_options, Trade  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

BEFORE = 5    # bars before entry to show context
AFTER = 20    # bars after entry to show the post-trade arc


@dataclass
class Path:
    t: Trade
    days_offset: list[int]      # [-BEFORE..+AFTER] integer trading-day offsets
    norm_close: list[float]     # close[i] / entry_close
    entry_offset: int           # index in days_offset where entry sits (== BEFORE)
    exit_offset: int | None     # index where exit sits (may be after AFTER)
    days_held_bars: int

    @property
    def is_win(self) -> bool:
        return self.t.gross_pl > 0


def _build_path(t: Trade, df: pd.DataFrame) -> Path | None:
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df = df.reset_index(drop=True)
    matches = df.index[df["date"] == t.entry_date].tolist()
    if not matches:
        # Find closest bar on or before entry_date.
        before = df[df["date"] <= t.entry_date]
        if before.empty:
            return None
        i = before.index[-1]
    else:
        i = matches[0]
    # Likewise for exit_date.
    matches_x = df.index[df["date"] == t.exit_date].tolist()
    if matches_x:
        xi = matches_x[0]
    else:
        bf = df[df["date"] <= t.exit_date]
        xi = bf.index[-1] if len(bf) else None

    closes = df["close"].to_numpy(dtype=float)
    if i < BEFORE or i + AFTER >= len(closes):
        # Trade is too close to edge of data; pad with NaN.
        pass
    entry_close = closes[i]
    if entry_close <= 0:
        return None

    days_offset: list[int] = []
    norm_close: list[float] = []
    for off in range(-BEFORE, AFTER + 1):
        j = i + off
        if 0 <= j < len(closes):
            days_offset.append(off)
            norm_close.append(float(closes[j] / entry_close))
        else:
            days_offset.append(off)
            norm_close.append(float("nan"))

    if xi is not None:
        exit_off = xi - i
    else:
        exit_off = None

    days_held_bars = (xi - i) if (xi is not None and xi > i) else 0

    return Path(
        t=t, days_offset=days_offset, norm_close=norm_close,
        entry_offset=BEFORE, exit_offset=exit_off,
        days_held_bars=days_held_bars,
    )


def _median_path(paths: list[Path]) -> list[float]:
    if not paths:
        return []
    n = len(paths[0].norm_close)
    med: list[float] = []
    for k in range(n):
        vals = [p.norm_close[k] for p in paths
                if not math.isnan(p.norm_close[k])]
        med.append(median(vals) if vals else float("nan"))
    return med


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows(os.path.join(ROOT, "reference", "DownloadTxnHistory.csv"))
    puts = [t for t in pair_options(rows) if t.right == "PUT"]
    tickers = sorted({t.ticker for t in puts})
    print(f"Fetching {len(tickers)} underlyings...")
    bars = fetch_bars_bulk(tickers, period="5y")
    print(f"  got {len(bars)}")

    paths: list[Path] = []
    for t in sorted(puts, key=lambda x: x.entry_date):
        df = bars.get(t.ticker)
        if df is None or df.empty:
            continue
        p = _build_path(t, df)
        if p is not None:
            paths.append(p)
    wins = [p for p in paths if p.is_win]
    losses = [p for p in paths if not p.is_win]
    print(f"  built paths for {len(paths)} puts ({len(wins)}W / {len(losses)}L)\n")

    out: list[str] = []
    out.append("USER PUT TRADES — UNDERLYING PRICE PATHS AROUND ENTRY")
    out.append(f"  n: {len(paths)}  ({len(wins)} wins, {len(losses)} losses)")
    out.append(f"  window: entry {-BEFORE}d ... entry +{AFTER}d (trading days)")
    out.append("")

    # ---- normalized %-change table at key offsets ----
    key_offsets = [-3, -1, 0, 1, 2, 3, 5, 7, 10, 15, 20]
    out.append("[MEDIAN UNDERLYING %CHANGE FROM ENTRY (close basis)]")
    out.append(f"  {'offset (trading-d)':>20s}  {'WINNERS':>10s}  {'LOSERS':>10s}  {'spread':>8s}")
    for off in key_offsets:
        idx = BEFORE + off
        w_vals = [(p.norm_close[idx] - 1.0) * 100 for p in wins
                  if 0 <= idx < len(p.norm_close) and not math.isnan(p.norm_close[idx])]
        l_vals = [(p.norm_close[idx] - 1.0) * 100 for p in losses
                  if 0 <= idx < len(p.norm_close) and not math.isnan(p.norm_close[idx])]
        wm = median(w_vals) if w_vals else float("nan")
        lm = median(l_vals) if l_vals else float("nan")
        sp = (wm - lm) if not (math.isnan(wm) or math.isnan(lm)) else float("nan")
        out.append(f"  {f'entry{off:+d}':>20s}  "
                   f"{wm:>+9.1f}%  {lm:>+9.1f}%  {sp:>+7.1f}%")
    out.append("")

    # ---- did the price recover after the user EXITED winners? ----
    out.append("[POST-EXIT BEHAVIOR ON WINNERS]")
    out.append("  if user 'caught the breather', price should recover (go UP)")
    out.append("  in the 5 bars AFTER the exit. measured as %change exit_close")
    out.append("  to exit+5 close.")
    recoveries: list[float] = []
    for p in wins:
        if p.exit_offset is None or p.exit_offset >= AFTER:
            continue
        exit_idx = BEFORE + p.exit_offset
        end_idx = min(exit_idx + 5, len(p.norm_close) - 1)
        ev = p.norm_close[exit_idx]
        fv = p.norm_close[end_idx]
        if math.isnan(ev) or math.isnan(fv) or ev <= 0:
            continue
        recoveries.append(((fv / ev) - 1.0) * 100)
    if recoveries:
        out.append(f"  n: {len(recoveries)}")
        out.append(f"  median: {median(recoveries):+.1f}%   "
                   f"mean: {sum(recoveries) / len(recoveries):+.1f}%")
        up = sum(1 for r in recoveries if r > 0)
        out.append(f"  bounced UP in 5d after exit: {up}/{len(recoveries)} "
                   f"({100 * up / len(recoveries):.0f}%)")
    out.append("")

    # ---- bottom-of-drop vs exit timing (winners only) ----
    out.append("[DID THE USER EXIT NEAR THE TROUGH?]")
    out.append("  For each winner: find the lowest underlying close within the")
    out.append("  entry+15d window. Compare to user's actual exit timing.")
    diffs: list[tuple] = []
    for p in wins:
        idx_entry = BEFORE
        post = p.norm_close[idx_entry:idx_entry + 16]
        post = [v for v in post if not math.isnan(v)]
        if len(post) < 3:
            continue
        trough_idx = int(np.argmin(post))
        trough_val = post[trough_idx]
        exit_idx_rel = p.exit_offset if p.exit_offset is not None else len(post) - 1
        exit_val = p.norm_close[idx_entry + exit_idx_rel] if (idx_entry + exit_idx_rel < len(p.norm_close)) else float("nan")
        if math.isnan(exit_val):
            continue
        diffs.append((p.t.ticker, str(p.t.entry_date), trough_idx,
                      exit_idx_rel, (trough_val - 1) * 100, (exit_val - 1) * 100))
    out.append(f"  {'tkr':>5s}  {'entry':>10s}  {'trough_d':>8s} {'exit_d':>6s}  "
               f"{'trough%':>8s} {'exit%':>7s}  {'exit-trough':>12s}")
    for tk, ed, td, xd, tv, xv in diffs:
        out.append(f"  {tk:>5s}  {ed:>10s}  {td:>8d} {xd:>6d}  "
                   f"{tv:>+7.1f}% {xv:>+6.1f}%  {xv - tv:>+11.1f}pts")
    out.append("")

    # ---- per-trade outcome detail ----
    out.append("[PER-TRADE DETAIL — winners then losers]")
    out.append(f"  {'tkr':>5s}  {'entry':>10s}  {'W/L':>3s}  "
               f"{'opt%':>7s}  {'held':>5s}  "
               f"{'u_d1':>6s} {'u_d2':>6s} {'u_d3':>6s} {'u_d5':>6s} {'u_d10':>6s}  {'min%':>7s}")
    for grp in (wins, losses):
        for p in grp:
            d1 = (p.norm_close[BEFORE + 1] - 1) * 100 if BEFORE + 1 < len(p.norm_close) and not math.isnan(p.norm_close[BEFORE + 1]) else float("nan")
            d2 = (p.norm_close[BEFORE + 2] - 1) * 100 if BEFORE + 2 < len(p.norm_close) and not math.isnan(p.norm_close[BEFORE + 2]) else float("nan")
            d3 = (p.norm_close[BEFORE + 3] - 1) * 100 if BEFORE + 3 < len(p.norm_close) and not math.isnan(p.norm_close[BEFORE + 3]) else float("nan")
            d5 = (p.norm_close[BEFORE + 5] - 1) * 100 if BEFORE + 5 < len(p.norm_close) and not math.isnan(p.norm_close[BEFORE + 5]) else float("nan")
            d10 = (p.norm_close[BEFORE + 10] - 1) * 100 if BEFORE + 10 < len(p.norm_close) and not math.isnan(p.norm_close[BEFORE + 10]) else float("nan")
            window = [v for v in p.norm_close[BEFORE:BEFORE + 16] if not math.isnan(v)]
            mn = (min(window) - 1) * 100 if window else float("nan")
            out.append(
                f"  {p.t.ticker:>5s}  {str(p.t.entry_date):>10s}  "
                f"{'W' if p.is_win else 'L':>3s}  "
                f"{p.t.ret_pct * 100:>+6.1f}%  {p.days_held_bars:>4d}d  "
                f"{d1:>+5.1f}% {d2:>+5.1f}% {d3:>+5.1f}% {d5:>+5.1f}% {d10:>+5.1f}%  "
                f"{mn:>+6.1f}%"
            )
        out.append("")

    # ---- text save ----
    text = "\n".join(out)
    print(text)
    out_txt = os.path.join(OUT_DIR, "personal_puts_post_entry_paths.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

    # ---- combined overlay plot ----
    fig, ax = plt.subplots(figsize=(13, 7))
    xs = list(range(-BEFORE, AFTER + 1))
    for p in wins:
        ax.plot(xs, [(v - 1) * 100 for v in p.norm_close], color="#26a69a",
                alpha=0.25, linewidth=1.0)
        if p.exit_offset is not None and 0 <= BEFORE + p.exit_offset < len(p.norm_close):
            ex = (p.norm_close[BEFORE + p.exit_offset] - 1) * 100
            ax.plot([p.exit_offset], [ex], "o", color="#26a69a", markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5)
    for p in losses:
        ax.plot(xs, [(v - 1) * 100 for v in p.norm_close], color="#e74c3c",
                alpha=0.25, linewidth=1.0)
        if p.exit_offset is not None and 0 <= BEFORE + p.exit_offset < len(p.norm_close):
            ex = (p.norm_close[BEFORE + p.exit_offset] - 1) * 100
            ax.plot([p.exit_offset], [ex], "o", color="#e74c3c", markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5)
    if wins:
        med_w = _median_path(wins)
        ax.plot(xs, [(v - 1) * 100 for v in med_w], color="#26a69a", linewidth=2.8,
                label=f"WINNERS median (n={len(wins)})")
    if losses:
        med_l = _median_path(losses)
        ax.plot(xs, [(v - 1) * 100 for v in med_l], color="#e74c3c", linewidth=2.8,
                label=f"LOSERS median (n={len(losses)})")
    ax.axvline(0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axhline(0, color="gray", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("trading days from entry (entry = 0)")
    ax.set_ylabel("underlying %change from entry close")
    ax.set_title("User's 2025 put trades — underlying price paths around entry "
                 "(dots = actual exit)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png_combined = os.path.join(OUT_DIR, "personal_puts_post_entry_paths_combined.png")
    fig.savefig(out_png_combined, dpi=110)
    plt.close(fig)

    # ---- small-multiples plot ----
    n = len(paths)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2.4), squeeze=False)
    for k, p in enumerate(paths):
        ax = axes[k // cols][k % cols]
        color = "#26a69a" if p.is_win else "#e74c3c"
        ys = [(v - 1) * 100 for v in p.norm_close]
        ax.plot(xs, ys, color=color, linewidth=1.4)
        ax.axvline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
        if p.exit_offset is not None and 0 <= BEFORE + p.exit_offset < len(p.norm_close):
            ex_x = p.exit_offset
            ex_y = (p.norm_close[BEFORE + p.exit_offset] - 1) * 100
            ax.plot([ex_x], [ex_y], "o", color="black", markersize=5)
        ax.set_title(
            f"{p.t.ticker} {p.t.entry_date}  "
            f"opt {'+' if p.is_win else ''}{p.t.ret_pct * 100:.0f}%  hold {p.days_held_bars}d",
            fontsize=9
        )
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.25)
    # Hide unused subplots.
    for k in range(len(paths), rows * cols):
        axes[k // cols][k % cols].set_visible(False)
    fig.suptitle("Per-put price paths around entry — black dot is the user's actual exit",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    out_png_sm = os.path.join(OUT_DIR, "personal_puts_post_entry_paths.png")
    fig.savefig(out_png_sm, dpi=110)
    plt.close(fig)

    print(f"\n  -> {out_txt}")
    print(f"  -> {out_png_combined}")
    print(f"  -> {out_png_sm}")


if __name__ == "__main__":
    main()
