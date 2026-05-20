"""Sigma-band overlay for a single put trade: shows where the underlying sat
in the 252d LOG regression channel at BTO and STC, with the centerline +
+/-1, +/-2, +/-2.5, +/-3 sigma bands drawn at every bar.

This is the bot's actual signal model — same formula as ai_strategy. Used
to confirm that the user's manually-drawn price levels coincide with (or
diverge from) the sigma-band structure the bot looks at.

CLI:
    python research/personal_put_sigma_overlay.py --ticker GOOG --entry 2025-11-10
    python research/personal_put_sigma_overlay.py --ticker APLD --entry 2025-10-15
    python research/personal_put_sigma_overlay.py --ticker IONQ --entry 2025-09-24
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars  # noqa: E402
from research.personal_trade_audit import CSV_PATH, load_rows, pair_options  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
LOG_WINDOW = 252


def _channel_at_bar(close: np.ndarray, t: int) -> tuple[float, float, float] | None:
    """Returns (fit_end_price, sigma_log, z_t) for the 252-bar window ENDING
    at bar t. Same regression as ai_strategy._log_channel_z. fit_end_price is
    the centerline price at bar t (i.e. exp(fit_end))."""
    if t < LOG_WINDOW - 1:
        return None
    y = np.log(close[t - LOG_WINDOW + 1: t + 1])
    x = np.arange(LOG_WINDOW, dtype=float)
    x_mean = x.mean()
    xc = x - x_mean
    denom = float(np.sum(xc ** 2))
    y_mean = y.mean()
    slope = float(np.sum(xc * (y - y_mean)) / denom)
    intercept = y_mean - slope * x_mean
    fit_end = slope * (LOG_WINDOW - 1) + intercept
    resid = y - (slope * x + intercept)
    sigma = float(np.std(resid, ddof=1))
    if sigma <= 0:
        return None
    z = (y[-1] - fit_end) / sigma
    return fit_end, sigma, z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--entry", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    rows = load_rows(CSV_PATH)
    opt_trades = pair_options(rows)
    puts = [t for t in opt_trades if t.right == "PUT"
            and t.ticker == args.ticker.upper()
            and t.entry_date == date.fromisoformat(args.entry)]
    if not puts:
        print(f"no matching put: {args.ticker} {args.entry}"); return
    trade = puts[0]

    df = fetch_bars(args.ticker, period="3y")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"])
    close = df["close"].values.astype(float)
    n = len(df)

    entry_pos = int(ts.searchsorted(pd.Timestamp(trade.entry_date), side="left"))
    exit_pos = int(ts.searchsorted(pd.Timestamp(trade.exit_date), side="left"))
    if entry_pos >= n or exit_pos >= n:
        print("trade dates past data; aborting"); return

    # Build per-bar channel projection: at each bar t, the channel computed
    # over [t-251..t]. For display we want the BAND VALUES as a continuous
    # series so we can plot them.
    centerline = np.full(n, np.nan)
    band_p1 = np.full(n, np.nan); band_n1 = np.full(n, np.nan)
    band_p2 = np.full(n, np.nan); band_n2 = np.full(n, np.nan)
    band_p25 = np.full(n, np.nan); band_n25 = np.full(n, np.nan)
    band_p3 = np.full(n, np.nan); band_n3 = np.full(n, np.nan)
    z_series = np.full(n, np.nan)

    for t in range(LOG_WINDOW - 1, n):
        out = _channel_at_bar(close, t)
        if out is None:
            continue
        fit_end, sigma, z = out
        centerline[t] = float(np.exp(fit_end))
        band_p1[t] = float(np.exp(fit_end + 1.0 * sigma))
        band_n1[t] = float(np.exp(fit_end - 1.0 * sigma))
        band_p2[t] = float(np.exp(fit_end + 2.0 * sigma))
        band_n2[t] = float(np.exp(fit_end - 2.0 * sigma))
        band_p25[t] = float(np.exp(fit_end + 2.5 * sigma))
        band_n25[t] = float(np.exp(fit_end - 2.5 * sigma))
        band_p3[t] = float(np.exp(fit_end + 3.0 * sigma))
        band_n3[t] = float(np.exp(fit_end - 3.0 * sigma))
        z_series[t] = z

    z_bto = z_series[entry_pos]
    z_stc = z_series[exit_pos]
    px_bto = close[entry_pos]
    px_stc = close[exit_pos]
    band_p2_bto = band_p2[entry_pos]
    band_p25_bto = band_p25[entry_pos]
    band_p3_bto = band_p3[entry_pos]
    cline_bto = centerline[entry_pos]

    # Numeric report first — this is the headline answer.
    print(f"\n{args.ticker} PUT ${trade.strike:.0f}  BTO {trade.entry_date}  STC {trade.exit_date}")
    print(f"  underlying  : BTO close = ${px_bto:.2f}   STC close = ${px_stc:.2f}   "
          f"move = ${px_stc - px_bto:+.2f}  ({(px_stc/px_bto - 1)*100:+.1f}%)")
    print(f"  z-score     : BTO z = {z_bto:+.2f}sigma   STC z = {z_stc:+.2f}sigma   "
          f"reversion = {z_bto - z_stc:+.2f}sigma")
    print(f"  centerline  : BTO day = ${cline_bto:.2f}   "
          f"(BTO price was ${px_bto - cline_bto:+.2f} above centerline)")
    print(f"  +/-2 sigma  : BTO day = +/- ${band_p2_bto:.2f} / ${band_n2[entry_pos]:.2f}   "
          f"(BTO price ${'ABOVE' if px_bto > band_p2_bto else 'below'} +2 sigma)")
    print(f"  +/-2.5      : BTO day = +/- ${band_p25_bto:.2f} / ${band_n25[entry_pos]:.2f}")
    print(f"  +/-3 sigma  : BTO day = +/- ${band_p3_bto:.2f} / ${band_n3[entry_pos]:.2f}")
    # What band did BTO fall into?
    if z_bto >= 3.0: band_label = "z >= +3sigma (EXTREME)"
    elif z_bto >= 2.5: band_label = "z in [+2.5, +3.0)"
    elif z_bto >= 2.0: band_label = "z in [+2.0, +2.5)  (the put-side band we tested and falsified)"
    elif z_bto >= 1.0: band_label = "z in [+1.0, +2.0)  (between centerline and +2 sigma)"
    elif z_bto >= 0.0: band_label = "z in [0, +1.0)  (above centerline but inside normal range)"
    else: band_label = "z < 0  (BELOW centerline)"
    print(f"  band class  : {band_label}")
    print()

    # ---- chart ----
    # Show a slice ending shortly after STC so the user can see the full
    # trajectory; start the slice at entry-180 td for context.
    lo = max(0, entry_pos - 180)
    hi = min(n, exit_pos + 21)
    sl = slice(lo, hi)
    sub_dates = ts.iloc[lo:hi]
    sub_close = close[lo:hi]

    fig, ax = plt.subplots(figsize=(16, 9))

    # Bands (drawn back to front so labels overlap cleanly)
    band_kwargs = dict(linewidth=1.0, alpha=0.65)
    ax.plot(sub_dates, centerline[sl], color="#888", linestyle="-",
            label="centerline (252d LOG regression)", **band_kwargs)
    ax.plot(sub_dates, band_p1[sl], color="#5b9bd5", linestyle="--",
            label="+/-1 sigma", **band_kwargs)
    ax.plot(sub_dates, band_n1[sl], color="#5b9bd5", linestyle="--", **band_kwargs)
    ax.plot(sub_dates, band_p2[sl], color="#ff8c1a", linestyle="--",
            label="+/-2 sigma", **band_kwargs)
    ax.plot(sub_dates, band_n2[sl], color="#ff8c1a", linestyle="--", **band_kwargs)
    ax.plot(sub_dates, band_p25[sl], color="#d4a017", linestyle="-",
            label="+/-2.5 sigma", linewidth=1.2, alpha=0.7)
    ax.plot(sub_dates, band_n25[sl], color="#d4a017", linestyle="-",
            linewidth=1.2, alpha=0.7)
    ax.plot(sub_dates, band_p3[sl], color="#c83232", linestyle="-",
            label="+/-3 sigma", linewidth=1.2, alpha=0.7)
    ax.plot(sub_dates, band_n3[sl], color="#c83232", linestyle="-",
            linewidth=1.2, alpha=0.7)

    # Gold zone above +2sigma to +2.5sigma (where the bot's UI shades) for visual reference
    ax.fill_between(sub_dates, band_p2[sl], band_p25[sl], color="#d4a017", alpha=0.10)
    ax.fill_between(sub_dates, band_n25[sl], band_n2[sl], color="#d4a017", alpha=0.10)

    # Price line on top
    ax.plot(sub_dates, sub_close, color="#1a3d5a", linewidth=1.6, label=f"{args.ticker} close")

    # BTO / STC markers
    ax.scatter([sub_dates.iloc[entry_pos - lo]], [px_bto], color="#1f6feb",
               s=120, marker="^", zorder=8, edgecolors="white", linewidths=1.2,
               label=f"BTO ({trade.entry_date}, z={z_bto:+.2f})")
    ax.scatter([sub_dates.iloc[exit_pos - lo]], [px_stc], color="#ff8c1a",
               s=120, marker="v", zorder=8, edgecolors="white", linewidths=1.2,
               label=f"STC ({trade.exit_date}, z={z_stc:+.2f})")

    ax.set_title(
        f"{args.ticker} PUT ${trade.strike:.0f}  BTO {trade.entry_date} -> "
        f"STC {trade.exit_date}  (ret {trade.ret_pct*100:+.0f}%, hold "
        f"{trade.days_held}d)\n"
        f"BTO z = {z_bto:+.2f}sigma   STC z = {z_stc:+.2f}sigma   "
        f"reverted {z_bto - z_stc:+.2f}sigma",
        fontsize=11
    )
    ax.set_ylabel(f"{args.ticker} close ($)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.20, linewidth=0.5)

    fig.tight_layout()
    out_name = f"personal_put_sigma_overlay_{args.ticker}_{trade.entry_date}.png"
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
