"""Tune the winning PUTS variant from divergence_sweep.py: variant C
(DIV-inside-bounce, lb=25). Sweep the RSI divergence threshold and the
pivot-gap minimum to see whether PF can clear the shipped puts sleeve
bar (2.53) without giving up CAGR.

Frozen from divergence_sweep.py Variant C, puts side:
  band              [SWEET_LO_PUT, SWEET_HI_PUT] = [+2.0, +2.5]
  lookback          25 bars
  resistance        >= 5 prior bars in 252d within +/-2% of recent 20b high
  cooldown          20 bars
  exits             R6 mirror (no TP, sigma_target=0, $200 disaster,
                    45 td time stop, 5% OTM put, 45 DTE)

Swept:
  RSI_DIVERG_MIN    {2, 3, 5, 7}    RSI pts; current shipped: 3
  PIVOT_GAP         {3, 5, 8}       min bars between current bar and
                                    the prior pivot used for comparison

12 cells. Reports the 12-row grid, then best PF / best CAGR vs.

GO/NO-GO (must clear to ship as a 2nd puts variant):
  PF >= 2.53 AND CAGR >= +4.5% AND MaxDD >= -17.1%
  (the shipped puts sleeve bar; see [[puts_sleeve_shipped]])

Outputs:
  research/out/divergence_puts_tune.png
  research/out/divergence_puts_tune.txt
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
from backend.indicators import rsi as rsi_series  # noqa: E402
from research.volatile_universe import universe  # noqa: E402
from research.oversold_call_exit_sweep import (  # noqa: E402
    ExitCfg, _log_channel_z, _fmt_with_n, LOG_WINDOW,
)
from research.volatile_universe_puts_sweep import (  # noqa: E402
    Sig as SigPut, simulate_put, SWEET_LO_PUT, SWEET_HI_PUT,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, HEADER, STARTING_CAPITAL, MIN_ADV_M,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Frozen Variant-C constants
LOOKBACK = 25
SUPPORT_PCT = 0.02
SUPPORT_HITS_MIN = 5
COOLDOWN_BARS = 20
RSI_PERIOD = 14

# Sweep grids
RSI_THRESHOLDS = (2.0, 3.0, 5.0, 7.0)
PIVOT_GAPS = (3, 5, 8)


def _rsi_arr(closes: np.ndarray) -> np.ndarray:
    return rsi_series(pd.Series(closes), period=RSI_PERIOD).to_numpy(dtype=float)


def _divergence_signals_put(
    series: dict[str, Series],
    z_by_tk: dict[str, np.ndarray],
    *,
    rsi_diverg_min: float,
    pivot_gap: int,
) -> dict[object, list[SigPut]]:
    """Variant C, puts side. Tight band + resistance confluence +
    bearish RSI divergence (lower-high in RSI on a new local high)."""
    out: dict[object, list[SigPut]] = {}
    start_min = LOG_WINDOW + 2

    for tk, s in series.items():
        z = z_by_tk[tk]
        rsi_a = _rsi_arr(s.close)
        last_fire_idx = -10_000
        for i in range(start_min, len(s.close)):
            if i - last_fire_idx < COOLDOWN_BARS:
                continue
            zi = z[i]
            if np.isnan(zi):
                continue
            if not (SWEET_LO_PUT <= zi <= SWEET_HI_PUT):
                continue

            win_highs = s.high[i - LOOKBACK + 1: i + 1]
            recent_hi = float(np.nanmax(win_highs))
            if not np.isfinite(s.high[i]) or s.high[i] < recent_hi - 1e-12:
                continue

            prior_end = i - pivot_gap + 1
            prior_start = i - LOOKBACK + 1
            if prior_end - prior_start < 2:
                continue
            prior_slice = s.high[prior_start: prior_end]
            if not np.any(np.isfinite(prior_slice)):
                continue
            prior_arg = int(np.nanargmax(prior_slice))
            prior_idx = prior_start + prior_arg
            prior_high = float(s.high[prior_idx])
            if not np.isfinite(prior_high) or s.high[i] <= prior_high:
                continue

            ri = rsi_a[i]
            rp = rsi_a[prior_idx]
            if not (np.isfinite(ri) and np.isfinite(rp)):
                continue
            if ri > rp - rsi_diverg_min:
                continue

            # Resistance confluence
            band_lo_p = recent_hi * (1.0 - SUPPORT_PCT)
            band_hi_p = recent_hi * (1.0 + SUPPORT_PCT)
            prior_highs = s.high[i - LOG_WINDOW + 1: i - LOOKBACK + 1]
            hits = int(((prior_highs >= band_lo_p) & (prior_highs <= band_hi_p)).sum())
            if hits < SUPPORT_HITS_MIN:
                continue

            adv = s.adv_m[i]
            if not (np.isfinite(adv) and adv >= MIN_ADV_M):
                continue
            if not np.isfinite(s.rv[i]):
                continue

            sg = SigPut(tk=tk, i=i, z=float(zi), date=s.dates[i],
                        tier="DIV", breadth=0, adv=float(adv))
            out.setdefault(s.dates[i], []).append(sg)
            last_fire_idx = i
    return out


def _pf(st):
    return st.get("pf") or st.get("profit_factor") or 0.0


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tks = universe()
    print(f"Fetching {len(tks)} volatile-universe tickers x 5y...")
    bars = fetch_bars_bulk(tks, period="5y")
    print(f"  got {len(bars)} usable tickers")

    series: dict[str, Series] = {}
    z_by_tk: dict[str, np.ndarray] = {}
    for tk, df in bars.items():
        s = build_series(tk, df)
        if s is not None:
            series[tk] = s
            z_by_tk[tk] = _log_channel_z(s.close)
    print(f"  {len(series)} tickers with full 252d history\n")

    R6_PUT = ExitCfg(
        tp_pct=None,
        disaster_pct=None, disaster_usd=200.0,
        time_stop=45,
        sigma_target=0.0,
        iv_elevation=1.00, crush_td=30,
        otm_pct=5.0, dte=45,
    )

    EHEADER = HEADER + "  | exit-reason mix"

    lines: list[str] = []
    lines.append("DIV-inside-bounce PUTS tune -- Variant C, lookback fixed at 25.")
    lines.append(f"Sweep RSI_DIVERG_MIN x PIVOT_GAP = {len(RSI_THRESHOLDS)} x {len(PIVOT_GAPS)} = "
                 f"{len(RSI_THRESHOLDS)*len(PIVOT_GAPS)} cells.")
    lines.append(f"Band [{SWEET_LO_PUT},{SWEET_HI_PUT}], resistance +/-{SUPPORT_PCT*100:.1f}% >={SUPPORT_HITS_MIN} bars,")
    lines.append(f"cooldown {COOLDOWN_BARS}, R6 exits (no TP, sigma|<=0, $200 disaster, 45td, 5% OTM, 45 DTE).")
    lines.append(f"Universe: {len(series)} tickers, full 252d history.\n")

    lines.append("[GRID  rows=RSI threshold, cols=pivot gap]")
    lines.append(EHEADER)

    all_rows: list[tuple[str, dict]] = []
    plot_rows: list[tuple[str, dict]] = []

    # Track grid for compact summary table
    grid_pf: dict[tuple[float, int], float] = {}
    grid_cagr: dict[tuple[float, int], float] = {}
    grid_dd: dict[tuple[float, int], float] = {}
    grid_n: dict[tuple[float, int], int] = {}

    for rsi_thr in RSI_THRESHOLDS:
        for pgap in PIVOT_GAPS:
            sigs = _divergence_signals_put(
                series, z_by_tk,
                rsi_diverg_min=rsi_thr,
                pivot_gap=pgap,
            )
            n = sum(len(v) for v in sigs.values())
            st = simulate_put(series, z_by_tk, sigs, R6_PUT)
            tag = f"rsi>={rsi_thr:.0f} gap={pgap} (n_sig={n})"
            lines.append(_fmt_with_n(tag, st))
            all_rows.append((tag, st))
            plot_rows.append((tag, st))
            grid_pf[(rsi_thr, pgap)] = _pf(st)
            grid_cagr[(rsi_thr, pgap)] = st.get("cagr", 0.0) * 100
            grid_dd[(rsi_thr, pgap)] = st.get("max_dd", 0.0) * 100
            grid_n[(rsi_thr, pgap)] = st.get("n", 0)

    lines.append("")
    lines.append("[GRID SUMMARY -- PF / CAGR%/ MaxDD% / n]")
    header_cols = " | ".join(f"gap={g:>1}" for g in PIVOT_GAPS)
    lines.append(f"  rsi\\gap | {header_cols}")
    for rsi_thr in RSI_THRESHOLDS:
        cells = []
        for pgap in PIVOT_GAPS:
            cells.append(
                f"PF {grid_pf[(rsi_thr, pgap)]:.2f} "
                f"C{grid_cagr[(rsi_thr, pgap)]:+5.1f} "
                f"DD{grid_dd[(rsi_thr, pgap)]:5.1f} "
                f"n{grid_n[(rsi_thr, pgap)]:>3}"
            )
        lines.append(f"  rsi>={int(rsi_thr):>2} | " + " | ".join(cells))
    lines.append("")

    # Best by each metric
    best_pf = max(all_rows, key=lambda kv: _pf(kv[1]))
    best_cagr = max(all_rows, key=lambda kv: kv[1].get("cagr", -1e9))
    best_mar = max(all_rows, key=lambda kv: kv[1].get("mar", -1e9) if np.isfinite(kv[1].get("mar", -1e9)) else -1e9)

    lines.append("[HEADLINE]")
    for label, row in (("best PF  ", best_pf), ("best CAGR", best_cagr), ("best MAR ", best_mar)):
        st = row[1]
        lines.append(
            f"  {label} : {row[0]}   "
            f"PF {_pf(st):.2f}  CAGR {st.get('cagr', 0)*100:+.1f}%  "
            f"Sharpe {st.get('sharpe', 0):.2f}  "
            f"MaxDD {st.get('max_dd', 0)*100:.1f}%  "
            f"MAR {st.get('mar', 0):.2f}  "
            f"n={st.get('n', 0)}"
        )
    lines.append("")
    lines.append("[GO/NO-GO -- shipped puts sleeve]")
    lines.append("  must clear: PF >= 2.53  AND  CAGR >= +4.5%  AND  MaxDD >= -17.1%")
    lines.append("  reference  : rsi>=3 gap=3 (the existing variant-C cell, lb=25)")
    lines.append("               PF 2.20  CAGR +12.6%  Sharpe 0.69  MaxDD -15.6%  n=35")
    lines.append("")

    # Filter cells that clear all three bars
    qual = []
    for (rsi_thr, pgap), pf in grid_pf.items():
        cagr = grid_cagr[(rsi_thr, pgap)]
        dd = grid_dd[(rsi_thr, pgap)]
        n = grid_n[(rsi_thr, pgap)]
        if pf >= 2.53 and cagr >= 4.5 and dd >= -17.1:
            qual.append((rsi_thr, pgap, pf, cagr, dd, n))
    if qual:
        lines.append("[QUALIFIERS -- cells that clear all three bars]")
        for rsi_thr, pgap, pf, cagr, dd, n in qual:
            lines.append(f"  rsi>={int(rsi_thr)} gap={pgap}: PF {pf:.2f} CAGR {cagr:+.1f}% MaxDD {dd:.1f}% n={n}")
    else:
        lines.append("[QUALIFIERS] none of the 12 cells clear all three bars.")
    lines.append("")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    for tag, st in plot_rows:
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.2,
                label=f"{tag} (CAGR {st.get('cagr', 0)*100:+.1f}% | "
                      f"PF {_pf(st):.2f} | n={st.get('n', 0)})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title(f"DIV-inside-bounce PUTS tune -- lb=25, "
                 f"RSI threshold x pivot gap ({len(RSI_THRESHOLDS)*len(PIVOT_GAPS)} cells)")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "divergence_puts_tune.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "divergence_puts_tune.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
