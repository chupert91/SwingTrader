"""Backtest the BOUNCE-CONFIRM signal on the volatile-universe options sim.

Motivation
----------
The validated edge (backend/ai_strategy.scan_candidates) only fires when
broad-market same-day-of-touch breadth >= 3. It misses a different shape
the user catches by eye on individual charts: a name still inside the
[-2.5,-2.0] primary band but with a CONFIRMED bounce off structural
support and a turning Stoch RSI -- broad-market breadth irrelevant.

Five features, all must be true:

  F1  in primary band           z in [SWEET_LO, SWEET_HI] = [-2.5, -2.0]
  F2  z reclaim                 today's z >= 20-bar local-z min + RECLAIM_SIGMA
  F3  higher low (252d)         min(low, last 20 bars) > min(low, prior 232 bars)
  F4  Stoch RSI bottoming       %K <= 30 AND %K > min(%K, last 20 bars) + 1
  F5  support confluence        >= 5 prior bars in 252d window with low within
                                +/- 2% of current 20-bar low

A name that fires gets one entry; we then cool down for COOLDOWN_BARS so
a continuous streak doesn't spam entries. Same option instrument
(otm/dte/IV model) and same EXIT rails as live (R6 / volatile-best):
no TP, sigma-target z>=0, $200 disaster cap, 45d time stop, 5% OTM, dte
from live setting (45 trading days).

Outputs:
    research/out/bounce_confirm_sweep.png
    research/out/bounce_confirm_sweep.txt

Comparison block:
    BASELINE    -- live edge (primary band + tier in {OK,PRIME,PANIC})
    BOUNCE-only -- only the 5-of-5 bounce_confirm signal
    COMBINED    -- BASELINE union BOUNCE-only (dedup per (ticker, date))

F2 sweep over RECLAIM_SIGMA in {0.05, 0.10, 0.20, 0.30}.
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.indicators import stoch_rsi  # noqa: E402
from research.volatile_universe import universe  # noqa: E402
from research.oversold_call_exit_sweep import (  # noqa: E402
    ExitCfg, simulate, _log_channel_z, _signals as baseline_signals,
    _fmt_with_n, Sig, SWEET_LO, SWEET_HI, LOG_WINDOW,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, HEADER, STARTING_CAPITAL, MIN_ADV_M,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Bounce-confirm config
LOOKBACK = 20                  # F2/F3/F4 short-frame window
PRIOR_WINDOW = LOG_WINDOW       # F3/F5 long-frame window (matches the regression window)
SUPPORT_PCT = 0.02             # F5 +/- 2% of recent 20d low
SUPPORT_HITS_MIN = 5
STOCH_OS_MAX = 30.0
STOCH_TURN_MIN = 1.0           # F4: %K must lift at least 1.0 above the 20b min
COOLDOWN_BARS = 20             # dedup: skip the next N bars after a fire


def _bounce_signals(
    series: dict[str, Series],
    z_by_tk: dict[str, np.ndarray],
    reclaim_sigma: float,
) -> dict[object, list[Sig]]:
    """date -> list[Sig]. Fires on the FIRST bar in a window where the
    5-feature pattern is true; then cooldown for COOLDOWN_BARS before
    eligible to re-fire on the same ticker."""
    out: dict[object, list[Sig]] = {}
    for tk, s in series.items():
        z = z_by_tk[tk]
        # Stoch RSI %K (default 14/14/3/3 from backend.indicators)
        k_ser, _ = stoch_rsi(pd.Series(s.close))
        k_arr = k_ser.to_numpy(dtype=float)

        last_fire_idx = -10_000
        # Need PRIOR_WINDOW history for F3/F5, plus LOOKBACK for short frame.
        start = PRIOR_WINDOW + LOOKBACK + 2
        for i in range(start, len(s.close)):
            if i - last_fire_idx < COOLDOWN_BARS:
                continue
            zi = z[i]
            if np.isnan(zi):
                continue

            # F1: primary band
            if not (SWEET_LO <= zi <= SWEET_HI):
                continue

            # F2: reclaim from recent local-z low
            recent_z_min = float(np.nanmin(z[i - LOOKBACK + 1: i + 1]))
            if zi < recent_z_min + reclaim_sigma:
                continue

            # F3: higher low on the 252d frame
            recent_lo = float(np.nanmin(s.low[i - LOOKBACK + 1: i + 1]))
            prior_lo = float(np.nanmin(s.low[i - PRIOR_WINDOW + 1: i - LOOKBACK + 1]))
            if not np.isfinite(prior_lo) or recent_lo <= prior_lo:
                continue

            # F4: Stoch RSI floored and turning up
            ki = k_arr[i]
            if not np.isfinite(ki) or ki > STOCH_OS_MAX:
                continue
            k_min_recent = float(np.nanmin(k_arr[i - LOOKBACK + 1: i + 1]))
            if ki <= k_min_recent + STOCH_TURN_MIN:
                continue

            # F5: support confluence -- prior bars within +/- 2% of recent low
            band_lo_p = recent_lo * (1.0 - SUPPORT_PCT)
            band_hi_p = recent_lo * (1.0 + SUPPORT_PCT)
            prior_lows = s.low[i - PRIOR_WINDOW + 1: i - LOOKBACK + 1]
            hits = int(((prior_lows >= band_lo_p) & (prior_lows <= band_hi_p)).sum())
            if hits < SUPPORT_HITS_MIN:
                continue

            # Liquidity gate (matches baseline edge).
            adv = s.adv_m[i]
            if not (np.isfinite(adv) and adv >= MIN_ADV_M):
                continue
            # RV must be finite (sim needs it to price the entry).
            if not np.isfinite(s.rv[i]):
                continue

            sg = Sig(tk=tk, i=i, z=float(zi), date=s.dates[i],
                     tier="BOUNCE", breadth=hits, adv=float(adv))
            out.setdefault(s.dates[i], []).append(sg)
            last_fire_idx = i
    return out


def _combine(a: dict[object, list[Sig]],
             b: dict[object, list[Sig]]) -> dict[object, list[Sig]]:
    """Union by (date, ticker); a wins on collision so baseline-tier is
    preserved over bounce-tier when both fire on the same name/day."""
    out: dict[object, list[Sig]] = {d: list(lst) for d, lst in a.items()}
    for d, lst in b.items():
        seen = {sg.tk for sg in out.get(d, [])}
        for sg in lst:
            if sg.tk not in seen:
                out.setdefault(d, []).append(sg)
                seen.add(sg.tk)
    return out


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
    print(f"  {len(series)} tickers with full 252d-of-history\n")

    # Live R6 exits — the only +CAGR / PF>2 row from volatile_universe_sweep.
    # No TP, sigma-revert at z=0, $200 absolute disaster cap, 45d time stop,
    # 5% OTM, dte 45 (earliest in live 45-120 band, lowest theta over-pay).
    LIVE_R6 = ExitCfg(
        tp_pct=None,
        disaster_pct=None, disaster_usd=200.0,
        time_stop=45,
        sigma_target=0.0,
        iv_elevation=1.00, crush_td=30,
        otm_pct=5.0, dte=45,
    )

    # Baseline signals (validated edge).
    base_sig = baseline_signals(series, z_by_tk)
    n_base = sum(len(v) for v in base_sig.values())
    print(f"baseline signals: {n_base}")

    lines: list[str] = []
    lines.append("Bounce-confirm sweep on the VOLATILE THEMATIC universe.")
    lines.append("Pattern: F1 z in [-2.5,-2.0], F2 reclaim from 20b z-min, F3 higher")
    lines.append("low on 252d, F4 Stoch RSI oversold+turning, F5 >=5 prior lows")
    lines.append(f"within +/-{SUPPORT_PCT*100:.1f}% of recent 20b low. Cooldown {COOLDOWN_BARS}b.")
    lines.append(f"Exit rails: R6 live (no TP, sigma>=0, $200 disaster, 45td, 5% OTM, 45 DTE).")
    lines.append(f"Tickers with full 252d history: {len(series)}.\n")

    EHEADER = HEADER + "  | exit-reason mix"

    # --- Baseline (live edge) ---
    base_st = simulate(series, z_by_tk, base_sig, LIVE_R6)
    lines.append("[BASELINE -- live edge (primary band + tier in OK/PRIME/PANIC)]")
    lines.append(EHEADER)
    lines.append(_fmt_with_n(f"baseline (n_sig={n_base})", base_st))
    lines.append("")

    # --- F2 sweep ---
    sweep_results: list[tuple[str, ExitCfg, dict]] = []
    bounce_sig_by_recl: dict[float, dict] = {}
    lines.append("[BOUNCE-ONLY F2 SWEEP (no baseline included)]")
    lines.append(EHEADER)
    for recl in (0.05, 0.10, 0.20, 0.30):
        sigs = _bounce_signals(series, z_by_tk, reclaim_sigma=recl)
        bounce_sig_by_recl[recl] = sigs
        n = sum(len(v) for v in sigs.values())
        st = simulate(series, z_by_tk, sigs, LIVE_R6)
        tag = f"bounce reclaim>={recl:.2f} (n_sig={n})"
        sweep_results.append((tag, LIVE_R6, st))
        lines.append(_fmt_with_n(tag, st))
    lines.append("")

    # --- Combined: baseline union bounce ---
    lines.append("[COMBINED -- baseline UNION bounce (dedup by ticker+date)]")
    lines.append(EHEADER)
    combined_results: list[tuple[str, dict]] = []
    for recl, sigs in bounce_sig_by_recl.items():
        merged = _combine(base_sig, sigs)
        n = sum(len(v) for v in merged.values())
        st = simulate(series, z_by_tk, merged, LIVE_R6)
        tag = f"combined reclaim>={recl:.2f} (n_sig={n})"
        combined_results.append((tag, st))
        lines.append(_fmt_with_n(tag, st))
    lines.append("")

    # --- Headline ---
    def pf(st): return st.get("pf", 0) or 0
    def cagr(st): return st.get("cagr", -1e9)
    all_rows = [("baseline", base_st)] \
        + [(t, st) for t, _, st in sweep_results] \
        + [(t, st) for t, st in combined_results]
    best_pf = max(all_rows, key=lambda kv: pf(kv[1]))
    best_cagr = max(all_rows, key=lambda kv: cagr(kv[1]))
    lines.append("[HEADLINE]")
    lines.append(f"  best PF   : {best_pf[0]}   "
                 f"PF {pf(best_pf[1]):.2f}  CAGR {cagr(best_pf[1])*100:+.1f}%  "
                 f"Sharpe {best_pf[1].get('sharpe', 0):.2f}  n={best_pf[1].get('n', 0)}")
    lines.append(f"  best CAGR : {best_cagr[0]}   "
                 f"PF {pf(best_cagr[1]):.2f}  CAGR {cagr(best_cagr[1])*100:+.1f}%  "
                 f"Sharpe {best_cagr[1].get('sharpe', 0):.2f}  n={best_cagr[1].get('n', 0)}")
    lines.append("")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 7))
    plot_rows = [("baseline (live edge)", base_st)] \
        + [(t, st) for t, _, st in sweep_results] \
        + [(t, st) for t, st in combined_results]
    for tag, st in plot_rows:
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.4,
                label=f"{tag} (CAGR {st.get('cagr', 0)*100:+.1f}% | PF "
                      f"{pf(st):.2f} | n={st.get('n', 0)})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title("Bounce-confirm vs live edge -- volatile universe, R6 exits")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "bounce_confirm_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "bounce_confirm_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
