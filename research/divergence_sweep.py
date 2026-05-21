"""RSI(14) regular-divergence entry gate, sweep on the volatile universe.

Question
--------
Does classic bullish/bearish RSI(14) divergence improve entry quality when
price is OUTSIDE +/-1 sigma on the 252d log-channel? "Outside 1 sigma" is
looser than the current production bands (calls fire only in [-2.5, -2.0]),
so the divergence rule has to do the candidate-pruning work itself; without
it the wide band overwhelms the sim.

Three roles per side, three lookback windows each. RSI divergence threshold
fixed at 3.0 RSI points (meaningful but not strict; tighter sweep can come
later if any cell breaks the GO/NO-GO bar below).

Calls (bullish divergence)
  F-BAND-CALL    z in [-3.5, -1.0]                  ("outside 1 sigma" gate)
  F-DIV-CALL     pivot rule:
                   - current bar low <= min(low, last LOOKBACK bars)  (new local low)
                   - prior pivot low: argmin(low) over the window
                     i - LOOKBACK ... i - PIVOT_GAP
                   - low[i] < low[prior_pivot]    (price lower-low)
                   - RSI[i] > RSI[prior_pivot] + RSI_DIVERG_MIN
                     (RSI higher-low => classic bullish divergence)

Puts (bearish divergence, mirror)
  F-BAND-PUT     z in [+1.0, +3.5]
  F-DIV-PUT      mirror: new local HIGH, lower-high in RSI

Three variants per side:

  A. DIV-only      F-BAND + F-DIV alone (loose band, divergence is the gate)
  B. DIV union R6  union with the live production signal
                   (calls: scan_candidates tier-rank; puts: _signals_put)
  C. DIV inside    F1 tight band [-2.5,-2.0] (or [+2.0,+2.5] puts)
     bounce_       + F5 support/resistance confluence (>=5 prior bars in 252d
     confirm        within +/-2% of recent 20b low/high)
                   + F-DIV  (replaces F2 reclaim / F3 higher-low / F4 stoch)

All variants share the live R6 exit rails (no TP, sigma-revert at z=0, $200
disaster cap, 45d time stop, 5% OTM, 45 DTE).

Cooldown COOLDOWN_BARS after each fire on the same ticker.

Outputs:
    research/out/divergence_sweep.png
    research/out/divergence_sweep.txt

GO/NO-GO
    Calls: must beat bounce_confirm combined-0.20:
           CAGR +15.5% | PF 1.71 | Sharpe 0.71 | MaxDD -23.8% | n=61
    Puts:  must beat shipped puts sleeve:
           PF 2.53 | CAGR +4.5% | MaxDD -17.1% | n=32
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from dataclasses import dataclass

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
    ExitCfg, simulate, _log_channel_z, _signals as baseline_call_signals,
    _fmt_with_n, Sig, SWEET_LO, SWEET_HI, LOG_WINDOW,
)
from research.volatile_universe_puts_sweep import (  # noqa: E402
    Sig as SigPut, simulate_put, _signals_put,
    SWEET_LO_PUT, SWEET_HI_PUT,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, HEADER, STARTING_CAPITAL, MIN_ADV_M,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# ---- Divergence config (fixed) ----
RSI_PERIOD = 14
RSI_DIVERG_MIN = 3.0           # RSI delta required for "higher low" / "lower high"
PIVOT_GAP = 3                  # min bars between current bar and the prior pivot
SUPPORT_PCT = 0.02             # variant-C F5 confluence width
SUPPORT_HITS_MIN = 5           # variant-C F5 confluence count
COOLDOWN_BARS = 20             # dedup: skip the next N bars after a fire on a ticker

# Wide "outside 1 sigma" gates for variant A; capped at +/-3.5 to skip the
# already-explored deep extreme tail (see hybrid_band_shipped).
CALL_BAND_LO, CALL_BAND_HI = -3.5, -1.0
PUT_BAND_LO, PUT_BAND_HI = 1.0, 3.5

LOOKBACKS = (10, 25, 60)


def _rsi_arr(closes: np.ndarray) -> np.ndarray:
    return rsi_series(pd.Series(closes), period=RSI_PERIOD).to_numpy(dtype=float)


def _divergence_signals_call(
    series: dict[str, Series],
    z_by_tk: dict[str, np.ndarray],
    lookback: int,
    *,
    band_lo: float,
    band_hi: float,
    tight_band_with_support: bool = False,
) -> dict[object, list[Sig]]:
    """date -> list[Sig] for the CALL leg.

    If tight_band_with_support=True (variant C), the band is forced to
    [SWEET_LO, SWEET_HI] and a 252d support-confluence filter is required
    on top of the divergence rule.
    """
    out: dict[object, list[Sig]] = {}

    if tight_band_with_support:
        lo, hi = SWEET_LO, SWEET_HI
    else:
        lo, hi = band_lo, band_hi

    start_min = (LOG_WINDOW if tight_band_with_support else lookback) + 2

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

            # Band gate
            if not (lo <= zi <= hi):
                continue

            # Divergence on a `lookback`-bar window ending at i
            if i - lookback + 1 < 0:
                continue
            win_lows = s.low[i - lookback + 1: i + 1]
            recent_lo = float(np.nanmin(win_lows))
            # New local low (ties allowed)
            if not np.isfinite(s.low[i]) or s.low[i] > recent_lo + 1e-12:
                continue

            # Prior pivot low excludes the last PIVOT_GAP bars (so we're
            # comparing two DISTINCT troughs).
            prior_end = i - PIVOT_GAP + 1   # exclusive
            prior_start = i - lookback + 1
            if prior_end - prior_start < 2:
                continue
            prior_slice = s.low[prior_start: prior_end]
            if not np.any(np.isfinite(prior_slice)):
                continue
            prior_arg = int(np.nanargmin(prior_slice))
            prior_idx = prior_start + prior_arg
            prior_low = float(s.low[prior_idx])

            # Price must be making a LOWER LOW than the prior pivot.
            if not np.isfinite(prior_low) or s.low[i] >= prior_low:
                continue

            # RSI higher-low (bullish divergence proper).
            ri = rsi_a[i]
            rp = rsi_a[prior_idx]
            if not (np.isfinite(ri) and np.isfinite(rp)):
                continue
            if ri < rp + RSI_DIVERG_MIN:
                continue

            # Variant C: support confluence (mirrors bounce_confirm F5).
            if tight_band_with_support:
                band_lo_p = recent_lo * (1.0 - SUPPORT_PCT)
                band_hi_p = recent_lo * (1.0 + SUPPORT_PCT)
                if i - LOG_WINDOW + 1 < 0:
                    continue
                prior_lows = s.low[i - LOG_WINDOW + 1: i - lookback + 1]
                hits = int(((prior_lows >= band_lo_p) & (prior_lows <= band_hi_p)).sum())
                if hits < SUPPORT_HITS_MIN:
                    continue

            # Liquidity / RV gates (match production).
            adv = s.adv_m[i]
            if not (np.isfinite(adv) and adv >= MIN_ADV_M):
                continue
            if not np.isfinite(s.rv[i]):
                continue

            sg = Sig(tk=tk, i=i, z=float(zi), date=s.dates[i],
                     tier="DIV", breadth=0, adv=float(adv))
            out.setdefault(s.dates[i], []).append(sg)
            last_fire_idx = i
    return out


def _divergence_signals_put(
    series: dict[str, Series],
    z_by_tk: dict[str, np.ndarray],
    lookback: int,
    *,
    band_lo: float,
    band_hi: float,
    tight_band_with_resistance: bool = False,
) -> dict[object, list[SigPut]]:
    """Mirror of _divergence_signals_call: bearish divergence on a new
    local high. Variant C uses tight [SWEET_LO_PUT, SWEET_HI_PUT] + 252d
    resistance confluence."""
    out: dict[object, list[SigPut]] = {}

    if tight_band_with_resistance:
        lo, hi = SWEET_LO_PUT, SWEET_HI_PUT
    else:
        lo, hi = band_lo, band_hi

    start_min = (LOG_WINDOW if tight_band_with_resistance else lookback) + 2

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
            if not (lo <= zi <= hi):
                continue

            if i - lookback + 1 < 0:
                continue
            win_highs = s.high[i - lookback + 1: i + 1]
            recent_hi = float(np.nanmax(win_highs))
            if not np.isfinite(s.high[i]) or s.high[i] < recent_hi - 1e-12:
                continue

            prior_end = i - PIVOT_GAP + 1
            prior_start = i - lookback + 1
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
            if ri > rp - RSI_DIVERG_MIN:
                continue

            if tight_band_with_resistance:
                band_lo_p = recent_hi * (1.0 - SUPPORT_PCT)
                band_hi_p = recent_hi * (1.0 + SUPPORT_PCT)
                if i - LOG_WINDOW + 1 < 0:
                    continue
                prior_highs = s.high[i - LOG_WINDOW + 1: i - lookback + 1]
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


def _combine(a, b):
    """Union by (date, ticker); 'a' wins on collision."""
    out = {d: list(lst) for d, lst in a.items()}
    for d, lst in b.items():
        seen = {sg.tk for sg in out.get(d, [])}
        for sg in lst:
            if sg.tk not in seen:
                out.setdefault(d, []).append(sg)
                seen.add(sg.tk)
    return out


def _pf(st):
    # calls sim uses 'pf'; puts sim uses 'profit_factor' -- accept both
    return st.get("pf") or st.get("profit_factor") or 0.0


def _cagr(st):
    return st.get("cagr", -1e9)


def _headline_block(rows, header_lbl):
    best_pf = max(rows, key=lambda kv: _pf(kv[1]))
    best_cagr = max(rows, key=lambda kv: _cagr(kv[1]))
    lines = [f"[{header_lbl} HEADLINE]"]
    lines.append(f"  best PF   : {best_pf[0]}   "
                 f"PF {_pf(best_pf[1]):.2f}  CAGR {_cagr(best_pf[1])*100:+.1f}%  "
                 f"Sharpe {best_pf[1].get('sharpe', 0):.2f}  "
                 f"MaxDD {best_pf[1].get('max_dd', 0)*100:.1f}%  "
                 f"n={best_pf[1].get('n', 0)}")
    lines.append(f"  best CAGR : {best_cagr[0]}   "
                 f"PF {_pf(best_cagr[1]):.2f}  CAGR {_cagr(best_cagr[1])*100:+.1f}%  "
                 f"Sharpe {best_cagr[1].get('sharpe', 0):.2f}  "
                 f"MaxDD {best_cagr[1].get('max_dd', 0)*100:.1f}%  "
                 f"n={best_cagr[1].get('n', 0)}")
    return lines


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

    # Live R6 exits (calls)
    R6 = ExitCfg(
        tp_pct=None,
        disaster_pct=None, disaster_usd=200.0,
        time_stop=45,
        sigma_target=0.0,
        iv_elevation=1.00, crush_td=30,
        otm_pct=5.0, dte=45,
    )
    # R6-mirror exits (puts) — same numeric config; sigma_target=0 means
    # close when overbought un-winds in the puts simulator.
    R6_PUT = ExitCfg(
        tp_pct=None,
        disaster_pct=None, disaster_usd=200.0,
        time_stop=45,
        sigma_target=0.0,
        iv_elevation=1.00, crush_td=30,
        otm_pct=5.0, dte=45,
    )

    base_call_sig = baseline_call_signals(series, z_by_tk)
    n_base_call = sum(len(v) for v in base_call_sig.values())
    base_put_sig = _signals_put(series, z_by_tk)
    n_base_put = sum(len(v) for v in base_put_sig.values())
    print(f"baseline calls (R6 live edge) n_sig = {n_base_call}")
    print(f"baseline puts  (R6 mirror)    n_sig = {n_base_put}")

    EHEADER = HEADER + "  | exit-reason mix"

    lines: list[str] = []
    lines.append("RSI(14) regular-divergence sweep on the VOLATILE THEMATIC universe.")
    lines.append(f"Divergence threshold {RSI_DIVERG_MIN:.1f} RSI pts; pivot gap {PIVOT_GAP} bars;")
    lines.append(f"cooldown {COOLDOWN_BARS} bars. Lookbacks {LOOKBACKS}.")
    lines.append(f"Calls band (DIV-only): z in [{CALL_BAND_LO}, {CALL_BAND_HI}] -- 'outside 1 sigma'.")
    lines.append(f"Puts  band (DIV-only): z in [{PUT_BAND_LO}, {PUT_BAND_HI}]   -- mirror.")
    lines.append(f"Variant C tight band: [{SWEET_LO}, {SWEET_HI}] calls / [{SWEET_LO_PUT}, {SWEET_HI_PUT}] puts + support confluence.")
    lines.append("Exit rails: R6 live (no TP, sigma|<=0, $200 disaster, 45td, 5% OTM, 45 DTE).")
    lines.append(f"Tickers with full 252d history: {len(series)}.\n")

    # ----------------------------------------------------------------
    # CALLS
    # ----------------------------------------------------------------
    call_rows_for_headline: list[tuple[str, dict]] = []
    call_plot_rows: list[tuple[str, dict]] = []

    base_call_st = simulate(series, z_by_tk, base_call_sig, R6)
    call_rows_for_headline.append(("R6-baseline-calls", base_call_st))
    call_plot_rows.append(("R6-baseline (live edge)", base_call_st))

    lines.append("=" * 78)
    lines.append("CALLS  (bullish RSI divergence)")
    lines.append("=" * 78)
    lines.append("[BASELINE -- live R6 edge (primary band + tier in OK/PRIME/PANIC)]")
    lines.append(EHEADER)
    lines.append(_fmt_with_n(f"R6-baseline-calls (n_sig={n_base_call})", base_call_st))
    lines.append("")

    lines.append("[A. DIV-only (loose 'outside 1 sigma' band; divergence is the gate)]")
    lines.append(EHEADER)
    div_call_sig_by_lb: dict[int, dict] = {}
    for lb in LOOKBACKS:
        sigs = _divergence_signals_call(
            series, z_by_tk, lookback=lb,
            band_lo=CALL_BAND_LO, band_hi=CALL_BAND_HI,
            tight_band_with_support=False,
        )
        div_call_sig_by_lb[lb] = sigs
        n = sum(len(v) for v in sigs.values())
        st = simulate(series, z_by_tk, sigs, R6)
        tag = f"DIV-only-calls lb={lb} (n_sig={n})"
        lines.append(_fmt_with_n(tag, st))
        call_rows_for_headline.append((tag, st))
        call_plot_rows.append((tag, st))
    lines.append("")

    lines.append("[B. DIV union R6-baseline (dedup by ticker+date)]")
    lines.append(EHEADER)
    for lb in LOOKBACKS:
        merged = _combine(base_call_sig, div_call_sig_by_lb[lb])
        n = sum(len(v) for v in merged.values())
        st = simulate(series, z_by_tk, merged, R6)
        tag = f"DIV+R6 union calls lb={lb} (n_sig={n})"
        lines.append(_fmt_with_n(tag, st))
        call_rows_for_headline.append((tag, st))
        call_plot_rows.append((tag, st))
    lines.append("")

    lines.append("[C. DIV inside bounce-confirm shell (tight band + support + divergence)]")
    lines.append(EHEADER)
    for lb in LOOKBACKS:
        sigs = _divergence_signals_call(
            series, z_by_tk, lookback=lb,
            band_lo=CALL_BAND_LO, band_hi=CALL_BAND_HI,  # ignored when tight=True
            tight_band_with_support=True,
        )
        n = sum(len(v) for v in sigs.values())
        st = simulate(series, z_by_tk, sigs, R6)
        tag = f"DIV-inside-bounce-calls lb={lb} (n_sig={n})"
        lines.append(_fmt_with_n(tag, st))
        call_rows_for_headline.append((tag, st))
        call_plot_rows.append((tag, st))
    lines.append("")

    lines.extend(_headline_block(call_rows_for_headline, "CALLS"))
    lines.append("  GO/NO-GO bar (bounce_confirm combined-0.20):")
    lines.append("    CAGR +15.5% | PF 1.71 | Sharpe 0.71 | MaxDD -23.8% | n=61")
    lines.append("")

    # ----------------------------------------------------------------
    # PUTS
    # ----------------------------------------------------------------
    put_rows_for_headline: list[tuple[str, dict]] = []
    put_plot_rows: list[tuple[str, dict]] = []

    base_put_st = simulate_put(series, z_by_tk, base_put_sig, R6_PUT)
    put_rows_for_headline.append(("R6-baseline-puts", base_put_st))
    put_plot_rows.append(("R6-baseline-puts", base_put_st))

    lines.append("=" * 78)
    lines.append("PUTS  (bearish RSI divergence, mirror)")
    lines.append("=" * 78)
    lines.append("[BASELINE -- existing puts breadth-tier signals + R6 exits]")
    lines.append(EHEADER)
    lines.append(_fmt_with_n(f"R6-baseline-puts (n_sig={n_base_put})", base_put_st))
    lines.append("")

    lines.append("[A. DIV-only (loose 'outside 1 sigma' band; divergence is the gate)]")
    lines.append(EHEADER)
    div_put_sig_by_lb: dict[int, dict] = {}
    for lb in LOOKBACKS:
        sigs = _divergence_signals_put(
            series, z_by_tk, lookback=lb,
            band_lo=PUT_BAND_LO, band_hi=PUT_BAND_HI,
            tight_band_with_resistance=False,
        )
        div_put_sig_by_lb[lb] = sigs
        n = sum(len(v) for v in sigs.values())
        st = simulate_put(series, z_by_tk, sigs, R6_PUT)
        tag = f"DIV-only-puts lb={lb} (n_sig={n})"
        lines.append(_fmt_with_n(tag, st))
        put_rows_for_headline.append((tag, st))
        put_plot_rows.append((tag, st))
    lines.append("")

    lines.append("[B. DIV union R6-baseline-puts (dedup by ticker+date)]")
    lines.append(EHEADER)
    for lb in LOOKBACKS:
        merged = _combine(base_put_sig, div_put_sig_by_lb[lb])
        n = sum(len(v) for v in merged.values())
        st = simulate_put(series, z_by_tk, merged, R6_PUT)
        tag = f"DIV+R6 union puts lb={lb} (n_sig={n})"
        lines.append(_fmt_with_n(tag, st))
        put_rows_for_headline.append((tag, st))
        put_plot_rows.append((tag, st))
    lines.append("")

    lines.append("[C. DIV inside bounce-confirm shell (tight band + resistance + divergence)]")
    lines.append(EHEADER)
    for lb in LOOKBACKS:
        sigs = _divergence_signals_put(
            series, z_by_tk, lookback=lb,
            band_lo=PUT_BAND_LO, band_hi=PUT_BAND_HI,
            tight_band_with_resistance=True,
        )
        n = sum(len(v) for v in sigs.values())
        st = simulate_put(series, z_by_tk, sigs, R6_PUT)
        tag = f"DIV-inside-bounce-puts lb={lb} (n_sig={n})"
        lines.append(_fmt_with_n(tag, st))
        put_rows_for_headline.append((tag, st))
        put_plot_rows.append((tag, st))
    lines.append("")

    lines.extend(_headline_block(put_rows_for_headline, "PUTS"))
    lines.append("  GO/NO-GO bar (shipped puts sleeve):")
    lines.append("    PF 2.53 | CAGR +4.5% | MaxDD -17.1% | n=32")
    lines.append("")

    # ----------------------------------------------------------------
    # Plot: one figure, two stacked panes (calls top, puts bottom).
    # ----------------------------------------------------------------
    fig, (ax_c, ax_p) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

    for tag, st in call_plot_rows:
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax_c.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.3,
                  label=f"{tag} (CAGR {st.get('cagr', 0)*100:+.1f}% | "
                        f"PF {_pf(st):.2f} | n={st.get('n', 0)})")
    ax_c.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax_c.set_title("CALLS -- RSI(14) bullish divergence variants vs R6 live edge")
    ax_c.set_ylabel("equity ($)")
    ax_c.legend(loc="upper left", fontsize=7, ncol=2)
    ax_c.grid(True, alpha=0.3)

    for tag, st in put_plot_rows:
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax_p.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.3,
                  label=f"{tag} (CAGR {st.get('cagr', 0)*100:+.1f}% | "
                        f"PF {_pf(st):.2f} | n={st.get('n', 0)})")
    ax_p.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax_p.set_title("PUTS -- RSI(14) bearish divergence variants vs R6-mirror baseline")
    ax_p.set_xlabel("date"); ax_p.set_ylabel("equity ($)")
    ax_p.legend(loc="upper left", fontsize=7, ncol=2)
    ax_p.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "divergence_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "divergence_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
