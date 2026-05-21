"""Sweep the LOG-channel REGRESSION WINDOW for the two winning divergence
variants. 252d is the bot's frame; check if a shorter (more responsive)
fit catches the same edge faster or breaks it.

Two variants tested at four windows each (=8 sim cells per side + 4
baselines = 12 calls / 12 puts):

  CALLS  R6-baseline only        z fresh cross into [-2.5,-2.0] + tier OK/PRIME/PANIC
         DIV-only (lb=60)        z in [-3.5,-1.0] + bullish RSI(14) divergence
         DIV+R6 union (lb=60)    union of the above (best from divergence_sweep)

  PUTS   R6-baseline only        z fresh cross into [+2.0,+2.5] + tier OK/PRIME/PANIC
         DIV-inside-bounce       z in [+2.0,+2.5] + 252d resistance + bearish div
                                 (rsi_thr=2, gap=8, lb=25) -- best from divergence_puts_tune

Windows swept: {60, 100, 150, 252}.

The band thresholds ([-2.5,-2.0] etc.) stay constant in standard-deviation
units; only the fitting window changes. A 60d window will have a tighter
sigma scale, so candidates that hit "-2σ" on a 60d fit are different from
those that hit "-2σ" on a 252d fit.

Output: research/out/divergence_window_sweep.png
        research/out/divergence_window_sweep.txt

GO/NO-GO at 252d (from prior runs):
  CALLS DIV+R6 union lb=60: CAGR +19.2% / PF 3.02 / Sharpe 0.88 / MaxDD -22.5% / n=83
  PUTS  DIV-inside rsi>=2 gap=8 lb=25: CAGR +5.7% / PF 3.10 / MaxDD -14.0% / n=35
"""
from __future__ import annotations

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
    ExitCfg, simulate, _fmt_with_n, Sig, SWEET_LO, SWEET_HI,
    TIER_THRESH, ACCEPT_TIERS,
)
from research.volatile_universe_puts_sweep import (  # noqa: E402
    Sig as SigPut, simulate_put, SWEET_LO_PUT, SWEET_HI_PUT,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, HEADER, STARTING_CAPITAL, MIN_ADV_M,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

WINDOWS = (60, 100, 150, 252)

# CALL divergence params (lb=60 winner) -- matches divergence_sweep.py
# CALL_DIV_PIVOT_GAP=3, not 8 -- the calls baseline never tuned the gap and
# the prior winner (PF 3.02 / CAGR +19.2%) was at the default gap=3.
CALL_DIV_LOOKBACK = 60
CALL_DIV_RSI_MIN = 3.0
CALL_DIV_PIVOT_GAP = 3
CALL_DIV_BAND_LO = -3.5
CALL_DIV_BAND_HI = -1.0

# PUT divergence params (rsi>=2 gap=8 lb=25 winner)
PUT_DIV_LOOKBACK = 25
PUT_DIV_RSI_MIN = 2.0
PUT_DIV_PIVOT_GAP = 8
SUPPORT_PCT = 0.02
SUPPORT_HITS_MIN = 5

COOLDOWN_BARS = 20
RSI_PERIOD = 14


def _log_channel_z_w(close: np.ndarray, window: int) -> np.ndarray:
    """Rolling LOG-channel z at arbitrary window."""
    n = len(close)
    z = np.full(n, np.nan)
    if n < window:
        return z
    y_all = np.log(close)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    xc = x - x_mean
    denom = float(np.sum(xc ** 2))
    for t in range(window - 1, n):
        chunk = y_all[t - window + 1: t + 1]
        y_mean = chunk.mean()
        slope = float(np.sum(xc * (chunk - y_mean)) / denom)
        intercept = y_mean - slope * x_mean
        fit_end = slope * (window - 1) + intercept
        sigma = float(np.std(chunk - (slope * x + intercept), ddof=1))
        if sigma > 0:
            z[t] = (y_all[t] - fit_end) / sigma
    return z


def _rsi_arr(closes: np.ndarray) -> np.ndarray:
    return rsi_series(pd.Series(closes), period=RSI_PERIOD).to_numpy(dtype=float)


def _tier_for(b: int) -> str:
    if b >= TIER_THRESH["PANIC"]: return "PANIC"
    if b >= TIER_THRESH["PRIME"]: return "PRIME"
    if b >= TIER_THRESH["OK"]:    return "OK"
    if b >= TIER_THRESH["WEAK"]:  return "WEAK"
    return "?"


def _r6_signals_call(series, z_by_tk) -> dict:
    """R6 calls baseline -- copy of _signals from oversold_call_exit_sweep,
    inlined so we can run it on z arrays computed at any window."""
    breadth_by_day: Counter = Counter()
    eligible: list[Sig] = []
    for tk, s in series.items():
        z = z_by_tk[tk]
        for i in range(1, len(s.close)):
            if np.isnan(z[i]) or np.isnan(z[i - 1]):
                continue
            crossed = z[i] <= SWEET_HI and z[i - 1] > SWEET_HI
            if crossed:
                breadth_by_day[s.dates[i]] += 1
            if not crossed:
                continue
            if not (SWEET_LO <= z[i] <= SWEET_HI):
                continue
            if np.isnan(s.rv[i]) or not (s.adv_m[i] >= MIN_ADV_M):
                continue
            eligible.append(Sig(tk=tk, i=i, z=float(z[i]), date=s.dates[i],
                                adv=float(s.adv_m[i])))
    out: dict = {}
    for sg in eligible:
        breadth = int(breadth_by_day.get(sg.date, 0))
        tier = _tier_for(breadth)
        if tier not in ACCEPT_TIERS:
            continue
        sg.tier = tier; sg.breadth = breadth
        out.setdefault(sg.date, []).append(sg)
    return out


def _r6_signals_put(series, z_by_tk) -> dict:
    """R6 puts baseline -- mirror of _r6_signals_call for the put leg."""
    breadth_by_day: Counter = Counter()
    eligible: list[SigPut] = []
    for tk, s in series.items():
        z = z_by_tk[tk]
        for i in range(1, len(s.close)):
            if np.isnan(z[i]) or np.isnan(z[i - 1]):
                continue
            crossed = z[i] >= SWEET_LO_PUT and z[i - 1] < SWEET_LO_PUT
            if crossed:
                breadth_by_day[s.dates[i]] += 1
            if not crossed:
                continue
            if not (SWEET_LO_PUT <= z[i] <= SWEET_HI_PUT):
                continue
            if np.isnan(s.rv[i]) or not (s.adv_m[i] >= MIN_ADV_M):
                continue
            eligible.append(SigPut(tk=tk, i=i, z=float(z[i]), date=s.dates[i],
                                   adv=float(s.adv_m[i])))
    out: dict = {}
    for sg in eligible:
        breadth = int(breadth_by_day.get(sg.date, 0))
        tier = _tier_for(breadth)
        if tier not in ACCEPT_TIERS:
            continue
        sg.tier = tier; sg.breadth = breadth
        out.setdefault(sg.date, []).append(sg)
    return out


def _divergence_signals_call(series, z_by_tk, window: int) -> dict:
    """Bullish RSI divergence call signal at arbitrary regression window."""
    out: dict = {}
    for tk, s in series.items():
        z = z_by_tk[tk]
        rsi_a = _rsi_arr(s.close)
        last_fire = -10_000
        start = window + CALL_DIV_LOOKBACK + 2
        for i in range(start, len(s.close)):
            if i - last_fire < COOLDOWN_BARS:
                continue
            zi = z[i]
            if np.isnan(zi):
                continue
            if not (CALL_DIV_BAND_LO <= zi <= CALL_DIV_BAND_HI):
                continue
            if i - CALL_DIV_LOOKBACK + 1 < 0:
                continue
            win_lows = s.low[i - CALL_DIV_LOOKBACK + 1: i + 1]
            recent_lo = float(np.nanmin(win_lows))
            if not np.isfinite(s.low[i]) or s.low[i] > recent_lo + 1e-12:
                continue
            prior_end = i - CALL_DIV_PIVOT_GAP + 1
            prior_start = i - CALL_DIV_LOOKBACK + 1
            if prior_end - prior_start < 2:
                continue
            prior_slice = s.low[prior_start: prior_end]
            if not np.any(np.isfinite(prior_slice)):
                continue
            prior_arg = int(np.nanargmin(prior_slice))
            prior_idx = prior_start + prior_arg
            prior_low = float(s.low[prior_idx])
            if not np.isfinite(prior_low) or s.low[i] >= prior_low:
                continue
            ri = rsi_a[i]; rp = rsi_a[prior_idx]
            if not (np.isfinite(ri) and np.isfinite(rp)):
                continue
            if ri < rp + CALL_DIV_RSI_MIN:
                continue
            adv = s.adv_m[i]
            if not (np.isfinite(adv) and adv >= MIN_ADV_M):
                continue
            if not np.isfinite(s.rv[i]):
                continue
            sg = Sig(tk=tk, i=i, z=float(zi), date=s.dates[i],
                     tier="DIV", breadth=0, adv=float(adv))
            out.setdefault(s.dates[i], []).append(sg)
            last_fire = i
    return out


def _divergence_signals_put(series, z_by_tk, window: int) -> dict:
    """Bearish RSI divergence put signal, Variant-C shell (tight band +
    resistance), at arbitrary regression window."""
    out: dict = {}
    for tk, s in series.items():
        z = z_by_tk[tk]
        rsi_a = _rsi_arr(s.close)
        last_fire = -10_000
        start = window + PUT_DIV_LOOKBACK + 2
        for i in range(start, len(s.close)):
            if i - last_fire < COOLDOWN_BARS:
                continue
            zi = z[i]
            if np.isnan(zi):
                continue
            if not (SWEET_LO_PUT <= zi <= SWEET_HI_PUT):
                continue
            if i - PUT_DIV_LOOKBACK + 1 < 0:
                continue
            win_highs = s.high[i - PUT_DIV_LOOKBACK + 1: i + 1]
            recent_hi = float(np.nanmax(win_highs))
            if not np.isfinite(s.high[i]) or s.high[i] < recent_hi - 1e-12:
                continue
            prior_end = i - PUT_DIV_PIVOT_GAP + 1
            prior_start = i - PUT_DIV_LOOKBACK + 1
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
            ri = rsi_a[i]; rp = rsi_a[prior_idx]
            if not (np.isfinite(ri) and np.isfinite(rp)):
                continue
            if ri > rp - PUT_DIV_RSI_MIN:
                continue
            band_lo_p = recent_hi * (1.0 - SUPPORT_PCT)
            band_hi_p = recent_hi * (1.0 + SUPPORT_PCT)
            if i - window + 1 < 0:
                continue
            prior_highs = s.high[i - window + 1: i - PUT_DIV_LOOKBACK + 1]
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
            last_fire = i
    return out


def _combine(a, b):
    out = {d: list(lst) for d, lst in a.items()}
    for d, lst in b.items():
        seen = {sg.tk for sg in out.get(d, [])}
        for sg in lst:
            if sg.tk not in seen:
                out.setdefault(d, []).append(sg)
                seen.add(sg.tk)
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
    for tk, df in bars.items():
        s = build_series(tk, df)
        if s is not None:
            series[tk] = s
    print(f"  {len(series)} tickers built\n")

    # Precompute z per window
    z_by_window: dict[int, dict[str, np.ndarray]] = {}
    for w in WINDOWS:
        print(f"  computing z at window={w}...")
        z_by_window[w] = {tk: _log_channel_z_w(s.close, w) for tk, s in series.items()}

    R6 = ExitCfg(
        tp_pct=None, disaster_pct=None, disaster_usd=200.0,
        time_stop=45, sigma_target=0.0,
        iv_elevation=1.00, crush_td=30, otm_pct=5.0, dte=45,
    )

    EHEADER = HEADER + "  | exit-reason mix"

    lines: list[str] = []
    lines.append("DIVERGENCE x REGRESSION-WINDOW sweep on the volatile universe.")
    lines.append(f"Windows tested: {WINDOWS}.  R6 exits everywhere.")
    lines.append(f"Universe: {len(series)} tickers, 5y history.")
    lines.append("")

    # -------- CALLS --------
    lines.append("=" * 78)
    lines.append("CALLS  (DIV+R6 union vs R6-only at each window; lb=60 divergence)")
    lines.append("=" * 78)
    lines.append(EHEADER)

    call_curves: list[tuple[str, dict]] = []
    call_rows: list[tuple[str, dict]] = []
    for w in WINDOWS:
        z_by_tk = z_by_window[w]
        base_sig = _r6_signals_call(series, z_by_tk)
        n_base = sum(len(v) for v in base_sig.values())
        base_st = simulate(series, z_by_tk, base_sig, R6)
        tag = f"w={w:>3} R6-only (n_sig={n_base})"
        lines.append(_fmt_with_n(tag, base_st))
        call_curves.append((tag, base_st)); call_rows.append((tag, base_st))

        div_sig = _divergence_signals_call(series, z_by_tk, w)
        n_div = sum(len(v) for v in div_sig.values())
        union = _combine(base_sig, div_sig)
        n_union = sum(len(v) for v in union.values())
        union_st = simulate(series, z_by_tk, union, R6)
        tag = f"w={w:>3} DIV+R6 union (n_div={n_div}, n_union={n_union})"
        lines.append(_fmt_with_n(tag, union_st))
        call_curves.append((tag, union_st)); call_rows.append((tag, union_st))
        lines.append("")

    # -------- PUTS --------
    lines.append("=" * 78)
    lines.append("PUTS  (DIV-inside-bounce rsi>=2 gap=8 lb=25 vs R6-only at each window)")
    lines.append("=" * 78)
    lines.append(EHEADER)

    put_curves: list[tuple[str, dict]] = []
    put_rows: list[tuple[str, dict]] = []
    for w in WINDOWS:
        z_by_tk = z_by_window[w]
        base_sig = _r6_signals_put(series, z_by_tk)
        n_base = sum(len(v) for v in base_sig.values())
        base_st = simulate_put(series, z_by_tk, base_sig, R6)
        tag = f"w={w:>3} R6-baseline-puts (n_sig={n_base})"
        lines.append(_fmt_with_n(tag, base_st))
        put_curves.append((tag, base_st)); put_rows.append((tag, base_st))

        div_sig = _divergence_signals_put(series, z_by_tk, w)
        n_div = sum(len(v) for v in div_sig.values())
        div_st = simulate_put(series, z_by_tk, div_sig, R6)
        tag = f"w={w:>3} DIV-inside-bounce (n_sig={n_div})"
        lines.append(_fmt_with_n(tag, div_st))
        put_curves.append((tag, div_st)); put_rows.append((tag, div_st))
        lines.append("")

    # Headlines per side
    lines.append("[CALLS HEADLINE -- best across all windows]")
    best_pf = max(call_rows, key=lambda kv: _pf(kv[1]))
    best_cagr = max(call_rows, key=lambda kv: kv[1].get("cagr", -1e9))
    for label, row in (("best PF", best_pf), ("best CAGR", best_cagr)):
        st = row[1]
        lines.append(
            f"  {label} : {row[0]}   PF {_pf(st):.2f}  CAGR {st.get('cagr', 0)*100:+.1f}%  "
            f"Sharpe {st.get('sharpe', 0):.2f}  MaxDD {st.get('max_dd', 0)*100:.1f}%  n={st.get('n', 0)}"
        )
    lines.append("  Reference (w=252 DIV+R6 union, prior result): PF 3.02  CAGR +19.2%  Sharpe 0.88  MaxDD -22.5%  n=83")
    lines.append("")

    lines.append("[PUTS HEADLINE -- best across all windows]")
    best_pf = max(put_rows, key=lambda kv: _pf(kv[1]))
    best_cagr = max(put_rows, key=lambda kv: kv[1].get("cagr", -1e9))
    for label, row in (("best PF", best_pf), ("best CAGR", best_cagr)):
        st = row[1]
        lines.append(
            f"  {label} : {row[0]}   PF {_pf(st):.2f}  CAGR {st.get('cagr', 0)*100:+.1f}%  "
            f"Sharpe {st.get('sharpe', 0):.2f}  MaxDD {st.get('max_dd', 0)*100:.1f}%  n={st.get('n', 0)}"
        )
    lines.append("  Reference (w=252 DIV-inside, prior result): PF 3.10  CAGR +5.7%  Sharpe 0.54  MaxDD -14.0%  n=35")
    lines.append("")

    # Plot: two-pane figure
    fig, (ax_c, ax_p) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    for tag, st in call_curves:
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax_c.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.2,
                  label=f"{tag} (CAGR {st.get('cagr', 0)*100:+.1f}% | PF {_pf(st):.2f} | n={st.get('n', 0)})")
    ax_c.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax_c.set_title("CALLS -- DIV+R6 union vs R6-only across regression windows")
    ax_c.set_ylabel("equity ($)")
    ax_c.legend(loc="upper left", fontsize=7, ncol=2)
    ax_c.grid(True, alpha=0.3)

    for tag, st in put_curves:
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax_p.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.2,
                  label=f"{tag} (CAGR {st.get('cagr', 0)*100:+.1f}% | PF {_pf(st):.2f} | n={st.get('n', 0)})")
    ax_p.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax_p.set_title("PUTS -- DIV-inside-bounce vs R6-baseline across regression windows")
    ax_p.set_xlabel("date"); ax_p.set_ylabel("equity ($)")
    ax_p.legend(loc="upper left", fontsize=7, ncol=2)
    ax_p.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "divergence_window_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "divergence_window_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
