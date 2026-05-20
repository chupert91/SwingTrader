"""Parabolic-exhaustion put backtest on the 5y volatile universe.

Reverse-engineering the user's PF 3.80 hand-picked 2025 puts surfaced
that the entry signal is NOT z-band reversion. It is a parabolic
exhaustion setup: stock has run up sharply (10-day return, distance
from SMA20) into overbought oscillators. See
  research/out/personal_puts_pattern_analysis.txt
  research/out/puts_threshold_sensitivity.txt

This script tests three filter variants on the full 5y volatile-
universe sim, each crossed with three exit configs tuned for fast
reversal capture (user's real puts had 2d median hold).

Filter variants (from the threshold sweep on the user's 20 trades):
  A. ret_10d >= 15  AND  dist_sma20 >= 15        ("the clean pair")
  B. ret_20d >= 10                                ("high recall")
  C. ret_10d >= 15  AND  dist_sma20 >= 15  AND  RSI(14) >= 70
                                                  ("conservative")

Exit configs (mirror of the call leg's R6, plus fast variants):
  FAST20   TP +20%, $200 cap, 5d time stop, IV 1.00/30
  FAST15   TP +15%, $200 cap, 3d time stop, IV 1.00/30
  R6-MIR   no TP, sigma <= 0 close, $200 cap, 45d time, IV 1.00/30

Signal mechanics (first-cross + cooldown):
  Fire on bar i if filter(i)=True AND filter(i-1)=False (i.e. the day
  the parabolic state begins). Cooldown 20 bars per ticker — same name
  can re-fire if its blow-off cleared and resumed.

Ship gate: any filter x exit cell with PF >= 1.5 AND CAGR > 0 on the
full 5y sim. This is the FORWARD test that puts_threshold_sensitivity
identified as the hypothesis.

Run:
    python research/puts_parabolic_exhaustion_sweep.py
Outputs:
    research/out/puts_parabolic_exhaustion_sweep.png
    research/out/puts_parabolic_exhaustion_sweep.txt
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.volatile_universe import universe, theme_of  # noqa: E402
from research.oversold_call_exit_sweep import (  # noqa: E402
    ExitCfg, _log_channel_z, _tier_for, _fmt_with_n,
    ACCEPT_TIERS, TIER_RANK,
)
from research.volatile_universe_puts_sweep import (  # noqa: E402
    simulate_put, Sig as _PutSig,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, _stats, HEADER, STARTING_CAPITAL, MIN_ADV_M,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

COOLDOWN_BARS = 20    # don't re-fire on the same name within 20 bars
RSI_PERIOD = 14
SMA20 = 20
LOG_WINDOW = 252


# ---------- per-bar feature computation ---------------------------------

def _features_per_bar(closes: np.ndarray) -> dict:
    """Vectorized per-bar features for the parabolic-exhaustion signal."""
    n = len(closes)
    out = {}

    # %returns over lookback k: (close[i]/close[i-k] - 1) * 100
    def _ret(k: int) -> np.ndarray:
        r = np.full(n, np.nan)
        if n > k:
            r[k:] = (closes[k:] / closes[:n - k] - 1.0) * 100.0
        return r

    out["ret_5d"] = _ret(5)
    out["ret_10d"] = _ret(10)
    out["ret_20d"] = _ret(20)

    # SMA(20) distance.
    sma20 = np.full(n, np.nan)
    if n >= SMA20:
        cs = np.cumsum(closes, dtype=float)
        cs = np.concatenate(([0.0], cs))
        sma20[SMA20 - 1:] = (cs[SMA20:] - cs[:n - SMA20 + 1]) / SMA20
    out["sma20"] = sma20
    out["dist_sma20"] = np.where(sma20 > 0, (closes / sma20 - 1.0) * 100.0, np.nan)

    # RSI (Wilder, vectorized).
    delta = np.diff(closes, prepend=closes[0])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    rsi = np.full(n, np.nan)
    if n >= RSI_PERIOD + 1:
        au = up[1:RSI_PERIOD + 1].mean()
        ad = dn[1:RSI_PERIOD + 1].mean()
        for i in range(RSI_PERIOD, n):
            if i > RSI_PERIOD:
                au = (au * (RSI_PERIOD - 1) + up[i]) / RSI_PERIOD
                ad = (ad * (RSI_PERIOD - 1) + dn[i]) / RSI_PERIOD
            if ad <= 0:
                rsi[i] = 100.0
            else:
                rs = au / ad
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    out["rsi"] = rsi
    return out


# ---------- signal generators -------------------------------------------

@dataclass
class FilterDef:
    name: str
    fn: object   # callable(features_dict, i) -> bool


def _filter_A(fb: dict, i: int) -> bool:
    """ret_10d >= 15 AND dist_sma20 >= 15"""
    if np.isnan(fb["ret_10d"][i]) or np.isnan(fb["dist_sma20"][i]):
        return False
    return fb["ret_10d"][i] >= 15.0 and fb["dist_sma20"][i] >= 15.0


def _filter_B(fb: dict, i: int) -> bool:
    """ret_20d >= 10"""
    if np.isnan(fb["ret_20d"][i]):
        return False
    return fb["ret_20d"][i] >= 10.0


def _filter_C(fb: dict, i: int) -> bool:
    """A + RSI(14) >= 70"""
    if not _filter_A(fb, i):
        return False
    if np.isnan(fb["rsi"][i]):
        return False
    return fb["rsi"][i] >= 70.0


def _signals_parabolic(series: dict[str, Series], filt: FilterDef) -> dict:
    """date -> list[Sig]. Fire on first-cross into the filter state with
    a per-ticker cooldown. Tier ranking is direction-agnostic: breadth
    counts how many names entered the parabolic state that day."""
    # Pre-compute features and per-bar parabolic state for every ticker.
    feat: dict[str, dict] = {}
    state: dict[str, np.ndarray] = {}
    for tk, s in series.items():
        fb = _features_per_bar(s.close)
        feat[tk] = fb
        st = np.zeros(len(s.close), dtype=bool)
        for i in range(len(s.close)):
            st[i] = filt.fn(fb, i)
        state[tk] = st

    # Breadth = same-day count of FIRST-CROSS events (state transitions
    # False -> True). Mirrors how the call leg's breadth counts crossings
    # into the band — a market-wide euphoria-peak day registers high.
    breadth_by_day: Counter = Counter()
    raw: list[tuple] = []   # (date, sg)
    for tk, s in series.items():
        st = state[tk]
        last_fire = -10**9
        for i in range(1, len(s.close)):
            if not st[i] or st[i - 1]:
                continue
            if i - last_fire < COOLDOWN_BARS:
                continue
            # eligibility gates
            if np.isnan(s.rv[i]) or not (s.adv_m[i] >= MIN_ADV_M):
                continue
            breadth_by_day[s.dates[i]] += 1
            raw.append((s.dates[i], tk, i, float(s.rv[i]), float(s.adv_m[i])))
            last_fire = i

    out: dict = {}
    for d, tk, i, rv, adv in raw:
        breadth = int(breadth_by_day.get(d, 0))
        tier = _tier_for(breadth)
        if tier not in ACCEPT_TIERS:
            continue
        out.setdefault(d, []).append(_PutSig(
            tk=tk, i=i, z=0.0, date=d, tier=tier, breadth=breadth, adv=adv,
        ))
    return out


# ---------- runner ------------------------------------------------------

FILTERS = [
    FilterDef("A: ret_10d>=15 AND dist_sma20>=15", _filter_A),
    FilterDef("B: ret_20d>=10 (high recall)",        _filter_B),
    FilterDef("C: A + RSI>=70",                        _filter_C),
]

EXITS = [
    ("FAST20 (TP20, t5, $200)",
     ExitCfg(tp_pct=20.0, disaster_pct=None, disaster_usd=200.0,
             time_stop=5, sigma_target=None,
             iv_elevation=1.00, crush_td=30)),
    ("FAST15 (TP15, t3, $200)",
     ExitCfg(tp_pct=15.0, disaster_pct=None, disaster_usd=200.0,
             time_stop=3, sigma_target=None,
             iv_elevation=1.00, crush_td=30)),
    ("R6-MIR (no-TP, sigma<=0, t45)",
     ExitCfg(tp_pct=None, disaster_pct=None, disaster_usd=200.0,
             time_stop=45, sigma_target=0.0,
             iv_elevation=1.00, crush_td=30)),
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tks = universe()
    print(f"Fetching {len(tks)} volatile-universe tickers x 5y...")
    bars = fetch_bars_bulk(tks, period="5y")
    series: dict[str, Series] = {}
    z_by_tk: dict[str, np.ndarray] = {}
    for tk, df in bars.items():
        s = build_series(tk, df)
        if s is not None:
            series[tk] = s
            z_by_tk[tk] = _log_channel_z(s.close)
    print(f"  {len(series)} tickers with full 252d history\n")

    EHEADER = HEADER + "  | exit-reason mix"
    lines: list[str] = []
    lines.append("PUTS PARABOLIC-EXHAUSTION SWEEP on volatile universe (5y).")
    lines.append("Forward test of the hypothesis from puts_threshold_sensitivity:")
    lines.append("  'shorting blow-off tops on volatile names is what worked for the user'")
    lines.append(f"Tickers: {len(series)}.")
    lines.append(f"Cooldown: {COOLDOWN_BARS} bars per ticker (no re-fire while still parabolic).")
    lines.append(f"Ship gate: PF >= 1.5 AND CAGR > 0 on the 5y sim.")
    lines.append("")

    all_results: list[tuple[str, str, dict, int]] = []  # (filter, exit, st, n_sig)
    sig_caches: dict[str, dict] = {}
    for filt in FILTERS:
        sig_by_day = _signals_parabolic(series, filt)
        sig_caches[filt.name] = sig_by_day
        n_sig = sum(len(v) for v in sig_by_day.values())

        # Signals-per-theme breakdown for context.
        by_theme: dict[str, int] = defaultdict(int)
        for sigs in sig_by_day.values():
            for sg in sigs:
                by_theme[theme_of(sg.tk)] += 1
        theme_str = ", ".join(f"{k}={v}" for k, v in sorted(by_theme.items(), key=lambda kv: -kv[1])[:5])

        lines.append(f"[FILTER {filt.name}]   raw_signals={n_sig}   themes(top5): {theme_str}")
        lines.append(EHEADER)
        for ex_label, cfg in EXITS:
            st = simulate_put(series, z_by_tk, sig_by_day, cfg)
            all_results.append((filt.name, ex_label, st, n_sig))
            lines.append(_fmt_with_n(ex_label, st))
        lines.append("")

    # Ship-gate passers.
    passers = [(f, e, st) for f, e, st, _ in all_results
               if (st.get("profit_factor", 0) or 0) >= 1.5
               and (st.get("cagr", 0) or 0) > 0
               and st.get("n", 0) >= 5]
    by_pf = max(all_results, key=lambda r: r[2].get("profit_factor", 0) or 0)
    by_cagr = max(all_results, key=lambda r: r[2].get("cagr", -1e9))
    by_sharpe = max(all_results, key=lambda r: r[2].get("sharpe", -1e9))

    lines.append("[SHIP-GATE PASSERS]  (PF >= 1.5 AND CAGR > 0 AND n >= 5)")
    if not passers:
        lines.append("  (none — hypothesis falsified on the 5y volatile-universe sim)")
    else:
        for f, e, st in sorted(passers, key=lambda x: -(x[2].get("profit_factor", 0) or 0)):
            lines.append(f"  {f}  +  {e}")
            lines.append(f"     CAGR {st['cagr']*100:+.1f}%  PF {st.get('profit_factor', 0):.2f}  "
                         f"Sharpe {st.get('sharpe', 0):.2f}  maxDD {st.get('max_dd', 0)*100:.0f}%  n={st['n']}")
    lines.append("")
    lines.append("[OVERALL BEST CELLS]")
    lines.append(f"  by CAGR   : {by_cagr[0]}  +  {by_cagr[1]}  ->  "
                 f"CAGR {by_cagr[2]['cagr']*100:+.1f}%  PF {by_cagr[2].get('profit_factor', 0):.2f}  "
                 f"n={by_cagr[2]['n']}")
    lines.append(f"  by PF     : {by_pf[0]}  +  {by_pf[1]}  ->  "
                 f"CAGR {by_pf[2]['cagr']*100:+.1f}%  PF {by_pf[2].get('profit_factor', 0):.2f}  "
                 f"n={by_pf[2]['n']}")
    lines.append(f"  by Sharpe : {by_sharpe[0]}  +  {by_sharpe[1]}  ->  "
                 f"CAGR {by_sharpe[2]['cagr']*100:+.1f}%  PF {by_sharpe[2].get('profit_factor', 0):.2f}  "
                 f"Sharpe {by_sharpe[2].get('sharpe', 0):.2f}  n={by_sharpe[2]['n']}")

    # Plot best-per-filter equity curves.
    fig, ax = plt.subplots(figsize=(14, 7))
    best_per_filter: dict = {}
    for f, e, st, _ in all_results:
        cur = best_per_filter.get(f)
        if cur is None or (st.get("profit_factor", 0) or 0) > (cur[1].get("profit_factor", 0) or 0):
            best_per_filter[f] = (e, st)
    for f, (e, st) in best_per_filter.items():
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.5,
                label=f"{f} | best: {e} (PF {st.get('profit_factor', 0):.2f}, "
                      f"CAGR {st['cagr']*100:+.1f}%, n={st['n']})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title("Puts parabolic-exhaustion sweep — best exit per filter on volatile universe")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "puts_parabolic_exhaustion_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "puts_parabolic_exhaustion_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
