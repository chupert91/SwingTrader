"""Mirror of research/bounce_confirm_sweep.py for the PUT leg.

The user's chart-eye reciprocal of the bounce-confirm pattern (overbought
distribution top with structural-resistance rejection + Stoch RSI rolling
over) was NOT among the 9 sweeps that falsified the predictive put leg
(see [[put_leg_falsified]]: PF ceiling 0.91 across extreme-band, level-
retest, time-stop, breather-scalp, parabolic-exhaustion, underlying-target,
ticker-specific level-retest, v3 level-entry). The original falsification
memory explicitly flags "consolidation-top level detection" as one of three
gaps worth revisiting -- F3m (lower high) + F5m (resistance confluence)
together are exactly that.

Five features, all must be true (direct mirrors of bounce_confirm F1-F5):

  F1m  z in mirrored band [+2.0, +2.5]
  F2m  z pullback >= 0.20σ from the 20b z-MAX (price rolling off the top)
  F3m  LOWER high (252d):  max(high, last 20b) < max(high, prior 232b)
  F4m  Stoch RSI %K >= 70 AND %K < max(%K, last 20b) - STOCH_TURN_MIN
  F5m  resistance confluence: >= 5 prior bars in 252d window with high
       within +/- 2% of the recent 20b high

Cooldown COOLDOWN_BARS after each fire to avoid spamming.

Same exit rails as the calls bot (R6-mirrored): no TP, sigma-revert at
z<=0 (puts engine fires when z_by_tk[idx] <= -sigma_target, so set 0.0
to mean "close when overbought is fully un-wound"), $200 disaster cap,
45d time stop, 5% OTM put (strike = 0.95 * spot), dte 45.

Outputs:
    research/out/bounce_confirm_puts_sweep.png
    research/out/bounce_confirm_puts_sweep.txt

Comparison block:
    BASELINE-PUTS   -- existing puts breadth-tier signals (the v1 leg)
    BOUNCE-PUTS     -- only the 5-of-5 mirror pattern
    COMBINED-PUTS   -- BASELINE union BOUNCE-PUTS (dedup by ticker+date)
F2m sweep over RECLAIM_SIGMA in {0.05, 0.10, 0.20, 0.30}.

GO/NO-GO: ship only if COMBINED-PUTS PF >= 1.5 AND CAGR >= 0 on at least
one reclaim level. The other 9 puts sweeps maxed at PF 0.91 -- this is
the threshold to declare the "consolidation-top" gap closed positively.
"""
from __future__ import annotations

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

from backend.indicators import stoch_rsi  # noqa: E402
from backend.data import fetch_bars_bulk  # noqa: E402
from research.volatile_universe import universe  # noqa: E402
from research.oversold_call_exit_sweep import (  # noqa: E402
    ExitCfg, _log_channel_z, _fmt_with_n, LOG_WINDOW,
)
from research.volatile_universe_puts_sweep import (  # noqa: E402
    Sig, simulate_put, _signals_put, SWEET_LO_PUT, SWEET_HI_PUT,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, HEADER, STARTING_CAPITAL, MIN_ADV_M,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Mirror of bounce_confirm constants.
LOOKBACK = 20
PRIOR_WINDOW = LOG_WINDOW
SUPPORT_PCT = 0.02          # F5m: resistance confluence band +/- 2% of recent 20b high
SUPPORT_HITS_MIN = 5
STOCH_OB_MIN = 70.0         # F4m: %K must be >= this (overbought)
STOCH_TURN_MIN = 1.0        # F4m: %K must roll over by at least this from 20b max
COOLDOWN_BARS = 20


def _bounce_signals_put(
    series: dict[str, Series],
    z_by_tk: dict[str, np.ndarray],
    reclaim_sigma: float,
) -> dict[object, list[Sig]]:
    """date -> list[Sig] for the put leg. Fires on first bar where the 5
    mirror features all hold, then cooldown for COOLDOWN_BARS."""
    out: dict[object, list[Sig]] = {}
    for tk, s in series.items():
        z = z_by_tk[tk]
        k_ser, _ = stoch_rsi(pd.Series(s.close))
        k_arr = k_ser.to_numpy(dtype=float)

        last_fire_idx = -10_000
        start = PRIOR_WINDOW + LOOKBACK + 2
        for i in range(start, len(s.close)):
            if i - last_fire_idx < COOLDOWN_BARS:
                continue
            zi = z[i]
            if np.isnan(zi):
                continue

            # F1m: mirrored band [+2.0, +2.5]
            if not (SWEET_LO_PUT <= zi <= SWEET_HI_PUT):
                continue

            # F2m: z PULLBACK >= reclaim_sigma from the 20b z-MAX
            recent_z_max = float(np.nanmax(z[i - LOOKBACK + 1: i + 1]))
            if zi > recent_z_max - reclaim_sigma:
                continue

            # F3m: LOWER high on 252d
            recent_hi = float(np.nanmax(s.high[i - LOOKBACK + 1: i + 1]))
            prior_hi = float(np.nanmax(s.high[i - PRIOR_WINDOW + 1: i - LOOKBACK + 1]))
            if not np.isfinite(prior_hi) or recent_hi >= prior_hi:
                continue

            # F4m: Stoch RSI overbought AND rolling down
            ki = k_arr[i]
            if not np.isfinite(ki) or ki < STOCH_OB_MIN:
                continue
            k_max_recent = float(np.nanmax(k_arr[i - LOOKBACK + 1: i + 1]))
            if ki >= k_max_recent - STOCH_TURN_MIN:
                continue

            # F5m: resistance confluence
            band_lo = recent_hi * (1.0 - SUPPORT_PCT)
            band_hi = recent_hi * (1.0 + SUPPORT_PCT)
            prior_highs = s.high[i - PRIOR_WINDOW + 1: i - LOOKBACK + 1]
            hits = int(((prior_highs >= band_lo) & (prior_highs <= band_hi)).sum())
            if hits < SUPPORT_HITS_MIN:
                continue

            # Liquidity gate.
            adv = s.adv_m[i]
            if not (np.isfinite(adv) and adv >= MIN_ADV_M):
                continue
            if not np.isfinite(s.rv[i]):
                continue

            sg = Sig(tk=tk, i=i, z=float(zi), date=s.dates[i],
                     tier="BOUNCE", breadth=hits, adv=float(adv))
            out.setdefault(s.dates[i], []).append(sg)
            last_fire_idx = i
    return out


def _combine(a: dict[object, list[Sig]],
             b: dict[object, list[Sig]]) -> dict[object, list[Sig]]:
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
    print(f"  {len(series)} tickers with full 252d history\n")

    # R6-mirrored put exits: no TP, sigma_target=0 (z<=0 closes when over-
    # bought un-winds), $200 disaster, 45td, 5% OTM put, dte 45.
    R6_PUT = ExitCfg(
        tp_pct=None,
        disaster_pct=None, disaster_usd=200.0,
        time_stop=45,
        sigma_target=0.0,
        iv_elevation=1.00, crush_td=30,
        otm_pct=5.0, dte=45,
    )

    base_sig = _signals_put(series, z_by_tk)
    n_base = sum(len(v) for v in base_sig.values())
    print(f"baseline puts breadth-tier signals: {n_base}")

    lines: list[str] = []
    lines.append("Bounce-confirm MIRROR sweep on the PUT leg (volatile universe).")
    lines.append("F1m z in [+2.0,+2.5], F2m pullback from 20b z-max, F3m LOWER high")
    lines.append("(252d), F4m Stoch RSI >=70 AND rolling down, F5m >=5 prior highs")
    lines.append(f"within +/-{SUPPORT_PCT*100:.1f}% of recent 20b high. Cooldown {COOLDOWN_BARS}b.")
    lines.append("Exit rails: R6-mirror (no TP, sigma<=0, $200 disaster, 45td, 5% OTM put, 45 DTE).")
    lines.append(f"Tickers with full 252d history: {len(series)}.")
    lines.append("GO/NO-GO: ship only if COMBINED PF >= 1.5 AND CAGR >= 0 on at least one")
    lines.append("reclaim level. Other 9 puts sweeps maxed at PF 0.91 -- this is the bar.\n")

    EHEADER = HEADER + "  | exit-reason mix"

    base_st = simulate_put(series, z_by_tk, base_sig, R6_PUT)
    lines.append("[BASELINE PUTS -- existing breadth-tier signals + R6 exits]")
    lines.append(EHEADER)
    lines.append(_fmt_with_n(f"baseline (n_sig={n_base})", base_st))
    lines.append("")

    sweep_results: list[tuple[str, dict]] = []
    bounce_sig_by_recl: dict[float, dict] = {}
    lines.append("[BOUNCE-PUTS F2m SWEEP (no baseline included)]")
    lines.append(EHEADER)
    for recl in (0.05, 0.10, 0.20, 0.30):
        sigs = _bounce_signals_put(series, z_by_tk, reclaim_sigma=recl)
        bounce_sig_by_recl[recl] = sigs
        n = sum(len(v) for v in sigs.values())
        st = simulate_put(series, z_by_tk, sigs, R6_PUT)
        tag = f"bounce-puts reclaim>={recl:.2f} (n_sig={n})"
        sweep_results.append((tag, st))
        lines.append(_fmt_with_n(tag, st))
    lines.append("")

    lines.append("[COMBINED PUTS -- baseline UNION bounce-puts]")
    lines.append(EHEADER)
    combined_results: list[tuple[str, dict]] = []
    for recl, sigs in bounce_sig_by_recl.items():
        merged = _combine(base_sig, sigs)
        n = sum(len(v) for v in merged.values())
        st = simulate_put(series, z_by_tk, merged, R6_PUT)
        tag = f"combined-puts reclaim>={recl:.2f} (n_sig={n})"
        combined_results.append((tag, st))
        lines.append(_fmt_with_n(tag, st))
    lines.append("")

    def pf(st): return st.get("profit_factor", 0) or 0
    def cagr(st): return st.get("cagr", -1e9)
    all_rows = ([("baseline-puts", base_st)]
                + sweep_results
                + combined_results)
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
    lines.append("[REFERENCE FROM CALLS BOUNCE-CONFIRM SWEEP]")
    lines.append("  calls combined-0.20: CAGR +15.5% | PF 1.71 | Sharpe 0.71 | MaxDD -23.8% | n=61")
    lines.append("  (research/out/bounce_confirm_sweep.txt -- comparable apples-to-apples).")

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_rows = [("baseline-puts (existing leg)", base_st)] + sweep_results + combined_results
    for tag, st in plot_rows:
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.4,
                label=f"{tag} (CAGR {st.get('cagr', 0)*100:+.1f}% | PF "
                      f"{pf(st):.2f} | n={st.get('n', 0)})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title("Bounce-confirm MIRROR (puts) -- volatile universe, R6-mirror exits")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "bounce_confirm_puts_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "bounce_confirm_puts_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
