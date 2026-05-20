"""Call-leg HYBRID-band sweep.

The single-band sigma sweep (research/out/sigma_band_sweep.txt) showed:
  [-2.5, -2.0]   PF 2.32   CAGR +13%   the live default
  [-3.5, -3.0]   PF 1.04   CAGR -18%   DEAD ZONE
  [-4.0, -3.5]   PF 12.59  CAGR +30%   GOLD (n=15)
  open >= -3.5   PF  6.91  CAGR +27%   open-ended deep

Question: can a UNION band (primary [-2.5,-2.0] + secondary z<=-3.5,
explicitly skipping the dead zone [-3.0,-3.5]) beat the current single
band while using the same 2-slot capital frame?

This is a capital-constrained sim — if both signals fire on the same
day they compete for the same slot, so the headline number is honest.

Rows:
  PRIMARY-ONLY        [-2.5, -2.0]                    baseline (live)
  DEEP-ONLY-3.5       z <= -3.5 (open)
  DEEP-ONLY-4.0       z <= -4.0 (open, ultra-rare)
  HYBRID 2.5+3.5      [-2.5,-2.0] OR z<=-3.5
  HYBRID 2.5+4.0      [-2.5,-2.0] OR z<=-4.0  (avoid the marginal -3.5)
  WIDE [-inf,-2.0]    open: anything past -2.0 (for context)
  HYBRID-RANK         primary preferred when both fire on same day
                      (instead of generic tier rank). Tests whether
                      slot-allocation matters.

Run:
    python research/calls_hybrid_band_sweep.py
Outputs:
    research/out/calls_hybrid_band_sweep.png
    research/out/calls_hybrid_band_sweep.txt
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.volatile_universe import universe, theme_of  # noqa: E402
from research.oversold_call_exit_sweep import (  # noqa: E402
    ExitCfg, simulate, _log_channel_z, _tier_for, _fmt_with_n,
    ACCEPT_TIERS, TIER_RANK,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, _stats, HEADER, STARTING_CAPITAL, MIN_ADV_M,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# R6 / LIVE-OTM-sigma-revert, pinned (the live default config).
PINNED_EXITS = ExitCfg(
    tp_pct=None,
    disaster_pct=None,
    disaster_usd=200.0,
    time_stop=45,
    sigma_target=0.0,
    iv_elevation=1.00,
    crush_td=30,
)

# Broad-market breadth still measured at z<=-2.0 crossings.
BREADTH_HI = -2.0


@dataclass
class Sig:
    tk: str
    i: int
    z: float
    date: object
    tier: str = "?"
    breadth: int = 0
    adv: float = 0.0
    source: str = "primary"   # "primary" | "deep" — for hybrid rank


def _signals_hybrid(series: dict[str, Series], z_by_tk: dict[str, np.ndarray],
                    *, primary_lo: float | None, primary_hi: float | None,
                    deep_threshold: float | None,
                    prefer_primary: bool = False) -> dict:
    """Generic band-union signal generator.

    primary band:  [primary_lo, primary_hi]  fires on first-touch (cross
                   into z <= primary_hi). Set both to None to disable.
    deep band:     z <= deep_threshold       fires on first-touch (cross
                   into z <= deep_threshold). Set to None to disable.

    Both signals can fire on the same bar for different tickers; they
    coexist in the same sig_by_day map and compete in the simulate
    engine's normal tier-ranked walk.

    prefer_primary: when True, primary-tagged sigs come ahead of deep-
                    tagged in the same-day ranking (within tier). When
                    False, use the engine's default rank.
    """
    breadth_by_day: Counter = Counter()
    eligible: list[Sig] = []

    for tk, s in series.items():
        z = z_by_tk[tk]
        for i in range(1, len(s.close)):
            if np.isnan(z[i]) or np.isnan(z[i - 1]):
                continue
            # Breadth — broad-market signal regardless of band.
            if z[i] <= BREADTH_HI and z[i - 1] > BREADTH_HI:
                breadth_by_day[s.dates[i]] += 1

            primary_hit = False
            deep_hit = False
            if primary_lo is not None and primary_hi is not None:
                if (z[i] <= primary_hi and z[i - 1] > primary_hi
                        and primary_lo <= z[i] <= primary_hi):
                    primary_hit = True
            if deep_threshold is not None:
                # Crossing into the deep zone — z[i-1] > deep_threshold
                # AND z[i] <= deep_threshold. A name that drops from
                # -2.5 to -3.7 crosses the -3.5 threshold this bar even
                # if it was already past the primary band, so deep
                # signals are CAPTURED INDEPENDENTLY of primary.
                if z[i] <= deep_threshold and z[i - 1] > deep_threshold:
                    deep_hit = True

            if not primary_hit and not deep_hit:
                continue
            if np.isnan(s.rv[i]) or not (s.adv_m[i] >= MIN_ADV_M):
                continue
            # If BOTH fire on the same bar (rare — would mean the bar
            # crossed both -2 and -3.5 in one go) we emit just the deep
            # one (it dominates).
            source = "deep" if deep_hit else "primary"
            eligible.append(Sig(tk=tk, i=i, z=float(z[i]), date=s.dates[i],
                                adv=float(s.adv_m[i]), source=source))

    out: dict = {}
    for sg in eligible:
        breadth = int(breadth_by_day.get(sg.date, 0))
        tier = _tier_for(breadth)
        if tier not in ACCEPT_TIERS:
            continue
        sg.tier = tier
        sg.breadth = breadth
        out.setdefault(sg.date, []).append(sg)

    if prefer_primary:
        # The simulate engine sorts by (TIER_RANK, -breadth, ticker).
        # We can't change its sort, but we can hand it sigs in our
        # preferred order — Python's sort is stable so within-tier
        # equal-breadth ties hold our order. So sort each day's sigs
        # with primary-first as the tie-break.
        for d in out:
            out[d].sort(key=lambda x: (0 if x.source == "primary" else 1,))
    return out


def _band_breakdown(sig_by_day: dict) -> tuple[int, int]:
    pri = sum(1 for sigs in sig_by_day.values() for s in sigs if s.source == "primary")
    dpe = sum(1 for sigs in sig_by_day.values() for s in sigs if s.source == "deep")
    return pri, dpe


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

    rows = [
        ("PRIMARY-ONLY [-2.5,-2.0]",
         dict(primary_lo=-2.5, primary_hi=-2.0, deep_threshold=None)),
        ("DEEP-ONLY z<=-3.5",
         dict(primary_lo=None, primary_hi=None, deep_threshold=-3.5)),
        ("DEEP-ONLY z<=-4.0",
         dict(primary_lo=None, primary_hi=None, deep_threshold=-4.0)),
        ("HYBRID 2.5+3.5",
         dict(primary_lo=-2.5, primary_hi=-2.0, deep_threshold=-3.5)),
        ("HYBRID 2.5+4.0 (skip marginal)",
         dict(primary_lo=-2.5, primary_hi=-2.0, deep_threshold=-4.0)),
        ("HYBRID 2.5+3.5 (prefer primary)",
         dict(primary_lo=-2.5, primary_hi=-2.0, deep_threshold=-3.5,
              prefer_primary=True)),
        ("WIDE [-inf,-2.0] (context)",
         dict(primary_lo=-1e9, primary_hi=-2.0, deep_threshold=None)),
    ]

    EHEADER = HEADER + "  | exit-reason mix"
    lines: list[str] = []
    lines.append("Call-leg HYBRID-band sweep on the volatile universe.")
    lines.append("Exit config PINNED to R6 / LIVE-OTM-sigma-revert (the live default).")
    lines.append(f"Tickers: {len(series)}.   Breadth always measured at z<=-2.0 crossings.")
    lines.append("")
    lines.append(EHEADER)

    results: list[tuple[str, dict, tuple[int, int]]] = []
    for label, kw in rows:
        sig_by_day = _signals_hybrid(series, z_by_tk, **kw)
        pri, dpe = _band_breakdown(sig_by_day)
        st = simulate(series, z_by_tk, sig_by_day, PINNED_EXITS)
        results.append((label, st, (pri, dpe)))
        lines.append(_fmt_with_n(label, st) + f"  | raw_sigs primary={pri}, deep={dpe}")
    lines.append("")

    # Compare hybrid vs. components.
    primary_only = next((st for l, st, _ in results if "PRIMARY-ONLY" in l), {})
    deep_only_3 = next((st for l, st, _ in results if "DEEP-ONLY z<=-3.5" in l), {})
    hybrid = next((st for l, st, _ in results if "HYBRID 2.5+3.5" in l and "prefer" not in l), {})

    lines.append("[VS. SINGLE-BAND BASELINES]")
    if primary_only and hybrid:
        d_cagr = (hybrid.get("cagr", 0) - primary_only.get("cagr", 0)) * 100
        d_pf = hybrid.get("profit_factor", 0) - primary_only.get("profit_factor", 0)
        d_n = hybrid.get("n", 0) - primary_only.get("n", 0)
        lines.append(f"  hybrid vs primary-only : CAGR {d_cagr:+.1f}pts, "
                     f"PF {d_pf:+.2f}, n {d_n:+d}")
    if deep_only_3 and hybrid:
        d_cagr = (hybrid.get("cagr", 0) - deep_only_3.get("cagr", 0)) * 100
        d_pf = hybrid.get("profit_factor", 0) - deep_only_3.get("profit_factor", 0)
        d_n = hybrid.get("n", 0) - deep_only_3.get("n", 0)
        lines.append(f"  hybrid vs deep-only-3.5 : CAGR {d_cagr:+.1f}pts, "
                     f"PF {d_pf:+.2f}, n {d_n:+d}")
    lines.append("")

    by_pf = max(results, key=lambda r: r[1].get("profit_factor", 0) or 0)
    by_cagr = max(results, key=lambda r: r[1].get("cagr", -1e9))
    by_sharpe = max(results, key=lambda r: r[1].get("sharpe", -1e9))
    lines.append("[HEADLINE]")
    lines.append(f"  best by CAGR   : {by_cagr[0]}   CAGR {by_cagr[1]['cagr']*100:+.1f}%  "
                 f"PF {by_cagr[1].get('profit_factor', 0):.2f}  Sharpe {by_cagr[1].get('sharpe', 0):.2f}  n={by_cagr[1].get('n', 0)}")
    lines.append(f"  best by PF     : {by_pf[0]}   CAGR {by_pf[1]['cagr']*100:+.1f}%  "
                 f"PF {by_pf[1].get('profit_factor', 0):.2f}  Sharpe {by_pf[1].get('sharpe', 0):.2f}  n={by_pf[1].get('n', 0)}")
    lines.append(f"  best by Sharpe : {by_sharpe[0]}   CAGR {by_sharpe[1]['cagr']*100:+.1f}%  "
                 f"PF {by_sharpe[1].get('profit_factor', 0):.2f}  Sharpe {by_sharpe[1].get('sharpe', 0):.2f}  n={by_sharpe[1].get('n', 0)}")

    fig, ax = plt.subplots(figsize=(14, 7))
    for label, st, _ in results:
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.5,
                label=f"{label}  (CAGR {st['cagr']*100:+.1f}%, "
                      f"PF {st.get('profit_factor', 0):.2f}, n={st['n']})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title("Call-leg hybrid-band sweep, R6 exits, volatile universe")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "calls_hybrid_band_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "calls_hybrid_band_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
