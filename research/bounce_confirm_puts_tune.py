"""Multi-axis tuning around the bounce-puts winner (reclaim=0.10).

bounce_confirm_puts_sweep found one positive cell:
  bounce-puts reclaim>=0.10: PF 1.83, CAGR +2.2%, Sharpe 0.22, n=45

This script holds the rest of the config at that anchor and sweeps one
axis at a time to find whether a tighter combo lifts CAGR/PF without
overfitting:

  F2m reclaim sigma     {0.05, 0.08, 0.10, 0.12, 0.15, 0.20}
  F4m turn threshold    {0.5, 1.0, 3.0, 5.0, 10.0}
  F4m overbought floor  {65, 70, 75, 80, 85}
  F5m support hits min  {3, 5, 8, 12}
  cooldown bars         {10, 20, 30, 40}
  DTE                   {30, 45, 60, 90}
  OTM%                  {0, 2.5, 5, 7.5, 10}
  disaster $ cap        {None, 100, 200, 300, 500}
  time stop (td)        {20, 30, 45, 60, None}

Then stacks best-per-axis into a TUNED combined config and reports it
alongside the anchor. Writes report to
research/out/bounce_confirm_puts_tune.txt.
"""
from __future__ import annotations

import copy
import os
import sys
from dataclasses import replace

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
    Sig, simulate_put, SWEET_LO_PUT, SWEET_HI_PUT,
)
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, HEADER, STARTING_CAPITAL, MIN_ADV_M,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Anchor (the winning cell from bounce_confirm_puts_sweep)
ANCHOR = {
    "reclaim": 0.10,
    "turn": 1.0,
    "ob_floor": 70.0,
    "hits": 5,
    "cooldown": 20,
    "support_pct": 0.02,
    "dte": 45,
    "otm": 5.0,
    "disaster_usd": 200.0,
    "time_stop": 45,
}


def make_signals(series, z_by_tk, p):
    """Return date -> [Sig] using parameters dict p (full feature config)."""
    out: dict[object, list[Sig]] = {}
    LB = 20
    PW = LOG_WINDOW
    for tk, s in series.items():
        z = z_by_tk[tk]
        k_ser, _ = stoch_rsi(pd.Series(s.close))
        k_arr = k_ser.to_numpy(dtype=float)
        last_fire = -10_000
        start = PW + LB + 2
        for i in range(start, len(s.close)):
            if i - last_fire < p["cooldown"]:
                continue
            zi = z[i]
            if np.isnan(zi):
                continue
            if not (SWEET_LO_PUT <= zi <= SWEET_HI_PUT):
                continue
            zmax = float(np.nanmax(z[i - LB + 1: i + 1]))
            if zi > zmax - p["reclaim"]:
                continue
            recent_hi = float(np.nanmax(s.high[i - LB + 1: i + 1]))
            prior_hi = float(np.nanmax(s.high[i - PW + 1: i - LB + 1]))
            if not np.isfinite(prior_hi) or recent_hi >= prior_hi:
                continue
            ki = k_arr[i]
            if not np.isfinite(ki) or ki < p["ob_floor"]:
                continue
            k_max = float(np.nanmax(k_arr[i - LB + 1: i + 1]))
            if ki >= k_max - p["turn"]:
                continue
            band_lo = recent_hi * (1.0 - p["support_pct"])
            band_hi = recent_hi * (1.0 + p["support_pct"])
            prior_highs = s.high[i - PW + 1: i - LB + 1]
            hits = int(((prior_highs >= band_lo) & (prior_highs <= band_hi)).sum())
            if hits < p["hits"]:
                continue
            adv = s.adv_m[i]
            if not (np.isfinite(adv) and adv >= MIN_ADV_M):
                continue
            if not np.isfinite(s.rv[i]):
                continue
            sg = Sig(tk=tk, i=i, z=float(zi), date=s.dates[i],
                     tier="BOUNCE", breadth=hits, adv=float(adv))
            out.setdefault(s.dates[i], []).append(sg)
            last_fire = i
    return out


def make_cfg(p):
    return ExitCfg(
        tp_pct=None,
        disaster_pct=None,
        disaster_usd=p["disaster_usd"],
        time_stop=p["time_stop"],
        sigma_target=0.0,
        iv_elevation=1.00, crush_td=30,
        otm_pct=p["otm"], dte=p["dte"],
    )


def run_with(p, series, z_by_tk):
    sigs = make_signals(series, z_by_tk, p)
    n = sum(len(v) for v in sigs.values())
    st = simulate_put(series, z_by_tk, sigs, make_cfg(p))
    st["n_sig"] = n
    return st


def fmt(tag, st):
    return _fmt_with_n(f"{tag} (n_sig={st.get('n_sig', '?')})", st)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tks = universe()
    print(f"Fetching {len(tks)} volatile-universe tickers x 5y...")
    bars = fetch_bars_bulk(tks, period="5y")
    series, z_by_tk = {}, {}
    for tk, df in bars.items():
        s = build_series(tk, df)
        if s is not None:
            series[tk] = s
            z_by_tk[tk] = _log_channel_z(s.close)
    print(f"  {len(series)} tickers with full 252d history\n")

    EHEADER = HEADER + "  | exit-reason mix"
    lines = []
    lines.append("Multi-axis tuning around bounce-puts anchor.")
    lines.append("Anchor: " + ", ".join(f"{k}={v}" for k, v in ANCHOR.items()))
    lines.append("Sweep one axis at a time, hold the rest at anchor.")
    lines.append("")

    anchor_st = run_with(ANCHOR, series, z_by_tk)
    lines.append("[ANCHOR]")
    lines.append(EHEADER)
    lines.append(fmt("anchor", anchor_st))
    lines.append("")

    def sweep(axis_name, axis_values, key):
        rows = []
        lines.append(f"[SWEEP {axis_name}]  (other axes = anchor)")
        lines.append(EHEADER)
        for v in axis_values:
            p = dict(ANCHOR)
            p[key] = v
            st = run_with(p, series, z_by_tk)
            tag = f"{axis_name}={v}"
            rows.append((v, st))
            lines.append(fmt(tag, st))
        lines.append("")
        return rows

    rec_rows  = sweep("reclaim",   [0.05, 0.08, 0.10, 0.12, 0.15, 0.20], "reclaim")
    turn_rows = sweep("turn",      [0.5, 1.0, 3.0, 5.0, 10.0],            "turn")
    ob_rows   = sweep("ob_floor",  [65, 70, 75, 80, 85],                   "ob_floor")
    hits_rows = sweep("hits",      [3, 5, 8, 12],                          "hits")
    cd_rows   = sweep("cooldown",  [10, 20, 30, 40],                       "cooldown")
    dte_rows  = sweep("dte",       [30, 45, 60, 90],                       "dte")
    otm_rows  = sweep("otm",       [0.0, 2.5, 5.0, 7.5, 10.0],             "otm")
    dis_rows  = sweep("dis_usd",   [None, 100, 200, 300, 500],             "disaster_usd")
    ts_rows   = sweep("time_stop", [20, 30, 45, 60, None],                 "time_stop")

    def best(rows, metric):
        getter = (lambda st: st.get(metric, -1e9) or -1e9) if metric == "cagr" \
                 else (lambda st: st.get("profit_factor", 0) or 0)
        return max(rows, key=lambda kv: getter(kv[1]))

    # Stack best-per-axis (by CAGR) into a tuned combined config.
    tuned = dict(ANCHOR)
    tuned["reclaim"]     = best(rec_rows,  "cagr")[0]
    tuned["turn"]        = best(turn_rows, "cagr")[0]
    tuned["ob_floor"]    = best(ob_rows,   "cagr")[0]
    tuned["hits"]        = best(hits_rows, "cagr")[0]
    tuned["cooldown"]    = best(cd_rows,   "cagr")[0]
    tuned["dte"]         = best(dte_rows,  "cagr")[0]
    tuned["otm"]         = best(otm_rows,  "cagr")[0]
    tuned["disaster_usd"] = best(dis_rows, "cagr")[0]
    tuned["time_stop"]   = best(ts_rows,   "cagr")[0]
    tuned_st = run_with(tuned, series, z_by_tk)

    # Also a PF-stacked version (more conservative).
    tuned_pf = dict(ANCHOR)
    tuned_pf["reclaim"]     = best(rec_rows,  "pf")[0]
    tuned_pf["turn"]        = best(turn_rows, "pf")[0]
    tuned_pf["ob_floor"]    = best(ob_rows,   "pf")[0]
    tuned_pf["hits"]        = best(hits_rows, "pf")[0]
    tuned_pf["cooldown"]    = best(cd_rows,   "pf")[0]
    tuned_pf["dte"]         = best(dte_rows,  "pf")[0]
    tuned_pf["otm"]         = best(otm_rows,  "pf")[0]
    tuned_pf["disaster_usd"] = best(dis_rows, "pf")[0]
    tuned_pf["time_stop"]   = best(ts_rows,   "pf")[0]
    tuned_pf_st = run_with(tuned_pf, series, z_by_tk)

    lines.append("[BEST PER AXIS (by CAGR)]")
    for n, rows in [("reclaim", rec_rows), ("turn", turn_rows), ("ob_floor", ob_rows),
                    ("hits", hits_rows), ("cooldown", cd_rows), ("dte", dte_rows),
                    ("otm", otm_rows), ("dis_usd", dis_rows), ("time_stop", ts_rows)]:
        v, st = best(rows, "cagr")
        lines.append(f"  {n:10s} = {v!s:6s} -> CAGR {st.get('cagr', 0)*100:+.1f}%  "
                     f"PF {st.get('profit_factor', 0):.2f}  n={st.get('n', 0)}")
    lines.append("")

    lines.append("[BEST PER AXIS (by PF)]")
    for n, rows in [("reclaim", rec_rows), ("turn", turn_rows), ("ob_floor", ob_rows),
                    ("hits", hits_rows), ("cooldown", cd_rows), ("dte", dte_rows),
                    ("otm", otm_rows), ("dis_usd", dis_rows), ("time_stop", ts_rows)]:
        v, st = best(rows, "pf")
        lines.append(f"  {n:10s} = {v!s:6s} -> PF {st.get('profit_factor', 0):.2f}  "
                     f"CAGR {st.get('cagr', 0)*100:+.1f}%  n={st.get('n', 0)}")
    lines.append("")

    lines.append("[STACKED TUNED CONFIGS]")
    lines.append(EHEADER)
    lines.append(fmt("anchor", anchor_st))
    lines.append(fmt("tuned-CAGR", tuned_st))
    lines.append(fmt("tuned-PF", tuned_pf_st))
    lines.append("")
    lines.append(f"tuned-CAGR  config: {tuned}")
    lines.append(f"tuned-PF    config: {tuned_pf}")

    out_txt = os.path.join(OUT_DIR, "bounce_confirm_puts_tune.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
