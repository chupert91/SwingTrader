"""Breather-scalp put backtest on the 5y volatile universe.

Final shot at the put leg. From research/out/personal_puts_post_entry_paths.txt:
  - Winners' median entry-1: -2.4%   (pullback ALREADY in progress)
  - Losers'  median entry-1: -0.3%   (flat — no pullback)
  - After exit, 11/14 winners' underlying bounces +7% in 5d
  -> user is scalping the brief pause INSIDE an ongoing uptrend, NOT
     shorting the top of the move.

So the signal needs TWO things together:
  1. Parabolic context (stock has been ripping)
  2. Pullback already in progress (today's close is BELOW recent)

And the exit needs to be opportunistic:
  - Exit on first GREEN underlying bar (close > prev close) after entry
    — that's the user catching the bounce attempt
  - OR TP +20%, OR 3-5d time stop, OR $200 absolute cap

Three entry variants:
  E1: parabolic + today is a red day  (close < prev_close by >= -1.5%)
  E2: parabolic + close is >= -2% below 3-bar high
  E3: parabolic + EITHER of E1/E2 (loosest)

Three exit configs:
  X1: green-bar OR TP+20% OR 3d time stop OR $200 cap (FAST)
  X2: green-bar OR TP+20% OR 5d time stop OR $200 cap (MEDIUM)
  X3:              TP+20% OR 5d time stop OR $200 cap (NO green-bar — baseline)

Ship gate: PF >= 1.5 AND CAGR > 0 on any of the 9 cells.

Run:
    python research/puts_breather_scalp_sweep.py
Outputs:
    research/out/puts_breather_scalp_sweep.png
    research/out/puts_breather_scalp_sweep.txt
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import timedelta

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
    PREMIUM_CAP, MAX_CONCURRENT, MAX_PER_DAY,
)
from research.volatile_universe_puts_sweep import (  # noqa: E402
    _bs_put, _price_put, Sig as _PutSig,
)
from research.oversold_call_exit_sweep import _iv_at  # noqa: E402
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, _stats, HEADER, STARTING_CAPITAL, MIN_ADV_M,
    OPTION_MULT,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

COOLDOWN_BARS = 5
LOG_WINDOW = 252
SMA20 = 20


# ---------- shared per-bar feature computation --------------------------

def _features_per_bar(closes: np.ndarray) -> dict:
    n = len(closes)
    out = {}

    def _ret(k: int) -> np.ndarray:
        r = np.full(n, np.nan)
        if n > k:
            r[k:] = (closes[k:] / closes[:n - k] - 1.0) * 100.0
        return r

    out["ret_10d"] = _ret(10)
    sma20 = np.full(n, np.nan)
    if n >= SMA20:
        cs = np.cumsum(closes, dtype=float)
        cs = np.concatenate(([0.0], cs))
        sma20[SMA20 - 1:] = (cs[SMA20:] - cs[:n - SMA20 + 1]) / SMA20
    out["dist_sma20"] = np.where(sma20 > 0, (closes / sma20 - 1.0) * 100.0, np.nan)

    # daily %change (close[i] vs close[i-1])
    dpct = np.full(n, np.nan)
    dpct[1:] = (closes[1:] / closes[:-1] - 1.0) * 100.0
    out["d_pct"] = dpct

    # rolling 3-bar high (look-back, ending at i-1 — exclude today so we
    # can compare today's close against the prior 3 days).
    high3 = np.full(n, np.nan)
    for i in range(3, n):
        high3[i] = max(closes[i - 3:i])
    out["high3"] = high3
    out["dist_high3"] = np.where(high3 > 0, (closes / high3 - 1.0) * 100.0, np.nan)
    return out


def _is_red_bar(d_pct: float) -> bool:
    return not (math.isnan(d_pct)) and d_pct <= -1.5


def _is_off_3d_high(dist_high3: float) -> bool:
    return not (math.isnan(dist_high3)) and dist_high3 <= -2.0


@dataclass
class EntryDef:
    name: str
    fn: object   # callable(fb: dict, i: int) -> bool


def _entry_E1(fb: dict, i: int) -> bool:
    if np.isnan(fb["ret_10d"][i]) or np.isnan(fb["dist_sma20"][i]):
        return False
    if not (fb["ret_10d"][i] >= 15.0 and fb["dist_sma20"][i] >= 10.0):
        return False
    return _is_red_bar(fb["d_pct"][i])


def _entry_E2(fb: dict, i: int) -> bool:
    if np.isnan(fb["ret_10d"][i]) or np.isnan(fb["dist_sma20"][i]):
        return False
    if not (fb["ret_10d"][i] >= 15.0 and fb["dist_sma20"][i] >= 10.0):
        return False
    return _is_off_3d_high(fb["dist_high3"][i])


def _entry_E3(fb: dict, i: int) -> bool:
    if np.isnan(fb["ret_10d"][i]) or np.isnan(fb["dist_sma20"][i]):
        return False
    if not (fb["ret_10d"][i] >= 15.0 and fb["dist_sma20"][i] >= 10.0):
        return False
    return _is_red_bar(fb["d_pct"][i]) or _is_off_3d_high(fb["dist_high3"][i])


ENTRIES = [
    EntryDef("E1 parabolic + red bar (>=-1.5%)", _entry_E1),
    EntryDef("E2 parabolic + off 3d-high (>=-2%)", _entry_E2),
    EntryDef("E3 parabolic + (E1 OR E2)", _entry_E3),
]


# ---------- breather-scalp simulator ------------------------------------

@dataclass
class _Pos:
    tk: str
    entry_idx: int
    entry_date: object
    exp_date: object
    strike: float
    entry_cost: float
    entry_iv: float
    iv_floor: float
    qty: int


@dataclass(frozen=True)
class ScalpExitCfg:
    tp_pct: float | None = 20.0
    time_stop: int | None = 5
    disaster_usd: float | None = 200.0
    green_bar_exit: bool = True   # exit on first green underlying bar
    iv_elevation: float = 1.00
    crush_td: int = 30
    half_spread: float = 0.015
    entry_buffer: float = 0.0
    otm_pct: float = 5.0
    dte: int = 45


def _signals_breather(series: dict[str, Series], entry: EntryDef) -> dict:
    """date -> list[_PutSig]. Fire on bars passing entry.fn; cooldown to
    avoid back-to-back entries on the same name."""
    breadth_by_day: Counter = Counter()
    raw = []
    feat_cache: dict[str, dict] = {}
    for tk, s in series.items():
        fb = _features_per_bar(s.close)
        feat_cache[tk] = fb
        last_fire = -10**9
        for i in range(1, len(s.close)):
            if not entry.fn(fb, i):
                continue
            if i - last_fire < COOLDOWN_BARS:
                continue
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


def simulate_scalp(series: dict[str, Series], sig_by_day: dict,
                   cfg: ScalpExitCfg) -> dict:
    """Put-leg sim with green-bar exit option."""
    all_dates = sorted({d for s in series.values() for d in s.dates})
    cash = STARTING_CAPITAL
    open_pos: dict[str, _Pos] = {}
    trades: list[dict] = []
    equity_curve: list[tuple] = []

    def mtm_close(pos: _Pos, s: Series, idx: int) -> float:
        iv = _iv_at(pos.entry_iv, pos.iv_floor, idx - pos.entry_idx, cfg.crush_td)
        mid = _price_put(s.close[idx], pos.strike, pos.exp_date, s.dates[idx], iv)
        return mid * (1 - cfg.half_spread) * OPTION_MULT * pos.qty

    for d in all_dates:
        # ---- exits ----
        for tk in sorted(list(open_pos.keys())):
            pos = open_pos[tk]
            s = series[tk]
            idx = s.by_date.get(d)
            if idx is None or idx <= pos.entry_idx:
                continue
            held = idx - pos.entry_idx
            iv = _iv_at(pos.entry_iv, pos.iv_floor, held, cfg.crush_td)
            total_cost_usd = pos.entry_cost * OPTION_MULT * pos.qty

            def realized_ret(S: float) -> float:
                proc = (_price_put(S, pos.strike, pos.exp_date, s.dates[idx], iv)
                        * (1 - cfg.half_spread) * OPTION_MULT * pos.qty)
                return proc / (pos.entry_cost * OPTION_MULT * pos.qty) - 1.0

            exit_ret, reason = None, None
            expired = s.dates[idx] >= pos.exp_date

            if expired:
                exit_ret, reason = realized_ret(s.close[idx]), "expiry"
            else:
                # Intraday TP touch on the bar LOW (put best mark).
                if cfg.tp_pct is not None and realized_ret(s.low[idx]) >= cfg.tp_pct / 100.0:
                    exit_ret, reason = cfg.tp_pct / 100.0, "take_profit"
                else:
                    cr = realized_ret(s.close[idx])
                    # Disaster $ cap.
                    if cfg.disaster_usd is not None and -cr * total_cost_usd >= cfg.disaster_usd:
                        exit_ret, reason = cr, "disaster"
                    # Green-bar exit: today's close > prior close (any green
                    # bar after entry triggers a close). Skip the entry bar
                    # itself since held > 0 guarantees we're past it.
                    elif (cfg.green_bar_exit and idx >= 1
                          and s.close[idx] > s.close[idx - 1]):
                        exit_ret, reason = cr, "green_bar"
                    elif cfg.time_stop is not None and held >= cfg.time_stop:
                        exit_ret, reason = cr, "time_stop"

            if exit_ret is not None:
                exit_ret = max(exit_ret, -1.0)
                proceeds = pos.entry_cost * OPTION_MULT * pos.qty * (1.0 + exit_ret)
                cash += proceeds
                trades.append({
                    "ticker": tk,
                    "entry_date": str(pos.entry_date),
                    "exit_date": str(s.dates[idx]),
                    "bars_held": held,
                    "opt_ret": exit_ret,
                    "exit_reason": reason,
                    "cost": pos.entry_cost * OPTION_MULT * pos.qty,
                })
                del open_pos[tk]

        # ---- entries ----
        sigs = sig_by_day.get(d, [])
        if sigs and len(open_pos) < MAX_CONCURRENT:
            placed_today = 0
            for sg in sorted(sigs, key=lambda x: (TIER_RANK.get(x.tier, 9),
                                                   -x.breadth, x.tk)):
                if len(open_pos) >= MAX_CONCURRENT or placed_today >= MAX_PER_DAY:
                    break
                tk, i = sg.tk, sg.i
                if tk in open_pos:
                    continue
                s = series[tk]
                S0 = s.close[i]
                K = S0 * (1.0 - cfg.otm_pct / 100.0)
                rv = s.rv[i]
                if np.isnan(rv):
                    continue
                entry_iv = min(max(rv * cfg.iv_elevation, 0.15), 1.20)
                iv_floor = min(max(rv, 0.10), 1.10)
                exp_date = s.dates[i] + timedelta(days=cfg.dte)
                mid0 = _price_put(S0, K, exp_date, s.dates[i], entry_iv)
                if mid0 <= 1e-6:
                    continue
                entry_cost = mid0 * (1.0 + cfg.half_spread) * (1.0 + cfg.entry_buffer)
                cost_usd = entry_cost * OPTION_MULT
                if cost_usd > PREMIUM_CAP or cost_usd > cash:
                    continue
                cash -= cost_usd
                open_pos[tk] = _Pos(tk, i, s.dates[i], exp_date, K, entry_cost,
                                    entry_iv, iv_floor, 1)
                placed_today += 1

        # ---- equity snapshot ----
        held_val = 0.0
        for tk, pos in open_pos.items():
            s = series[tk]
            idx = s.by_date.get(d)
            if idx is not None and idx >= pos.entry_idx:
                held_val += mtm_close(pos, s, idx)
        equity_curve.append((d, cash + held_val))

    return _stats(trades, equity_curve)


EXIT_CFGS = [
    ("X1 green-bar | TP20 | t3 | $200",
     ScalpExitCfg(tp_pct=20.0, time_stop=3, disaster_usd=200.0,
                  green_bar_exit=True)),
    ("X2 green-bar | TP20 | t5 | $200",
     ScalpExitCfg(tp_pct=20.0, time_stop=5, disaster_usd=200.0,
                  green_bar_exit=True)),
    ("X3 NO-green | TP20 | t5 | $200",
     ScalpExitCfg(tp_pct=20.0, time_stop=5, disaster_usd=200.0,
                  green_bar_exit=False)),
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tks = universe()
    print(f"Fetching {len(tks)} volatile-universe tickers x 5y...")
    bars = fetch_bars_bulk(tks, period="5y")
    series: dict[str, Series] = {}
    for tk, df in bars.items():
        s = build_series(tk, df)
        if s is not None:
            series[tk] = s
    print(f"  {len(series)} tickers with full 252d history\n")

    EHEADER = HEADER + "  | exit-reason mix"
    lines: list[str] = []
    lines.append("PUTS BREATHER-SCALP SWEEP on volatile universe (5y).")
    lines.append("Entry: parabolic context AND pullback already started.")
    lines.append("Exit:  green-bar (resumption) OR TP20 OR time stop OR $200 cap.")
    lines.append(f"Tickers: {len(series)}.   Cooldown: {COOLDOWN_BARS} bars/ticker.")
    lines.append("Ship gate: PF >= 1.5 AND CAGR > 0 AND n >= 10.")
    lines.append("")

    all_results: list[tuple[str, str, dict, int]] = []
    for entry in ENTRIES:
        sig_by_day = _signals_breather(series, entry)
        n_sig = sum(len(v) for v in sig_by_day.values())
        by_theme: dict[str, int] = defaultdict(int)
        for sigs in sig_by_day.values():
            for sg in sigs:
                by_theme[theme_of(sg.tk)] += 1
        theme_str = ", ".join(f"{k}={v}" for k, v in sorted(by_theme.items(), key=lambda kv: -kv[1])[:5])
        lines.append(f"[ENTRY {entry.name}]   raw_signals={n_sig}   themes(top5): {theme_str}")
        lines.append(EHEADER)
        for ex_label, cfg in EXIT_CFGS:
            st = simulate_scalp(series, sig_by_day, cfg)
            all_results.append((entry.name, ex_label, st, n_sig))
            lines.append(_fmt_with_n(ex_label, st))
        lines.append("")

    passers = [(e, x, st) for e, x, st, _ in all_results
               if (st.get("profit_factor", 0) or 0) >= 1.5
               and (st.get("cagr", 0) or 0) > 0
               and st.get("n", 0) >= 10]
    by_pf = max(all_results, key=lambda r: r[2].get("profit_factor", 0) or 0)
    by_cagr = max(all_results, key=lambda r: r[2].get("cagr", -1e9))
    by_sharpe = max(all_results, key=lambda r: r[2].get("sharpe", -1e9))

    lines.append("[SHIP-GATE PASSERS]  (PF >= 1.5 AND CAGR > 0 AND n >= 10)")
    if not passers:
        lines.append("  (none — breather-scalp hypothesis falsified)")
    else:
        for e, x, st in sorted(passers, key=lambda r: -(r[2].get("profit_factor", 0) or 0)):
            lines.append(f"  {e}  +  {x}")
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

    # Plot best-per-entry equity curves.
    fig, ax = plt.subplots(figsize=(14, 7))
    best_per_entry: dict = {}
    for e, x, st, _ in all_results:
        cur = best_per_entry.get(e)
        if cur is None or (st.get("profit_factor", 0) or 0) > (cur[1].get("profit_factor", 0) or 0):
            best_per_entry[e] = (x, st)
    for e, (x, st) in best_per_entry.items():
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.5,
                label=f"{e} | best: {x} "
                      f"(PF {st.get('profit_factor', 0):.2f}, "
                      f"CAGR {st['cagr']*100:+.1f}%, n={st['n']})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title("Puts breather-scalp sweep — best exit per entry on volatile universe")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "puts_breather_scalp_sweep.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "puts_breather_scalp_sweep.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
