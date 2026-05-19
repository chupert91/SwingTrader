"""Capital-constrained, theta-aware OPTION sim of the SMA-reversion strategy.

This answers the bot go/no-go the underlying study could not: does the
4-down-bars-below-SMA20 edge survive as the OTM-call instrument the live
bot trades, run as a *separate competing sleeve* under the same capital /
concurrency / risk frame as the existing oversold-call bot?

Mirrors backend/ai_store.DEFAULT_SETTINGS (the live bot):
  OTM 5%, DTE 45 (earliest in the 45-60 band, least theta over-pay --
  matches ai_strategy.pick_contract), premium cap $2000, max 2 concurrent,
  max 2 entries/day, 1 contract per trade, take-profit +30% (option),
  disaster stop -50% (option), expiry settles at intrinsic.

Option pricing reuses research/theta_overlay.py's validated model:
  Black-Scholes; entry IV = trailing 20d realized vol * 1.25 clamped
  [0.15, 1.20]; IV crushes linearly to the un-elevated RV over the first
  10 trading days then flat; r = 4%. Entry pays mid*(1+half_spread)*(1+
  limit_buffer); exit receives mid*(1-half_spread). Calendar/expiry use
  the bars' real dates.

Engine: event-driven over the union trading calendar. Each date: MTM +
exits on open positions first, then entries. Contested same-day signals
fill by FIRST-COME = most-stretched-below-SMA wins (deterministic, ticker
tie-break), bounded by free slots, the per-day cap, the premium cap, and
cash. Liquidity-gated (>= $50M 20d avg dollar vol) like the live scanner.

The strategy exit is the SMA reclaim (close back above SMA20). The
loss-only time stop is SWEPT {none, 8, 11, 15, 20} trading days: once held
>= N bars, exit any bar with the OPTION still in a loss (winners ride to
the SMA reclaim / TP). A one-axis sensitivity (DTE, OTM, spread) runs at
the no-stop exit to show how much the verdict hinges on option assumptions.

Run:
    python research/sma_reversion_option_sim.py
Outputs:
    research/out/sma_reversion_option_sim.png
    research/out/sma_reversion_option_sim.txt
"""
from __future__ import annotations

import math
import os
import sys
from datetime import timedelta
from dataclasses import dataclass, replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.sp500_tickers import SP500_TICKERS  # noqa: E402
from research.theta_overlay import (  # noqa: E402
    bs_call, IV_ELEVATION, CRUSH_TD, R, RV_LOOKBACK,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Strategy
DOWN_BARS = 4
SMA_PERIOD = 20
# Instrument / sizing (mirror backend/ai_store.DEFAULT_SETTINGS)
OTM_PCT = 5.0
DTE = 45
PREMIUM_CAP = 2000.0
MAX_CONCURRENT = 2
MAX_PER_DAY = 2
TP_PCT = 30.0
DISASTER_PCT = 50.0
STARTING_CAPITAL = 10000.0
MIN_ADV_M = 50.0          # 20d avg dollar volume gate ($M), like ai_strategy
# Frictions
HALF_SPREAD = 0.03        # option bid/ask half-spread
ENTRY_BUFFER = 0.01       # marketable-limit buffer (ai_store entry_limit_buffer_pct)
# Exit sweep: trading-day loss-only time stop. None = SMA/TP/disaster/expiry only.
STOP_SWEEP = [None, 8, 11, 15, 20]
OPTION_MULT = 100


@dataclass(frozen=True)
class Cfg:
    otm_pct: float = OTM_PCT
    dte: int = DTE
    half_spread: float = HALF_SPREAD
    time_loss_exit: int | None = None  # set per run


def down_run_lengths(closes: np.ndarray) -> np.ndarray:
    n = len(closes)
    run = np.zeros(n, dtype=int)
    for i in range(1, n):
        run[i] = run[i - 1] + 1 if closes[i] < closes[i - 1] else 0
    return run


@dataclass
class Series:
    ticker: str
    dates: np.ndarray            # datetime.date
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    sma: np.ndarray
    drun: np.ndarray
    rv: np.ndarray               # annualized 20d realized vol per bar
    adv_m: np.ndarray            # trailing 20d avg dollar volume ($M)
    by_date: dict                # date -> row index


def build_series(ticker: str, df: pd.DataFrame) -> Series | None:
    if df is None or df.empty or len(df) < SMA_PERIOD + RV_LOOKBACK + 5:
        return None
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    vol = (df["volume"].to_numpy(dtype=float)
           if "volume" in df.columns else np.zeros(len(close)))
    dts = pd.to_datetime(df["timestamp"]).dt.date.to_numpy()
    sma = pd.Series(close).rolling(SMA_PERIOD).mean().to_numpy()
    drun = down_run_lengths(close)
    logret = np.concatenate([[np.nan], np.diff(np.log(close))])
    rv = np.full(len(close), np.nan)
    for i in range(RV_LOOKBACK + 1, len(close)):
        rv[i] = float(np.std(logret[i - RV_LOOKBACK + 1: i + 1], ddof=1) * math.sqrt(252))
    dollar = close * vol
    adv_m = pd.Series(dollar).rolling(20).mean().to_numpy() / 1_000_000.0
    by_date = {d: i for i, d in enumerate(dts)}
    return Series(ticker, dts, close, high, low, sma, drun, rv, adv_m, by_date)


def _iv_at(entry_iv: float, iv_floor: float, held: int) -> float:
    crush = min(held / CRUSH_TD, 1.0) if CRUSH_TD > 0 else 1.0
    return entry_iv - (entry_iv - iv_floor) * crush


def _price(S: float, K: float, exp_date, bar_date, iv: float) -> float:
    T = max((exp_date - bar_date).days, 0) / 365.0
    if T <= 0.0:
        return max(S - K, 0.0)
    return bs_call(S, K, T, R, iv)


@dataclass
class Pos:
    ticker: str
    entry_idx: int
    entry_date: object
    exp_date: object
    strike: float
    entry_cost: float            # per-contract premium paid (incl. spread+buffer)
    entry_iv: float
    iv_floor: float
    qty: int


def _signals(series: dict[str, Series]) -> dict[object, list[tuple]]:
    """date -> list of (stretch, ticker, idx) entry signals (pre-filtered)."""
    by_day: dict[object, list[tuple]] = {}
    for tk, s in series.items():
        for i in range(len(s.close)):
            if np.isnan(s.sma[i]) or np.isnan(s.rv[i]):
                continue
            if s.drun[i] != DOWN_BARS:
                continue
            if s.close[i] >= s.sma[i]:
                continue
            if not (s.adv_m[i] >= MIN_ADV_M):
                continue
            stretch = (s.sma[i] - s.close[i]) / s.sma[i]  # most-stretched first
            by_day.setdefault(s.dates[i], []).append((stretch, tk, i))
    return by_day


def simulate(series: dict[str, Series], sig_by_day: dict[object, list[tuple]],
             cfg: Cfg) -> dict:
    all_dates = sorted({d for s in series.values() for d in s.dates})
    cash = STARTING_CAPITAL
    open_pos: dict[str, Pos] = {}
    trades: list[dict] = []
    equity_curve: list[tuple] = []

    def mtm(pos: Pos, s: Series, idx: int) -> float:
        iv = _iv_at(pos.entry_iv, pos.iv_floor, idx - pos.entry_idx)
        mid = _price(s.close[idx], pos.strike, pos.exp_date, s.dates[idx], iv)
        return mid * (1 - cfg.half_spread) * OPTION_MULT * pos.qty

    for d in all_dates:
        # ---- exits / MTM (deterministic ticker order) ----
        for tk in sorted(list(open_pos.keys())):
            pos = open_pos[tk]
            s = series[tk]
            idx = s.by_date.get(d)
            if idx is None or idx <= pos.entry_idx:
                continue
            held = idx - pos.entry_idx
            iv = _iv_at(pos.entry_iv, pos.iv_floor, held)

            def ret_at(S: float) -> float:
                proceeds = (_price(S, pos.strike, pos.exp_date, s.dates[idx], iv)
                            * (1 - cfg.half_spread) * OPTION_MULT * pos.qty)
                return proceeds / (pos.entry_cost * OPTION_MULT * pos.qty) - 1.0

            exit_ret = None
            reason = None
            # Conservative intraday order: disaster (low) -> TP (high).
            if ret_at(s.low[idx]) <= -DISASTER_PCT / 100.0:
                exit_ret, reason = -DISASTER_PCT / 100.0, "disaster"
            elif ret_at(s.high[idx]) >= TP_PCT / 100.0:
                exit_ret, reason = TP_PCT / 100.0, "take_profit"
            else:
                close_ret = ret_at(s.close[idx])
                expired = s.dates[idx] >= pos.exp_date
                crossed_up = s.close[idx] > s.sma[idx]
                time_stopped = (
                    cfg.time_loss_exit is not None
                    and held >= cfg.time_loss_exit
                    and close_ret < 0.0
                    and not crossed_up
                )
                if expired:
                    exit_ret, reason = close_ret, "expiry"
                elif crossed_up:
                    exit_ret, reason = close_ret, "sma_cross"
                elif time_stopped:
                    exit_ret, reason = close_ret, "time_loss"

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
            for stretch, tk, i in sorted(sigs, key=lambda x: (-x[0], x[1])):
                if len(open_pos) >= MAX_CONCURRENT or placed_today >= MAX_PER_DAY:
                    break
                if tk in open_pos:
                    continue
                s = series[tk]
                S0 = s.close[i]
                K = S0 * (1.0 + cfg.otm_pct / 100.0)
                rv = s.rv[i]
                if np.isnan(rv):
                    continue
                entry_iv = min(max(rv * IV_ELEVATION, 0.15), 1.20)
                iv_floor = min(max(rv, 0.10), 1.10)
                exp_date = s.dates[i] + timedelta(days=cfg.dte)
                mid0 = _price(S0, K, exp_date, s.dates[i], entry_iv)
                if mid0 <= 1e-6:
                    continue
                entry_cost = mid0 * (1.0 + cfg.half_spread) * (1.0 + ENTRY_BUFFER)
                cost_usd = entry_cost * OPTION_MULT
                if cost_usd > PREMIUM_CAP or cost_usd > cash:
                    continue
                cash -= cost_usd
                open_pos[tk] = Pos(tk, i, s.dates[i], exp_date, K, entry_cost,
                                   entry_iv, iv_floor, 1)
                placed_today += 1

        # ---- equity snapshot ----
        held_val = 0.0
        for tk, pos in open_pos.items():
            s = series[tk]
            idx = s.by_date.get(d)
            if idx is not None and idx >= pos.entry_idx:
                held_val += mtm(pos, s, idx)
        equity_curve.append((d, cash + held_val))

    return _stats(trades, equity_curve)


def _stats(trades: list[dict], equity_curve: list[tuple]) -> dict:
    if not equity_curve:
        return {"n": 0}
    vals = np.array([v for _, v in equity_curve], dtype=float)
    dates = [d for d, _ in equity_curve]
    final = float(vals[-1])
    total_ret = final / STARTING_CAPITAL - 1.0
    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    cagr = (final / STARTING_CAPITAL) ** (1.0 / years) - 1.0
    peak = vals[0]
    max_dd = 0.0
    for v in vals:
        peak = max(peak, v)
        max_dd = min(max_dd, v / peak - 1.0)
    dr = np.diff(vals) / vals[:-1]
    sharpe = float(np.sqrt(252) * dr.mean() / dr.std()) if dr.std() > 0 else 0.0
    mar = (cagr / abs(max_dd)) if max_dd < 0 else float("inf")

    r = np.array([t["opt_ret"] for t in trades], dtype=float) if trades else np.array([])
    wins = r[r > 0]
    losses = r[r <= 0]
    g = float(r[r > 0].sum()) if r.size else 0.0
    l = float(-r[r < 0].sum()) if r.size else 0.0
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
    held = np.array([t["bars_held"] for t in trades], dtype=float) if trades else np.array([])
    return {
        "n": len(trades),
        "final_equity": final,
        "total_ret": total_ret,
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "mar": mar,
        "win_rate": float((r > 0).mean()) if r.size else 0.0,
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "profit_factor": (g / l) if l > 0 else (float("inf") if g > 0 else 0.0),
        "mean_hold": float(held.mean()) if held.size else 0.0,
        "median_hold": float(np.median(held)) if held.size else 0.0,
        "reasons": reasons,
        "equity_curve": equity_curve,
    }


def _fmt_row(tag: str, s: dict) -> str:
    if not s.get("n"):
        return f"  {tag:<14} (no trades)"
    pf = s["profit_factor"]
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    mar = s["mar"]
    mar_s = "inf" if mar == float("inf") else f"{mar:.2f}"
    return (f"  {tag:<14} {s['n']:5d} | {s['cagr']*100:+6.1f} | {s['total_ret']*100:+7.1f} | "
            f"{s['max_dd']*100:6.1f} | {s['sharpe']:5.2f} | {mar_s:>5} | "
            f"{s['win_rate']*100:4.1f} | {pf_s:>5} | {s['mean_hold']:4.1f}d")


HEADER = ("  config         trades |  CAGR% | totRet% | maxDD% | Shrp |  MAR | win% |   PF | hold")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Fetching {len(SP500_TICKERS)} tickers x 5y...")
    bars = fetch_bars_bulk(SP500_TICKERS, period="5y")
    print(f"  got {len(bars)} usable tickers")

    series: dict[str, Series] = {}
    for tk, df in bars.items():
        s = build_series(tk, df)
        if s is not None:
            series[tk] = s
    print(f"  {len(series)} tickers with full history")
    sig_by_day = _signals(series)
    n_sig = sum(len(v) for v in sig_by_day.values())
    print(f"  {n_sig} raw entry signals (pre-capital-constraint)")

    lines: list[str] = []
    lines.append("Capital-constrained theta-aware OTM-call sim of the SMA-reversion strategy.")
    lines.append(f"Entry: {DOWN_BARS} down bars & close<SMA{SMA_PERIOD}, ADV>=${MIN_ADV_M:.0f}M. "
                 f"Instrument: {OTM_PCT:.0f}% OTM call, {DTE} DTE.")
    lines.append(f"Capital ${STARTING_CAPITAL:,.0f}; max {MAX_CONCURRENT} concurrent / "
                 f"{MAX_PER_DAY} per day; premium cap ${PREMIUM_CAP:,.0f}; 1 contract/trade.")
    lines.append(f"TP +{TP_PCT:.0f}% / disaster -{DISASTER_PCT:.0f}% (option); "
                 f"half-spread {HALF_SPREAD*100:.0f}% + {ENTRY_BUFFER*100:.0f}% entry buffer.")
    lines.append(f"Raw signals: {n_sig} across {len(series)} tickers "
                 f"(capital constraint admits far fewer).\n")

    base = Cfg()
    results: dict[str, dict] = {}

    lines.append("[STOP SWEEP]  exit = SMA-reclaim/TP/disaster/expiry + N-day loss-only stop")
    lines.append(HEADER)
    for sd in STOP_SWEEP:
        cfg = replace(base, time_loss_exit=sd)
        st = simulate(series, sig_by_day, cfg)
        tag = "no-stop" if sd is None else f"{sd}d-loss"
        results[tag] = st
        lines.append(_fmt_row(tag, st))
    lines.append("")

    # One-axis sensitivity at the no-stop exit (cleanest instrument-fit read).
    lines.append("[SENSITIVITY]  no-stop exit; one axis varied around base "
                 f"(DTE {DTE}, OTM {OTM_PCT:.0f}%, spread {HALF_SPREAD*100:.0f}%)")
    lines.append(HEADER)
    for dte in (30, 45, 60):
        st = simulate(series, sig_by_day, replace(base, dte=dte))
        lines.append(_fmt_row(f"DTE={dte}", st))
    for otm in (3.0, 5.0, 8.0):
        st = simulate(series, sig_by_day, replace(base, otm_pct=otm))
        lines.append(_fmt_row(f"OTM={otm:.0f}%", st))
    for hs in (0.01, 0.03, 0.05):
        st = simulate(series, sig_by_day, replace(base, half_spread=hs))
        lines.append(_fmt_row(f"spread={hs*100:.0f}%", st))
    lines.append("")

    # Exit-reason breakdown for the no-stop and 15d-loss configs.
    for tag in ("no-stop", "15d-loss"):
        st = results.get(tag)
        if st and st.get("n"):
            rs = ", ".join(f"{k} {v}" for k, v in sorted(st["reasons"].items()))
            lines.append(f"  exits [{tag}]: {rs}")
    lines.append("")

    # Equity-curve plot per stop.
    fig, ax = plt.subplots(figsize=(14, 7))
    for tag, st in results.items():
        if not st.get("n") and not st.get("equity_curve"):
            continue
        ec = st["equity_curve"]
        xs = [d for d, _ in ec]
        ys = [v for _, v in ec]
        ax.plot(xs, ys, linewidth=1.6, label=f"{tag} (CAGR {st['cagr']*100:+.1f}%, "
                f"maxDD {st['max_dd']*100:.0f}%, n={st['n']})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title("SMA-reversion as an OTM-call sleeve — equity by loss-stop length")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "sma_reversion_option_sim.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "sma_reversion_option_sim.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
