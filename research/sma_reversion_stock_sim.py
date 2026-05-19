"""Capital-constrained sim of the SMA-reversion strategy as LONG STOCK.

Same event-driven capital engine and signal set as
sma_reversion_option_sim.py, but the instrument is plain shares instead of
an OTM call: no theta, no IV crush, no leverage, no -50% disaster stop.
This isolates the strategy's portfolio behaviour from the option wrapper
that wiped the OTM-call sleeve out.

Engine (Series/stats shared with the option sim): union trading calendar;
each date MTM+exit open positions, then fill new signals; bounded by N
concurrent (equal-weight DEPLOY_FRAC/N of equity); liquidity-gated >=
$50M ADV; $10k start. Buy shares at close*(1+slip). Strategy exits only:
SMA reclaim (close back above SMA20) OR the loss-only time stop (held >=
N bars and still in a loss; winners ride to the reclaim); open at data
end closes "eod".

SIGNAL-SELECTION-RULE comparison: with ~5.4k signals competing for ~10
slots, which contested same-day signals you fill is the dominant lever.
Four rules are compared at the breadth sweet spot (N in RULE_N), at
no-stop and the 11d-loss stop:
  stretched : deepest below SMA20 first (what every prior run used)
  fresh     : fewest consecutive bars below SMA20 first
  adv       : highest 20d avg dollar volume first
  lowcorr   : lowest mean 60d return-correlation to the open book first
            (degrades to 'stretched' while the book is empty)

Run:
    python research/sma_reversion_stock_sim.py
Outputs:
    research/out/sma_reversion_stock_sim.png
    research/out/sma_reversion_stock_sim.txt
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.data import fetch_bars_bulk  # noqa: E402
from backend.sp500_tickers import SP500_TICKERS  # noqa: E402
from research.sma_reversion_option_sim import (  # noqa: E402
    Series, build_series, _stats, _fmt_row, HEADER,
    STARTING_CAPITAL, MIN_ADV_M, DOWN_BARS,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
STOCK_SLIP = 0.0005      # per-side equity slippage (5 bps)
DEPLOY_FRAC = 90.0       # target % of equity deployed when all slots full
LB_CORR = 60             # trailing-return lookback for the low-correlation rule
# Signal-selection rules tested when more signals fire than free slots:
#   stretched : deepest below SMA20 first (the rule all prior runs used)
#   fresh     : fewest consecutive bars below SMA20 first (recent breakdown)
#   adv       : highest 20d avg dollar volume first (most liquid)
#   lowcorr   : lowest mean 60d return-correlation to the open book first
RULES = ["stretched", "fresh", "adv", "lowcorr"]
RULE_N = [10, 20]        # concurrency points to compare rules at


def _logret(series: dict[str, Series]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for tk, s in series.items():
        lr = np.full(len(s.close), np.nan)
        lr[1:] = np.diff(np.log(s.close))
        out[tk] = lr
    return out


def _signals_rich(series: dict[str, Series]) -> dict:
    """date -> list of {tk, i, stretch, below, adv}. Same gate as the option
    sim's _signals (run==DOWN_BARS, close<SMA, ADV>=MIN_ADV_M), plus the
    fields the alternative selection rules need."""
    by_day: dict = {}
    for tk, s in series.items():
        below_run = 0
        for i in range(len(s.close)):
            below = (not np.isnan(s.sma[i])) and s.close[i] < s.sma[i]
            below_run = below_run + 1 if below else 0
            if np.isnan(s.sma[i]) or np.isnan(s.rv[i]):
                continue
            if s.drun[i] != DOWN_BARS or not below:
                continue
            if not (s.adv_m[i] >= MIN_ADV_M):
                continue
            by_day.setdefault(s.dates[i], []).append({
                "tk": tk, "i": i,
                "stretch": (s.sma[i] - s.close[i]) / s.sma[i],
                "below": below_run,
                "adv": float(s.adv_m[i]),
            })
    return by_day


def _mean_corr_to_book(sig: dict, open_tickers, d, series, logret) -> float:
    """Mean Pearson corr of the candidate's trailing LB_CORR log-returns vs
    each open position's, aligned to date d. 0.0 when the book is empty (so
    the rule degrades to most-stretched until a book exists)."""
    if not open_tickers:
        return 0.0
    ci = sig["i"]
    if ci < LB_CORR:
        return 1.0  # not enough history -> treat as maximally redundant
    cv = logret[sig["tk"]][ci - LB_CORR + 1: ci + 1]
    corrs = []
    for h in open_tickers:
        sh = series[h]
        hi = sh.by_date.get(d)
        if hi is None or hi < LB_CORR:
            continue
        hv = logret[h][hi - LB_CORR + 1: hi + 1]
        if cv.std() == 0 or hv.std() == 0:
            continue
        corrs.append(float(np.corrcoef(cv, hv)[0, 1]))
    return float(np.mean(corrs)) if corrs else 0.0


def _rank(rule: str, sigs: list[dict], open_tickers, d, series, logret) -> list[dict]:
    if rule == "stretched":
        key = lambda x: (-x["stretch"], x["tk"])
    elif rule == "fresh":
        key = lambda x: (x["below"], -x["stretch"], x["tk"])
    elif rule == "adv":
        key = lambda x: (-x["adv"], x["tk"])
    elif rule == "lowcorr":
        key = lambda x: (_mean_corr_to_book(x, open_tickers, d, series, logret),
                         -x["stretch"], x["tk"])
    else:
        raise ValueError(rule)
    return sorted(sigs, key=key)


def simulate_stock(series: dict[str, Series], sig_by_day: dict,
                   alloc_pct: float, time_loss_exit: int | None,
                   max_concurrent: int, max_per_day: int,
                   rule: str, logret: dict) -> dict:
    all_dates = sorted({d for s in series.values() for d in s.dates})
    cash = STARTING_CAPITAL
    open_pos: dict[str, dict] = {}
    trades: list[dict] = []
    equity_curve: list[tuple] = []
    last_date = all_dates[-1]

    def held_value(on_date) -> float:
        v = 0.0
        for tk, p in open_pos.items():
            s = series[tk]
            idx = s.by_date.get(on_date)
            if idx is not None and idx >= p["entry_idx"]:
                v += p["shares"] * s.close[idx]
        return v

    for d in all_dates:
        # ---- exits (deterministic ticker order) ----
        for tk in sorted(list(open_pos.keys())):
            p = open_pos[tk]
            s = series[tk]
            idx = s.by_date.get(d)
            if idx is None or idx <= p["entry_idx"]:
                continue
            held = idx - p["entry_idx"]
            close_ret = (s.close[idx] * (1 - STOCK_SLIP)) / p["entry_price"] - 1.0
            crossed_up = s.close[idx] > s.sma[idx]
            is_last = d == last_date
            time_stopped = (
                time_loss_exit is not None
                and held >= time_loss_exit
                and close_ret < 0.0
                and not crossed_up
            )
            reason = ("sma_cross" if crossed_up
                      else "time_loss" if time_stopped
                      else "eod" if is_last else None)
            if reason is not None:
                proceeds = p["shares"] * s.close[idx] * (1 - STOCK_SLIP)
                cash += proceeds
                trades.append({
                    "ticker": tk,
                    "entry_date": str(p["entry_date"]),
                    "exit_date": str(s.dates[idx]),
                    "bars_held": held,
                    "opt_ret": close_ret,          # keyed 'opt_ret' for shared _stats
                    "exit_reason": reason,
                    "cost": p["cost"],
                })
                del open_pos[tk]

        # ---- entries ----
        sigs = sig_by_day.get(d, [])
        if sigs and len(open_pos) < max_concurrent:
            equity_now = cash + held_value(d)
            placed_today = 0
            ranked = _rank(rule, sigs, set(open_pos.keys()), d, series, logret)
            for sig in ranked:
                if len(open_pos) >= max_concurrent or placed_today >= max_per_day:
                    break
                tk, i = sig["tk"], sig["i"]
                if tk in open_pos:
                    continue
                s = series[tk]
                px = s.close[i] * (1 + STOCK_SLIP)
                budget = min(equity_now * alloc_pct / 100.0, cash)
                shares = int(budget // px)
                if shares < 1:
                    continue
                cost = shares * px
                cash -= cost
                open_pos[tk] = {"entry_idx": i, "entry_date": s.dates[i],
                                "entry_price": px, "shares": shares, "cost": cost}
                placed_today += 1

        equity_curve.append((d, cash + held_value(d)))

    return _stats(trades, equity_curve)


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
    sig_by_day = _signals_rich(series)
    logret = _logret(series)
    n_sig = sum(len(v) for v in sig_by_day.values())
    print(f"  {n_sig} raw entry signals (pre-capital-constraint)")

    lines: list[str] = []
    lines.append("Signal-selection-rule comparison: SMA-reversion as LONG STOCK.")
    lines.append("Entry: 4 down bars & close<SMA20, ADV>=$50M. Instrument: shares (no leverage).")
    lines.append(f"Capital ${STARTING_CAPITAL:,.0f}; N slots equal-weighted at "
                 f"{DEPLOY_FRAC:.0f}%/N of equity; per-day throttle lifted (per_day=N).")
    lines.append("Exits: SMA reclaim, or the loss-only time stop. No TP / no disaster.")
    lines.append("Rules: stretched=deepest below SMA | fresh=fewest bars below SMA | "
                 "adv=most liquid | lowcorr=least corr to open book.")
    lines.append(f"Raw signals: {n_sig} across {len(series)} tickers "
                 f"(only the top-ranked fill the {RULE_N} slots).\n")

    plot_curves: list[tuple] = []

    for stop_tag, sd in (("no-stop", None), ("11d-loss", 11)):
        for n in RULE_N:
            lines.append(f"[SELECTION RULES — {stop_tag}, N={n}]  equal-weight "
                         f"{DEPLOY_FRAC:.0f}%/N")
            lines.append(HEADER)
            for rule in RULES:
                st = simulate_stock(series, sig_by_day, DEPLOY_FRAC / n, sd,
                                    n, n, rule, logret)
                lines.append(_fmt_row(rule, st))
                if sd is None and n == 10:
                    plot_curves.append((rule, st))
            lines.append("")

    fig, ax = plt.subplots(figsize=(14, 7))
    for rule, st in plot_curves:
        ec = st.get("equity_curve") or []
        if not ec:
            continue
        ax.plot([d for d, _ in ec], [v for _, v in ec], linewidth=1.6,
                label=f"{rule} (CAGR {st['cagr']*100:+.1f}%, maxDD {st['max_dd']*100:.0f}%, "
                      f"Sharpe {st['sharpe']:.2f}, n={st['n']})")
    ax.axhline(STARTING_CAPITAL, color="gray", lw=0.8, alpha=0.6)
    ax.set_title("SMA-reversion long-stock sleeve (no-stop, N=10) — equity by selection rule")
    ax.set_xlabel("date"); ax.set_ylabel("equity ($)")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "sma_reversion_stock_sim.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    out_txt = os.path.join(OUT_DIR, "sma_reversion_stock_sim.txt")
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"  -> {out_png}")
    print(f"  -> {out_txt}")


if __name__ == "__main__":
    main()
