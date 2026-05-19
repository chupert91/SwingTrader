"""Audit the user's real 2025 trading history (reference/DownloadTxnHistory.csv).

Pair Bought-To-Open with Sold-To-Close (and Bought with Sold for stock) by
contract spec, FIFO. Emit a per-trade table plus the distributions that
matter for standardizing rules into the bot:

  - hold-time distribution (the bot's 10d time stop has been a suspect)
  - %-return distribution (where do real exits actually land?)
  - DTE at entry / moneyness (am I really trading 5% OTM 45 DTE?)
  - per-trade $ P/L (user said "~$500 per win, tried to cap losses at $200")
  - profit factor / win rate / expectancy (real-world numbers)
  - call vs put split (mean-reversion both ways)

Outputs:
    research/out/personal_trade_audit.txt
    research/out/personal_trade_audit.csv  (per-trade rows)
"""
from __future__ import annotations

import csv
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime
from statistics import mean, median

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "reference", "DownloadTxnHistory.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

OPT_DESC = re.compile(
    r"^(CALL|PUT)\s+(\S+)\s+(\d{2}/\d{2}/\d{2})\s+([\d.]+)$"
)


def _parse_date(s: str) -> date:
    return datetime.strptime(s.strip(), "%m/%d/%y").date()


def _parse_opt(desc: str):
    """Returns (right, ticker, expiry_date, strike) or None."""
    m = OPT_DESC.match(desc.strip())
    if not m:
        return None
    right, ticker, expiry_s, strike_s = m.groups()
    return right, ticker, _parse_date(expiry_s), float(strike_s)


@dataclass
class OpenLot:
    activity_date: date
    contracts: int
    premium: float           # per-contract price (e.g., 8.10)
    cost_total: float        # signed cash impact incl commission (negative)
    commission: float


@dataclass
class Trade:
    ticker: str
    right: str               # CALL / PUT / STOCK
    strike: float | None
    expiry: date | None
    entry_date: date
    exit_date: date
    contracts: int           # for stock, shares
    entry_px: float          # per-contract or per-share
    exit_px: float
    entry_cost: float        # signed cash out at entry (negative for BTO/Bought)
    exit_proceeds: float     # signed cash in at exit (positive for STC/Sold)
    commission: float
    days_held: int = 0
    dte_at_entry: int | None = None
    moneyness_pct: float | None = None   # only filled if we can resolve spot

    @property
    def gross_pl(self) -> float:
        return self.exit_proceeds + self.entry_cost  # cost is negative

    @property
    def ret_pct(self) -> float:
        denom = abs(self.entry_cost)
        return self.gross_pl / denom if denom > 0 else 0.0


def load_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("Activity/Trade Date,"):
                header_line = ln.rstrip("\n")
                break
        header = next(csv.reader([header_line]))
        for line in f:
            if not line.strip() or line.startswith('"'):
                continue
            try:
                rec = next(csv.reader([line]))
            except StopIteration:
                continue
            if len(rec) < len(header):
                continue
            rows.append(dict(zip(header, rec)))
    return rows


def pair_options(rows: list[dict]) -> list[Trade]:
    open_books: dict[tuple, deque[OpenLot]] = defaultdict(deque)
    trades: list[Trade] = []

    # rows are newest-first; reverse so FIFO works
    rows = list(reversed(rows))

    for r in rows:
        atype = r["Activity Type"].strip()
        if atype not in ("Bought To Open", "Sold To Close"):
            continue
        opt = _parse_opt(r["Description"])
        if opt is None:
            continue
        right, ticker, expiry, strike = opt
        qty = float(r["Quantity #"])
        price = float(r["Price $"])
        amount = float(r["Amount $"])
        commission = float(r["Commission"] or 0)
        adate = _parse_date(r["Activity/Trade Date"])
        key = (ticker, right, strike, expiry)

        if atype == "Bought To Open":
            open_books[key].append(OpenLot(
                activity_date=adate, contracts=int(round(qty)),
                premium=price, cost_total=amount, commission=commission,
            ))
        else:  # Sold To Close
            remaining = int(round(-qty))      # qty is negative on closes
            proceeds_per_contract = amount / max(remaining, 1)
            book = open_books[key]
            while remaining > 0 and book:
                lot = book[0]
                take = min(lot.contracts, remaining)
                # Allocate proportional cost from the lot
                cost_alloc = lot.cost_total * (take / lot.contracts)
                comm_alloc = lot.commission * (take / lot.contracts)
                trades.append(Trade(
                    ticker=ticker, right=right, strike=strike, expiry=expiry,
                    entry_date=lot.activity_date, exit_date=adate,
                    contracts=take, entry_px=lot.premium, exit_px=price,
                    entry_cost=cost_alloc,
                    exit_proceeds=proceeds_per_contract * take,
                    commission=comm_alloc + commission * (take / max(int(round(-qty)), 1)),
                    days_held=(adate - lot.activity_date).days,
                    dte_at_entry=(expiry - lot.activity_date).days,
                ))
                lot.contracts -= take
                lot.cost_total -= cost_alloc
                lot.commission -= comm_alloc
                remaining -= take
                if lot.contracts == 0:
                    book.popleft()
            if remaining > 0:
                print(f"  WARN: STC with no open lot: {ticker} {right} {strike} {expiry} qty={remaining}")
    return trades


def pair_stock(rows: list[dict]) -> list[Trade]:
    open_books: dict[str, deque[OpenLot]] = defaultdict(deque)
    trades: list[Trade] = []
    rows = list(reversed(rows))
    for r in rows:
        atype = r["Activity Type"].strip()
        if atype not in ("Bought", "Sold"):
            continue
        ticker = r["Symbol"].strip()
        if not ticker:
            continue
        # description sometimes has "UNSOLICITED TRADE" — these ARE the stock rows
        try:
            qty = float(r["Quantity #"])
            price = float(r["Price $"])
            amount = float(r["Amount $"])
        except (ValueError, KeyError):
            continue
        adate = _parse_date(r["Activity/Trade Date"])
        if atype == "Bought":
            open_books[ticker].append(OpenLot(
                activity_date=adate, contracts=int(round(qty)),
                premium=price, cost_total=amount, commission=0.0,
            ))
        else:
            remaining = int(round(-qty))
            book = open_books[ticker]
            proceeds_per_share = amount / max(remaining, 1)
            while remaining > 0 and book:
                lot = book[0]
                take = min(lot.contracts, remaining)
                cost_alloc = lot.cost_total * (take / lot.contracts)
                trades.append(Trade(
                    ticker=ticker, right="STOCK", strike=None, expiry=None,
                    entry_date=lot.activity_date, exit_date=adate,
                    contracts=take, entry_px=lot.premium, exit_px=price,
                    entry_cost=cost_alloc,
                    exit_proceeds=proceeds_per_share * take,
                    commission=0.0,
                    days_held=(adate - lot.activity_date).days,
                    dte_at_entry=None,
                ))
                lot.contracts -= take
                lot.cost_total -= cost_alloc
                remaining -= take
                if lot.contracts == 0:
                    book.popleft()
    return trades


def _pctile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def _stats(trades: list[Trade], label: str) -> list[str]:
    if not trades:
        return [f"{label}: no trades"]
    wins = [t for t in trades if t.gross_pl > 0]
    losses = [t for t in trades if t.gross_pl <= 0]
    win_pls = [t.gross_pl for t in wins]
    loss_pls = [t.gross_pl for t in losses]
    rets = [t.ret_pct for t in trades]
    holds = [t.days_held for t in trades]
    dtes = [t.dte_at_entry for t in trades if t.dte_at_entry is not None]
    gross_win = sum(win_pls)
    gross_loss = abs(sum(loss_pls))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    lines = [
        f"--- {label} (n={len(trades)}) ---",
        f"  win rate    : {len(wins) / len(trades) * 100:5.1f}%  ({len(wins)}W / {len(losses)}L)",
        f"  total P/L   : ${sum(t.gross_pl for t in trades):+,.0f}",
        f"  profit fact : {pf:.2f}    (wins ${gross_win:+,.0f} / losses ${-gross_loss:+,.0f})",
        f"  expectancy  : ${mean(t.gross_pl for t in trades):+,.0f} / trade",
        f"  avg win     : ${mean(win_pls):+,.0f}    (median ${median(win_pls):+,.0f})" if wins else "  avg win     : --",
        f"  avg loss    : ${mean(loss_pls):+,.0f}    (median ${median(loss_pls):+,.0f})" if losses else "  avg loss    : --",
        f"  ret%        : median {median(rets) * 100:+5.1f}%  | mean {mean(rets) * 100:+5.1f}%  | p10 {_pctile(rets, 10) * 100:+5.1f}%  p25 {_pctile(rets, 25) * 100:+5.1f}%  p75 {_pctile(rets, 75) * 100:+5.1f}%  p90 {_pctile(rets, 90) * 100:+5.1f}%",
        f"  hold days   : median {median(holds):.0f}d   mean {mean(holds):.1f}d   p25 {_pctile([float(h) for h in holds], 25):.0f}d  p75 {_pctile([float(h) for h in holds], 75):.0f}d  max {max(holds)}d",
    ]
    if dtes:
        lines.append(f"  DTE @ entry : median {median(dtes):.0f}d   mean {mean(dtes):.1f}d   min {min(dtes)}d   max {max(dtes)}d")
    return lines


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows(CSV_PATH)
    opt_trades = pair_options(rows)
    stk_trades = pair_stock(rows)
    all_opt = sorted(opt_trades, key=lambda t: t.entry_date)
    all_stk = sorted(stk_trades, key=lambda t: t.entry_date)

    out: list[str] = []
    out.append("PERSONAL 2025 TRADING-HISTORY AUDIT")
    out.append(f"  source: reference/DownloadTxnHistory.csv  ({len(rows)} txn rows)")
    out.append(f"  paired option trades : {len(all_opt)}")
    out.append(f"  paired stock trades  : {len(all_stk)}")
    out.append("")

    # ---- per-trade table (options) ----
    out.append("PER-TRADE TABLE — OPTIONS")
    out.append(f"  {'entry':10}  {'exit':10}  {'tkr':5} {'cp':4} {'strike':>8} {'dte':>4} {'ctr':>3} {'ent$':>7} {'exit$':>7} {'P/L':>9} {'ret%':>7} {'hold':>5}")
    for t in all_opt:
        out.append(
            f"  {t.entry_date}  {t.exit_date}  {t.ticker:5} {t.right:4} {t.strike:>8.2f} {t.dte_at_entry or 0:>4d} {t.contracts:>3d}"
            f" {t.entry_px:>7.2f} {t.exit_px:>7.2f} {t.gross_pl:>+9.0f} {t.ret_pct * 100:>+6.1f}% {t.days_held:>4d}d"
        )
    out.append("")

    # ---- per-trade table (stock) ----
    if all_stk:
        out.append("PER-TRADE TABLE — STOCK")
        out.append(f"  {'entry':10}  {'exit':10}  {'tkr':5} {'qty':>5} {'ent$':>8} {'exit$':>8} {'P/L':>9} {'ret%':>7} {'hold':>5}")
        for t in all_stk:
            out.append(
                f"  {t.entry_date}  {t.exit_date}  {t.ticker:5} {t.contracts:>5d}"
                f" {t.entry_px:>8.2f} {t.exit_px:>8.2f} {t.gross_pl:>+9.0f} {t.ret_pct * 100:>+6.1f}% {t.days_held:>4d}d"
            )
        out.append("")

    # ---- summary blocks ----
    out.append("SUMMARY STATS")
    out.extend(_stats(all_opt, "ALL OPTION TRADES"))
    out.append("")
    out.extend(_stats([t for t in all_opt if t.right == "CALL"], "CALL TRADES (bullish mean-rev)"))
    out.append("")
    out.extend(_stats([t for t in all_opt if t.right == "PUT"], "PUT TRADES (bearish mean-rev)"))
    out.append("")
    if all_stk:
        out.extend(_stats(all_stk, "STOCK TRADES"))
        out.append("")

    # ---- per-ticker breakdown (options) ----
    by_tk: dict[str, list[Trade]] = defaultdict(list)
    for t in all_opt:
        by_tk[t.ticker].append(t)
    out.append("PER-TICKER (options)")
    out.append(f"  {'tkr':6} {'n':>3} {'win%':>6} {'P/L':>10} {'avg-ret%':>10} {'med-hold':>10}")
    for tk in sorted(by_tk, key=lambda k: -sum(t.gross_pl for t in by_tk[k])):
        ts = by_tk[tk]
        wins = sum(1 for t in ts if t.gross_pl > 0)
        out.append(
            f"  {tk:6} {len(ts):>3d} {wins / len(ts) * 100:>5.1f}% {sum(t.gross_pl for t in ts):>+10.0f} "
            f"{mean(t.ret_pct for t in ts) * 100:>+9.1f}% {median(t.days_held for t in ts):>9.0f}d"
        )
    out.append("")

    # ---- hold-time histogram ----
    bins = [(0, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 40), (41, 90)]
    out.append("HOLD-TIME HISTOGRAM (options)")
    out.append(f"  {'days':>10}  {'count':>6}  {'win%':>6}  {'mean-ret%':>10}")
    for lo, hi in bins:
        ts = [t for t in all_opt if lo <= t.days_held <= hi]
        if not ts:
            continue
        wr = sum(1 for t in ts if t.gross_pl > 0) / len(ts) * 100
        mr = mean(t.ret_pct for t in ts) * 100
        out.append(f"  {lo:>4d}-{hi:<4d}  {len(ts):>6d}  {wr:>5.1f}%  {mr:>+9.1f}%")
    out.append("")

    # ---- exit-bucket distribution ----
    buckets = [
        (-1.0, -0.50, "<-50% (disaster)"),
        (-0.50, -0.20, "-50% .. -20%"),
        (-0.20, 0.00, "-20% .. 0%"),
        (0.00, 0.20, "0% .. +20%"),
        (0.20, 0.40, "+20% .. +40%"),
        (0.40, 0.80, "+40% .. +80%"),
        (0.80, 100.0, ">+80%"),
    ]
    out.append("EXIT %-RETURN BUCKETS (options)")
    out.append(f"  {'bucket':>20}  {'count':>6}  {'sum P/L':>10}")
    for lo, hi, lbl in buckets:
        ts = [t for t in all_opt if lo <= t.ret_pct < hi]
        out.append(f"  {lbl:>20}  {len(ts):>6d}  {sum(t.gross_pl for t in ts):>+10.0f}")
    out.append("")

    # ---- comparison against current bot defaults ----
    bot_dte_lo, bot_dte_hi = 45, 60
    in_bot_dte = [t for t in all_opt if t.dte_at_entry is not None and bot_dte_lo <= t.dte_at_entry <= bot_dte_hi]
    out.append("COMPARISON vs CURRENT BOT DEFAULTS")
    out.append(f"  bot DTE band     : {bot_dte_lo}-{bot_dte_hi}d")
    out.append(f"  real trades w/ DTE in band : {len(in_bot_dte)} / {len(all_opt)} ({len(in_bot_dte) / max(len(all_opt), 1) * 100:.0f}%)")
    out.append(f"  real-trade DTE median       : {median([t.dte_at_entry for t in all_opt if t.dte_at_entry is not None]):.0f}d")
    bot_tp_pct = 0.30
    hit_30 = [t for t in all_opt if t.ret_pct >= bot_tp_pct]
    out.append(f"  bot TP threshold : +{bot_tp_pct * 100:.0f}%")
    out.append(f"  real trades at >=+30% : {len(hit_30)} / {len(all_opt)} ({len(hit_30) / max(len(all_opt), 1) * 100:.0f}%)")
    bot_disaster = -0.50
    hit_disaster = [t for t in all_opt if t.ret_pct <= bot_disaster]
    out.append(f"  bot disaster stop : {bot_disaster * 100:.0f}%")
    out.append(f"  real trades at <=-50% : {len(hit_disaster)} / {len(all_opt)} ({len(hit_disaster) / max(len(all_opt), 1) * 100:.0f}%)")
    bot_time_stop = 10
    held_past_time = [t for t in all_opt if t.days_held > bot_time_stop and t.gross_pl > 0]
    out.append(f"  bot time stop    : {bot_time_stop}d (loss-only)")
    out.append(f"  real WINS held past {bot_time_stop}d : {len(held_past_time)} / {sum(1 for t in all_opt if t.gross_pl > 0)} "
               f"({len(held_past_time) / max(sum(1 for t in all_opt if t.gross_pl > 0), 1) * 100:.0f}% — would the bot have cut?)")

    out.append("")
    out.append(f"NET REALIZED P/L (options): ${sum(t.gross_pl for t in all_opt):+,.0f}")
    out.append(f"NET REALIZED P/L (stock)  : ${sum(t.gross_pl for t in all_stk):+,.0f}")
    out.append(f"COMBINED                  : ${sum(t.gross_pl for t in all_opt) + sum(t.gross_pl for t in all_stk):+,.0f}")

    txt = "\n".join(out)
    print(txt)
    out_txt = os.path.join(OUT_DIR, "personal_trade_audit.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(txt)

    out_csv = os.path.join(OUT_DIR, "personal_trade_audit.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["entry_date", "exit_date", "ticker", "right", "strike",
                    "expiry", "dte_at_entry", "contracts", "entry_px",
                    "exit_px", "entry_cost", "exit_proceeds", "gross_pl",
                    "ret_pct", "days_held"])
        for t in all_opt + all_stk:
            w.writerow([t.entry_date, t.exit_date, t.ticker, t.right,
                        t.strike, t.expiry, t.dte_at_entry, t.contracts,
                        t.entry_px, t.exit_px, t.entry_cost, t.exit_proceeds,
                        t.gross_pl, t.ret_pct, t.days_held])

    print(f"\n  -> {out_txt}")
    print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()
