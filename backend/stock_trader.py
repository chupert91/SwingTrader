"""Long-stock bot - orchestration / cron brain.

The thesis-2 capitulation strategy traded as plain stock, running side by
side with the AI options trader. One entrypoint, run_cron(), called by the
Vercel cron and the manual "Run now" button. Fully separate from ai_trader:
own KV namespace (stock_store), own Alpaca paper account (alpaca_stock).

Each run, three phases:

  reconcile (always, even if the kill-switch is off, so in-flight trades
  stay managed): match each open trade to its Alpaca orders/positions,
  detect entry fills, detect the +target% sell-limit fill, enforce the
  time stop, finalize P&L on close.

  entry (market open + kill-switch on, capped by a per-ET-day counter):
  scan the S&P universe for the thesis-2 signal (stock_strategy), size each
  position at position_size_pct% of equity, place a marketable buy-limit.

  equity snapshot + run-log.

Exit design (thesis-2-faithful, research/thesis2/THESIS.md):
  - profit_target_pct (default 20): a resting GTC sell-limit at
    fill*(1+pct/100). A native sell of stock you own - no "uncovered"
    rejection, unlike the options bot.
  - time_stop_days (default 45): hard time stop regardless of P&L.
  No hard disaster stop - the thesis exit study used target + time-stop only.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from backend import alpaca_stock as alp
from backend import stock_store
from backend import stock_strategy

logger = logging.getLogger(__name__)

CASH_SLACK = 0.97  # leave a small cash buffer; never deploy into margin


# ---- helpers --------------------------------------------------------------

def _f(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _et_date(clock: dict) -> str:
    """Trading date from the Alpaca clock timestamp (DST-correct)."""
    ts = clock.get("timestamp")
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        dt = datetime.now(timezone.utc)
    return dt.date().isoformat()


def _trading_days_since(iso_ts: str) -> int:
    """Weekday count from a fill timestamp to now (holidays ignored - the
    45-day stop is a soft heuristic)."""
    try:
        start = datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00")).date()
    except Exception:
        return 0
    end = datetime.now(timezone.utc).date()
    days = 0
    d = start
    while d < end:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days += 1
    return days


# ---- exit / bookkeeping ---------------------------------------------------

def _arm_tp(trade: dict, settings: dict, actions: list) -> None:
    """Ensure a resting +profit_target% GTC sell-limit exists for an open
    trade. Re-arms if it is missing (e.g. externally canceled)."""
    entry = trade.get("entry") or {}
    fill = _f(entry.get("fill_price"))
    qty = int(entry.get("qty") or 0)
    if not fill or qty < 1:
        return
    tp = trade.get("tp") or {}
    if tp.get("order_id"):
        try:
            o = alp.get_order(tp["order_id"])
            if o.get("status") in ("new", "accepted", "pending_new",
                                    "partially_filled", "held"):
                return                       # still working
            if o.get("status") == "filled":
                return                       # _reconcile handles the close
        except alp.AlpacaError:
            pass
    target_pct = _f(settings.get("profit_target_pct"), 20.0) or 20.0
    tp_price = round(fill * (1 + target_pct / 100.0), 2)
    try:
        o = alp.submit_order(trade["ticker"], qty, "sell", "limit", "gtc",
                             limit_price=tp_price)
        trade["tp"] = {"order_id": o.get("id"), "limit_price": tp_price,
                       "status": "working"}
        actions.append(f"{trade['ticker']}: armed +{target_pct:.0f}% sell limit {tp_price}")
    except alp.AlpacaError as exc:
        actions.append(f"{trade['ticker']}: TP arm failed: {exc}")


def _finalize_close(trade: dict, exit_price: float, reason: str, actions: list) -> None:
    entry = trade.get("entry") or {}
    ep = _f(entry.get("fill_price")) or 0.0
    qty = int(entry.get("qty") or 0)
    realized = round((exit_price - ep) * qty, 2)
    pct = round((exit_price / ep - 1.0) * 100.0, 1) if ep else 0.0
    trade["status"] = "closed"
    trade["exit"] = {**(trade.get("exit") or {}), "reason": reason,
                     "fill_price": exit_price, "filled_at": stock_store.now_iso()}
    trade["pnl"] = {"realized_usd": realized, "realized_pct": pct}
    actions.append(f"{trade['ticker']}: CLOSED {reason} pnl ${realized} ({pct}%)")
    oid = (trade.get("tp") or {}).get("order_id")
    if oid:
        alp.cancel_order(oid)


def _reconcile(trade: dict, settings: dict, clock_open: bool, actions: list) -> None:
    st = trade.get("status")

    if st == "pending_entry":
        entry = trade.get("entry") or {}
        oid = entry.get("order_id")
        if not oid:
            return
        try:
            o = alp.get_order(oid)
        except alp.AlpacaError as exc:
            actions.append(f"{trade['ticker']}: entry order lookup failed: {exc}")
            return
        ostat = o.get("status")
        if ostat == "filled":
            entry["fill_price"] = _f(o.get("filled_avg_price"))
            entry["qty"] = int(_f(o.get("filled_qty"), 0) or 0)
            entry["filled_at"] = o.get("filled_at") or stock_store.now_iso()
            trade["entry"] = entry
            trade["status"] = "open"
            actions.append(f"{trade['ticker']}: ENTRY filled "
                           f"{entry['qty']}sh @ {entry['fill_price']}")
            _arm_tp(trade, settings, actions)
        elif ostat in ("canceled", "expired", "rejected", "done_for_day"):
            trade["status"] = "canceled"
            trade["exit"] = {"reason": "entry_unfilled"}
            actions.append(f"{trade['ticker']}: entry abandoned (unfilled)")
        return

    if st in ("open", "closing"):
        ticker = trade["ticker"]

        # Take-profit hit? (resting GTC limit - Alpaca executes it on its own.)
        tp = trade.get("tp") or {}
        if tp.get("order_id"):
            try:
                o = alp.get_order(tp["order_id"])
                if o.get("status") == "filled":
                    _finalize_close(trade, _f(o.get("filled_avg_price")) or 0.0,
                                    "take_profit", actions)
                    return
            except alp.AlpacaError:
                pass

        # Confirm a submitted time-stop close order.
        if st == "closing":
            ex = trade.get("exit") or {}
            if ex.get("order_id"):
                try:
                    o = alp.get_order(ex["order_id"])
                    if o.get("status") == "filled":
                        _finalize_close(trade, _f(o.get("filled_avg_price")) or 0.0,
                                        ex.get("reason", "time_stop"), actions)
                        return
                except alp.AlpacaError:
                    pass
            return

        # Still open: confirm the position still exists.
        entry = trade.get("entry") or {}
        pos = None
        positions_ok = True
        try:
            for p in alp.list_positions():
                if p.get("symbol") == ticker:
                    pos = p
                    break
        except alp.AlpacaError:
            positions_ok = False
        if positions_ok and pos is None:
            trade["status"] = "closed"
            trade["exit"] = {"reason": "external"}
            trade["pnl"] = trade.get("pnl") or {"realized_usd": None, "realized_pct": None}
            actions.append(f"{ticker}: position gone (external close)")
            return

        # Time stop (hard, regardless of P&L).
        held = _trading_days_since(entry.get("filled_at", ""))
        ts_days = settings.get("time_stop_days")
        if ts_days is not None and held >= int(ts_days) and clock_open:
            oid = (trade.get("tp") or {}).get("order_id")
            if oid:
                alp.cancel_order(oid)
            try:
                o = alp.close_position(ticker)
                trade["status"] = "closing"
                trade["exit"] = {"reason": "time_stop", "order_id": o.get("id"),
                                 "submitted_at": stock_store.now_iso()}
                actions.append(f"{ticker}: time_stop ({held}d) -> closing")
                return
            except alp.AlpacaError as exc:
                actions.append(f"{ticker}: time_stop close failed: {exc}")

        # Still open and healthy: keep the +target% sell limit armed.
        if trade.get("status") == "open":
            _arm_tp(trade, settings, actions)


# ---- entry ----------------------------------------------------------------

def _run_entries(settings: dict, account: dict, max_new: int,
                 actions: list, skips: list) -> int:
    placed = 0
    max_conc = int(settings["max_concurrent"])
    held = {t["ticker"] for t in stock_store.list_trades()
            if t.get("status") in ("pending_entry", "open", "closing")}
    slots = max_conc - len(held)
    if slots <= 0:
        skips.append(f"no slots ({len(held)}/{max_conc} used)")
        return 0
    # Clamp to the remaining per-day budget so a capitulation-day cluster of
    # correlated candidates doesn't fill every slot at once.
    if slots > max_new:
        slots = max_new
    if slots <= 0:
        skips.append("per-day entry budget exhausted")
        return 0

    equity = _f(account.get("equity"), 0.0) or 0.0
    cash = _f(account.get("cash"), 0.0) or 0.0   # unlevered: spend cash, not margin
    size_pct = float(settings["position_size_pct"])
    buf = float(settings["entry_limit_buffer_pct"])
    pos_dollars = equity * size_pct / 100.0
    if pos_dollars <= 0:
        skips.append("no equity to size positions")
        return 0

    try:
        candidates = stock_strategy.scan_candidates(settings)
    except Exception as exc:
        logger.exception("stock scan failed")
        skips.append(f"scan error: {exc!r}")
        return 0
    if not candidates:
        skips.append("no candidates pass the entry filters")
        return 0

    for c in candidates:
        if slots <= 0:
            break
        if c["ticker"] in held:
            continue
        price = float(c["price"])
        limit = round(price * (1 + buf / 100.0), 2)
        qty = int(pos_dollars // limit)
        if qty < 1:
            skips.append(f"{c['ticker']}: ${price:.0f}/sh > position size ${pos_dollars:.0f}")
            continue
        cost = qty * limit
        if cost > cash * CASH_SLACK:
            skips.append(f"{c['ticker']}: cost ${cost:.0f} > cash ${cash:.0f}")
            continue
        try:
            o = alp.submit_order(c["ticker"], qty, "buy", "limit", "day",
                                 limit_price=limit)
        except alp.AlpacaError as exc:
            skips.append(f"{c['ticker']}: order rejected: {exc}")
            continue
        trade = stock_store.new_trade(c["ticker"], {
            "z": c["z"], "drawdown_pct": c["drawdown_pct"], "rsi": c["rsi"],
            "price_at_signal": price, "touch_date": c["touch_date"],
        })
        stock_store.update_trade(trade["id"], {
            "status": "pending_entry",
            "entry": {"order_id": o.get("id"), "limit_price": limit, "qty": qty,
                      "submitted_at": stock_store.now_iso()},
        })
        cash -= cost
        slots -= 1
        placed += 1
        held.add(c["ticker"])
        actions.append(f"{c['ticker']}: BUY {qty}sh limit {limit} "
                       f"(z={c['z']}, dd={c['drawdown_pct']}%, rsi={c['rsi']})")
    return placed


# ---- entrypoint -----------------------------------------------------------

def run_cron(manual: bool = False) -> dict:
    rec: dict = {"phase": "manual" if manual else "cron", "actions": [],
                 "skips": [], "errors": [], "ran": False}

    if not alp.is_configured():
        rec["reason"] = "stock-bot Alpaca credentials not configured"
        stock_store.log_run(rec)
        return {"ok": False, **rec}

    settings = stock_store.get_settings()

    try:
        clock = alp.get_clock()
        account = alp.get_account()
    except alp.AlpacaError as exc:
        rec["errors"].append(f"alpaca unreachable: {exc}")
        stock_store.log_run(rec)
        return {"ok": False, **rec}

    is_open = bool(clock.get("is_open"))
    rec["market_open"] = is_open
    rec["ran"] = True

    # Reconcile always runs so in-flight trades stay managed even if the
    # kill-switch was turned off after entry.
    trades = stock_store.list_trades()
    dirty = False
    for t in trades:
        if t.get("status") in ("pending_entry", "open", "closing"):
            dirty = True
            try:
                _reconcile(t, settings, is_open, rec["actions"])
            except Exception as exc:
                logger.exception("reconcile failed for %s", t.get("id"))
                rec["errors"].append(f"{t.get('ticker')}: reconcile {exc!r}")
    if dirty:
        stock_store._save_trades(trades)

    # Entry phase: market open + kill-switch on, capped by a per-ET-day
    # counter (a slot freed intraday back-fills the same day).
    et_today = _et_date(clock)
    state = stock_store.get_state()
    placed_today = (int(state.get("entries_today", 0))
                    if state.get("entry_day") == et_today else 0)
    max_per_day = int(settings["max_entries_per_day"])

    if not settings.get("enabled"):
        rec["skips"].append("kill-switch off - entries skipped")
    elif not is_open:
        rec["skips"].append("market closed - entries skipped")
    elif placed_today >= max_per_day:
        rec["skips"].append(
            f"per-day entry cap reached ({placed_today}/{max_per_day} for {et_today})")
    else:
        placed = _run_entries(settings, account, max_per_day - placed_today,
                              rec["actions"], rec["skips"])
        rec["entries_placed"] = placed
        placed_today += placed

    stock_store.set_state({"entry_day": et_today, "entries_today": placed_today})

    # Equity snapshot + realized cumulative.
    realized_cum = round(sum(
        (_f((t.get("pnl") or {}).get("realized_usd"), 0.0) or 0.0)
        for t in stock_store.list_trades() if t.get("status") == "closed"
    ), 2)
    stock_store.append_equity(
        _f(account.get("equity"), 0.0) or 0.0,
        _f(account.get("cash"), 0.0) or 0.0,
        realized_cum,
    )
    rec["equity"] = _f(account.get("equity"))

    stock_store.log_run(rec)
    return {"ok": True, **rec}


def snapshot() -> dict:
    """Full read-only state for the stock-bot page. Degrades gracefully when
    the second Alpaca account's credentials are missing."""
    settings = stock_store.get_settings()
    trades = stock_store.list_trades()
    account = None
    marks: dict[str, dict] = {}
    configured = alp.is_configured()
    if configured:
        try:
            account = alp.get_account()
        except alp.AlpacaError:
            account = None
        try:
            for p in alp.list_positions():
                marks[p.get("symbol")] = p
        except alp.AlpacaError:
            pass

    open_t, closed_t = [], []
    for t in trades:
        if t.get("status") in ("closed", "canceled"):
            closed_t.append(t)
        else:
            tt = dict(t)
            m = marks.get(tt.get("ticker"))
            if m:
                tt["mark"] = {
                    "current_price": _f(m.get("current_price")),
                    "market_value": _f(m.get("market_value")),
                    "unrealized_pl": _f(m.get("unrealized_pl")),
                    "unrealized_plpc": _f(m.get("unrealized_plpc")),
                }
            ent = tt.get("entry") or {}
            if ent.get("filled_at"):
                tt["trading_days_held"] = _trading_days_since(ent["filled_at"])
            open_t.append(tt)

    closed_pnl = [t for t in closed_t if (t.get("pnl") or {}).get("realized_usd") is not None]
    wins = [t for t in closed_pnl if t["pnl"]["realized_usd"] > 0]
    losses = [t for t in closed_pnl if t["pnl"]["realized_usd"] <= 0]
    stats = {
        "closed_count": len(closed_pnl),
        "win_rate": round(100.0 * len(wins) / len(closed_pnl), 1) if closed_pnl else None,
        "avg_win": round(sum(t["pnl"]["realized_usd"] for t in wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(t["pnl"]["realized_usd"] for t in losses) / len(losses), 2) if losses else None,
        "total_realized": round(sum(t["pnl"]["realized_usd"] for t in closed_pnl), 2),
    }

    return {
        "configured": configured,
        "settings": settings,
        "account": account,
        "open_trades": list(reversed(open_t)),
        "closed_trades": list(reversed(closed_t))[:100],
        "stats": stats,
        "equity": stock_store.equity_series(),
        "run_log": stock_store.run_log(100),
    }


__all__ = ["run_cron", "snapshot"]
