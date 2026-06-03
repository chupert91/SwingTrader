"""Autonomous AI options trader — orchestration / cron brain.

One entrypoint, run_cron(), called by the hourly Vercel cron and the manual
"Run now" button. Responsibilities:

  bookkeeping (always, even if the kill-switch is off so in-flight trades
  stay managed): reconcile each open trade against Alpaca orders/positions,
  detect fills, (optionally) re-arm the resting take-profit, enforce the
  $-cap / %-cap disaster stop, sigma-target exit, and time stop, finalize
  P&L on close.

  entry (re-evaluated every run when market is open AND enabled, up to
  max_entries_per_day new positions per ET day via a per-day counter so a
  slot freed intraday back-fills the same day): scan the validated
  oversold-call signal on the volatile-universe basket, walk the tier-
  ranked list, and place marketable buy-limit orders subject to the
  per-trade premium cap, the max-concurrent cap, and Alpaca options
  buying power.

Exit design:
  - take_profit_pct (default None): if set, resting sell-limit at
    entry*(1+pct/100), Alpaca-day-TIF, re-armed each session.
  - disaster_stop (software, %-of-mark OR $-cap-on-unrealized, either or
    both can be set; whichever triggers first fires the close). Native
    stop-sell on a long option is rejected by Alpaca as "uncovered" so we
    enforce from the mark in this job.
  - sigma_target (default 0.0): close when the underlying's 252d-LOG z
    reverts back to >= target. The validated "let winners run to
    capitulation un-wind" exit, per the volatile-universe sweep (PF 2.32).
  - time_stop_days (default 45): hard time stop regardless of P&L.

Loss control is sizing + concurrency + $-cap disaster + time stop, not a
tight % stop. The default config tracks research/out/
volatile_universe_sweep.txt R6 (the only +CAGR / PF>2 / Sharpe>0.5 row).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from backend import ai_store
from backend import ai_strategy
from backend import alpaca_trading as at

logger = logging.getLogger(__name__)

OPTION_MULTIPLIER = 100
BUYING_POWER_SLACK = 0.97  # don't deploy the last few % of options BP
MAX_REPRICE_ATTEMPTS = 1


# ---- helpers --------------------------------------------------------------

def _f(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _et_date(clock: dict) -> str:
    """Eastern date string from the Alpaca clock timestamp (DST-correct
    because the clock carries an offset-aware ISO timestamp)."""
    ts = clock.get("timestamp")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.now(timezone.utc)
    # Alpaca's clock timestamp is already US/Eastern wall-clock with offset;
    # taking its date is the trading date.
    return dt.date().isoformat()


def _side_of(trade: dict) -> str:
    """'put' or 'call'. Pre-puts-sleeve trades have no type field; treat as
    calls so back-compat is preserved (the calls bot was the only writer
    until the puts sleeve shipped)."""
    return (trade.get("contract") or {}).get("type", "call")


def _trading_days_until(iso_date: str) -> int | None:
    """Weekday count from today to the expiration date (holidays ignored --
    same approximation as _trading_days_since). Returns None if the date
    can't be parsed. Used by the close-before-expiry exit to avoid the
    ITM auto-exercise / cash-settlement P&L-attribution loss."""
    try:
        y, m, d = (int(x) for x in iso_date.split("-"))
        target = date(y, m, d)
    except Exception:
        return None
    today = datetime.now(timezone.utc).date()
    if target <= today:
        return 0
    days = 0
    d = today
    while d < target:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days += 1
    return days


def _trading_days_since(iso_ts: str) -> int:
    """Weekday count from a fill timestamp to now (holidays ignored — the
    10-day stop is a soft heuristic; off-by-one over two weeks is fine)."""
    try:
        start = datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).date()
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

def _arm_exits(trade: dict, settings: dict, actions: list) -> None:
    """Ensure a resting +TP limit and (optional) disaster stop exist for an
    open trade. Re-arms anything that lapsed (day-TIF) or is missing.

    take_profit_pct=None disables the resting TP entirely — the bot then
    rides to the sigma-target / $-cap / time-stop exit checked from the
    mark in _reconcile. Default config since the volatile-universe sweep
    is "no TP, sigma-revert at z=0" (PF 2.32)."""
    entry = trade.get("entry") or {}
    fill = _f(entry.get("fill_price"))
    qty = int(entry.get("qty") or 1)
    symbol = (trade.get("contract") or {}).get("symbol")
    if not fill or not symbol:
        return

    # Per-leg settings (puts_* / call defaults) -- so the two sleeves can be
    # tuned independently without one stepping on the other's TP / disaster.
    side = _side_of(trade)
    tp_key  = "puts_take_profit_pct"      if side == "put" else "take_profit_pct"
    en_key  = "puts_disaster_stop_enabled" if side == "put" else "disaster_stop_enabled"
    pct_key = "puts_disaster_stop_pct"    if side == "put" else "disaster_stop_pct"
    usd_key = "puts_disaster_stop_usd"    if side == "put" else "disaster_stop_usd"

    tp_pct = _f(settings.get(tp_key))
    if tp_pct is None:
        # No TP configured. Cancel any lingering TP order from a prior config.
        oid = (trade.get("tp") or {}).get("order_id")
        if oid:
            try:
                at.cancel_order(oid)
            except at.AlpacaError:
                pass
        trade["tp"] = None
    else:
        tp = trade.get("tp") or {}
        need_tp = True
        if tp.get("order_id"):
            try:
                o = at.get_order(tp["order_id"])
                if o.get("status") in ("new", "accepted", "pending_new", "partially_filled", "held"):
                    need_tp = False
                elif o.get("status") == "filled":
                    need_tp = False  # handled by _reconcile as a close
            except at.AlpacaError:
                pass
        if need_tp:
            tp_price = round(fill * (1 + tp_pct / 100.0), 2)
            try:
                o = at.submit_order(symbol, qty, "sell", "limit", "day", limit_price=tp_price)
                trade["tp"] = {"order_id": o.get("id"), "limit_price": tp_price, "status": "working"}
                actions.append(f"{trade['ticker']}: armed TP limit {tp_price}")
            except at.AlpacaError as exc:
                actions.append(f"{trade['ticker']}: TP arm failed: {exc}")

    if not settings.get(en_key):
        trade["disaster_stop"] = None
        return
    # Software stop, not a native order: Alpaca rejects a resting stop-sell
    # on a long option as an "uncovered option" (account not approved for
    # naked short calls), even though it is a closing order. We record the
    # threshold(s) here and enforce them from the mark in _reconcile.
    # Either disaster_stop_pct OR disaster_stop_usd (or both) can be set —
    # whichever fires first triggers the close. Defaults are pct=None and
    # usd=200 ("cap each loss at ~$200" — the user's real rule shape).
    ds_pct = _f(settings.get(pct_key))
    ds_usd = _f(settings.get(usd_key))
    trade["disaster_stop"] = {
        "mode": "software",
        "pct": ds_pct,
        "usd": ds_usd,
        # NB: for puts the "stop_price_pct" hint is also a fall-from-fill
        # threshold -- option price still falls when the trade goes against
        # us regardless of side -- so the same formula applies.
        "stop_price_pct": round(fill * (1 - ds_pct / 100.0), 2) if ds_pct else None,
    }


def _finalize_close(trade: dict, exit_price: float, reason: str, actions: list) -> None:
    entry = trade.get("entry") or {}
    ep = _f(entry.get("fill_price")) or 0.0
    qty = int(entry.get("qty") or 1)
    realized = round((exit_price - ep) * OPTION_MULTIPLIER * qty, 2)
    pct = round((exit_price / ep - 1.0) * 100.0, 1) if ep else 0.0
    trade["status"] = "closed"
    trade["exit"] = {**(trade.get("exit") or {}), "reason": reason,
                     "fill_price": exit_price, "filled_at": ai_store.now_iso()}
    trade["pnl"] = {"realized_usd": realized, "realized_pct": pct}
    actions.append(f"{trade['ticker']}: CLOSED {reason} pnl ${realized} ({pct}%)")
    # Cancel any sibling resting order.
    for k in ("tp", "disaster_stop"):
        oid = (trade.get(k) or {}).get("order_id")
        if oid:
            at.cancel_order(oid)


def _reconcile(trade: dict, settings: dict, clock_open: bool, actions: list,
               z_by_ticker: dict[str, float] | None = None) -> None:
    z_by_ticker = z_by_ticker or {}
    st = trade.get("status")

    if st == "pending_entry":
        entry = trade.get("entry") or {}
        oid = entry.get("order_id")
        if not oid:
            return
        try:
            o = at.get_order(oid)
        except at.AlpacaError as exc:
            actions.append(f"{trade['ticker']}: entry order lookup failed: {exc}")
            return
        ostat = o.get("status")
        if ostat == "filled":
            fp = _f(o.get("filled_avg_price"))
            entry["fill_price"] = fp
            entry["qty"] = int(_f(o.get("filled_qty"), 1) or 1)
            entry["filled_at"] = o.get("filled_at") or ai_store.now_iso()
            trade["entry"] = entry
            trade["status"] = "open"
            actions.append(f"{trade['ticker']}: ENTRY filled @ {fp}")
            _arm_exits(trade, settings, actions)
        elif ostat in ("canceled", "expired", "rejected", "done_for_day"):
            attempts = entry.get("reprice_attempts", 0)
            if attempts < MAX_REPRICE_ATTEMPTS and clock_open:
                # One reprice at the fresh ask; else abandon (stale signal).
                q = at.latest_option_quote((trade.get("contract") or {}).get("symbol", ""))
                ask = (q or {}).get("ask")
                if ask:
                    lp = round(ask * (1 + settings["entry_limit_buffer_pct"] / 100.0), 2)
                    try:
                        no = at.submit_order(trade["contract"]["symbol"], 1, "buy", "limit", "day", limit_price=lp)
                        entry.update({"order_id": no.get("id"), "limit_price": lp,
                                      "reprice_attempts": attempts + 1})
                        trade["entry"] = entry
                        actions.append(f"{trade['ticker']}: entry repriced -> {lp}")
                        return
                    except at.AlpacaError as exc:
                        actions.append(f"{trade['ticker']}: reprice failed: {exc}")
            trade["status"] = "canceled"
            trade["exit"] = {"reason": "entry_unfilled"}
            actions.append(f"{trade['ticker']}: entry abandoned (unfilled)")
        return

    if st in ("open", "closing"):
        symbol = (trade.get("contract") or {}).get("symbol", "")

        # Take-profit hit? (resting limit — Alpaca executes this on its own.)
        tp = trade.get("tp") or {}
        if tp.get("order_id"):
            try:
                o = at.get_order(tp["order_id"])
                if o.get("status") == "filled":
                    _finalize_close(trade, _f(o.get("filled_avg_price")) or 0.0,
                                    "take_profit", actions)
                    return
            except at.AlpacaError:
                pass

        # Confirm a submitted closing order (time stop or software disaster).
        if st == "closing":
            ex = trade.get("exit") or {}
            if ex.get("order_id"):
                try:
                    o = at.get_order(ex["order_id"])
                    if o.get("status") == "filled":
                        _finalize_close(trade, _f(o.get("filled_avg_price")) or 0.0,
                                        ex.get("reason", "time_stop"), actions)
                        return
                except at.AlpacaError:
                    pass
            return

        # --- still open: fetch this position's mark once ------------------
        entry = trade.get("entry") or {}
        pos = None
        positions_ok = True
        try:
            for p in at.list_option_positions():
                if p.get("symbol") == symbol:
                    pos = p
                    break
        except at.AlpacaError:
            positions_ok = False

        # Position vanished (manual/external close).
        if positions_ok and symbol and pos is None:
            trade["status"] = "closed"
            trade["exit"] = {"reason": "external"}
            trade["pnl"] = trade.get("pnl") or {"realized_usd": None, "realized_pct": None}
            actions.append(f"{trade['ticker']}: position gone (external close)")
            return

        # ---- mark-based exit checks (priority: disaster -> sigma-target ->
        # time stop). All are enforced from the mark because a native
        # stop-sell on a long option is rejected by Alpaca as "uncovered".
        def _submit_close(reason: str, detail: str) -> bool:
            oid = (trade.get("tp") or {}).get("order_id")
            if oid:
                at.cancel_order(oid)
            try:
                o = at.close_position(symbol)
                trade["status"] = "closing"
                trade["exit"] = {"reason": reason, "order_id": o.get("id"),
                                 "submitted_at": ai_store.now_iso()}
                actions.append(f"{trade['ticker']}: {reason} ({detail}) -> closing")
                return True
            except at.AlpacaError as exc:
                actions.append(f"{trade['ticker']}: {reason} close failed: {exc}")
                return False

        # Disaster: % of premium OR $-cap (whichever triggers first).
        # Both ds_pct and ds_usd are treated as OFF when 0 / None / blank --
        # the UI form sends 0.0 when a field is empty, and a 0% stop would
        # close on the first negative tick (which closed CCJ at -$25 / -3.4%
        # on 2026-05-20 before this guard existed). Use any truthy value
        # to arm; 0 means "no stop on this axis". Puts use the puts_* keys.
        side_for_stop = _side_of(trade)
        en_key = "puts_disaster_stop_enabled" if side_for_stop == "put" else "disaster_stop_enabled"
        pct_key = "puts_disaster_stop_pct" if side_for_stop == "put" else "disaster_stop_pct"
        usd_key = "puts_disaster_stop_usd" if side_for_stop == "put" else "disaster_stop_usd"
        if pos is not None and clock_open and settings.get(en_key):
            ds_pct = _f(settings.get(pct_key))
            ds_usd = _f(settings.get(usd_key))
            plpc = _f(pos.get("unrealized_plpc"))
            plus = _f(pos.get("unrealized_pl"))   # absolute $ unrealized
            if plpc is None:
                cur = _f(pos.get("current_price"))
                ef = _f(entry.get("fill_price"))
                plpc = (cur / ef - 1.0) if (cur and ef) else None
            hit_pct = (bool(ds_pct) and plpc is not None
                       and plpc <= -ds_pct / 100.0)
            hit_usd = (bool(ds_usd) and plus is not None
                       and plus <= -ds_usd)
            if hit_pct or hit_usd:
                detail = (f"{plpc * 100:.0f}%" if hit_pct
                          else f"${plus:.0f}")
                if _submit_close("disaster_stop", detail):
                    return

        # Sigma-target exit: close when the underlying's 252d-LOG z reverts
        # back through the configured target. Direction depends on the
        # leg's side:
        #   CALL: close when z >= +sigma_target (oversold un-wound; default 0)
        #   PUT : close when z <= -sigma_target (overbought un-wound; default 0)
        # Calls use the standard sigma_target/time_stop_days settings; puts
        # use puts_sigma_target/puts_time_stop_days so the two legs can be
        # tuned independently. Skip silently if we don't have a current z
        # (data error / illiquid ticker) — time-stop will catch eventually.
        side = _side_of(trade)
        if clock_open:
            tgt_key = "puts_sigma_target" if side == "put" else "sigma_target"
            tgt = _f(settings.get(tgt_key))
            cur_z = z_by_ticker.get(trade.get("ticker", "").upper())
            if tgt is not None and cur_z is not None:
                if side == "put":
                    hit = cur_z <= -tgt
                    label = f"z={cur_z:+.2f}<=-tgt {-tgt:+.2f}"
                else:
                    hit = cur_z >= tgt
                    label = f"z={cur_z:+.2f}>=tgt {tgt:+.2f}"
                if hit and _submit_close("sigma_target", label):
                    return

        # Close-before-expiry: fires when remaining DTE drops to <= the
        # configured threshold. Ranked AFTER sigma_target (winners get the
        # cleaner exit reason if both hit the same day) and BEFORE time_stop
        # (assignment risk > "I've waited long enough"). Eliminates the ITM
        # auto-exercise problem (Alpaca assigns 100 shares = $10k+ on our
        # ~$5k account) AND fixes the "external" P&L attribution gap when
        # the option vanishes from the positions list at expiration.
        cbe_key = "puts_close_before_expiry_days" if side == "put" else "close_before_expiry_days"
        cbe = settings.get(cbe_key)
        contract = trade.get("contract") or {}
        exp_iso = contract.get("expiration")
        if cbe is not None and exp_iso and clock_open:
            td_to_exp = _trading_days_until(exp_iso)
            if td_to_exp is not None and td_to_exp <= int(cbe):
                if _submit_close("close_before_expiry", f"{td_to_exp}td to exp"):
                    return

        # Time stop (hard, regardless of P&L). Puts use the puts-side setting.
        held = _trading_days_since(entry.get("filled_at", ""))
        ts_key = "puts_time_stop_days" if side == "put" else "time_stop_days"
        ts_days = settings.get(ts_key)
        if ts_days is not None and held >= int(ts_days) and clock_open:
            if _submit_close("time_stop", f"{held}d"):
                return

        # Still open and healthy: keep the TP limit armed (no-op if TP=None).
        if trade.get("status") == "open":
            _arm_exits(trade, settings, actions)


# ---- entry ----------------------------------------------------------------

def _run_entries(settings: dict, account: dict, max_new: int,
                 actions: list, skips: list) -> int:
    placed = 0
    max_conc = int(settings["max_concurrent"])
    held_tickers = {t["ticker"] for t in ai_store.list_trades()
                    if t.get("status") in ("pending_entry", "open", "closing")}
    slots = max_conc - len(held_tickers)
    if slots <= 0:
        skips.append(f"no slots ({len(held_tickers)}/{max_conc} used)")
        return 0

    # Clamp this pass to the remaining per-day budget (run_cron tracks a
    # per-day counter and passes what is left). Stops a capitulation-day
    # cluster of correlated candidates from filling every free slot at once;
    # the book builds over multiple days instead.
    if slots > max_new:
        skips.append(f"per-day budget {max_new} left (< {slots} free slots)")
        slots = max_new
    if slots <= 0:
        skips.append("per-day entry budget exhausted")
        return 0

    opt_bp = _f(account.get("options_buying_power"), 0.0) or 0.0
    cap = float(settings["premium_cap_usd"])

    try:
        candidates = ai_strategy.scan_candidates(
            stoch_mode=str(settings.get("stoch_rsi_mode", "off")),
            stoch_oversold_max=float(settings.get("stoch_oversold_max", 30.0)),
            divergence_enabled=bool(settings.get("divergence_enabled", True)),
        )
    except Exception as exc:
        logger.exception("scan failed")
        skips.append(f"scan error: {exc!r}")
        return 0

    # Funnel counters so the run log shows WHY 0 placed even when no
    # per-candidate skip fires (e.g. scan returned all WEAK, or empty).
    n_scan = len(candidates)
    n_weak = 0
    n_held = 0
    n_eligible = 0
    for c in candidates:
        if slots <= 0:
            break
        tier = c.get("tier")
        if tier in ("WEAK", "?"):
            n_weak += 1
            continue  # playbook: skip weak / lone
        if c["ticker"] in held_tickers:
            n_held += 1
            continue
        n_eligible += 1
        contract = ai_strategy.pick_contract(
            c["ticker"], c["price"],
            otm_pct=settings["otm_pct"],
            dte_min=int(settings["dte_min"]),
            dte_max=int(settings["dte_max"]),
        )
        if not contract:
            skips.append(f"{c['ticker']}: no contract in DTE/strike band")
            continue
        q = at.latest_option_quote(contract["symbol"])
        ask = (q or {}).get("ask")
        if not ask:
            skips.append(f"{c['ticker']}: no option quote")
            continue
        premium = ask * OPTION_MULTIPLIER
        if premium > cap:
            skips.append(f"{c['ticker']}: premium ${premium:.0f} > cap ${cap:.0f}")
            continue
        limit_price = round(ask * (1 + settings["entry_limit_buffer_pct"] / 100.0), 2)
        cost = limit_price * OPTION_MULTIPLIER
        if cost > opt_bp * BUYING_POWER_SLACK:
            skips.append(f"{c['ticker']}: cost ${cost:.0f} > options BP ${opt_bp:.0f}")
            continue
        try:
            o = at.submit_order(contract["symbol"], 1, "buy", "limit", "day",
                                limit_price=limit_price)
        except at.AlpacaError as exc:
            skips.append(f"{c['ticker']}: order rejected: {exc}")
            continue
        trade = ai_store.new_trade(c["ticker"], {
            "z_log": c["z_log"], "tier": tier, "touch_date": c["touch_date"],
            "first_touch": c["first_touch"], "same_day_breadth": c["same_day_breadth"],
            "price_at_signal": c["price"], "reversion_to_mean_pct": c["reversion_to_mean_pct"],
            "source": c.get("source", "primary"),   # "primary" | "deep" (z<=-3.5)
        })
        ai_store.update_trade(trade["id"], {
            "status": "pending_entry",
            "contract": {**contract, "type": "call"},
            "entry": {"order_id": o.get("id"), "limit_price": limit_price,
                      "submitted_at": ai_store.now_iso(), "reprice_attempts": 0},
        })
        opt_bp -= cost
        slots -= 1
        placed += 1
        held_tickers.add(c["ticker"])
        actions.append(
            f"{c['ticker']} [{tier}]: BUY 1x {contract['symbol']} "
            f"limit {limit_price} (premium ${premium:.0f})"
        )
    bars_n = getattr(ai_strategy.scan_candidates, "last_bars", None)
    uni_n = getattr(ai_strategy.scan_candidates, "last_universe", None)
    skips.append(
        f"calls: bars={bars_n}/{uni_n}, scan={n_scan}, "
        f"weak/?={n_weak}, held={n_held}, eligible={n_eligible}"
    )
    return placed


def _run_put_entries(settings: dict, account: dict, max_new: int,
                     actions: list, skips: list) -> int:
    """Puts-sleeve entry phase. Same shape as _run_entries but with the
    puts settings, the puts scanner, and the puts contract picker.
    Counts only its own held tickers (calls and puts share Alpaca BP but
    not concurrency slots), and tags every new trade with side='put' so
    _reconcile flips the sigma-target direction."""
    placed = 0
    max_conc = int(settings["puts_max_concurrent"])
    open_puts = [t for t in ai_store.list_trades()
                 if t.get("status") in ("pending_entry", "open", "closing")
                 and _side_of(t) == "put"]
    held_tickers = {t["ticker"] for t in open_puts}
    slots = max_conc - len(held_tickers)
    if slots <= 0:
        skips.append(f"puts: no slots ({len(held_tickers)}/{max_conc} used)")
        return 0
    if slots > max_new:
        skips.append(f"puts: per-day budget {max_new} left (< {slots} free slots)")
        slots = max_new
    if slots <= 0:
        skips.append("puts: per-day entry budget exhausted")
        return 0

    opt_bp = _f(account.get("options_buying_power"), 0.0) or 0.0
    cap = float(settings["puts_premium_cap_usd"])
    buf = float(settings["puts_entry_limit_buffer_pct"])

    try:
        candidates = ai_strategy.scan_put_candidates()
    except Exception as exc:
        logger.exception("puts scan failed")
        skips.append(f"puts scan error: {exc!r}")
        return 0

    if not candidates:
        bars_n = getattr(ai_strategy.scan_put_candidates, "last_bars", None)
        uni_n = getattr(ai_strategy.scan_put_candidates, "last_universe", None)
        skips.append(
            f"puts: bars={bars_n}/{uni_n}, no candidates pass all 5 features")
        return 0

    for c in candidates:
        if slots <= 0:
            break
        if c["ticker"] in held_tickers:
            continue
        contract = ai_strategy.pick_put_contract(
            c["ticker"], c["price"],
            otm_pct=settings["puts_otm_pct"],
            dte_min=int(settings["puts_dte_min"]),
            dte_max=int(settings["puts_dte_max"]),
        )
        if not contract:
            skips.append(f"puts {c['ticker']}: no contract in DTE/strike band")
            continue
        q = at.latest_option_quote(contract["symbol"])
        ask = (q or {}).get("ask")
        if not ask:
            skips.append(f"puts {c['ticker']}: no option quote")
            continue
        premium = ask * OPTION_MULTIPLIER
        if premium > cap:
            skips.append(f"puts {c['ticker']}: premium ${premium:.0f} > cap ${cap:.0f}")
            continue
        limit_price = round(ask * (1 + buf / 100.0), 2)
        cost = limit_price * OPTION_MULTIPLIER
        if cost > opt_bp * BUYING_POWER_SLACK:
            skips.append(f"puts {c['ticker']}: cost ${cost:.0f} > options BP ${opt_bp:.0f}")
            continue
        try:
            o = at.submit_order(contract["symbol"], 1, "buy", "limit", "day",
                                limit_price=limit_price)
        except at.AlpacaError as exc:
            skips.append(f"puts {c['ticker']}: order rejected: {exc}")
            continue
        trade = ai_store.new_trade(c["ticker"], {
            "z_log": c["z_log"], "tier": c["tier"],
            "price_at_signal": c["price"],
            "reversion_to_mean_pct": c["reversion_to_mean_pct"],
            "source": c.get("source", "bounce_put"),
            "bounce": c.get("bounce"),
            "side": "put",
        })
        ai_store.update_trade(trade["id"], {
            "status": "pending_entry",
            "contract": {**contract, "type": "put"},
            "entry": {"order_id": o.get("id"), "limit_price": limit_price,
                      "submitted_at": ai_store.now_iso(), "reprice_attempts": 0},
        })
        opt_bp -= cost
        slots -= 1
        placed += 1
        held_tickers.add(c["ticker"])
        actions.append(
            f"PUT {c['ticker']} [{c['tier']}]: BUY 1x {contract['symbol']} "
            f"limit {limit_price} (premium ${premium:.0f})"
        )
    return placed


# ---- entrypoint -----------------------------------------------------------

def run_cron(manual: bool = False) -> dict:
    rec: dict = {"phase": "manual" if manual else "cron", "actions": [],
                 "skips": [], "errors": [], "ran": False}

    if not at.is_configured():
        rec["reason"] = "alpaca credentials not configured"
        ai_store.log_run(rec)
        return {"ok": False, **rec}

    settings = ai_store.get_settings()

    try:
        clock = at.get_clock()
        account = at.get_account()
    except at.AlpacaError as exc:
        rec["errors"].append(f"alpaca unreachable: {exc}")
        ai_store.log_run(rec)
        return {"ok": False, **rec}

    is_open = bool(clock.get("is_open"))
    rec["market_open"] = is_open
    rec["ran"] = True

    # Bookkeeping always runs so in-flight trades stay managed even if the
    # kill-switch was turned off after entry. We batch-fetch the underlying
    # z for every open trade once per tick (rather than per-trade) so the
    # sigma-target exit check is cheap. Both legs share this fetch.
    trades = ai_store.list_trades()
    open_tickers = sorted({(t.get("ticker") or "").upper()
                           for t in trades
                           if t.get("status") in ("pending_entry", "open", "closing")
                           and t.get("ticker")})
    z_by_ticker: dict[str, float] = {}
    # Fetch if EITHER leg's sigma-target is configured -- both reuse the
    # same z dict, just with different threshold semantics in _reconcile.
    need_z = (settings.get("sigma_target") is not None
              or settings.get("puts_sigma_target") is not None)
    if open_tickers and need_z:
        try:
            z_by_ticker = ai_strategy.current_z_for_tickers(open_tickers)
        except Exception as exc:
            logger.exception("sigma-target z-fetch failed")
            rec["errors"].append(f"sigma-target z-fetch failed: {exc!r}")

    dirty = False
    for t in trades:
        if t.get("status") in ("pending_entry", "open", "closing"):
            dirty = True
            try:
                _reconcile(t, settings, is_open, rec["actions"],
                           z_by_ticker=z_by_ticker)
            except Exception as exc:
                logger.exception("reconcile failed for %s", t.get("id"))
                rec["errors"].append(f"{t.get('ticker')}: reconcile {exc!r}")
    if dirty:
        ai_store._save_trades(trades)

    # Entry phase: market open + kill-switch on. Re-evaluated every run so a
    # slot freed intraday (take-profit/stop close) or a raised max_concurrent
    # is picked up the same day; a per-day COUNTER (not a calendar stamp)
    # preserves the capitulation-clustering throttle. Each leg has its own
    # counter (entries_today / puts_entries_today) so they can't steal each
    # other's daily budget.
    et_today = _et_date(clock)
    state = ai_store.get_state()
    placed_today = (int(state.get("entries_today", 0))
                    if state.get("entry_day") == et_today else 0)
    puts_placed_today = (int(state.get("puts_entries_today", 0))
                         if state.get("entry_day") == et_today else 0)
    max_per_day = int(settings["max_entries_per_day"])
    puts_max_per_day = int(settings["puts_max_entries_per_day"])

    # CALL leg
    if not settings.get("enabled"):
        rec["skips"].append("kill-switch off - call entries skipped")
    elif not is_open:
        rec["skips"].append("market closed - call entries skipped")
    elif placed_today >= max_per_day:
        rec["skips"].append(
            f"call per-day entry cap reached ({placed_today}/{max_per_day} for {et_today})")
    else:
        placed = _run_entries(settings, account, max_per_day - placed_today,
                              rec["actions"], rec["skips"])
        rec["entries_placed"] = placed
        placed_today += placed

    # PUT leg (separate kill switch -- starts disabled by default)
    if not settings.get("puts_enabled"):
        rec["skips"].append("puts kill-switch off - put entries skipped")
    elif not is_open:
        rec["skips"].append("market closed - put entries skipped")
    elif puts_placed_today >= puts_max_per_day:
        rec["skips"].append(
            f"puts per-day entry cap reached "
            f"({puts_placed_today}/{puts_max_per_day} for {et_today})")
    else:
        placed = _run_put_entries(settings, account,
                                  puts_max_per_day - puts_placed_today,
                                  rec["actions"], rec["skips"])
        rec["puts_entries_placed"] = placed
        puts_placed_today += placed

    ai_store.set_state({"entry_day": et_today,
                        "entries_today": placed_today,
                        "puts_entries_today": puts_placed_today})

    # Equity snapshot + realized cumulative.
    realized_cum = round(sum(
        (_f((t.get("pnl") or {}).get("realized_usd"), 0.0) or 0.0)
        for t in ai_store.list_trades() if t.get("status") == "closed"
    ), 2)
    ai_store.append_equity(
        _f(account.get("equity"), 0.0) or 0.0,
        _f(account.get("cash"), 0.0) or 0.0,
        realized_cum,
    )
    rec["equity"] = _f(account.get("equity"))

    ai_store.log_run(rec)
    return {"ok": True, **rec}


def snapshot() -> dict:
    """Full read-only state for the AI page. Degrades gracefully when
    Alpaca creds are missing (account=None, no live marks)."""
    settings = ai_store.get_settings()
    trades = ai_store.list_trades()
    account = None
    marks: dict[str, dict] = {}
    configured = at.is_configured()
    if configured:
        try:
            account = at.get_account()
        except at.AlpacaError:
            account = None
        try:
            for p in at.list_option_positions():
                marks[p.get("symbol")] = p
        except at.AlpacaError:
            pass

    open_t, closed_t = [], []
    for t in trades:
        if t.get("status") == "closed" or t.get("status") == "canceled":
            closed_t.append(t)
        else:
            tt = dict(t)
            sym = (tt.get("contract") or {}).get("symbol")
            if sym and sym in marks:
                m = marks[sym]
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
    total_real = round(sum(t["pnl"]["realized_usd"] for t in closed_pnl), 2)
    stats = {
        "closed_count": len(closed_pnl),
        "win_rate": round(100.0 * len(wins) / len(closed_pnl), 1) if closed_pnl else None,
        "avg_win": round(sum(t["pnl"]["realized_usd"] for t in wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(t["pnl"]["realized_usd"] for t in losses) / len(losses), 2) if losses else None,
        "total_realized": total_real,
    }

    return {
        "configured": configured,
        "settings": settings,
        "account": account,
        "open_trades": list(reversed(open_t)),
        "closed_trades": list(reversed(closed_t))[:100],
        "stats": stats,
        "equity": ai_store.equity_series(),
        "run_log": ai_store.run_log(100),
    }


__all__ = ["run_cron", "snapshot"]
