"""Strategy backtester for options-style trading (asymmetric risk frame).

At each bar i, fit a 252-day regression on bars [i-251, i] (no lookahead) and
derive that bar's sd_position. Entries fire on configured σ crosses (typically
extreme: ≥±3σ).

Exit triggers (both in OPTION P&L terms — symmetric):
- Hard stop loss: OPTION P&L drops -stop_loss_pct (default 10%)
- Hard profit target: OPTION P&L reaches +profit_target_pct (default 20%)
  (Both can fire from small stock moves amplified by leverage)
- Optional trailing stop (both trail_* params must be set; on UNDERLYING %)
- Optional σ exits (tp_sigma / sl_sigma) for mean-reversion safety nets
- Time stop: max time_stop_bars in the trade

CAVEAT: stop is evaluated at the bar's CLOSE. Gap days can overshoot the stop
significantly. Intraday stop modeling (using bar high/low) is a separate fix.

Equity uses linear leverage approximation, capped at -100% per trade:
    option_return = max(underlying_return * leverage, -1.0)
Real options have delta/gamma so this overstates winners somewhat.
"""
from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend import ichimoku
from backend.channels import REGRESSION_WINDOW
from backend.indicators import macd, stoch_rsi


@dataclass
class BacktestConfig:
    # Sides
    long_enabled: bool = True
    short_enabled: bool = True
    # Entry: extreme σ levels by default
    long_entry_sigma: float = -3.0
    short_entry_sigma: float = 3.0
    min_confidence: int = 0
    # Option leverage (used for equity + profit-target trigger)
    leverage: float = 5.0
    # Primary exits: both in OPTION P&L %
    stop_loss_pct: float = 10.0                  # exit when option P&L drops this %
    profit_target_pct: float | None = 20.0       # exit when option P&L gains this %
    # Optional trail (None = disabled). Operates on UNDERLYING %.
    trail_activation_pct: float | None = None
    trail_distance_pct: float | None = None
    # Optional σ-based exits
    tp_sigma: float | None = None
    sl_sigma: float | None = None
    # Time
    time_stop_bars: int = 14
    # Capital + sizing
    starting_capital: float = 10000.0
    allocation_pct: float = 10.0       # % of current equity per trade
    window: int = REGRESSION_WINDOW
    # Trend filter: when True, only take longs if regression slope > 0,
    # only take shorts if slope < 0 (regime alignment).
    require_trend_alignment: bool = False
    # Minimum |annualized slope %| to consider a trend "real". 0 = sign only.
    # Expressed as annual % move (slope*252/price*100).
    min_trend_pct: float = 0.0
    # Independent slope-sign filter (orthogonal to require_trend_alignment):
    # "any" = no filter, "up" = require slope > 0, "down" = require slope < 0.
    trend_direction: str = "any"
    # Require stoch RSI K to be in the extreme matching the side:
    # longs need K <= stoch_oversold; shorts need K >= stoch_overbought.
    require_stoch_extreme: bool = False
    stoch_oversold: float = 25.0
    stoch_overbought: float = 75.0


def compute_rolling_channels(df: pd.DataFrame, window: int = REGRESSION_WINDOW) -> pd.DataFrame:
    n = len(df)
    closes = df["close"].to_numpy(dtype=float)
    reg_line = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    sd_pos = np.full(n, np.nan)
    slope_arr = np.full(n, np.nan)
    trend_pct = np.full(n, np.nan)  # annualized slope as % of current price

    x = np.arange(window, dtype=float)
    for i in range(window - 1, n):
        y = closes[i - window + 1 : i + 1]
        slope, intercept = np.polyfit(x, y, 1)
        fit = slope * x + intercept
        residuals = y - fit
        s = float(residuals.std(ddof=1))
        if s > 0:
            reg_line[i] = float(fit[-1])
            sigma[i] = s
            sd_pos[i] = (closes[i] - fit[-1]) / s
            slope_arr[i] = float(slope)
            if closes[i] > 0:
                trend_pct[i] = float(slope) * 252.0 / float(closes[i]) * 100.0

    out = df.copy()
    out["regression_line"] = reg_line
    out["sigma"] = sigma
    out["sd_position"] = sd_pos
    out["slope"] = slope_arr
    out["trend_pct"] = trend_pct
    return out


def prepare(df: pd.DataFrame, window: int = REGRESSION_WINDOW) -> pd.DataFrame:
    df = compute_rolling_channels(df, window=window)
    df["stoch_rsi_k"], df["stoch_rsi_d"] = stoch_rsi(df["close"])
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
    ichi = ichimoku.compute(df)
    df["tenkan"] = ichi.tenkan
    df["kijun"] = ichi.kijun
    df["senkou_a"] = ichi.senkou_a
    df["senkou_b"] = ichi.senkou_b
    df["chikou"] = ichi.chikou
    return df


def simulate(df: pd.DataFrame, config: BacktestConfig) -> dict:
    """Public entry: prepares the df and runs the simulation."""
    df_prepared = prepare(df, window=config.window)
    return _simulate_prepared(df_prepared, config)


def _simulate_prepared(df: pd.DataFrame, config: BacktestConfig) -> dict:
    """Run simulation on an already-prepared df. Used by sweep for speed."""
    trades: list[dict] = []
    equity_curve: list[dict] = []
    cash = config.starting_capital
    position: dict | None = None

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        sd_prev = prev["sd_position"]
        sd_cur = cur["sd_position"]
        cur_time = _datestr(cur["timestamp"])
        cur_close = float(cur["close"])
        cur_high = float(cur["high"])
        cur_low = float(cur["low"])

        if pd.isna(sd_prev) or pd.isna(sd_cur):
            equity_curve.append({"time": cur_time, "value": cash + _position_value(position, cur_close, config)})
            continue

        # ENTRY (no exit check on entry bar — fill is at close)
        if position is None:
            entry = None
            if (config.long_enabled
                and sd_prev >= config.long_entry_sigma
                and sd_cur < config.long_entry_sigma):
                entry = "long"
            elif (config.short_enabled
                  and sd_prev <= config.short_entry_sigma
                  and sd_cur > config.short_entry_sigma):
                entry = "short"

            if entry is not None and config.require_trend_alignment:
                trend = cur.get("trend_pct")
                if pd.isna(trend):
                    entry = None
                elif entry == "long" and trend <= config.min_trend_pct:
                    entry = None
                elif entry == "short" and trend >= -config.min_trend_pct:
                    entry = None

            if entry is not None and config.trend_direction in ("up", "down"):
                slope = cur.get("slope")
                if pd.isna(slope):
                    entry = None
                elif config.trend_direction == "up" and slope <= 0:
                    entry = None
                elif config.trend_direction == "down" and slope >= 0:
                    entry = None

            if entry is not None and config.require_stoch_extreme:
                k = cur.get("stoch_rsi_k")
                if pd.isna(k):
                    entry = None
                elif entry == "long" and k > config.stoch_oversold:
                    entry = None
                elif entry == "short" and k < config.stoch_overbought:
                    entry = None

            if entry is not None:
                conf = _confidence(entry, cur, df.iloc[: i + 1])
                if conf >= config.min_confidence:
                    equity_at_entry = cash
                    position_size = equity_at_entry * (config.allocation_pct / 100.0)
                    cash -= position_size
                    position = {
                        "direction": entry,
                        "entry_idx": i,
                        "entry_price": cur_close,
                        "entry_time": cur_time,
                        "position_size": position_size,
                        "equity_at_entry": equity_at_entry,
                        "confidence": conf,
                        "peak_underlying_pnl_pct": 0.0,
                        "peak_option_pnl_pct": 0.0,
                    }
            equity_curve.append({"time": cur_time, "value": cash + _position_value(position, cur_close, config)})
            continue

        # EXIT — intraday first (touch detection), then close-based
        exit_price, reason = _intraday_exit_price(position, cur_high, cur_low, config)

        if exit_price is None:
            # No intraday touch. Update peaks and check close-based exits.
            close_underlying_ret = _underlying_return(position, cur_close)
            close_underlying_pnl_pct = close_underlying_ret * 100.0
            close_option_ret = _option_return(close_underlying_ret, config.leverage)
            close_option_pnl_pct = close_option_ret * 100.0

            if close_underlying_pnl_pct > position["peak_underlying_pnl_pct"]:
                position["peak_underlying_pnl_pct"] = close_underlying_pnl_pct
            if close_option_pnl_pct > position["peak_option_pnl_pct"]:
                position["peak_option_pnl_pct"] = close_option_pnl_pct

            should_exit, reason = _check_close_exit(position, close_underlying_pnl_pct, sd_cur, i, config)
            if should_exit:
                exit_price = cur_close
            else:
                # MTM and continue
                position_value = position["position_size"] * (1.0 + close_option_ret)
                equity_curve.append({"time": cur_time, "value": cash + position_value})
                continue

        # We have an exit_price and reason — close the trade.
        underlying_ret = _underlying_return(position, exit_price)
        underlying_pnl_pct = underlying_ret * 100.0
        option_ret = _option_return(underlying_ret, config.leverage)
        option_pnl_pct = option_ret * 100.0
        position_value = position["position_size"] * (1.0 + option_ret)
        trade_pnl = position_value - position["position_size"]
        cash += position_value
        trades.append({
            "direction": position["direction"],
            "entry_time": position["entry_time"],
            "exit_time": cur_time,
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "underlying_return_pct": underlying_pnl_pct,
            "option_return_pct": option_pnl_pct,
            "return_pct": option_pnl_pct,                    # legacy display
            "peak_underlying_pnl_pct": position["peak_underlying_pnl_pct"],
            "peak_option_pnl_pct": position["peak_option_pnl_pct"],
            "position_size": position["position_size"],
            "trade_pnl_dollars": trade_pnl,
            "account_impact_pct": (trade_pnl / position["equity_at_entry"]) * 100.0,
            "exit_reason": reason,
            "confidence": position["confidence"],
            "bars_held": i - position["entry_idx"],
        })
        position = None
        equity_curve.append({"time": cur_time, "value": cash})

    stats = _compute_stats(trades, equity_curve, df, config.starting_capital)
    return {"trades": trades, "equity_curve": equity_curve, "stats": stats}


DEFAULT_SWEEP_GRID: dict[str, list] = {
    "long_entry_sigma": [-1.5, -2.0, -2.5, -3.0],
    "min_confidence": [0, 1, 2],
    "stop_loss_pct": [7.0, 10.0, 15.0],
    "profit_target_pct": [15.0, 20.0, 30.0, 50.0],
    "time_stop_bars": [10, 14, 20],
}


def sweep(
    df: pd.DataFrame,
    base_config: BacktestConfig,
    grid: dict[str, list] | None = None,
    mirror_sigma: bool = True,
) -> list[dict]:
    """Run simulate() for every combination in grid; return all results.

    Caller is responsible for sorting/filtering. grid keys must be valid
    BacktestConfig field names. If mirror_sigma is True and the grid varies
    long_entry_sigma, short_entry_sigma is automatically set to -long_entry_sigma
    so the strategy is symmetric across sides.
    """
    grid = grid if grid is not None else DEFAULT_SWEEP_GRID
    df_prepared = prepare(df, window=base_config.window)

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    results: list[dict] = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        cfg = dataclasses.replace(base_config, **params)
        if mirror_sigma and "long_entry_sigma" in params:
            cfg = dataclasses.replace(cfg, short_entry_sigma=-params["long_entry_sigma"])
        sim = _simulate_prepared(df_prepared, cfg)
        results.append({"params": params, "stats": sim["stats"]})
    return results


def _position_value(position: dict | None, cur_price: float, config: BacktestConfig) -> float:
    if position is None:
        return 0.0
    underlying_ret = _underlying_return(position, cur_price)
    option_ret = _option_return(underlying_ret, config.leverage)
    return position["position_size"] * (1.0 + option_ret)


def _underlying_return(position: dict, cur_price: float) -> float:
    r = (cur_price - position["entry_price"]) / position["entry_price"]
    return r if position["direction"] == "long" else -r


def _option_return(underlying_ret: float, leverage: float) -> float:
    # Long options: max loss = 100% of premium.
    return max(underlying_ret * leverage, -1.0)


def _intraday_exit_price(position: dict, cur_high: float, cur_low: float, config: BacktestConfig) -> tuple[float | None, str | None]:
    """Detect intraday touch of stop or target thresholds. Returns the exit
    price (cap loss / lock gain at the threshold) and reason, or (None, None).

    Convert option-pct thresholds to underlying-pct via leverage.
    If both stop and target are touched on the same bar, assume stop fires
    first (conservative — without tick data we can't know the order).
    """
    entry = position["entry_price"]
    lev = max(config.leverage, 0.001)
    stop_under_frac = abs(config.stop_loss_pct) / lev / 100.0
    target_under_frac = (config.profit_target_pct / lev / 100.0) if config.profit_target_pct is not None else None

    if position["direction"] == "long":
        stop_price = entry * (1.0 - stop_under_frac)
        target_price = entry * (1.0 + target_under_frac) if target_under_frac is not None else None
        if cur_low <= stop_price:
            return stop_price, "stop"
        if target_price is not None and cur_high >= target_price:
            return target_price, "target"
    else:  # short
        stop_price = entry * (1.0 + stop_under_frac)
        target_price = entry * (1.0 - target_under_frac) if target_under_frac is not None else None
        if cur_high >= stop_price:
            return stop_price, "stop"
        if target_price is not None and cur_low <= target_price:
            return target_price, "target"
    return None, None


def _check_close_exit(position: dict, close_underlying_pnl_pct: float, sd_cur: float, i: int, config: BacktestConfig) -> tuple[bool, str | None]:
    """Close-based exits: trail, σ-based, time stop."""
    # Trailing stop on UNDERLYING (both params must be set)
    if (config.trail_activation_pct is not None
        and config.trail_distance_pct is not None
        and position["peak_underlying_pnl_pct"] >= config.trail_activation_pct):
        drawdown_from_peak = position["peak_underlying_pnl_pct"] - close_underlying_pnl_pct
        if drawdown_from_peak >= config.trail_distance_pct:
            return True, "trail"
    # σ-based exits (optional)
    if config.tp_sigma is not None:
        if position["direction"] == "long" and sd_cur >= config.tp_sigma:
            return True, "tp_sigma"
        if position["direction"] == "short" and sd_cur <= -config.tp_sigma:
            return True, "tp_sigma"
    if config.sl_sigma is not None:
        mag = abs(config.sl_sigma)
        if position["direction"] == "long" and sd_cur <= -mag:
            return True, "sl_sigma"
        if position["direction"] == "short" and sd_cur >= mag:
            return True, "sl_sigma"
    # Time stop
    if (i - position["entry_idx"]) >= config.time_stop_bars:
        return True, "time"
    return False, None


def _confidence(direction: str, cur: pd.Series, df_so_far: pd.DataFrame) -> int:
    score = 0
    hist = cur.get("macd_hist")
    prev_hist = df_so_far["macd_hist"].iloc[-2] if "macd_hist" in df_so_far.columns and len(df_so_far) >= 2 else None
    if pd.notna(hist) and prev_hist is not None and pd.notna(prev_hist):
        if direction == "long" and hist > prev_hist:
            score += 1
        elif direction == "short" and hist < prev_hist:
            score += 1
    k = cur.get("stoch_rsi_k")
    d = cur.get("stoch_rsi_d")
    if pd.notna(k) and pd.notna(d):
        if direction == "long" and (k < 25 or (k > d and k < 50)):
            score += 1
        elif direction == "short" and (k > 75 or (k < d and k > 50)):
            score += 1
    if "senkou_a" in df_so_far.columns and len(df_so_far) > 26:
        a = df_so_far["senkou_a"].iloc[-1 - 26]
        b = df_so_far["senkou_b"].iloc[-1 - 26]
        if pd.notna(a) and pd.notna(b):
            pos = ichimoku.cloud_position(cur["close"], a, b)
            if direction == "long" and pos == "above":
                score += 1
            elif direction == "short" and pos == "below":
                score += 1
    return score


def _compute_stats(trades: list[dict], equity_curve: list[dict], df: pd.DataFrame, starting_capital: float) -> dict:
    if not equity_curve:
        return _empty_stats()

    final_equity = equity_curve[-1]["value"]
    total_return = (final_equity / starting_capital) - 1.0

    peak = starting_capital
    max_dd = 0.0
    for pt in equity_curve:
        if pt["value"] > peak:
            peak = pt["value"]
        dd = (pt["value"] - peak) / peak
        if dd < max_dd:
            max_dd = dd

    bh_start_idx = next((i for i, r in df.iterrows() if _datestr(r["timestamp"]) == equity_curve[0]["time"]), None)
    bh_end_idx = next((i for i, r in df.iterrows() if _datestr(r["timestamp"]) == equity_curve[-1]["time"]), None)
    if bh_start_idx is not None and bh_end_idx is not None:
        bh_return = (df["close"].iloc[bh_end_idx] / df["close"].iloc[bh_start_idx]) - 1.0
    else:
        bh_return = float("nan")

    values = np.array([pt["value"] for pt in equity_curve], dtype=float)
    if len(values) > 1:
        daily_ret = np.diff(values) / values[:-1]
        sharpe = float(np.sqrt(252) * daily_ret.mean() / daily_ret.std()) if daily_ret.std() > 0 else 0.0
    else:
        sharpe = 0.0

    wins = [t for t in trades if t["option_return_pct"] > 0]
    losses = [t for t in trades if t["option_return_pct"] <= 0]
    return {
        "total_return_pct": total_return * 100.0,
        "buy_and_hold_pct": bh_return * 100.0 if bh_return == bh_return else None,
        "trade_count": len(trades),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate_pct": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "avg_win_pct": float(np.mean([t["option_return_pct"] for t in wins])) if wins else 0.0,
        "avg_loss_pct": float(np.mean([t["option_return_pct"] for t in losses])) if losses else 0.0,
        "profit_factor": _profit_factor(trades),
        "max_drawdown_pct": max_dd * 100.0,
        "sharpe": sharpe,
        "final_equity": final_equity,
        "starting_capital": starting_capital,
        "exit_reason_counts": _exit_reason_counts(trades),
    }


def _exit_reason_counts(trades: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in trades:
        counts[t["exit_reason"]] = counts.get(t["exit_reason"], 0) + 1
    return counts


def _profit_factor(trades: list[dict]) -> float:
    gains = sum(t["option_return_pct"] for t in trades if t["option_return_pct"] > 0)
    losses = -sum(t["option_return_pct"] for t in trades if t["option_return_pct"] < 0)
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _empty_stats() -> dict:
    return {
        "total_return_pct": 0.0,
        "buy_and_hold_pct": None,
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "win_rate_pct": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "profit_factor": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe": 0.0,
        "final_equity": 0.0,
        "starting_capital": 0.0,
        "exit_reason_counts": {},
    }


def _datestr(ts) -> str:
    if hasattr(ts, "date"):
        return str(ts.date())
    return str(ts)[:10]
