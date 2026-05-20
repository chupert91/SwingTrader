"""FastAPI app: serves chart data API + static frontend."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import backtest, ichimoku, indicator_registry, kv, scheduler, state, watchlist
from backend.alert_engine import AlertRule
from backend.channels import REGRESSION_WINDOW, SIGMA_LEVELS, compute_channels
from backend.data import fetch_bars, fetch_ticker_name
from backend.volume import split_volume
from backend.volume_profile import compute_profile
import backend.indicators_lib  # noqa: F401 — side-effect: registers indicators

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

ICHIMOKU_SHIFT = 26

VOL_BUY_COLOR = "#26a69a"
VOL_SELL_COLOR = "#ef5350"

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# The in-process scheduler (scheduler.run_forever) only makes sense in the
# long-running local dev server. On Vercel each request is a new container,
# so we skip the lifespan there and rely on the Vercel Cron entry that hits
# /api/scan instead.
_ON_VERCEL = bool(os.environ.get("VERCEL"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(scheduler.run_forever())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Trisigma",
    lifespan=None if _ON_VERCEL else lifespan,
)


FETCH_PERIOD = "14mo"  # > 252 trading days so compute_channels has a full fit window


def _prepare(ticker: str, period: str) -> pd.DataFrame:
    df = fetch_bars(ticker, period=period)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for ticker '{ticker}'")
    return compute_channels(df, window=REGRESSION_WINDOW)


@app.get("/api/chart/{ticker}")
def get_chart(ticker: str, period: str = FETCH_PERIOD):
    full = _prepare(ticker, period)
    ichi = ichimoku.compute(full)

    crop_start = max(0, len(full) - REGRESSION_WINDOW)
    df = full.iloc[crop_start:].reset_index(drop=True)
    ichi_cropped = ichimoku.IchimokuComponents(
        tenkan=ichi.tenkan.iloc[crop_start:].reset_index(drop=True),
        kijun=ichi.kijun.iloc[crop_start:].reset_index(drop=True),
        senkou_a=ichi.senkou_a.iloc[crop_start:].reset_index(drop=True),
        senkou_b=ichi.senkou_b.iloc[crop_start:].reset_index(drop=True),
        chikou=ichi.chikou.iloc[crop_start:].reset_index(drop=True),
    )
    payload = _serialize(df, ticker, ichi_cropped)
    payload["custom_indicators"] = indicator_registry.compute_active(
        df, kv.get_active_indicators()
    )
    return JSONResponse(payload)


@app.get("/api/indicators")
def list_indicators() -> dict:
    """Catalog of every indicator the user can enable from the picker."""
    return {"indicators": [i.to_summary_dict() for i in indicator_registry.list_all()]}


class ActiveIndicatorsPayload(BaseModel):
    active: list[dict]


@app.get("/api/indicators/active")
def get_active_indicators() -> dict:
    return {"active": kv.get_active_indicators()}


@app.put("/api/indicators/active")
def put_active_indicators(payload: ActiveIndicatorsPayload) -> dict:
    kv.set_active_indicators(payload.active)
    return {"ok": True, "active": payload.active}


class WatchlistPayload(BaseModel):
    tickers: list[str]


@app.get("/api/watchlist")
def get_watchlist() -> dict:
    return {"tickers": watchlist.get()}


@app.put("/api/watchlist")
def put_watchlist(payload: WatchlistPayload) -> dict:
    saved = watchlist.set_(payload.tickers)
    return {"tickers": saved}


@app.get("/api/alerts")
def get_alerts(limit: int = 25) -> dict:
    # KV-backed history — same store the cron writes to. Frontend expects
    # the key "alerts" so we re-key here from kv's "history" shape.
    return {"alerts": kv.history(limit=limit)}


@app.post("/api/alerts/scan")
def post_manual_scan() -> dict:
    """Browser-triggered scan from the chart sidebar's ⟳ button.

    Same KV-backed pipeline the cron uses; no auth (this is the UI path)."""
    from backend.scan import run_scan
    return run_scan()


# ---- Rule-based alert system (new) ----------------------------------------
# Local mirrors of the Vercel serverless endpoints. The FastAPI handlers
# call directly into backend.kv / backend.alert_engine so the same code
# powers both deployment modes.

@app.get("/api/rules")
def get_rules() -> dict:
    return {"rules": kv.list_rules()}


class RulePayload(BaseModel):
    id: str = ""
    name: str = "Unnamed rule"
    tickers: list[str] = []
    side: str = "both"
    entry_sigma: float = -1.0
    require_trend: bool = False
    min_trend_pct: float = 0.0
    exit_target_pct: float = 20.0
    exit_stop_pct: float = 10.0
    leverage: float = 5.0
    enabled: bool = True
    notify_email: str = ""
    trend_direction: str = "any"
    require_stoch_extreme: bool = True
    stoch_oversold: float = 35.0
    stoch_overbought: float = 65.0
    min_avg_volume_m: float = 50.0


@app.post("/api/rules")
def post_rule(payload: RulePayload) -> dict:
    rule = AlertRule.from_dict(payload.model_dump())
    existing = kv.list_rules()
    updated = [r for r in existing if r.get("id") != rule.id]
    updated.append(rule.to_dict())
    kv.save_rules(updated)
    return {"ok": True, "rule": rule.to_dict()}


@app.delete("/api/rules")
def delete_rule(id: str) -> dict:
    existing = kv.list_rules()
    updated = [r for r in existing if r.get("id") != id]
    kv.save_rules(updated)
    return {"ok": True, "removed": id}


class DrawingsPayload(BaseModel):
    hlines: list[dict] = []
    trendlines: list[dict] = []
    fibs: list[dict] = []
    trades: list[dict] = []


@app.get("/api/drawings/{ticker}")
def get_drawings(ticker: str) -> dict:
    return kv.get_ticker_drawings(ticker)


@app.put("/api/drawings/{ticker}")
def put_drawings(ticker: str, payload: DrawingsPayload) -> dict:
    kv.set_ticker_drawings(ticker, payload.model_dump())
    return {"ok": True}


@app.get("/api/history")
def get_history(limit: int = 50) -> dict:
    limit = max(1, min(200, int(limit)))
    return {"history": kv.history(limit=limit)}


@app.api_route("/api/discover", methods=["GET", "POST"])
def run_discover_endpoint(request: Request) -> dict:
    """Cron-driven S&P 500 discovery scan. CRON_SECRET-gated when set."""
    expected = os.environ.get("CRON_SECRET")
    if expected:
        auth = request.headers.get("authorization") or request.headers.get("Authorization", "")
        if auth != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="unauthorized")
    from backend.discover import run_discovery
    return run_discovery()


@app.get("/api/discoveries")
def get_discoveries() -> dict:
    from backend.discover import get_latest
    return get_latest()


@app.api_route("/api/scan", methods=["GET", "POST"])
def run_rule_scan(request: Request) -> dict:
    """Rule-based scan endpoint. Hit by Vercel Cron (GET with Bearer auth)
    and by manual curl/local dev (POST without auth).

    If CRON_SECRET env var is set, the request must carry
    `Authorization: Bearer <CRON_SECRET>` (Vercel Cron does this automatically).
    """
    expected = os.environ.get("CRON_SECRET")
    if expected:
        auth = request.headers.get("authorization") or request.headers.get("Authorization", "")
        if auth != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="unauthorized")
    from backend.scan import run_scan
    return run_scan()


# ---- Autonomous AI options trader ----------------------------------------

class AISettingsPayload(BaseModel):
    enabled: bool | None = None
    premium_cap_usd: float | None = None
    max_concurrent: int | None = None
    max_entries_per_day: int | None = None
    disaster_stop_enabled: bool | None = None
    disaster_stop_pct: float | None = None
    disaster_stop_usd: float | None = None
    take_profit_pct: float | None = None
    sigma_target: float | None = None
    time_stop_days: int | None = None
    close_before_expiry_days: int | None = None
    otm_pct: float | None = None
    dte_min: int | None = None
    dte_max: int | None = None
    entry_limit_buffer_pct: float | None = None
    stoch_rsi_mode: str | None = None
    stoch_oversold_max: float | None = None
    # Puts sleeve (separate kill switch and capacity from calls). Without
    # these fields the Pydantic model would silently drop the UI's puts_*
    # POSTs before they reached ai_store.save_settings.
    puts_enabled: bool | None = None
    puts_premium_cap_usd: float | None = None
    puts_max_concurrent: int | None = None
    puts_max_entries_per_day: int | None = None
    puts_disaster_stop_enabled: bool | None = None
    puts_disaster_stop_pct: float | None = None
    puts_disaster_stop_usd: float | None = None
    puts_take_profit_pct: float | None = None
    puts_sigma_target: float | None = None
    puts_time_stop_days: int | None = None
    puts_close_before_expiry_days: int | None = None
    puts_otm_pct: float | None = None
    puts_dte_min: int | None = None
    puts_dte_max: int | None = None
    puts_entry_limit_buffer_pct: float | None = None


@app.get("/api/ai/state")
def get_ai_state() -> dict:
    from backend import ai_trader
    return ai_trader.snapshot()


@app.post("/api/ai/settings")
def post_ai_settings(payload: AISettingsPayload) -> dict:
    from backend import ai_store
    # exclude_unset preserves explicit nulls (so the UI can DISABLE a field
    # like take_profit_pct or sigma_target by sending {key: null}) while
    # still skipping fields that weren't submitted at all.
    patch = payload.model_dump(exclude_unset=True)
    return {"ok": True, "settings": ai_store.save_settings(patch)}


@app.post("/api/ai/run")
def post_ai_run() -> dict:
    """Browser 'Run now' — same pipeline as the cron, no auth (UI path)."""
    from backend import ai_trader
    return ai_trader.run_cron(manual=True)


@app.api_route("/api/ai/cron", methods=["GET", "POST"])
def ai_cron(request: Request) -> dict:
    """Hourly Vercel cron. CRON_SECRET-gated (Bearer) like /api/scan. The
    handler self-gates to US market hours via the Alpaca clock, so a wide
    UTC cron window is fine."""
    expected = os.environ.get("CRON_SECRET")
    if expected:
        auth = request.headers.get("authorization") or request.headers.get("Authorization", "")
        if auth != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="unauthorized")
    from backend import ai_trader
    return ai_trader.run_cron(manual=False)


class BacktestConfigPayload(BaseModel):
    long_enabled: bool = True
    short_enabled: bool = True
    long_entry_sigma: float = -3.0
    short_entry_sigma: float = 3.0
    min_confidence: int = 0
    leverage: float = 5.0
    stop_loss_pct: float = 10.0
    profit_target_pct: float | None = 20.0
    trail_activation_pct: float | None = None
    trail_distance_pct: float | None = None
    tp_sigma: float | None = None
    sl_sigma: float | None = None
    time_stop_bars: int = 14
    starting_capital: float = 10000.0
    allocation_pct: float = 10.0
    require_trend_alignment: bool = False
    min_trend_pct: float = 0.0
    trend_direction: str = "any"
    require_stoch_extreme: bool = False
    stoch_oversold: float = 25.0
    stoch_overbought: float = 75.0
    period: str = "5y"


class SweepPayload(BacktestConfigPayload):
    metric: str = "sharpe"
    top_n: int = 10
    min_trades: int = 5


@app.post("/api/sweep/{ticker}")
def post_sweep(ticker: str, payload: SweepPayload) -> dict:
    df = fetch_bars(ticker, period=payload.period)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for ticker '{ticker}'")
    base = backtest.BacktestConfig(
        long_enabled=payload.long_enabled,
        short_enabled=payload.short_enabled,
        long_entry_sigma=payload.long_entry_sigma,
        short_entry_sigma=payload.short_entry_sigma,
        min_confidence=payload.min_confidence,
        leverage=payload.leverage,
        stop_loss_pct=payload.stop_loss_pct,
        profit_target_pct=payload.profit_target_pct,
        trail_activation_pct=payload.trail_activation_pct,
        trail_distance_pct=payload.trail_distance_pct,
        tp_sigma=payload.tp_sigma,
        sl_sigma=payload.sl_sigma,
        time_stop_bars=payload.time_stop_bars,
        starting_capital=payload.starting_capital,
        allocation_pct=payload.allocation_pct,
        require_trend_alignment=payload.require_trend_alignment,
        min_trend_pct=payload.min_trend_pct,
        trend_direction=payload.trend_direction,
        require_stoch_extreme=payload.require_stoch_extreme,
        stoch_oversold=payload.stoch_oversold,
        stoch_overbought=payload.stoch_overbought,
    )
    raw_results = backtest.sweep(df, base)
    # Drop runs with too few trades or NaN-ish metrics
    filtered = [
        r for r in raw_results
        if r["stats"]["trade_count"] >= payload.min_trades
        and r["stats"].get(payload.metric) is not None
        and np.isfinite(r["stats"].get(payload.metric, np.nan))
    ]
    filtered.sort(key=lambda r: r["stats"][payload.metric], reverse=True)
    return {
        "ticker": ticker.upper(),
        "metric": payload.metric,
        "total_evaluated": len(raw_results),
        "filtered_count": len(filtered),
        "grid": backtest.DEFAULT_SWEEP_GRID,
        "results": filtered[: payload.top_n],
    }


@app.post("/api/backtest/{ticker}")
def post_backtest(ticker: str, payload: BacktestConfigPayload) -> dict:
    df = fetch_bars(ticker, period=payload.period)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for ticker '{ticker}'")
    cfg = backtest.BacktestConfig(
        long_enabled=payload.long_enabled,
        short_enabled=payload.short_enabled,
        long_entry_sigma=payload.long_entry_sigma,
        short_entry_sigma=payload.short_entry_sigma,
        min_confidence=payload.min_confidence,
        leverage=payload.leverage,
        stop_loss_pct=payload.stop_loss_pct,
        profit_target_pct=payload.profit_target_pct,
        trail_activation_pct=payload.trail_activation_pct,
        trail_distance_pct=payload.trail_distance_pct,
        tp_sigma=payload.tp_sigma,
        sl_sigma=payload.sl_sigma,
        time_stop_bars=payload.time_stop_bars,
        starting_capital=payload.starting_capital,
        allocation_pct=payload.allocation_pct,
        require_trend_alignment=payload.require_trend_alignment,
        min_trend_pct=payload.min_trend_pct,
        trend_direction=payload.trend_direction,
        require_stoch_extreme=payload.require_stoch_extreme,
        stoch_oversold=payload.stoch_oversold,
        stoch_overbought=payload.stoch_overbought,
    )
    result = backtest.simulate(df, cfg)
    result["ticker"] = ticker.upper()
    return result


@app.get("/api/quote/{ticker}")
def get_quote(ticker: str) -> dict:
    """Latest IEX trade for the symbol (Alpaca free tier).

    Returns {ticker, price, timestamp, source}. 404 if Alpaca isn't
    configured or the symbol has no recent trade data.
    """
    from backend import alpaca
    if not alpaca.is_configured():
        raise HTTPException(status_code=503, detail="alpaca credentials not configured")
    try:
        result = alpaca.latest_trade(ticker)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}")
    if result is None:
        raise HTTPException(status_code=404, detail="no recent trade")
    return {"ticker": ticker.upper(), **result, "source": "alpaca-iex"}


@app.get("/api/summary/{ticker}")
def get_summary(ticker: str, period: str = FETCH_PERIOD):
    df = _prepare(ticker, period)
    latest = df.iloc[-1]
    return {
        "ticker": ticker.upper(),
        "current_price": _f(latest["close"]),
        "sd_position": _f(latest.get("sd_position", np.nan)),
        "r_squared": _f(latest.get("r_squared", np.nan)),
        "slope_annual_pct": _f(_annual_slope_pct(df)),
    }


def _serialize(df: pd.DataFrame, ticker: str, ichi: ichimoku.IchimokuComponents) -> dict:
    times = df["timestamp"].dt.strftime("%Y-%m-%d").tolist()

    candles = [
        {"time": t, "open": _f(o), "high": _f(h), "low": _f(l), "close": _f(c)}
        for t, o, h, l, c in zip(
            times, df["open"], df["high"], df["low"], df["close"]
        )
    ]
    buy_vol, sell_vol = split_volume(df)
    volume_total = [
        {"time": t, "value": _f(b + s), "color": VOL_SELL_COLOR}
        for t, b, s in zip(times, buy_vol, sell_vol)
    ]
    volume_buy = [
        {"time": t, "value": _f(b), "color": VOL_BUY_COLOR}
        for t, b in zip(times, buy_vol)
    ]

    overlays: dict[str, list] = {}
    if "regression_line" in df.columns and df["regression_line"].notna().any():
        overlays["regression_line"] = _line(times, df["regression_line"])
        for k in SIGMA_LEVELS:
            overlays[f"upper_{k}sd"] = _line(times, df[f"upper_{k}sd"])
            overlays[f"lower_{k}sd"] = _line(times, df[f"lower_{k}sd"])
        overlays["upper_2_5sd"] = _line(times, df["upper_2_5sd"])
        overlays["lower_2_5sd"] = _line(times, df["lower_2_5sd"])

    ichi_payload = _ichimoku_payload(times, ichi, shift=ICHIMOKU_SHIFT)

    latest = df.iloc[-1]
    summary = {
        "current_price": _f(latest["close"]),
        "sd_position": _f(latest.get("sd_position", np.nan)),
        "r_squared": _f(latest.get("r_squared", np.nan)),
        "slope_annual_pct": _f(_annual_slope_pct(df)),
        "window": REGRESSION_WINDOW,
        "name": fetch_ticker_name(ticker),
    }

    return {
        "ticker": ticker.upper(),
        "candles": candles,
        "volume_total": volume_total,
        "volume_buy": volume_buy,
        "volume_profile": compute_profile(df),
        "overlays": overlays,
        "ichimoku": ichi_payload,
        "summary": summary,
    }


def _ichimoku_payload(times: list[str], ichi: ichimoku.IchimokuComponents, shift: int) -> dict:
    last_ts = pd.to_datetime(times[-1])
    future = pd.bdate_range(start=last_ts, periods=shift + 1, freq="B")[1:]
    plot_times = list(times) + future.strftime("%Y-%m-%d").tolist()

    senkou_a_points: list[dict] = []
    senkou_b_points: list[dict] = []
    for i, (a, b) in enumerate(zip(ichi.senkou_a, ichi.senkou_b)):
        plot_idx = i + shift
        if plot_idx >= len(plot_times):
            break
        t = plot_times[plot_idx]
        if pd.notna(a):
            senkou_a_points.append({"time": t, "value": _f(a)})
        if pd.notna(b):
            senkou_b_points.append({"time": t, "value": _f(b)})

    return {
        "tenkan": _line(times, ichi.tenkan),
        "kijun": _line(times, ichi.kijun),
        "senkou_a": senkou_a_points,
        "senkou_b": senkou_b_points,
        "chikou": _line(times, ichi.chikou),
    }


def _line(times: list[str], series: pd.Series) -> list[dict]:
    return [
        {"time": t, "value": _f(v)}
        for t, v in zip(times, series)
        if pd.notna(v)
    ]


def _annual_slope_pct(df: pd.DataFrame) -> float:
    slope = df["slope"].iloc[-1] if "slope" in df.columns else np.nan
    price = df["close"].iloc[-1]
    if not np.isfinite(slope) or price == 0:
        return float("nan")
    return float(slope * 252 / price * 100)


def _f(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
