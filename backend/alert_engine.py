"""Rule-driven alert detection.

An AlertRule is a saved configuration that decides which σ cross + trend
combination should fire an email. The default rule mirrors the optimized
strategy from optimize_trend.py (long-only, -2σ entry, slope > 30%/yr,
TSLA/NVDA/PLTR/MP), but users can clone and edit rules to widen the net.

Detection is pure: given (rule, prepared_df, ticker) -> Signal | None.
Persistence (dedup + history) lives in kv.py.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal

import pandas as pd

from backend.alerts import Signal, prepare_df  # reuse the existing Signal + indicator prep
from backend.channels import REGRESSION_WINDOW, SIGMA_LEVELS

Side = Literal["long", "short", "both"]

DEFAULT_OPTIMIZED_TICKERS = ["TSLA", "NVDA", "PLTR", "MP"]


@dataclass
class AlertRule:
    """A saved alert configuration.

    The defaults match the Sharpe-optimal config from the 2026-05-12 sweep.
    """
    id: str
    name: str
    tickers: list[str]
    side: Side = "both"
    # Minimum |σ| level to start alerting on. Engine fires for every band
    # in SIGMA_LEVELS (1/2/3) whose absolute value is >= |entry_sigma|, so
    # entry_sigma=-1 generates alerts at -1σ, -2σ, -3σ (longs) and +1σ/+2σ/+3σ (shorts).
    entry_sigma: float = -1.0
    require_trend: bool = False
    min_trend_pct: float = 0.0        # annualized slope %
    exit_target_pct: float = 20.0     # option P&L target (advisory — shown in email)
    exit_stop_pct: float = 10.0       # option P&L stop (advisory)
    leverage: float = 5.0             # for translating option pcts -> underlying pcts
    enabled: bool = True
    notify_email: str = ""            # set to user's address; empty = no email
    # Slope-sign confluence: "any" = off, "up" = require slope > 0, "down" = require slope < 0
    trend_direction: str = "any"
    # Stoch RSI extreme confluence
    require_stoch_extreme: bool = True
    stoch_oversold: float = 35.0
    stoch_overbought: float = 65.0
    # Liquidity filter: 20-bar avg dollar volume in millions. 0 = off.
    # Applies to both watchlist alerts (detect) and discovery scans.
    min_avg_volume_m: float = 10.0

    @classmethod
    def optimized_default(cls, notify_email: str = "") -> "AlertRule":
        return cls(
            id=str(uuid.uuid4()),
            name="σ + trend + stoch confluence (both sides)",
            tickers=list(DEFAULT_OPTIMIZED_TICKERS),
            side="both",
            entry_sigma=-1.0,
            require_trend=True,
            min_trend_pct=0.0,
            exit_target_pct=20.0,
            exit_stop_pct=10.0,
            leverage=5.0,
            enabled=True,
            notify_email=notify_email,
            trend_direction="any",
            require_stoch_extreme=True,
            stoch_oversold=35.0,
            stoch_overbought=65.0,
            min_avg_volume_m=10.0,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AlertRule":
        td = str(data.get("trend_direction", "any")).lower()
        if td not in ("any", "up", "down"):
            td = "any"
        return cls(
            id=str(data.get("id") or uuid.uuid4()),
            name=str(data.get("name", "Unnamed rule")),
            tickers=[t.upper() for t in data.get("tickers", []) if isinstance(t, str) and t.strip()],
            side=data.get("side", "both"),
            entry_sigma=float(data.get("entry_sigma", -1.0)),
            require_trend=bool(data.get("require_trend", False)),
            min_trend_pct=float(data.get("min_trend_pct", 0.0)),
            exit_target_pct=float(data.get("exit_target_pct", 20.0)),
            exit_stop_pct=float(data.get("exit_stop_pct", 10.0)),
            leverage=float(data.get("leverage", 5.0)),
            enabled=bool(data.get("enabled", True)),
            notify_email=str(data.get("notify_email", "")),
            trend_direction=td,
            require_stoch_extreme=bool(data.get("require_stoch_extreme", True)),
            stoch_oversold=float(data.get("stoch_oversold", 35.0)),
            stoch_overbought=float(data.get("stoch_overbought", 65.0)),
            min_avg_volume_m=float(data.get("min_avg_volume_m", 10.0)),
        )


@dataclass
class RuleSignal:
    """A Signal fired by a specific rule. Adds rule_id + trend_pct context."""
    rule_id: str
    rule_name: str
    ticker: str
    direction: Literal["long", "short"]
    bar_date: str
    price: float
    sd_position: float
    trend_pct: float                  # annualized regression slope at signal bar
    exit_target_pct: float            # echo from rule for email body
    exit_stop_pct: float
    leverage: float
    sigma_level: float = 0.0          # absolute σ band that triggered this signal (1, 2, or 3)
    confidence: int = 0               # not used by optimized rule; kept for legacy display
    confirmations: list[str] = field(default_factory=list)

    def dedup_key(self) -> str:
        return f"{self.rule_id}|{self.ticker}|{self.direction}|{self.sigma_level}|{self.bar_date}"

    def to_dict(self) -> dict:
        return asdict(self)


def detect(rule: AlertRule, ticker: str, df: pd.DataFrame) -> list[RuleSignal]:
    """Return all RuleSignals `rule` produces for `ticker` on the latest bar.

    For each σ band in SIGMA_LEVELS whose |level| >= |rule.entry_sigma|, fires
    one signal per side ("long"/"short" per rule.side) if price crossed that
    band on this bar AND confluence gates pass. Returns [] when nothing triggers.
    """
    if not rule.enabled:
        return []
    if len(df) < REGRESSION_WINDOW + 1:
        return []
    # NOTE: rule.tickers is no longer consulted here — scan.py iterates the
    # live watchlist directly, so detect() just answers "does this rule
    # trigger on this ticker?" without an implicit allow-list filter.

    prepared = prepare_df(df)
    prev = prepared.iloc[-2]
    cur = prepared.iloc[-1]

    sd_prev = prev.get("sd_position")
    sd_cur = cur.get("sd_position")
    if pd.isna(sd_prev) or pd.isna(sd_cur):
        return []

    # Liquidity gate: skip thinly traded names (rule-configurable, 0 = off).
    if rule.min_avg_volume_m > 0:
        avg_dv_m = _avg_dollar_volume_m(prepared)
        if avg_dv_m < rule.min_avg_volume_m:
            return []

    trend_pct = _annualized_trend_pct(prepared)
    slope = cur.get("slope")
    k = cur.get("stoch_rsi_k")
    confidence, confirmations = _legacy_confidence("long", cur, prepared)  # context for email

    threshold = abs(float(rule.entry_sigma))
    active_levels = [float(lvl) for lvl in SIGMA_LEVELS if float(lvl) >= threshold]

    signals: list[RuleSignal] = []
    for lvl in active_levels:
        long_band = -lvl
        short_band = lvl

        candidates: list[Literal["long", "short"]] = []
        if rule.side in ("long", "both") and sd_prev >= long_band and sd_cur < long_band:
            candidates.append("long")
        if rule.side in ("short", "both") and sd_prev <= short_band and sd_cur > short_band:
            candidates.append("short")
        if not candidates:
            continue

        for direction in candidates:
            if rule.require_trend:
                if direction == "long" and trend_pct <= rule.min_trend_pct:
                    continue
                if direction == "short" and trend_pct >= -rule.min_trend_pct:
                    continue
            if rule.trend_direction in ("up", "down"):
                if pd.isna(slope):
                    continue
                if rule.trend_direction == "up" and slope <= 0:
                    continue
                if rule.trend_direction == "down" and slope >= 0:
                    continue
            if rule.require_stoch_extreme:
                if pd.isna(k):
                    continue
                if direction == "long" and k > rule.stoch_oversold:
                    continue
                if direction == "short" and k < rule.stoch_overbought:
                    continue

            signals.append(RuleSignal(
                rule_id=rule.id,
                rule_name=rule.name,
                ticker=ticker.upper(),
                direction=direction,
                bar_date=_iso_date(cur["timestamp"]),
                price=float(cur["close"]),
                sd_position=float(sd_cur),
                trend_pct=float(trend_pct),
                exit_target_pct=rule.exit_target_pct,
                exit_stop_pct=rule.exit_stop_pct,
                leverage=rule.leverage,
                sigma_level=lvl,
                confidence=confidence,
                confirmations=confirmations,
            ))
    return signals


def _avg_dollar_volume_m(prepared: pd.DataFrame, window: int = 20) -> float:
    """Average dollar volume (close × volume) over the last `window` bars,
    expressed in millions. Returns 0.0 if there's not enough data."""
    if "volume" not in prepared.columns or len(prepared) < window:
        return 0.0
    tail = prepared.iloc[-window:]
    dollar_vol = (tail["close"] * tail["volume"]).mean()
    if pd.isna(dollar_vol):
        return 0.0
    return float(dollar_vol) / 1_000_000.0


def _annualized_trend_pct(prepared: pd.DataFrame) -> float:
    """Annualized regression slope as % of latest price.

    compute_channels() adds a single 'slope' value for the final fit. We use
    that — it represents the trend of the most recent 252-day regression.
    """
    slope = prepared["slope"].iloc[-1] if "slope" in prepared.columns else None
    price = prepared["close"].iloc[-1]
    if slope is None or pd.isna(slope) or price <= 0:
        return 0.0
    return float(slope) * 252.0 / float(price) * 100.0


def _legacy_confidence(direction, cur, df) -> tuple[int, list[str]]:
    """Reuse the existing 3-indicator scoring for email-body context only.
    The optimized rule doesn't gate on this, but seeing it helps the human."""
    from backend.alerts import _score
    return _score(direction, cur, df)


def _iso_date(ts) -> str:
    if hasattr(ts, "date"):
        return str(ts.date())
    return str(ts)[:10]


__all__ = [
    "AlertRule",
    "RuleSignal",
    "DEFAULT_OPTIMIZED_TICKERS",
    "detect",
]
