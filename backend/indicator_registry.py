"""Pluggable technical-indicator registry.

Each indicator declares itself via `register(Indicator(...))`. The chart
endpoint asks the registry to compute whichever indicators the user has
turned on in their /api/indicators/active list, and the registry returns
a serializable payload the frontend pane manager can render generically.

This is intentionally separate from backend/indicators.py — that module's
contents (Stoch RSI, MACD) plus ichimoku/channels stay hardcoded on the
chart and are not part of this registry. The registry is purely additive:
indicators a user opts into via the Indicators picker.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


ParamType = Literal["int", "float", "bool", "select", "color"]


@dataclass
class IndicatorParam:
    id: str
    label: str
    type: ParamType
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[list[str]] = None  # for type="select"
    help: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "options": self.options,
            "help": self.help,
        }


# How each emitted plot item should be drawn on the frontend.
PlotKind = Literal["line", "candle", "histogram", "marker", "price_line", "fill"]


@dataclass
class PlotItem:
    """One drawable item produced by an indicator compute.

    - kind=line: data = [{time, value}, ...]
    - kind=candle: data = [{time, open, high, low, close}, ...]
    - kind=histogram: data = [{time, value, color?}, ...]
    - kind=marker: data = [{time, position: "aboveBar"|"belowBar", color, shape, text?}, ...]
    - kind=price_line: data = [{price, title?, color, lineStyle?, lineWidth?}] — one or more horizontal levels on the price pane (used by Reverse Cutler's RSI projections)
    - kind=fill: data = [{time, top, bottom}, ...] — paired top/bottom values per bar
    """
    kind: PlotKind
    name: str
    data: list[dict]
    pane: Literal["price", "own"] = "own"
    style: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "pane": self.pane,
            "data": self.data,
            "style": self.style,
        }


@dataclass
class IndicatorResult:
    """What an indicator's compute_fn returns."""
    pane_title: str
    items: list[PlotItem]
    # Optional Y-axis range hints for the indicator's own pane (e.g. RSI 0-100)
    pane_y_range: Optional[tuple[float, float]] = None

    def to_dict(self) -> dict:
        return {
            "pane_title": self.pane_title,
            "pane_y_range": list(self.pane_y_range) if self.pane_y_range else None,
            "items": [i.to_dict() for i in self.items],
        }


ComputeFn = Callable[[pd.DataFrame, dict], IndicatorResult]


@dataclass
class Indicator:
    id: str
    name: str
    category: str
    description: str
    params: list[IndicatorParam]
    compute_fn: ComputeFn
    # True if this indicator wants its own pane below the price chart.
    # False = overlays render onto the price pane only (e.g. Supertrend).
    has_own_pane: bool

    def default_params(self) -> dict:
        return {p.id: p.default for p in self.params}

    def to_summary_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "has_own_pane": self.has_own_pane,
            "params": [p.to_dict() for p in self.params],
        }


_registry: dict[str, Indicator] = {}


def register(ind: Indicator) -> None:
    if ind.id in _registry:
        logger.warning("re-registering indicator %s", ind.id)
    _registry[ind.id] = ind


def get(indicator_id: str) -> Optional[Indicator]:
    return _registry.get(indicator_id)


def list_all() -> list[Indicator]:
    return sorted(_registry.values(), key=lambda i: (i.category, i.name))


def compute_active(df: pd.DataFrame, active: list[dict]) -> list[dict]:
    """Compute every active indicator and return serializable results.

    `active` is the user's saved config list — each entry shape:
      {"indicator_id": "supertrend", "params": {...}}

    Bad/missing indicator ids are silently skipped (compute failures too —
    we log and skip rather than fail the whole chart payload).
    """
    out: list[dict] = []
    for entry in active or []:
        iid = entry.get("indicator_id") if isinstance(entry, dict) else None
        if not iid:
            continue
        ind = get(iid)
        if not ind:
            continue
        params = ind.default_params()
        user_params = entry.get("params") or {}
        if isinstance(user_params, dict):
            params.update(user_params)
        try:
            result = ind.compute_fn(df, params)
        except Exception as exc:
            logger.warning("indicator %s compute failed: %s", iid, exc, exc_info=True)
            continue
        out.append({
            "indicator_id": ind.id,
            "name": ind.name,
            "has_own_pane": ind.has_own_pane,
            "params": params,
            **result.to_dict(),
        })
    return out


# Serialization helpers shared by indicator impls -------------------------

def _f(x: Any) -> Optional[float]:
    """Coerce to JSON-safe float (None for NaN/inf), matching backend/main.py."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def line_points(times: list[str], series: pd.Series) -> list[dict]:
    """Standard {time, value} list, NaN-skipped — same shape as backend/main.py overlays."""
    return [
        {"time": t, "value": _f(v)}
        for t, v in zip(times, series)
        if pd.notna(v)
    ]


def histogram_points(times: list[str], series: pd.Series, color_fn=None) -> list[dict]:
    """Histogram data; optional `color_fn(value, prev_value) -> hex_color`."""
    out: list[dict] = []
    prev = None
    for t, v in zip(times, series):
        if pd.isna(v):
            prev = v
            continue
        pt = {"time": t, "value": _f(v)}
        if color_fn is not None:
            try:
                c = color_fn(v, prev)
                if c is not None:
                    pt["color"] = c
            except Exception:
                pass
        out.append(pt)
        prev = v
    return out


__all__ = [
    "Indicator",
    "IndicatorParam",
    "PlotItem",
    "IndicatorResult",
    "register",
    "get",
    "list_all",
    "compute_active",
    "line_points",
    "histogram_points",
]
