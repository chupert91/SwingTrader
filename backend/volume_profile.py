"""Fixed-window volume profile: volume distributed across price buckets.

Daily OHLCV only (no intraday prints), so each bar's volume is spread
uniformly across its [low, high] range into the price buckets it overlaps.
This is the standard daily-bar approximation of a true volume profile.
Returns per-bucket volume plus the Point of Control and the 70% Value Area.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUM_BINS = 50
VALUE_AREA_FRAC = 0.70


def compute_profile(df: pd.DataFrame, num_bins: int = NUM_BINS) -> dict | None:
    if df.empty or not {"high", "low", "volume"}.issubset(df.columns):
        return None

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)

    ok = np.isfinite(high) & np.isfinite(low) & np.isfinite(vol) & (vol > 0)
    if not ok.any():
        return None
    high, low, vol = high[ok], low[ok], vol[ok]

    p_min = float(low.min())
    p_max = float(high.max())
    if not (np.isfinite(p_min) and np.isfinite(p_max)) or p_max <= p_min:
        return None

    edges = np.linspace(p_min, p_max, num_bins + 1)
    bin_w = (p_max - p_min) / num_bins
    totals = np.zeros(num_bins)

    for h, l, v in zip(high, low, vol):
        if h <= l:
            # No intraday range (e.g. halted): drop the whole bar in one bucket.
            idx = min(int((l - p_min) / bin_w), num_bins - 1)
            totals[idx] += v
            continue
        # Volume share per bucket = fraction of [l, h] overlapping that bucket.
        lo = np.maximum(edges[:-1], l)
        hi = np.minimum(edges[1:], h)
        overlap = np.clip(hi - lo, 0.0, None)
        totals += v * (overlap / (h - l))

    total_vol = float(totals.sum())
    if total_vol <= 0:
        return None

    poc_idx = int(totals.argmax())

    # Value area: start at the POC bucket, expand toward whichever neighbour
    # holds more volume, until VALUE_AREA_FRAC of total volume is enclosed.
    target = total_vol * VALUE_AREA_FRAC
    lo_idx = hi_idx = poc_idx
    running = float(totals[poc_idx])
    while running < target and (lo_idx > 0 or hi_idx < num_bins - 1):
        below = float(totals[lo_idx - 1]) if lo_idx > 0 else -1.0
        above = float(totals[hi_idx + 1]) if hi_idx < num_bins - 1 else -1.0
        if above >= below:
            hi_idx += 1
            running += float(totals[hi_idx])
        else:
            lo_idx -= 1
            running += float(totals[lo_idx])

    bins = [
        {"low": float(edges[i]), "high": float(edges[i + 1]), "volume": float(totals[i])}
        for i in range(num_bins)
    ]
    return {
        "bins": bins,
        "max_volume": float(totals.max()),
        "poc": float((edges[poc_idx] + edges[poc_idx + 1]) / 2.0),
        "va_low": float(edges[lo_idx]),
        "va_high": float(edges[hi_idx + 1]),
    }
