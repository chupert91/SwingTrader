"""252-day linear regression channels with residual-based ±σ bands."""
from __future__ import annotations

import numpy as np
import pandas as pd

REGRESSION_WINDOW = 252
SIGMA_LEVELS = (1, 2, 3)


def compute_channels(df: pd.DataFrame, window: int = REGRESSION_WINDOW) -> pd.DataFrame:
    """Add regression line and ±1σ/2σ/3σ bands as new columns.

    The regression is fit on the most recent `window` closes using x = bar index.
    Residual stdev is computed on the same window and applied as horizontal-offset
    bands above and below the regression line.

    Returns the input df with added columns:
        regression_line, upper_1sd, lower_1sd, upper_2sd, lower_2sd,
        upper_3sd, lower_3sd, sd_position, slope, r_squared
    """
    out = df.copy()
    n = len(out)
    if n < window:
        for col in _channel_columns():
            out[col] = np.nan
        return out

    closes = out["close"].to_numpy(dtype=float)
    y = closes[-window:]
    x = np.arange(window, dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    residuals = y - fit
    sigma = float(residuals.std(ddof=1))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Project the regression line across the entire df length so the chart can
    # show the line extending through earlier history too (using the same fit).
    full_x = np.arange(n, dtype=float) - (n - window)
    full_fit = slope * full_x + intercept

    out["regression_line"] = full_fit
    for k in SIGMA_LEVELS:
        out[f"upper_{k}sd"] = full_fit + k * sigma
        out[f"lower_{k}sd"] = full_fit - k * sigma

    out["sd_position"] = (closes - full_fit) / sigma if sigma > 0 else 0.0
    out["slope"] = slope
    out["r_squared"] = r_squared
    return out


def _channel_columns() -> list[str]:
    cols = ["regression_line"]
    for k in SIGMA_LEVELS:
        cols += [f"upper_{k}sd", f"lower_{k}sd"]
    cols += ["sd_position", "slope", "r_squared"]
    return cols
