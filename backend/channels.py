"""252-day linear regression channels with residual-based ±σ bands."""
from __future__ import annotations

import numpy as np
import pandas as pd

REGRESSION_WINDOW = 252
SIGMA_LEVELS = (1, 2, 3)
# "Gold zone": the strip between the 2σ and 2.5σ bands (both sides). Kept
# separate from SIGMA_LEVELS so alert/threshold logic over the integer
# levels stays unchanged.
GOLD_ZONE = (2.0, 2.5)


def compute_channels(df: pd.DataFrame, window: int = REGRESSION_WINDOW) -> pd.DataFrame:
    """Add regression line and ±1σ/2σ/3σ bands as new columns.

    The regression is fit on the most recent `window` closes using x = bar index.
    Residual stdev is computed on the same window and applied as horizontal-offset
    bands above and below the regression line.

    Returns the input df with added columns:
        regression_line, upper_1sd, lower_1sd, upper_2sd, lower_2sd,
        upper_3sd, lower_3sd, upper_2_5sd, lower_2_5sd,
        sd_position, slope, r_squared
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
    out["upper_2_5sd"] = full_fit + 2.5 * sigma
    out["lower_2_5sd"] = full_fit - 2.5 * sigma

    out["sd_position"] = (closes - full_fit) / sigma if sigma > 0 else 0.0
    out["slope"] = slope
    out["r_squared"] = r_squared
    return out


def _channel_columns() -> list[str]:
    cols = ["regression_line"]
    for k in SIGMA_LEVELS:
        cols += [f"upper_{k}sd", f"lower_{k}sd"]
    cols += ["upper_2_5sd", "lower_2_5sd"]
    cols += ["sd_position", "slope", "r_squared"]
    return cols
