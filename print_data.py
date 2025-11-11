# compare_regression_methods.py
# Compare single regression (TD Ameritrade style) vs rolling regression (current method)

from data.alpaca_fetcher import AlpacaDataFetcher
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import scipy.stats as stats

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


def add_single_regression(df):
    """
    Calculate ONE regression line through ALL data (TD Ameritrade style)
    No NaN values - every row gets a calculation
    """
    df = df.copy()

    x = np.arange(len(df))
    y = df['close'].values

    # Calculate single regression through entire dataset
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Calculate regression line for ALL points
    df['single_regression_line'] = slope * x + intercept
    df['single_regression_slope'] = slope
    df['single_regression_r2'] = r_value ** 2

    # Calculate residuals for ALL points
    residuals = y - df['single_regression_line']
    residual_std = np.std(residuals)

    # Channels for ALL points
    df['single_upper_2sd'] = df['single_regression_line'] + (2 * residual_std)
    df['single_lower_2sd'] = df['single_regression_line'] - (2 * residual_std)
    df['single_upper_3sd'] = df['single_regression_line'] + (3 * residual_std)
    df['single_lower_3sd'] = df['single_regression_line'] - (3 * residual_std)
    df['single_sd_position'] = residuals / residual_std
    df['single_channel_width'] = residual_std * 6

    print(f"Single Regression Stats:")
    print(f"  Slope: ${slope:.4f} per day")
    print(f"  R²: {r_value ** 2:.4f}")
    print(f"  Residual SD: ${residual_std:.2f}")

    return df


# Fetch data
fetcher = AlpacaDataFetcher()
df = fetcher.get_bars(
    'PLTR',
    start_date=datetime.now() - timedelta(days=500)
)

print(f"Fetched {len(df)} rows\n")

# Method 1: Rolling regression (current method - 252 day)
print("=" * 100)
print("METHOD 1: ROLLING REGRESSION (252-day window)")
print("=" * 100)
df_rolling = fetcher.add_regression_channels(df, period=252)
print(f"NaN values in first rows: {df_rolling['regression_line'].isna().sum()}")
print(f"First valid row: {df_rolling['regression_line'].first_valid_index()}\n")

# Method 2: Single regression (TD Ameritrade style)
print("=" * 100)
print("METHOD 2: SINGLE REGRESSION (entire dataset)")
print("=" * 100)
df_single = add_single_regression(df)
print(f"NaN values: {df_single['single_regression_line'].isna().sum()}")
print()

# Combine both for comparison
df_compare = df.copy()
df_compare['rolling_reg'] = df_rolling['regression_line']
df_compare['rolling_sd_pos'] = df_rolling['sd_position']
df_compare['single_reg'] = df_single['single_regression_line']
df_compare['single_sd_pos'] = df_single['single_sd_position']

# Show first 10 rows
print("=" * 100)
print("FIRST 10 ROWS - Comparison")
print("=" * 100)
print(df_compare[['timestamp', 'close', 'rolling_reg', 'single_reg', 'rolling_sd_pos', 'single_sd_pos']].head(10))

# Show middle rows (around day 251)
print("\n" + "=" * 100)
print("ROWS 248-258 - Around first rolling calculation")
print("=" * 100)
print(df_compare[['timestamp', 'close', 'rolling_reg', 'single_reg', 'rolling_sd_pos', 'single_sd_pos']].iloc[248:258])

# Show last 10 rows
print("\n" + "=" * 100)
print("LAST 10 ROWS - Both methods active")
print("=" * 100)
print(df_compare[['timestamp', 'close', 'rolling_reg', 'single_reg', 'rolling_sd_pos', 'single_sd_pos']].tail(10))

# Statistical comparison
print("\n" + "=" * 100)
print("STATISTICAL COMPARISON (for rows where both exist)")
print("=" * 100)

# Only compare rows where rolling regression exists
valid_rows = df_compare.dropna(subset=['rolling_reg'])

diff_reg_line = valid_rows['rolling_reg'] - valid_rows['single_reg']
diff_sd_pos = valid_rows['rolling_sd_pos'] - valid_rows['single_sd_pos']

print(f"\nRegression Line Differences:")
print(f"  Mean difference: ${diff_reg_line.mean():.2f}")
print(f"  Max difference: ${diff_reg_line.max():.2f}")
print(f"  Min difference: ${diff_reg_line.min():.2f}")
print(f"  Std deviation: ${diff_reg_line.std():.2f}")

print(f"\nSD Position Differences:")
print(f"  Mean difference: {diff_sd_pos.mean():.4f}")
print(f"  Max difference: {diff_sd_pos.max():.4f}")
print(f"  Min difference: {diff_sd_pos.min():.4f}")

# Show where they diverge the most
max_diff_idx = diff_reg_line.abs().idxmax()
print(f"\nLargest divergence at row {max_diff_idx}:")
print(f"  Date: {df_compare.loc[max_diff_idx, 'timestamp']}")
print(f"  Close: ${df_compare.loc[max_diff_idx, 'close']:.2f}")
print(f"  Rolling regression: ${df_compare.loc[max_diff_idx, 'rolling_reg']:.2f}")
print(f"  Single regression: ${df_compare.loc[max_diff_idx, 'single_reg']:.2f}")
print(f"  Difference: ${diff_reg_line[max_diff_idx]:.2f}")

print("\n" + "=" * 100)
print("KEY DIFFERENCES SUMMARY")
print("=" * 100)
print("\nROLLING REGRESSION (Current Method):")
print("  ✓ Adapts to recent price trends")
print("  ✓ Better for detecting regime changes")
print("  ✓ More responsive to volatility shifts")
print("  ✗ First 251 rows are NaN")
print("  ✗ More complex to calculate")

print("\nSINGLE REGRESSION (TD Ameritrade Style):")
print("  ✓ No NaN values - works from day 1")
print("  ✓ Simpler calculation")
print("  ✓ Shows overall trend of entire period")
print("  ✗ Doesn't adapt to changing conditions")
print("  ✗ Less useful for real-time trading signals")
print("  ✗ Can be misleading if trend changes mid-period")

print("\nRECOMMENDATION:")
print("  For ML trading models: Use ROLLING regression")
print("  For static chart visualization: Use SINGLE regression")