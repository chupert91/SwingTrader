# data/alpaca_fetcher.py (using 'ta' library)
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
import ta  # Using 'ta' instead of 'pandas_ta'

load_dotenv()


class AlpacaDataFetcher:
    def __init__(self):
        self.data_client = StockHistoricalDataClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY')
        )

        self.trading_client = TradingClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            paper=True
        )

    def get_bars(self, symbols, start_date, end_date=None, timeframe=TimeFrame.Day):
        """Fetch historical bar data for given symbols"""
        if isinstance(symbols, str):
            symbols = [symbols]

        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start_date,
            end=end_date
        )

        try:
            bars = self.data_client.get_stock_bars(request_params)
        except Exception as e:
            print(f"Error fetching bars: {e}")
            return pd.DataFrame()

        df_list = []
        for symbol in symbols:
            if symbol in bars.data:
                symbol_bars = bars.data[symbol]
                df = pd.DataFrame([{
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'trade_count': bar.trade_count,
                    'vwap': bar.vwap,
                } for bar in symbol_bars])
                df['symbol'] = symbol
                df_list.append(df)

        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            return combined_df
        return pd.DataFrame()

    # Add this method to your AlpacaDataFetcher class in data/alpaca_fetcher.py

    def add_regression_channels(self, df, period=249):
        """
        Add linear regression and residual-based standard deviation channels

        Parameters:
        - df: DataFrame with OHLCV data
        - period: Regression period (default 252 for yearly)
        """
        import scipy.stats as stats

        df = df.copy()

        # Ensure we have enough data
        if len(df) < period:
            print(f"Warning: Only {len(df)} days of data, need {period} for regression")
            period = min(len(df), period)

        # Initialize columns with NaN
        df['regression_line'] = np.nan
        df['regression_slope'] = np.nan
        df['regression_r2'] = np.nan
        df['residual'] = np.nan
        df['residual_sd'] = np.nan

        # SD channel columns
        for sd in [1, 2, 3]:
            df[f'upper_{sd}sd'] = np.nan
            df[f'lower_{sd}sd'] = np.nan

        df['sd_position'] = np.nan  # Current position in SDs from regression
        df['channel_width'] = np.nan

        # Calculate rolling regression
        for i in range(period - 1, len(df)):
            # Get the window of data
            window_data = df.iloc[i - period + 1: i + 1]

            # X values (days) and Y values (prices)
            x = np.arange(len(window_data))
            y = window_data['close'].values

            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Calculate regression line value for current point
            regression_value = slope * (len(window_data) - 1) + intercept

            # Store regression values
            df.loc[df.index[i], 'regression_line'] = regression_value
            df.loc[df.index[i], 'regression_slope'] = slope
            df.loc[df.index[i], 'regression_r2'] = r_value ** 2

            # Calculate residuals for the entire window
            regression_line_window = slope * x + intercept
            residuals = y - regression_line_window

            # Calculate standard deviation of residuals
            residual_std = np.std(residuals)

            # Current residual (distance from regression line)
            current_residual = y[-1] - regression_value

            df.loc[df.index[i], 'residual'] = current_residual
            df.loc[df.index[i], 'residual_sd'] = residual_std

            # Calculate SD channels
            for sd_level in [1, 2, 3]:
                df.loc[df.index[i], f'upper_{sd_level}sd'] = regression_value + (residual_std * sd_level)
                df.loc[df.index[i], f'lower_{sd_level}sd'] = regression_value - (residual_std * sd_level)

            # Calculate current position in standard deviations
            if residual_std > 0:
                df.loc[df.index[i], 'sd_position'] = current_residual / residual_std
            else:
                df.loc[df.index[i], 'sd_position'] = 0

            # Channel width (6 SD total, from -3 to +3)
            df.loc[df.index[i], 'channel_width'] = residual_std * 6

        # Add derived features for trading signals
        df['at_extreme'] = (abs(df['sd_position']) >= 2.5).astype(int)
        df['at_3sd'] = (abs(df['sd_position']) >= 2.8).astype(int)  # Near 3 SD
        df['above_regression'] = (df['close'] > df['regression_line']).astype(int)
        df['sd_cross_signal'] = df['sd_position'].diff()  # Momentum through SD levels

        # Trend strength features
        df['trend_strength'] = df['regression_slope'] * df['regression_r2']  # Slope weighted by R²
        df['trend_consistency'] = df['regression_r2']  # How well price follows trend

        # Distance features (as percentages)
        df['pct_from_regression'] = (df['close'] - df['regression_line']) / df['regression_line'] * 100
        df['pct_to_upper_3sd'] = (df['upper_3sd'] - df['close']) / df['close'] * 100
        df['pct_to_lower_3sd'] = (df['close'] - df['lower_3sd']) / df['close'] * 100

        return df

    def add_regression_features_for_ml(self, df):
        """
        Add specific features for ML model based on regression channels
        """
        df = df.copy()

        # Ensure regression channels are calculated
        if 'regression_line' not in df.columns:
            df = self.add_regression_channels(df)

        # Mean reversion features
        df['sd_position_lag1'] = df['sd_position'].shift(1)
        df['sd_position_lag2'] = df['sd_position'].shift(2)
        df['sd_position_change'] = df['sd_position'] - df['sd_position_lag1']

        # Extreme reversion counting
        df['days_above_2sd'] = 0
        df['days_below_2sd'] = 0

        for i in range(1, len(df)):
            if df['sd_position'].iloc[i] > 2:
                df.loc[df.index[i], 'days_above_2sd'] = df['days_above_2sd'].iloc[i - 1] + 1
            else:
                df.loc[df.index[i], 'days_above_2sd'] = 0

            if df['sd_position'].iloc[i] < -2:
                df.loc[df.index[i], 'days_below_2sd'] = df['days_below_2sd'].iloc[i - 1] + 1
            else:
                df.loc[df.index[i], 'days_below_2sd'] = 0

        # Historical behavior at SD levels
        df['touches_3sd_20d'] = (abs(df['sd_position']) >= 2.8).rolling(20).sum()
        df['mean_reversion_20d'] = df['sd_position'].rolling(20).apply(
            lambda x: np.sum(np.diff(np.sign(x)) != 0) if len(x) > 1 else 0
        )

        # Volatility regime
        df['channel_width_ma20'] = df['channel_width'].rolling(20).mean()
        df['channel_width_expanding'] = (df['channel_width'] > df['channel_width_ma20']).astype(int)

        # Trend regime changes
        df['regression_slope_change'] = df['regression_slope'].diff()
        df['trend_accelerating'] = (df['regression_slope_change'] > 0).astype(int)

        return df

    def add_technical_indicators(self, df):
        """Add technical indicators using 'ta' library"""
        df = df.sort_values('timestamp').copy()

        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']

        # Trend Indicators
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bbands_u'] = bollinger.bollinger_hband()
        df['bbands_m'] = bollinger.bollinger_mavg()
        df['bbands_l'] = bollinger.bollinger_lband()
        df['bbands_b'] = bollinger.bollinger_pband()  # %B
        df['bbands_w'] = bollinger.bollinger_wband()  # Bandwidth

        # ATR
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

        # MACD
        macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macds'] = macd.macd_signal()
        df['macdh'] = macd.macd_diff()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        df['stochk'] = stoch.stoch()
        df['stochd'] = stoch.stoch_signal()

        # Volume indicators
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # MFI
        df['mfi'] = ta.volume.MFIIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            window=14
        ).money_flow_index()

        # VWAP (simple implementation since ta doesn't have it)
        df['vwap_ta'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

        # Support and Resistance
        df['resistance_20'] = df['high'].rolling(window=20).max()
        df['support_20'] = df['low'].rolling(window=20).min()
        df['resistance_52w'] = df['high'].rolling(window=252, min_periods=20).max()
        df['support_52w'] = df['low'].rolling(window=252, min_periods=20).min()

        # Price position
        df['price_position_20'] = (df['close'] - df['support_20']) / (df['resistance_20'] - df['support_20'])

        # Volatility
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

        # Add regression channels (252-day for yearly)
        df = self.add_regression_channels(df, period=249)

        # Add ML-specific regression features
        df = self.add_regression_features_for_ml(df)

        # Fill NaN values
        df = df.ffill()

        return df

    def get_latest_quote(self, symbols):
        """Get real-time quotes for symbols"""
        if isinstance(symbols, str):
            symbols = [symbols]

        try:
            request_params = StockQuotesRequest(symbol_or_symbols=symbols)
            quotes = self.data_client.get_stock_latest_quote(request_params)

            return {
                symbol: {
                    'ask_price': quote.ask_price,
                    'bid_price': quote.bid_price,
                    'ask_size': quote.ask_size,
                    'bid_size': quote.bid_size,
                    'timestamp': quote.timestamp,
                    'spread': quote.ask_price - quote.bid_price if quote.ask_price and quote.bid_price else None
                }
                for symbol, quote in quotes.items()
            }
        except Exception as e:
            print(f"Error fetching quotes: {e}")
            return {}