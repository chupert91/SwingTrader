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

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')

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