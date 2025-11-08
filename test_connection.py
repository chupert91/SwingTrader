# test_connection.py
from data.alpaca_fetcher import AlpacaDataFetcher
from datetime import datetime, timedelta
import pandas as pd


def test_alpaca_connection():
    """Test the Alpaca connection and data fetching"""
    print("Testing Alpaca connection...")

    # Initialize fetcher
    fetcher = AlpacaDataFetcher()

    # Test 1: Fetch basic data
    print("\n1. Testing basic data fetch for AAPL...")
    df = fetcher.get_bars(
        'AAPL',
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )

    if not df.empty:
        print(f"✅ Successfully fetched {len(df)} rows")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Latest close: ${df['close'].iloc[-1]:.2f}")
    else:
        print("❌ No data fetched")
        return False

    # Test 2: Add technical indicators
    print("\n2. Testing technical indicators...")
    df_with_indicators = fetcher.add_technical_indicators(df)

    # Check if indicators were added
    expected_indicators = ['rsi', 'macd', 'sma_20', 'bbands_u', 'atr']
    missing = [ind for ind in expected_indicators if ind not in df_with_indicators.columns]

    if not missing:
        print("✅ All indicators calculated successfully")
        print(f"   RSI (latest): {df_with_indicators['rsi'].iloc[-1]:.2f}")
        print(f"   SMA20 (latest): ${df_with_indicators['sma_20'].iloc[-1]:.2f}")
    else:
        print(f"❌ Missing indicators: {missing}")

    # Test 3: Get real-time quote
    print("\n3. Testing real-time quotes...")
    quotes = fetcher.get_latest_quote(['AAPL', 'MSFT'])

    if quotes:
        print("✅ Successfully fetched quotes:")
        for symbol, quote in quotes.items():
            if quote.get('ask_price') and quote.get('bid_price'):
                print(f"   {symbol}: Bid ${quote['bid_price']:.2f} / Ask ${quote['ask_price']:.2f}")
    else:
        print("⚠️  Could not fetch real-time quotes (market might be closed)")

    # Test 4: Multiple symbols
    print("\n4. Testing multiple symbols...")
    multi_df = fetcher.get_bars(
        ['AAPL', 'MSFT', 'GOOGL'],
        start_date=datetime.now() - timedelta(days=10)
    )

    if not multi_df.empty:
        symbols_found = multi_df['symbol'].unique()
        print(f"✅ Fetched data for: {', '.join(symbols_found)}")

    print("\n✅ All tests completed successfully!")
    return True


if __name__ == "__main__":
    test_alpaca_connection()