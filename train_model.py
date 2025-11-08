# train_model.py
from data.alpaca_fetcher import AlpacaDataFetcher
from models.multi_timeframe_predictor import MultiTimeframePredictor
from datetime import datetime, timedelta
import pandas as pd
import os


def train_models():
    """Train the prediction models"""
    print("🚀 Starting model training...")

    # Initialize components
    fetcher = AlpacaDataFetcher()
    predictor = MultiTimeframePredictor()

    # Define training symbols (diverse set for better generalization)
    training_symbols = [
        # Your watchlist
        'PLTR', 'HOOD', 'SOFI', 'MP', 'IONQ',
        'AMD', 'NVDA', 'TSLA', 'AAPL',

        # Market indices
        'SPY', 'QQQ', 'DIA', 'IWM',

        # Balance symbols
        'MSFT', 'GOOGL',  # Stable tech
        'JPM', 'BAC',  # Financials
        'XOM', 'CVX',  # Energy
        'JNJ', 'PG',  # Defensive
        'GLD', 'TLT',  # Safe havens
        'VIX'  # Volatility
    ]

    print(f"Fetching data for {len(training_symbols)} symbols...")
    all_data = []

    for symbol in training_symbols:
        print(f"  Fetching {symbol}...")
        try:
            df = fetcher.get_bars(
                symbol,
                start_date=datetime.now() - timedelta(days=500),
                end_date=datetime.now()
            )

            if not df.empty:
                # Add indicators
                df = fetcher.add_technical_indicators(df)
                all_data.append(df)
                print(f"    ✅ {symbol}: {len(df)} rows fetched")
            else:
                print(f"    ⚠️  {symbol}: No data")

        except Exception as e:
            print(f"    ❌ {symbol}: Error - {e}")

    if all_data:
        # Combine all data
        print("\nCombining data...")
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total training samples: {len(combined_df)}")

        # Train models
        print("\nTraining models for different timeframes...")
        predictor.train_models(combined_df)

        # Create models directory if it doesn't exist
        os.makedirs('models/saved', exist_ok=True)

        # Save models
        save_path = 'models/saved/timeframe_models.pkl'
        predictor.save_models(save_path)
        print(f"\n✅ Models saved to {save_path}")

        return True
    else:
        print("\n❌ No data available for training")
        return False


if __name__ == "__main__":
    success = train_models()
    if success:
        print("\n🎉 Training completed successfully!")
        print("You can now run predictions using run_predictions.py")
    else:
        print("\n❌ Training failed. Please check your data connection.")