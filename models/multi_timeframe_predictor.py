# models/multi_timeframe_predictor.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib


@dataclass
class PredictionResult:
    ticker: str
    current_price: float
    predictions: Dict[str, float]  # timeframe -> predicted_price
    confidence_scores: Dict[str, float]  # timeframe -> confidence
    technical_signals: Dict[str, any]
    risk_metrics: Dict[str, float]
    recommendation: str
    analysis_timestamp: pd.Timestamp


class MultiTimeframePredictor:
    def __init__(self):
        self.models = {}  # Store different models for different timeframes
        self.scalers = {}
        self.timeframes = {
            '1_day': 1,
            '3_days': 3,
            '1_week': 5,
            '2_weeks': 10,
            '1_month': 22,
        }

    def train_models(self, df: pd.DataFrame):
        """Train separate models for each timeframe"""
        for timeframe_name, periods in self.timeframes.items():
            print(f"Training {timeframe_name} model...")

            # Create target for this timeframe
            df_timeframe = df.copy()
            df_timeframe['target'] = df_timeframe['close'].shift(-periods)
            df_timeframe['target_return'] = (df_timeframe['target'] / df_timeframe['close']) - 1

            # Create features
            features = self._engineer_features(df_timeframe, periods)

            # Remove NaN rows
            valid_data = features.join(df_timeframe[['target', 'target_return']]).dropna()

            X = valid_data.drop(['target', 'target_return'], axis=1)
            y = valid_data['target']

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train model
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.01,
                objective='reg:squarederror',
                random_state=42
            )

            # Use time series split for validation
            split_point = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=False
            )

            # Store model and scaler
            self.models[timeframe_name] = model
            self.scalers[timeframe_name] = scaler

            # Calculate accuracy metrics
            predictions = model.predict(X_test)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            print(f"  MAPE: {mape:.2f}%")

    def _engineer_features(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Create features optimized for specific prediction horizon"""
        features = pd.DataFrame(index=df.index)

        # Price momentum features (different lookbacks based on horizon)
        lookbacks = [1, 3, 5, 10, 20, 50] if horizon > 5 else [1, 2, 3, 5, 10]

        for lb in lookbacks:
            features[f'return_{lb}d'] = df['close'].pct_change(lb)
            features[f'volume_ratio_{lb}d'] = df['volume'] / df['volume'].rolling(lb).mean()

        # Volatility features
        features['volatility_20'] = df['close'].pct_change().rolling(20).std()
        features['atr_ratio'] = df['atr'] / df['close']
        features['bb_width'] = (df['bbands_u'] - df['bbands_l']) / df['close']
        features['bb_position'] = (df['close'] - df['bbands_l']) / (df['bbands_u'] - df['bbands_l'])

        # Technical indicators
        features['rsi'] = df['rsi']
        features['macd_signal'] = df['macd'] - df['macds']
        features['stoch_k'] = df['stochk']

        # Moving average features
        for ma in [10, 20, 50]:
            features[f'price_to_ma{ma}'] = df['close'] / df[f'sma_{ma}'] - 1

        # Market microstructure
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        # Support/Resistance features
        features['dist_from_52w_high'] = df['close'] / df['high'].rolling(252).max() - 1
        features['dist_from_52w_low'] = df['close'] / df['low'].rolling(252).min() - 1

        return features

    def predict(self, ticker: str, df: pd.DataFrame) -> PredictionResult:
        """Generate predictions for all timeframes"""

        # Prepare current features
        current_features = self._engineer_features(df, 1)
        current_row = current_features.iloc[-1:].fillna(0)

        predictions = {}
        confidence_scores = {}

        for timeframe_name, model in self.models.items():
            # Scale features
            X_scaled = self.scalers[timeframe_name].transform(current_row)

            # Make prediction
            pred_price = model.predict(X_scaled)[0]
            predictions[timeframe_name] = round(pred_price, 2)

            # Calculate confidence based on prediction interval
            # Using model's feature importance and recent prediction accuracy
            confidence = self._calculate_confidence(model, df, timeframe_name)
            confidence_scores[timeframe_name] = confidence

        # Technical analysis for context
        technical_signals = self._analyze_technical_signals(df)

        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(df, predictions)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            predictions, confidence_scores, technical_signals, df
        )

        return PredictionResult(
            ticker=ticker,
            current_price=df['close'].iloc[-1],
            predictions=predictions,
            confidence_scores=confidence_scores,
            technical_signals=technical_signals,
            risk_metrics=risk_metrics,
            recommendation=recommendation,
            analysis_timestamp=pd.Timestamp.now()
        )

    def _calculate_confidence(self, model, df, timeframe_name):
        """Calculate confidence score for prediction"""
        # Simplified confidence based on recent volatility and model performance
        recent_volatility = df['close'].pct_change().tail(20).std()

        # Lower confidence for higher volatility
        volatility_factor = max(0.5, 1 - (recent_volatility * 10))

        # You could also track model's recent prediction accuracy here
        base_confidence = 0.7  # This should come from backtesting

        return min(0.95, base_confidence * volatility_factor)

    def _analyze_technical_signals(self, df):
        """Analyze current technical indicators"""
        latest = df.iloc[-1]

        signals = {
            'trend': 'bullish' if latest['close'] > latest['sma_50'] else 'bearish',
            'rsi_signal': 'oversold' if latest['rsi'] < 30 else ('overbought' if latest['rsi'] > 70 else 'neutral'),
            'macd_crossover': latest['macd'] > latest['macds'],
            'volume_surge': latest['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5,
            'support_level': df['low'].rolling(20).min().iloc[-1],
            'resistance_level': df['high'].rolling(20).max().iloc[-1],
        }

        return signals

    def _calculate_risk_metrics(self, df, predictions):
        """Calculate risk metrics for the predictions"""
        current_price = df['close'].iloc[-1]

        returns = {}
        for timeframe, pred_price in predictions.items():
            returns[timeframe] = (pred_price - current_price) / current_price

        # Historical volatility
        hist_volatility = df['close'].pct_change().std() * np.sqrt(252)

        # Maximum drawdown in last 3 months
        rolling_max = df['close'].tail(63).expanding().max()
        drawdown = (df['close'].tail(63) - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            'expected_returns': returns,
            'historical_volatility': hist_volatility,
            'max_drawdown_3m': max_drawdown,
            'sharpe_estimate': returns.get('1_month', 0) / (hist_volatility / np.sqrt(12)) if hist_volatility > 0 else 0
        }

    def _generate_recommendation(self, predictions, confidence, signals, df):
        """Generate trading recommendation based on all factors"""
        current_price = df['close'].iloc[-1]

        # Calculate average expected return
        returns = [(pred - current_price) / current_price for pred in predictions.values()]
        avg_return = np.mean(returns)

        # Weight by confidence
        weighted_return = avg_return * np.mean(list(confidence.values()))

        if weighted_return > 0.03 and signals['trend'] == 'bullish':
            return "STRONG BUY - Positive outlook across timeframes with bullish trend"
        elif weighted_return > 0.01:
            return "BUY - Moderate positive outlook"
        elif weighted_return < -0.03 and signals['trend'] == 'bearish':
            return "SELL - Negative outlook with bearish trend"
        elif weighted_return < -0.01:
            return "WEAK SELL - Moderate negative outlook"
        else:
            return "HOLD - Mixed signals, await clearer direction"