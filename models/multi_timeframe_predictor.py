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
            '3_days': 3,
            '1_week': 5,
            '2_weeks': 10,
            '1_month': 22,
            '3_months': 66,  # Approximately 66 trading days in 3 months
        }

    def save_models(self, path):
        import joblib
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'timeframes': self.timeframes
        }
        joblib.dump(model_data, path)
        print(f"Models saved to {path}")

    def load_models(self, path):
        import joblib
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        model_data = joblib.load(path)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.timeframes = model_data['timeframes']
        print(f"Models loaded from {path}")

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
            valid_data = features.join(df_timeframe[['target', 'target_return', 'close']]).dropna()

            current_prices = valid_data['close'].copy()

            X = valid_data.drop(['target', 'target_return', 'close'], axis=1)
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
                random_state=42,
                early_stopping_rounds=50  # Add here instead
            )

            # Use time series split for validation
            split_point = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            # Store model and scaler
            self.models[timeframe_name] = model
            self.scalers[timeframe_name] = scaler

            # Calculate accuracy metrics
            predictions = model.predict(X_test)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            print(f"  MAPE: {mape:.2f}%")

            from models.validation.metrics import validate_predictions
            # Get the actual current prices from the test set
            current_prices_test = current_prices.iloc[split_point:].values
            metrics = validate_predictions(y_test, predictions, current_prices_test)
            print(f"  Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
            print(f"  Win Rate: {metrics['win_rate']:.1f}%")
            print(f"  Profit Factor: {metrics['profit_factor']:.2f}")

    def _engineer_features(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Create features optimized for specific prediction horizon"""
        features = pd.DataFrame(index=df.index)

        # REGRESSION CHANNEL FEATURES (PRIMARY)
        # Current position in channel
        features['sd_position'] = df['sd_position']
        features['sd_position_squared'] = df['sd_position'] ** 2  # Non-linear effects at extremes
        features['at_extreme'] = df['at_extreme']
        features['at_3sd'] = df['at_3sd']

        # Regression trend features
        features['regression_slope'] = df['regression_slope']
        features['regression_r2'] = df['regression_r2']
        features['trend_strength'] = df['trend_strength']
        features['above_regression'] = df['above_regression']

        # Distance features
        features['pct_from_regression'] = df['pct_from_regression']
        features['residual_normalized'] = df['residual'] / df['close']  # Normalized residual

        # Channel characteristics
        features['channel_width'] = df['channel_width']
        features['channel_width_expanding'] = df['channel_width_expanding']

        # Mean reversion signals
        features['days_above_2sd'] = df['days_above_2sd']
        features['days_below_2sd'] = df['days_below_2sd']
        features['sd_position_change'] = df['sd_position_change']

        # Historical behavior
        features['touches_3sd_20d'] = df['touches_3sd_20d']
        features['mean_reversion_20d'] = df['mean_reversion_20d']

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

        # Price momentum (adjusted for regression context)
        for lb in [1, 3, 5, 10, 20]:
            features[f'return_{lb}d'] = df['close'].pct_change(lb)
            # How SD position changed over period
            features[f'sd_change_{lb}d'] = df['sd_position'] - df['sd_position'].shift(lb)

        # RSI with context
        features['rsi'] = df['rsi']
        features['rsi_vs_sd'] = df['rsi'] - (df['sd_position'] * 10 + 50)  # RSI vs expected from SD

        # Volume at extremes
        features['volume_at_extreme'] = df['volume'] * df['at_extreme']
        features['volume_ratio'] = df['volume_ratio']

        # Interaction features
        features['sd_rsi_interaction'] = df['sd_position'] * df['rsi']
        features['sd_volume_interaction'] = df['sd_position'] * df['volume_ratio']

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
        """
        Enhanced confidence calculation with emphasis on SD position and context
        """
        # Base confidence from model
        base_confidence = 0.6

        # Get current SD position
        current_sd_position = abs(df['sd_position'].iloc[-1])

        # Higher confidence at extremes (mean reversion is reliable)
        if current_sd_position >= 2.5:
            sd_confidence_boost = 0.25  # Very high confidence
        elif current_sd_position >= 2.0:
            sd_confidence_boost = 0.20
        elif current_sd_position >= 1.5:
            sd_confidence_boost = 0.15
        elif current_sd_position >= 1.0:
            sd_confidence_boost = 0.10
        else:
            sd_confidence_boost = 0.05  # Lower confidence in neutral zone

        # Trend consistency factor
        r2 = df['regression_r2'].iloc[-1]
        trend_factor = r2 * 0.1  # Up to 10% boost for strong trends

        # Time at extreme (more time = higher confidence in reversion)
        days_extended = max(
            df['days_above_2sd'].iloc[-1] if 'days_above_2sd' in df.columns else 0,
            df['days_below_2sd'].iloc[-1] if 'days_below_2sd' in df.columns else 0
        )
        if days_extended >= 3:
            extreme_time_boost = 0.15
        elif days_extended >= 2:
            extreme_time_boost = 0.10
        else:
            extreme_time_boost = 0

        # Recent volatility (lower confidence if channel is expanding rapidly)
        if df['channel_width_expanding'].iloc[-1]:
            volatility_penalty = -0.05
        else:
            volatility_penalty = 0

        # Historical mean reversion success rate
        if 'mean_reversion_20d' in df.columns:
            reversion_rate = df['mean_reversion_20d'].iloc[-1] / 20
            reversion_boost = reversion_rate * 0.1
        else:
            reversion_boost = 0

        # Combine factors
        final_confidence = (base_confidence +
                            sd_confidence_boost +
                            trend_factor +
                            extreme_time_boost +
                            volatility_penalty +
                            reversion_boost)

        return min(0.95, max(0.30, final_confidence))

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
        """
        Generate trading recommendation based on regression channel position
        Returns: LONG, SHORT, or HOLD with reasoning
        """
        current_price = df['close'].iloc[-1]
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Key metrics
        sd_position = latest['sd_position']
        regression_line = latest['regression_line']
        days_above_2sd = latest.get('days_above_2sd', 0)
        days_below_2sd = latest.get('days_below_2sd', 0)

        # Price action analysis
        upper_wick = (latest['high'] - latest['close']) / latest['close']
        lower_wick = (latest['close'] - latest['low']) / latest['close']
        body_size = abs(latest['close'] - latest['open']) / latest['open']

        # Trend context
        regression_slope = latest['regression_slope']
        above_regression = latest['above_regression']

        # Volume context
        volume_surge = latest['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5

        # ============================================================================
        # DECISION LOGIC - Based on Standard Deviation Position
        # ============================================================================

        # 1. EXTREME OVERSOLD - Strong Long Signal
        if sd_position <= -2.5:
            if days_below_2sd >= 3:
                return "LONG - Extreme oversold (3+ days below -2σ), high probability mean reversion"
            else:
                return "LONG - Extreme oversold below -2.5σ, expect bounce to regression line"

        # 2. OVERSOLD ZONE (-2σ to -1.5σ) - Long Signal
        elif sd_position <= -2.0:
            # Check for signs of reversal
            if lower_wick > 0.015:  # 1.5% lower wick = buying pressure
                return "LONG - Oversold with buyer support, reversal likely"
            elif sd_position > prev['sd_position']:  # Starting to move up
                return "LONG - Oversold and starting mean reversion"
            else:
                return "LONG - Oversold below -2σ, favorable risk/reward for long"

        elif sd_position <= -1.5:
            if volume_surge and latest['close'] > latest['open']:  # Bullish volume spike
                return "LONG - Oversold with bullish volume, bounce expected"
            else:
                return "LONG - Below -1.5σ, good entry for mean reversion play"

        # 3. LOWER BAND (-1σ to -1.5σ) - Potential Long
        elif sd_position <= -1.0:
            # Near term predictions
            near_term_returns = [
                (predictions.get('3_days', current_price) - current_price) / current_price,
                (predictions.get('1_week', current_price) - current_price) / current_price
            ]
            avg_near_term = sum(near_term_returns) / len(near_term_returns)

            if avg_near_term > 0.02:  # 2%+ expected return
                return "LONG - Below -1σ with positive near-term outlook"
            else:
                return "HOLD - Near -1σ, wait for clearer signal or deeper pullback"

        # 4. NEUTRAL ZONE (-1σ to +1σ) - Hold or Trend Follow
        elif -1.0 < sd_position < 1.0:
            # Within 1 SD = HOLD unless strong trend

            # Check if price is crossing regression line
            if not above_regression and prev['above_regression']:  # Just crossed below
                return "SHORT - Broke below regression line, momentum shifting bearish"
            elif above_regression and not prev['above_regression']:  # Just crossed above
                return "LONG - Broke above regression line, momentum shifting bullish"

            # Check trend strength
            if abs(regression_slope) > latest['close'] * 0.001:  # Strong trend
                r2 = latest['regression_r2']
                if r2 > 0.8 and regression_slope > 0:  # Strong uptrend
                    if above_regression:
                        return "LONG - Strong uptrend with price above regression, ride the trend"
                    else:
                        return "HOLD - Uptrend but price below regression, wait for confirmation"
                elif r2 > 0.8 and regression_slope < 0:  # Strong downtrend
                    if not above_regression:
                        return "SHORT - Strong downtrend with price below regression"
                    else:
                        return "HOLD - Downtrend but price above regression, wait for confirmation"

            return "HOLD - Within 1σ of regression line, no clear edge"

        # 5. UPPER BAND (+1σ to +1.5σ) - Potential Short
        elif sd_position >= 1.0 and sd_position < 1.5:
            # Near term predictions
            near_term_returns = [
                (predictions.get('3_days', current_price) - current_price) / current_price,
                (predictions.get('1_week', current_price) - current_price) / current_price
            ]
            avg_near_term = sum(near_term_returns) / len(near_term_returns)

            if avg_near_term < -0.02:  # -2% expected return
                return "SHORT - Above +1σ with negative near-term outlook"
            elif upper_wick > 0.015:  # Rejection
                return "SHORT - Above +1σ with rejection signal, pullback expected"
            else:
                return "HOLD - Above +1σ, extended but no clear reversal signal yet"

        # 6. OVERBOUGHT ZONE (+1.5σ to +2σ) - Short Signal
        elif sd_position >= 1.5 and sd_position < 2.0:
            if upper_wick > 0.015:  # 1.5% upper wick = selling pressure
                return "SHORT - Overbought with seller pressure, reversal likely"
            elif sd_position < prev['sd_position']:  # Starting to move down
                return "SHORT - Overbought and starting mean reversion"
            else:
                return "SHORT - Overbought above +1.5σ, favorable risk/reward for short"

        elif sd_position >= 2.0 and sd_position < 2.5:
            # Check for signs of exhaustion
            if volume_surge and latest['close'] < latest['open']:  # Bearish volume spike
                return "SHORT - Overbought with bearish volume, pullback expected"
            else:
                return "SHORT - Above +2σ, good entry for mean reversion short"

        # 7. EXTREME OVERBOUGHT - Strong Short Signal
        elif sd_position >= 2.5:
            if days_above_2sd >= 3:
                return "SHORT - Extreme overbought (3+ days above +2σ), high probability mean reversion"
            else:
                return "SHORT - Extreme overbought above +2.5σ, expect pullback to regression line"

        # Fallback
        return "HOLD - Unable to determine clear signal"
