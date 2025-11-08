# monitoring/alert_system.py
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
from dataclasses import dataclass
import time
from enum import Enum


class AlertType(Enum):
    BREAKOUT = "Price Breakout"
    RSI_OVERSOLD = "RSI Oversold"
    RSI_OVERBOUGHT = "RSI Overbought"
    VOLUME_SPIKE = "Volume Spike"
    MACD_CROSSOVER = "MACD Crossover"
    SUPPORT_BOUNCE = "Support Bounce"
    RESISTANCE_BREAK = "Resistance Break"
    PREDICTION_OPPORTUNITY = "ML Prediction Signal"
    SD_3_TOUCH = "3 Sigma Touch"
    SD_2_REVERSAL = "2 Sigma Reversal"
    REGRESSION_BREAK = "Regression Line Break"
    CHANNEL_SQUEEZE = "Volatility Squeeze"
    EXTREME_EXHAUSTION = "Extreme Exhaustion"


@dataclass
class Alert:
    ticker: str
    alert_type: AlertType
    message: str
    current_price: float
    target_price: float
    confidence: float
    timestamp: datetime
    metadata: Dict


class AlertSystem:
    def __init__(self, data_fetcher, predictor):
        self.data_fetcher = data_fetcher
        self.predictor = predictor
        self.alerts = []
        self.watchlist = []
        self.alert_conditions = self._default_alert_conditions()

    def _default_alert_conditions(self):
        """Define default alert conditions"""
        return {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_spike_multiplier': 2.0,
            'breakout_periods': 20,
            'min_prediction_return': 0.02,  # 2% minimum expected return
            'min_confidence': 0.65
        }

    def add_to_watchlist(self, tickers: List[str]):
        """Add tickers to monitoring watchlist"""
        self.watchlist.extend(tickers)
        self.watchlist = list(set(self.watchlist))  # Remove duplicates

    def scan_for_alerts(self):
        """Scan all watchlist items for alert conditions"""
        new_alerts = []

        for ticker in self.watchlist:
            try:
                # Fetch recent data
                df = self.data_fetcher.get_bars(
                    ticker,
                    start_date=datetime.now() - timedelta(days=100),
                    end_date=datetime.now()
                )

                if df.empty:
                    continue

                # Add technical indicators
                df = self.data_fetcher.add_technical_indicators(df)

                # Check various alert conditions
                alerts = []

                # 1. RSI Alerts
                alerts.extend(self._check_rsi_alerts(ticker, df))

                # 2. Volume Alerts
                alerts.extend(self._check_volume_alerts(ticker, df))

                # 3. Price Breakout Alerts
                alerts.extend(self._check_breakout_alerts(ticker, df))

                # 4. MACD Crossover
                alerts.extend(self._check_macd_alerts(ticker, df))

                # 5. ML Prediction Alerts
                alerts.extend(self._check_prediction_alerts(ticker, df))

                new_alerts.extend(alerts)

            except Exception as e:
                print(f"Error scanning {ticker}: {e}")

        self.alerts.extend(new_alerts)
        return new_alerts

    def _check_regression_alerts(self, ticker, df):
        """Check for regression channel-based alerts"""
        alerts = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 1. Three Sigma Touch (RARE - High Priority)
        if abs(latest['sd_position']) >= 2.8:
            direction = "overbought" if latest['sd_position'] > 0 else "oversold"
            target = latest['regression_line']

            alerts.append(Alert(
                ticker=ticker,
                alert_type=AlertType.SD_3_TOUCH,
                message=f"{ticker} at {latest['sd_position']:.1f} SD ({direction}) - RARE EVENT",
                current_price=latest['close'],
                target_price=target,
                confidence=0.85,  # High confidence in mean reversion from 3SD
                timestamp=datetime.now(),
                metadata={
                    'sd_position': latest['sd_position'],
                    'regression_line': latest['regression_line'],
                    'channel_width': latest['channel_width'],
                    'regression_slope': latest['regression_slope']
                }
            ))

        # 2. Two Sigma Reversal Signal
        if prev['sd_position'] >= 2.0 and latest['sd_position'] < prev['sd_position']:
            # Starting to revert from upper 2SD
            alerts.append(Alert(
                ticker=ticker,
                alert_type=AlertType.SD_2_REVERSAL,
                message=f"{ticker} reverting from +2SD (potential short)",
                current_price=latest['close'],
                target_price=latest['regression_line'],
                confidence=0.7,
                timestamp=datetime.now(),
                metadata={'sd_position': latest['sd_position']}
            ))

        elif prev['sd_position'] <= -2.0 and latest['sd_position'] > prev['sd_position']:
            # Starting to revert from lower 2SD
            alerts.append(Alert(
                ticker=ticker,
                alert_type=AlertType.SD_2_REVERSAL,
                message=f"{ticker} bouncing from -2SD (potential long)",
                current_price=latest['close'],
                target_price=latest['regression_line'],
                confidence=0.7,
                timestamp=datetime.now(),
                metadata={'sd_position': latest['sd_position']}
            ))

        # 3. Regression Line Cross
        if prev['above_regression'] != latest['above_regression']:
            direction = "above" if latest['above_regression'] else "below"
            alerts.append(Alert(
                ticker=ticker,
                alert_type=AlertType.REGRESSION_BREAK,
                message=f"{ticker} crossed {direction} regression line",
                current_price=latest['close'],
                target_price=latest[f'{"upper" if direction == "above" else "lower"}_2sd'],
                confidence=0.6,
                timestamp=datetime.now(),
                metadata={
                    'regression_line': latest['regression_line'],
                    'trend_slope': latest['regression_slope']
                }
            ))

        # 4. Channel Squeeze (Volatility Contraction)
        recent_width = df['channel_width'].tail(20)
        if latest['channel_width'] < recent_width.quantile(0.2):
            alerts.append(Alert(
                ticker=ticker,
                alert_type=AlertType.CHANNEL_SQUEEZE,
                message=f"{ticker} channel squeeze - potential breakout setup",
                current_price=latest['close'],
                target_price=latest['close'] * 1.03,  # 3% breakout target
                confidence=0.5,
                timestamp=datetime.now(),
                metadata={'channel_width': latest['channel_width']}
            ))

        # 5. Extreme Exhaustion (3+ days beyond 2SD)
        if latest['days_above_2sd'] >= 3 or latest['days_below_2sd'] >= 3:
            direction = "overbought" if latest['days_above_2sd'] >= 3 else "oversold"
            alerts.append(Alert(
                ticker=ticker,
                alert_type=AlertType.EXTREME_EXHAUSTION,
                message=f"{ticker} exhaustion - {direction} for {max(latest['days_above_2sd'], latest['days_below_2sd'])} days",
                current_price=latest['close'],
                target_price=latest['regression_line'],
                confidence=0.75,
                timestamp=datetime.now(),
                metadata={
                    'days_extended': max(latest['days_above_2sd'], latest['days_below_2sd']),
                    'sd_position': latest['sd_position']
                }
            ))

        return alerts

    def _check_rsi_alerts(self, ticker, df):
        """Check RSI conditions"""
        alerts = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Oversold
        if latest['rsi'] < self.alert_conditions['rsi_oversold'] and prev['rsi'] >= self.alert_conditions[
            'rsi_oversold']:
            alerts.append(Alert(
                ticker=ticker,
                alert_type=AlertType.RSI_OVERSOLD,
                message=f"{ticker} RSI entered oversold territory ({latest['rsi']:.1f})",
                current_price=latest['close'],
                target_price=latest['close'] * 1.02,  # 2% bounce target
                confidence=0.6,
                timestamp=datetime.now(),
                metadata={'rsi': latest['rsi']}
            ))

        # Overbought
        if latest['rsi'] > self.alert_conditions['rsi_overbought'] and prev['rsi'] <= self.alert_conditions[
            'rsi_overbought']:
            alerts.append(Alert(
                ticker=ticker,
                alert_type=AlertType.RSI_OVERBOUGHT,
                message=f"{ticker} RSI entered overbought territory ({latest['rsi']:.1f})",
                current_price=latest['close'],
                target_price=latest['close'] * 0.98,  # 2% pullback target
                confidence=0.6,
                timestamp=datetime.now(),
                metadata={'rsi': latest['rsi']}
            ))

        return alerts

    def _check_volume_alerts(self, ticker, df):
        """Check for volume spikes"""
        alerts = []
        latest = df.iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]

        if latest['volume'] > avg_volume * self.alert_conditions['volume_spike_multiplier']:
            # Determine direction based on price action
            price_change = (latest['close'] - latest['open']) / latest['open']

            alerts.append(Alert(
                ticker=ticker,
                alert_type=AlertType.VOLUME_SPIKE,
                message=f"{ticker} volume spike detected ({latest['volume'] / avg_volume:.1f}x average)",
                current_price=latest['close'],
                target_price=latest['close'] * (1 + price_change),  # Continue in same direction
                confidence=0.5,
                timestamp=datetime.now(),
                metadata={
                    'volume': latest['volume'],
                    'avg_volume': avg_volume,
                    'price_change': price_change
                }
            ))

        return alerts

    def _check_breakout_alerts(self, ticker, df):
        """Check for price breakouts"""
        alerts = []
        latest = df.iloc[-1]

        # Resistance breakout
        resistance = df['high'].rolling(self.alert_conditions['breakout_periods']).max().iloc[-2]
        if latest['close'] > resistance and df.iloc[-2]['close'] <= resistance:
            alerts.append(Alert(
                ticker=ticker,
                alert_type=AlertType.RESISTANCE_BREAK,
                message=f"{ticker} broke resistance at ${resistance:.2f}",
                current_price=latest['close'],
                target_price=latest['close'] * 1.03,  # 3% continuation target
                confidence=0.7,
                timestamp=datetime.now(),
                metadata={'resistance_level': resistance}
            ))

        # Support bounce
        support = df['low'].rolling(self.alert_conditions['breakout_periods']).min().iloc[-2]
        if latest['low'] <= support and latest['close'] > support:
            alerts.append(Alert(
                ticker=ticker,
                alert_type=AlertType.SUPPORT_BOUNCE,
                message=f"{ticker} bounced off support at ${support:.2f}",
                current_price=latest['close'],
                target_price=latest['close'] * 1.02,  # 2% bounce target
                confidence=0.65,
                timestamp=datetime.now(),
                metadata={'support_level': support}
            ))

        return alerts

    def _check_macd_alerts(self, ticker, df):
        """Check for MACD crossovers"""
        alerts = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Bullish crossover
        if latest['macd'] > latest['macds'] and prev['macd'] <= prev['macds']:
            alerts.append(Alert(
                ticker=ticker,
                alert_type=AlertType.MACD_CROSSOVER,
                message=f"{ticker} MACD bullish crossover",
                current_price=latest['close'],
                target_price=latest['close'] * 1.025,
                confidence=0.6,
                timestamp=datetime.now(),
                metadata={'macd': latest['macd'], 'signal': latest['macds']}
            ))

        return alerts

    def _check_prediction_alerts(self, ticker, df):
        """Check ML model predictions"""
        alerts = []

        # Get predictions
        prediction_result = self.predictor.predict(ticker, df)

        # Check if any timeframe shows strong opportunity
        current_price = prediction_result.current_price

        for timeframe, predicted_price in prediction_result.predictions.items():
            expected_return = (predicted_price - current_price) / current_price
            confidence = prediction_result.confidence_scores[timeframe]

            if (expected_return > self.alert_conditions['min_prediction_return'] and
                    confidence > self.alert_conditions['min_confidence']):
                alerts.append(Alert(
                    ticker=ticker,
                    alert_type=AlertType.PREDICTION_OPPORTUNITY,
                    message=f"{ticker} ML model predicts {expected_return:.1%} return in {timeframe}",
                    current_price=current_price,
                    target_price=predicted_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    metadata={
                        'timeframe': timeframe,
                        'expected_return': expected_return,
                        'technical_signals': prediction_result.technical_signals,
                        'recommendation': prediction_result.recommendation
                    }
                ))

        return alerts