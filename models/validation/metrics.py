import numpy as np

def validate_predictions(y_true, y_pred, prices):
    """Add 3 quick metrics to existing validation"""

    # Already have: MAPE (from current code)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # NEW Metric 1: Directional Accuracy
    actual_direction = np.sign(y_true - prices)
    pred_direction = np.sign(y_pred - prices)
    directional_accuracy = (actual_direction == pred_direction).mean() * 100

    # NEW Metric 2: Win Rate (when predicted up, did it go up?)
    predicted_up = pred_direction > 0
    wins = (actual_direction[predicted_up] > 0).sum()
    win_rate = wins / predicted_up.sum() * 100 if predicted_up.sum() > 0 else 0

    # NEW Metric 3: Profit Factor
    returns = (y_true - prices) / prices
    pred_returns = (y_pred - prices) / prices
    profitable_trades = pred_returns > 0.01  # 1% threshold
    profits = returns[profitable_trades & (returns > 0)].sum()
    losses = abs(returns[profitable_trades & (returns < 0)].sum())
    profit_factor = profits / losses if losses > 0 else 0

    return {
        'mape': mape,
        'directional_accuracy': directional_accuracy,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }


def validate_by_regime(df, predictions, actuals):
    """Validate performance in different market conditions"""

    # Extreme SD positions (your edge!)
    extreme_mask = abs(df['sd_position']) > 2
    normal_mask = abs(df['sd_position']) <= 2

    return {
        'extreme_sd': validate_predictions(
            actuals[extreme_mask],
            predictions[extreme_mask],
            df['close'][extreme_mask]
        ),
        'normal': validate_predictions(
            actuals[normal_mask],
            predictions[normal_mask],
            df['close'][normal_mask]
        )
    }


def validate_by_stock(predictions_dict):
    """Show which stocks perform best"""
    results = {}
    for stock, preds in predictions_dict.items():
        results[stock] = validate_predictions(...)
    return results