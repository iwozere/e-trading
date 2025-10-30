"""
HybridNNStrategy
---------------

A strategy that uses a pretrained CNN+LSTM+Bahdanau Attention (PyTorch) model and an XGBoost model in sequence for trading signal generation.

Use HybridNNStrategy for research, batch inference, or as a callable ML component.
Use HybridNNBacktraderStrategy for live trading or backtesting in Backtrader.

Workflow:
- For each new data window:
    1. Compute OHLCV window and technical indicators (using TA-Lib)
    2. Pass OHLCV window through CNN+LSTM+Attention to get feature vector
    3. Concatenate with technical indicators
    4. Pass combined features to XGBoost for up/down prediction
    5. Generate trading signals accordingly

Usage Example:
--------------
from src.strategy.hybrid_nn_strategy import HybridNNStrategy

strategy = HybridNNStrategy(
    cnn_lstm_path='cnn_lstm_attention.pt',
    xgb_path='xgboost_model.json',
    window_size=100
)
signal = strategy.predict_signal(ohlcv_df)
print('Signal:', signal)  # 'buy', 'sell', or 'hold'
"""

from src.strategy.future.hybrid_nn_core import HybridNNCore

class HybridNNStrategy(HybridNNCore):
    def __init__(self, cnn_lstm_path, xgb_path, window_size=100, device=None):
        super().__init__(cnn_lstm_path, xgb_path, window_size, device)
    # All logic is inherited from HybridNNCore
