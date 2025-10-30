"""
HybridNNBacktraderStrategy
-------------------------
Use HybridNNStrategy for research, batch inference, or as a callable ML component.
Use HybridNNBacktraderStrategy for live trading or backtesting in Backtrader.

A Backtrader-compatible strategy that uses a pretrained CNN+LSTM+Bahdanau Attention (PyTorch) model and an XGBoost model in sequence for trading signal generation.

Usage Example:
--------------
from src.strategy.hybrid_nn_backtrader import HybridNNBacktraderStrategy
import backtrader as bt

cerebro = bt.Cerebro()
cerebro.addstrategy(HybridNNBacktraderStrategy,
    cnn_lstm_path='cnn_lstm_attention.pt',
    xgb_path='xgboost_model.json',
    window_size=100
)
# Add data, set broker, etc.
cerebro.run()
"""

import backtrader as bt
import pandas as pd
from src.strategy.future.hybrid_nn_core import HybridNNCore

class HybridNNBacktraderStrategy(bt.Strategy):
    params = (
        ('cnn_lstm_path', 'cnn_lstm_attention.pt'),
        ('xgb_path', 'xgboost_model.json'),
        ('window_size', 100),
        ('device', None),
    )

    def __init__(self):
        self.window_size = self.p.window_size
        self.core = HybridNNCore(self.p.cnn_lstm_path, self.p.xgb_path, self.p.window_size, self.p.device)
        self.ohlcv_buffer = []
        self.datafields = ['open', 'high', 'low', 'close', 'volume']

    def next(self):
        bar = [float(getattr(self.data, f)[0]) for f in self.datafields]
        self.ohlcv_buffer.append(bar)
        if len(self.ohlcv_buffer) < self.window_size:
            return
        if len(self.ohlcv_buffer) > self.window_size:
            self.ohlcv_buffer.pop(0)
        ohlcv_df = pd.DataFrame(self.ohlcv_buffer, columns=self.datafields)
        try:
            signal = self.core.predict_signal(ohlcv_df)
        except Exception as e:
            print(f"[HybridNNBacktraderStrategy] Prediction error: {e}")
            return
        if not self.position:
            if signal == 'buy':
                self.buy()
        else:
            if signal == 'sell':
                self.sell()
