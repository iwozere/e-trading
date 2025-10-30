"""
HybridNNCore
------------
A shared base class for hybrid neural network + XGBoost trading models.
Encapsulates model loading, feature extraction, and signal prediction logic.

Use HybridNNStrategy for research, batch inference, or as a callable ML component.
Use HybridNNBacktraderStrategy for live trading or backtesting in Backtrader.

Usage Example:
--------------
from src.strategy.hybrid_nn_core import HybridNNCore
core = HybridNNCore('cnn_lstm_attention.pt', 'xgboost_model.json', window_size=100)
signal = core.predict_signal(ohlcv_df)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import talib
import xgboost as xgb

# --- Bahdanau Attention ---
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
    def forward(self, encoder_outputs, hidden):
        hidden = hidden.unsqueeze(1)
        score = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden))
        attention_weights = torch.softmax(self.V(score), dim=1)
        context = (attention_weights * encoder_outputs).sum(dim=1)
        return context, attention_weights.squeeze(-1)

# --- CNN+LSTM+Attention Model ---
class HybridModel(nn.Module):
    def __init__(self, cnn_params, lstm_params, tech_feat_dim, out_dim=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(5, cnn_params['filters'], cnn_params['kernel_size']),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_params['filters'],
            hidden_size=lstm_params['hidden_size'],
            num_layers=lstm_params['num_layers'],
            batch_first=True,
            dropout=lstm_params['dropout'] if lstm_params['num_layers'] > 1 else 0.0
        )
        self.attn = BahdanauAttention(lstm_params['hidden_size'])
        self.fc = nn.Linear(lstm_params['hidden_size'], out_dim)
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, _) = self.lstm(x)
        context, attn_weights = self.attn(lstm_out, h_n[-1])
        out = self.fc(context)
        return out, context, attn_weights

class HybridNNCore:
    def __init__(self, cnn_lstm_path, xgb_path, window_size=100, device=None):
        self.window_size = window_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Model params should match those used in training
        cnn_params = {'filters': 64, 'kernel_size': 5}  # Update as needed
        lstm_params = {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3}
        self.model = HybridModel(cnn_params, lstm_params, tech_feat_dim=10).to(self.device)
        self.model.load_state_dict(torch.load(cnn_lstm_path, map_location=self.device))
        self.model.eval()
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(xgb_path)
        self.datafields = ['open', 'high', 'low', 'close', 'volume']

    def compute_tech_indicators(self, df):
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_ = df['open'].values
        volume = df['volume'].values
        rsi = talib.RSI(close, timeperiod=14)[-1]
        atr = talib.ATR(high, low, close, timeperiod=14)[-1]
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        macd = macd[-1]
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        bb_bbm = middle[-1]
        bb_bbh = upper[-1]
        bb_bbl = lower[-1]
        nine_high = pd.Series(high).rolling(window=9).max().iloc[-1]
        nine_low = pd.Series(low).rolling(window=9).min().iloc[-1]
        tenkan_sen = (nine_high + nine_low) / 2
        period26_high = pd.Series(high).rolling(window=26).max().iloc[-1]
        period26_low = pd.Series(low).rolling(window=26).min().iloc[-1]
        kijun_sen = (period26_high + period26_low) / 2
        ichimoku_a = ((tenkan_sen + kijun_sen) / 2)
        period52_high = pd.Series(high).rolling(window=52).max().iloc[-1]
        period52_low = pd.Series(low).rolling(window=52).min().iloc[-1]
        ichimoku_b = ((period52_high + period52_low) / 2)
        obv = talib.OBV(close, volume)[-1]
        slowk, slowd = talib.STOCH(high, low, close)
        stoch = slowk[-1]
        tech = np.array([rsi, atr, macd, bb_bbm, bb_bbh, bb_bbl, ichimoku_a, ichimoku_b, obv, stoch], dtype=np.float32)
        return tech

    def predict_signal(self, ohlcv_df):
        # ohlcv_df: DataFrame with columns ['open','high','low','close','volume'], length=window_size
        assert len(ohlcv_df) == self.window_size
        ohlcv = ohlcv_df[self.datafields].values.T  # (5, window_size)
        ohlcv_tensor = torch.tensor(ohlcv[None, ...], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, lstm_features, _ = self.model(ohlcv_tensor)
        tech = self.compute_tech_indicators(ohlcv_df)
        features = np.concatenate([lstm_features.cpu().numpy().flatten(), tech])
        pred = self.xgb.predict(features.reshape(1, -1))[0]
        if pred == 1:
            return 'buy'
        elif pred == 0:
            return 'sell'
        else:
            return 'hold' 
