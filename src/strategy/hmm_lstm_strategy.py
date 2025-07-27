"""
regime_lstm_strategy.py

Backtrader strategy that combines HMM regime detection with regime-specific LSTM models
to predict future returns. Each regime (bearish, sideways, bullish) is handled by a
dedicated PyTorch LSTM trained only on data from that regime.

Requirements:
- PyTorch
- Backtrader
- scikit-learn
- NumPy
- Pandas

Usage:
    python regime_lstm_strategy.py --csv results/LTCUSDT_1h_20220101_20250707.csv
"""

import backtrader as bt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import argparse

# Mapping regime integers to names
REGIME_NAMES = {0: "bearish", 1: "sideways", 2: "bullish"}

# LSTM settings
WINDOW = 20
FEATURE_COLS = ['log_return', 'volatility', 'rsi', 'macd', 'boll_width']


class LSTMModel(nn.Module):
    """
    Simple LSTM regression model that predicts the next log return
    based on a window of past features.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze()


class RegimeLSTMStrategy(bt.Strategy):
    """
    Backtrader strategy that uses HMM regime labels and regime-specific LSTM
    predictions to decide when to buy or sell.
    """
    def __init__(self):
        self.buffer = deque(maxlen=WINDOW)  # Sliding window of features
        self.models = {}  # Loaded LSTM models per regime
        self.scaler = StandardScaler()

    def load_model(self, regime_name, filename_prefix):
        """
        Load the PyTorch LSTM model corresponding to a regime.
        """
        model_path = Path(f"models/lstm/{filename_prefix}_{regime_name}.pt")
        model = LSTMModel(input_dim=len(FEATURE_COLS))
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model

    def next(self):
        """
        Called on each bar. Predicts next return using the regime-specific LSTM model
        and makes trading decisions based on the prediction.
        """
        # Get the most recent data values for this bar
        row = {col: self.datas[0].lines.__getattribute__(col)[0] for col in FEATURE_COLS + ['regime']}
        self.buffer.append([row[col] for col in FEATURE_COLS])

        if len(self.buffer) < WINDOW:
            return  # Wait for full window

        regime_id = int(row['regime'])
        regime_name = REGIME_NAMES.get(regime_id, "unknown")

        filename_prefix = self.datas[0]._name
        if regime_name not in self.models:
            self.models[regime_name] = self.load_model(regime_name, filename_prefix)

        model = self.models[regime_name]
        input_tensor = torch.tensor([self.buffer], dtype=torch.float32)
        prediction = model(input_tensor).item()

        # Basic trading logic: long if prediction > threshold, close if prediction < -threshold
        if prediction > 0.001 and not self.position:
            self.buy()
        elif prediction < -0.001 and self.position:
            self.close()


def run_backtest(csv_path):
    """
    Loads a regime-labeled CSV, prepares the data for Backtrader,
    and runs the regime-aware LSTM strategy.
    """
    filename_prefix = Path(csv_path).stem
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])

    # Normalize features before feeding into strategy
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    df.set_index('timestamp', inplace=True)

    dfbt = bt.feeds.PandasData(
        dataname=df,
        fromdate=df.index[0],
        todate=df.index[-1],
        timeframe=bt.TimeFrame.Minutes,
        name=filename_prefix
    )

    cerebro = bt.Cerebro()
    cerebro.adddata(dfbt)
    cerebro.addstrategy(RegimeLSTMStrategy)
    cerebro.broker.setcash(10000)
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)
    cerebro.run()
    cerebro.plot()


def main():
    """
    CLI entry point to launch a backtest from a given regime-labeled CSV.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        required=False,
        default="results/LTCUSDT_1h_20220101_20250707.csv",
        help="Path to regime-labeled CSV file"
    )
    args = parser.parse_args()
    run_backtest(args.csv)


if __name__ == "__main__":
    main()
