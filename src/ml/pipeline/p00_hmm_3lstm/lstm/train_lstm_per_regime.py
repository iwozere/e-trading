"""
Trains LSTM models for different market regimes based on pre-processed CSV data.

This script iterates through CSV files in the 'results' directory. For each file,
it trains a separate LSTM model for each market regime (bullish, bearish, sideways).
The trained model weights are saved as '.pt' files in 'src/ml/lstm/model/',
and the corresponding hyperparameters are saved as '.json' files in the 'results'
directory.

The script assumes the input CSVs contain feature columns and a 'regime' column.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3] # Go up 3 levels from 'src/ml/hmm'
sys.path.append(str(PROJECT_ROOT))

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from glob import glob

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# ---- Config ----
RESULTS_DIR = "results"
MODEL_DIR = "src/ml/lstm/model"
FEATURE_COLS = ['log_return', 'volatility', 'rsi', 'macd', 'boll_width']
TARGET_COL = 'log_return'
WINDOW = 20         # Size of the look-back window for sequences
EPOCHS = 30         # Number of training epochs
BATCH_SIZE = 32     # Batch size for training
LR = 1e-3           # Learning rate for the Adam optimizer
HIDDEN_DIM = 64     # Number of hidden units in the LSTM layers
NUM_LAYERS = 2      # Number of LSTM layers
REGIME_NAMES = {0: "bearish", 1: "sideways", 2: "bullish"}

# Create necessary directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Model ----
class LSTMModel(nn.Module):
    """
    A simple Long Short-Term Memory (LSTM) network for time series prediction.

    The model consists of an LSTM layer followed by a fully connected (Linear)
    layer to produce a single output value.

    Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of features in the hidden state h.
        num_layers (int): The number of recurrent layers.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size,).
        """
        # LSTM returns output, (hidden_state, cell_state)
        # We only need the final hidden state
        _, (h_n, _) = self.lstm(x)
        # h_n is of shape (num_layers, batch_size, hidden_dim)
        # We take the hidden state of the last layer
        return self.fc(h_n[-1]).squeeze()

# ---- Data prep ----
def create_sequences(data, target_idx, window):
    """
    Creates sequences and corresponding labels from time series data.

    Args:
        data (np.ndarray): The input data array of shape (n_samples, n_features).
        target_idx (int): The index of the target column in the data array.
        window (int): The length of each sequence (look-back period).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the input sequences (X)
                                       and their corresponding target values (y).
    """
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window][target_idx])
    return np.array(X), np.array(y)

def prepare_data(csv_path, regime_id, feature_cols, window):
    """
    Loads, filters by regime, scales, and prepares data for the LSTM model.

    Args:
        csv_path (str): The path to the input CSV file.
        regime_id (int): The integer ID of the market regime to filter by.
        feature_cols (list[str]): A list of column names to be used as features.
        window (int): The sequence length for the time series windows.

    Returns:
        tuple: A tuple containing the prepared sequences (X) and targets (y).
               Returns (None, None) if there is not enough data for the regime.
    """
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df[df['regime'] == regime_id].dropna()

    if df.empty or len(df) <= window:
        return None, None

    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols].values)

    target_idx = feature_cols.index(TARGET_COL)
    X, y = create_sequences(features, target_idx, window)
    return X, y

# ---- Training ----
def train_lstm(X, y, input_dim, save_path, hidden_dim, num_layers):
    """
    Trains an LSTM model and saves its state dictionary.

    Args:
        X (np.ndarray): The input feature sequences.
        y (np.ndarray): The target values.
        input_dim (int): The number of input features.
        save_path (str): The path where the trained model's state_dict will be saved.
        hidden_dim (int): The number of hidden units for the LSTM.
        num_layers (int): The number of layers for the LSTM.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_dim, hidden_dim, num_layers).to(device)

    dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    final_loss = 0
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        final_loss = total_loss / len(loader)
        # Log every 5 epochs to reduce verbosity
        if (epoch + 1) % 5 == 0:
            _logger.info("Epoch %s/%s - Loss: %s")

    _logger.info("Final training loss: %s")
    torch.save(model.state_dict(), save_path)
    _logger.info("✅ Saved model: %s")

# ---- Runner ----
if __name__ == "__main__":
    # Find all CSV files in the results directory
    csv_files = glob(os.path.join(RESULTS_DIR, "*.csv"))

    if not csv_files:
        _logger.error("❌ No CSV files found in 'results/'")
        exit()

    # Process each CSV file found
    for csv_path in csv_files:
        # Extract base filename for naming output files
        filename = os.path.splitext(os.path.basename(csv_path))[0]
        _logger.info("\n📂 Processing: %s")

        # Train a separate model for each defined regime
        for regime_id, regime_name in REGIME_NAMES.items():
            _logger.info("--- Regime: %s (%s) ---")

            # Prepare data for the current regime
            X, y = prepare_data(csv_path, regime_id, FEATURE_COLS, window=WINDOW)

            if X is None or len(X) == 0:
                _logger.warning("⚠️ Not enough data for regime '%s', skipping.")
                continue

            # Train the model
            model_path = f"{MODEL_DIR}/{filename.replace("HMM", "LSTM")}_{regime_name}.pt"
            train_lstm(
                X, y,
                input_dim=X.shape[2],
                save_path=model_path,
                hidden_dim=HIDDEN_DIM,
                num_layers=NUM_LAYERS
            )

            # ---- Save parameters to a JSON file ----
            params = {
                "model_type": "LSTM",
                "model_path": model_path,
                "data_source": csv_path,
                "symbol_info": filename,
                "regime": regime_name,
                "hyperparameters": {
                    "feature_cols": FEATURE_COLS,
                    "target_col": TARGET_COL,
                    "window": WINDOW,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LR,
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS
                }
            }

            # Construct the JSON filename as requested: LSTM_{symbol_info}_{regime}.json
            # e.g., LSTM_BTCUSDT_4h_20220101_20250707_bullish.json
            json_filename = f"LSTM_{filename}_{regime_name}.json"
            json_path = os.path.join(RESULTS_DIR, json_filename)

            with open(json_path, 'w') as f:
                json.dump(params, f, indent=4)

            _logger.info("📄 Saved params to: %s")
