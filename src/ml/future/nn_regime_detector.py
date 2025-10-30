"""
nn_regime_detector.py
--------------------

A PyTorch-based neural network for market regime detection.
Implements an LSTM classifier for bull/bear/sideways regimes.

Features:
- LSTM-based sequence classifier
- fit, predict, save, and load methods
- GPU support if available

Uses LSTM-based deep learning architecture
Supervised learning approach - requires pre-labeled regime data for training
Works with multiple engineered features (returns, volatility, RSI, MACD, etc.)
Learns complex non-linear patterns through neural networks
More complex with multiple hyperparameters (hidden size, layers, epochs, etc.)



Example usage:
--------------
from src.ml.nn_regime_detector import NNRegimeDetector
import pandas as pd

df = pd.read_csv('your_feature_data.csv')
features = df[['return', 'volatility', 'rsi', 'macd']].values
labels = df['regime'].values  # 0=bull, 1=bear, 2=sideways

model = NNRegimeDetector(input_size=4, n_regimes=3)
model.fit(features, labels, epochs=10)
preds = model.predict(features)
model.save('regime_model.pt')
model.load('regime_model.pt')
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTMRegimeClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, n_regimes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_regimes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last output
        out = self.fc(out)
        return out

class NNRegimeDetector:
    def __init__(self, input_size, n_regimes=3, hidden_size=32, num_layers=1, device=None):
        self.input_size = input_size
        self.n_regimes = n_regimes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMRegimeClassifier(input_size, hidden_size, num_layers, n_regimes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _prepare_sequences(self, X, y=None, seq_len=20):
        # X: (n_samples, n_features), y: (n_samples,)
        n_samples = X.shape[0]
        X_seq = []
        y_seq = []
        for i in range(seq_len, n_samples):
            X_seq.append(X[i-seq_len:i, :])
            if y is not None:
                y_seq.append(y[i])
        X_seq = np.stack(X_seq)
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        return X_seq

    def fit(self, X, y, epochs=10, batch_size=32, seq_len=20, verbose=True):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        X_seq, y_seq = self._prepare_sequences(X, y, seq_len)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(dataset)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    def predict(self, X, seq_len=20, batch_size=128):
        X = np.asarray(X, dtype=np.float32)
        X_seq = self._prepare_sequences(X, None, seq_len)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_seq), batch_size):
                xb = torch.tensor(X_seq[i:i+batch_size], dtype=torch.float32).to(self.device)
                out = self.model(xb)
                pred = torch.argmax(out, dim=1).cpu().numpy()
                preds.append(pred)
        return np.concatenate(preds)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
