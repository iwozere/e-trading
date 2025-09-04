"""
Hybrid CNN+LSTM+Attention Training Script
----------------------------------------

- Loads/generates OHLCV data
- Computes technical indicators for each window using TA-Lib
- Defines CNN+LSTM+Bahdanau Attention model (PyTorch)
- Uses Optuna for hyperparameter optimization
- Trains to predict next bar's close price
- Saves LSTM+attention output and tech indicators for XGBoost
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from tqdm import tqdm
import talib

# --- Data Preparation ---
def generate_ohlcv_data(n_samples=2000, seed=42):
    np.random.seed(seed)
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='h')
    price = 100 + np.cumsum(np.random.randn(n_samples))
    high = price + np.abs(np.random.randn(n_samples))
    low = price - np.abs(np.random.randn(n_samples))
    open_ = price + np.random.randn(n_samples)
    close = price
    volume = np.random.randint(1000, 10000, n_samples)
    df = pd.DataFrame({'datetime': dates, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    return df

def compute_tech_indicators(df):
    # Add technical indicators using TA-Lib
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    volume = df['volume'].values
    df['rsi'] = talib.RSI(close, timeperiod=14)
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    upper, middle, lower = talib.BBANDS(close, timeperiod=20)
    df['bb_bbm'] = middle
    df['bb_bbh'] = upper
    df['bb_bbl'] = lower
    # Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B)
    nine_high = pd.Series(high).rolling(window=9).max()
    nine_low = pd.Series(low).rolling(window=9).min()
    df['tenkan_sen'] = (nine_high + nine_low) / 2
    period26_high = pd.Series(high).rolling(window=26).max()
    period26_low = pd.Series(low).rolling(window=26).min()
    df['kijun_sen'] = (period26_high + period26_low) / 2
    df['ichimoku_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    period52_high = pd.Series(high).rolling(window=52).max()
    period52_low = pd.Series(low).rolling(window=52).min()
    df['ichimoku_b'] = ((period52_high + period52_low) / 2).shift(26)
    df['obv'] = talib.OBV(close, volume)
    slowk, slowd = talib.STOCH(high, low, close)
    df['stoch'] = slowk
    # Fill NaNs
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

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
        cnn_out_len = (100 - cnn_params['kernel_size'] + 1) // 2
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
        return out, context.detach(), attn_weights.detach()

# --- Data Loader ---
def create_windows(df, window_size=100):
    X_ohlcv = []
    X_tech = []
    y = []
    for i in range(window_size, len(df)-1):
        ohlcv = df[['open','high','low','close','volume']].iloc[i-window_size:i].values.T
        tech = df[['rsi','atr','macd','bb_bbm','bb_bbh','bb_bbl','ichimoku_a','ichimoku_b','obv','stoch']].iloc[i-window_size:i].values
        X_ohlcv.append(ohlcv)
        X_tech.append(tech)
        y.append(df['close'].iloc[i+1])
    return np.stack(X_ohlcv), np.stack(X_tech), np.array(y)

# --- Optuna Objective ---
def objective(trial, X_ohlcv, y):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kernel_size = trial.suggest_categorical('kernel_size', [3,5,7])
    filters = trial.suggest_int('filters', 16, 256)
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    model = HybridModel(
        cnn_params={'filters': filters, 'kernel_size': kernel_size},
        lstm_params={'hidden_size': hidden_size, 'num_layers': num_layers, 'dropout': dropout},
        tech_feat_dim=10
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True))
    criterion = nn.MSELoss()
    X = torch.tensor(X_ohlcv, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        out, _, _ = model(X)
        loss = criterion(out.squeeze(), y_t)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        out, _, _ = model(X)
        val_loss = criterion(out.squeeze(), y_t).item()
    return val_loss

# --- Main Training Script ---
def main():
    print("Generating data and computing indicators...")
    df = generate_ohlcv_data()
    df = compute_tech_indicators(df)
    print("Creating windows...")
    X_ohlcv, X_tech, y = create_windows(df, window_size=100)
    print(f"X_ohlcv: {X_ohlcv.shape}, X_tech: {X_tech.shape}, y: {y.shape}")
    print("Starting Optuna hyperparameter search...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_ohlcv, y), n_trials=10)
    print("Best trial:", study.best_trial.params)
    best_params = study.best_trial.params
    model = HybridModel(
        cnn_params={'filters': best_params['filters'], 'kernel_size': best_params['kernel_size']},
        lstm_params={'hidden_size': best_params['hidden_size'], 'num_layers': best_params['num_layers'], 'dropout': best_params['dropout']},
        tech_feat_dim=10
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()
    X = torch.tensor(X_ohlcv, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)
    print("Training best model...")
    for epoch in tqdm(range(10)):
        model.train()
        optimizer.zero_grad()
        out, _, _ = model(X)
        loss = criterion(out.squeeze(), y_t)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")
    print("Saving features for XGBoost...")
    model.eval()
    with torch.no_grad():
        _, lstm_features, _ = model(X)
    tech_last = X_tech[:, -1, :]
    features = np.concatenate([lstm_features.cpu().numpy(), tech_last], axis=1)
    np.savez('hybrid_xgboost_features.npz', features=features, target=(y[1:] > y[:-1]).astype(int)[:features.shape[0]])
    print("Done. Features saved to hybrid_xgboost_features.npz")

if __name__ == '__main__':
    main()