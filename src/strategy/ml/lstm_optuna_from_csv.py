import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
import os
import glob
import json
import datetime

# --- Parameters ---
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"] # Change as needed
TARGET_COLUMN = "close" # Change as needed
SEQ_LEN = 30 # Lookback window (can be tuned)

# --- Data Loading and Preparation ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.common.technicals import calculate_technicals_from_df

def load_sequence_data(csv_file, feature_cols, target_col, seq_len, indicator_params=None):
    df = pd.read_csv(csv_file)
    # Add TA-Lib indicators
    indicators = ['rsi', 'bb_lower', 'bb_upper', 'atr', 'macd']
    # Map to names used in calculate_technicals_from_df
    indicator_map = {
        'rsi': 'rsi',
        'bb_lower': 'bb_lower',
        'bb_upper': 'bb_upper',
        'atr': 'adr',  # ADR is used for ATR in that file
        'macd': 'macd',
    }
    calc_indicators = [indicator_map[k] for k in indicators]
    df, _ = calculate_technicals_from_df(df, indicators=calc_indicators, indicator_params=indicator_params)
    print(f"[DEBUG] DataFrame length after indicator calculation: {len(df)}")
    print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")
    # Fill NA after indicator calculation
    df = df.bfill().ffill()
    # Add indicator columns to features
    feature_cols = feature_cols + [indicator_map[k] for k in indicators]
    features = df[feature_cols].values
    target = df[target_col].values.reshape(-1, 1)
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # Build sequences
    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(features_scaled[i:i+seq_len])
        y.append(target[i+seq_len])
    X = np.array(X)
    y = np.array(y)
    print(f"[DEBUG] After sequence creation: X shape = {X.shape}, y shape = {y.shape}")
    if X.shape[0] != y.shape[0]:
        print(f"[ERROR] Shape mismatch after sequence creation: X={X.shape}, y={y.shape}")
        raise ValueError("Shape mismatch after sequence creation")
    return X, y

def prepare_data(csv_file, indicator_params=None):
    X, y = load_sequence_data(csv_file, FEATURE_COLUMNS, TARGET_COLUMN, SEQ_LEN, indicator_params=indicator_params)
    # Split: 80% for Optuna (train+val), 20% for test
    X_optuna, X_test, y_optuna, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_optuna, y_optuna, test_size=0.2, shuffle=False)
    return X_train, X_val, X_optuna, X_test, y_train, y_val, y_optuna, y_test, X

def make_loader(X, y, batch_size):
    print(f"[DEBUG] make_loader: X shape = {X.shape}, y shape = {y.shape}")
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# --- Optuna Objective Function ---
def objective(trial, X_train, y_train, X_val, y_val, input_size, csv_file):
    # Suggest indicator parameters
    indicator_params = {
        'rsi': {'timeperiod': trial.suggest_int('rsi_period', 7, 30)},
        'bb_upper': {'timeperiod': trial.suggest_int('bb_period', 10, 30)},
        'bb_lower': {'timeperiod': trial.suggest_int('bb_period', 10, 30)},
        'adr': {'timeperiod': trial.suggest_int('atr_period', 7, 30)},
        'macd': {
            'fastperiod': trial.suggest_int('macd_fast', 8, 20),
            'slowperiod': trial.suggest_int('macd_slow', 18, 30),
            'signalperiod': trial.suggest_int('macd_signal', 5, 15),
        },
    }
    # Prepare data with these indicator params
    X_train, X_val, _, _, y_train, y_val, _, _, _ = prepare_data(csv_file, indicator_params=indicator_params)

    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 10, 100)

    train_loader = make_loader(X_train, y_train, batch_size)
    val_loader = make_loader(X_val, y_val, batch_size)

    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"[Optuna Trial {trial.number}] Finished epoch {epoch+1}/{epochs}")

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            val_losses.append(loss.item())
    return np.mean(val_losses)

# --- Run Optuna Study ---
def main(csv_file):
    def get_indicator_params_from_trial(trial):
        return {
            'rsi': {'timeperiod': trial.suggest_int('rsi_period', 7, 30)},
            'bb_upper': {'timeperiod': trial.suggest_int('bb_period', 10, 30)},
            'bb_lower': {'timeperiod': trial.suggest_int('bb_period', 10, 30)},
            'adr': {'timeperiod': trial.suggest_int('atr_period', 7, 30)},
            'macd': {
                'fastperiod': trial.suggest_int('macd_fast', 8, 20),
                'slowperiod': trial.suggest_int('macd_slow', 18, 30),
                'signalperiod': trial.suggest_int('macd_signal', 5, 15),
            },
        }
    # Prepare initial data (default params)
    X_train, X_val, X_optuna, X_test, y_train, y_val, y_optuna, y_test, X = prepare_data(csv_file)
    input_size = X.shape[2]

    best_epochs = 10  # default fallback
    best_indicator_params = None
    def optuna_objective(trial):
        indicator_params = get_indicator_params_from_trial(trial)
        X_train, X_val, _, _, y_train, y_val, _, _, X = prepare_data(csv_file, indicator_params=indicator_params)
        return objective(trial, X_train, y_train, X_val, y_val, X.shape[2], csv_file)

    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, n_trials=30)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    best_params = trial.params
    best_epochs = best_params.get('epochs', 10)
    best_indicator_params = get_indicator_params_from_trial(trial)

    # --- Retrain Best Model on Full Optuna Data ---
    X_train, X_val, X_optuna, X_test, y_train, y_val, y_optuna, y_test, X = prepare_data(csv_file, indicator_params=best_indicator_params)
    input_size = X.shape[2]
    model = LSTMModel(
        input_size=input_size,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()
    full_loader = make_loader(X_optuna, y_optuna, best_params['batch_size'])

    for epoch in range(best_epochs):
        model.train()
        for xb, yb in full_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == best_epochs - 1:
            print(f"[Retrain] Finished epoch {epoch+1}/{best_epochs}")

    # --- Final Test Evaluation ---
    test_loader = make_loader(X_test, y_test, best_params['batch_size'])
    model.eval()
    test_losses = []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            test_losses.append(loss.item())
    print(f"Test MSE: {np.mean(test_losses):.4f}")

    # --- Save Model ---
    # Parse symbol and period from filename
    base = os.path.basename(csv_file)
    parts = base.split('_')
    if len(parts) >= 2:
        symbol = parts[0]
        period = parts[1]
    else:
        symbol = 'model'
        period = 'unknown'
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{symbol}_{period}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # --- Save Results as JSON ---
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(results_dir, f"LSTM_OPTUNA_{symbol}_{period}_{timestamp}.json")
    results_data = {
        'symbol': symbol,
        'period': period,
        'timestamp': timestamp,
        'best_params': best_params,
        'best_indicator_params': best_indicator_params,
        'test_mse': float(np.mean(test_losses)),
    }
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to {results_path}")


def run_all_csvs_in_data():
    # Find project root (three levels up from this file)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    data_dir = os.path.join(project_root, 'data')
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    for csv_file in csv_files:
        print(f"Processing {csv_file}")
        main(csv_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            run_all_csvs_in_data()
        else:
            csv_file = sys.argv[1]
            main(csv_file)
    else:
        run_all_csvs_in_data()


