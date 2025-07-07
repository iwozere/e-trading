import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna

# --- Parameters ---
CSV_FILE = "your_data.csv" # Change to your CSV filename
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"] # Change as needed
TARGET_COLUMN = "close" # Change as needed
SEQ_LEN = 30 # Lookback window (can be tuned)

# --- Data Loading and Preparation ---
def load_sequence_data(csv_file, feature_cols, target_col, seq_len):
    df = pd.read_csv(csv_file)
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
    return X, y

X, y = load_sequence_data(CSV_FILE, FEATURE_COLUMNS, TARGET_COLUMN, SEQ_LEN)

# Split: 80% for Optuna (train+val), 20% for test
X_optuna, X_test, y_optuna, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_optuna, y_optuna, test_size=0.2, shuffle=False)

def make_loader(X, y, batch_size):
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
def objective(trial):
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    train_loader = make_loader(X_train, y_train, batch_size)
    val_loader = make_loader(X_val, y_val, batch_size)

    model = LSTMModel(input_size=X.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            val_losses.append(loss.item())
    return np.mean(val_losses)

# --- Run Optuna Study ---
if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # --- Retrain Best Model on Full Optuna Data ---
    best_params = trial.params
    model = LSTMModel(
        input_size=X.shape[2],
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()
    full_loader = make_loader(X_optuna, y_optuna, best_params['batch_size'])

    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in full_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

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