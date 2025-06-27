import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------
# DATA PROCESSING
# -------------------------------
def load_btc_data():
    df = yf.download("BTC-USD", start="2017-01-01", end="2024-06-30")
    df = df.asfreq("D")
    return df[["Close"]].dropna()


def load_btc_data2():
    df = yf.download("BTC-USD", start="2024-05-01", end="2025-06-01")
    df = df.asfreq("D")
    return df[["Close"]].dropna()


def holt_winters_smoothing(series):
    model = ExponentialSmoothing(
        series, trend="add", seasonal="add", seasonal_periods=365
    )
    fit = model.fit(optimized=True)
    fitted = fit.fittedvalues
    deseasonalized = series / fitted
    return deseasonalized.bfill()


def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window : i])
        y.append(data[i])
    return np.array(X), np.array(y)


# -------------------------------
# HELFORMER MODEL
# -------------------------------
class Helformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, num_layers, dropout):
        super(Helformer, self).__init__()
        self.input_proj = nn.Linear(1, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# -------------------------------
# OBJECTIVE FUNCTION
# -------------------------------
def objective(trial):
    window = 30
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    embed_dim = trial.suggest_categorical("embed_dim", [8, 16, 32, 64])
    n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Ensure embed_dim is divisible by n_heads
    if embed_dim % n_heads != 0:
        print(f"Pruned: embed_dim {embed_dim} not divisible by n_heads {n_heads}")
        raise optuna.exceptions.TrialPruned()

    model = Helformer(
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    df = load_btc_data()
    deseason = holt_winters_smoothing(df["Close"])
    deseason = deseason.dropna()

    # Prune trial if no data left
    if len(deseason) == 0:
        print("Pruned: deseasonalized data is empty after preprocessing.")
        raise optuna.exceptions.TrialPruned()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(deseason.values.reshape(-1, 1))
    scaled = np.nan_to_num(scaled, nan=0.0)

    X, y = create_sequences(scaled, window)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(10):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).squeeze()
        loss = criterion(preds, y_tensor.squeeze())

    return (
        loss.item(),
        model.state_dict(),
        {
            "hidden_dim": hidden_dim,
            "n_heads": n_heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "embed_dim": embed_dim,
        },
    )


# -------------------------------
# LOAD AND INFER
# -------------------------------
def load_and_predict():
    checkpoint = torch.load("helformer_best_model.pt")
    params = checkpoint["params"]
    model = Helformer(
        input_dim=params["embed_dim"],
        hidden_dim=params["hidden_dim"],
        n_heads=params["n_heads"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = load_btc_data2()
    deseason = holt_winters_smoothing(df["Close"])
    deseason = deseason.dropna()

    # Prune trial if no data left
    if len(deseason) == 0:
        raise optuna.exceptions.TrialPruned()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(deseason.values.reshape(-1, 1))
    scaled = np.nan_to_num(scaled, nan=0.0)

    window = 30
    X, y = create_sequences(scaled, window)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X_tensor).squeeze().numpy()

    preds_rescaled = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actuals_rescaled = scaler.inverse_transform(np.array(y).reshape(-1, 1)).flatten()

    print("Last 10 predictions:", preds_rescaled[-10:])

    # Buy/Sell Strategy with Trailing Stop-Loss and Plotting Signals
    trailing_pct = 0.05  # 5% trailing stop
    balance = 10000
    btc_holding = 0
    peak_price = 0
    bought_price = 0

    buy_signals = []
    sell_signals = []

    for i in range(1, len(preds_rescaled)):
        price = actuals_rescaled[i - 1]

        if preds_rescaled[i] > price and balance > 0:
            btc_holding = balance / price
            bought_price = price
            peak_price = price
            balance = 0
            buy_signals.append((i - 1, price))

        elif btc_holding > 0:
            if price > peak_price:
                peak_price = price

            if price < peak_price * (1 - trailing_pct):
                balance = btc_holding * price
                btc_holding = 0
                peak_price = 0
                bought_price = 0
                sell_signals.append((i - 1, price))

            elif preds_rescaled[i] < price:
                balance = btc_holding * price
                btc_holding = 0
                peak_price = 0
                bought_price = 0
                sell_signals.append((i - 1, price))

    # Plotting with signals
    plt.figure(figsize=(14, 6))
    plt.plot(actuals_rescaled, label="Actual Prices", alpha=0.6)
    plt.plot(preds_rescaled, label="Predicted Prices", alpha=0.6)

    if buy_signals:
        buy_x, buy_y = zip(*buy_signals)
        plt.scatter(buy_x, buy_y, marker="^", color="green", label="Buy Signal", s=100)

    if sell_signals:
        sell_x, sell_y = zip(*sell_signals)
        plt.scatter(sell_x, sell_y, marker="v", color="red", label="Sell Signal", s=100)

    plt.title("BTC Forecast with Buy/Sell Signals")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    final_value = balance + btc_holding * actuals_rescaled[-1]
    roi = (final_value - 10000) / 10000 * 100
    print(f"Final portfolio value: ${final_value:.2f}, ROI: {roi:.2f}%")


def test_minimal_pipeline():
    window = 30
    hidden_dim = 32
    embed_dim = 8
    n_heads = 2
    num_layers = 1
    dropout = 0.1
    lr = 0.001
    batch_size = 32

    model = Helformer(
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    df = yf.download("BTC-USD", start="2017-01-01", end="2024-06-30")
    print("Downloaded data shape:", df.shape)
    print(df.head())
    df = df.asfreq("D")
    print("After asfreq('D') shape:", df.shape)
    print(df.head(10))
    df = df[["Close"]].dropna()
    print("After dropna shape:", df.shape)
    print(df.head(10))

    deseason = holt_winters_smoothing(df["Close"])
    print("After holt_winters_smoothing, shape:", deseason.shape)
    print(deseason.head(10))
    deseason = deseason.dropna()
    print("After final dropna, shape:", deseason.shape)
    print(deseason.head(10))
    if len(deseason) == 0:
        print("No data after preprocessing!")
        return False

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(deseason.values.reshape(-1, 1))
    scaled = np.nan_to_num(scaled, nan=0.0)

    X, y = create_sequences(scaled, window)
    print("After create_sequences, X shape:", X.shape, "y shape:", y.shape)
    if len(X) == 0:
        print("No sequences created!")
        return False

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(3):  # Fewer epochs for quick test
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    print("Minimal pipeline ran successfully.")
    return True


# -------------------------------
# MAIN EXECUTION
# -------------------------------
def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    # Test minimal pipeline first
    if not test_minimal_pipeline():
        print("Minimal pipeline failed. Please check your data and model setup.")
        return
    best_loss = float("inf")
    best_state = None
    best_params = None

    def optuna_objective(trial):
        nonlocal best_loss, best_state, best_params
        loss, state_dict, params = objective(trial)
        if loss < best_loss:
            best_loss = loss
            best_state = state_dict
            best_params = params
        return loss

    from optuna.visualization import (plot_optimization_history,
                                      plot_param_importances)

    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=20, n_jobs=1)

    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        print("No completed trials. All trials were pruned or failed.")
        return
    print("Best trial:")
    print(study.best_trial)
    print("Best params:")
    print(study.best_params)

    if not os.path.exists("optuna_results"):
        os.mkdir("optuna_results")
    study.trials_dataframe().to_csv("optuna_results/helformer_trials.csv", index=False)

    # Save full trial log
    with open("optuna_results/optuna_log.txt", "w") as f:
        for t in study.trials:
            f.write(f"Trial {t.number}: Loss={t.value}, Params={t.params}")

        torch.save(
            {"model_state_dict": best_state, "params": best_params},
            "helformer_best_model.pt",
        )

        # Visualize parameter importance
        fig = plot_param_importances(study)
    fig.write_image("optuna_results/param_importance.png")

    # Visualize optimization history
    fig_hist = plot_optimization_history(study)
    fig_hist.write_image("optuna_results/optimization_history.png")

    load_and_predict()


if __name__ == "__main__":
    main()
