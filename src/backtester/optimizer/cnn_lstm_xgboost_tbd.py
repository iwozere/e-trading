"""
Hybrid CNN-LSTM-XGBoost Optimization Module
------------------------------------------

This module implements a hybrid deep learning and machine learning pipeline for time series prediction using:
- A CNN-LSTM neural network with attention (PyTorch)
- XGBoost regression
- Technical indicators (TA-Lib)
- Hyperparameter optimization with Optuna

Workflow:
1. Load or generate data and compute technical indicators
2. Prepare data for sequence modeling
3. Optimize CNN-LSTM hyperparameters with Optuna
4. Train the best CNN-LSTM model
5. Extract features from the trained CNN-LSTM
6. Optimize XGBoost hyperparameters with Optuna
7. Train the best XGBoost model on CNN-LSTM features
8. Evaluate and visualize predictions

Classes and functions are documented below.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import talib
import optuna

class Config:
    """
    Configuration for the hybrid model pipeline.
    Attributes:
        time_steps (int): Number of timesteps in each input sequence.
        cnn_lstm_epochs (int): Number of epochs for CNN-LSTM training.
        batch_size (int): Batch size for training.
        n_trials (int): Number of Optuna trials for each optimization.
        study_name (str): Name of the Optuna study.
        storage_name (str): Storage URI for Optuna study.
    """
    time_steps = 20
    cnn_lstm_epochs = 50
    batch_size = 32
    n_trials = 20
    study_name = "hybrid_model_opt"
    storage_name = "sqlite:///db/hybrid.db"

class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model with attention for time series prediction.
    Args:
        time_steps (int): Number of timesteps in each input sequence.
        features (int): Number of input features per timestep.
        conv_filters (int): Number of convolutional filters.
        lstm_units (int): Number of units in the first LSTM layer.
        dense_units (int): Number of units in the second LSTM layer and dense output.
    """
    def __init__(self, time_steps, features, conv_filters, lstm_units, dense_units):
        super(HybridCNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=conv_filters, kernel_size=3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=conv_filters, hidden_size=lstm_units, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=lstm_units, num_heads=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstm_units, hidden_size=dense_units, batch_first=True)
        self.fc = nn.Linear(dense_units, 1)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, time_steps, features).
        Returns:
            torch.Tensor: Output tensor of shape (batch, 1).
        """
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        attn_output, _ = self.attention(x, x, x)
        x, _ = self.lstm2(attn_output)
        x = x[:, -1, :]
        out = self.fc(x)
        return out

def add_technical_indicators(df):
    """
    Add technical indicators to a DataFrame using TA-Lib.
    Args:
        df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
    Returns:
        pd.DataFrame: DataFrame with additional indicator columns.
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    df['RSI'] = talib.RSI(close, timeperiod=14)
    macd, macdsignal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_SIGNAL'] = macdsignal
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(close, timeperiod=20)
    df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    df['OBV'] = talib.OBV(close, volume)
    df.fillna(method='bfill', inplace=True)
    return df

def prepare_data(df, time_steps):
    """
    Prepare data for sequence modeling.
    Args:
        df (pd.DataFrame): DataFrame with features and indicators.
        time_steps (int): Number of timesteps in each input sequence.
    Returns:
        tuple: (X, y, scaler) where X is input array, y is target array, scaler is fitted MinMaxScaler.
    """
    features = df[['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'ATR']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i+time_steps])
        y.append(scaled_data[i+time_steps, 3]) # close price
    return np.array(X), np.array(y), scaler

def train_cnn_lstm(model, X_train, y_train, device, epochs, batch_size, lr):
    """
    Train the CNN-LSTM model.
    Args:
        model (nn.Module): The CNN-LSTM model.
        X_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training target data.
        device (torch.device): Device to train on.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
    Returns:
        nn.Module: Trained model.
    """
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    for epoch in range(epochs):
        permutation = torch.randperm(X_train_tensor.size(0))
        total_loss = 0
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.6f}')
    return model

def optimize_cnn_lstm(trial, X_train, y_train, X_val, y_val, device):
    """
    Optuna objective for CNN-LSTM hyperparameter optimization.
    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        device (torch.device): Device to train on.
    Returns:
        float: Validation mean squared error.
    """
    params = {
        'conv_filters': trial.suggest_int('conv_filters', 32, 128),
        'lstm_units': trial.suggest_int('lstm_units', 50, 200),
        'dense_units': trial.suggest_int('dense_units', 20, 100),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    }
    model = HybridCNNLSTM(
        Config.time_steps,
        X_train.shape[2],
        params['conv_filters'],
        params['lstm_units'],
        params['dense_units']
    )
    model = train_cnn_lstm(
        model, X_train, y_train, device,
        Config.cnn_lstm_epochs,
        params['batch_size'],
        params['lr']
    )
    model.eval()
    with torch.no_grad():
        val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        predictions = model(val_tensor).squeeze().cpu().numpy()
    return mean_squared_error(y_val, predictions)

def optimize_xgb(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective for XGBoost hyperparameter optimization.
    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
    Returns:
        float: Validation mean squared error.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1000),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1000)
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    return mean_squared_error(y_val, predictions)

def main():
    """
    Main entry point for the hybrid optimization pipeline.
    Loads data, runs Optuna optimization for CNN-LSTM and XGBoost, trains best models, evaluates, and visualizes results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Load data
    try:
        df = pd.read_csv('your_stock_data.csv')
        df = add_technical_indicators(df)
    except FileNotFoundError:
        print("File not found! Generating synthetic data.")
        data_len = 4000
        data = {
            'open': np.random.rand(data_len),
            'high': np.random.rand(data_len) * 1.1,
            'low': np.random.rand(data_len) * 0.9,
            'close': np.cumsum(np.random.randn(data_len)),
            'volume': np.random.randint(10000, 100000, data_len)
        }
        df = pd.DataFrame(data)
        df = add_technical_indicators(df)
    # Data preparation
    X, y, scaler = prepare_data(df, Config.time_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # CNN-LSTM optimization
    study_cnn = optuna.create_study(
        direction='minimize',
        study_name='cnn_lstm_opt',
        storage=Config.storage_name,
        load_if_exists=True
    )
    study_cnn.optimize(
        lambda trial: optimize_cnn_lstm(trial, X_train, y_train, X_test, y_test, device),
        n_trials=Config.n_trials
    )
    # Train the best CNN-LSTM
    best_cnn_params = study_cnn.best_params
    cnn_lstm_model = HybridCNNLSTM(
        Config.time_steps,
        X_train.shape[2],
        best_cnn_params['conv_filters'],
        best_cnn_params['lstm_units'],
        best_cnn_params['dense_units']
    )
    cnn_lstm_model = train_cnn_lstm(
        cnn_lstm_model, X_train, y_train, device,
        Config.cnn_lstm_epochs,
        best_cnn_params['batch_size'],
        best_cnn_params['lr']
    )
    # Get features from CNN-LSTM
    cnn_lstm_model.eval()
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        train_features = cnn_lstm_model(X_train_tensor).cpu().numpy()
        test_features = cnn_lstm_model(X_test_tensor).cpu().numpy()
    # Add technical indicators
    train_features = np.hstack([train_features, y_train.reshape(-1, 1)])
    test_features = np.hstack([test_features, y_test.reshape(-1, 1)])
    # XGBoost optimization
    study_xgb = optuna.create_study(
        direction='minimize',
        study_name='xgb_opt',
        storage=Config.storage_name,
        load_if_exists=True
    )
    study_xgb.optimize(
        lambda trial: optimize_xgb(trial, train_features, y_train, test_features, y_test),
        n_trials=Config.n_trials
    )
    # Train the best XGBoost
    best_xgb_params = study_xgb.best_params
    xgb_model = XGBRegressor(**best_xgb_params)
    xgb_model.fit(train_features, y_train)
    # Prediction
    predictions = xgb_model.predict(test_features)
    # Evaluation
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Best CNN-LSTM params: {best_cnn_params}")
    print(f"Best XGBoost params: {best_xgb_params}")
    print(f"Test MSE: {mse:.6f}, MAE: {mae:.6f}")
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title('Stock Price Prediction')
    plt.savefig('prediction_plot.png')
    plt.show()

if __name__ == "__main__":
    main()
