"""
End-to-End LSTM Time Series Forecasting Pipeline with Hyperparameter Optimization.

This script provides a complete, automated pipeline for training, evaluating, and
saving Long Short-Term Memory (LSTM) models for financial time series prediction.
It leverages the Optuna library to perform extensive hyperparameter tuning,
optimizing not only the model's architecture but also the feature engineering
process itself by selecting the best parameters for technical indicators.

Workflow:
1.  **Data Ingestion**: The script can process a single CSV file or automatically
    discover and iterate through all CSV files in the project's `data/` directory.
2.  **Hyperparameter and Feature Optimization**: For each dataset, an Optuna
    study is launched. A single trial consists of:
    a.  Suggesting hyperparameters for technical indicators (e.g., RSI period,
        MACD settings).
    b.  Suggesting hyperparameters for the LSTM model (e.g., hidden size,
        number of layers, dropout, learning rate, batch size).
    c.  Dynamically generating features based on the trial's indicator parameters.
    d.  Creating sequence data (lookback windows) for the LSTM.
    e.  Splitting the data into training and validation sets.
    f.  Scaling features (StandardScaler) and the target (MinMaxScaler).
    g.  Training the LSTM model on the trial data and evaluating it on the
        validation set.
    h.  Reporting the validation loss to Optuna, which uses a MedianPruner
        to terminate unpromising trials early.
3.  **Final Model Training**: After the study concludes, the best hyperparameters
    are used to train a new, final model on the complete training dataset
    (training + validation sets).
4.  **Evaluation**: The final model is evaluated on a hold-out test set. Its
    performance (MSE and RMSE) is calculated and compared against a naive
    baseline model (predicting the previous time step's log return).
5.  **Artifact Storage**: The script saves the results of the experiment:
    - The trained PyTorch model state dictionary (`.pt`).
    - A detailed JSON file containing the best hyperparameters, final
      evaluation metrics, and other metadata.

Key Features:
-   **Dual Optimization**: Simultaneously tunes model architecture and feature
    engineering parameters for a more holistic optimization.
-   **Automated Pipeline**: Capable of running unattended on an entire directory
    of datasets.
-   **Reproducibility**: Uses a fixed random seed for consistent results in
    data splitting and model weight initialization.
-   **Efficient Tuning**: Employs Optuna's pruners to save time by stopping
    poor-performing trials early.
-   **GPU Acceleration**: Automatically utilizes a CUDA-enabled GPU if available.
-   **Robust Path Management**: Uses `pathlib` for cross-platform compatibility.

Input Requirements:
-   **Directory Structure**: Expects a `data/` directory at the project root.
-   **File Format**: Input files must be in CSV format.
-   **Filename Convention**: Files should be named `symbol_timeframe.csv`
    (e.g., 'btcusdt_15m.csv') for proper parsing.
-   **Required CSV Columns**: Each CSV must contain `timestamp`, `open`, `high`,
    `low`, `close`, `volume`, and `log_return`.

Output Artifacts:
-   All outputs are saved to the `results/` directory at the project root.
-   **Model File**: `LSTM_{symbol}_{timeframe}.pt` - The PyTorch state_dict of
    the best-trained model.
-   **Results File**: `LSTM_OPTUNA_{symbol}_{period}_{timestamp}.json` - A JSON
    file containing:
    - Best hyperparameters found by Optuna.
    - Final evaluation metrics (MSE, RMSE) on the test set.
    - Performance metrics of the naive baseline model for comparison.
    - Metadata about the experiment (symbol, timestamp, etc.).

Usage (from the command line):
-   **Process all CSVs in the `data` directory:**
    `python your_script_name.py --all`
-   **Process a single, specific CSV file:**
    `python your_script_name.py path/to/your/file.csv`
-   **Default behavior (no arguments):**
    Falls back to processing all CSVs in the `data` directory.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import optuna
import json
import datetime
import sys
from pathlib import Path  # ## IMPROVEMENT ##: Using modern, object-oriented pathlib
import random # ## IMPROVEMENT ##: For setting seeds

# --- Configuration ---
## IMPROVEMENT ##: Centralized configuration for clarity and easy modification
class Config:
    FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "log_return"]
    TARGET_COLUMN = "log_return"
    SEQ_LEN = 30  # Lookback window
    TEST_SIZE = 0.2
    VAL_SIZE = 0.25 # Validation size from the training set
    N_TRIALS = 50 # Number of Optuna trials
    DEFAULT_EPOCHS = 30 # Default epochs, can be tuned
    RANDOM_SEED = 42

# --- Path and Logger Setup ---
## IMPROVEMENT ##: Using pathlib for cleaner path management
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))
from src.indicators.service import get_unified_indicator_service
from src.notification.logger import setup_logger

# Helper function to replace calculate_technicals_from_df for ML pipeline
def calculate_technicals_for_ml(df: pd.DataFrame, indicators: list, indicator_params: dict):
    """
    Calculate technical indicators for ML pipeline using TA-Lib directly.
    Returns (DataFrame with indicators, None) to maintain compatibility with legacy code.
    """
    if df is None or df.empty:
        return None, None

    # Convert DataFrame to OHLCV format expected by unified service
    # The DataFrame should have columns: ['open', 'high', 'low', 'close', 'volume']
    df_copy = df.copy()

    # For ML pipeline, we'll use TA-Lib directly since we need the full DataFrame with indicators
    try:
        # Use TA-Lib directly for ML pipeline since we need the full DataFrame with indicators
        import talib
        import numpy as np

        high = df_copy['high'].values.astype(float)
        low = df_copy['low'].values.astype(float)
        close = df_copy['close'].values.astype(float)
        volume = df_copy['volume'].values.astype(float)

        # Calculate indicators based on the requested list
        for indicator in indicators:
            if indicator == 'rsi' and 'rsi' in indicator_params:
                df_copy['rsi'] = talib.RSI(close, **indicator_params['rsi'])
            elif indicator == 'bb_upper' and 'bb_upper' in indicator_params:
                bb_upper, _, _ = talib.BBANDS(close, **indicator_params['bb_upper'])
                df_copy['bb_upper'] = bb_upper
            elif indicator == 'bb_lower' and 'bb_lower' in indicator_params:
                _, _, bb_lower = talib.BBANDS(close, **indicator_params['bb_lower'])
                df_copy['bb_lower'] = bb_lower
            elif indicator == 'macd' and 'macd' in indicator_params:
                macd, _, _ = talib.MACD(close, **indicator_params['macd'])
                df_copy['macd'] = macd
            elif indicator == 'adr' and 'adr' in indicator_params:
                daily_range = high - low
                df_copy['adr'] = talib.SMA(daily_range, **indicator_params['adr'])

        return df_copy, None

    except Exception as e:
        _logger.exception("Error calculating technical indicators for ML: ")
        return None, None

_logger = setup_logger(__name__)

## IMPROVEMENT ##: Set seeds for reproducibility
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds(Config.RANDOM_SEED)

# ## IMPROVEMENT ##: Added device handling for GPU acceleration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_logger.info("Using device: %s")

# --- Data Preparation ---
def create_sequences(features: np.ndarray, target: np.ndarray, seq_len: int):
    """Creates sequences from feature and target arrays."""
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(target[i + seq_len])

    if not X:
        return np.array([]), np.array([])
    return np.array(X), np.array(y)

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """Creates a PyTorch DataLoader."""
    tensor_x = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    tensor_y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# --- Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the output of the last time step
        return self.fc(out)

# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial, df: pd.DataFrame):
    """Defines a single Optuna trial, now more efficient."""
    # 1. Suggest hyperparameters
    bb_period = trial.suggest_int('bb_period', 10, 30)
    macd_fast = trial.suggest_int('macd_fast', 8, 20)
    macd_slow = trial.suggest_int('macd_slow', macd_fast + 5, 40) # Ensure slow > fast

    indicator_params = {
        'rsi': {'timeperiod': trial.suggest_int('rsi_period', 7, 30)},
        'bb_upper': {'timeperiod': bb_period},
        'bb_lower': {'timeperiod': bb_period},
        # Align key with function parameter
        'adr': {'timeperiod': trial.suggest_int('atr_period', 7, 30)},
        'macd': {
            'fastperiod': macd_fast,
            'slowperiod': macd_slow,
            'signalperiod': trial.suggest_int('macd_signal', 5, 15),
        },
    }

    hidden_size = trial.suggest_int('hidden_size', 32, 256, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 10, 50)

    # 2. Prepare data for this trial (more efficient)
    trial_df = df.copy()
    indicators_to_add = list(indicator_params.keys())
    trial_df, _ = calculate_technicals_for_ml(trial_df, indicators=indicators_to_add, indicator_params=indicator_params)
    trial_df = trial_df.dropna().reset_index(drop=True)

    all_feature_cols = Config.FEATURE_COLUMNS + indicators_to_add
    features = trial_df[all_feature_cols].values
    target = trial_df[Config.TARGET_COLUMN].values.reshape(-1, 1)

    X, y = create_sequences(features, target, Config.SEQ_LEN)

    if X.shape[0] < 50:
        raise optuna.exceptions.TrialPruned("Not enough data after feature engineering.")

    # 3. Split and scale data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=Config.VAL_SIZE, shuffle=False)

    # ## IMPROVEMENT ##: Scaling both features and target variable
    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler()

    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    X_train_scaled = feature_scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    y_train_scaled = target_scaler.fit_transform(y_train)

    X_val_scaled = feature_scaler.transform(X_val.reshape(-1, X_val.shape[2])).reshape(X_val.shape)
    y_val_scaled = target_scaler.transform(y_val)

    # 4. Create DataLoaders and Model
    train_loader = make_loader(X_train_scaled, y_train_scaled, batch_size)
    val_loader = make_loader(X_val_scaled, y_val_scaled, batch_size, shuffle=False)

    model = LSTMModel(X.shape[2], hidden_size, num_layers, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 5. Train and Validate with pruning
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += criterion(pred, yb).item()

        avg_val_loss = val_loss / len(val_loader)
        trial.report(avg_val_loss, epoch)

        # ## IMPROVEMENT ##: Intermediate value pruning
        if trial.should_prune():
            _logger.info("Trial %s pruned at epoch %s.")
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss

# --- Main Execution Block ---
def run_experiment(csv_file: Path):
    df_base = pd.read_csv(csv_file)

    # Run Optuna study
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, df_base), n_trials=Config.N_TRIALS)

    _logger.info("\n" + "="*50 + "\nOptuna Study Complete.\n" + "="*50)
    best_trial = study.best_trial
    _logger.info("Best Trial Validation MSE: %s")
    _logger.info("Best Hyperparameters:")
    for key, value in best_trial.params.items():
        _logger.info("  %s: %s")

    # --- Retrain Best Model on Full Training Data ---
    _logger.info("\n--- Retraining model with best hyperparameters ---")
    best_params = best_trial.params
    best_indicator_params = {
        'rsi': {'timeperiod': best_params['rsi_period']},
        'bb_upper': {'timeperiod': best_params['bb_period']},
        'bb_lower': {'timeperiod': best_params['bb_period']},
        'adr': {'timeperiod': best_params['atr_period']},
        'macd': {
            'fastperiod': best_params['macd_fast'],
            'slowperiod': best_params['macd_slow'],
            'signalperiod': best_params['macd_signal'],
        },
    }

    # Generate features on the full dataset
    final_df = df_base.copy()
    indicators_to_add = list(best_indicator_params.keys())
    final_df, _ = calculate_technicals_for_ml(final_df, indicators=indicators_to_add, indicator_params=best_indicator_params)
    final_df = final_df.dropna().reset_index(drop=True)

    all_feature_cols = Config.FEATURE_COLUMNS + indicators_to_add
    features = final_df[all_feature_cols].values
    target = final_df[Config.TARGET_COLUMN].values.reshape(-1, 1)

    X, y = create_sequences(features, target, Config.SEQ_LEN)

    # Split into final training and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, shuffle=False)

    # Scale the final datasets
    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler()

    X_train_full_scaled = feature_scaler.fit_transform(X_train_full.reshape(-1, X.shape[2])).reshape(X_train_full.shape)
    y_train_full_scaled = target_scaler.fit_transform(y_train_full)

    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, X.shape[2])).reshape(X_test.shape)

    # Build and train the final model
    final_model = LSTMModel(
        input_size=X.shape[2],
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(DEVICE)

    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()
    full_loader = make_loader(X_train_full_scaled, y_train_full_scaled, best_params['batch_size'])

    for epoch in range(best_params['epochs']):
        final_model.train()
        for xb, yb in full_loader:
            optimizer.zero_grad()
            loss = criterion(final_model(xb), yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            _logger.info("[Retrain] Epoch %s/%s complete.")

    # --- Final Test Evaluation ---
    final_model.eval()
    with torch.no_grad():
        test_preds_scaled = final_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE))

    # Inverse transform predictions to original scale for MSE calculation
    test_preds = target_scaler.inverse_transform(test_preds_scaled.cpu().numpy())
    final_mse = np.mean((test_preds - y_test)**2)
    _logger.info("Final Test MSE: %s")
    _logger.info("Final Test RMSE: %s")

    # --- Naive Reference Model ---
    close_idx = Config.FEATURE_COLUMNS.index('log_return')
    naive_preds = X_test[:, -1, close_idx]
    naive_targets = y_test.flatten()
    naive_mse = np.mean((naive_preds - naive_targets)**2)
    _logger.info("Naive (previous log_return) Test MSE: %s")
    _logger.info("Naive (previous log_return) Test RMSE: %s")

    # --- Save Model and Results ---
    symbol, period = csv_file.stem.split('_')[:2]

    model_dir = PROJECT_ROOT / 'results'
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"LSTM_{csv_file.stem}.pt"
    torch.save(final_model.state_dict(), model_path)
    _logger.info("Model saved to %s")

    results_dir = PROJECT_ROOT / 'results'
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f"LSTM_OPTUNA_{symbol}_{period}_{timestamp}.json"

    results_data = {
        'symbol': symbol,
        'period': period,
        'timestamp': timestamp,
        'best_hyperparameters': best_params,
        'best_validation_mse_scaled': best_trial.value,
        'final_test_mse': float(final_mse),
        'final_test_rmse': np.sqrt(final_mse),
        'naive_test_mse': float(naive_mse),
        'naive_test_rmse': np.sqrt(naive_mse),
        'data_source': csv_file.name,
        'random_seed': Config.RANDOM_SEED
    }
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    _logger.info("Results saved to %s")

def run_all_csvs_in_data():
    data_dir = PROJECT_ROOT / 'data'
    csv_files = list(data_dir.glob('*.csv'))
    _logger.info("Found %s CSV files in %s")
    for csv_file in csv_files:
        _logger.info("\n%s Processing %s %s")
        try:
            run_experiment(csv_file)
        except Exception as e:
            _logger.exception("ERROR processing %s", csv_file.name)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            run_all_csvs_in_data()
        else:
            csv_path = Path(sys.argv[1])
            if csv_path.exists():
                run_experiment(csv_path)
            else:
                _logger.error("Error: File not found at %s", csv_path)
    else:
        _logger.info("Usage: python your_script.py [--all | path/to/your.csv]")
        # Default to running all
        run_all_csvs_in_data()
