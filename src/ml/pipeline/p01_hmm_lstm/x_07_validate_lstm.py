"""
LSTM Model Validation and Reporting

This module validates trained LSTM models against naive baselines and generates
comprehensive performance reports including charts, metrics, and comparisons.
It evaluates models on hold-out test sets and provides insights into model
performance across different market regimes.

Features:
- Loads trained LSTM models and evaluates on test data
- Compares against naive baseline (previous close prediction)
- Calculates comprehensive performance metrics
- Generates visualizations and PDF reports
- Analyzes performance by market regime
- Saves results in JSON format for tracking
"""

# Set matplotlib backend to non-interactive to prevent file handle issues
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
from typing import Dict, Optional, Tuple
import talib # Added for technical indicators
import re # Added for regex

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    """Enhanced LSTM model for time series prediction with regime awareness."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float = 0.2, output_size: int = 1, n_regimes: int = 3):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_regimes = n_regimes

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # Additional dense layers for better feature extraction
        self.dense1 = nn.Linear(hidden_size + n_regimes, hidden_size // 2)
        self.dense2 = nn.Linear(hidden_size // 2, hidden_size // 4)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.5)

        # Output layer
        self.linear = nn.Linear(hidden_size // 4, output_size)

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, regime_onehot):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]

        # Concatenate with regime information early
        combined = torch.cat([last_output, regime_onehot], dim=1)

        # Apply dense layers with activation and dropout
        dense1_out = self.dropout1(self.relu(self.dense1(combined)))
        dense2_out = self.dropout2(self.relu(self.dense2(dense1_out)))

        # Final prediction
        output = self.linear(dense2_out)

        return output

class LSTMValidator:
    def __init__(self, config_path: str = "config/pipeline/p01.yaml"):
        """
        Initialize LSTM validator with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.labeled_data_dir = Path(self.config['paths']['data_labeled'])
        self.models_dir = Path(self.config['paths']['models_lstm'])
        self.results_dir = Path(self.config['paths']['models_lstm'])  # Optimization results are stored with models
        self.reports_dir = Path(self.config['paths']['reports'])
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        _logger.info("Loaded configuration from %s", self.config_path)
        return config

    def find_latest_model(self, symbol: str, timeframe: str) -> Optional[Path]:
        """
        Find the latest trained LSTM model.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Path to latest model file or None if not found
        """
        pattern = f"lstm_{symbol}_{timeframe}_*.pkl"
        model_files = list(self.models_dir.glob(pattern))

        if not model_files:
            _logger.warning("No LSTM model found for %s %s", symbol, timeframe)
            return None

        # Return the most recent model
        latest_model = sorted(model_files)[-1]
        _logger.info("Found latest model: %s", latest_model)
        return latest_model

    def load_model(self, model_path: Path) -> Dict:
        """
        Load trained LSTM model and metadata.

        Args:
            model_path: Path to model pickle file

        Returns:
            Dict containing model, scalers, features, and metadata
        """
        try:
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)

            # Recreate model
            arch = model_package['model_architecture']
            model = LSTMModel(
                input_size=arch['input_size'],
                hidden_size=arch['hidden_size'],
                num_layers=arch['num_layers'],
                n_regimes=arch['n_regimes']
            )

            # Load state dict
            model.load_state_dict(model_package['model_state_dict'])
            model.to(DEVICE)
            model.eval()

            model_package['model'] = model

            _logger.info("Loaded LSTM model from %s", model_path)
            _logger.info("Model features: %d", len(model_package['features']))

            return model_package

        except Exception:
            _logger.exception("Failed to load model from %s: ", model_path)
            raise

    def load_optimization_parameters(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """
        Load optimization parameters for indicators and LSTM.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dict with optimization parameters or None if not found
        """
        # Load indicator optimization results
        indicator_pattern = f"indicators_{symbol}_{timeframe}_*.json"
        indicator_files = list(self.results_dir.glob(indicator_pattern))

        indicator_params = None
        if indicator_files:
            indicator_file = sorted(indicator_files)[-1]
            with open(indicator_file, 'r') as f:
                results = json.load(f)
            indicator_params = results['best_params']
            _logger.info("Loaded indicator parameters from %s", indicator_file)
        else:
            _logger.warning("No indicator optimization results found for %s %s", symbol, timeframe)

        # Load LSTM optimization results
        lstm_pattern = f"lstm_params_{symbol}_{timeframe}_*.json"
        lstm_files = list(self.results_dir.glob(lstm_pattern))

        lstm_params = None
        if lstm_files:
            lstm_file = sorted(lstm_files)[-1]
            with open(lstm_file, 'r') as f:
                results = json.load(f)
            lstm_params = results['best_params']
            _logger.info("Loaded LSTM parameters from %s", lstm_file)
        else:
            _logger.warning("No LSTM optimization results found for %s %s", symbol, timeframe)

        return {
            'indicator_params': indicator_params,
            'lstm_params': lstm_params
        }

    def apply_optimized_indicators(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Apply optimized technical indicators to the data.

        Args:
            df: DataFrame with OHLCV data
            params: Optimized indicator parameters

        Returns:
            DataFrame with optimized indicators
        """
        df = df.copy()

        # Extract OHLCV arrays
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values

        try:
            # Apply optimized indicators

            # RSI
            if 'rsi_period' in params:
                df['rsi_optimized'] = talib.RSI(close, timeperiod=params['rsi_period'])

            # Bollinger Bands
            if 'bb_period' in params and 'bb_std' in params:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close,
                    timeperiod=params['bb_period'],
                    nbdevup=params['bb_std'],
                    nbdevdn=params['bb_std']
                )
                df['bb_upper_opt'] = bb_upper
                df['bb_middle_opt'] = bb_middle
                df['bb_lower_opt'] = bb_lower

                # Avoid division by zero in bb_position calculation
                bb_range = bb_upper - bb_lower
                mask = bb_range != 0
                bb_position = np.full_like(close, 0.5)  # Default value
                bb_position[mask] = (close[mask] - bb_lower[mask]) / bb_range[mask]
                df['bb_position_opt'] = bb_position

                # Avoid division by zero in bb_width calculation
                mask_width = bb_middle != 0
                bb_width = np.full_like(close, 0)  # Default value
                bb_width[mask_width] = bb_range[mask_width] / bb_middle[mask_width]
                df['bb_width_opt'] = bb_width

            # MACD
            if all(param in params for param in ['macd_fast', 'macd_slow', 'macd_signal']):
                macd, macd_signal, macd_hist = talib.MACD(
                    close,
                    fastperiod=params['macd_fast'],
                    slowperiod=params['macd_slow'],
                    signalperiod=params['macd_signal']
                )
                df['macd_opt'] = macd
                df['macd_signal_opt'] = macd_signal
                df['macd_histogram_opt'] = macd_hist

            # EMAs
            if 'ema_fast' in params:
                df['ema_fast_opt'] = talib.EMA(close, timeperiod=params['ema_fast'])
            if 'ema_slow' in params:
                df['ema_slow_opt'] = talib.EMA(close, timeperiod=params['ema_slow'])

            # EMA spread
            if 'ema_fast_opt' in df.columns and 'ema_slow_opt' in df.columns:
                df['ema_spread_opt'] = (df['ema_fast_opt'] - df['ema_slow_opt']) / df['close']

            # ATR
            if 'atr_period' in params:
                df['atr_opt'] = talib.ATR(high, low, close, timeperiod=params['atr_period'])

            # Stochastic
            if 'stoch_k' in params and 'stoch_d' in params:
                stoch_k, stoch_d = talib.STOCH(
                    high, low, close,
                    fastk_period=params['stoch_k'],
                    slowk_period=params['stoch_d'],
                    slowd_period=params['stoch_d']
                )
                df['stoch_k_opt'] = stoch_k
                df['stoch_d_opt'] = stoch_d

            # Williams %R
            if 'williams_period' in params:
                df['williams_r_opt'] = talib.WILLR(high, low, close, timeperiod=params['williams_period'])

            # MFI
            if 'mfi_period' in params:
                df['mfi_opt'] = talib.MFI(high, low, close, volume, timeperiod=params['mfi_period'])

            # SMA
            if 'sma_period' in params:
                df['sma_opt'] = talib.SMA(close, timeperiod=params['sma_period'])

        except Exception as e:
            _logger.exception("Error applying optimized indicators: %s", str(e))

        return df

    def prepare_test_data(self, df: pd.DataFrame, model_package: Dict) -> Dict:
        """
        Prepare test data for validation.

        Args:
            df: DataFrame with labeled data
            model_package: Loaded model package

        Returns:
            Dict with prepared test data
        """
        features = model_package['features']
        scalers = model_package['scalers']
        sequence_length = model_package['hyperparameters']['sequence_length']

        # Load optimization parameters and apply optimized indicators
        symbol = model_package.get('symbol', 'UNKNOWN')
        timeframe = model_package.get('timeframe', 'UNKNOWN')

        # Try to extract symbol and timeframe from model path if not in metadata
        if symbol == 'UNKNOWN' or timeframe == 'UNKNOWN':
            model_path = model_package.get('model_path', '')
            if model_path:
                # Extract from filename like "lstm_BTCUSDT_1h_20250815_123456.pkl"
                # Updated pattern to handle various symbol formats (including numbers)
                match = re.search(r'lstm_([A-Z0-9]+)_([0-9a-z]+)_', str(model_path))
                if match:
                    symbol = match.group(1)
                    timeframe = match.group(2)
                    _logger.info("Extracted symbol=%s, timeframe=%s from model path", symbol, timeframe)

        optimization_params = self.load_optimization_parameters(symbol, timeframe)

        if optimization_params and optimization_params['indicator_params']:
            _logger.info("Applying optimized indicators for %s %s", symbol, timeframe)
            df = self.apply_optimized_indicators(df, optimization_params['indicator_params'])
        else:
            _logger.warning("No optimization parameters found, using existing features")

        # Check if all required features are available
        missing_features = [feat for feat in features if feat not in df.columns]
        if missing_features:
            _logger.warning("Missing features after applying optimized indicators: %s", missing_features)
            _logger.warning("Available features: %s", [col for col in df.columns if any(indicator in col for indicator in ['rsi', 'bb_', 'macd', 'ema_', 'atr', 'stoch', 'williams', 'mfi', 'sma'])])
            raise ValueError(f"Missing required features: {missing_features}")

        # Extract features and regimes
        feature_data = df[features].ffill().bfill().fillna(0)

        # Handle infinite and extremely large values
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        feature_data = feature_data.ffill().bfill()
        feature_data = feature_data.fillna(0)

        # Clip extreme values to prevent numerical issues
        for col in feature_data.columns:
            if feature_data[col].dtype in ['float64', 'float32']:
                # Get the 1st and 99th percentiles
                q1 = feature_data[col].quantile(0.01)
                q99 = feature_data[col].quantile(0.99)
                # Clip values outside this range
                feature_data[col] = feature_data[col].clip(lower=q1, upper=q99)

        regime_data = df['regime'].fillna(0).astype(int)
        target_data = df['log_return'].shift(-1).fillna(0)

        # Create sequences
        X, regime_onehot, y = self.create_sequences(
            feature_data.values, regime_data.values, target_data.values, sequence_length
        )

        # Use test split (same as training)
        test_size = self.config['evaluation']['test_split']
        split_idx = int(len(X) * (1 - test_size))

        X_test = X[split_idx:]
        regime_test = regime_onehot[split_idx:]
        y_test = y[split_idx:]

        # Scale test data using fitted scalers
        feature_scaler = scalers['feature_scaler']
        target_scaler = scalers['target_scaler']

        X_test_reshaped = X_test.reshape(-1, X_test.shape[2])
        X_test_scaled = feature_scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)

        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

        return {
            'X_test': X_test_scaled,
            'regime_test': regime_test,
            'y_test': y_test_scaled,
            'y_test_original': y_test,
            'target_scaler': target_scaler,
            'test_start_idx': split_idx + sequence_length  # For aligning with original data
        }

    def create_sequences(self, data: np.ndarray, regimes: np.ndarray, target: np.ndarray,
                        sequence_length: int, n_regimes: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences for LSTM prediction."""
        X, regime_onehot, y = [], [], []

        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])

            # One-hot encode the current regime
            current_regime = int(regimes[i])
            onehot = np.zeros(n_regimes)
            if 0 <= current_regime < n_regimes:
                onehot[current_regime] = 1
            regime_onehot.append(onehot)

            y.append(target[i])

        return np.array(X), np.array(regime_onehot), np.array(y)

    def make_predictions(self, model: nn.Module, test_data: Dict) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            model: Trained LSTM model
            test_data: Prepared test data

        Returns:
            Predictions in original scale
        """
        model.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(test_data['X_test'], dtype=torch.float32).to(DEVICE)
            regime_tensor = torch.tensor(test_data['regime_test'], dtype=torch.float32).to(DEVICE)

            predictions_scaled = model(X_tensor, regime_tensor).squeeze().cpu().numpy()

            # Convert back to original scale
            predictions = test_data['target_scaler'].inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).flatten()

        return predictions

    def calculate_baseline_predictions(self, df: pd.DataFrame, test_start_idx: int) -> np.ndarray:
        """
        Calculate naive baseline predictions (previous log return).

        Args:
            df: Original DataFrame
            test_start_idx: Starting index for test data

        Returns:
            Baseline predictions
        """
        # Naive prediction: use previous log return
        test_data = df.iloc[test_start_idx:].copy()
        baseline_predictions = test_data['log_return'].shift(1).fillna(0).values

        # Remove the first element to align with LSTM predictions
        if len(baseline_predictions) > 0:
            baseline_predictions = baseline_predictions[1:]

        return baseline_predictions

    def calculate_performance_metrics(self, predictions: np.ndarray, actual: np.ndarray,
                                    baseline_predictions: np.ndarray) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Args:
            predictions: Model predictions
            actual: Actual values
            baseline_predictions: Naive baseline predictions

        Returns:
            Dict with performance metrics
        """
        # Align arrays (handle any length differences)
        min_len = min(len(predictions), len(actual), len(baseline_predictions))
        pred = predictions[:min_len]
        act = actual[:min_len]
        base = baseline_predictions[:min_len]

        # Basic regression metrics
        mse_lstm = mean_squared_error(act, pred)
        mse_baseline = mean_squared_error(act, base)

        mae_lstm = mean_absolute_error(act, pred)
        mae_baseline = mean_absolute_error(act, base)

        rmse_lstm = np.sqrt(mse_lstm)
        rmse_baseline = np.sqrt(mse_baseline)

        # R-squared
        r2_lstm = r2_score(act, pred)
        r2_baseline = r2_score(act, base)

        # Directional accuracy
        dir_acc_lstm = self.calculate_directional_accuracy(pred, act)
        dir_acc_baseline = self.calculate_directional_accuracy(base, act)

        # Hit rate (percentage of predictions within certain tolerance)
        tolerance = np.std(act) * 0.5
        hit_rate_lstm = np.mean(np.abs(pred - act) <= tolerance) * 100
        hit_rate_baseline = np.mean(np.abs(base - act) <= tolerance) * 100

        # Sharpe-like ratio (mean return / volatility)
        sharpe_lstm = np.mean(pred) / (np.std(pred) + 1e-8)
        sharpe_baseline = np.mean(base) / (np.std(base) + 1e-8)

        # Performance improvement
        mse_improvement = (mse_baseline - mse_lstm) / mse_baseline * 100
        dir_acc_improvement = dir_acc_lstm - dir_acc_baseline

        return {
            'lstm_metrics': {
                'mse': mse_lstm,
                'mae': mae_lstm,
                'rmse': rmse_lstm,
                'r2': r2_lstm,
                'directional_accuracy': dir_acc_lstm,
                'hit_rate': hit_rate_lstm,
                'sharpe_ratio': sharpe_lstm
            },
            'baseline_metrics': {
                'mse': mse_baseline,
                'mae': mae_baseline,
                'rmse': rmse_baseline,
                'r2': r2_baseline,
                'directional_accuracy': dir_acc_baseline,
                'hit_rate': hit_rate_baseline,
                'sharpe_ratio': sharpe_baseline
            },
            'improvements': {
                'mse_improvement_pct': mse_improvement,
                'directional_accuracy_improvement': dir_acc_improvement,
                'mae_improvement_pct': (mae_baseline - mae_lstm) / mae_baseline * 100,
                'r2_improvement': r2_lstm - r2_baseline
            },
            'sample_size': min_len
        }

    def calculate_directional_accuracy(self, predictions: np.ndarray, actual: np.ndarray) -> float:
        """Calculate directional accuracy (percentage of correct direction predictions)."""
        pred_direction = np.sign(predictions)
        true_direction = np.sign(actual)

        correct_directions = (pred_direction == true_direction).sum()
        accuracy = correct_directions / len(predictions) * 100

        return accuracy

    def analyze_regime_performance(self, predictions: np.ndarray, actual: np.ndarray,
                                 regimes: np.ndarray) -> Dict:
        """
        Analyze performance by market regime.

        Args:
            predictions: Model predictions
            actual: Actual values
            regimes: Regime labels

        Returns:
            Dict with regime-specific performance
        """
        regime_performance = {}

        unique_regimes = np.unique(regimes)

        for regime in unique_regimes:
            regime_mask = regimes == regime

            if np.sum(regime_mask) < 5:  # Skip if too few samples
                continue

            regime_pred = predictions[regime_mask]
            regime_actual = actual[regime_mask]

            regime_performance[f'regime_{int(regime)}'] = {
                'sample_count': int(np.sum(regime_mask)),
                'mse': mean_squared_error(regime_actual, regime_pred),
                'mae': mean_absolute_error(regime_actual, regime_pred),
                'directional_accuracy': self.calculate_directional_accuracy(regime_pred, regime_actual),
                'r2': r2_score(regime_actual, regime_pred),
                'mean_prediction': float(np.mean(regime_pred)),
                'mean_actual': float(np.mean(regime_actual)),
                'volatility_prediction': float(np.std(regime_pred)),
                'volatility_actual': float(np.std(regime_actual))
            }

        return regime_performance

    def create_visualizations(self, df: pd.DataFrame, predictions: np.ndarray, actual: np.ndarray,
                            baseline_predictions: np.ndarray, regimes: np.ndarray,
                            test_start_idx: int, symbol: str, timeframe: str) -> Dict[str, plt.Figure]:
        """
        Create comprehensive visualizations.

        Args:
            df: Original DataFrame
            predictions: Model predictions
            actual: Actual values
            baseline_predictions: Baseline predictions
            regimes: Regime labels
            test_start_idx: Test data start index
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dict with figure names as keys and matplotlib figures as values
        """
        figures = {}

        # Align data for plotting
        min_len = min(len(predictions), len(actual), len(baseline_predictions))
        pred = predictions[:min_len]
        act = actual[:min_len]
        base = baseline_predictions[:min_len]
        reg = regimes[:min_len]

        # Create time index for test period
        test_data = df.iloc[test_start_idx:test_start_idx + min_len].copy()
        if 'timestamp' in test_data.columns:
            time_index = pd.to_datetime(test_data['timestamp'])
        else:
            time_index = range(len(test_data))

        # Figure 1: Predictions vs Actual - Time Series
        fig1, ax = plt.subplots(1, 1, figsize=(60, 30))
        ax.plot(time_index, act, label='Actual', color='red', alpha=0.8, linewidth=1.5)
        ax.plot(time_index, pred, label='LSTM Prediction', color='blue', alpha=0.8, linewidth=1.5)
        ax.plot(time_index, base, label='Naive Baseline', color='orange', alpha=0.7, linewidth=1.5)
        ax.set_title(f'Log Return Predictions Over Time - {symbol} {timeframe}', fontsize=24)
        ax.set_ylabel('Log Return', fontsize=20)
        ax.set_xlabel('Time', fontsize=20)
        ax.legend(fontsize=16)  # Twice bigger than default (8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures['predictions_time_series'] = fig1

        # Figure 2: Scatter Plot
        fig2, ax = plt.subplots(1, 1, figsize=(60, 30))
        ax.scatter(act, pred, alpha=0.7, label='LSTM', s=30, color='blue')
        ax.scatter(act, base, alpha=0.6, label='Baseline', s=30, color='red')

        # Perfect prediction line
        min_val, max_val = min(np.min(act), np.min(pred)), max(np.max(act), np.max(pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='Perfect Prediction')

        ax.set_xlabel('Actual Log Return', fontsize=20)
        ax.set_ylabel('Predicted Log Return', fontsize=20)
        ax.set_title(f'Prediction Accuracy Scatter Plot - {symbol} {timeframe}', fontsize=24)
        ax.legend(fontsize=16)  # Twice bigger than default (8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures['predictions_scatter'] = fig2

        # Figure 3: Error Distribution
        fig3, ax = plt.subplots(1, 1, figsize=(60, 30))
        lstm_errors = pred - act
        baseline_errors = base - act

        ax.hist(lstm_errors, bins=50, alpha=0.7, label='LSTM Errors', density=True, color='blue')
        ax.hist(baseline_errors, bins=50, alpha=0.5, label='Baseline Errors', density=True, color='red')
        ax.set_title(f'Error Distribution - {symbol} {timeframe}', fontsize=24)
        ax.set_xlabel('Prediction Error', fontsize=20)
        ax.set_ylabel('Density', fontsize=20)
        ax.legend(fontsize=16)  # Twice bigger than default (8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures['error_distribution'] = fig3

        # Figure 4: Error Over Time
        fig4, ax = plt.subplots(1, 1, figsize=(60, 30))
        ax.plot(time_index, np.abs(lstm_errors), label='LSTM |Error|', alpha=0.8, linewidth=1.5, color='blue')
        ax.plot(time_index, np.abs(baseline_errors), label='Baseline |Error|', alpha=0.8, linewidth=1.5, color='red')
        ax.set_title(f'Absolute Error Over Time - {symbol} {timeframe}', fontsize=24)
        ax.set_xlabel('Time', fontsize=20)
        ax.set_ylabel('Absolute Error', fontsize=20)
        ax.legend(fontsize=16)  # Twice bigger than default (8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures['error_over_time'] = fig4

        # Figure 5: Cumulative Squared Error
        fig5, ax = plt.subplots(1, 1, figsize=(60, 30))
        cumulative_lstm_error = np.cumsum(lstm_errors ** 2)
        cumulative_baseline_error = np.cumsum(baseline_errors ** 2)

        ax.plot(time_index, cumulative_lstm_error, label='LSTM Cumulative SE', linewidth=1.5, color='blue')
        ax.plot(time_index, cumulative_baseline_error, label='Baseline Cumulative SE', linewidth=1.5, color='red')
        ax.set_title(f'Cumulative Squared Error - {symbol} {timeframe}', fontsize=24)
        ax.set_xlabel('Time', fontsize=20)
        ax.set_ylabel('Cumulative Squared Error', fontsize=20)
        ax.legend(fontsize=16)  # Twice bigger than default (8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures['cumulative_error'] = fig5

        # Figure 6: Predictions by Market Regime
        fig6, ax = plt.subplots(1, 1, figsize=(60, 30))
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
        for regime_id in np.unique(reg):
            regime_mask = reg == regime_id
            if np.sum(regime_mask) > 0:
                color = colors[int(regime_id) % len(colors)]
                ax.scatter(act[regime_mask], pred[regime_mask],
                          c=color, alpha=0.7, s=30, label=f'Regime {int(regime_id)}')

        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2)
        ax.set_xlabel('Actual', fontsize=20)
        ax.set_ylabel('Predicted', fontsize=20)
        ax.set_title(f'Predictions by Market Regime - {symbol} {timeframe}', fontsize=24)
        ax.legend(fontsize=16)  # Twice bigger than default (8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures['predictions_by_regime'] = fig6

        # Figure 7: Rolling Performance Metrics
        fig7, ax = plt.subplots(1, 1, figsize=(60, 30))
        window = min(50, len(pred) // 10)
        rolling_mse_lstm = pd.Series(lstm_errors ** 2).rolling(window).mean()
        rolling_mse_baseline = pd.Series(baseline_errors ** 2).rolling(window).mean()

        ax.plot(time_index, rolling_mse_lstm, label=f'LSTM Rolling MSE (window={window})', linewidth=1.5, color='blue')
        ax.plot(time_index, rolling_mse_baseline, label=f'Baseline Rolling MSE (window={window})', linewidth=1.5, color='red')
        ax.set_title(f'Rolling Mean Squared Error Comparison - {symbol} {timeframe}', fontsize=24)
        ax.set_xlabel('Time', fontsize=20)
        ax.set_ylabel('Rolling MSE', fontsize=20)
        ax.legend(fontsize=16)  # Twice bigger than default (8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures['rolling_performance'] = fig7

        return figures

    def save_png_files(self, figures: Dict[str, plt.Figure], symbol: str, timeframe: str) -> Dict[str, str]:
        """
        Save individual PNG files for each visualization.

        Args:
            figures: Dict with figure names as keys and matplotlib figures as values
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dict with figure names as keys and PNG file paths as values
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_files = {}

        # Create a subdirectory for this validation run
        validation_dir = self.reports_dir / f"lstm_validation_{symbol}_{timeframe}_{timestamp}"
        validation_dir.mkdir(parents=True, exist_ok=True)

        for figure_name, fig in figures.items():
            png_filename = f"{figure_name}_{symbol}_{timeframe}_{timestamp}.png"
            png_path = validation_dir / png_filename

            try:
                # Save with high DPI for better quality
                fig.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
                png_files[figure_name] = str(png_path)
                _logger.info("Saved PNG: %s", png_path)
            except Exception as e:
                _logger.exception("Failed to save PNG %s: %s", png_path, str(e))
            finally:
                # Ensure figure is closed even if save fails
                plt.close(fig)

        return png_files

    def validate_lstm(self, symbol: str, timeframe: str) -> Dict:
        """
        Validate LSTM model for a specific symbol-timeframe combination.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dict with validation results
        """
        _logger.info("Validating LSTM for %s %s", symbol, timeframe)

        try:
            # Find and load model
            model_path = self.find_latest_model(symbol, timeframe)
            if model_path is None:
                raise FileNotFoundError(f"No trained LSTM model found for {symbol} {timeframe}")

            model_package = self.load_model(model_path)
            # Store symbol and timeframe in model package for later use
            model_package['symbol'] = symbol
            model_package['timeframe'] = timeframe

            # Find labeled data file - look for both patterns
            patterns = [
                f"labeled_{symbol}_{timeframe}_*.csv",  # Original pattern
                f"*_{symbol}_{timeframe}_*_labeled.csv"  # New pattern with provider prefix
            ]

            csv_files = []
            for pattern in patterns:
                csv_files.extend(list(self.labeled_data_dir.glob(pattern)))

            if not csv_files:
                raise FileNotFoundError(f"No labeled data found for {symbol} {timeframe}")

            # Use the most recent file
            csv_file = sorted(csv_files)[-1]
            _logger.info("Using data file: %s", csv_file)

            # Load data
            df = pd.read_csv(csv_file)

            # Prepare test data
            test_data = self.prepare_test_data(df, model_package)

            # Make predictions
            predictions = self.make_predictions(model_package['model'], test_data)

            # Calculate baseline predictions
            baseline_predictions = self.calculate_baseline_predictions(df, test_data['test_start_idx'])

            # Ensure arrays are aligned
            min_len = min(len(predictions), len(test_data['y_test_original']), len(baseline_predictions))
            predictions = predictions[:min_len]
            actual = test_data['y_test_original'][:min_len]
            baseline_predictions = baseline_predictions[:min_len]

            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(predictions, actual, baseline_predictions)

            # Analyze regime performance
            regime_performance = self.analyze_regime_performance(
                predictions, actual, test_data['regime_test'][:min_len, 0]  # Get regime class
            )

            # Create visualizations
            figures = self.create_visualizations(
                df, predictions, actual, baseline_predictions,
                test_data['regime_test'][:min_len, 0], test_data['test_start_idx'],
                symbol, timeframe
            )

            # Save PNG files
            png_files = self.save_png_files(figures, symbol, timeframe)

            # Save results to JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'validation_timestamp': timestamp,
                'model_path': str(model_path),
                'data_file': str(csv_file),
                'performance_metrics': metrics,
                'regime_performance': regime_performance,
                'png_files': png_files,
                'model_metadata': {
                    'architecture': model_package['model_architecture'],
                    'hyperparameters': model_package['hyperparameters'],
                    'features_count': len(model_package['features']),
                    'training_results': model_package['training_results']
                }
            }

            json_filename = f"lstm_validation_{symbol}_{timeframe}_{timestamp}.json"
            json_path = self.results_dir / json_filename

            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            # Convert results for JSON serialization
            json_results = convert_numpy_types(results)

            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)

            _logger.info("[OK] LSTM validation completed for %s %s", symbol, timeframe)
            _logger.info("  MSE improvement: %.2f%%", metrics['improvements']['mse_improvement_pct'])
            _logger.info("  Directional accuracy: %.2f%%", metrics['lstm_metrics']['directional_accuracy'])
            _logger.info("  PNG files: %d files saved", len(png_files))
            _logger.info("  JSON results: %s", json_path)

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': True,
                'png_files': png_files,
                'json_results': str(json_path),
                'mse_improvement': metrics['improvements']['mse_improvement_pct'],
                'directional_accuracy': metrics['lstm_metrics']['directional_accuracy'],
                'sample_size': metrics['sample_size']
            }

        except Exception as e:
            error_msg = f"Failed to validate LSTM for {symbol} {timeframe}: {str(e)}"
            _logger.exception(error_msg)
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': False,
                'error': error_msg
            }

    def validate_all(self) -> Dict:
        """
        Validate LSTM models for all symbol-timeframe combinations.

        Returns:
            Dict with summary of validation results
        """
        # Extract symbols and timeframes from data_sources configuration
        symbols = []
        timeframes = []

        for provider, config in self.config.get('data_sources', {}).items():
            symbols.extend(config.get('symbols', []))
            timeframes.extend(config.get('timeframes', []))

        # Remove duplicates while preserving order
        symbols = list(dict.fromkeys(symbols))
        timeframes = list(dict.fromkeys(timeframes))

        _logger.info("Validating LSTM models for %d symbols x %d timeframes", len(symbols), len(timeframes))

        results = {
            'total': len(symbols) * len(timeframes),
            'successful': [],
            'failed': []
        }

        for symbol in symbols:
            for timeframe in timeframes:
                result = self.validate_lstm(symbol, timeframe)

                if result['success']:
                    results['successful'].append(result)
                else:
                    results['failed'].append(result)

        # Log summary
        _logger.info("\n%s", "="*50)
        _logger.info("LSTM Validation Summary:")
        _logger.info("  Total: %d", results['total'])
        _logger.info("  Successful: %d", len(results['successful']))
        _logger.info("  Failed: %d", len(results['failed']))

        if results['successful']:
            avg_mse_improvement = np.mean([r['mse_improvement'] for r in results['successful']])
            avg_dir_accuracy = np.mean([r['directional_accuracy'] for r in results['successful']])
            _logger.info("  Average MSE improvement: %.2f%%", avg_mse_improvement)
            _logger.info("  Average directional accuracy: %.2f%%", avg_dir_accuracy)

        _logger.info("%s", "="*50)

        if results['failed']:
            _logger.warning("Failed validations:")
            for failure in results['failed']:
                _logger.warning("  %s %s: %s", failure['symbol'], failure['timeframe'], failure['error'])

        return results

def main():
    """Main function to run LSTM validation."""
    try:
        validator = LSTMValidator()
        results = validator.validate_all()

        _logger.info("LSTM validation completed!")

    except Exception:
        _logger.exception("LSTM validation failed: ")
        raise

if __name__ == "__main__":
    main()
