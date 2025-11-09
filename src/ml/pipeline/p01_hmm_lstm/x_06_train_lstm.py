"""
LSTM Model Training with Optimized Parameters

This module trains the final LSTM model using the best hyperparameters found
by Optuna optimization. It loads optimized indicator parameters and LSTM
hyperparameters, trains the model on the full dataset, and saves the trained
model for later use in validation and prediction.

Features:
- Uses optimized hyperparameters from Optuna
- Applies optimized technical indicators
- Trains on full dataset with proper validation split
- Implements early stopping and learning rate scheduling
- Saves trained models with comprehensive metadata
- Supports regime-aware LSTM with one-hot encoding
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from pathlib import Path
import json
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import talib
import sys
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_logger.info("Using device: %s", DEVICE)

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

class LSTMTrainer:
    def __init__(self, config_path: str = "config/pipeline/p01.yaml"):
        """
        Initialize LSTM trainer with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.labeled_data_dir = Path(self.config['paths']['data_labeled'])
        self.results_dir = Path(self.config['paths']['models_lstm'])
        self.models_dir = Path(self.config['paths']['models_lstm'])
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        _logger.info("Loaded configuration from %s", self.config_path)
        return config

    def _get_checkpoint_path(self, symbol: str, timeframe: str) -> Path:
        """
        Build checkpoint filename based on labeled CSV file.
        """
        fname = f"checkpoint_{symbol}_{timeframe}.pth"
        return self.models_dir / fname

    def save_checkpoint(self, model, optimizer, epoch, train_losses, val_losses,
                        symbol: str, timeframe: str):
        """Save model/optimizer state so training can resume later."""
        ckpt_path = self._get_checkpoint_path(symbol, timeframe)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses
        }, ckpt_path)
        _logger.info("Checkpoint saved at %s (epoch %d)", ckpt_path, epoch+1)

    def load_checkpoint(self, model, optimizer, symbol: str, timeframe: str):
        """Load model/optimizer state to resume training if checkpoint exists."""
        ckpt_path = self._get_checkpoint_path(symbol, timeframe)
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            _logger.info("Resuming from checkpoint %s (epoch %d)", ckpt_path, start_epoch)
            return start_epoch, checkpoint['train_losses'], checkpoint['val_losses']
        else:
            return 0, [], []

    def load_optimization_results(self, symbol: str, timeframe: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Load optimization results for indicators and LSTM parameters.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Tuple of (indicator_params, lstm_params) or None if not found
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
            # Use default parameters from config
            lstm_params = {
                'sequence_length': self.config['lstm']['sequence_length'],
                'hidden_size': self.config['lstm']['hidden_size'],
                'num_layers': self.config['lstm']['num_layers'],
                'dropout': self.config['lstm']['dropout'],
                'learning_rate': self.config['lstm']['learning_rate'],
                'batch_size': self.config['lstm']['batch_size'],
                'epochs': self.config['lstm']['epochs']
            }
            _logger.info("Using default LSTM parameters: %s", lstm_params)

        return indicator_params, lstm_params

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
            # Apply optimized indicators (same as in x_06_optuna_lstm.py)

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
                df['bb_position_opt'] = (close - bb_lower) / (bb_upper - bb_lower)
                df['bb_width_opt'] = (bb_upper - bb_lower) / bb_middle

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
            _logger.warning("Error applying optimized indicators: %s", str(e))

        return df

    def prepare_lstm_features(self, df: pd.DataFrame, exclude_regime: bool = False) -> List[str]:
        """
        Prepare feature columns for LSTM training.
        Only uses optimized indicators to avoid timeframe noise from baseline indicators.

        Args:
            df: DataFrame with all features
            exclude_regime: Whether to exclude regime from features (it will be handled separately)

        Returns:
            List of selected feature column names
        """
        # Base OHLCV features
        base_features = ['open', 'high', 'low', 'close', 'volume', 'log_return']

        # Regime features (exclude the categorical regime itself)
        regime_features = ['regime_confidence', 'regime_duration']
        if not exclude_regime:
            regime_features.append('regime')

        # Optimized indicator features (prioritize these)
        optimized_features = [col for col in df.columns if col.endswith('_opt')]

        # Time features
        time_features = [col for col in df.columns if any(t in col for t in ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos'])]

        # Only use optimized indicators - no baseline indicators to avoid timeframe noise
        # Additional technical features (only if they're optimized)
        additional_features = []
        for col in df.columns:
            if any(indicator in col for indicator in ['rsi', 'bb_', 'macd', 'ema_', 'atr', 'stoch', 'williams', 'mfi', 'sma']):
                if col.endswith('_opt') and col not in optimized_features:
                    additional_features.append(col)

        # Combine features (only optimized indicators to avoid timeframe noise)
        selected_features = (base_features +
                           regime_features +
                           optimized_features +
                           time_features +
                           additional_features[:5])  # Limit additional optimized features

        # Filter to only include features that exist in the DataFrame
        available_features = [feat for feat in selected_features if feat in df.columns]

        _logger.info("Selected %d features for LSTM: %s...", len(available_features), available_features[:10])
        return available_features

    def create_sequences(self, data: np.ndarray, regimes: np.ndarray, target: np.ndarray,
                        sequence_length: int, n_regimes: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training with regime information.

        Args:
            data: Feature matrix
            regimes: Regime labels
            target: Target values
            sequence_length: Length of input sequences
            n_regimes: Number of regime classes

        Returns:
            Tuple of (X sequences, regime one-hot, y targets)
        """
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

    def prepare_data(self, df: pd.DataFrame, features: List[str],
                    sequence_length: int, test_size: float = 0.1) -> Dict:
        """
        Prepare data for LSTM training.

        Args:
            df: DataFrame with features
            features: List of feature column names (excluding regime)
            sequence_length: Length of input sequences
            test_size: Proportion of data for testing

        Returns:
            Dict with prepared data splits
        """
        # Extract features, regimes, and target
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

        # Target is next period's log return
        target_data = df['log_return'].shift(-1).fillna(0)  # Predict next period

        # Create sequences
        X, regime_onehot, y = self.create_sequences(
            feature_data.values, regime_data.values, target_data.values, sequence_length
        )

        if len(X) < 100:
            raise ValueError(f"Not enough data after sequence creation: {len(X)} samples")

        # Split data (preserve temporal order)
        split_idx = int(len(X) * (1 - test_size))

        X_train = X[:split_idx]
        regime_train = regime_onehot[:split_idx]
        y_train = y[:split_idx]

        X_test = X[split_idx:]
        regime_test = regime_onehot[split_idx:]
        y_test = y[split_idx:]

        # Further split training data for validation
        val_split_idx = int(len(X_train) * 0.8)

        X_val = X_train[val_split_idx:]
        regime_val = regime_train[val_split_idx:]
        y_val = y_train[val_split_idx:]

        X_train = X_train[:val_split_idx]
        regime_train = regime_train[:val_split_idx]
        y_train = y_train[:val_split_idx]

        _logger.info("Data splits - Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))

        return {
            'X_train': X_train,
            'regime_train': regime_train,
            'y_train': y_train,
            'X_val': X_val,
            'regime_val': regime_val,
            'y_val': y_val,
            'X_test': X_test,
            'regime_test': regime_test,
            'y_test': y_test,
            'n_features': X.shape[2]
        }

    def scale_data(self, data_dict: Dict) -> Tuple[Dict, Dict]:
        """
        Scale features and targets.

        Args:
            data_dict: Dictionary with data splits

        Returns:
            Tuple of (scaled data dict, scalers dict)
        """
        # Scale features
        feature_scaler = StandardScaler()

        # Reshape for scaling
        X_train_reshaped = data_dict['X_train'].reshape(-1, data_dict['n_features'])
        X_train_scaled = feature_scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(data_dict['X_train'].shape)

        X_val_reshaped = data_dict['X_val'].reshape(-1, data_dict['n_features'])
        X_val_scaled = feature_scaler.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled.reshape(data_dict['X_val'].shape)

        X_test_reshaped = data_dict['X_test'].reshape(-1, data_dict['n_features'])
        X_test_scaled = feature_scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled.reshape(data_dict['X_test'].shape)

        # Scale targets
        target_scaler = MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(data_dict['y_train'].reshape(-1, 1)).flatten()
        y_val_scaled = target_scaler.transform(data_dict['y_val'].reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(data_dict['y_test'].reshape(-1, 1)).flatten()

        scaled_data = {
            'X_train': X_train_scaled,
            'regime_train': data_dict['regime_train'],  # Don't scale one-hot
            'y_train': y_train_scaled,
            'X_val': X_val_scaled,
            'regime_val': data_dict['regime_val'],
            'y_val': y_val_scaled,
            'X_test': X_test_scaled,
            'regime_test': data_dict['regime_test'],
            'y_test': y_test_scaled,
            'n_features': data_dict['n_features']
        }

        scalers = {
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        }

        return scaled_data, scalers

    def create_data_loaders(self, data_dict: Dict, batch_size: int) -> Dict:
        """
        Create PyTorch data loaders.

        Args:
            data_dict: Dictionary with scaled data
            batch_size: Batch size for training

        Returns:
            Dict with data loaders
        """
        # Convert to tensors
        X_train_tensor = torch.tensor(data_dict['X_train'], dtype=torch.float32)
        regime_train_tensor = torch.tensor(data_dict['regime_train'], dtype=torch.float32)
        y_train_tensor = torch.tensor(data_dict['y_train'], dtype=torch.float32)

        X_val_tensor = torch.tensor(data_dict['X_val'], dtype=torch.float32)
        regime_val_tensor = torch.tensor(data_dict['regime_val'], dtype=torch.float32)
        y_val_tensor = torch.tensor(data_dict['y_val'], dtype=torch.float32)

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, regime_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, regime_val_tensor, y_val_tensor)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return {
            'train_loader': train_loader,
            'val_loader': val_loader
        }

    def train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                   learning_rate: float, epochs: int, symbol: str = "LTCUSDT", timeframe: str = "15m",
                   early_stopping_patience: int = 15) -> Dict:
        """
        Train LSTM model with checkpointing, early stopping, and LR scheduling.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Try to load checkpoint
        start_epoch, train_losses, val_losses = self.load_checkpoint(model, optimizer, symbol, timeframe)

        best_val_loss = float('inf') if not val_losses else min(val_losses)
        patience_counter = 0
        best_model_state = None

        for epoch in range(start_epoch, epochs):
            # Training loop
            model.train()
            train_loss = 0.0
            for batch_X, batch_regime, batch_y in train_loader:
                batch_X, batch_regime, batch_y = batch_X.to(DEVICE), batch_regime.to(DEVICE), batch_y.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(batch_X, batch_regime).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_regime, batch_y in val_loader:
                    batch_X, batch_regime, batch_y = batch_X.to(DEVICE), batch_regime.to(DEVICE), batch_y.to(DEVICE)
                    outputs = model(batch_X, batch_regime).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            # Save checkpoint after each epoch
            self.save_checkpoint(model, optimizer, epoch, train_losses, val_losses, symbol, timeframe)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    _logger.info("Early stopping at epoch %d", epoch+1)
                    break

            if (epoch + 1) % 5 == 0:
                _logger.info("Epoch %d/%d - Train Loss: %.6f, Val Loss: %.6f", epoch+1, epochs, train_loss, val_loss)

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1
        }

    def save_model(self, model: nn.Module, scalers: Dict, features: List[str],
                  training_results: Dict, lstm_params: Dict, symbol: str, timeframe: str) -> Path:
        """
        Save trained LSTM model and metadata.

        Args:
            model: Trained model
            scalers: Feature and target scalers
            features: List of features used
            training_results: Training results
            lstm_params: LSTM hyperparameters
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Path to saved model file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lstm_{symbol}_{timeframe}_{timestamp}.pkl"
        filepath = self.models_dir / filename

        # Prepare model package
        model_package = {
            'model_state_dict': model.state_dict(),
            'model_architecture': {
                'input_size': model.lstm.input_size,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'n_regimes': model.n_regimes
            },
            'scalers': scalers,
            'features': features,
            'training_results': training_results,
            'hyperparameters': lstm_params,
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': timestamp,
                'training_date': datetime.now().isoformat(),
                'device': str(DEVICE)
            }
        }

        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)

        _logger.info("Saved LSTM model to %s", filepath)
        return filepath

    def train_lstm(self, symbol: str, timeframe: str) -> Dict:
        """
        Train LSTM model for a specific symbol-timeframe combination.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dict with training results
        """
        _logger.info("Training LSTM for %s %s", symbol, timeframe)

        try:
            # Load optimization results
            indicator_params, lstm_params = self.load_optimization_results(symbol, timeframe)

            # Find labeled data file - try multiple patterns
            csv_files = []

            # Try provider-specific patterns first
            for provider in ['binance', 'yfinance']:
                pattern = f"{provider}_{symbol}_{timeframe}_*_labeled.csv"
                provider_files = list(self.labeled_data_dir.glob(pattern))
                csv_files.extend(provider_files)

            # Try legacy pattern as fallback
            if not csv_files:
                pattern = f"labeled_{symbol}_{timeframe}_*.csv"
                csv_files = list(self.labeled_data_dir.glob(pattern))

            # Try any file containing the symbol and timeframe
            if not csv_files:
                pattern = f"*{symbol}_{timeframe}*_labeled.csv"
                csv_files = list(self.labeled_data_dir.glob(pattern))

            if not csv_files:
                raise FileNotFoundError(f"No labeled data found for {symbol} {timeframe}")

            # Use the most recent file
            csv_file = sorted(csv_files)[-1]
            _logger.info("Using data file: %s", csv_file)

            # Load data
            df = pd.read_csv(csv_file)

            # Apply optimized indicators if available
            if indicator_params:
                df = self.apply_optimized_indicators(df, indicator_params)

            # Prepare features (exclude regime as it's handled separately)
            features = self.prepare_lstm_features(df, exclude_regime=True)

            # Prepare data
            data_dict = self.prepare_data(
                df, features,
                lstm_params['sequence_length'],
                test_size=self.config['evaluation']['test_split']
            )

            # Scale data
            scaled_data, scalers = self.scale_data(data_dict)

            # Create data loaders
            loaders = self.create_data_loaders(scaled_data, lstm_params['batch_size'])

            # Create model
            model = LSTMModel(
                input_size=scaled_data['n_features'],
                hidden_size=lstm_params['hidden_size'],
                num_layers=lstm_params['num_layers'],
                dropout=lstm_params['dropout'],
                n_regimes=3  # From HMM configuration
            ).to(DEVICE)

            _logger.info("Created LSTM model with %d parameters", sum(p.numel() for p in model.parameters()))

            # Train model
            training_results = self.train_model(
                model,
                loaders['train_loader'],
                loaders['val_loader'],
                lstm_params['learning_rate'],
                lstm_params['epochs']
            )

            # Save model
            model_path = self.save_model(
                model, scalers, features, training_results, lstm_params, symbol, timeframe
            )

            _logger.info("[OK] LSTM training completed for %s %s", symbol, timeframe)
            _logger.info("  Best validation loss: %.6f", training_results['best_val_loss'])
            _logger.info("  Training epochs: %d", training_results['final_epoch'])

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': True,
                'model_path': str(model_path),
                'best_val_loss': training_results['best_val_loss'],
                'training_epochs': training_results['final_epoch'],
                'n_parameters': sum(p.numel() for p in model.parameters()),
                'features_used': len(features)
            }

        except Exception as e:
            error_msg = f"Failed to train LSTM for {symbol} {timeframe}: {str(e)}"
            _logger.error(error_msg)
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': False,
                'error': error_msg
            }

    def _get_symbols_and_timeframes(self) -> Tuple[List[str], List[str]]:
        """
        Extract symbols and timeframes from config, supporting both legacy and multi-provider formats.

        Returns:
            Tuple of (symbols, timeframes) lists
        """
        if 'data_sources' in self.config:
            # New multi-provider format
            symbols = []
            timeframes = []

            for provider, config in self.config['data_sources'].items():
                provider_symbols = config.get('symbols', [])
                provider_timeframes = config.get('timeframes', [])

                symbols.extend(provider_symbols)
                timeframes.extend(provider_timeframes)

            # Remove duplicates while preserving order
            symbols = list(dict.fromkeys(symbols))
            timeframes = list(dict.fromkeys(timeframes))

            _logger.info("Using multi-provider configuration")
            _logger.info("  Symbols: %s", symbols)
            _logger.info("  Timeframes: %s", timeframes)

        else:
            # Legacy format
            symbols = self.config['symbols']
            timeframes = self.config['timeframes']
            _logger.info("Using legacy configuration")

        return symbols, timeframes

    def get_available_combinations(self) -> List[Tuple[str, str]]:
        """
        Get available symbol-timeframe combinations that have BOTH labeled data AND LSTM optimization results.

        Returns:
            List of (symbol, timeframe) tuples that have both data and LSTM params
        """
        available_combinations = []

        # Check all possible combinations by looking at actual files
        # First, get all labeled data files
        labeled_files = list(self.labeled_data_dir.glob("*_labeled.csv"))

        for labeled_file in labeled_files:
            # Extract symbol and timeframe from filename
            filename = labeled_file.name

            # Parse provider_symbol_timeframe pattern
            if filename.startswith(('binance_', 'yfinance_')):
                # Format: provider_symbol_timeframe_date_labeled.csv
                parts = filename.replace('_labeled.csv', '').split('_')
                if len(parts) >= 3:
                    provider = parts[0]
                    symbol = parts[1]
                    timeframe = parts[2]

                    # Check if LSTM params exist for this combination
                    lstm_pattern = f"lstm_params_{symbol}_{timeframe}_*.json"
                    lstm_files = list(self.results_dir.glob(lstm_pattern))

                    if lstm_files:
                        available_combinations.append((symbol, timeframe))
                        _logger.info("Found complete setup for %s %s: data=%s, lstm_params=%s",
                                   symbol, timeframe, labeled_file.name, lstm_files[0].name)
                    else:
                        _logger.debug("Missing LSTM params for %s %s (has data: %s)",
                                    symbol, timeframe, labeled_file.name)
            else:
                # Try legacy pattern: labeled_symbol_timeframe_date.csv
                if filename.startswith('labeled_'):
                    parts = filename.replace('labeled_', '').replace('.csv', '').split('_')
                    if len(parts) >= 2:
                        symbol = parts[0]
                        timeframe = parts[1]

                        # Check if LSTM params exist
                        lstm_pattern = f"lstm_params_{symbol}_{timeframe}_*.json"
                        lstm_files = list(self.results_dir.glob(lstm_pattern))

                        if lstm_files:
                            available_combinations.append((symbol, timeframe))
                            _logger.info("Found complete setup for %s %s: data=%s, lstm_params=%s",
                                       symbol, timeframe, labeled_file.name, lstm_files[0].name)

        # Remove duplicates while preserving order
        unique_combinations = []
        seen = set()
        for combo in available_combinations:
            if combo not in seen:
                unique_combinations.append(combo)
                seen.add(combo)

        return unique_combinations

    def train_all(self) -> Dict:
        """
        Train LSTM models for all available symbol-timeframe combinations.

        Returns:
            Dict with summary of training results
        """
        available_combinations = self.get_available_combinations()

        if not available_combinations:
            _logger.error("No labeled data found for any symbol-timeframe combination")
            return {
                'total': 0,
                'successful': [],
                'failed': [],
                'error': 'No labeled data available'
            }

        _logger.info("Training LSTM models for %d available combinations", len(available_combinations))

        results = {
            'total': len(available_combinations),
            'successful': [],
            'failed': []
        }

        for symbol, timeframe in available_combinations:
            result = self.train_lstm(symbol, timeframe)

            if result['success']:
                results['successful'].append(result)
            else:
                results['failed'].append(result)

        # Log summary
        _logger.info("\n%s", "="*50)
        _logger.info("LSTM Training Summary:")
        _logger.info("  Total: %d", results['total'])
        _logger.info("  Successful: %d", len(results['successful']))
        _logger.info("  Failed: %d", len(results['failed']))
        _logger.info("%s", "="*50)

        if results['failed']:
            _logger.warning("Failed training:")
            for failure in results['failed']:
                _logger.warning("  %s %s: %s", failure['symbol'], failure['timeframe'], failure['error'])

        return results

def main():
    """Main function to run LSTM training."""
    try:
        trainer = LSTMTrainer()
        results = trainer.train_all()

        _logger.info("LSTM training completed!")

    except Exception:
        _logger.exception("LSTM training failed: ")
        raise

if __name__ == "__main__":
    main()
