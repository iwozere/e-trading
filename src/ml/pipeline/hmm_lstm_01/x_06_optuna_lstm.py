"""
Optuna LSTM Hyperparameter Optimization

This module optimizes LSTM model hyperparameters using Optuna for time series
prediction. It searches for optimal combinations of sequence length, hidden size,
learning rate, dropout, batch size, and other architecture parameters.

Features:
- Optimizes LSTM architecture and training parameters
- Uses regime information as additional features
- Applies optimized technical indicators from previous step
- Multi-objective optimization (MSE, directional accuracy)
- Early stopping and pruning for efficiency
- Saves best parameters for final model training
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from pathlib import Path
import logging
import json
from datetime import datetime
import optuna
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import sys
from typing import Dict, List, Optional, Tuple, Any
import talib

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float = 0.2, output_size: int = 1):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]

        # Apply dropout
        dropped = self.dropout(last_output)

        # Final prediction
        output = self.linear(dropped)

        return output

class LSTMOptimizer:
    def __init__(self, config_path: str = "config/pipeline/x01.yaml"):
        """
        Initialize LSTM optimizer with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.labeled_data_dir = Path(self.config['paths']['data_labeled'])
        self.results_dir = Path(self.config['paths']['results'])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Optuna configuration
        self.n_trials = self.config['optuna']['n_trials']
        self.timeout = self.config['optuna'].get('timeout', 3600)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {self.config_path}")
        return config

    def load_optimized_indicators(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """
        Load optimized indicator parameters from previous optimization step.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dict with optimized parameters or None if not found
        """
        pattern = f"indicators_{symbol}_{timeframe}_*.json"
        json_files = list(self.results_dir.glob(pattern))

        if not json_files:
            logger.warning(f"No optimized indicator parameters found for {symbol} {timeframe}")
            return None

        # Use the most recent file
        json_file = sorted(json_files)[-1]

        with open(json_file, 'r') as f:
            results = json.load(f)

        logger.info(f"Loaded optimized indicators from {json_file}")
        return results['best_params']

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
            logger.warning(f"Error applying optimized indicators: {str(e)}")

        return df

    def prepare_lstm_features(self, df: pd.DataFrame) -> List[str]:
        """
        Prepare feature columns for LSTM training.

        Args:
            df: DataFrame with all features

        Returns:
            List of selected feature column names
        """
        # Base OHLCV features
        base_features = ['open', 'high', 'low', 'close', 'volume', 'log_return']

        # Regime features
        regime_features = ['regime', 'regime_confidence', 'regime_duration']

        # Optimized indicator features (prioritize these)
        optimized_features = [col for col in df.columns if col.endswith('_opt')]

        # Time features
        time_features = [col for col in df.columns if any(t in col for t in ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos'])]

        # Additional technical features (fallback if optimized not available)
        additional_features = []
        for col in df.columns:
            if any(indicator in col for indicator in ['rsi', 'bb_', 'macd', 'ema_', 'atr', 'stoch', 'williams', 'mfi', 'sma']):
                if not col.endswith('_opt') and col not in optimized_features:
                    additional_features.append(col)

        # Combine features (prioritize optimized indicators)
        selected_features = (base_features +
                           regime_features +
                           optimized_features +
                           time_features +
                           additional_features[:5])  # Limit additional features

        # Filter to only include features that exist in the DataFrame
        available_features = [feat for feat in selected_features if feat in df.columns]

        logger.info(f"Selected {len(available_features)} features for LSTM: {available_features[:10]}...")
        return available_features

    def create_sequences(self, data: np.ndarray, target: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.

        Args:
            data: Feature matrix
            target: Target values
            sequence_length: Length of input sequences

        Returns:
            Tuple of (X sequences, y targets)
        """
        X, y = [], []

        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(target[i])

        return np.array(X), np.array(y)

    def prepare_data(self, df: pd.DataFrame, features: List[str], sequence_length: int, test_size: float = 0.2) -> Dict:
        """
        Prepare data for LSTM training.

        Args:
            df: DataFrame with features
            features: List of feature column names
            sequence_length: Length of input sequences
            test_size: Proportion of data for testing

        Returns:
            Dict with prepared data splits
        """
        # Extract features and target
        feature_data = df[features].fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Target is next period's log return
        target_data = df['log_return'].shift(-1).fillna(0)  # Predict next period

        # Create sequences
        X, y = self.create_sequences(feature_data.values, target_data.values, sequence_length)

        if len(X) < 50:
            raise ValueError(f"Not enough data after sequence creation: {len(X)} samples")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Don't shuffle time series
        )

        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, shuffle=False
        )

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
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
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_scaled,
            'y_val': y_val_scaled,
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
        y_train_tensor = torch.tensor(data_dict['y_train'], dtype=torch.float32)

        X_val_tensor = torch.tensor(data_dict['X_val'], dtype=torch.float32)
        y_val_tensor = torch.tensor(data_dict['y_val'], dtype=torch.float32)

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return {
            'train_loader': train_loader,
            'val_loader': val_loader
        }

    def train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                   learning_rate: float, epochs: int, early_stopping_patience: int = 10) -> Dict:
        """
        Train LSTM model.

        Args:
            model: LSTM model
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping

        Returns:
            Dict with training history and metrics
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1
        }

    def calculate_directional_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculate directional accuracy (percentage of correct direction predictions).

        Args:
            predictions: Model predictions
            targets: True target values

        Returns:
            Directional accuracy as a percentage
        """
        pred_direction = np.sign(predictions)
        true_direction = np.sign(targets)

        correct_directions = (pred_direction == true_direction).sum()
        accuracy = correct_directions / len(predictions) * 100

        return accuracy

    def objective(self, trial: optuna.trial.Trial, df: pd.DataFrame, optimized_indicators: Optional[Dict]) -> float:
        """
        Optuna objective function for LSTM optimization.

        Args:
            trial: Optuna trial object
            df: DataFrame with features and target
            optimized_indicators: Optimized indicator parameters

        Returns:
            Objective value (validation loss)
        """
        try:
            # Apply optimized indicators if available
            if optimized_indicators:
                df = self.apply_optimized_indicators(df, optimized_indicators)

            # Suggest hyperparameters
            sequence_length = trial.suggest_int('sequence_length', 10, 120)
            hidden_size = trial.suggest_int('hidden_size', 32, 256)
            num_layers = trial.suggest_int('num_layers', 1, 4)
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            epochs = trial.suggest_int('epochs', 20, 100)

            # Prepare features
            features = self.prepare_lstm_features(df)

            # Prepare data
            data_dict = self.prepare_data(df, features, sequence_length)
            scaled_data, scalers = self.scale_data(data_dict)

            # Create data loaders
            loaders = self.create_data_loaders(scaled_data, batch_size)

            # Create model
            model = LSTMModel(
                input_size=scaled_data['n_features'],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            ).to(DEVICE)

            # Train model
            training_results = self.train_model(
                model, loaders['train_loader'], loaders['val_loader'],
                learning_rate, epochs, early_stopping_patience=7
            )

            # Calculate additional metrics for validation
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(scaled_data['X_val'], dtype=torch.float32).to(DEVICE)
                val_predictions = model(X_val_tensor).squeeze().cpu().numpy()

                # Inverse transform predictions and targets
                val_predictions_orig = scalers['target_scaler'].inverse_transform(val_predictions.reshape(-1, 1)).flatten()
                val_targets_orig = scalers['target_scaler'].inverse_transform(scaled_data['y_val'].reshape(-1, 1)).flatten()

                # Calculate directional accuracy
                directional_accuracy = self.calculate_directional_accuracy(val_predictions_orig, val_targets_orig)

            # Multi-objective: combine MSE and directional accuracy
            val_mse = training_results['best_val_loss']

            # Penalize poor directional accuracy
            if directional_accuracy < 45:  # Below random chance
                return val_mse * 10

            # Combined objective (minimize MSE, maximize directional accuracy)
            combined_objective = val_mse * (1 + (55 - directional_accuracy) / 100)

            # Report intermediate value for pruning
            trial.report(combined_objective, training_results['final_epoch'])

            # Pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return combined_objective

        except Exception as e:
            logger.warning(f"Error in LSTM objective function: {str(e)}")
            return 999.0  # Return large value for failed trials

    def optimize_lstm(self, symbol: str, timeframe: str) -> Dict:
        """
        Optimize LSTM hyperparameters for a specific symbol-timeframe combination.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dict with optimization results
        """
        logger.info(f"Optimizing LSTM for {symbol} {timeframe}")

        try:
            # Load optimized indicator parameters
            optimized_indicators = self.load_optimized_indicators(symbol, timeframe)

            # Find labeled data file
            pattern = f"labeled_{symbol}_{timeframe}_*.csv"
            csv_files = list(self.labeled_data_dir.glob(pattern))

            if not csv_files:
                raise FileNotFoundError(f"No labeled data found for {symbol} {timeframe}")

            # Use the most recent file
            csv_file = sorted(csv_files)[-1]
            logger.info(f"Using data file: {csv_file}")

            # Load data
            df = pd.read_csv(csv_file)

            # Use subset for optimization (speed up the process)
            if len(df) > 5000:
                df = df.iloc[-5000:].copy().reset_index(drop=True)

            logger.info(f"Using {len(df)} samples for LSTM optimization")

            # Create Optuna study
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
            )

            # Optimize
            study.optimize(
                lambda trial: self.objective(trial, df, optimized_indicators),
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True
            )

            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value

            logger.info(f"LSTM optimization completed for {symbol} {timeframe}")
            logger.info(f"Best objective value: {best_value:.6f}")
            logger.info(f"Best parameters: {best_params}")

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'optimization_timestamp': timestamp,
                'best_params': best_params,
                'best_objective_value': best_value,
                'n_trials': study.n_trials,
                'optimization_samples': len(df),
                'optimized_indicators_used': optimized_indicators is not None
            }

            # Save to JSON file
            output_filename = f"lstm_params_{symbol}_{timeframe}_{timestamp}.json"
            output_path = self.results_dir / output_filename

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"✓ Saved LSTM optimization results to {output_path}")

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': True,
                'results_file': str(output_path),
                'best_params': best_params,
                'best_objective_value': best_value
            }

        except Exception as e:
            error_msg = f"Failed to optimize LSTM for {symbol} {timeframe}: {str(e)}"
            logger.error(error_msg)
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': False,
                'error': error_msg
            }

    def optimize_all(self) -> Dict:
        """
        Optimize LSTM hyperparameters for all symbol-timeframe combinations.

        Returns:
            Dict with summary of optimization results
        """
        symbols = self.config['symbols']
        timeframes = self.config['timeframes']

        logger.info(f"Optimizing LSTM for {len(symbols)} symbols x {len(timeframes)} timeframes")

        results = {
            'total': len(symbols) * len(timeframes),
            'successful': [],
            'failed': []
        }

        for symbol in symbols:
            for timeframe in timeframes:
                result = self.optimize_lstm(symbol, timeframe)

                if result['success']:
                    results['successful'].append(result)
                else:
                    results['failed'].append(result)

        # Log summary
        logger.info(f"\n{'='*50}")
        logger.info(f"LSTM Optimization Summary:")
        logger.info(f"  Total: {results['total']}")
        logger.info(f"  Successful: {len(results['successful'])}")
        logger.info(f"  Failed: {len(results['failed'])}")
        logger.info(f"{'='*50}")

        if results['failed']:
            logger.warning("Failed optimizations:")
            for failure in results['failed']:
                logger.warning(f"  {failure['symbol']} {failure['timeframe']}: {failure['error']}")

        return results

def main():
    """Main function to run LSTM optimization."""
    try:
        optimizer = LSTMOptimizer()
        results = optimizer.optimize_all()

        logger.info("LSTM optimization completed!")

    except Exception as e:
        logger.error(f"LSTM optimization failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
