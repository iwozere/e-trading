"""
CNN-LSTM Optimization Module for CNN-LSTM-XGBoost Pipeline

This module implements Optuna-based hyperparameter optimization for the CNN-LSTM model.
It optimizes the CNN-LSTM architecture parameters to minimize validation MSE.

Features:
- Optuna-based hyperparameter optimization
- Configurable search spaces via YAML
- Study persistence in SQLite
- GPU support for training
- Early stopping and timeout handling
- Optimization visualization and reporting
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import yaml
import pickle
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model with attention mechanism for time series prediction.

    Args:
        time_steps (int): Number of timesteps in each input sequence
        features (int): Number of input features per timestep
        conv_filters (int): Number of convolutional filters
        lstm_units (int): Number of units in the first LSTM layer
        dense_units (int): Number of units in the second LSTM layer and dense output
        attention_heads (int): Number of attention heads
        dropout (float): Dropout rate
        kernel_size (int): Kernel size for convolutional layer
    """
    def __init__(self, time_steps: int, features: int, conv_filters: int,
                 lstm_units: int, dense_units: int, attention_heads: int = 1,
                 dropout: float = 0.3, kernel_size: int = 3):
        super(HybridCNNLSTM, self).__init__()

        self.time_steps = time_steps
        self.features = features
        self.conv_filters = conv_filters
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.kernel_size = kernel_size

        # Convolutional layer
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=conv_filters,
                              kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size=conv_filters, hidden_size=lstm_units,
                           batch_first=True, dropout=dropout)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_units,
            num_heads=attention_heads,
            batch_first=True,
            dropout=dropout
        )

        # Second LSTM layer
        self.lstm2 = nn.LSTM(input_size=lstm_units, hidden_size=dense_units,
                           batch_first=True, dropout=dropout)

        # Output layer
        self.fc = nn.Linear(dense_units, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time_steps, features)

        Returns:
            torch.Tensor: Output tensor of shape (batch, 1)
        """
        batch_size = x.size(0)

        # Transpose for convolutional layer: (batch, features, time_steps)
        x = x.permute(0, 2, 1)

        # Convolutional layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Transpose back for LSTM: (batch, time_steps, conv_filters)
        x = x.permute(0, 2, 1)

        # First LSTM layer
        lstm_out, _ = self.lstm1(x)

        # Attention mechanism
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Second LSTM layer
        lstm_out2, _ = self.lstm2(attn_output)

        # Take the last output
        x = lstm_out2[:, -1, :]

        # Output layer
        out = self.fc(x)

        return out

class CNNLSTMOptimizer:
    def __init__(self, config_path: str = "config/pipeline/x02.yaml"):
        """
        Initialize CNN-LSTM optimizer with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Directory setup
        self.labeled_data_dir = Path(self.config['paths']['data_labeled'])
        self.models_dir = Path(self.config['paths']['models_cnn_lstm'])
        self.studies_dir = self.models_dir / "studies"
        self.studies_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.cnn_lstm_config = self.config.get('cnn_lstm', {})
        self.optuna_config = self.config.get('optuna', {})
        self.hardware_config = self.config.get('hardware', {})

        # Device setup
        self.device = self._setup_device()

        # Load data
        self.data = self._load_data()

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        _logger.info("Loaded configuration from %s", self.config_path)
        return config

    def _setup_device(self) -> torch.device:
        """Setup device for training."""
        device_setting = self.hardware_config.get('device', 'auto')

        if device_setting == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_setting == 'cuda':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')

        _logger.info("Using device: %s", device)
        return device

    def _load_data(self) -> Dict[str, np.ndarray]:
        """Load processed data from labeled data directory."""
        _logger.info("Loading processed data...")

        # Find processed data files
        npz_files = list(self.labeled_data_dir.glob("processed_*.npz"))

        if not npz_files:
            raise FileNotFoundError("No processed data files found")

        # Load the first file (you might want to modify this to handle multiple files)
        data_file = npz_files[0]
        _logger.info("Loading data from: %s", data_file.name)

        data = np.load(data_file)

        return {
            'X_train': data['X_train'],
            'X_val': data['X_val'],
            'X_test': data['X_test'],
            'y_train': data['y_train'],
            'y_val': data['y_val'],
            'y_test': data['y_test'],
            'feature_names': data['feature_names'].tolist() if 'feature_names' in data else []
        }

    def train_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, epochs: int,
                   batch_size: int, learning_rate: float) -> Tuple[nn.Module, List[float], List[float]]:
        """
        Train the CNN-LSTM model.

        Args:
            model: CNN-LSTM model
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Tuple of (trained_model, train_losses, val_losses)
        """
        model.to(self.device)
        model.train()

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = self.cnn_lstm_config.get('early_stopping_patience', 10)

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0

            # Shuffle training data
            indices = torch.randperm(X_train_tensor.size(0))

            for i in range(0, X_train_tensor.size(0), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_x = X_train_tensor[batch_indices]
                batch_y = y_train_tensor[batch_indices]

                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
                val_loss = criterion(val_outputs, y_val_tensor).item()

            train_losses.append(train_loss / (X_train_tensor.size(0) // batch_size))
            val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                _logger.info("Early stopping at epoch %d", (epoch + 1))
                break

            if (epoch + 1) % 10 == 0:
                _logger.info('Epoch [%d/%d], Train Loss: %.6f, Val Loss: %.6f', (epoch+1), epochs, train_losses[-1], val_losses[-1])

        return model, train_losses, val_losses

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation MSE
        """
        # Suggest hyperparameters
        conv_filters = trial.suggest_int('conv_filters',
                                       self.cnn_lstm_config['conv_filters_range'][0],
                                       self.cnn_lstm_config['conv_filters_range'][1])

        lstm_units = trial.suggest_int('lstm_units',
                                     self.cnn_lstm_config['lstm_units_range'][0],
                                     self.cnn_lstm_config['lstm_units_range'][1])

        dense_units = trial.suggest_int('dense_units',
                                      self.cnn_lstm_config['dense_units_range'][0],
                                      self.cnn_lstm_config['dense_units_range'][1])

        learning_rate = trial.suggest_float('learning_rate',
                                          self.cnn_lstm_config['learning_rate_range'][0],
                                          self.cnn_lstm_config['learning_rate_range'][1],
                                          log=True)

        batch_size = trial.suggest_categorical('batch_size',
                                             self.cnn_lstm_config['batch_size_options'])

        dropout = trial.suggest_float('dropout', 0.1, 0.5)

        attention_heads = trial.suggest_int('attention_heads', 1, 4)

        # Create model
        time_steps = self.cnn_lstm_config.get('time_steps', 20)
        features = self.data['X_train'].shape[2]

        model = HybridCNNLSTM(
            time_steps=time_steps,
            features=features,
            conv_filters=conv_filters,
            lstm_units=lstm_units,
            dense_units=dense_units,
            attention_heads=attention_heads,
            dropout=dropout
        )

        # Train model
        epochs = self.cnn_lstm_config.get('epochs', 50)

        try:
            trained_model, train_losses, val_losses = self.train_model(
                model=model,
                X_train=self.data['X_train'],
                y_train=self.data['y_train'],
                X_val=self.data['X_val'],
                y_val=self.data['y_val'],
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )

            # Return best validation loss
            best_val_loss = min(val_losses)

            # Report intermediate values
            trial.report(best_val_loss, epoch=len(val_losses))

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.TrialPruned()

            return best_val_loss

        except Exception as e:
            _logger.exception("Error in trial:")
            raise optuna.TrialPruned()

    def run_optimization(self) -> optuna.Study:
        """
        Run the hyperparameter optimization.

        Returns:
            Optuna study object
        """
        _logger.info("Starting CNN-LSTM hyperparameter optimization...")

        # Study configuration
        study_name = self.optuna_config['study_names']['cnn_lstm']
        storage = self.optuna_config['storage']
        n_trials = self.optuna_config['n_trials']
        timeout = self.optuna_config['timeout']

        # Create or load study
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        # Save study results
        self._save_study_results(study)

        _logger.info("CNN-LSTM optimization completed")
        _logger.info("Best trial: %s", study.best_trial.number)
        _logger.info("Best value: %s", study.best_trial.value)
        _logger.info("Best params: %s", study.best_trial.params)

        return study

    def _save_study_results(self, study: optuna.Study):
        """Save study results and visualizations."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save study object
        study_path = self.studies_dir / f"cnn_lstm_study_{timestamp}.pkl"
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)

        # Save best parameters
        best_params_path = self.studies_dir / f"cnn_lstm_best_params_{timestamp}.json"
        with open(best_params_path, 'w') as f:
            json.dump(study.best_trial.params, f, indent=2)

        # Create visualizations
        self._create_optimization_plots(study, timestamp)

        _logger.info("Study results saved to %s", self.studies_dir)

    def _create_optimization_plots(self, study: optuna.Study, timestamp: str):
        """Create optimization visualization plots."""
        try:
            # Optimization history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot optimization history
            optuna.visualization.matplotlib.plot_optimization_history(study, ax=ax1)
            ax1.set_title('Optimization History')

            # Plot parameter importance
            optuna.visualization.matplotlib.plot_param_importances(study, ax=ax2)
            ax2.set_title('Parameter Importance')

            plt.tight_layout()
            plot_path = self.studies_dir / f"cnn_lstm_optimization_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Parameter relationships
            fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            fig.update_layout(title='Parameter Relationships')
            parallel_path = self.studies_dir / f"cnn_lstm_parallel_{timestamp}.html"
            fig.write_html(str(parallel_path))

            _logger.info("Optimization plots saved to %s", self.studies_dir)

        except Exception as e:
            _logger.warning("Could not create optimization plots: %s", e)

def main():
    """Main entry point for CNN-LSTM optimization."""
    import argparse

    parser = argparse.ArgumentParser(description='CNN-LSTM hyperparameter optimization')
    parser.add_argument('--config', default='config/pipeline/x02.yaml', help='Configuration file path')
    parser.add_argument('--trials', type=int, help='Number of trials (overrides config)')
    parser.add_argument('--timeout', type=int, help='Timeout in seconds (overrides config)')

    args = parser.parse_args()

    try:
        optimizer = CNNLSTMOptimizer(args.config)

        # Override config if provided
        if args.trials:
            optimizer.optuna_config['n_trials'] = args.trials
        if args.timeout:
            optimizer.optuna_config['timeout'] = args.timeout

        study = optimizer.run_optimization()

        print("Optimization completed successfully!")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_trial.value}")
        print(f"Best params: {study.best_trial.params}")

    except Exception as e:
        _logger.exception("CNN-LSTM optimization failed:")
        sys.exit(1)

if __name__ == "__main__":
    main()
