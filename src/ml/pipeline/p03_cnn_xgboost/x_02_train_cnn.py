"""
CNN Training Stage for CNN + XGBoost Pipeline.

This module implements the CNN training stage that extracts features from OHLCV time series data.
The CNN is designed as a 1D convolutional network optimized for financial time series analysis.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.utils.logging import setup_logger
from src.utils.config import load_config

_logger = setup_logger(__name__)


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for time series feature extraction.

    This CNN is specifically designed for OHLCV financial data with configurable
    architecture parameters for optimal feature extraction.
    """

    def __init__(self,
                 input_channels: int = 5,
                 sequence_length: int = 120,
                 embedding_dim: int = 64,
                 num_filters: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [3, 5, 7],
                 dropout_rate: float = 0.3) -> None:
        """
        Initialize the 1D CNN architecture.

        Args:
            input_channels: Number of input features (OHLCV = 5)
            sequence_length: Length of time series sequence
            embedding_dim: Dimension of output embeddings
            num_filters: List of filter counts for each convolutional layer
            kernel_sizes: List of kernel sizes for each convolutional layer
            dropout_rate: Dropout rate for regularization
        """
        super(CNN1D, self).__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        # Build convolutional layers
        layers = []
        in_channels = input_channels

        for i, (filters, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            layers.extend([
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_channels = filters

        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final embedding layer
        self.embedding_layer = nn.Linear(num_filters[-1], embedding_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.

        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)

        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim)
        """
        # Apply convolutional layers
        x = self.conv_layers(x)

        # Global average pooling
        x = self.global_pool(x)

        # Flatten and apply embedding layer
        x = x.view(x.size(0), -1)
        x = self.embedding_layer(x)

        return x


class CNNTrainer:
    """
    CNN Trainer for the CNN + XGBoost pipeline.

    Handles data preparation, model training, hyperparameter optimization,
    and model saving for the CNN feature extraction stage.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the CNN trainer.

        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.cnn_config = config.get("cnn", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _logger.info("Initializing CNN trainer on device: %s", self.device)

        # Create output directories
        self.models_dir = Path("src/ml/pipeline/p03_cnn_xgboost/models/cnn")
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = []

    def run(self) -> Dict[str, Any]:
        """
        Execute the CNN training stage.

        Returns:
            Dictionary containing training results and metadata
        """
        _logger.info("Starting CNN training stage")

        try:
            # Load processed data
            data_files = self._discover_processed_data()
            if not data_files:
                raise ValueError("No processed data files found")

            _logger.info("Found %d processed data files", len(data_files))

            # Prepare training data
            X_train, y_train = self._prepare_training_data(data_files)

            # Optimize hyperparameters if enabled
            if self.cnn_config.get("optimize_hyperparameters", True):
                best_params = self._optimize_hyperparameters(X_train, y_train)
                self.cnn_config.update(best_params)
                _logger.info("Hyperparameter optimization completed")

            # Train final model
            training_results = self._train_model(X_train, y_train)

            # Save model and artifacts
            self._save_model_and_artifacts(training_results)

            # Generate training summary
            summary = self._generate_training_summary(training_results)

            _logger.info("CNN training stage completed successfully")
            return summary

        except Exception as e:
            _logger.exception("Error in CNN training stage: %s", e)
            raise

    def _discover_processed_data(self) -> List[Path]:
        """
        Discover processed data files from the data loader stage.

        Returns:
            List of paths to processed data files
        """
        data_dir = Path("data/processed")
        if not data_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found: {data_dir}")

        # Look for Parquet files
        data_files = list(data_dir.glob("*.parquet"))

        if not data_files:
            # Fallback to CSV files
            data_files = list(data_dir.glob("*.csv"))

        return data_files

    def _get_checkpoint_path(self) -> Path:
        """Get checkpoint file path."""
        return self.checkpoints_dir / "cnn_checkpoint.pth"

    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                       epoch: int, history: Dict[str, List[float]], best_val_loss: float):
        """Save model/optimizer state so training can resume later."""
        ckpt_path = self._get_checkpoint_path()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'best_val_loss': best_val_loss,
            'scaler_state': self.scaler
        }, ckpt_path)
        _logger.info("Checkpoint saved at %s (epoch %d)", ckpt_path, epoch+1)

    def load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer) -> Tuple[int, Dict[str, List[float]], float]:
        """Load model/optimizer state to resume training if checkpoint exists."""
        ckpt_path = self._get_checkpoint_path()
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint['history']
            best_val_loss = checkpoint['best_val_loss']
            self.scaler = checkpoint['scaler_state']
            _logger.info("Resuming from checkpoint %s (epoch %d)", ckpt_path, start_epoch)
            return start_epoch, history, best_val_loss
        else:
            return 0, {'train_loss': [], 'val_loss': [], 'learning_rate': []}, float('inf')

    def _prepare_training_data(self, data_files: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from processed files.

        Args:
            data_files: List of paths to processed data files

        Returns:
            Tuple of (X_train, y_train) arrays
        """
        _logger.info("Preparing training data from %d files", len(data_files))

        all_sequences = []
        all_targets = []

        sequence_length = self.cnn_config.get("sequence_length", 120)

        for file_path in data_files:
            try:
                # Load data
                if file_path.suffix == ".parquet":
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)

                # Extract OHLCV features
                ohlcv_cols = ["open", "high", "low", "close", "volume"]
                if not all(col in df.columns for col in ohlcv_cols):
                    _logger.warning("Missing OHLCV columns in %s, skipping", file_path)
                    continue

                ohlcv_data = df[ohlcv_cols].values

                # Create sequences
                sequences, targets = self._create_sequences(ohlcv_data, sequence_length)

                all_sequences.extend(sequences)
                all_targets.extend(targets)

                _logger.debug("Processed %s: %d sequences", file_path.name, len(sequences))

            except Exception as e:
                _logger.warning("Error processing %s: %s", file_path, e)
                continue

        if not all_sequences:
            raise ValueError("No valid sequences found in processed data")

        # Convert to numpy arrays
        X = np.array(all_sequences)
        y = np.array(all_targets)

        # Normalize features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X = X_normalized.reshape(X.shape)

        _logger.info("Prepared training data: X shape %s, y shape %s", X.shape, y.shape)

        return X, y

    def _create_sequences(self,
                         data: np.ndarray,
                         sequence_length: int) -> Tuple[List[np.ndarray], List[int]]:
        """
        Create sequences and targets from time series data.

        Args:
            data: OHLCV data array
            sequence_length: Length of each sequence

        Returns:
            Tuple of (sequences, targets) lists
        """
        sequences = []
        targets = []

        for i in range(len(data) - sequence_length):
            # Extract sequence
            sequence = data[i:i + sequence_length]

            # Create target (next period return direction)
            current_close = data[i + sequence_length - 1, 3]  # Close price
            next_close = data[i + sequence_length, 3] if i + sequence_length < len(data) else current_close

            # Simple binary target: 1 if price goes up, 0 if down
            target = 1 if next_close > current_close else 0

            sequences.append(sequence)
            targets.append(target)

        return sequences, targets

    def _optimize_hyperparameters(self,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray) -> Dict[str, Any]:
        """
        Optimize CNN hyperparameters using Optuna.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Dictionary of best hyperparameters
        """
        _logger.info("Starting hyperparameter optimization")

        def objective(trial):
            # Define hyperparameter search space
            embedding_dim = trial.suggest_int("embedding_dim", 32, 128)
            num_filters = [
                trial.suggest_int("filters_1", 16, 64),
                trial.suggest_int("filters_2", 32, 128),
                trial.suggest_int("filters_3", 64, 256)
            ]
            kernel_sizes = [
                trial.suggest_int("kernel_1", 3, 7),
                trial.suggest_int("kernel_2", 3, 9),
                trial.suggest_int("kernel_3", 3, 11)
            ]
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

            # Create model with trial parameters
            model = CNN1D(
                input_channels=5,
                sequence_length=X_train.shape[1],
                embedding_dim=embedding_dim,
                num_filters=num_filters,
                kernel_sizes=kernel_sizes,
                dropout_rate=dropout_rate
            ).to(self.device)

            # Train model
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.BCEWithLogitsLoss()

            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            val_scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                # Convert to tensors
                X_fold_train = torch.FloatTensor(X_fold_train).to(self.device)
                y_fold_train = torch.FloatTensor(y_fold_train).to(self.device)
                X_fold_val = torch.FloatTensor(X_fold_val).to(self.device)
                y_fold_val = torch.FloatTensor(y_fold_val).to(self.device)

                # Train for a few epochs
                model.train()
                for epoch in range(5):  # Quick training for optimization
                    optimizer.zero_grad()
                    outputs = model(X_fold_train.transpose(1, 2))
                    loss = criterion(outputs.squeeze(), y_fold_train)
                    loss.backward()
                    optimizer.step()

                # Evaluate
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_fold_val.transpose(1, 2))
                    val_loss = criterion(val_outputs.squeeze(), y_fold_val)
                    val_scores.append(val_loss.item())

            return np.mean(val_scores)

        # Run optimization
        study = optuna.create_study(direction="minimize")
        n_trials = self.cnn_config.get("optimization_trials", 50)
        study.optimize(objective, n_trials=n_trials)

        _logger.info("Best trial: %s", study.best_trial.value)
        _logger.info("Best parameters: %s", study.best_trial.params)

        return study.best_trial.params

    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train the final CNN model.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Dictionary containing training results
        """
        _logger.info("Training final CNN model")

        # Create model
        self.model = CNN1D(
            input_channels=5,
            sequence_length=X_train.shape[1],
            embedding_dim=self.cnn_config.get("embedding_dim", 64),
            num_filters=self.cnn_config.get("num_filters", [32, 64, 128]),
            kernel_sizes=self.cnn_config.get("kernel_sizes", [3, 5, 7]),
            dropout_rate=self.cnn_config.get("dropout_rate", 0.3)
        ).to(self.device)

        # Prepare data
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = self.cnn_config.get("batch_size", 32)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training setup
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cnn_config.get("learning_rate", 0.001)
        )
        criterion = nn.BCEWithLogitsLoss()
        num_epochs = self.cnn_config.get("num_epochs", 50)

        # Training loop
        self.model.train()
        training_history = []
        best_loss = float('inf')

        # Try to load checkpoint
        start_epoch, history, best_loss = self.load_checkpoint(self.model, optimizer)
        if start_epoch > 0:
            training_history = history['train_loss']
            _logger.info("Resuming training from epoch %d", start_epoch)

        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X.transpose(1, 2))
                loss = criterion(outputs.squeeze(), batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            training_history.append(avg_loss)

            # Save checkpoint after each epoch
            self.save_checkpoint(self.model, optimizer, epoch,
                               {'train_loss': training_history}, best_loss)

            if (epoch + 1) % 10 == 0:
                _logger.info("Epoch %d/%d, Loss: %.4f", epoch + 1, num_epochs, avg_loss)

        self.training_history = training_history

        return {
            "final_loss": training_history[-1],
            "training_history": training_history,
            "model_parameters": sum(p.numel() for p in self.model.parameters())
        }

    def _save_model_and_artifacts(self, training_results: Dict[str, Any]) -> None:
        """
        Save the trained model and training artifacts.

        Args:
            training_results: Results from model training
        """
        _logger.info("Saving model and artifacts")

        # Save model
        model_path = self.models_dir / "cnn_model.pth"
        torch.save(self.model.state_dict(), model_path)

        # Save scaler
        scaler_path = self.models_dir / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        # Save training results
        results_path = self.models_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(training_results, f, indent=2)

        # Save model configuration
        config_path = self.models_dir / "model_config.json"
        model_config = {
            "input_channels": 5,
            "sequence_length": self.cnn_config.get("sequence_length", 120),
            "embedding_dim": self.cnn_config.get("embedding_dim", 64),
            "num_filters": self.cnn_config.get("num_filters", [32, 64, 128]),
            "kernel_sizes": self.cnn_config.get("kernel_sizes", [3, 5, 7]),
            "dropout_rate": self.cnn_config.get("dropout_rate", 0.3)
        }
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)

        _logger.info("Model and artifacts saved to %s", self.models_dir)

    def _generate_training_summary(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of the training process.

        Args:
            training_results: Results from model training

        Returns:
            Dictionary containing training summary
        """
        return {
            "stage": "cnn_training",
            "status": "completed",
            "model_path": str(self.models_dir / "cnn_model.pth"),
            "scaler_path": str(self.models_dir / "scaler.pkl"),
            "final_loss": training_results["final_loss"],
            "model_parameters": training_results["model_parameters"],
            "device_used": str(self.device),
            "training_epochs": len(training_results["training_history"]),
            "embedding_dim": self.cnn_config.get("embedding_dim", 64)
        }


def train_cnn(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to train the CNN model.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        Dictionary containing training results
    """
    trainer = CNNTrainer(config)
    return trainer.run()


if __name__ == "__main__":
    # Load configuration
    config_path = Path("config/pipeline/p03.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(str(config_path))

    # Run CNN training
    results = train_cnn(config)
    print("CNN Training Results:", results)
