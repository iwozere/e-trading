"""
CNN Training Stage for CNN + XGBoost Pipeline.

This module implements the CNN training stage that extracts features from OHLCV time series data.
The CNN is designed as a 1D convolutional network optimized for financial time series analysis.
Each data file gets its own trained model with individual artifacts.
"""
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import yaml

project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger

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
                 num_filters: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [3, 5, 7],
                 dropout_rate: float = 0.3) -> None:
        """
        Initialize the 1D CNN architecture.

        Args:
            input_channels: Number of input features (OHLCV = 5)
            sequence_length: Length of time series sequence
            num_filters: List of filter counts for each convolutional layer
            kernel_sizes: List of kernel sizes for each convolutional layer
            dropout_rate: Dropout rate for regularization
        """
        super(CNN1D, self).__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length

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

        # Final classification layer (single output for binary classification)
        self.classification_layer = nn.Linear(num_filters[-1], 1)

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
            Logits tensor of shape (batch_size, 1) for binary classification
        """
        # Apply convolutional layers
        x = self.conv_layers(x)

        # Global average pooling
        x = self.global_pool(x)

        # Flatten and apply classification layer
        x = x.view(x.size(0), -1)
        x = self.classification_layer(x)

        return x


class CNNTrainer:
    """
    CNN Trainer for the CNN + XGBoost pipeline.

    Handles data preparation, model training, hyperparameter optimization,
    and model saving for the CNN feature extraction stage.
    Each data file gets its own trained model.
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

        # Create output directories using configurable paths
        self.models_dir = Path(self.config["paths"]["models_cnn"])
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.reports_dir = self.models_dir / "reports"
        self.visualizations_dir = self.models_dir / "visualizations"

        for dir_path in [self.models_dir, self.checkpoints_dir, self.reports_dir, self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = []
        self.optimization_results = []

    def run(self) -> Dict[str, Any]:
        """
        Execute the CNN training stage.

        Returns:
            Dictionary containing training results and metadata
        """
        _logger.info("Starting CNN training stage")

        try:
            # Load raw data
            data_files = self._discover_raw_data()
            if not data_files:
                raise ValueError("No raw data files found")

            _logger.info("Found %d raw data files", len(data_files))

            # Train individual models for each data file
            all_results = []

            for data_file in data_files:
                try:
                    _logger.info("Training CNN model for %s", data_file.name)
                    result = self._train_model_for_file(data_file)
                    all_results.append(result)
                except Exception as e:
                    _logger.error("Failed to train model for %s: %s", data_file.name, e)
                    continue

            # Generate overall summary
            summary = self._generate_overall_summary(all_results)

            _logger.info("CNN training stage completed successfully")
            return summary

        except Exception as e:
            _logger.exception("Error in CNN training stage: %s", e)
            raise

    def _train_model_for_file(self, data_file: Path) -> Dict[str, Any]:
        """
        Train a CNN model for a specific data file.

        Args:
            data_file: Path to the data file

        Returns:
            Dictionary containing training results for this file
        """
        # Parse filename to extract metadata
        file_metadata = self._parse_filename(data_file.name)

        # Generate timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model identifier
        model_id = f"cnn_{file_metadata['provider']}_{file_metadata['symbol']}_{file_metadata['timeframe']}_{file_metadata['start_date']}_{file_metadata['end_date']}_{timestamp}"

        _logger.info("Training model: %s", model_id)

        # Prepare training data for this file
        X_train, y_train = self._prepare_training_data_for_file(data_file)

        if len(X_train) == 0:
            raise ValueError(f"No valid training data found in {data_file.name}")

        # Optimize hyperparameters if enabled
        best_params = None
        if self.cnn_config.get("optimize_hyperparameters", True):
            best_params = self._optimize_hyperparameters(X_train, y_train, model_id)
            self.cnn_config.update(best_params)
            _logger.info("Hyperparameter optimization completed for %s", model_id)

        # Train final model
        training_results = self._train_model(X_train, y_train, model_id)

        # Save model and artifacts
        self._save_model_and_artifacts(training_results, model_id, file_metadata, best_params)

        # Generate visualization
        self._create_visualization(training_results, model_id, file_metadata)

        return {
            "model_id": model_id,
            "data_file": data_file.name,
            "file_metadata": file_metadata,
            "training_results": training_results,
            "best_params": best_params,
            "model_path": str(self.models_dir / f"{model_id}.pth"),
            "config_path": str(self.models_dir / f"{model_id}_config.json"),
            "report_path": str(self.reports_dir / f"{model_id}_report.json"),
            "visualization_path": str(self.visualizations_dir / f"{model_id}.png")
        }

    def _parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse filename to extract metadata.

        Expected format: provider_symbol_timeframe_startdate_enddate.csv
        Example: yfinance_VT_1d_20210829_20250828.csv
        """
        # Remove .csv extension
        name = filename.replace('.csv', '')

        # Split by underscore
        parts = name.split('_')

        if len(parts) >= 5:
            return {
                "provider": parts[0],
                "symbol": parts[1],
                "timeframe": parts[2],
                "start_date": parts[3],
                "end_date": parts[4]
            }
        else:
            # Fallback for non-standard filenames
            return {
                "provider": "unknown",
                "symbol": "unknown",
                "timeframe": "unknown",
                "start_date": "unknown",
                "end_date": "unknown"
            }

    def _discover_raw_data(self) -> List[Path]:
        """
        Discover raw data files from the data loader stage.

        Returns:
            List of paths to raw data files
        """
        data_dir = Path(self.config["paths"]["data_raw"])
        if not data_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {data_dir}")

        # Look for CSV files (raw data is saved as CSV)
        data_files = list(data_dir.glob("*.csv"))

        if not data_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        return data_files

    def _prepare_training_data_for_file(self, data_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from a single file.

        Args:
            data_file: Path to the data file

        Returns:
            Tuple of (X_train, y_train) arrays
        """
        _logger.info("Preparing training data from %s", data_file.name)

        try:
            # Load data (raw data is saved as CSV)
            df = pd.read_csv(data_file)

            # Extract OHLCV features
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in ohlcv_cols):
                _logger.warning("Missing OHLCV columns in %s, skipping", data_file.name)
                return np.array([]), np.array([])

            ohlcv_data = df[ohlcv_cols].values

            # Create sequences
            sequence_length = self.cnn_config.get("sequence_length", 120)
            sequences, targets = self._create_sequences(ohlcv_data, sequence_length)

            if len(sequences) == 0:
                _logger.warning("No valid sequences found in %s", data_file.name)
                return np.array([]), np.array([])

            # Convert to numpy arrays
            X = np.array(sequences)
            y = np.array(targets)

            # Sample data if we have too much to avoid memory issues
            max_samples = self.cnn_config.get("max_samples", 10000)
            if len(X) > max_samples:
                # Randomly sample to reduce memory usage
                indices = np.random.choice(len(X), max_samples, replace=False)
                X = X[indices]
                y = y[indices]
                _logger.info("Sampled %d sequences from %s", max_samples, data_file.name)

            # Normalize features
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_normalized = self.scaler.fit_transform(X_reshaped)
            X = X_normalized.reshape(X.shape)

            _logger.info("Prepared %d sequences from %s", len(X), data_file.name)
            return X, y

        except Exception as e:
            _logger.error("Error processing %s: %s", data_file.name, e)
            return np.array([]), np.array([])

    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[List[np.ndarray], List[int]]:
        """
        Create sequences and targets from OHLCV data.

        Args:
            data: OHLCV data array
            sequence_length: Length of each sequence

        Returns:
            Tuple of (sequences, targets) lists
        """
        sequences = []
        targets = []

        for i in range(len(data) - sequence_length):
            # Create sequence
            sequence = data[i:i + sequence_length]
            sequences.append(sequence)

            # Create target (simple binary classification: price goes up or down)
            current_price = data[i + sequence_length - 1, 3]  # Close price
            next_price = data[i + sequence_length, 3]  # Next close price
            target = 1 if next_price > current_price else 0
            targets.append(target)

        return sequences, targets

    def _optimize_hyperparameters(self,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 model_id: str) -> Dict[str, Any]:
        """
        Optimize CNN hyperparameters using Optuna.

        Args:
            X_train: Training features
            y_train: Training targets
            model_id: Model identifier for logging

        Returns:
            Dictionary of best hyperparameters
        """
        _logger.info("Starting hyperparameter optimization for %s", model_id)

        def objective(trial):
            # Define hyperparameter search space
            num_filters = [
                trial.suggest_int("filters_1", 16, 32),
                trial.suggest_int("filters_2", 32, 64),
                trial.suggest_int("filters_3", 64, 128)
            ]
            kernel_sizes = [
                trial.suggest_int("kernel_1", 3, 5),
                trial.suggest_int("kernel_2", 3, 7),
                trial.suggest_int("kernel_3", 3, 9)
            ]
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

            # Create model with trial parameters
            model = CNN1D(
                input_channels=5,
                sequence_length=X_train.shape[1],
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
        n_trials = self.cnn_config.get("optimization_trials", 10)
        study.optimize(objective, n_trials=n_trials)

        _logger.info("Best trial for %s: %s", model_id, study.best_trial.value)
        _logger.info("Best parameters for %s: %s", model_id, study.best_trial.params)

        # Save optimization results
        self.optimization_results.append({
            "model_id": model_id,
            "best_value": study.best_trial.value,
            "best_params": study.best_trial.params,
            "n_trials": n_trials
        })

        return study.best_trial.params

    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray, model_id: str) -> Dict[str, Any]:
        """
        Train the final CNN model.

        Args:
            X_train: Training features
            y_train: Training targets
            model_id: Model identifier

        Returns:
            Dictionary containing training results
        """
        _logger.info("Training final CNN model for %s", model_id)

        # Create model with proper parameter handling
        num_filters = self.cnn_config.get("num_filters", [32, 64, 128])
        kernel_sizes = self.cnn_config.get("kernel_sizes", [3, 5, 7])
        dropout_rate = self.cnn_config.get("dropout_rate", 0.3)

        # Handle parameters if they're lists (take first values)
        if isinstance(dropout_rate, list):
            dropout_rate = dropout_rate[0]

        self.model = CNN1D(
            input_channels=5,
            sequence_length=X_train.shape[1],
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout_rate=dropout_rate
        ).to(self.device)

        # Training parameters
        learning_rate = self.cnn_config.get("learning_rate", 0.001)
        if isinstance(learning_rate, list):
            learning_rate = learning_rate[0]

        batch_size = self.cnn_config.get("batch_size", 32)
        if isinstance(batch_size, list):
            batch_size = batch_size[0]

        num_epochs = self.cnn_config.get("epochs", 50)
        if isinstance(num_epochs, list):
            num_epochs = num_epochs[0]

        # Create data loader
        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup training
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        training_history = []
        best_loss = float('inf')

        for epoch in range(num_epochs):
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

            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 10 == 0:
                _logger.info("Epoch %d/%d, Loss: %.4f", epoch + 1, num_epochs, avg_loss)

        self.training_history = training_history

        return {
            "final_loss": training_history[-1],
            "best_loss": best_loss,
            "training_history": training_history,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }

    def _save_model_and_artifacts(self, training_results: Dict[str, Any], model_id: str,
                                 file_metadata: Dict[str, str], best_params: Dict[str, Any]) -> None:
        """
        Save the trained model and training artifacts.

        Args:
            training_results: Results from model training
            model_id: Model identifier
            file_metadata: Metadata from filename
            best_params: Best hyperparameters from optimization
        """
        _logger.info("Saving model and artifacts for %s", model_id)

        # Save model
        model_path = self.models_dir / f"{model_id}.pth"
        torch.save(self.model.state_dict(), model_path)

        # Save scaler
        scaler_path = self.models_dir / f"{model_id}_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        # Save model configuration
        config_path = self.models_dir / f"{model_id}_config.json"
        model_config = {
            "input_channels": 5,
            "sequence_length": self.cnn_config.get("sequence_length", 120),
            "num_filters": self.cnn_config.get("num_filters", [32, 64, 128]),
            "kernel_sizes": self.cnn_config.get("kernel_sizes", [3, 5, 7]),
            "dropout_rate": self.cnn_config.get("dropout_rate", 0.3),
            "file_metadata": file_metadata,
            "best_params": best_params,
            "training_config": {
                "learning_rate": training_results.get("learning_rate"),
                "batch_size": training_results.get("batch_size"),
                "num_epochs": training_results.get("num_epochs")
            }
        }
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)

        # Save training report
        report_path = self.reports_dir / f"{model_id}_report.json"
        report = {
            "model_id": model_id,
            "file_metadata": file_metadata,
            "training_results": training_results,
            "best_params": best_params,
            "device_used": str(self.device),
            "training_timestamp": datetime.now().isoformat()
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        _logger.info("Model and artifacts saved for %s", model_id)

    def _create_visualization(self, training_results: Dict[str, Any], model_id: str,
                            file_metadata: Dict[str, str]) -> None:
        """
        Create training visualization.

        Args:
            training_results: Results from model training
            model_id: Model identifier
            file_metadata: Metadata from filename
        """
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Plot training loss
            training_history = training_results["training_history"]
            epochs = range(1, len(training_history) + 1)

            ax1.plot(epochs, training_history, 'b-', label='Training Loss')
            ax1.set_title(f'Training Loss - {file_metadata["symbol"]} ({file_metadata["timeframe"]})')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            # Plot loss distribution
            ax2.hist(training_history, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title(f'Loss Distribution - {file_metadata["symbol"]}')
            ax2.set_xlabel('Loss')
            ax2.set_ylabel('Frequency')
            ax2.grid(True)

            # Add model info text
            info_text = f"""
            Model: {model_id}
            Symbol: {file_metadata['symbol']}
            Timeframe: {file_metadata['timeframe']}
            Provider: {file_metadata['provider']}
            Final Loss: {training_results['final_loss']:.4f}
            Best Loss: {training_results['best_loss']:.4f}
            Parameters: {training_results['model_parameters']:,}
            """

            plt.figtext(0.02, 0.02, info_text, fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

            plt.tight_layout()

            # Save visualization
            viz_path = self.visualizations_dir / f"{model_id}.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()

            _logger.info("Visualization saved: %s", viz_path)

        except Exception as e:
            _logger.warning("Failed to create visualization for %s: %s", model_id, e)

    def _generate_overall_summary(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate overall summary of all training runs.

        Args:
            all_results: List of results from all training runs

        Returns:
            Dictionary containing overall summary
        """
        # Save optimization summary report
        if self.optimization_results:
            optimization_report_path = self.reports_dir / f"full_cnn_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            opt_df = pd.DataFrame(self.optimization_results)
            opt_df.to_csv(optimization_report_path, index=False)
            _logger.info("Optimization report saved: %s", optimization_report_path)

        # Calculate summary statistics
        successful_models = len(all_results)
        total_parameters = sum(r["training_results"]["model_parameters"] for r in all_results)
        avg_final_loss = np.mean([r["training_results"]["final_loss"] for r in all_results])

        return {
            "stage": "cnn_training",
            "status": "completed",
            "total_models_trained": successful_models,
            "total_parameters": total_parameters,
            "average_final_loss": avg_final_loss,
            "device_used": str(self.device),
            "models": [r["model_id"] for r in all_results],
            "optimization_report_path": str(optimization_report_path) if self.optimization_results else None
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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # Load configuration
    config_path = Path("config/pipeline/p03.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(str(config_path))

    # Run CNN training
    results = train_cnn(config)
    _logger.info("CNN Training Results: %s", results)
