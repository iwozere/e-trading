"""
CNN-LSTM Training Module for CNN-LSTM-XGBoost Pipeline

This module trains the CNN-LSTM model using the optimized hyperparameters from the optimization stage.
It loads the best hyperparameters and trains the final model for feature extraction.

Features:
- Loads optimized hyperparameters from Optuna study
- Trains CNN-LSTM model with best parameters
- Model checkpointing and saving with training restart capability
- Training visualization and monitoring
- GPU support and mixed precision training
- Comprehensive logging and error handling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import json
import pickle
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

# Import the HybridCNNLSTM model from the optimization module
from x_03_optuna_cnn_lstm import HybridCNNLSTM

class CNNLSTMTrainer:
    def __init__(self, config_path: str = "config/pipeline/x02.yaml"):
        """
        Initialize CNN-LSTM trainer with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Directory setup
        self.labeled_data_dir = Path(self.config['paths']['data_labeled'])
        self.models_dir = Path(self.config['paths']['models_cnn_lstm'])
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.studies_dir = self.models_dir / "studies"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.cnn_lstm_config = self.config.get('cnn_lstm', {})
        self.hardware_config = self.config.get('hardware', {})

        # Device setup
        self.device = self._setup_device()

        # Load data
        self.data = self._load_data()

        # Load best hyperparameters
        self.best_params = self._load_best_params()

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

        # Load train data
        train_data = np.load(self.labeled_data_dir / "combined_features_train.npz")
        X_train = train_data['features']
        y_train = train_data['targets']

        # Load validation data
        val_data = np.load(self.labeled_data_dir / "combined_features_val.npz")
        X_val = val_data['features']
        y_val = val_data['targets']

        # Load test data
        test_data = np.load(self.labeled_data_dir / "combined_features_test.npz")
        X_test = test_data['features']
        y_test = test_data['targets']

        # Load feature names
        feature_names_path = self.labeled_data_dir / "feature_names.json"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[2])]

        _logger.info("Loaded data:")
        _logger.info("  Train: %d, %d", X_train.shape, y_train.shape)
        _logger.info("  Validation: %d, %d", X_val.shape, y_val.shape)
        _logger.info("  Test: %d, %d", X_test.shape, y_test.shape)
        _logger.info("  Features: %d", len(feature_names))

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_names
        }

    def _load_best_params(self) -> dict:
        """Load the best hyperparameters from optimization stage."""
        best_params_path = self.studies_dir / "best_cnn_lstm_params.json"

        if not best_params_path.exists():
            _logger.warning("Best parameters not found, using default parameters")
            return {
                'conv_filters': 64,
                'lstm_units': 128,
                'dense_units': 64,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            }

        try:
            with open(best_params_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            _logger.exception("Failed to load best parameters:")
            raise

    def _get_checkpoint_path(self) -> Path:
        """Get checkpoint file path."""
        return self.checkpoints_dir / "cnn_lstm_checkpoint.pth"

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
            'best_params': self.best_params
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
            _logger.info("Resuming from checkpoint %s (epoch %d)", ckpt_path, start_epoch)
            return start_epoch, history, best_val_loss
        else:
            return 0, {'train_loss': [], 'val_loss': [], 'learning_rate': []}, float('inf')

    def create_model(self) -> HybridCNNLSTM:
        """Create the CNN-LSTM model with best hyperparameters."""
        time_steps = self.cnn_lstm_config.get('time_steps', 20)
        features = self.data['X_train'].shape[2]

        model = HybridCNNLSTM(
            time_steps=time_steps,
            features=features,
            conv_filters=self.best_params['conv_filters'],
            lstm_units=self.best_params['lstm_units'],
            dense_units=self.best_params['dense_units'],
            attention_heads=self.best_params.get('attention_heads', 1),
            dropout=self.best_params['dropout'],
            kernel_size=self.cnn_lstm_config.get('kernel_size', 3)
        )

        _logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        return model

    def train_model(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        Train the CNN-LSTM model.

        Args:
            model: CNN-LSTM model

        Returns:
            Tuple of (trained_model, training_history)
        """
        _logger.info("Starting CNN-LSTM training...")

        model.to(self.device)
        model.train()

        # Training parameters
        epochs = self.cnn_lstm_config.get('epochs', 50)
        batch_size = self.best_params['batch_size']
        learning_rate = self.best_params['learning_rate']
        early_stopping_patience = self.cnn_lstm_config.get('early_stopping_patience', 10)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Convert to tensors
        X_train_tensor = torch.tensor(self.data['X_train'], dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(self.data['y_train'], dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(self.data['X_val'], dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(self.data['y_val'], dtype=torch.float32).to(self.device)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Load checkpoint if it exists
        start_epoch, history, best_val_loss = self.load_checkpoint(model, optimizer)

        for epoch in range(start_epoch, epochs):
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

            # Record history
            avg_train_loss = train_loss / (X_train_tensor.size(0) // batch_size)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            # Save checkpoint after each epoch
            self.save_checkpoint(model, optimizer, epoch, history, best_val_loss)

            if patience_counter >= early_stopping_patience:
                _logger.info("Early stopping at epoch %d", (epoch + 1))
                break

            # Logging
            if (epoch + 1) % 10 == 0:
                _logger.info('Epoch [%d/%d], Train Loss: %.6f, Val Loss: %.6f, LR: %.6f', (epoch+1), epochs, avg_train_loss, val_loss, optimizer.param_groups[0]["lr"])

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            _logger.info("Loaded best model with validation loss: %.6f", best_val_loss)

        return model, history

    def save_model(self, model: nn.Module, history: Dict[str, List[float]]):
        """Save the trained model and training history."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save model
        model_path = self.checkpoints_dir / f"cnn_lstm_model_{timestamp}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'time_steps': self.cnn_lstm_config.get('time_steps', 20),
                'features': self.data['X_train'].shape[2],
                'conv_filters': self.best_params['conv_filters'],
                'lstm_units': self.best_params['lstm_units'],
                'dense_units': self.best_params['dense_units'],
                'attention_heads': self.best_params.get('attention_heads', 1),
                'dropout': self.best_params['dropout'],
                'kernel_size': self.cnn_lstm_config.get('kernel_size', 3)
            },
            'best_params': self.best_params,
            'training_history': history,
            'feature_names': self.data['feature_names']
        }, model_path)

        # Save training history
        history_path = self.checkpoints_dir / f"cnn_lstm_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Save model info
        info_path = self.checkpoints_dir / f"cnn_lstm_info_{timestamp}.json"
        model_info = {
            'model_path': str(model_path),
            'history_path': str(history_path),
            'best_params': self.best_params,
            'model_config': {
                'time_steps': self.cnn_lstm_config.get('time_steps', 20),
                'features': self.data['X_train'].shape[2],
                'conv_filters': self.best_params['conv_filters'],
                'lstm_units': self.best_params['lstm_units'],
                'dense_units': self.best_params['dense_units'],
                'attention_heads': self.best_params.get('attention_heads', 1),
                'dropout': self.best_params['dropout'],
                'kernel_size': self.cnn_lstm_config.get('kernel_size', 3)
            },
            'data_info': {
                'train_samples': len(self.data['X_train']),
                'val_samples': len(self.data['X_val']),
                'test_samples': len(self.data['X_test']),
                'feature_count': len(self.data['feature_names'])
            },
            'training_info': {
                'best_val_loss': min(history['val_loss']),
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'epochs_trained': len(history['train_loss'])
            }
        }

        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        _logger.info("Model saved to %s", model_path)
        _logger.info("Training history saved to %s", history_path)
        _logger.info("Model info saved to %s", info_path)

        return model_path, history_path, info_path

    def create_training_plots(self, history: Dict[str, List[float]], save_path: Path):
        """Create training visualization plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Training and validation loss
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Learning rate
        ax2.plot(epochs, history['learning_rate'], 'g-', label='Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')

        # Loss difference
        loss_diff = [abs(t - v) for t, v in zip(history['train_loss'], history['val_loss'])]
        ax3.plot(epochs, loss_diff, 'm-', label='|Train Loss - Val Loss|')
        ax3.set_title('Training vs Validation Loss Difference')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Difference')
        ax3.legend()
        ax3.grid(True)

        # Loss ratio
        loss_ratio = [t / v if v > 0 else 0 for t, v in zip(history['train_loss'], history['val_loss'])]
        ax4.plot(epochs, loss_ratio, 'c-', label='Train Loss / Val Loss')
        ax4.set_title('Training vs Validation Loss Ratio')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Ratio')
        ax4.legend()
        ax4.grid(True)
        ax4.axhline(y=1, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        _logger.info("Training plots saved to %s", save_path)

    def evaluate_model(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate the trained model on test data."""
        _logger.info("Evaluating model on test data...")

        model.eval()
        criterion = nn.MSELoss()

        X_test_tensor = torch.tensor(self.data['X_test'], dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(self.data['y_test'], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            test_outputs = model(X_test_tensor).squeeze()
            test_loss = criterion(test_outputs, y_test_tensor).item()

            # Convert to numpy for additional metrics
            y_pred = test_outputs.cpu().numpy()
            y_true = y_test_tensor.cpu().numpy()

            # Calculate additional metrics
            mae = np.mean(np.abs(y_pred - y_true))
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

            # Directional accuracy
            pred_direction = np.diff(y_pred) > 0
            true_direction = np.diff(y_true) > 0
            directional_accuracy = np.mean(pred_direction == true_direction)

        metrics = {
            'test_mse': test_loss,
            'test_mae': mae,
            'test_rmse': rmse,
            'directional_accuracy': directional_accuracy
        }

        _logger.info("Test MSE: %.6f", test_loss)
        _logger.info("Test MAE: %.6f", mae)
        _logger.info("Test RMSE: %.6f", rmse)
        _logger.info("Directional Accuracy: %.4f", directional_accuracy)

        return metrics

    def run(self) -> Dict[str, Any]:
        """
        Run the complete CNN-LSTM training process.

        Returns:
            Dictionary with training results
        """
        _logger.info("Starting CNN-LSTM training process...")

        try:
            # Create model
            model = self.create_model()

            # Train model
            trained_model, history = self.train_model(model)

            # Save model
            model_path, history_path, info_path = self.save_model(trained_model, history)

            # Create training plots
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plots_path = self.checkpoints_dir / f"cnn_lstm_training_plots_{timestamp}.png"
            self.create_training_plots(history, plots_path)

            # Evaluate model
            metrics = self.evaluate_model(trained_model)

            # Save metrics
            metrics_path = self.checkpoints_dir / f"cnn_lstm_metrics_{timestamp}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            _logger.info("CNN-LSTM training completed successfully!")

            return {
                'success': True,
                'model_path': str(model_path),
                'history_path': str(history_path),
                'info_path': str(info_path),
                'plots_path': str(plots_path),
                'metrics_path': str(metrics_path),
                'metrics': metrics,
                'best_params': self.best_params
            }

        except Exception as e:
            _logger.exception("CNN-LSTM training failed:")
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """Main entry point for CNN-LSTM training."""
    import argparse

    parser = argparse.ArgumentParser(description='CNN-LSTM model training')
    parser.add_argument('--config', default='config/pipeline/x02.yaml', help='Configuration file path')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')

    args = parser.parse_args()

    try:
        trainer = CNNLSTMTrainer(args.config)

        # Override config if provided
        if args.epochs:
            trainer.cnn_lstm_config['epochs'] = args.epochs

        results = trainer.run()

        if results['success']:
            print("Training completed successfully!")
            print(f"Model saved to: {results['model_path']}")
            print(f"Test metrics: {results['metrics']}")
        else:
            print(f"Training failed: {results['error']}")
            sys.exit(1)

    except Exception as e:
        _logger.error("CNN-LSTM training failed: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
