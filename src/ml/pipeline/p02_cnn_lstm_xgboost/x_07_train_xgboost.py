"""
Stage 7: XGBoost Training
=========================

This stage trains the final XGBoost model using the best hyperparameters
found in the optimization stage and the combined features.

Features:
- Loads optimized hyperparameters from Optuna study
- Trains XGBoost model with best parameters
- Model checkpointing and saving with training restart capability
- Comprehensive metrics and visualization
- Time series cross-validation support
"""

import sys
import yaml
import numpy as np
import xgboost as xgb
import json
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class XGBoostTrainer:
    """XGBoost model trainer."""

    def __init__(self, config_path: str = "config/pipeline/x02.yaml"):
        """Initialize the XGBoost trainer."""
        self.config_path = config_path
        self.config = self._load_config()

        # Create directories
        self.models_dir = Path("src/ml/pipeline/p02_cnn_lstm_xgboost/models")
        self.xgboost_dir = self.models_dir / "xgboost"
        self.models_path = self.xgboost_dir / "models"
        self.configs_dir = self.xgboost_dir / "configs"
        self.results_dir = self.models_dir / "results"
        self.predictions_dir = self.results_dir / "predictions"
        self.visualizations_dir = self.results_dir / "visualizations"
        self.checkpoints_dir = self.xgboost_dir / "checkpoints"

        for dir_path in [self.models_path, self.predictions_dir, self.visualizations_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise Exception(f"Failed to load config from {self.config_path}: {e}")

    def _load_best_params(self) -> dict:
        """Load the best hyperparameters from optimization stage."""
        best_params_path = self.configs_dir / "best_xgboost_params.json"

        if not best_params_path.exists():
            _logger.warning("Best parameters not found, using default parameters")
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_lambda': 1,
                'reg_alpha': 0
            }

        try:
            with open(best_params_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            _logger.exception("Failed to load best parameters:")
            raise

    def _get_checkpoint_path(self) -> Path:
        """Get checkpoint file path."""
        return self.checkpoints_dir / "xgboost_checkpoint.pkl"

    def save_checkpoint(self, model: xgb.XGBRegressor, iteration: int,
                       train_metrics: dict, val_metrics: dict):
        """Save model state so training can resume later."""
        ckpt_path = self._get_checkpoint_path()
        checkpoint = {
            'model_state': model.get_booster().save_raw()[4],  # Save raw booster state
            'iteration': iteration,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_params': self._load_best_params()
        }

        with open(ckpt_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        _logger.info("Checkpoint saved at %s (iteration %d)", ckpt_path, iteration)

    def load_checkpoint(self, model: xgb.XGBRegressor) -> Tuple[int, dict, dict]:
        """Load model state to resume training if checkpoint exists."""
        ckpt_path = self._get_checkpoint_path()
        if ckpt_path.exists():
            with open(ckpt_path, 'rb') as f:
                checkpoint = pickle.load(f)

            # Load model state
            model.get_booster().load_model(ckpt_path)
            start_iteration = checkpoint['iteration'] + 1
            train_metrics = checkpoint['train_metrics']
            val_metrics = checkpoint['val_metrics']

            _logger.info("Resuming from checkpoint %s (iteration %d)", ckpt_path, start_iteration)
            return start_iteration, train_metrics, val_metrics
        else:
            return 0, {}, {}

    def _load_combined_features(self) -> tuple:
        """Load the combined features from the feature extraction stage."""
        data_dir = Path("data/labeled")

        # Load train data
        train_data = np.load(data_dir / "combined_features_train.npz")
        X_train = train_data['features']
        y_train = train_data['targets']

        # Load validation data
        val_data = np.load(data_dir / "combined_features_val.npz")
        X_val = val_data['features']
        y_val = val_data['targets']

        # Load test data
        test_data = np.load(data_dir / "combined_features_test.npz")
        X_test = test_data['features']
        y_test = test_data['targets']

        _logger.info("Loaded combined features:")
        _logger.info("  Train: %s, %s", X_train.shape, y_train.shape)
        _logger.info("  Validation: %s, %s", X_val.shape, y_val.shape)
        _logger.info("  Test: %s, %s", X_test.shape, y_test.shape)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # Calculate directional accuracy
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        directional_accuracy = np.mean((y_true_diff > 0) == (y_pred_diff > 0))

        # Calculate R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'r_squared': r_squared
        }

    def _create_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                     title: str, save_path: Path):
        """Create and save evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)

        # Actual vs Predicted scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)

        # Residuals plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)

        # Time series plot (first 100 points)
        n_points = min(100, len(y_true))
        axes[1, 0].plot(range(n_points), y_true[:n_points], label='Actual', alpha=0.7)
        axes[1, 0].plot(range(n_points), y_pred[:n_points], label='Predicted', alpha=0.7)
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Values')
        axes[1, 0].set_title('Time Series Comparison (First 100 points)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Residuals histogram
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def train(self) -> dict:
        """Train the XGBoost model."""
        _logger.info("Starting XGBoost training...")

        # Load best parameters
        best_params = self._load_best_params()
        _logger.info("Using parameters: %s", best_params)

        # Load data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self._load_combined_features()

        # Combine train and validation for final training
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])

        _logger.info("Final training data shape: %s", X_train_full.shape)

        # Create model
        model = xgb.XGBRegressor(
            **best_params,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=50
        )

        # Try to load checkpoint
        start_iteration, train_metrics, val_metrics = self.load_checkpoint(model)

        if start_iteration > 0:
            _logger.info("Resuming training from iteration %s", start_iteration)
        else:
            _logger.info("Starting fresh training")

        # Train with early stopping on validation set
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            xgb_model=None if start_iteration == 0 else model
        )

        # Save checkpoint after training
        final_iteration = model.n_estimators if hasattr(model, 'n_estimators') else 100
        self.save_checkpoint(model, final_iteration, train_metrics, val_metrics)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        val_metrics = self._calculate_metrics(y_val, y_val_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)

        # Save model
        model_path = self.models_path / "xgboost_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save predictions
        predictions = {
            'train': {'actual': y_train, 'predicted': y_train_pred},
            'validation': {'actual': y_val, 'predicted': y_val_pred},
            'test': {'actual': y_test, 'predicted': y_test_pred}
        }

        predictions_path = self.predictions_dir / "xgboost_predictions.npz"
        np.savez(predictions_path,
                train_actual=y_train, train_predicted=y_train_pred,
                val_actual=y_val, val_predicted=y_val_pred,
                test_actual=y_test, test_predicted=y_test_pred)

        # Create plots
        self._create_plots(y_train, y_train_pred,
                          "XGBoost Training Set Performance",
                          self.visualizations_dir / "xgboost_train_plots.png")

        self._create_plots(y_val, y_val_pred,
                          "XGBoost Validation Set Performance",
                          self.visualizations_dir / "xgboost_val_plots.png")

        self._create_plots(y_test, y_test_pred,
                          "XGBoost Test Set Performance",
                          self.visualizations_dir / "xgboost_test_plots.png")

        # Save metrics
        metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics,
            'model_path': str(model_path),
            'predictions_path': str(predictions_path)
        }

        metrics_path = self.results_dir / "xgboost_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Log results
        _logger.info("XGBoost training completed")
        _logger.info("Train MSE: %.6f", train_metrics['mse'])
        _logger.info("Validation MSE: %.6f", val_metrics['mse'])
        _logger.info("Test MSE: %s.6f", test_metrics['mse'])
        _logger.info("Test Directional Accuracy: %.4f", test_metrics['directional_accuracy'])
        _logger.info("Test R-squared: %.4f", test_metrics['r_squared'])

        return metrics

def main():
    """Main function to run XGBoost training."""
    import argparse

    parser = argparse.ArgumentParser(description="XGBoost Training")
    parser.add_argument("--config", default="config/pipeline/x02.yaml",
                       help="Path to configuration file")

    args = parser.parse_args()

    try:
        trainer = XGBoostTrainer(args.config)
        metrics = trainer.train()

        print("\n" + "="*50)
        print("XGBOOST TRAINING RESULTS")
        print("="*50)
        print(f"Train MSE: {metrics['train']['mse']:.6f}")
        print(f"Validation MSE: {metrics['validation']['mse']:.6f}")
        print(f"Test MSE: {metrics['test']['mse']:.6f}")
        print(f"Test MAE: {metrics['test']['mae']:.6f}")
        print(f"Test RMSE: {metrics['test']['rmse']:.6f}")
        print(f"Test Directional Accuracy: {metrics['test']['directional_accuracy']:.4f}")
        print(f"Test R-squared: {metrics['test']['r_squared']:.4f}")
        print("="*50)

    except Exception as e:
        _logger.exception("Error during XGBoost training:")
        sys.exit(1)

if __name__ == "__main__":
    main()
