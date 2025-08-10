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

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import logging
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
from typing import Dict, List, Optional, Tuple, Any

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

class LSTMModel(nn.Module):
    """LSTM model for time series prediction with regime awareness."""

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
            batch_first=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output layer with regime conditioning
        self.linear = nn.Linear(hidden_size + n_regimes, output_size)

    def forward(self, x, regime_onehot):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]

        # Apply dropout
        dropped = self.dropout(last_output)

        # Concatenate with regime information
        combined = torch.cat([dropped, regime_onehot], dim=1)

        # Final prediction
        output = self.linear(combined)

        return output

class LSTMValidator:
    def __init__(self, config_path: str = "config/pipeline/x01.yaml"):
        """
        Initialize LSTM validator with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.labeled_data_dir = Path(self.config['paths']['data_labeled'])
        self.models_dir = Path(self.config['paths']['models_lstm'])
        self.results_dir = Path(self.config['paths']['results'])
        self.reports_dir = Path(self.config['paths']['reports'])
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {self.config_path}")
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
            logger.warning(f"No LSTM model found for {symbol} {timeframe}")
            return None

        # Return the most recent model
        latest_model = sorted(model_files)[-1]
        logger.info(f"Found latest model: {latest_model}")
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

            logger.info(f"Loaded LSTM model from {model_path}")
            logger.info(f"Model features: {len(model_package['features'])}")

            return model_package

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise

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
                            test_start_idx: int, symbol: str, timeframe: str) -> List[plt.Figure]:
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
            List of matplotlib figures
        """
        figures = []

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

        # Figure 1: Predictions vs Actual
        fig1, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig1.suptitle(f'LSTM Predictions vs Actual - {symbol} {timeframe}', fontsize=16)

        # Plot 1: Time series comparison
        axes[0].plot(time_index, act, label='Actual', alpha=0.7, linewidth=1)
        axes[0].plot(time_index, pred, label='LSTM Prediction', alpha=0.8, linewidth=1)
        axes[0].plot(time_index, base, label='Naive Baseline', alpha=0.6, linewidth=1)
        axes[0].set_title('Log Return Predictions Over Time')
        axes[0].set_ylabel('Log Return')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Scatter plot
        axes[1].scatter(act, pred, alpha=0.6, label='LSTM', s=10)
        axes[1].scatter(act, base, alpha=0.4, label='Baseline', s=10)

        # Perfect prediction line
        min_val, max_val = min(np.min(act), np.min(pred)), max(np.max(act), np.max(pred))
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Prediction')

        axes[1].set_xlabel('Actual Log Return')
        axes[1].set_ylabel('Predicted Log Return')
        axes[1].set_title('Prediction Accuracy Scatter Plot')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig1)

        # Figure 2: Error Analysis
        fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig2.suptitle(f'Error Analysis - {symbol} {timeframe}', fontsize=16)

        lstm_errors = pred - act
        baseline_errors = base - act

        # Error distribution
        axes[0, 0].hist(lstm_errors, bins=50, alpha=0.7, label='LSTM Errors', density=True)
        axes[0, 0].hist(baseline_errors, bins=50, alpha=0.5, label='Baseline Errors', density=True)
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].set_xlabel('Prediction Error')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Error over time
        axes[0, 1].plot(time_index, np.abs(lstm_errors), label='LSTM |Error|', alpha=0.7)
        axes[0, 1].plot(time_index, np.abs(baseline_errors), label='Baseline |Error|', alpha=0.7)
        axes[0, 1].set_title('Absolute Error Over Time')
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Cumulative squared error
        cumulative_lstm_error = np.cumsum(lstm_errors ** 2)
        cumulative_baseline_error = np.cumsum(baseline_errors ** 2)

        axes[1, 0].plot(time_index, cumulative_lstm_error, label='LSTM Cumulative SE')
        axes[1, 0].plot(time_index, cumulative_baseline_error, label='Baseline Cumulative SE')
        axes[1, 0].set_title('Cumulative Squared Error')
        axes[1, 0].set_ylabel('Cumulative Squared Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Regime-colored predictions
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for regime_id in np.unique(reg):
            regime_mask = reg == regime_id
            if np.sum(regime_mask) > 0:
                color = colors[int(regime_id) % len(colors)]
                axes[1, 1].scatter(act[regime_mask], pred[regime_mask],
                                 c=color, alpha=0.6, s=10, label=f'Regime {int(regime_id)}')

        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        axes[1, 1].set_xlabel('Actual')
        axes[1, 1].set_ylabel('Predicted')
        axes[1, 1].set_title('Predictions by Market Regime')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig2)

        # Figure 3: Performance Metrics
        fig3, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig3.suptitle(f'Performance Comparison - {symbol} {timeframe}', fontsize=16)

        # Calculate rolling metrics
        window = min(50, len(pred) // 10)
        rolling_mse_lstm = pd.Series(lstm_errors ** 2).rolling(window).mean()
        rolling_mse_baseline = pd.Series(baseline_errors ** 2).rolling(window).mean()

        ax.plot(time_index, rolling_mse_lstm, label=f'LSTM Rolling MSE (window={window})')
        ax.plot(time_index, rolling_mse_baseline, label=f'Baseline Rolling MSE (window={window})')
        ax.set_title('Rolling Mean Squared Error Comparison')
        ax.set_ylabel('Rolling MSE')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig3)

        return figures

    def generate_pdf_report(self, metrics: Dict, regime_performance: Dict, figures: List[plt.Figure],
                          symbol: str, timeframe: str, model_metadata: Dict) -> Path:
        """
        Generate comprehensive PDF report.

        Args:
            metrics: Performance metrics
            regime_performance: Regime-specific performance
            figures: List of matplotlib figures
            symbol: Trading symbol
            timeframe: Timeframe
            model_metadata: Model metadata

        Returns:
            Path to generated PDF report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"lstm_validation_{symbol}_{timeframe}_{timestamp}.pdf"
        pdf_path = self.reports_dir / pdf_filename

        with PdfPages(pdf_path) as pdf:
            # Title page
            fig_title = plt.figure(figsize=(8.5, 11))
            fig_title.text(0.5, 0.7, f'LSTM Model Validation Report',
                          ha='center', va='center', fontsize=24, weight='bold')
            fig_title.text(0.5, 0.6, f'{symbol} - {timeframe}',
                          ha='center', va='center', fontsize=18)
            fig_title.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                          ha='center', va='center', fontsize=12)

            # Model information
            model_info = [
                f"Model Architecture:",
                f"  - Input Size: {model_metadata['model_architecture']['input_size']}",
                f"  - Hidden Size: {model_metadata['model_architecture']['hidden_size']}",
                f"  - Layers: {model_metadata['model_architecture']['num_layers']}",
                f"  - Regimes: {model_metadata['model_architecture']['n_regimes']}",
                f"",
                f"Training Information:",
                f"  - Best Validation Loss: {model_metadata['training_results']['best_val_loss']:.6f}",
                f"  - Training Epochs: {model_metadata['training_results']['final_epoch']}",
                f"  - Features Used: {len(model_metadata['features'])}",
            ]

            y_pos = 0.4
            for line in model_info:
                fig_title.text(0.1, y_pos, line, ha='left', va='top', fontsize=10, family='monospace')
                y_pos -= 0.03

            pdf.savefig(fig_title, bbox_inches='tight')
            plt.close(fig_title)

            # Performance metrics page
            fig_metrics = plt.figure(figsize=(8.5, 11))
            fig_metrics.text(0.5, 0.95, 'Performance Metrics', ha='center', va='top',
                           fontsize=18, weight='bold')

            # LSTM vs Baseline comparison
            metrics_text = [
                "LSTM Model Performance:",
                f"  MSE: {metrics['lstm_metrics']['mse']:.6f}",
                f"  RMSE: {metrics['lstm_metrics']['rmse']:.6f}",
                f"  MAE: {metrics['lstm_metrics']['mae']:.6f}",
                f"  R^2: {metrics['lstm_metrics']['r2']:.4f}",
                f"  Directional Accuracy: {metrics['lstm_metrics']['directional_accuracy']:.2f}%",
                f"  Hit Rate: {metrics['lstm_metrics']['hit_rate']:.2f}%",
                f"  Sharpe Ratio: {metrics['lstm_metrics']['sharpe_ratio']:.4f}",
                "",
                "Naive Baseline Performance:",
                f"  MSE: {metrics['baseline_metrics']['mse']:.6f}",
                f"  RMSE: {metrics['baseline_metrics']['rmse']:.6f}",
                f"  MAE: {metrics['baseline_metrics']['mae']:.6f}",
                f"  R^2: {metrics['baseline_metrics']['r2']:.4f}",
                f"  Directional Accuracy: {metrics['baseline_metrics']['directional_accuracy']:.2f}%",
                f"  Hit Rate: {metrics['baseline_metrics']['hit_rate']:.2f}%",
                f"  Sharpe Ratio: {metrics['baseline_metrics']['sharpe_ratio']:.4f}",
                "",
                "Improvements (LSTM vs Baseline):",
                f"  MSE Improvement: {metrics['improvements']['mse_improvement_pct']:.2f}%",
                f"  MAE Improvement: {metrics['improvements']['mae_improvement_pct']:.2f}%",
                f"  R^2 Improvement: {metrics['improvements']['r2_improvement']:.4f}",
                f"  Directional Accuracy Gain: {metrics['improvements']['directional_accuracy_improvement']:.2f}%",
                f"",
                f"Test Sample Size: {metrics['sample_size']} observations"
            ]

            y_pos = 0.85
            for line in metrics_text:
                weight = 'bold' if line.endswith(':') else 'normal'
                fig_metrics.text(0.1, y_pos, line, ha='left', va='top', fontsize=10,
                               family='monospace', weight=weight)
                y_pos -= 0.03

            # Regime performance
            if regime_performance:
                y_pos -= 0.05
                fig_metrics.text(0.1, y_pos, "Performance by Market Regime:",
                               ha='left', va='top', fontsize=12, weight='bold')
                y_pos -= 0.04

                for regime, perf in regime_performance.items():
                    regime_text = [
                        f"{regime.replace('_', ' ').title()}:",
                        f"  Samples: {perf['sample_count']}",
                        f"  MSE: {perf['mse']:.6f}",
                        f"  MAE: {perf['mae']:.6f}",
                        f"  Directional Accuracy: {perf['directional_accuracy']:.2f}%",
                        f"  R^2: {perf['r2']:.4f}",
                        ""
                    ]

                    for line in regime_text:
                        weight = 'bold' if line.endswith(':') else 'normal'
                        fig_metrics.text(0.1, y_pos, line, ha='left', va='top', fontsize=9,
                                       family='monospace', weight=weight)
                        y_pos -= 0.025

            pdf.savefig(fig_metrics, bbox_inches='tight')
            plt.close(fig_metrics)

            # Add all visualization figures
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        logger.info(f"Generated PDF report: {pdf_path}")
        return pdf_path

    def validate_lstm(self, symbol: str, timeframe: str) -> Dict:
        """
        Validate LSTM model for a specific symbol-timeframe combination.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dict with validation results
        """
        logger.info(f"Validating LSTM for {symbol} {timeframe}")

        try:
            # Find and load model
            model_path = self.find_latest_model(symbol, timeframe)
            if model_path is None:
                raise FileNotFoundError(f"No trained LSTM model found for {symbol} {timeframe}")

            model_package = self.load_model(model_path)

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

            # Generate PDF report
            pdf_path = self.generate_pdf_report(
                metrics, regime_performance, figures, symbol, timeframe, model_package
            )

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
                'model_metadata': {
                    'architecture': model_package['model_architecture'],
                    'hyperparameters': model_package['hyperparameters'],
                    'features_count': len(model_package['features']),
                    'training_results': model_package['training_results']
                }
            }

            json_filename = f"lstm_validation_{symbol}_{timeframe}_{timestamp}.json"
            json_path = self.results_dir / json_filename

            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"[OK] LSTM validation completed for {symbol} {timeframe}")
            logger.info(f"  MSE improvement: {metrics['improvements']['mse_improvement_pct']:.2f}%")
            logger.info(f"  Directional accuracy: {metrics['lstm_metrics']['directional_accuracy']:.2f}%")
            logger.info(f"  PDF report: {pdf_path}")
            logger.info(f"  JSON results: {json_path}")

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': True,
                'pdf_report': str(pdf_path),
                'json_results': str(json_path),
                'mse_improvement': metrics['improvements']['mse_improvement_pct'],
                'directional_accuracy': metrics['lstm_metrics']['directional_accuracy'],
                'sample_size': metrics['sample_size']
            }

        except Exception as e:
            error_msg = f"Failed to validate LSTM for {symbol} {timeframe}: {str(e)}"
            logger.error(error_msg)
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
        symbols = self.config['symbols']
        timeframes = self.config['timeframes']

        logger.info(f"Validating LSTM models for {len(symbols)} symbols x {len(timeframes)} timeframes")

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
        logger.info(f"\n{'='*50}")
        logger.info(f"LSTM Validation Summary:")
        logger.info(f"  Total: {results['total']}")
        logger.info(f"  Successful: {len(results['successful'])}")
        logger.info(f"  Failed: {len(results['failed'])}")

        if results['successful']:
            avg_mse_improvement = np.mean([r['mse_improvement'] for r in results['successful']])
            avg_dir_accuracy = np.mean([r['directional_accuracy'] for r in results['successful']])
            logger.info(f"  Average MSE improvement: {avg_mse_improvement:.2f}%")
            logger.info(f"  Average directional accuracy: {avg_dir_accuracy:.2f}%")

        logger.info(f"{'='*50}")

        if results['failed']:
            logger.warning("Failed validations:")
            for failure in results['failed']:
                logger.warning(f"  {failure['symbol']} {failure['timeframe']}: {failure['error']}")

        return results

def main():
    """Main function to run LSTM validation."""
    try:
        validator = LSTMValidator()
        results = validator.validate_all()

        logger.info("LSTM validation completed!")

    except Exception as e:
        logger.error(f"LSTM validation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
