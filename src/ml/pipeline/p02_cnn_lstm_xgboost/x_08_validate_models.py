"""
Stage 8: Model Validation
=========================

This stage performs comprehensive validation of both CNN-LSTM and XGBoost models,
comparing their performance and generating final evaluation reports.
"""

import sys
import yaml
import numpy as np
import json
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

class ModelValidator:
    """Comprehensive model validator for CNN-LSTM-XGBoost pipeline."""

    def __init__(self, config_path: str = "config/pipeline/x02.yaml"):
        """Initialize the model validator."""
        self.config_path = config_path
        self.config = self._load_config()

        # Create directories
        self.models_dir = Path("src/ml/pipeline/p02_cnn_lstm_xgboost/models")
        self.results_dir = self.models_dir / "results"
        self.reports_dir = self.results_dir / "reports"
        self.predictions_dir = self.results_dir / "predictions"

        for dir_path in [self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise Exception(f"Failed to load config from {self.config_path}: {e}")

    def _load_predictions(self) -> dict:
        """Load predictions from both models."""
        predictions = {}

        # Load CNN-LSTM predictions
        cnn_lstm_pred_path = self.predictions_dir / "cnn_lstm_predictions.npz"
        if cnn_lstm_pred_path.exists():
            cnn_lstm_data = np.load(cnn_lstm_pred_path)
            predictions['cnn_lstm'] = {
                'test': {'actual': cnn_lstm_data['test_actual'],
                        'predicted': cnn_lstm_data['test_predicted']}
            }

        # Load XGBoost predictions
        xgboost_pred_path = self.predictions_dir / "xgboost_predictions.npz"
        if xgboost_pred_path.exists():
            xgboost_data = np.load(xgboost_pred_path)
            predictions['xgboost'] = {
                'test': {'actual': xgboost_data['test_actual'],
                        'predicted': xgboost_data['test_predicted']}
            }

        return predictions

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

    def _generate_report(self, predictions: dict) -> dict:
        """Generate validation report."""
        report = {
            'pipeline_info': {
                'name': 'CNN-LSTM-XGBoost Pipeline',
                'version': '2.0'
            },
            'models': {},
            'comparison': {},
            'recommendations': []
        }

        # Calculate metrics for each model
        for model_name, model_predictions in predictions.items():
            report['models'][model_name] = {}

            for dataset_name, dataset_data in model_predictions.items():
                metrics = self._calculate_metrics(dataset_data['actual'], dataset_data['predicted'])
                report['models'][model_name][dataset_name] = metrics

        # Model comparison
        if 'cnn_lstm' in predictions and 'xgboost' in predictions:
            cnn_lstm_test = report['models']['cnn_lstm']['test']
            xgboost_test = report['models']['xgboost']['test']

            report['comparison'] = {
                'best_mse': 'cnn_lstm' if cnn_lstm_test['mse'] < xgboost_test['mse'] else 'xgboost',
                'best_directional_accuracy': 'cnn_lstm' if cnn_lstm_test['directional_accuracy'] > xgboost_test['directional_accuracy'] else 'xgboost'
            }

            # Generate recommendations
            if cnn_lstm_test['mse'] < xgboost_test['mse'] * 0.95:
                report['recommendations'].append("CNN-LSTM shows better MSE performance")
            elif xgboost_test['mse'] < cnn_lstm_test['mse'] * 0.95:
                report['recommendations'].append("XGBoost shows better MSE performance")
            else:
                report['recommendations'].append("Both models show similar MSE performance")

        return report

    def validate(self) -> dict:
        """Perform model validation."""
        _logger.info("Starting model validation...")

        # Load predictions
        predictions = self._load_predictions()

        if not predictions:
            raise Exception("No predictions found for validation")

        # Generate report
        report = self._generate_report(predictions)

        # Save report
        report_path = self.reports_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Create summary table
        summary_data = []
        for model_name, model_results in report['models'].items():
            for dataset_name, metrics in model_results.items():
                summary_data.append({
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'MSE': f"{metrics['mse']:.6f}",
                    'MAE': f"{metrics['mae']:.6f}",
                    'RMSE': f"{metrics['rmse']:.6f}",
                    'Directional Accuracy': f"{metrics['directional_accuracy']:.4f}",
                    'R-squared': f"{metrics['r_squared']:.4f}"
                })

        summary_df = pd.DataFrame(summary_data)
        summary_path = self.reports_dir / "validation_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        # Log results
        _logger.info("Model validation completed")

        # Print summary
        print("\n" + "="*60)
        print("MODEL VALIDATION SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False))

        if 'comparison' in report and report['comparison']:
            print("\n" + "="*40)
            print("MODEL COMPARISON")
            print("="*40)
            for metric, best_model in report['comparison'].items():
                print(f"Best {metric}: {best_model}")

        if report['recommendations']:
            print("\n" + "="*40)
            print("RECOMMENDATIONS")
            print("="*40)
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")

        print("="*60)

        return report

def main():
    """Main function to run model validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Model Validation")
    parser.add_argument("--config", default="config/pipeline/x02.yaml",
                       help="Path to configuration file")

    args = parser.parse_args()

    try:
        validator = ModelValidator(args.config)
        report = validator.validate()

    except Exception as e:
        print(f"Error during model validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
