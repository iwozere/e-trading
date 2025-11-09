"""
Model Validation Stage for CNN + XGBoost Pipeline.

This module performs comprehensive validation of the trained models including
backtesting, performance analysis, and generation of detailed reports.
"""

import sys
from pathlib import Path

# Add project root to path to import common utilities
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import TimeSeriesSplit

from src.notification.logger import setup_logger
from src.util.config import load_config

_logger = setup_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class ModelValidator:
    """
    Model Validator for the CNN + XGBoost pipeline.

    Performs comprehensive validation including backtesting, performance analysis,
    and generation of detailed reports and visualizations.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the model validator.

        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.validation_config = config.get("validation", {})

        _logger.info("Initializing model validator")

        # Create output directories
        self.results_dir = Path("src/ml/pipeline/p03_cnn_xgboost/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Define target variables
        self.targets = ["target_direction", "target_volatility", "target_trend", "target_magnitude"]

        # Validation settings
        self.cv_splits = self.validation_config.get("cv_splits", 5)
        self.backtest_window = self.validation_config.get("backtest_window", 252)  # 1 year

    def run(self) -> Dict[str, Any]:
        """
        Execute the model validation stage.

        Returns:
            Dictionary containing validation results and metadata
        """
        _logger.info("Starting model validation stage")

        try:
            # Load trained models
            trained_models = self._load_trained_models()

            # Load feature data
            feature_files = self._discover_feature_data()
            if not feature_files:
                raise ValueError("No feature data files found")

            _logger.info("Found %d feature data files", len(feature_files))

            # Prepare validation data
            X_val, y_val_dict = self._prepare_validation_data(feature_files)

            # Perform comprehensive validation
            validation_results = self._perform_validation(trained_models, X_val, y_val_dict)

            # Perform backtesting
            backtest_results = self._perform_backtesting(trained_models, feature_files)

            # Generate performance reports
            performance_reports = self._generate_performance_reports(validation_results, backtest_results)

            # Create visualizations
            self._create_visualizations(validation_results, backtest_results)

            # Save validation summary
            self._save_validation_summary(validation_results, backtest_results, performance_reports)

            _logger.info("Model validation stage completed successfully")
            return {
                "stage": "model_validation",
                "status": "completed",
                "validation_results": validation_results,
                "backtest_results": backtest_results,
                "performance_reports": performance_reports
            }

        except Exception as e:
            _logger.exception("Error in model validation stage: %s", e)
            raise

    def _load_trained_models(self) -> Dict[str, Any]:
        """
        Load trained XGBoost models.

        Returns:
            Dictionary of trained models for each target
        """
        models_dir = Path("src/ml/pipeline/p03_cnn_xgboost/models/xgboost")
        trained_models = {}

        for target in self.targets:
            model_path = models_dir / f"{target}_model.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    trained_models[target] = pickle.load(f)
                _logger.info("Loaded model for target: %s", target)
            else:
                _logger.warning("Model not found for target: %s", target)

        return trained_models

    def _discover_feature_data(self) -> List[Path]:
        """
        Discover feature data files from the TA feature engineering stage.

        Returns:
            List of paths to feature data files
        """
        features_dir = Path("data/features")
        if not features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {features_dir}")

        # Look for CSV files with features
        feature_files = list(features_dir.glob("*_features.csv"))

        if not feature_files:
            # Fallback to any CSV files
            feature_files = list(features_dir.glob("*.csv"))

        return feature_files

    def _prepare_validation_data(self, feature_files: List[Path]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare validation data from feature files.

        Args:
            feature_files: List of paths to feature data files

        Returns:
            Tuple of (X_val, y_val_dict) where y_val_dict contains targets for each target variable
        """
        _logger.info("Preparing validation data from %d files", len(feature_files))

        all_features = []
        all_targets = {target: [] for target in self.targets}

        for file_path in feature_files:
            try:
                # Load feature data
                df = pd.read_csv(file_path)

                # Extract feature columns (exclude targets and metadata)
                exclude_cols = self.targets + ["date", "timestamp", "sequence_start_idx", "sequence_end_idx"]
                feature_cols = [col for col in df.columns if col not in exclude_cols]

                if not feature_cols:
                    _logger.warning("No feature columns found in %s, skipping", file_path)
                    continue

                # Extract features and targets
                features = df[feature_cols].values
                all_features.append(features)

                for target in self.targets:
                    if target in df.columns:
                        # Convert target to numeric type to handle string/int mismatches
                        target_values = pd.to_numeric(df[target], errors='coerce')

                        # Check for any NaN values after conversion
                        nan_count = target_values.isna().sum()
                        if nan_count > 0:
                            _logger.warning("Found %d NaN values in target %s for file %s",
                                          nan_count, target, file_path.name)
                            # Fill NaN values with mode or most common value
                            target_values = target_values.fillna(target_values.mode().iloc[0] if len(target_values.mode()) > 0 else 0)

                        all_targets[target].append(target_values.values)

            except Exception as e:
                _logger.warning("Error processing %s: %s", file_path, e)
                continue

        if not all_features:
            raise ValueError("No valid feature data found")

        # Combine all data
        X_val = np.vstack(all_features)

        # Combine targets for each target variable
        y_val_dict = {}
        for target in self.targets:
            if all_targets[target]:
                y_val_dict[target] = np.concatenate(all_targets[target])

        _logger.info("Prepared validation data: X shape %s, targets: %s",
                    X_val.shape, list(y_val_dict.keys()))

        return X_val, y_val_dict

    def _perform_validation(self,
                           trained_models: Dict[str, Any],
                           X_val: np.ndarray,
                           y_val_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Perform comprehensive validation of trained models.

        Args:
            trained_models: Dictionary of trained models for each target
            X_val: Validation features
            y_val_dict: Validation targets for each target variable

        Returns:
            Dictionary containing validation results
        """
        _logger.info("Performing comprehensive validation")

        validation_results = {}

        for target, model in trained_models.items():
            if target in y_val_dict:
                _logger.info("Validating model for target: %s", target)

                y_true = y_val_dict[target]
                y_pred_proba = model.predict_proba(X_val)
                y_pred = model.predict(X_val)

                # Calculate comprehensive metrics
                target_results = self._calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)

                # Time series cross-validation
                cv_results = self._perform_time_series_cv(model, X_val, y_true)

                # Feature importance analysis
                feature_importance = self._analyze_feature_importance(model, X_val)

                validation_results[target] = {
                    "metrics": target_results,
                    "cv_results": cv_results,
                    "feature_importance": feature_importance,
                    "predictions": y_pred.tolist(),
                    "probabilities": y_pred_proba.tolist(),
                    "true_labels": y_true.tolist()
                }

        return validation_results

    def _calculate_comprehensive_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for model evaluation.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary of comprehensive metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, log_loss,
            confusion_matrix, classification_report
        )

        # Basic metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "log_loss": log_loss(y_true, y_pred_proba)
        }

        # ROC and PR metrics (for binary classification)
        if len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                metrics["average_precision"] = average_precision_score(y_true, y_pred_proba[:, 1])
            except Exception as e:
                _logger.warning("Could not calculate ROC/PR metrics: %s", e)
                metrics["roc_auc"] = None
                metrics["average_precision"] = None

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Classification report
        metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True)

        return metrics

    def _perform_time_series_cv(self,
                               model: Any,
                               X: np.ndarray,
                               y: np.ndarray) -> Dict[str, Any]:
        """
        Perform time series cross-validation.

        Args:
            model: Trained model
            X: Features
            y: Targets

        Returns:
            Dictionary containing CV results
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Retrain model on this fold
            fold_model = type(model)(**model.get_params())
            fold_model.fit(X_train, y_train)

            # Evaluate
            y_pred_proba = fold_model.predict_proba(X_val)
            y_pred = fold_model.predict(X_val)

            # Calculate metrics
            fold_metrics = self._calculate_comprehensive_metrics(y_val, y_pred, y_pred_proba)
            cv_scores.append(fold_metrics)

        # Aggregate CV results
        cv_results = {
            "mean_accuracy": np.mean([score["accuracy"] for score in cv_scores]),
            "std_accuracy": np.std([score["accuracy"] for score in cv_scores]),
            "mean_f1": np.mean([score["f1_score"] for score in cv_scores]),
            "std_f1": np.std([score["f1_score"] for score in cv_scores]),
            "fold_scores": cv_scores
        }

        return cv_results

    def _analyze_feature_importance(self, model: Any, X: np.ndarray) -> Dict[str, Any]:
        """
        Analyze feature importance for the model.

        Args:
            model: Trained model
            X: Features

        Returns:
            Dictionary containing feature importance analysis
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            # Get top features
            top_indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            top_features = {
                f"feature_{i}": {
                    "importance": float(importances[i]),
                    "rank": int(np.where(np.argsort(importances)[::-1] == i)[0][0])
                }
                for i in top_indices
            }

            return {
                "importances": importances.tolist(),
                "top_features": top_features,
                "total_features": len(importances)
            }
        else:
            return {"error": "Model does not support feature importance"}

    def _perform_backtesting(self,
                            trained_models: Dict[str, Any],
                            feature_files: List[Path]) -> Dict[str, Any]:
        """
        Perform backtesting on historical data.

        Args:
            trained_models: Dictionary of trained models for each target
            feature_files: List of paths to feature data files

        Returns:
            Dictionary containing backtesting results
        """
        _logger.info("Performing backtesting analysis")

        backtest_results = {}

        for file_path in feature_files:
            try:
                # Load feature data
                df = pd.read_csv(file_path)

                # Extract features and targets
                exclude_cols = self.targets + ["date", "timestamp", "sequence_start_idx", "sequence_end_idx"]
                feature_cols = [col for col in df.columns if col not in exclude_cols]

                if not feature_cols:
                    continue

                features = df[feature_cols].values

                # Perform walk-forward backtesting
                file_results = self._walk_forward_backtest(trained_models, features, df)

                backtest_results[file_path.stem] = file_results

            except Exception as e:
                _logger.warning("Error in backtesting %s: %s", file_path, e)
                continue

        return backtest_results

    def _walk_forward_backtest(self,
                              trained_models: Dict[str, Any],
                              features: np.ndarray,
                              df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform walk-forward backtesting.

        Args:
            trained_models: Dictionary of trained models for each target
            features: Feature array
            df: Original DataFrame with targets

        Returns:
            Dictionary containing backtesting results
        """
        results = {}

        for target, model in trained_models.items():
            if target not in df.columns:
                continue

            y_true = df[target].values

            # Walk-forward prediction
            predictions = []
            probabilities = []

            for i in range(self.backtest_window, len(features)):
                # Use data up to current point for prediction
                X_current = features[i:i+1]
                y_pred_proba = model.predict_proba(X_current)
                y_pred = model.predict(X_current)

                predictions.append(y_pred[0])
                probabilities.append(y_pred_proba[0].tolist())

            # Calculate backtesting metrics
            if len(predictions) > 0:
                y_true_backtest = y_true[self.backtest_window:]
                y_pred_backtest = np.array(predictions)
                y_pred_proba_backtest = np.array(probabilities)

                backtest_metrics = self._calculate_comprehensive_metrics(
                    y_true_backtest, y_pred_backtest, y_pred_proba_backtest
                )

                results[target] = {
                    "metrics": backtest_metrics,
                    "predictions": predictions,
                    "probabilities": probabilities,
                    "true_labels": y_true_backtest.tolist()
                }

        return results

    def _generate_performance_reports(self,
                                    validation_results: Dict[str, Any],
                                    backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive performance reports.

        Args:
            validation_results: Results from model validation
            backtest_results: Results from backtesting

        Returns:
            Dictionary containing performance reports
        """
        _logger.info("Generating performance reports")

        reports = {
            "summary": self._generate_summary_report(validation_results, backtest_results),
            "detailed": self._generate_detailed_reports(validation_results, backtest_results),
            "recommendations": self._generate_recommendations(validation_results, backtest_results)
        }

        return reports

    def _generate_summary_report(self,
                                validation_results: Dict[str, Any],
                                backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary performance report.

        Args:
            validation_results: Results from model validation
            backtest_results: Results from backtesting

        Returns:
            Dictionary containing summary report
        """
        summary = {
            "total_targets": len(validation_results),
            "targets": list(validation_results.keys()),
            "overall_performance": {},
            "best_performing_target": None,
            "worst_performing_target": None
        }

        # Calculate overall performance
        accuracies = []
        f1_scores = []

        for target, results in validation_results.items():
            accuracies.append(results["metrics"]["accuracy"])
            f1_scores.append(results["metrics"]["f1_score"])

        summary["overall_performance"] = {
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "mean_f1": np.mean(f1_scores),
            "std_f1": np.std(f1_scores)
        }

        # Find best and worst performing targets
        if accuracies:
            best_idx = np.argmax(accuracies)
            worst_idx = np.argmin(accuracies)
            summary["best_performing_target"] = list(validation_results.keys())[best_idx]
            summary["worst_performing_target"] = list(validation_results.keys())[worst_idx]

        return summary

    def _generate_detailed_reports(self,
                                  validation_results: Dict[str, Any],
                                  backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed performance reports.

        Args:
            validation_results: Results from model validation
            backtest_results: Results from backtesting

        Returns:
            Dictionary containing detailed reports
        """
        detailed = {
            "validation_reports": validation_results,
            "backtest_reports": backtest_results
        }

        return detailed

    def _generate_recommendations(self,
                                 validation_results: Dict[str, Any],
                                 backtest_results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on validation results.

        Args:
            validation_results: Results from model validation
            backtest_results: Results from backtesting

        Returns:
            List of recommendations
        """
        recommendations = []

        # Analyze performance
        for target, results in validation_results.items():
            accuracy = results["metrics"]["accuracy"]
            f1 = results["metrics"]["f1_score"]

            if accuracy < 0.6:
                recommendations.append(f"Target '{target}' has low accuracy ({accuracy:.3f}). Consider feature engineering or hyperparameter tuning.")

            if f1 < 0.5:
                recommendations.append(f"Target '{target}' has low F1 score ({f1:.3f}). Check for class imbalance or model complexity.")

        # General recommendations
        if len(validation_results) > 0:
            mean_accuracy = np.mean([r["metrics"]["accuracy"] for r in validation_results.values()])
            if mean_accuracy > 0.7:
                recommendations.append("Overall model performance is good. Consider ensemble methods for further improvement.")
            else:
                recommendations.append("Overall model performance needs improvement. Consider data quality, feature selection, or model architecture changes.")

        return recommendations

    def _create_visualizations(self,
                              validation_results: Dict[str, Any],
                              backtest_results: Dict[str, Any]) -> None:
        """
        Create performance visualizations.

        Args:
            validation_results: Results from model validation
            backtest_results: Results from backtesting
        """
        _logger.info("Creating performance visualizations")

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Create visualizations directory
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # 1. Performance comparison across targets
        self._plot_performance_comparison(validation_results, viz_dir)

        # 2. Confusion matrices
        self._plot_confusion_matrices(validation_results, viz_dir)

        # 3. Feature importance plots
        self._plot_feature_importance(validation_results, viz_dir)

        # 4. ROC curves (for binary targets)
        self._plot_roc_curves(validation_results, viz_dir)

        _logger.info("Visualizations saved to %s", viz_dir)

    def _plot_performance_comparison(self,
                                   validation_results: Dict[str, Any],
                                   viz_dir: Path) -> None:
        """Plot performance comparison across targets."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for i, metric in enumerate(metrics):
            values = [results["metrics"][metric] for results in validation_results.values()]
            targets = list(validation_results.keys())

            axes[i].bar(targets, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()} by Target')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(viz_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrices(self,
                                validation_results: Dict[str, Any],
                                viz_dir: Path) -> None:
        """Plot confusion matrices for each target."""
        n_targets = len(validation_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for i, (target, results) in enumerate(validation_results.items()):
            if i < 4:  # Limit to 4 plots
                cm = np.array(results["metrics"]["confusion_matrix"])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(f'Confusion Matrix - {target}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig(viz_dir / "confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self,
                                validation_results: Dict[str, Any],
                                viz_dir: Path) -> None:
        """Plot feature importance for each target."""
        for target, results in validation_results.items():
            if "feature_importance" in results and "importances" in results["feature_importance"]:
                importances = results["feature_importance"]["importances"]
                top_indices = np.argsort(importances)[::-1][:20]

                plt.figure(figsize=(12, 8))
                plt.barh(range(len(top_indices)), [importances[i] for i in top_indices])
                plt.yticks(range(len(top_indices)), [f'Feature_{i}' for i in top_indices])
                plt.xlabel('Feature Importance')
                plt.title(f'Top 20 Feature Importance - {target}')
                plt.gca().invert_yaxis()

                plt.tight_layout()
                plt.savefig(viz_dir / f"feature_importance_{target}.png", dpi=300, bbox_inches='tight')
                plt.close()

    def _plot_roc_curves(self,
                        validation_results: Dict[str, Any],
                        viz_dir: Path) -> None:
        """Plot ROC curves for binary targets."""
        plt.figure(figsize=(10, 8))

        for target, results in validation_results.items():
            y_true = np.array(results["true_labels"])
            y_pred_proba = np.array(results["probabilities"])

            if len(np.unique(y_true)) == 2 and y_pred_proba.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr, label=f'{target} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(viz_dir / "roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _save_validation_summary(self,
                                validation_results: Dict[str, Any],
                                backtest_results: Dict[str, Any],
                                performance_reports: Dict[str, Any]) -> None:
        """
        Save comprehensive validation summary.

        Args:
            validation_results: Results from model validation
            backtest_results: Results from backtesting
            performance_reports: Performance reports
        """
        _logger.info("Saving validation summary")

        summary_path = self.results_dir / "validation_summary.json"

        summary = {
            "stage": "model_validation",
            "status": "completed",
            "timestamp": pd.Timestamp.now().isoformat(),
            "validation_results": validation_results,
            "backtest_results": backtest_results,
            "performance_reports": performance_reports
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)

        _logger.info("Validation summary saved to %s", summary_path)


def validate_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to validate models.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        Dictionary containing validation results
    """
    validator = ModelValidator(config)
    return validator.run()


if __name__ == "__main__":
    # Load configuration
    config_path = Path("config/pipeline/p03.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(str(config_path))

    # Run model validation
    results = validate_models(config)
    _logger.info("Model Validation Results: %s", results)
