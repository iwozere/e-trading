"""
XGBoost Hyperparameter Optimization Stage for CNN + XGBoost Pipeline.

This module uses Optuna to optimize XGBoost hyperparameters for the classification task.
It performs time series cross-validation to ensure robust parameter selection.
"""

import sys
from pathlib import Path

# Add project root to path to import common utilities
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

import json
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score

from src.notification.logger import setup_logger
from src.util.config import load_config

_logger = setup_logger(__name__)


class XGBoostOptimizer:
    """
    XGBoost Hyperparameter Optimizer for the CNN + XGBoost pipeline.

    Uses Optuna to find optimal hyperparameters for XGBoost classification
    with time series cross-validation to ensure robust parameter selection.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the XGBoost optimizer.

        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.xgb_config = config.get("xgboost", {})

        _logger.info("Initializing XGBoost optimizer")

        # Create output directories
        self.models_dir = Path("src/ml/pipeline/p03_cnn_xgboost/models/xgboost")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Define target variables
        self.targets = ["target_direction", "target_volatility", "target_trend", "target_magnitude"]

        # Optimization settings
        self.n_trials = self.xgb_config.get("optimization_trials", 100)
        self.cv_splits = self.xgb_config.get("cv_splits", 5)

    def run(self) -> Dict[str, Any]:
        """
        Execute the XGBoost hyperparameter optimization stage.

        Returns:
            Dictionary containing optimization results and metadata
        """
        _logger.info("Starting XGBoost hyperparameter optimization stage")

        try:
            # Load feature data
            feature_files = self._discover_feature_data()
            if not feature_files:
                raise ValueError("No feature data files found")

            _logger.info("Found %d feature data files", len(feature_files))

            # Prepare training data
            X_train, y_train_dict = self._prepare_training_data(feature_files)

            # Optimize hyperparameters for each target
            optimization_results = {}
            best_params_dict = {}

            for target in self.targets:
                if target in y_train_dict:
                    _logger.info("Optimizing hyperparameters for target: %s", target)

                    target_results = self._optimize_target_hyperparameters(
                        X_train, y_train_dict[target], target
                    )

                    optimization_results[target] = target_results
                    best_params_dict[target] = target_results["best_params"]

                    _logger.info("Completed optimization for %s: best score = %.4f",
                                target, target_results["best_score"])

            # Save optimization results
            self._save_optimization_results(optimization_results, best_params_dict)

            _logger.info("XGBoost hyperparameter optimization stage completed successfully")
            return {
                "stage": "xgboost_optimization",
                "status": "completed",
                "targets_optimized": list(optimization_results.keys()),
                "best_params": best_params_dict,
                "optimization_results": optimization_results
            }

        except Exception as e:
            _logger.exception("Error in XGBoost optimization stage: %s", e)
            raise

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

    def _prepare_training_data(self, feature_files: List[Path]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare training data from feature files.

        Args:
            feature_files: List of paths to feature data files

        Returns:
            Tuple of (X_train, y_train_dict) where y_train_dict contains targets for each target variable
        """
        _logger.info("Preparing training data from %d files", len(feature_files))

        all_features = []
        all_targets = {target: [] for target in self.targets}

        # Track processing statistics
        processed_count = 0
        error_count = 0

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

                _logger.debug("Processed %s: %d samples, %d features",
                             file_path.name, len(features), len(feature_cols))

                processed_count += 1

            except Exception as e:
                _logger.error("Error processing %s: %s", file_path, e)
                _logger.exception("Full traceback for %s:", file_path.name)
                error_count += 1
                continue

        if not all_features:
            raise ValueError("No valid feature data found")

        _logger.info("Data preparation summary: %d files processed successfully, %d files failed",
                    processed_count, error_count)

        # Combine all data
        X_train = np.vstack(all_features)

        # Combine targets for each target variable
        y_train_dict = {}
        for target in self.targets:
            if all_targets[target]:
                y_train_dict[target] = np.concatenate(all_targets[target])

        _logger.info("Prepared training data: X shape %s, targets: %s",
                    X_train.shape, list(y_train_dict.keys()))

        return X_train, y_train_dict

    def _optimize_target_hyperparameters(self,
                                        X_train: np.ndarray,
                                        y_train: np.ndarray,
                                        target: str) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific target variable.

        Args:
            X_train: Training features
            y_train: Training targets for the specific target variable
            target: Name of the target variable

        Returns:
            Dictionary containing optimization results
        """
        def objective(trial):
            # Define hyperparameter search space
            params = {
                # Core parameters
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),

                # Regularization parameters
                "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),

                # Tree parameters
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "min_split_loss": trial.suggest_float("min_split_loss", 0.0, 1.0),

                # Other parameters
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss"
            }

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            cv_scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                # Train XGBoost model
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False
                )

                # Predict and evaluate
                y_pred_proba = model.predict_proba(X_fold_val)
                y_pred = model.predict(X_fold_val)

                # Calculate metrics
                logloss = log_loss(y_fold_val, y_pred_proba)
                accuracy = accuracy_score(y_fold_val, y_pred)
                precision = precision_score(y_fold_val, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_fold_val, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_fold_val, y_pred, average='weighted', zero_division=0)

                # Use log loss as primary metric (minimize)
                cv_scores.append(logloss)

            return np.mean(cv_scores)

        # Run optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)

        # Get best parameters and score
        best_params = study.best_trial.params
        best_score = study.best_trial.value

        # Train final model with best parameters
        final_model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1)
        final_model.fit(X_train, y_train)

        # Calculate final metrics
        y_pred_proba = final_model.predict_proba(X_train)
        y_pred = final_model.predict(X_train)

        final_metrics = {
            "log_loss": log_loss(y_train, y_pred_proba),
            "accuracy": accuracy_score(y_train, y_pred),
            "precision": precision_score(y_train, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_train, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_train, y_pred, average='weighted', zero_division=0)
        }

        return {
            "target": target,
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": self.n_trials,
            "cv_splits": self.cv_splits,
            "final_metrics": final_metrics,
            "study": study
        }

    def _save_optimization_results(self,
                                  optimization_results: Dict[str, Any],
                                  best_params_dict: Dict[str, Any]) -> None:
        """
        Save optimization results and best parameters.

        Args:
            optimization_results: Results from hyperparameter optimization
            best_params_dict: Dictionary of best parameters for each target
        """
        _logger.info("Saving optimization results")

        # Save best parameters
        params_path = self.models_dir / "best_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params_dict, f, indent=2)

        # Save optimization summary
        summary_path = self.models_dir / "optimization_summary.json"

        summary = {
            "stage": "xgboost_optimization",
            "status": "completed",
            "timestamp": pd.Timestamp.now().isoformat(),
            "n_trials": self.n_trials,
            "cv_splits": self.cv_splits,
            "targets": self.targets,
            "best_params": best_params_dict,
            "optimization_results": {
                target: {
                    "best_score": results["best_score"],
                    "final_metrics": results["final_metrics"],
                    "n_trials": results["n_trials"]
                }
                for target, results in optimization_results.items()
            }
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save detailed results for each target
        for target, results in optimization_results.items():
            target_dir = self.models_dir / target
            target_dir.mkdir(exist_ok=True)

            # Save detailed results
            detailed_path = target_dir / "optimization_results.json"
            with open(detailed_path, "w") as f:
                json.dump({
                    "target": target,
                    "best_params": results["best_params"],
                    "best_score": results["best_score"],
                    "final_metrics": results["final_metrics"],
                    "n_trials": results["n_trials"],
                    "cv_splits": results["cv_splits"]
                }, f, indent=2)

        _logger.info("Saved optimization results to %s", self.models_dir)


def optimize_xgboost(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to optimize XGBoost hyperparameters.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        Dictionary containing optimization results
    """
    optimizer = XGBoostOptimizer(config)
    return optimizer.run()


if __name__ == "__main__":
    # Load configuration
    config_path = Path("config/pipeline/p03.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(str(config_path))

    # Run XGBoost optimization
    results = optimize_xgboost(config)
    _logger.info("XGBoost Optimization Results: %s", results)
