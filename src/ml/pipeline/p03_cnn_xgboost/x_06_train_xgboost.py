"""
XGBoost Training Stage for CNN + XGBoost Pipeline.

This module trains the final XGBoost models using the optimized hyperparameters
for each target variable. It implements a multi-target classification approach
with time series cross-validation for robust model training.
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
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, classification_report

from src.notification.logger import setup_logger
from src.util.config import load_config

_logger = setup_logger(__name__)


class XGBoostTrainer:
    """
    XGBoost Trainer for the CNN + XGBoost pipeline.

    Trains final XGBoost models using optimized hyperparameters for each target variable.
    Implements time series cross-validation and saves trained models for deployment.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the XGBoost trainer.

        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.xgb_config = config.get("xgboost", {})

        _logger.info("Initializing XGBoost trainer")

        # Create output directories
        self.models_dir = Path("src/ml/pipeline/p03_cnn_xgboost/models/xgboost")
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Define target variables
        self.targets = ["target_direction", "target_volatility", "target_trend", "target_magnitude"]

        # Training settings
        self.cv_splits = self.xgb_config.get("cv_splits", 5)
        self.test_size = self.xgb_config.get("test_size", 0.2)

        # Load optimized hyperparameters
        self.best_params = self._load_best_params()

    def _get_checkpoint_path(self, target: str) -> Path:
        """Get checkpoint file path for a specific target."""
        return self.checkpoints_dir / f"xgboost_checkpoint_{target}.pkl"

    def save_checkpoint(self, model: xgb.XGBClassifier, target: str, iteration: int,
                       train_metrics: dict, val_metrics: dict):
        """Save model state so training can resume later."""
        ckpt_path = self._get_checkpoint_path(target)
        checkpoint = {
            'iteration': iteration,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_params': self.best_params.get(target, {})
        }

        with open(ckpt_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        _logger.info("Checkpoint saved for %s at %s (iteration %d)", target, ckpt_path, iteration)

    def load_checkpoint(self, model: xgb.XGBClassifier, target: str) -> Tuple[int, dict, dict]:
        """Load model state to resume training if checkpoint exists."""
        ckpt_path = self._get_checkpoint_path(target)
        if ckpt_path.exists():
            with open(ckpt_path, 'rb') as f:
                checkpoint = pickle.load(f)

            # For now, just return the checkpoint info without loading the model state
            # since we can't access booster on unfitted model in XGBoost 3.x
            start_iteration = checkpoint['iteration'] + 1
            train_metrics = checkpoint['train_metrics']
            val_metrics = checkpoint['val_metrics']

            _logger.info("Found checkpoint for %s at %s (iteration %d) - will start fresh training",
                        target, ckpt_path, start_iteration)
            return start_iteration, train_metrics, val_metrics
        else:
            return 0, {}, {}

    def run(self) -> Dict[str, Any]:
        """
        Execute the XGBoost training stage.

        Returns:
            Dictionary containing training results and metadata
        """
        _logger.info("Starting XGBoost training stage")

        try:
            # Load feature data
            feature_files = self._discover_feature_data()
            if not feature_files:
                raise ValueError("No feature data files found")

            _logger.info("Found %d feature data files", len(feature_files))

            # Prepare training data
            X_train, y_train_dict = self._prepare_training_data(feature_files)

            # Train models for each target
            training_results = {}
            trained_models = {}

            for target in self.targets:
                if target in y_train_dict and target in self.best_params:
                    _logger.info("Training XGBoost model for target: %s", target)

                    target_results = self._train_target_model(
                        X_train, y_train_dict[target], target
                    )

                    training_results[target] = target_results
                    trained_models[target] = target_results["model"]

                    _logger.info("Completed training for %s: final accuracy = %.4f",
                                target, target_results["final_metrics"]["accuracy"])

            # Save trained models
            self._save_trained_models(trained_models)

            # Generate ensemble predictions
            ensemble_results = self._generate_ensemble_predictions(trained_models, X_train, y_train_dict)

            # Save training summary
            self._save_training_summary(training_results, ensemble_results)

            _logger.info("XGBoost training stage completed successfully")
            return {
                "stage": "xgboost_training",
                "status": "completed",
                "targets_trained": list(training_results.keys()),
                "training_results": training_results,
                "ensemble_results": ensemble_results
            }

        except Exception as e:
            _logger.exception("Error in XGBoost training stage: %s", e)
            raise

    def _load_best_params(self) -> Dict[str, Any]:
        """
        Load optimized hyperparameters from the optimization stage.

        Returns:
            Dictionary of best parameters for each target
        """
        params_path = self.models_dir / "best_params.json"
        if not params_path.exists():
            raise FileNotFoundError(f"Best parameters file not found: {params_path}")

        with open(params_path, "r") as f:
            best_params = json.load(f)

        _logger.info("Loaded best parameters for targets: %s", list(best_params.keys()))
        return best_params

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

            except Exception as e:
                _logger.warning("Error processing %s: %s", file_path, e)
                continue

        if not all_features:
            raise ValueError("No valid feature data found")

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

    def _train_target_model(self,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           target: str) -> Dict[str, Any]:
        """
        Train XGBoost model for a specific target variable.

        Args:
            X_train: Training features
            y_train: Training targets for the specific target variable
            target: Name of the target variable

        Returns:
            Dictionary containing training results
        """
        _logger.info("Training model for target: %s", target)

        # Get best parameters for this target
        best_params = self.best_params[target].copy()
        best_params.update({
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        })

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        cv_results = {
            "train_scores": [],
            "val_scores": [],
            "train_metrics": [],
            "val_metrics": []
        }

        models = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            _logger.debug("Training fold %d/%d", fold + 1, self.cv_splits)

            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # Train XGBoost model
            model = xgb.XGBClassifier(**best_params, early_stopping_rounds=50)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )

            models.append(model)

            # Evaluate on train and validation sets
            y_train_pred_proba = model.predict_proba(X_fold_train)
            y_train_pred = model.predict(X_fold_train)
            y_val_pred_proba = model.predict_proba(X_fold_val)
            y_val_pred = model.predict(X_fold_val)

            # Calculate metrics
            train_metrics = self._calculate_metrics(y_fold_train, y_train_pred, y_train_pred_proba)
            val_metrics = self._calculate_metrics(y_fold_val, y_val_pred, y_val_pred_proba)

            cv_results["train_scores"].append(train_metrics["log_loss"])
            cv_results["val_scores"].append(val_metrics["log_loss"])
            cv_results["train_metrics"].append(train_metrics)
            cv_results["val_metrics"].append(val_metrics)

        # Train final model on full dataset
        final_model = xgb.XGBClassifier(**best_params)

        # Check if checkpoint exists (for logging purposes only)
        start_iteration, train_metrics, val_metrics = self.load_checkpoint(final_model, target)

        final_model.fit(X_train, y_train)

        # Save checkpoint after training
        final_iteration = final_model.n_estimators if hasattr(final_model, 'n_estimators') else 100
        self.save_checkpoint(final_model, target, final_iteration, train_metrics, val_metrics)

        # Final evaluation
        y_pred_proba = final_model.predict_proba(X_train)
        y_pred = final_model.predict(X_train)
        final_metrics = self._calculate_metrics(y_train, y_pred, y_pred_proba)

        # Feature importance
        feature_importance = final_model.feature_importances_

        # Classification report
        classification_rep = classification_report(y_train, y_pred, output_dict=True)

        return {
            "target": target,
            "model": final_model,
            "cv_models": models,
            "best_params": best_params,
            "cv_results": cv_results,
            "final_metrics": final_metrics,
            "feature_importance": feature_importance.tolist(),
            "classification_report": classification_rep,
            "cv_mean_val_score": np.mean(cv_results["val_scores"]),
            "cv_std_val_score": np.std(cv_results["val_scores"])
        }

    def _calculate_metrics(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for model evaluation.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        return {
            "log_loss": log_loss(y_true, y_pred_proba),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def _generate_ensemble_predictions(self,
                                     trained_models: Dict[str, xgb.XGBClassifier],
                                     X_train: np.ndarray,
                                     y_train_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Generate ensemble predictions using all trained models.

        Args:
            trained_models: Dictionary of trained models for each target
            X_train: Training features
            y_train_dict: Training targets for each target variable

        Returns:
            Dictionary containing ensemble results
        """
        _logger.info("Generating ensemble predictions")

        ensemble_results = {}

        for target, model in trained_models.items():
            if target in y_train_dict:
                # Generate predictions
                y_pred_proba = model.predict_proba(X_train)
                y_pred = model.predict(X_train)
                y_true = y_train_dict[target]

                # Calculate ensemble metrics
                ensemble_metrics = self._calculate_metrics(y_true, y_pred, y_pred_proba)

                # Store results
                ensemble_results[target] = {
                    "predictions": y_pred.tolist(),
                    "probabilities": y_pred_proba.tolist(),
                    "true_labels": y_true.tolist(),
                    "metrics": ensemble_metrics
                }

        return ensemble_results

    def _save_trained_models(self, trained_models: Dict[str, xgb.XGBClassifier]) -> None:
        """
        Save trained XGBoost models.

        Args:
            trained_models: Dictionary of trained models for each target
        """
        _logger.info("Saving trained models")

        for target, model in trained_models.items():
            # Save model
            model_path = self.models_dir / f"{target}_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Save model info
            info_path = self.models_dir / f"{target}_model_info.json"
            model_info = {
                "target": target,
                "model_type": "XGBClassifier",
                "n_features": model.n_features_in_,
                "n_classes": len(model.classes_),
                "classes": model.classes_.tolist(),
                "feature_importances": model.feature_importances_.tolist()
            }

            with open(info_path, "w") as f:
                json.dump(model_info, f, indent=2)

        _logger.info("Saved %d trained models", len(trained_models))

    def _save_training_summary(self,
                              training_results: Dict[str, Any],
                              ensemble_results: Dict[str, Any]) -> None:
        """
        Save training summary and results.

        Args:
            training_results: Results from model training
            ensemble_results: Results from ensemble predictions
        """
        _logger.info("Saving training summary")

        # Save training summary
        summary_path = self.models_dir / "training_summary.json"

        summary = {
            "stage": "xgboost_training",
            "status": "completed",
            "timestamp": pd.Timestamp.now().isoformat(),
            "targets_trained": list(training_results.keys()),
            "cv_splits": self.cv_splits,
            "training_results": {
                target: {
                    "cv_mean_val_score": results["cv_mean_val_score"],
                    "cv_std_val_score": results["cv_std_val_score"],
                    "final_metrics": results["final_metrics"]
                }
                for target, results in training_results.items()
            },
            "ensemble_results": {
                target: {
                    "metrics": results["metrics"]
                }
                for target, results in ensemble_results.items()
            }
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save detailed results for each target
        for target, results in training_results.items():
            target_dir = self.models_dir / target
            target_dir.mkdir(exist_ok=True)

            # Save detailed training results
            detailed_path = target_dir / "training_results.json"
            with open(detailed_path, "w") as f:
                json.dump({
                    "target": target,
                    "best_params": results["best_params"],
                    "cv_results": results["cv_results"],
                    "final_metrics": results["final_metrics"],
                    "classification_report": results["classification_report"],
                    "feature_importance": results["feature_importance"]
                }, f, indent=2)

        _logger.info("Saved training summary to %s", self.models_dir)


def train_xgboost(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to train XGBoost models.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        Dictionary containing training results
    """
    trainer = XGBoostTrainer(config)
    return trainer.run()


if __name__ == "__main__":
    # Load configuration
    config_path = Path("config/pipeline/p03.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(str(config_path))

    # Run XGBoost training
    results = train_xgboost(config)
    _logger.info("XGBoost Training Results: %s", results)
