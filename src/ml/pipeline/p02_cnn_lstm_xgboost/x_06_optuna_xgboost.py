"""
Stage 6: XGBoost Hyperparameter Optimization
============================================

This stage optimizes XGBoost hyperparameters using Optuna.
It uses the combined features (CNN-LSTM features + technical indicators)
extracted in the previous stage.
"""

import sys
import yaml
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

class XGBoostOptimizer:
    """XGBoost hyperparameter optimizer using Optuna."""

    def __init__(self, config_path: str = "config/pipeline/x02.yaml"):
        """Initialize the XGBoost optimizer."""
        self.config_path = config_path
        self.config = self._load_config()

        # Create directories
        self.models_dir = Path("src/ml/pipeline/p02_cnn_lstm_xgboost/models")
        self.xgboost_dir = self.models_dir / "xgboost"
        self.studies_dir = self.xgboost_dir / "studies"
        self.configs_dir = self.xgboost_dir / "configs"

        for dir_path in [self.studies_dir, self.configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise Exception(f"Failed to load config from {self.config_path}: {e}")

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
        _logger.info("  Train: %d, %d", X_train.shape, y_train.shape)
        _logger.info("  Validation: %d, %d", X_val.shape, y_val.shape)
        _logger.info("  Test: %d, %d", X_test.shape, y_test.shape)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Optuna objective function for XGBoost hyperparameter optimization."""
        # Get hyperparameters from trial
        params = {
            'n_estimators': trial.suggest_int('n_estimators',
                                            self.config['xgboost']['n_estimators']['min'],
                                            self.config['xgboost']['n_estimators']['max']),
            'learning_rate': trial.suggest_float('learning_rate',
                                               self.config['xgboost']['learning_rate']['min'],
                                               self.config['xgboost']['learning_rate']['max'],
                                               log=True),
            'max_depth': trial.suggest_int('max_depth',
                                         self.config['xgboost']['max_depth']['min'],
                                         self.config['xgboost']['max_depth']['max']),
            'subsample': trial.suggest_float('subsample',
                                           self.config['xgboost']['subsample']['min'],
                                           self.config['xgboost']['subsample']['max']),
            'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                  self.config['xgboost']['colsample_bytree']['min'],
                                                  self.config['xgboost']['colsample_bytree']['max']),
            'gamma': trial.suggest_float('gamma',
                                       self.config['xgboost']['gamma']['min'],
                                       self.config['xgboost']['gamma']['max']),
            'reg_lambda': trial.suggest_float('reg_lambda',
                                            self.config['xgboost']['reg_lambda']['min'],
                                            self.config['xgboost']['reg_lambda']['max']),
            'reg_alpha': trial.suggest_float('reg_alpha',
                                           self.config['xgboost']['reg_alpha']['min'],
                                           self.config['xgboost']['reg_alpha']['max']),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }

        # Train XGBoost model
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 verbose=False)

        # Make predictions on validation set
        y_pred = model.predict(X_val)

        # Calculate validation MSE
        mse = mean_squared_error(y_val, y_pred)

        return mse

    def optimize(self) -> dict:
        """Run XGBoost hyperparameter optimization."""
        _logger.info("Starting XGBoost hyperparameter optimization...")

        # Load data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self._load_combined_features()

        # Create Optuna study
        study_name = self.config['optuna']['xgboost_study_name']
        storage = self.config['optuna']['storage']
        n_trials = self.config['optuna']['n_trials']
        timeout = self.config['optuna']['timeout']

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction='minimize'
        )

        # Run optimization
        _logger.info("Running %s trials with %ss timeout...", n_trials, timeout)
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            timeout=timeout
        )

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        _logger.info("Best validation MSE: %s.6f", best_value)
        _logger.info("Best parameters: %s", best_params)

        # Save study results
        study_path = self.studies_dir / f"{study_name}.pkl"
        study.export_data(study_path)

        # Save best parameters
        best_params_path = self.configs_dir / "best_xgboost_params.json"
        import json
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=2)

        # Create optimization plots
        try:

            # Parameter importance plot
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(str(self.studies_dir / "xgboost_param_importance.html"))

            # Optimization history plot
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(str(self.studies_dir / "xgboost_optimization_history.html"))

            # Parameter relationships plot
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_html(str(self.studies_dir / "xgboost_parallel_coordinate.html"))

            _logger.info("Optimization plots saved")

        except Exception as e:
            _logger.warning("Failed to create optimization plots: %s", e)

        # Evaluate best model on test set
        best_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0, early_stopping_rounds=50)
        best_model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)

        y_test_pred = best_model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)

        # Calculate directional accuracy
        y_test_diff = np.diff(y_test)
        y_pred_diff = np.diff(y_test_pred)
        directional_accuracy = np.mean((y_test_diff > 0) == (y_pred_diff > 0))

        results = {
            'best_params': best_params,
            'best_validation_mse': best_value,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'directional_accuracy': directional_accuracy,
            'study_path': str(study_path),
            'best_params_path': str(best_params_path)
        }

        _logger.info("XGBoost optimization completed")
        _logger.info("Test MSE: %.6f", test_mse)
        _logger.info("Test MAE: %.6f", test_mae)
        _logger.info("Test RMSE: %.6f", test_rmse)
        _logger.info("Directional Accuracy: %.4f", directional_accuracy)

        return results

def main():
    """Main function to run XGBoost optimization."""
    import argparse

    parser = argparse.ArgumentParser(description="XGBoost Hyperparameter Optimization")
    parser.add_argument("--config", default="config/pipeline/x02.yaml",
                       help="Path to configuration file")

    args = parser.parse_args()

    try:
        optimizer = XGBoostOptimizer(args.config)
        results = optimizer.optimize()

        print("\n" + "="*50)
        print("XGBOOST OPTIMIZATION RESULTS")
        print("="*50)
        print(f"Best Validation MSE: {results['best_validation_mse']:.6f}")
        print(f"Test MSE: {results['test_mse']:.6f}")
        print(f"Test MAE: {results['test_mae']:.6f}")
        print(f"Test RMSE: {results['test_rmse']:.6f}")
        print(f"Directional Accuracy: {results['directional_accuracy']:.4f}")
        print(f"Best Parameters: {results['best_params']}")
        print("="*50)

    except Exception as e:
        print(f"Error during XGBoost optimization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
