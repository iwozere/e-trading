"""
Automated Training Pipeline

This module provides comprehensive automated training capabilities:
- Scheduled model retraining
- Performance monitoring
- A/B testing framework
- Model drift detection
"""

import pandas as pd
import numpy as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime
import json
import schedule
import time
import threading
from dataclasses import asdict
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import optuna
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.model.machine_learning import TrainingConfig, ModelType, PerformanceMetrics, TrainingTrigger
from src.ml.future.mlflow_integration import MLflowManager, ModelDeployer
from src.ml.future.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """Handles model training and optimization."""

    def __init__(self, config: TrainingConfig, mlflow_manager: MLflowManager):
        self.config = config
        self.mlflow_manager = mlflow_manager
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.current_model = None
        self.best_params = {}

        # Model factory
        self.model_factory = {
            ModelType.RANDOM_FOREST: RandomForestRegressor,
            ModelType.GRADIENT_BOOSTING: GradientBoostingRegressor,
            ModelType.XGBOOST: xgb.XGBRegressor,
            ModelType.LIGHTGBM: lgb.LGBMRegressor,
            ModelType.LINEAR_REGRESSION: LinearRegression
        }

    def train_model(self,
                   X: pd.DataFrame,
                   y: pd.Series,
                   optimize_hyperparameters: bool = True) -> Any:
        """Train a model with optional hyperparameter optimization."""
        try:
            logger.info("Starting model training for %s", self.config.model_type.value)

            # Feature engineering
            X_features = self.feature_pipeline.generate_features(X)

            # Feature selection
            X_selected = self.feature_pipeline.select_features(
                X_features, y,
                self.config.feature_selection_method,
                self.config.n_features
            )

            # Feature scaling
            X_scaled = self.feature_pipeline.scale_features(
                X_selected, self.config.scaler_type
            )

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y,
                test_size=self.config.validation_split,
                random_state=42
            )

            # Optimize hyperparameters if requested
            if optimize_hyperparameters:
                best_params = self._optimize_hyperparameters(X_train, y_train)
                self.best_params = best_params
            else:
                best_params = self.config.hyperparameters

            # Train final model
            model_class = self.model_factory[self.config.model_type]
            model = model_class(**best_params)

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config.cross_validation_folds,
                scoring='neg_mean_squared_error'
            )

            # Train on full training set
            model.fit(X_train, y_train)

            # Evaluate on validation set
            y_pred = model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)

            # Store model
            self.current_model = model

            logger.info("Model training completed. CV MSE: %.4f", -cv_scores.mean())
            return model, metrics

        except Exception:
            logger.exception("Error in model training: ")
            raise

    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            if self.config.model_type == ModelType.RANDOM_FOREST:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42
                }
            elif self.config.model_type == ModelType.XGBOOST:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
            elif self.config.model_type == ModelType.LIGHTGBM:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
            else:
                params = self.config.hyperparameters

            model_class = self.model_factory[self.config.model_type]
            model = model_class(**params)

            scores = cross_val_score(
                model, X, y,
                cv=self.config.cross_validation_folds,
                scoring='neg_mean_squared_error'
            )

            return -scores.mean()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, timeout=self.config.max_training_time * 60)

        return study.best_params

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> PerformanceMetrics:
        """Calculate performance metrics."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Trading-specific metrics
        returns = y_true - y_true.shift(1)
        predicted_returns = y_pred - y_true.shift(1)

        sharpe_ratio = np.mean(predicted_returns) / np.std(predicted_returns) if np.std(predicted_returns) > 0 else 0

        # Calculate drawdown
        cumulative_returns = (1 + predicted_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (predicted_returns > 0).mean()

        # Profit factor
        gains = predicted_returns[predicted_returns > 0].sum()
        losses = abs(predicted_returns[predicted_returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')

        return PerformanceMetrics(
            mse=mse,
            mae=mae,
            r2=r2,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            timestamp=datetime.now()
        )


class PerformanceMonitor:
    """Monitors model performance and detects degradation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_history = []
        self.alert_threshold = config.get("alert_threshold", 0.1)
        self.degradation_threshold = config.get("degradation_threshold", 0.2)

    def update_performance(self, metrics: PerformanceMetrics):
        """Update performance history."""
        self.performance_history.append(metrics)

        # Keep only recent history
        max_history = self.config.get("max_history", 100)
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]

    def check_performance_degradation(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if performance has degraded."""
        if len(self.performance_history) < 10:
            return False, {}

        recent_metrics = self.performance_history[-10:]
        historical_metrics = self.performance_history[:-10]

        if not historical_metrics:
            return False, {}

        # Calculate performance changes
        changes = {}
        for metric_name in ['mse', 'mae', 'r2', 'sharpe_ratio', 'win_rate']:
            recent_avg = np.mean([getattr(m, metric_name) for m in recent_metrics])
            historical_avg = np.mean([getattr(m, metric_name) for m in historical_metrics])

            if historical_avg != 0:
                change = (recent_avg - historical_avg) / abs(historical_avg)
                changes[metric_name] = change

        # Check for degradation
        degradation_detected = False
        degradation_reasons = []

        for metric, change in changes.items():
            if metric in ['mse', 'mae'] and change > self.degradation_threshold:
                degradation_detected = True
                degradation_reasons.append(f"{metric} increased by {change:.2%}")
            elif metric in ['r2', 'sharpe_ratio', 'win_rate'] and change < -self.degradation_threshold:
                degradation_detected = True
                degradation_reasons.append(f"{metric} decreased by {abs(change):.2%}")

        return degradation_detected, {
            'changes': changes,
            'reasons': degradation_reasons,
            'recent_avg': {name: np.mean([getattr(m, name) for m in recent_metrics])
                          for name in ['mse', 'mae', 'r2', 'sharpe_ratio', 'win_rate']},
            'historical_avg': {name: np.mean([getattr(m, name) for m in historical_metrics])
                              for name in ['mse', 'mae', 'r2', 'sharpe_ratio', 'win_rate']}
        }

    def get_performance_trend(self, metric_name: str, window: int = 20) -> Dict[str, Any]:
        """Get performance trend for a specific metric."""
        if len(self.performance_history) < window:
            return {}

        recent_metrics = self.performance_history[-window:]
        values = [getattr(m, metric_name) for m in recent_metrics]

        # Calculate trend
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

        return {
            'values': values,
            'trend': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'mean': np.mean(values),
            'std': np.std(values)
        }


class DriftDetector:
    """Detects data and concept drift."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reference_distribution = None
        self.drift_threshold = config.get("drift_threshold", 0.05)

    def set_reference_distribution(self, data: pd.DataFrame):
        """Set reference distribution for drift detection."""
        self.reference_distribution = {
            'mean': data.mean(),
            'std': data.std(),
            'quantiles': data.quantile([0.25, 0.5, 0.75])
        }

    def detect_data_drift(self, current_data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Detect data distribution drift."""
        if self.reference_distribution is None:
            return False, {}

        drift_detected = False
        drift_details = {}

        for column in current_data.columns:
            if column in self.reference_distribution['mean']:
                # Kolmogorov-Smirnov test
                try:
                    ks_statistic, p_value = stats.ks_2samp(
                        current_data[column].dropna(),
                        np.random.normal(
                            self.reference_distribution['mean'][column],
                            self.reference_distribution['std'][column],
                            size=len(current_data)
                        )
                    )

                    if p_value < self.drift_threshold:
                        drift_detected = True
                        drift_details[column] = {
                            'ks_statistic': ks_statistic,
                            'p_value': p_value,
                            'drift_detected': True
                        }
                    else:
                        drift_details[column] = {
                            'ks_statistic': ks_statistic,
                            'p_value': p_value,
                            'drift_detected': False
                        }
                except:
                    drift_details[column] = {'error': 'Could not compute drift'}

        return drift_detected, drift_details

    def detect_concept_drift(self,
                           X: pd.DataFrame,
                           y: pd.Series,
                           model: Any) -> Tuple[bool, Dict[str, Any]]:
        """Detect concept drift using model performance."""
        try:
            # Split data into time periods
            split_point = len(X) // 2
            X_old = X.iloc[:split_point]
            y_old = y.iloc[:split_point]
            X_new = X.iloc[split_point:]
            y_new = y.iloc[split_point:]

            # Train model on old data
            model_old = type(model)(**model.get_params())
            model_old.fit(X_old, y_old)

            # Predict on new data
            y_pred_old = model_old.predict(X_new)
            y_pred_new = model.predict(X_new)

            # Compare prediction distributions
            ks_statistic, p_value = stats.ks_2samp(y_pred_old, y_pred_new)

            drift_detected = p_value < self.drift_threshold

            return drift_detected, {
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'old_performance': mean_squared_error(y_new, y_pred_old),
                'new_performance': mean_squared_error(y_new, y_pred_new)
            }

        except Exception:
            logger.exception("Error in concept drift detection: ")
            return False, {}


class ABTestingFramework:
    """A/B testing framework for model comparison."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiments = {}
        self.traffic_split = config.get("traffic_split", 0.5)
        self.significance_level = config.get("significance_level", 0.05)

    def create_experiment(self,
                         experiment_name: str,
                         model_a: Any,
                         model_b: Any,
                         metrics: List[str]) -> str:
        """Create a new A/B testing experiment."""
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.experiments[experiment_id] = {
            'name': experiment_name,
            'model_a': model_a,
            'model_b': model_b,
            'metrics': metrics,
            'results_a': [],
            'results_b': [],
            'start_time': datetime.now(),
            'status': 'running'
        }

        logger.info("Created A/B experiment: %s", experiment_id)
        return experiment_id

    def run_experiment(self,
                      experiment_id: str,
                      X: pd.DataFrame,
                      y: pd.Series) -> Dict[str, Any]:
        """Run A/B testing experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]

        # Split data
        split_point = int(len(X) * self.traffic_split)
        X_a = X.iloc[:split_point]
        y_a = y.iloc[:split_point]
        X_b = X.iloc[split_point:]
        y_b = y.iloc[split_point:]

        # Get predictions
        y_pred_a = experiment['model_a'].predict(X_a)
        y_pred_b = experiment['model_b'].predict(X_b)

        # Calculate metrics
        metrics_a = self._calculate_experiment_metrics(y_a, y_pred_a)
        metrics_b = self._calculate_experiment_metrics(y_b, y_pred_b)

        # Store results
        experiment['results_a'].append(metrics_a)
        experiment['results_b'].append(metrics_b)

        # Statistical significance test
        significance_results = self._test_significance(experiment)

        return {
            'experiment_id': experiment_id,
            'metrics_a': metrics_a,
            'metrics_b': metrics_b,
            'significance': significance_results,
            'recommendation': self._get_recommendation(significance_results)
        }

    def _calculate_experiment_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for experiment."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'sharpe_ratio': np.mean(y_pred) / np.std(y_pred) if np.std(y_pred) > 0 else 0
        }

    def _test_significance(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Test statistical significance of differences."""
        if len(experiment['results_a']) < 2 or len(experiment['results_b']) < 2:
            return {}

        significance_results = {}

        for metric in experiment['metrics']:
            values_a = [r[metric] for r in experiment['results_a']]
            values_b = [r[metric] for r in experiment['results_b']]

            # T-test
            t_stat, p_value = stats.ttest_ind(values_a, values_b)

            significance_results[metric] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.significance_level,
                'mean_a': np.mean(values_a),
                'mean_b': np.mean(values_b),
                'improvement': (np.mean(values_b) - np.mean(values_a)) / abs(np.mean(values_a)) if np.mean(values_a) != 0 else 0
            }

        return significance_results

    def _get_recommendation(self, significance_results: Dict[str, Any]) -> str:
        """Get recommendation based on significance results."""
        if not significance_results:
            return "Insufficient data"

        significant_metrics = [metric for metric, result in significance_results.items()
                             if result.get('significant', False)]

        if not significant_metrics:
            return "No significant difference detected"

        # Count improvements
        improvements = 0
        for metric in significant_metrics:
            if significance_results[metric]['improvement'] > 0:
                improvements += 1

        if improvements > len(significant_metrics) / 2:
            return "Model B is significantly better"
        else:
            return "Model A is significantly better"


class AutomatedTrainingPipeline:
    """Main automated training pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize components
        self.mlflow_manager = MLflowManager(
            tracking_uri=config.get("mlflow_tracking_uri", "sqlite:///mlflow.db"),
            registry_uri=config.get("mlflow_registry_uri", "sqlite:///mlflow.db")
        )

        self.model_deployer = ModelDeployer(config.get("deployment", {}))
        self.trainer = ModelTrainer(config.get("training"), self.mlflow_manager)
        self.performance_monitor = PerformanceMonitor(config.get("monitoring", {}))
        self.drift_detector = DriftDetector(config.get("drift_detection", {}))
        self.ab_testing = ABTestingFramework(config.get("ab_testing", {}))

        # Training state
        self.is_training = False
        self.last_training_time = None
        self.training_scheduler = None

    def start_scheduled_training(self):
        """Start scheduled training."""
        training_schedule = self.config.get("training_schedule", "0 2 * * *")  # Daily at 2 AM

        schedule.every().day.at("02:00").do(self._scheduled_training_job)

        logger.info("Started scheduled training")

        # Run scheduler in separate thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)

        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()

    def _scheduled_training_job(self):
        """Scheduled training job."""
        if self.is_training:
            logger.warning("Training already in progress, skipping scheduled job")
            return

        try:
            logger.info("Starting scheduled training job")
            self.is_training = True

            # Get latest data
            data = self._get_training_data()

            # Train model
            model, metrics = self.trainer.train_model(data['X'], data['y'])

            # Log to MLflow
            run_id = self.mlflow_manager.start_run("scheduled_training")
            self.mlflow_manager.log_parameters(self.trainer.best_params)
            self.mlflow_manager.log_metrics(asdict(metrics))

            # Register model
            model_uri = f"runs:/{run_id}/model"
            version = self.mlflow_manager.register_model(
                "crypto_prediction_model",
                model_uri,
                "Staging"
            )

            self.mlflow_manager.end_run()

            # Deploy if performance is good
            if metrics.r2 > self.config.get("deployment_threshold", 0.6):
                self.model_deployer.deploy_model(
                    "crypto_prediction_model",
                    version,
                    self.mlflow_manager
                )

            self.last_training_time = datetime.now()
            logger.info("Scheduled training completed successfully")

        except Exception:
            logger.exception("Error in scheduled training: ")
        finally:
            self.is_training = False

    def _get_training_data(self) -> Dict[str, Any]:
        """Get training data (placeholder - implement based on your data source)."""
        # This should be implemented based on your data source
        # For now, return dummy data
        n_samples = 1000
        n_features = 50

        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = pd.Series(np.random.randn(n_samples))

        return {'X': X, 'y': y}

    def trigger_training(self, trigger: TrainingTrigger, **kwargs):
        """Trigger model training."""
        if self.is_training:
            logger.warning("Training already in progress")
            return

        try:
            logger.info("Triggering training due to %s", trigger.value)
            self.is_training = True

            # Get data
            data = self._get_training_data()

            # Train model
            model, metrics = self.trainer.train_model(data['X'], data['y'])

            # Update performance monitor
            self.performance_monitor.update_performance(metrics)

            # Check for performance degradation
            degradation_detected, details = self.performance_monitor.check_performance_degradation()

            if degradation_detected:
                logger.warning("Performance degradation detected: %s", details)

            self.last_training_time = datetime.now()
            logger.info("Training completed successfully")

        except Exception:
            logger.exception("Error in training: ")
        finally:
            self.is_training = False

    def run_ab_test(self,
                   model_a: Any,
                   model_b: Any,
                   experiment_name: str = "model_comparison") -> Dict[str, Any]:
        """Run A/B testing between two models."""
        try:
            # Create experiment
            experiment_id = self.ab_testing.create_experiment(
                experiment_name, model_a, model_b, ['mse', 'r2', 'sharpe_ratio']
            )

            # Get test data
            data = self._get_training_data()

            # Run experiment
            results = self.ab_testing.run_experiment(
                experiment_id, data['X'], data['y']
            )

            logger.info("A/B test completed: %s", results['recommendation'])
            return results

        except Exception:
            logger.exception("Error in A/B testing: ")
            return {}

    def check_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for data and concept drift."""
        try:
            # Data drift detection
            data_drift_detected, data_drift_details = self.drift_detector.detect_data_drift(current_data)

            # Concept drift detection (if model is available)
            concept_drift_detected = False
            concept_drift_details = {}

            if self.trainer.current_model is not None:
                # Get target data (placeholder)
                y = pd.Series(np.random.randn(len(current_data)))

                concept_drift_detected, concept_drift_details = self.drift_detector.detect_concept_drift(
                    current_data, y, self.trainer.current_model
                )

            return {
                'data_drift_detected': data_drift_detected,
                'data_drift_details': data_drift_details,
                'concept_drift_detected': concept_drift_detected,
                'concept_drift_details': concept_drift_details,
                'overall_drift_detected': data_drift_detected or concept_drift_detected
            }

        except Exception:
            logger.exception("Error in drift detection: ")
            return {}

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        try:
            # Performance trends
            trends = {}
            for metric in ['mse', 'r2', 'sharpe_ratio']:
                trends[metric] = self.performance_monitor.get_performance_trend(metric)

            # Drift status
            data = self._get_training_data()
            drift_status = self.check_drift(data['X'])

            # Model registry status
            models = self.mlflow_manager.list_models()

            return {
                'performance_trends': trends,
                'drift_status': drift_status,
                'registered_models': models,
                'last_training_time': self.last_training_time,
                'is_training': self.is_training
            }

        except Exception:
            logger.exception("Error generating performance report: ")
            return {}

    def save_pipeline_state(self, filepath: str):
        """Save pipeline state."""
        state = {
            'last_training_time': self.last_training_time,
            'best_params': self.trainer.best_params,
            'performance_history': self.performance_monitor.performance_history,
            'reference_distribution': self.drift_detector.reference_distribution
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, default=str, indent=2)

        logger.info("Saved pipeline state to %s", filepath)

    def load_pipeline_state(self, filepath: str):
        """Load pipeline state."""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.last_training_time = datetime.fromisoformat(state['last_training_time']) if state['last_training_time'] else None
        self.trainer.best_params = state['best_params']

        # Load performance history
        self.performance_monitor.performance_history = []
        for metric_data in state['performance_history']:
            metric = PerformanceMetrics(**metric_data)
            metric.timestamp = datetime.fromisoformat(metric_data['timestamp'])
            self.performance_monitor.performance_history.append(metric)

        # Load reference distribution
        self.drift_detector.reference_distribution = state['reference_distribution']

        logger.info("Loaded pipeline state from %s", filepath)
