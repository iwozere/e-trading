"""
Advanced ML Features Example

This example demonstrates the comprehensive ML capabilities:
- MLflow Integration (Model versioning, tracking, registry, deployment)
- Feature Engineering Pipeline (Automated feature extraction, selection)
- Automated Training Pipeline (Scheduled retraining, monitoring, A/B testing, drift detection)
"""

import pandas as pd
import numpy as np
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import ML modules
from src.ml.future.mlflow_integration import MLflowManager, ModelDeployer, ExperimentManager
from src.ml.future.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.ml.future.automated_training_pipeline import (
    AutomatedTrainingPipeline, TrainingConfig, ModelType,
    TrainingTrigger, PerformanceMetrics
)
from src.notification.logger import setup_logger

# Configure logging
logger = setup_logger(__name__)


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample trading data for demonstration."""
    np.random.seed(42)

    # Generate time series data
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')

    # Generate price data with some patterns
    base_price = 100
    trend = np.cumsum(np.random.randn(n_samples) * 0.01)
    seasonality = 5 * np.sin(np.arange(n_samples) * 2 * np.pi / 24)  # Daily seasonality
    noise = np.random.randn(n_samples) * 2

    close_prices = base_price + trend + seasonality + noise

    # Generate OHLCV data
    data = pd.DataFrame({
        'datetime': dates,
        'open': close_prices + np.random.randn(n_samples) * 0.5,
        'high': close_prices + np.abs(np.random.randn(n_samples) * 1.0),
        'low': close_prices - np.abs(np.random.randn(n_samples) * 1.0),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })

    # Ensure OHLC relationships
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))

    return data


def create_target_variable(data: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Create target variable for prediction (future returns)."""
    future_returns = data['close'].pct_change(horizon).shift(-horizon)
    return future_returns


def demonstrate_mlflow_integration():
    """Demonstrate MLflow integration features."""
    logger.info("=== MLflow Integration Demo ===")

    # Initialize MLflow manager
    mlflow_manager = MLflowManager(
        tracking_uri="sqlite:///mlflow_demo.db",
        registry_uri="sqlite:///mlflow_demo.db",
        experiment_name="crypto_trading_demo"
    )

    # Start a new run
    run_id = mlflow_manager.start_run("demo_run")

    # Log parameters
    params = {
        'model_type': 'xgboost',
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }
    mlflow_manager.log_parameters(params)

    # Simulate training and get metrics
    metrics = {
        'train_accuracy': 0.85,
        'val_accuracy': 0.82,
        'test_accuracy': 0.80,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.05
    }
    mlflow_manager.log_metrics(metrics)

    # Create sample model metadata
    from src.ml.future.mlflow_integration import ModelMetadata

    metadata = ModelMetadata(
        model_name="crypto_prediction_model",
        version="1.0.0",
        model_type="xgboost",
        framework_version="1.7.0",
        created_at=datetime.now(),
        author="Trading System",
        description="Crypto price prediction model",
        tags={"asset": "BTC", "timeframe": "1h"},
        hyperparameters=params,
        metrics=metrics,
        feature_names=["rsi_14", "macd", "bb_position_20"],
        target_column="future_return",
        data_version="2023.1.0",
        git_commit="abc123"
    )

    # Log model (simulated)
    logger.info("Logging model to MLflow...")
    # mlflow_manager.log_model(model, "crypto_model", "xgboost", metadata)

    # Register model
    logger.info("Registering model in MLflow registry...")
    # version = mlflow_manager.register_model("crypto_prediction_model", model_uri, "Staging")

    # List models
    models = mlflow_manager.list_models()
    logger.info("Registered models: %d", len(models))

    # End run
    mlflow_manager.end_run()

    logger.info("MLflow integration demo completed")


def demonstrate_feature_engineering():
    """Demonstrate feature engineering pipeline."""
    logger.info("=== Feature Engineering Pipeline Demo ===")

    # Create sample data
    data = create_sample_data(500)
    logger.info("Created sample data with %d rows", len(data))

    # Initialize feature engineering pipeline
    config = {
        "technical": {
            "enabled": True,
            "indicators": ["rsi", "macd", "bollinger_bands", "atr"]
        },
        "microstructure": {
            "enabled": True,
            "volume_analysis": True
        },
        "statistical": {
            "enabled": True,
            "rolling_stats": True
        },
        "selection": {
            "method": "mutual_info",
            "n_features": 20,
            "threshold": 0.01
        }
    }

    feature_pipeline = FeatureEngineeringPipeline(config)

    # Generate features
    logger.info("Generating features...")
    features_df = feature_pipeline.generate_features(data)

    logger.info("Original features: %d", len(data.columns))
    logger.info("Generated features: %d", len(features_df.columns))
    logger.info("New features: %d", len(features_df.columns) - len(data.columns))

    # Create target variable
    target = create_target_variable(data)

    # Remove rows with NaN values
    valid_indices = ~(features_df.isnull().any(axis=1) | target.isnull())
    features_df = features_df[valid_indices]
    target = target[valid_indices]

    # Feature selection
    logger.info("Performing feature selection...")
    selected_features = feature_pipeline.select_features(
        features_df, target, method="mutual_info", n_features=20
    )

    logger.info("Selected %d features", len(selected_features.columns))

    # Get feature importance
    importance = feature_pipeline.get_feature_importance()
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    logger.info("Top 10 features by importance:")
    for feature, score in top_features:
        logger.info("  %s: %.4f", feature, score)

    # Correlation analysis
    logger.info("Performing correlation analysis...")
    correlation_analysis = feature_pipeline.get_correlation_analysis(selected_features)

    high_corr_pairs = correlation_analysis.get('high_correlation_pairs', [])
    logger.info("Found %d highly correlated feature pairs", len(high_corr_pairs))

    # Feature scaling
    logger.info("Scaling features...")
    scaled_features = feature_pipeline.scale_features(selected_features, scaler_type="standard")

    logger.info("Feature engineering demo completed")
    return scaled_features, target


def demonstrate_automated_training():
    """Demonstrate automated training pipeline."""
    logger.info("=== Automated Training Pipeline Demo ===")

    # Create training configuration
    training_config = TrainingConfig(
        model_type=ModelType.XGBOOST,
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        },
        training_schedule="0 2 * * *",  # Daily at 2 AM
        performance_threshold=0.7,
        retrain_threshold=0.1,
        max_training_time=30,  # minutes
        validation_split=0.2,
        cross_validation_folds=5,
        feature_selection_method="mutual_info",
        n_features=20,
        scaler_type="standard"
    )

    # Pipeline configuration
    pipeline_config = {
        "mlflow_tracking_uri": "sqlite:///automated_training.db",
        "mlflow_registry_uri": "sqlite:///automated_training.db",
        "deployment": {
            "deployment_dir": "deployments",
            "backup_dir": "backups"
        },
        "monitoring": {
            "alert_threshold": 0.1,
            "degradation_threshold": 0.2,
            "max_history": 100
        },
        "drift_detection": {
            "drift_threshold": 0.05
        },
        "ab_testing": {
            "traffic_split": 0.5,
            "significance_level": 0.05
        },
        "deployment_threshold": 0.6
    }

    # Initialize automated training pipeline
    pipeline = AutomatedTrainingPipeline(pipeline_config)

    # Get training data
    data = create_sample_data(1000)
    target = create_target_variable(data)

    # Remove NaN values
    valid_indices = ~(data.isnull().any(axis=1) | target.isnull())
    data = data[valid_indices]
    target = target[valid_indices]

    # Trigger manual training
    logger.info("Triggering manual training...")
    pipeline.trigger_training(TrainingTrigger.MANUAL)

    # Check drift
    logger.info("Checking for data drift...")
    drift_results = pipeline.check_drift(data)

    if drift_results.get('overall_drift_detected'):
        logger.warning("Drift detected! Consider retraining model.")
        logger.info("Data drift: %s", drift_results.get('data_drift_detected'))
        logger.info("Concept drift: %s", drift_results.get('concept_drift_detected'))
    else:
        logger.info("No drift detected")

    # Get performance report
    logger.info("Generating performance report...")
    performance_report = pipeline.get_performance_report()

    logger.info("Performance report generated:")
    logger.info("  Performance trends: %d", len(performance_report.get('performance_trends', {})))
    logger.info("  Registered models: %d", len(performance_report.get('registered_models', [])))
    logger.info("  Is training: %s", performance_report.get('is_training', False))

    logger.info("Automated training demo completed")


def demonstrate_ab_testing():
    """Demonstrate A/B testing framework."""
    logger.info("=== A/B Testing Framework Demo ===")

    # Create sample data
    data = create_sample_data(1000)
    target = create_target_variable(data)

    # Remove NaN values
    valid_indices = ~(data.isnull().any(axis=1) | target.isnull())
    data = data[valid_indices]
    target = target[valid_indices]

    # Initialize A/B testing framework
    from src.ml.future.automated_training_pipeline import ABTestingFramework

    ab_config = {
        "traffic_split": 0.5,
        "significance_level": 0.05
    }

    ab_testing = ABTestingFramework(ab_config)

    # Create two different models for comparison
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    model_a = RandomForestRegressor(n_estimators=100, random_state=42)
    model_b = GradientBoostingRegressor(n_estimators=100, random_state=42)

    # Train models
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3, random_state=42
    )

    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    # Create A/B experiment
    experiment_id = ab_testing.create_experiment(
        "model_comparison_demo",
        model_a,
        model_b,
        ["mse", "r2", "sharpe_ratio"]
    )

    # Run experiment
    logger.info("Running A/B experiment...")
    results = ab_testing.run_experiment(experiment_id, X_test, y_test)

    logger.info("A/B Testing Results:")
    logger.info("  Experiment ID: %s", results['experiment_id'])
    logger.info("  Recommendation: %s", results['recommendation'])

    if 'significance' in results:
        for metric, sig_result in results['significance'].items():
            logger.info("  %s:", metric)
            logger.info("    Significant: %s", sig_result.get('significant', False))
            logger.info("    P-value: %.4f", sig_result.get('p_value', 0))
            logger.info("    Improvement: %.2%", sig_result.get('improvement', 0))

    logger.info("A/B testing demo completed")


def demonstrate_experiment_management():
    """Demonstrate experiment management."""
    logger.info("=== Experiment Management Demo ===")

    # Initialize MLflow manager and experiment manager
    mlflow_manager = MLflowManager(
        tracking_uri="sqlite:///experiments.db",
        experiment_name="experiment_demo"
    )

    experiment_manager = ExperimentManager(mlflow_manager)

    # Create multiple experiments
    experiments = [
        ("xgboost_optimization", "XGBoost hyperparameter optimization"),
        ("feature_selection", "Feature selection methods comparison"),
        ("ensemble_models", "Ensemble model performance comparison")
    ]

    for exp_name, exp_desc in experiments:
        experiment_id = experiment_manager.create_experiment(
            exp_name, exp_desc, {"type": "optimization"}
        )
        logger.info("Created experiment: %s (ID: %s)", exp_name, experiment_id)

    # Simulate multiple runs
    for i in range(5):
        run_id = mlflow_manager.start_run(f"run_{i+1}")

        # Log different parameters
        params = {
            'learning_rate': 0.01 + i * 0.05,
            'max_depth': 3 + i,
            'n_estimators': 50 + i * 25
        }
        mlflow_manager.log_parameters(params)

        # Log metrics
        metrics = {
            'test_accuracy': 0.75 + i * 0.02,
            'sharpe_ratio': 1.0 + i * 0.1,
            'max_drawdown': -0.05 - i * 0.01
        }
        mlflow_manager.log_metrics(metrics)

        mlflow_manager.end_run()

    # Compare runs
    logger.info("Comparing experiment runs...")
    comparison_df = experiment_manager.compare_runs(
        "xgboost_optimization", "test_accuracy", max_results=5
    )

    logger.info("Top 5 runs by test accuracy:")
    for _, run in comparison_df.iterrows():
        logger.info("  Run %s: %.4f", run['run_id'][:8], run['metric'])

    # Get best run
    best_run = experiment_manager.get_best_run("xgboost_optimization", "test_accuracy")

    if best_run:
        logger.info("Best run: %s", best_run['run_id'][:8])
        logger.info("Best accuracy: %.4f", best_run['metric_value'])

    logger.info("Experiment management demo completed")


def create_configuration_files():
    """Create configuration files for the ML system."""
    logger.info("=== Creating Configuration Files ===")

    # MLflow configuration
    mlflow_config = {
        "tracking_uri": "sqlite:///mlflow.db",
        "registry_uri": "sqlite:///mlflow.db",
        "experiment_name": "crypto_trading",
        "artifacts_dir": "mlruns"
    }

    with open("config/mlflow_config.yaml", "w") as f:
        yaml.dump(mlflow_config, f, default_flow_style=False)

    # Feature engineering configuration
    feature_config = {
        "technical": {
            "enabled": True,
            "trend_indicators": True,
            "momentum_indicators": True,
            "volatility_indicators": True,
            "volume_indicators": True,
            "oscillator_indicators": True,
            "pattern_indicators": True
        },
        "microstructure": {
            "enabled": True,
            "orderbook_features": True,
            "volume_profile": True,
            "price_impact": True,
            "liquidity_features": True
        },
        "statistical": {
            "enabled": True,
            "rolling_statistics": True,
            "cross_sectional": True,
            "time_series": True
        },
        "selection": {
            "method": "mutual_info",
            "n_features": 50,
            "threshold": 0.01,
            "correlation_threshold": 0.8
        },
        "scaling": {
            "method": "standard",
            "fit_on_train": True
        }
    }

    with open("config/feature_engineering_config.yaml", "w") as f:
        yaml.dump(feature_config, f, default_flow_style=False)

    # Training configuration
    training_config = {
        "model_type": "xgboost",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        "training_schedule": "0 2 * * *",  # Daily at 2 AM
        "performance_threshold": 0.7,
        "retrain_threshold": 0.1,
        "max_training_time": 30,
        "validation_split": 0.2,
        "cross_validation_folds": 5,
        "feature_selection": {
            "method": "mutual_info",
            "n_features": 50
        },
        "scaling": {
            "method": "standard"
        }
    }

    with open("config/training_config.yaml", "w") as f:
        yaml.dump(training_config, f, default_flow_style=False)

    # Monitoring configuration
    monitoring_config = {
        "performance_monitoring": {
            "alert_threshold": 0.1,
            "degradation_threshold": 0.2,
            "max_history": 100,
            "metrics": ["mse", "r2", "sharpe_ratio", "win_rate"]
        },
        "drift_detection": {
            "drift_threshold": 0.05,
            "check_frequency": "daily",
            "reference_window": 30
        },
        "ab_testing": {
            "traffic_split": 0.5,
            "significance_level": 0.05,
            "min_sample_size": 100
        },
        "deployment": {
            "deployment_threshold": 0.6,
            "deployment_type": "rolling",
            "backup_enabled": True
        }
    }

    with open("config/monitoring_config.yaml", "w") as f:
        yaml.dump(monitoring_config, f, default_flow_style=False)

    logger.info("Configuration files created in config/ directory")


def main():
    """Main function to run all demonstrations."""
    logger.info("Starting Advanced ML Features Demonstration")
    logger.info("=" * 50)

    try:
        # Create configuration files
        create_configuration_files()

        # Demonstrate MLflow integration
        demonstrate_mlflow_integration()

        # Demonstrate feature engineering
        features, target = demonstrate_feature_engineering()

        # Demonstrate automated training
        demonstrate_automated_training()

        # Demonstrate A/B testing
        demonstrate_ab_testing()

        # Demonstrate experiment management
        demonstrate_experiment_management()

        logger.info("=" * 50)
        logger.info("All demonstrations completed successfully!")

        # Summary
        logger.info("\nSummary of Advanced ML Features:")
        logger.info("✓ MLflow Integration - Model versioning, tracking, registry, deployment")
        logger.info("✓ Feature Engineering - Automated extraction, selection, validation")
        logger.info("✓ Automated Training - Scheduled retraining, monitoring, optimization")
        logger.info("✓ A/B Testing - Statistical comparison of models")
        logger.info("✓ Experiment Management - Organized experimentation")
        logger.info("✓ Drift Detection - Data and concept drift monitoring")
        logger.info("✓ Performance Monitoring - Real-time performance tracking")

    except Exception as e:
        logger.exception("Error in demonstration: ")
        raise


if __name__ == "__main__":
    main()