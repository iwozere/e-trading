# Advanced ML Features Documentation

This document provides comprehensive documentation for the advanced machine learning features implemented in the crypto trading system.

## Table of Contents

1. [Overview](#overview)
2. [MLflow Integration](#mlflow-integration)
3. [Feature Engineering Pipeline](#feature-engineering-pipeline)
4. [Automated Training Pipeline](#automated-training-pipeline)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Neural Network Regime Detection (PyTorch)](#neural-network-regime-detection-pytorch)
10. [HMM-Based Regime Detection](#hmm-based-regime-detection)
11. [Hybrid Deep Learning + XGBoost Pipeline (Future Improvements)](#hybrid-deep-learning-xgb-pipeline-future-improvements)

## Overview

The advanced ML features provide a complete machine learning lifecycle management system for crypto trading strategies:

- **MLflow Integration**: Model versioning, tracking, registry, and deployment
- **Feature Engineering Pipeline**: Automated feature extraction, selection, and validation
- **Automated Training Pipeline**: Scheduled retraining, performance monitoring, A/B testing, and drift detection

## MLflow Integration

### Features

#### Model Versioning and Tracking
- **Experiment Management**: Organize ML experiments with proper naming and tagging
- **Run Tracking**: Track all model versions, parameters, and performance metrics
- **Reproducibility**: Enable reproducible experiments with exact environment snapshots
- **Lineage Tracking**: Track data to model to predictions lineage

#### Model Registry
- **Centralized Storage**: Store production-ready models in a centralized registry
- **Lifecycle Management**: Manage model stages (staging, production, archived)
- **Automated Promotion**: Promote models based on performance criteria
- **Version Control**: Track model versions and changes

#### Automated Model Deployment
- **Seamless Deployment**: Deploy new models to production seamlessly
- **Rolling Updates**: Perform rolling updates with zero downtime
- **Automatic Rollback**: Rollback on performance degradation
- **Blue-Green Deployment**: Support for blue-green deployment strategies

### Key Classes

#### MLflowManager
Main class for MLflow integration:

```python
from src.ml.mlflow_integration import MLflowManager

# Initialize
mlflow_manager = MLflowManager(
    tracking_uri="sqlite:///mlflow.db",
    registry_uri="sqlite:///mlflow.db",
    experiment_name="crypto_trading"
)

# Start run
run_id = mlflow_manager.start_run("experiment_name")

# Log parameters and metrics
mlflow_manager.log_parameters({"learning_rate": 0.1})
mlflow_manager.log_metrics({"accuracy": 0.85})

# Register model
version = mlflow_manager.register_model("model_name", model_uri, "Production")

# Load model
model = mlflow_manager.load_model("model_name", "Production")
```

#### ModelDeployer
Handles automated model deployment:

```python
from src.ml.mlflow_integration import ModelDeployer

deployer = ModelDeployer(deployment_config)

# Deploy model
success = deployer.deploy_model(
    model_name="crypto_model",
    model_version=1,
    mlflow_manager=mlflow_manager,
    deployment_type="rolling"
)
```

#### ExperimentManager
Manages ML experiments:

```python
from src.ml.mlflow_integration import ExperimentManager

exp_manager = ExperimentManager(mlflow_manager)

# Create experiment
experiment_id = exp_manager.create_experiment("optimization_study")

# Compare runs
comparison_df = exp_manager.compare_runs("experiment_name", "accuracy")

# Get best run
best_run = exp_manager.get_best_run("experiment_name", "accuracy")
```

## Feature Engineering Pipeline

### Features

#### Automated Feature Extraction
- **Technical Indicators**: Comprehensive set of technical analysis indicators
- **Market Microstructure**: Order book, volume profile, and liquidity features
- **Statistical Features**: Rolling statistics, cross-sectional features, time series features
- **Multi-timeframe**: Features from different timeframes
- **Batch and Real-time**: Support for both batch and real-time feature computation

#### Technical Indicator Features
- **Trend Indicators**: Moving averages, MACD, Parabolic SAR, ADX, Ichimoku
- **Momentum Indicators**: RSI, Stochastic, Williams %R, CCI, ROC, Momentum
- **Volatility Indicators**: Bollinger Bands, ATR, Keltner Channel, Historical Volatility
- **Volume Indicators**: OBV, Volume SMA, Chaikin Money Flow, MFI, AD, VPT
- **Oscillator Indicators**: Stochastic RSI, Ultimate Oscillator, TRIX, AROON
- **Pattern Recognition**: 50+ candlestick patterns

#### Market Microstructure Features
- **Order Book Features**: Order imbalance, depth, pressure
- **Volume Profile**: Volume-weighted price levels, concentration, trends
- **Price Impact**: Kyle's lambda, Amihud illiquidity
- **Liquidity Features**: Roll's spread estimator, effective spread, quoted spread

#### Feature Selection and Validation
- **Multiple Methods**: Mutual information, F-regression, correlation, PCA
- **Automated Selection**: Select best features based on importance scores
- **Correlation Analysis**: Detect multicollinearity and highly correlated features
- **Feature Stability**: Test feature stability across time periods
- **Validation**: Cross-validation and holdout validation

### Key Classes

#### FeatureEngineeringPipeline
Main pipeline for feature engineering:

```python
from src.ml.feature_engineering_pipeline import FeatureEngineeringPipeline

# Initialize
config = {
    "technical": {"enabled": True},
    "microstructure": {"enabled": True},
    "statistical": {"enabled": True},
    "selection": {"method": "mutual_info", "n_features": 50}
}

pipeline = FeatureEngineeringPipeline(config)

# Generate features
features_df = pipeline.generate_features(data)

# Select features
selected_features = pipeline.select_features(
    features_df, target, method="mutual_info", n_features=20
)

# Scale features
scaled_features = pipeline.scale_features(selected_features, "standard")

# Get feature importance
importance = pipeline.get_feature_importance()
```

#### TechnicalIndicatorFeatures
Generates technical indicator features:

```python
from src.ml.feature_engineering_pipeline import TechnicalIndicatorFeatures

tech_features = TechnicalIndicatorFeatures()
features_df = tech_features.generate_all_features(data)
```

#### MarketMicrostructureFeatures
Generates market microstructure features:

```python
from src.ml.feature_engineering_pipeline import MarketMicrostructureFeatures

micro_features = MarketMicrostructureFeatures()
features_df = micro_features.generate_features(data, orderbook_data)
```

#### FeatureSelector
Handles feature selection:

```python
from src.ml.feature_engineering_pipeline import FeatureSelector

selector = FeatureSelector()

# Select features
selected_X = selector.select_features(X, y, method="mutual_info", n_features=20)

# Analyze correlations
correlation_analysis = selector.analyze_correlations(X)

# Check feature stability
stability_scores = selector.get_feature_stability(X_train, X_test)
```

## Automated Training Pipeline

### Features

#### Scheduled Model Retraining
- **Time-based Scheduling**: Retrain models on schedule (daily, weekly, etc.)
- **Performance-based Triggering**: Retrain when performance degrades
- **Drift-based Triggering**: Retrain when data or concept drift is detected
- **Resource Optimization**: Optimize training resources and time

#### Performance Monitoring
- **Real-time Tracking**: Monitor model performance in real-time
- **Alert System**: Alert on performance degradation
- **Automated Reporting**: Generate performance reports automatically
- **Trend Analysis**: Analyze performance trends over time

#### A/B Testing Framework
- **Controlled Rollout**: Controlled rollout of new models
- **Statistical Testing**: Statistical significance testing
- **Performance Comparison**: Compare performance between model versions
- **Traffic Splitting**: Split traffic between model versions

#### Model Drift Detection
- **Data Drift**: Detect changes in data distribution
- **Concept Drift**: Detect changes in relationship between features and target
- **Automated Triggers**: Automatically trigger retraining on drift detection
- **Drift Analysis**: Analyze drift patterns and causes

### Key Classes

#### AutomatedTrainingPipeline
Main pipeline for automated training:

```python
from src.ml.automated_training_pipeline import (
    AutomatedTrainingPipeline, TrainingConfig, ModelType
)

# Configuration
config = {
    "mlflow_tracking_uri": "sqlite:///training.db",
    "deployment": {"deployment_dir": "deployments"},
    "monitoring": {"alert_threshold": 0.1},
    "drift_detection": {"drift_threshold": 0.05}
}

# Initialize pipeline
pipeline = AutomatedTrainingPipeline(config)

# Start scheduled training
pipeline.start_scheduled_training()

# Trigger manual training
pipeline.trigger_training(TrainingTrigger.MANUAL)

# Check drift
drift_results = pipeline.check_drift(current_data)

# Get performance report
report = pipeline.get_performance_report()
```

#### ModelTrainer
Handles model training and optimization:

```python
from src.ml.automated_training_pipeline import ModelTrainer, TrainingConfig

# Training configuration
training_config = TrainingConfig(
    model_type=ModelType.XGBOOST,
    hyperparameters={"n_estimators": 100, "max_depth": 6},
    training_schedule="0 2 * * *",
    performance_threshold=0.7,
    validation_split=0.2
)

# Initialize trainer
trainer = ModelTrainer(training_config, mlflow_manager)

# Train model
model, metrics = trainer.train_model(X, y, optimize_hyperparameters=True)
```

#### PerformanceMonitor
Monitors model performance:

```python
from src.ml.automated_training_pipeline import PerformanceMonitor

monitor = PerformanceMonitor({"alert_threshold": 0.1})

# Update performance
monitor.update_performance(metrics)

# Check degradation
degradation_detected, details = monitor.check_performance_degradation()

# Get trends
trends = monitor.get_performance_trend("accuracy", window=20)
```

#### DriftDetector
Detects data and concept drift:

```python
from src.ml.automated_training_pipeline import DriftDetector

detector = DriftDetector({"drift_threshold": 0.05})

# Set reference distribution
detector.set_reference_distribution(reference_data)

# Detect data drift
data_drift, details = detector.detect_data_drift(current_data)

# Detect concept drift
concept_drift, details = detector.detect_concept_drift(X, y, model)
```

#### ABTestingFramework
Handles A/B testing:

```python
from src.ml.automated_training_pipeline import ABTestingFramework

ab_testing = ABTestingFramework({"traffic_split": 0.5})

# Create experiment
experiment_id = ab_testing.create_experiment(
    "model_comparison", model_a, model_b, ["accuracy", "sharpe_ratio"]
)

# Run experiment
results = ab_testing.run_experiment(experiment_id, X_test, y_test)
```

## Configuration

### MLflow Configuration

```yaml
# config/mlflow_config.yaml
tracking_uri: "sqlite:///mlflow.db"
registry_uri: "sqlite:///mlflow.db"
experiment_name: "crypto_trading"
artifacts_dir: "mlruns"
```

### Feature Engineering Configuration

```yaml
# config/feature_engineering_config.yaml
technical:
  enabled: true
  trend_indicators: true
  momentum_indicators: true
  volatility_indicators: true
  volume_indicators: true
  oscillator_indicators: true
  pattern_indicators: true

microstructure:
  enabled: true
  orderbook_features: true
  volume_profile: true
  price_impact: true
  liquidity_features: true

statistical:
  enabled: true
  rolling_statistics: true
  cross_sectional: true
  time_series: true

selection:
  method: "mutual_info"
  n_features: 50
  threshold: 0.01
  correlation_threshold: 0.8

scaling:
  method: "standard"
  fit_on_train: true
```

### Training Configuration

```yaml
# config/training_config.yaml
model_type: "xgboost"
hyperparameters:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42

training_schedule: "0 2 * * *"  # Daily at 2 AM
performance_threshold: 0.7
retrain_threshold: 0.1
max_training_time: 30
validation_split: 0.2
cross_validation_folds: 5

feature_selection:
  method: "mutual_info"
  n_features: 50

scaling:
  method: "standard"
```

### Monitoring Configuration

```yaml
# config/monitoring_config.yaml
performance_monitoring:
  alert_threshold: 0.1
  degradation_threshold: 0.2
  max_history: 100
  metrics: ["mse", "r2", "sharpe_ratio", "win_rate"]

drift_detection:
  drift_threshold: 0.05
  check_frequency: "daily"
  reference_window: 30

ab_testing:
  traffic_split: 0.5
  significance_level: 0.05
  min_sample_size: 100

deployment:
  deployment_threshold: 0.6
  deployment_type: "rolling"
  backup_enabled: true
```

## Usage Examples

### Complete ML Workflow

```python
import pandas as pd
from src.ml.mlflow_integration import MLflowManager
from src.ml.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.ml.automated_training_pipeline import AutomatedTrainingPipeline

# 1. Initialize MLflow
mlflow_manager = MLflowManager()

# 2. Initialize feature engineering
feature_pipeline = FeatureEngineeringPipeline()

# 3. Initialize training pipeline
training_pipeline = AutomatedTrainingPipeline({
    "mlflow_tracking_uri": "sqlite:///mlflow.db"
})

# 4. Load data
data = pd.read_csv("data/crypto_data.csv")

# 5. Generate features
features_df = feature_pipeline.generate_features(data)
target = data['future_return']

# 6. Select features
selected_features = feature_pipeline.select_features(
    features_df, target, method="mutual_info", n_features=20
)

# 7. Scale features
scaled_features = feature_pipeline.scale_features(selected_features)

# 8. Train model
model, metrics = training_pipeline.trainer.train_model(
    scaled_features, target, optimize_hyperparameters=True
)

# 9. Log to MLflow
run_id = mlflow_manager.start_run("model_training")
mlflow_manager.log_parameters(training_pipeline.trainer.best_params)
mlflow_manager.log_metrics(metrics.__dict__)
mlflow_manager.end_run()

# 10. Deploy model
training_pipeline.model_deployer.deploy_model(
    "crypto_model", 1, mlflow_manager
)

# 11. Start monitoring
training_pipeline.start_scheduled_training()
```

### Feature Engineering Example

```python
from src.ml.feature_engineering_pipeline import FeatureEngineeringPipeline

# Initialize pipeline
config = {
    "technical": {"enabled": True},
    "microstructure": {"enabled": True},
    "statistical": {"enabled": True},
    "selection": {"method": "mutual_info", "n_features": 30}
}

pipeline = FeatureEngineeringPipeline(config)

# Generate features
features_df = pipeline.generate_features(data)

# Get feature importance
importance = pipeline.get_feature_importance()
print("Top 10 features:")
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {feature}: {score:.4f}")

# Analyze correlations
correlation_analysis = pipeline.get_correlation_analysis(features_df)
high_corr_pairs = correlation_analysis['high_correlation_pairs']
print(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
```

### A/B Testing Example

```python
from src.ml.automated_training_pipeline import ABTestingFramework
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Create models
model_a = RandomForestRegressor(n_estimators=100, random_state=42)
model_b = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train models
model_a.fit(X_train, y_train)
model_b.fit(X_train, y_train)

# Initialize A/B testing
ab_testing = ABTestingFramework({"traffic_split": 0.5})

# Create experiment
experiment_id = ab_testing.create_experiment(
    "model_comparison", model_a, model_b, ["mse", "r2", "sharpe_ratio"]
)

# Run experiment
results = ab_testing.run_experiment(experiment_id, X_test, y_test)

print(f"Recommendation: {results['recommendation']}")
for metric, sig_result in results['significance'].items():
    print(f"{metric}: Significant={sig_result['significant']}, P-value={sig_result['p_value']:.4f}")
```

## Best Practices

### MLflow Best Practices

1. **Consistent Naming**: Use consistent naming conventions for experiments and runs
2. **Comprehensive Logging**: Log all parameters, metrics, and artifacts
3. **Model Registry**: Use model registry for production model management
4. **Version Control**: Track model versions and changes
5. **Automated Deployment**: Use automated deployment with rollback capabilities

### Feature Engineering Best Practices

1. **Feature Validation**: Validate features for stability and relevance
2. **Correlation Analysis**: Remove highly correlated features
3. **Feature Selection**: Use multiple selection methods and compare results
4. **Scaling**: Apply appropriate scaling methods
5. **Feature Monitoring**: Monitor feature distributions over time

### Training Pipeline Best Practices

1. **Scheduled Retraining**: Set up regular retraining schedules
2. **Performance Monitoring**: Monitor model performance continuously
3. **Drift Detection**: Implement drift detection and alerting
4. **A/B Testing**: Use A/B testing for model comparison
5. **Resource Management**: Optimize training resources and time

### General Best Practices

1. **Configuration Management**: Use configuration files for all settings
2. **Logging**: Implement comprehensive logging
3. **Error Handling**: Handle errors gracefully
4. **Testing**: Test all components thoroughly
5. **Documentation**: Document all processes and decisions

## Troubleshooting

### Common Issues

#### MLflow Issues

**Problem**: MLflow connection errors
**Solution**: Check tracking URI and network connectivity

**Problem**: Model registration failures
**Solution**: Verify model format and registry permissions

**Problem**: Experiment not found
**Solution**: Check experiment name and create if necessary

#### Feature Engineering Issues

**Problem**: Missing TA-Lib indicators
**Solution**: Install TA-Lib and verify installation

**Problem**: Feature selection errors
**Solution**: Check data quality and remove NaN values

**Problem**: Memory issues with large datasets
**Solution**: Use batch processing or reduce feature set

#### Training Pipeline Issues

**Problem**: Training timeouts
**Solution**: Increase max_training_time or optimize hyperparameters

**Problem**: Performance degradation alerts
**Solution**: Check data quality and consider retraining

**Problem**: Drift detection false positives
**Solution**: Adjust drift thresholds and reference windows

### Debugging Tips

1. **Enable Debug Logging**: Set logging level to DEBUG for detailed information
2. **Check Data Quality**: Verify data quality before processing
3. **Monitor Resources**: Monitor CPU, memory, and disk usage
4. **Validate Results**: Cross-validate results with multiple methods
5. **Test Incrementally**: Test components individually before integration

### Performance Optimization

1. **Feature Caching**: Cache computed features for reuse
2. **Parallel Processing**: Use parallel processing for feature generation
3. **Memory Management**: Optimize memory usage for large datasets
4. **Model Optimization**: Use hyperparameter optimization
5. **Resource Scaling**: Scale resources based on workload

## Neural Network Regime Detection (PyTorch)

A neural network can be used to detect market regimes (e.g., bull, bear, sideways) from features such as returns, volatility, RSI, MACD, etc. This approach can capture non-linear and temporal dependencies in the data.

### Overview
- Uses an LSTM-based classifier implemented in PyTorch (`src/ml/nn_regime_detector.py`)
- Supports GPU acceleration if available
- Can be trained on any set of features and regime labels
- Provides `fit`, `predict`, `save`, and `load` methods

### Example Usage
```python
from src.ml.nn_regime_detector import NNRegimeDetector
import pandas as pd

df = pd.read_csv('your_feature_data.csv')
features = df[['return', 'volatility', 'rsi', 'macd']].values
labels = df['regime'].values  # 0=bull, 1=bear, 2=sideways

model = NNRegimeDetector(input_size=4, n_regimes=3)
model.fit(features, labels, epochs=10)
preds = model.predict(features)
model.save('regime_model.pt')
model.load('regime_model.pt')
```

### Example Integration with Trading Logic
```python
# After predicting regimes, you can use them in your trading logic:
df['predicted_regime'] = preds
df['signal'] = np.where(df['predicted_regime'] == 0, 'buy',
                 np.where(df['predicted_regime'] == 1, 'sell', 'hold'))
```

### Full Example
See `examples/nn_regime_detection_example.py` for a complete script that:
- Generates synthetic data with regime labels
- Trains the neural network regime detector
- Predicts regimes
- Shows how to use regime predictions in trading logic
- Plots the detected regimes

### Requirements
- `torch` (PyTorch)
- `numpy`, `pandas`, `matplotlib`
- GPU support is automatic if available

## HMM-Based Regime Detection

A Hidden Markov Model (HMM) is a statistical model that can infer hidden market regimes (e.g., bull, bear, sideways) from observable features such as returns or volatility. HMMs are unsupervised and can automatically cluster periods with similar statistical properties.

### Overview
- Uses a Gaussian HMM to model regime switching in time series
- Provided as `src/ml/hmm_regime_detector.py`
- Can be used with any 1D or 2D feature set (e.g., returns, volatility, technical indicators)
- Provides `fit`, `predict`, and `plot_regimes` methods

### Example Usage
```python
from src.ml.hmm_regime_detector import HMMRegimeDetector
import pandas as pd

df = pd.read_csv('your_price_data.csv', parse_dates=['datetime'])
detector = HMMRegimeDetector(n_regimes=2)
detector.fit(df['close'])
df['regime'] = detector.predict(df['close'])
detector.plot_regimes(df['datetime'], df['close'], df['regime'])
```

### Example Integration with Trading Logic
```python
# After predicting regimes, you can use them in your trading logic:
df['signal'] = np.where(df['regime'] == 0, 'buy',
                 np.where(df['regime'] == 1, 'sell', 'hold'))
```

### Features
- Unsupervised regime detection (no need for labeled data)
- Can use more than two regimes or multiple features
- Visualizes detected regimes on price series

### Requirements
- `hmmlearn`
- `numpy`, `pandas`, `matplotlib`

See the source code and docstring in `src/ml/hmm_regime_detector.py` for more details and advanced usage.

## Hybrid Deep Learning + XGBoost Pipeline (Future Improvements)

A powerful hybrid pipeline can be built by combining deep learning and gradient boosting models for financial time series prediction. This approach leverages the strengths of CNNs for local pattern detection, LSTMs (with Bahdanau attention) for long-term dependencies, and XGBoost for robust tabular modeling.

### Pipeline Overview
- **Input:** OHLCV data (5 features) over a 100-timestep window
- **Technical Indicators:** RSI, ATR, MACD, Bollinger Bands, Ichimoku, OBV, Stochastic (computed on the same window)
- **CNN:** 1D convolutions + max-pooling for local feature extraction
- **LSTM:** 2+ layers, captures sequence dependencies
- **Bahdanau Attention:** Focuses on relevant LSTM outputs
- **XGBoost:** Takes LSTM+attention output concatenated with technical indicators for up/down classification

### Workflow
1. **Train CNN+LSTM+Attention** (`examples/hybrid_cnn_lstm_train.py`)
   - Input: OHLCV windows
   - Output: LSTM+attention features
   - Hyperparameters optimized with Optuna (TPE/Bayesian)
2. **Train XGBoost** (`examples/hybrid_xgboost_train.py`)
   - Input: Concatenation of LSTM+attention features and technical indicators
   - Output: Up/down classification
   - Hyperparameters optimized with Hyperopt

### Example Model Structure
```python
# CNN+LSTM+Attention (PyTorch)
class HybridModel(nn.Module):
    ... # see examples/hybrid_cnn_lstm_train.py

# XGBoost (scikit-learn API)
from xgboost import XGBClassifier
clf = XGBClassifier(...)
```

### Hyperparameters Optimized
- **CNN:** kernel size, number of layers, number of filters, learning rate
- **LSTM:** timesteps, number of neurons, number of layers, dropout
- **XGBoost:** learning_rate, max_depth, subsample

### Future Improvements
- Implement real-time model renormalization
- Add risk management (stop-loss, position sizing)

### References
- See `examples/hybrid_cnn_lstm_train.py` and `examples/hybrid_xgboost_train.py` for full scripts and details.

## Conclusion

The advanced ML features provide a comprehensive machine learning ecosystem for crypto trading. The system supports the complete ML lifecycle from feature engineering to model deployment and monitoring. By following the best practices and using the provided examples, you can build robust and scalable ML systems for crypto trading strategies.

For more information, refer to the individual module documentation and examples in the `examples/` directory. 