# CNN + XGBoost Pipeline Design Document

## Overview

The CNN + XGBoost pipeline is a hybrid machine learning system designed for financial time series prediction. It combines the temporal pattern recognition capabilities of 1D Convolutional Neural Networks with the robust classification performance of XGBoost, enhanced by traditional technical analysis indicators.

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw OHLCV     │───▶│   1D CNN        │───▶│   Embeddings    │
│   Data (CSV)    │    │   Feature       │    │   (32-64 dim)   │
│                 │    │   Extractor     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐            │
│   Technical     │───▶│   Feature       │            │
│   Indicators    │    │   Combination   │◀───────────┘
│   (15+ TA)      │    │   (Weighted)    │
└─────────────────┘    └─────────────────┘
                                │
┌─────────────────┐    ┌─────────────────┐
│   Validation    │◀───│   XGBoost       │
│   & Reports     │    │   Classifier    │
└─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Input**: OHLCV CSV files with format `{provider}_{symbol}_{timeframe}_{start_date}_{end_date}.csv`
2. **Preprocessing**: Data cleaning, log returns calculation, normalization
3. **CNN Processing**: 1D CNN extracts temporal features from 120-candle windows
4. **Embedding Generation**: CNN outputs 32-64 dimensional embeddings
5. **TA Features**: 15+ technical indicators calculated
6. **Feature Fusion**: Weighted combination of embeddings and TA features
7. **XGBoost Training**: Multi-target classification with time series CV
8. **Output**: Predictions, feature importance, and performance metrics

## Stage 1: Data Loading (x_01_data_loader.py)

### Purpose
Load and preprocess OHLCV data from CSV files, preparing it for CNN training and feature extraction.

### Input Format
- **File Pattern**: `{provider}_{symbol}_{timeframe}_{start_date}_{end_date}.csv`
- **Columns**: timestamp, open, high, low, close, volume
- **Data Types**: Numeric for OHLCV, datetime for timestamp

### Processing Steps
1. **Data Loading**: Read CSV files using pandas
2. **Data Cleaning**: 
   - Remove rows with missing values
   - Handle outliers using IQR method
   - Ensure chronological order
3. **Feature Engineering**:
   - Calculate log returns: `log(close_t / close_{t-1})`
   - Add price ratios: `high/low`, `close/open`
   - Normalize volume: `volume / volume_sma_20`
4. **Data Validation**:
   - Check for sufficient data points (minimum 1000 candles)
   - Verify data quality and consistency
   - Generate data quality reports

### Output
- **Processed Data**: Clean, normalized OHLCV data with additional features
- **Metadata**: Data statistics, quality metrics, processing logs
- **Format**: Parquet files for efficient storage and access

## Stage 2: CNN Training (x_02_train_cnn.py)

### Purpose
Train 1D Convolutional Neural Networks to extract meaningful temporal features from OHLCV data.

### CNN Architecture

#### Input Layer
- **Shape**: (batch_size, sequence_length, features)
- **Sequence Length**: 120 candles (configurable)
- **Features**: 5 (OHLCV) + additional engineered features

#### Convolutional Layers
```python
# Layer 1: Initial feature extraction
Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')
BatchNormalization()
MaxPooling1D(pool_size=2)

# Layer 2: Pattern recognition
Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')
BatchNormalization()
MaxPooling1D(pool_size=2)

# Layer 3: High-level features
Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')
BatchNormalization()
GlobalAveragePooling1D()
```

#### Embedding Layer
- **Output Dimension**: 32-64 (optimized with Optuna)
- **Activation**: Linear (for embedding extraction)
- **Regularization**: Dropout (0.3-0.5)

### Training Configuration
- **Loss Function**: MSE (for embedding quality)
- **Optimizer**: Adam with learning rate 0.001
- **Batch Size**: 32-128 (optimized)
- **Epochs**: 50-200 (early stopping)
- **Validation**: 20% holdout for early stopping

### Hyperparameter Optimization
- **Framework**: Optuna
- **Trials**: 50-100
- **Parameters**:
  - Embedding dimension: [32, 64]
  - Number of filters: [32, 64, 128, 256]
  - Kernel sizes: [3, 5, 7]
  - Dropout rates: [0.1, 0.3, 0.5]
  - Learning rate: [0.0001, 0.001, 0.01]

### Output
- **Trained CNN Models**: Saved as PyTorch models
- **Optimization Results**: Optuna study with best parameters
- **Training Logs**: Loss curves, validation metrics

## Stage 3: Embedding Generation (x_03_generate_embeddings.py)

### Purpose
Generate CNN embeddings for all available data using trained CNN models.

### Processing Steps
1. **Model Loading**: Load trained CNN models for each symbol/timeframe
2. **Data Preparation**: Create 120-candle sliding windows
3. **Embedding Generation**: Extract embeddings for each window
4. **Label Creation**: Generate target variables for supervised learning
5. **Data Storage**: Save embeddings with corresponding labels

### Target Variables (Multiple Targets Strategy)
1. **Price Direction**: 
   - Up: log_return > threshold
   - Down: log_return < -threshold
   - Sideways: |log_return| <= threshold
2. **Volatility Regime**:
   - Low: rolling_std < percentile_33
   - Medium: percentile_33 <= rolling_std <= percentile_66
   - High: rolling_std > percentile_66
3. **Trend Strength**:
   - Strong: abs(sma_20_slope) > threshold
   - Weak: abs(sma_20_slope) <= threshold

### Output Format
```python
{
    'timestamp': [...],
    'embeddings': [...],  # CNN embeddings
    'target_direction': [...],  # 0, 1, 2
    'target_volatility': [...],  # 0, 1, 2
    'target_trend': [...],  # 0, 1
    'features': {
        'log_return': [...],
        'volume_ratio': [...],
        # ... other features
    }
}
```

## Stage 4: TA Feature Engineering (x_04_ta_features.py)

### Purpose
Calculate technical analysis indicators and combine them with CNN embeddings using weighted fusion.

### Technical Indicators

#### Momentum Indicators
- **RSI (14)**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
  - MACD line, signal line, histogram
- **Stochastic**: Stochastic oscillator (K%, D%)

#### Trend Indicators
- **SMA (20)**: Simple Moving Average
- **EMA (12)**: Exponential Moving Average
- **Price vs SMA/EMA**: Price position relative to moving averages

#### Volatility Indicators
- **ATR (14)**: Average True Range
- **Bollinger Bands**: Upper, lower bands, position

#### Volume Indicators
- **Volume Ratio**: Current volume / SMA volume

### Feature Combination Strategy

#### Weighted Fusion
```python
# Combine CNN embeddings and TA features
combined_features = (
    cnn_weight * normalized_embeddings + 
    ta_weight * normalized_ta_features
)

# Weights optimized during training
cnn_weight + ta_weight = 1.0
```

#### Feature Selection
- **Correlation Analysis**: Remove highly correlated features
- **Feature Importance**: Use XGBoost feature importance for selection
- **Dimensionality Reduction**: PCA if needed

### Output
- **Combined Features**: CNN embeddings + TA indicators
- **Feature Weights**: Optimized weights for fusion
- **Feature Importance**: Ranking of feature importance

## Stage 5: XGBoost Optimization (x_05_optuna_xgboost.py)

### Purpose
Optimize XGBoost hyperparameters using Optuna with time series cross-validation.

### Optimization Strategy

#### Time Series Cross-Validation
- **Method**: Walk-Forward Analysis
- **Folds**: 5-10 folds
- **Validation**: Out-of-sample performance
- **Metric**: Multi-class log loss

#### Hyperparameters to Optimize
```python
param_space = {
    'max_depth': [3, 6, 9, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500, 1000],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3]
}
```

#### Optimization Configuration
- **Trials**: 100-200
- **Objective**: Minimize log loss
- **Early Stopping**: 20 trials without improvement
- **Pruning**: Median pruner for efficiency

### Output
- **Best Parameters**: Optimized hyperparameters
- **Study Results**: Optuna study with all trials
- **Performance Metrics**: CV scores and confidence intervals

## Stage 6: XGBoost Training (x_06_train_xgboost.py)

### Purpose
Train final XGBoost models with optimized hyperparameters for multiple targets.

### Training Configuration

#### Model Architecture
- **Algorithm**: XGBoost with multi-class classification
- **Number of Classes**: 3 for direction/volatility, 2 for trend
- **Objective**: Multi-class log loss
- **Evaluation Metric**: Multi-class log loss

#### Training Strategy
- **Data Split**: 70% train, 15% validation, 15% test
- **Time Series Split**: Maintain chronological order
- **Early Stopping**: Based on validation loss
- **Class Balancing**: Handle imbalanced classes

#### Feature Engineering
- **Feature Scaling**: StandardScaler for numerical features
- **Categorical Encoding**: One-hot encoding if needed
- **Feature Selection**: Based on importance scores

### Output
- **Trained Models**: XGBoost models for each target
- **Feature Importance**: Ranking of feature importance
- **Training Metrics**: Loss curves, accuracy scores
- **Model Artifacts**: Pickled models and configurations

## Stage 7: Validation (x_07_validate_model.py)

### Purpose
Validate trained models on out-of-sample data and generate comprehensive performance reports.

### Validation Metrics

#### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class metrics
- **Confusion Matrix**: Detailed classification results
- **Log Loss**: Multi-class logarithmic loss

#### Time Series Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst historical loss
- **Win Rate**: Percentage of profitable predictions
- **Profit Factor**: Gross profit / gross loss

#### Feature Analysis
- **Feature Importance**: XGBoost feature rankings
- **Feature Correlation**: Correlation analysis
- **SHAP Values**: Model interpretability

### Backtesting Framework
- **Walk-Forward Analysis**: Rolling window validation
- **Transaction Costs**: Realistic trading costs
- **Position Sizing**: Risk management rules
- **Performance Attribution**: Analysis of returns

### Output
- **Performance Reports**: Comprehensive validation results
- **Visualizations**: Charts and graphs
- **Model Comparison**: Compare with baseline models
- **Recommendations**: Model deployment suggestions

## Configuration Management

### Configuration File Structure
```yaml
# config/pipeline/p03.yaml
data:
  symbols: ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
  timeframes: ["15m", "1h", "4h"]
  start_date: "2022-01-01"
  end_date: "2025-07-07"

cnn:
  sequence_length: 120
  embedding_dim: [32, 64]
  filters: [64, 128, 256]
  kernel_sizes: [3, 5, 7]
  dropout: [0.1, 0.3, 0.5]

technical_indicators:
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bb_period: 20
  bb_std: 2
  atr_period: 14
  sma_period: 20
  ema_period: 12

xgboost:
  n_trials: 200
  cv_folds: 5
  early_stopping_rounds: 20
  eval_metric: "mlogloss"

targets:
  direction_threshold: 0.001
  volatility_percentiles: [33, 66]
  trend_threshold: 0.0001
```

## Error Handling and Logging

### Error Handling Strategy
- **Data Validation**: Comprehensive input validation
- **Model Training**: Graceful handling of training failures
- **Resource Management**: Memory and GPU management
- **Recovery Mechanisms**: Checkpoint and restart capabilities

### Logging Framework
- **Structured Logging**: JSON format for machine readability
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Performance Monitoring**: Training time, memory usage
- **Audit Trail**: Complete pipeline execution history

## Performance Considerations

### Computational Requirements
- **CPU**: Multi-core processing for data preprocessing
- **GPU**: CUDA support for CNN training
- **Memory**: Sufficient RAM for large datasets
- **Storage**: Fast I/O for data loading/saving

### Optimization Strategies
- **Data Parallelism**: Multi-GPU training
- **Batch Processing**: Efficient data loading
- **Caching**: Intermediate results caching
- **Compression**: Data compression for storage

## Security and Privacy

### Data Security
- **Input Validation**: Sanitize all inputs
- **Access Control**: Restrict model access
- **Audit Logging**: Track all operations
- **Data Encryption**: Encrypt sensitive data

### Model Security
- **Model Validation**: Validate model outputs
- **Adversarial Testing**: Test against adversarial inputs
- **Version Control**: Track model versions
- **Deployment Security**: Secure model deployment

## Future Enhancements

### Planned Improvements
- **Ensemble Methods**: Combine multiple models
- **Online Learning**: Incremental model updates
- **Real-time Processing**: Stream processing capabilities
- **Advanced Features**: Additional technical indicators

### Research Directions
- **Attention Mechanisms**: Self-attention for time series
- **Graph Neural Networks**: Market relationship modeling
- **Reinforcement Learning**: RL for trading strategy
- **Causal Inference**: Causal relationships in markets
