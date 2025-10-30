# CNN-LSTM-XGBoost Pipeline Requirements

## Overview
This document outlines the requirements for the CNN-LSTM-XGBoost hybrid trading pipeline, which combines deep learning (CNN-LSTM with attention) and machine learning (XGBoost) for time series prediction and trading signal generation.

## Core Requirements

### 1. Data Management
- **Multi-provider support**: Binance (crypto), Yahoo Finance (stocks/ETFs)
- **Data formats**: OHLCV data with consistent naming convention
- **Data validation**: Quality checks and missing data handling
- **Rate limiting**: Respect API limits for all providers
- **Storage**: Raw data in `data/raw`, processed data in `data/labeled`

### 2. Feature Engineering
- **Technical indicators**: Configurable via YAML
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands (upper, middle, lower)
  - ATR (Average True Range)
  - ADX (Average Directional Index)
  - OBV (On-Balance Volume)
- **Data preprocessing**: MinMaxScaler normalization
- **Sequence preparation**: Configurable time steps for LSTM input
- **Feature selection**: Configurable feature sets

### 3. CNN-LSTM Model
- **Architecture**: Hybrid CNN-LSTM with attention mechanism
- **Configurable parameters**:
  - Convolutional filters
  - LSTM units (both layers)
  - Dense units
  - Dropout rates
  - Learning rate
  - Batch size
  - Number of epochs
- **Attention mechanism**: Multi-head attention for sequence modeling
- **Output**: Feature representations for XGBoost

### 4. XGBoost Model
- **Purpose**: Final prediction using CNN-LSTM features
- **Configurable parameters**:
  - Number of estimators
  - Learning rate
  - Max depth
  - Subsample and colsample ratios
  - Regularization parameters (alpha, lambda, gamma)
- **Input**: CNN-LSTM features + technical indicators
- **Output**: Price predictions and trading signals

### 5. Hyperparameter Optimization
- **Framework**: Optuna for both models
- **Optimization targets**:
  - CNN-LSTM: Validation MSE
  - XGBoost: Validation MSE
- **Configurable trials**: Number of optimization trials
- **Study persistence**: SQLite storage for optimization results

### 6. Model Evaluation
- **Metrics**:
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - Directional accuracy
  - Sharpe ratio
  - Maximum drawdown
  - Win rate
- **Visualization**: Prediction plots, performance charts
- **Reports**: Comprehensive evaluation reports

### 7. Pipeline Architecture
- **Modular stages**: 8 distinct pipeline stages
- **Configuration**: YAML-based configuration (`config/pipeline/x02.yaml`)
- **Error handling**: Comprehensive error handling and logging
- **Progress tracking**: Stage-by-stage progress monitoring
- **Parallel processing**: Where applicable

### 8. Output Generation
- **Trading signals**: Buy/sell/hold recommendations
- **Price predictions**: Future price forecasts
- **Model artifacts**: Saved models, optimization results
- **Reports**: Performance analysis and visualization
- **Logs**: Detailed execution logs

## Technical Requirements

### Dependencies
- PyTorch (CNN-LSTM)
- XGBoost
- TA-Lib (technical indicators)
- Optuna (hyperparameter optimization)
- Pandas, NumPy, Scikit-learn
- Matplotlib, Plotly (visualization)
- PyYAML (configuration)

### Performance Requirements
- **Training time**: Configurable timeout per optimization trial
- **Memory usage**: Efficient memory management for large datasets
- **Scalability**: Support for multiple symbols and timeframes
- **Reproducibility**: Deterministic results with seed setting

### Quality Requirements
- **Code quality**: Comprehensive error handling and logging
- **Documentation**: Detailed docstrings and README files
- **Testing**: Unit tests for critical components
- **Validation**: Data and model validation at each stage

## Configuration Requirements

### YAML Configuration Structure
```yaml
data_sources:
  binance:
    symbols: [BTCUSDT, LTCUSDT]
    timeframes: [5m, 15m, 1h, 4h]
  yfinance:
    symbols: [AAPL, MSFT]
    timeframes: [1d]

cnn_lstm:
  time_steps: 20
  conv_filters: [32, 64, 128]
  lstm_units: [50, 100, 200]
  dense_units: [20, 50, 100]
  dropout: 0.3
  epochs: 50
  batch_size: [16, 32, 64]
  learning_rate: [1e-5, 1e-2]

xgboost:
  n_estimators: [100, 500, 1000, 2000]
  learning_rate: [0.001, 0.3]
  max_depth: [3, 6, 12]
  subsample: [0.5, 1.0]
  colsample_bytree: [0.5, 1.0]

technical_indicators:
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bb_period: 20
  atr_period: 14
  adx_period: 14

optuna:
  n_trials: 20
  timeout: 3600
  storage: "sqlite:///db/cnn_lstm_xgboost.db"

evaluation:
  test_split: 0.2
  metrics: [mse, mae, directional_accuracy, sharpe_ratio]
```

## Success Criteria
1. **Accuracy**: Improved prediction accuracy over baseline models
2. **Performance**: Reasonable training and inference times
3. **Reliability**: Robust error handling and recovery
4. **Usability**: Clear documentation and easy configuration
5. **Extensibility**: Modular design for future enhancements
