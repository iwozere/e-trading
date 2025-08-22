# CNN + XGBoost Trading Pipeline

## Overview

The CNN + XGBoost pipeline is a hybrid machine learning system for financial time series analysis and prediction. It combines 1D Convolutional Neural Networks (CNN) for feature extraction from OHLCV data with XGBoost for final prediction, enhanced by technical analysis indicators.

## Pipeline Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│ Data Loader │───▶│ CNN Training │───▶│ Embedding   │───▶│ TA Features │
│ (x_01)      │    │ (x_02)       │    │ Generation  │    │ (x_04)      │
│             │    │              │    │ (x_03)      │    │             │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
                                                                    │
┌─────────────┐    ┌──────────────┐    ┌─────────────┐            │
│ Validation  │◀───│ XGBoost Train│◀───│ XGBoost Opt.│◀───────────┘
│ (x_07)      │    │ (x_06)       │    │ (x_05)      │
└─────────────┘    └──────────────┘    └─────────────┘
```

## Key Features

### CNN Architecture
- **Input**: 1D time series of OHLCV data (120 candles)
- **Architecture**: 1D Convolutional Neural Network
- **Output**: Embeddings (32-64 dimensions, optimized with Optuna)
- **Purpose**: Extract temporal patterns and features from price data

### Technical Analysis Integration
- **Indicators**: 15+ technical indicators including RSI, MACD, Bollinger Bands, ATR, etc.
- **Combination**: Weighted combination of CNN embeddings and TA features
- **Optimization**: Feature weights optimized during training

### XGBoost Configuration
- **Target**: Multiple targets strategy (price direction, volatility, etc.)
- **Validation**: Time Series Cross-Validation (Walk-Forward)
- **Optimization**: Optuna with 100-200 trials, optimized by log_loss
- **Features**: CNN embeddings + TA indicators

## Pipeline Stages

### Stage 1: Data Loading (x_01_data_loader.py)
- Load and preprocess OHLCV data from CSV files
- Clean data, add log returns, handle missing values
- Support for multiple symbols and timeframes
- Input format: `{provider}_{symbol}_{timeframe}_{start_date}_{end_date}.csv`

### Stage 2: CNN Training (x_02_train_cnn.py)
- Train 1D CNN on OHLCV time series data
- Optimize embedding dimension (32-64) with Optuna
- Generate embeddings for feature extraction
- Save trained CNN models

### Stage 3: Embedding Generation (x_03_generate_embeddings.py)
- Generate CNN embeddings for all available data
- Create labeled datasets with embeddings
- Store embeddings in `data/labeled/` directory

### Stage 4: TA Feature Engineering (x_04_ta_features.py)
- Calculate technical analysis indicators:
  - RSI, MACD, Bollinger Bands, ATR
  - SMA, EMA, Stochastic, Volume ratios
- Combine with CNN embeddings using weighted approach

### Stage 5: XGBoost Optimization (x_05_optuna_xgboost.py)
- Optimize XGBoost hyperparameters with Optuna
- 100-200 trials, optimized by log_loss
- Time series cross-validation

### Stage 6: XGBoost Training (x_06_train_xgboost.py)
- Train final XGBoost model with optimized parameters
- Multiple target strategy
- Save trained models and feature importance

### Stage 7: Validation (x_07_validate_model.py)
- Validate models on out-of-sample data
- Generate performance reports and metrics
- Backtesting results and analysis

## Usage

### Quick Start
```bash
# Run complete pipeline
python run_pipeline.py

# Run specific stages
python run_pipeline.py --skip-stages "4,5"

# Custom configuration
python run_pipeline.py --config config/pipeline/p03.yaml
```

### Configuration
Edit `config/pipeline/p03.yaml` to customize:
- Symbols and timeframes
- CNN architecture parameters
- Technical indicators
- XGBoost hyperparameters
- Training parameters

### Data Requirements
- Input data in `data/` directory
- Format: `{provider}_{symbol}_{timeframe}_{start_date}_{end_date}.csv`
- Columns: timestamp, open, high, low, close, volume

## Output

### Models
- **CNN Models**: `models/cnn/cnn_{symbol}_{timeframe}_{timestamp}.pkl`
- **XGBoost Models**: `models/xgboost/xgb_{symbol}_{timeframe}_{timestamp}.pkl`
- **Configurations**: JSON files with optimized parameters

### Data
- **Labeled Data**: `data/labeled/` with embeddings and features
- **Predictions**: Model predictions and confidence scores
- **Reports**: Performance metrics and validation results

## Performance Metrics

- **Accuracy**: Classification accuracy for directional predictions
- **Log Loss**: Multi-class logarithmic loss
- **Feature Importance**: XGBoost feature importance rankings
- **Backtesting**: Historical performance analysis

## Dependencies

- **Deep Learning**: PyTorch, torchvision
- **ML**: XGBoost, scikit-learn, optuna
- **Data Processing**: pandas, numpy
- **Technical Analysis**: ta-lib, pandas-ta
- **Visualization**: matplotlib, seaborn

## Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Use consistent logging and error handling

## License

This pipeline is part of the e-trading project.
