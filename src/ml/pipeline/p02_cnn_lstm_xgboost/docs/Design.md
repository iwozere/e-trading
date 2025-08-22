# CNN-LSTM-XGBoost Pipeline Design

## Architecture Overview

The CNN-LSTM-XGBoost pipeline implements a hybrid approach combining deep learning and machine learning for time series prediction. The architecture consists of three main components:

1. **CNN-LSTM with Attention**: Feature extraction from time series data
2. **XGBoost**: Final prediction using extracted features
3. **Technical Indicators**: Additional features for enhanced prediction

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Data      │    │  Technical      │    │  CNN-LSTM       │
│   (OHLCV)       │───▶│  Indicators     │───▶│  Feature        │
│                 │    │                 │    │  Extraction     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Predictions   │◀───│   XGBoost       │◀───│  Combined       │
│   & Signals     │    │   Model         │    │  Features       │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Pipeline Stages

### Stage 1: Data Loading (`x_01_data_loader.py`)
- **Purpose**: Download OHLCV data from multiple providers
- **Input**: Configuration file with symbols and timeframes
- **Output**: Raw CSV files in `data/raw/`
- **Features**:
  - Multi-provider support (Binance, Yahoo Finance)
  - Rate limiting and error handling
  - Consistent file naming convention
  - Parallel downloading

### Stage 2: Feature Engineering (`x_02_feature_engineering.py`)
- **Purpose**: Calculate technical indicators and prepare features
- **Input**: Raw OHLCV data
- **Output**: Processed data with indicators in `data/labeled/`
- **Features**:
  - Configurable technical indicators
  - Data normalization (MinMaxScaler)
  - Missing data handling
  - Feature validation

### Stage 3: CNN-LSTM Optimization (`x_03_optuna_cnn_lstm.py`)
- **Purpose**: Optimize CNN-LSTM hyperparameters using Optuna
- **Input**: Processed data with features
- **Output**: Optimized hyperparameters and study results
- **Features**:
  - Optuna-based hyperparameter optimization
  - Validation MSE as objective
  - Study persistence in SQLite
  - Configurable search spaces

### Stage 4: CNN-LSTM Training (`x_04_train_cnn_lstm.py`)
- **Purpose**: Train CNN-LSTM model with optimized parameters
- **Input**: Processed data and optimized hyperparameters
- **Output**: Trained CNN-LSTM model
- **Features**:
  - PyTorch-based training
  - Attention mechanism
  - Model checkpointing
  - Training visualization

### Stage 5: Feature Extraction (`x_05_extract_features.py`)
- **Purpose**: Extract features from trained CNN-LSTM model
- **Input**: Trained CNN-LSTM model and data
- **Output**: CNN-LSTM features for XGBoost
- **Features**:
  - Feature extraction from intermediate layers
  - Feature combination with technical indicators
  - Feature validation and quality checks

### Stage 6: XGBoost Optimization (`x_06_optuna_xgboost.py`)
- **Purpose**: Optimize XGBoost hyperparameters using Optuna
- **Input**: Combined features (CNN-LSTM + technical indicators)
- **Output**: Optimized XGBoost hyperparameters
- **Features**:
  - Optuna-based optimization
  - Validation MSE as objective
  - Study persistence
  - Configurable search spaces

### Stage 7: XGBoost Training (`x_07_train_xgboost.py`)
- **Purpose**: Train XGBoost model with optimized parameters
- **Input**: Combined features and optimized hyperparameters
- **Output**: Trained XGBoost model
- **Features**:
  - XGBoost training with early stopping
  - Feature importance analysis
  - Model validation
  - Training visualization

### Stage 8: Model Validation (`x_08_validate_models.py`)
- **Purpose**: Evaluate model performance and generate reports
- **Input**: Trained models and test data
- **Output**: Performance metrics, visualizations, and reports
- **Features**:
  - Comprehensive evaluation metrics
  - Trading signal generation
  - Performance visualization
  - Detailed reports

## Model Architecture

### CNN-LSTM Model
```python
class HybridCNNLSTM(nn.Module):
    def __init__(self, time_steps, features, conv_filters, lstm_units, dense_units):
        super().__init__()
        # Convolutional layer
        self.conv1 = nn.Conv1d(features, conv_filters, kernel_size=3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # First LSTM layer
        self.lstm = nn.LSTM(conv_filters, lstm_units, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(lstm_units, num_heads=1, batch_first=True)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(lstm_units, dense_units, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(dense_units, 1)
```

### XGBoost Model
- **Input**: CNN-LSTM features + technical indicators
- **Output**: Price predictions
- **Configurable parameters**: All XGBoost hyperparameters
- **Training**: Early stopping with validation set

## Data Flow

### 1. Data Preparation
```
Raw OHLCV → Technical Indicators → Normalization → Sequence Creation
```

### 2. CNN-LSTM Processing
```
Sequences → CNN → LSTM → Attention → LSTM → Feature Extraction
```

### 3. XGBoost Processing
```
CNN-LSTM Features + Technical Indicators → XGBoost → Predictions
```

## Configuration Design

### YAML Structure
```yaml
# Data sources configuration
data_sources:
  binance:
    symbols: [BTCUSDT, LTCUSDT]
    timeframes: [5m, 15m, 1h, 4h]
  yfinance:
    symbols: [AAPL, MSFT]
    timeframes: [1d]

# CNN-LSTM configuration
cnn_lstm:
  time_steps: 20
  conv_filters_range: [32, 128]
  lstm_units_range: [50, 200]
  dense_units_range: [20, 100]
  dropout: 0.3
  epochs: 50
  batch_size_options: [16, 32, 64]
  learning_rate_range: [1e-5, 1e-2]

# XGBoost configuration
xgboost:
  n_estimators_range: [100, 2000]
  learning_rate_range: [0.001, 0.3]
  max_depth_range: [3, 12]
  subsample_range: [0.5, 1.0]
  colsample_bytree_range: [0.5, 1.0]

# Technical indicators configuration
technical_indicators:
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bb_period: 20
  atr_period: 14
  adx_period: 14

# Optimization configuration
optuna:
  n_trials: 20
  timeout: 3600
  storage: "sqlite:///db/cnn_lstm_xgboost.db"

# Evaluation configuration
evaluation:
  test_split: 0.2
  metrics: [mse, mae, directional_accuracy, sharpe_ratio]
```

## File Structure

```
src/ml/pipeline/p02_cnn_lstm_xgboost/
├── docs/
│   ├── Requirements.md
│   ├── Design.md
│   └── Tasks.md
├── models/
│   ├── cnn_lstm/
│   │   ├── checkpoints/
│   │   ├── studies/
│   │   └── configs/
│   ├── xgboost/
│   │   ├── models/
│   │   ├── studies/
│   │   └── configs/
│   └── results/
│       ├── predictions/
│       ├── visualizations/
│       └── reports/
├── x_01_data_loader.py
├── x_02_feature_engineering.py
├── x_03_optuna_cnn_lstm.py
├── x_04_train_cnn_lstm.py
├── x_05_extract_features.py
├── x_06_optuna_xgboost.py
├── x_07_train_xgboost.py
├── x_08_validate_models.py
├── run_pipeline.py
└── README.md
```

## Error Handling and Logging

### Error Handling Strategy
1. **Data validation**: Check data quality at each stage
2. **Model validation**: Validate model outputs and performance
3. **Graceful degradation**: Continue pipeline with warnings for non-critical errors
4. **Recovery mechanisms**: Retry failed operations with exponential backoff

### Logging Strategy
1. **Structured logging**: JSON-formatted logs for easy parsing
2. **Log levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
3. **Log rotation**: Automatic log file rotation
4. **Performance monitoring**: Track execution times and resource usage

## Performance Considerations

### Memory Management
- **Batch processing**: Process data in configurable batches
- **Garbage collection**: Explicit garbage collection after large operations
- **Memory monitoring**: Track memory usage and optimize where needed

### Parallel Processing
- **Data loading**: Parallel downloads with rate limiting
- **Model training**: GPU utilization for CNN-LSTM
- **Optimization**: Parallel Optuna trials where possible

### Scalability
- **Modular design**: Easy to extend with new models or features
- **Configuration-driven**: All parameters configurable via YAML
- **Resource management**: Configurable timeouts and resource limits
