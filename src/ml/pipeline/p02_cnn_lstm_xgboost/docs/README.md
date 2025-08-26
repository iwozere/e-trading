# CNN-LSTM-XGBoost Trading Pipeline

A hybrid deep learning and machine learning pipeline for time series prediction and trading signal generation, combining CNN-LSTM with attention mechanism and XGBoost regression.

## Overview

This pipeline implements a sophisticated approach to financial time series prediction by:

1. **Feature Extraction**: Using CNN-LSTM with attention mechanism to extract complex patterns from time series data
2. **Final Prediction**: Using XGBoost to combine CNN-LSTM features with technical indicators for robust predictions
3. **Trading Signals**: Generating buy/sell/hold recommendations based on model predictions

## Architecture

```
Raw OHLCV Data → Technical Indicators → CNN-LSTM → Feature Extraction → XGBoost → Predictions & Signals
```

### Key Components

- **CNN-LSTM Model**: Hybrid architecture with convolutional layers for feature extraction and LSTM layers with attention for temporal modeling
- **XGBoost Model**: Gradient boosting for final prediction using CNN-LSTM features and technical indicators
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, OBV
- **Hyperparameter Optimization**: Optuna-based optimization for both models

## Pipeline Stages

1. **Data Loading** (`x_01_data_loader.py`): Download OHLCV data from multiple providers
2. **Feature Engineering** (`x_02_feature_engineering.py`): Calculate technical indicators and prepare features
3. **CNN-LSTM Optimization** (`x_03_optuna_cnn_lstm.py`): Optimize CNN-LSTM hyperparameters
4. **CNN-LSTM Training** (`x_04_train_cnn_lstm.py`): Train CNN-LSTM model with optimized parameters
5. **Feature Extraction** (`x_05_extract_features.py`): Extract features from trained CNN-LSTM
6. **XGBoost Optimization** (`x_06_optuna_xgboost.py`): Optimize XGBoost hyperparameters
7. **XGBoost Training** (`x_07_train_xgboost.py`): Train XGBoost model with optimized parameters
8. **Model Validation** (`x_08_validate_models.py`): Evaluate models and generate reports

## Quick Start

### Prerequisites

```bash
pip install torch xgboost optuna talib pandas numpy scikit-learn matplotlib pyyaml
```

### Configuration

Create or modify `config/pipeline/x02.yaml`:

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
  conv_filters_range: [32, 128]
  lstm_units_range: [50, 200]
  dense_units_range: [20, 100]
  dropout: 0.3
  epochs: 50
  batch_size_options: [16, 32, 64]
  learning_rate_range: [1e-5, 1e-2]

xgboost:
  n_estimators_range: [100, 2000]
  learning_rate_range: [0.001, 0.3]
  max_depth_range: [3, 12]
  subsample_range: [0.5, 1.0]
  colsample_bytree_range: [0.5, 1.0]

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

### Running the Pipeline

```bash
# Run complete pipeline
python src/ml/pipeline/p02_cnn_lstm_xgboost/run_pipeline.py

# Run specific stages
python src/ml/pipeline/p02_cnn_lstm_xgboost/run_pipeline.py --skip-stages "3,4,5,6,7,8"

# Run with custom configuration
python src/ml/pipeline/p02_cnn_lstm_xgboost/run_pipeline.py --config config/pipeline/x02_custom.yaml
```

## Model Architecture

### CNN-LSTM Model

```python
class HybridCNNLSTM(nn.Module):
    def __init__(self, time_steps, features, conv_filters, lstm_units, dense_units):
        super().__init__()
        # Convolutional layer for feature extraction
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
- **Training**: Early stopping with validation set
- **Optimization**: Optuna-based hyperparameter tuning

## Features

### Multi-Provider Data Support
- **Binance**: Cryptocurrency data (BTCUSDT, LTCUSDT, etc.)
- **Yahoo Finance**: Stock and ETF data (AAPL, MSFT, etc.)
- **Rate Limiting**: Respects API limits for all providers
- **Parallel Downloading**: Efficient data acquisition

### Technical Indicators
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Upper, middle, and lower bands
- **ATR**: Average True Range
- **ADX**: Average Directional Index
- **OBV**: On-Balance Volume

### Hyperparameter Optimization
- **Optuna Framework**: State-of-the-art optimization
- **Study Persistence**: SQLite storage for optimization results
- **Configurable Search Spaces**: All parameters configurable via YAML
- **Timeout Handling**: Configurable timeouts for optimization trials

### Model Evaluation
- **Comprehensive Metrics**: MSE, MAE, directional accuracy, Sharpe ratio
- **Trading Signals**: Buy/sell/hold recommendations
- **Visualization**: Performance charts and prediction plots
- **Reports**: Detailed evaluation reports

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

## Configuration Options

### Data Sources
- **Symbols**: List of trading symbols to process
- **Timeframes**: List of timeframes (5m, 15m, 1h, 4h, 1d)
- **Providers**: Data provider configuration

### CNN-LSTM Parameters
- **Time Steps**: Number of timesteps in input sequences
- **Convolutional Filters**: Range for number of filters
- **LSTM Units**: Range for LSTM layer sizes
- **Dense Units**: Range for dense layer sizes
- **Dropout**: Dropout rate for regularization
- **Epochs**: Number of training epochs
- **Batch Size**: Training batch size options
- **Learning Rate**: Learning rate range for optimization

### XGBoost Parameters
- **Number of Estimators**: Range for number of trees
- **Learning Rate**: Learning rate range
- **Max Depth**: Maximum tree depth range
- **Subsample**: Subsample ratio range
- **Colsample by Tree**: Column sample ratio range

### Technical Indicators
- **RSI Period**: Period for RSI calculation
- **MACD Parameters**: Fast, slow, and signal periods
- **Bollinger Bands Period**: Period for BB calculation
- **ATR Period**: Period for ATR calculation
- **ADX Period**: Period for ADX calculation

### Optimization
- **Number of Trials**: Number of Optuna trials
- **Timeout**: Timeout per optimization trial
- **Storage**: SQLite storage location

### Evaluation
- **Test Split**: Fraction of data for testing
- **Metrics**: List of evaluation metrics

## Usage Examples

### Basic Usage

```python
from src.ml.pipeline.p02_cnn_lstm_xgboost.run_pipeline import PipelineRunner

# Initialize pipeline
pipeline = PipelineRunner("config/pipeline/x02.yaml")

# Run complete pipeline
pipeline.run()

# Run specific stages
pipeline.run_stages([1, 2, 3])  # Run stages 1, 2, and 3
```

### Custom Configuration

```python
import yaml

# Load custom configuration
with open("config/pipeline/x02_custom.yaml", "r") as f:
    config = yaml.safe_load(f)

# Modify configuration
config["cnn_lstm"]["epochs"] = 100
config["optuna"]["n_trials"] = 50

# Save modified configuration
with open("config/pipeline/x02_modified.yaml", "w") as f:
    yaml.dump(config, f)

# Run with modified configuration
pipeline = PipelineRunner("config/pipeline/x02_modified.yaml")
pipeline.run()
```

### Individual Stage Execution

```python
# Run individual stages
from src.ml.pipeline.p02_cnn_lstm_xgboost.x_01_data_loader import DataLoader
from src.ml.pipeline.p02_cnn_lstm_xgboost.x_02_feature_engineering import FeatureEngineer

# Load data
data_loader = DataLoader("config/pipeline/x02.yaml")
data_loader.run()

# Engineer features
feature_engineer = FeatureEngineer("config/pipeline/x02.yaml")
feature_engineer.run()
```

## Performance Considerations

### Memory Management
- **Batch Processing**: Process data in configurable batches
- **Garbage Collection**: Explicit cleanup after large operations
- **Memory Monitoring**: Track memory usage and optimize

### GPU Utilization
- **Automatic Detection**: Automatically detect and use available GPUs
- **Memory Optimization**: Efficient GPU memory management
- **Mixed Precision**: Optional mixed precision training

### Parallel Processing
- **Data Loading**: Parallel downloads with rate limiting
- **Optimization**: Parallel Optuna trials where possible
- **Model Training**: GPU-accelerated training

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Reduce sequence length
   - Use CPU training if GPU memory is insufficient

2. **Data Download Failures**
   - Check internet connection
   - Verify API keys (if required)
   - Check rate limiting settings

3. **Optimization Timeout**
   - Increase timeout in configuration
   - Reduce number of trials
   - Use smaller search spaces

4. **Poor Model Performance**
   - Increase training epochs
   - Adjust hyperparameter ranges
   - Add more technical indicators
   - Check data quality

### Debug Mode

```bash
# Run with debug logging
python src/ml/pipeline/p02_cnn_lstm_xgboost/run_pipeline.py --debug

# Run individual stage with debug
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from src.ml.pipeline.p02_cnn_lstm_xgboost.x_01_data_loader import DataLoader
DataLoader('config/pipeline/x02.yaml').run()
"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **PyTorch**: Deep learning framework
- **XGBoost**: Gradient boosting library
- **Optuna**: Hyperparameter optimization
- **TA-Lib**: Technical analysis library
- **Pandas**: Data manipulation library
