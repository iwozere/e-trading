# CNN + XGBoost Pipeline

## Overview
The CNN + XGBoost pipeline is a hybrid machine learning approach that combines Convolutional Neural Networks (CNN) for feature extraction from time series data with XGBoost for classification and regression tasks. This pipeline is designed for financial time series analysis and trading strategy development.

**Key Features:**
- **Individual Model Training**: Each data file gets its own trained CNN model
- **Intelligent Data Batching**: Automatic handling of yfinance period limits with intelligent batching
- **Comprehensive Artifacts**: Individual model files, parameters, reports, and visualizations
- **Multi-provider Support**: Support for yfinance, Binance, and other data providers
- **Hyperparameter Optimization**: Automated optimization using Optuna
- **Advanced Visualization**: Training progress and performance visualizations
- **CSV Format Consistency**: All data files use CSV format for compatibility and ease of processing

## Recent Updates

### Enhanced Data Loading
- **Intelligent Batching**: yfinance downloader now automatically splits large date ranges into smaller chunks to respect API limits
- **Period Flexibility**: Support for any period length (4y, 5y, 10y, etc.) with automatic batching
- **Error Recovery**: Robust error handling with fallback mechanisms

### Individual Model Training
- **Per-File Models**: Each data file (e.g., `yfinance_VT_1d_20210829_20250828.csv`) gets its own trained CNN model
- **Proper Naming**: Models follow the convention: `cnn_{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}.pth`
- **Individual Artifacts**: Each model has its own configuration, report, and visualization files

### Data Format Consistency
- **CSV Throughout**: All pipeline stages use CSV format for data storage and exchange
- **Compatibility**: CSV format ensures compatibility with various tools and systems
- **Ease of Processing**: Human-readable format for debugging and analysis
- **Consistent Naming**: Standardized file naming across all pipeline stages

### Comprehensive Output Structure
```
src/ml/pipeline/p03_cnn_xgboost/models/cnn/
├── cnn_yfinance_VT_1d_20210829_20250828_20250812_221148.pth          # Model file
├── cnn_yfinance_VT_1d_20210829_20250828_20250812_221148_config.json  # Model parameters
├── cnn_yfinance_VT_1d_20210829_20250828_20250812_221148_scaler.pkl   # Data scaler
├── reports/
│   ├── cnn_yfinance_VT_1d_20210829_20250828_20250812_221148_report.json
│   └── full_cnn_optimization_report_20250812_221148.csv
└── visualizations/
    └── cnn_yfinance_VT_1d_20210829_20250828_20250812_221148.png

data/
├── raw/                    # Original downloaded data (CSV)
├── labeled/                # Data with CNN embeddings (CSV)
└── features/               # Data with technical indicators (CSV)
```

## Features
- Multi-provider data loading with intelligent batching
- Individual CNN model training per data file
- Hyperparameter optimization with Optuna
- Technical indicator feature engineering
- Cross-validation and backtesting capabilities
- Provider-specific configuration management
- Comprehensive visualization and reporting
- Automatic error handling and recovery
- **CSV format consistency** throughout the pipeline

## Quick Start
Example code showing how to use this pipeline:

```python
from src.ml.pipeline.p03_cnn_xgboost.x_01_data_loader import DataLoader
from src.ml.pipeline.p03_cnn_xgboost.x_02_train_cnn import CNNTrainer
from src.ml.pipeline.p03_cnn_xgboost.x_06_train_xgboost import XGBoostTrainer

# Load and preprocess data (with intelligent batching)
data_loader = DataLoader(config)
result = data_loader.run()

# Train individual CNN models for each data file
cnn_trainer = CNNTrainer(config)
cnn_results = cnn_trainer.run()  # Returns results for all trained models

# Train XGBoost model
xgboost_trainer = XGBoostTrainer(config)
xgboost_model = xgboost_trainer.train()
```

## Configuration
The pipeline is configured through `config/pipeline/p03.yaml` with the following key sections:

### Data Sources
```yaml
data_sources:
  yfinance:
    symbols: [VT, PSNY, GOOG]
    timeframes: [1d]
    period: "4y"  # Will be automatically batched if needed
  binance:
    symbols: [BTCUSDT, ETHUSDT]
    timeframes: [15m, 1h, 4h]
    period: "2y"
```

### CNN Configuration
```yaml
cnn:
  sequence_length: 120
  max_samples: 10000
  optimization_trials: 10
  filters: [32, 64, 128]
  kernel_sizes: [3, 5, 7]
  dropout: [0.1, 0.2, 0.3]
  batch_size: [32, 64]
  learning_rate: [0.0001, 0.001]
  epochs: [50, 100]
```

## Data Flow

### Pipeline Stages and Data Formats
1. **Data Loading** (`x_01_data_loader.py`)
   - Input: None (downloads from providers)
   - Output: `data/raw/*.csv` (original OHLCV data)

2. **CNN Training** (`x_02_train_cnn.py`)
   - Input: `data/raw/*.csv`
   - Output: `models/cnn/*.pth`, `*.json`, `*.pkl` (trained models and artifacts)

3. **Embedding Generation** (`x_03_generate_embeddings.py`)
   - Input: `data/raw/*.csv`
   - Output: `data/labeled/*_labeled.csv` (data with CNN embeddings)

4. **Technical Analysis** (`x_04_ta_features.py`)
   - Input: `data/labeled/*_labeled.csv`
   - Output: `data/features/*_features.csv` (data with technical indicators)

5. **XGBoost Training** (`x_05_optuna_xgboost.py`, `x_06_train_xgboost.py`)
   - Input: `data/features/*_features.csv`
   - Output: `models/xgboost/*.pkl` (trained XGBoost models)

6. **Validation** (`x_07_validate_model.py`)
   - Input: `data/features/*_features.csv`
   - Output: `results/*.json` (validation reports)

## Integration
This module integrates with:
- `src.data` - For data retrieval and preprocessing (with enhanced batching)
- `src.ml` - For machine learning utilities
- `src.config` - For configuration management
- `src.notification` - For logging and notifications

## Output Files

### Model Files
- **`.pth`**: PyTorch model weights
- **`_config.json`**: Model architecture and training parameters
- **`_scaler.pkl`**: Data normalization scaler

### Data Files (CSV Format)
- **`data/raw/*.csv`**: Original OHLCV data from providers
- **`data/labeled/*_labeled.csv`**: Data with CNN embeddings
- **`data/features/*_features.csv`**: Data with technical indicators and targets

### Reports
- **`_report.json`**: Individual model training report
- **`full_cnn_optimization_report_*.csv`**: Overall optimization summary

### Visualizations
- **`.png`**: Training loss plots and model performance visualizations

## Related Documentation
- [Requirements](Requirements.md) - Technical requirements
- [Design](Design.md) - Architecture and design
- [Tasks](Tasks.md) - Implementation roadmap
- [Data Format Guide](Data_Format_Guide.md) - CSV format consistency and data flow
