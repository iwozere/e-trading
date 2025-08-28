# CNN + XGBoost Pipeline

## Overview
The CNN + XGBoost pipeline is a hybrid machine learning approach that combines Convolutional Neural Networks (CNN) for feature extraction from time series data with XGBoost for classification and regression tasks. This pipeline is designed for financial time series analysis and trading strategy development.

## Features
- Multi-provider data loading and preprocessing
- CNN-based feature extraction from OHLCV data
- XGBoost model training with hyperparameter optimization
- Technical indicator feature engineering
- Cross-validation and backtesting capabilities
- Provider-specific configuration management

## Quick Start
Example code showing how to use this pipeline:

```python
from src.ml.pipeline.p03_cnn_xgboost.x_01_data_loader import DataLoader
from src.ml.pipeline.p03_cnn_xgboost.x_02_train_cnn import CNNTrainer
from src.ml.pipeline.p03_cnn_xgboost.x_06_train_xgboost import XGBoostTrainer

# Load and preprocess data
data_loader = DataLoader(config)
result = data_loader.run()

# Train CNN model
cnn_trainer = CNNTrainer(config)
cnn_model = cnn_trainer.train()

# Train XGBoost model
xgboost_trainer = XGBoostTrainer(config)
xgboost_model = xgboost_trainer.train()
```

## Integration
This module integrates with:
- `src.data` - For data retrieval and preprocessing
- `src.ml` - For machine learning utilities
- `src.config` - For configuration management
- `src.notification` - For logging and notifications

## Configuration
The pipeline is configured through `config/pipeline/p03.yaml` with the following key sections:
- `data_sources` - Multi-provider data configuration
- `cnn` - CNN model hyperparameters
- `xgboost` - XGBoost model parameters
- `technical_indicators` - Feature engineering settings
- `training` - Training and validation parameters

## Related Documentation
- [Requirements](Requirements.md) - Technical requirements
- [Design](Design.md) - Architecture and design
- [Tasks](Tasks.md) - Implementation roadmap
