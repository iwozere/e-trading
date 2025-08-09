# HMM-LSTM Trading Pipeline

A complete automated trading pipeline that combines Hidden Markov Models (HMM) for market regime detection with LSTM neural networks for price prediction. The pipeline uses Optuna for hyperparameter optimization of both technical indicators and model architecture.

## Overview

This pipeline implements a sophisticated approach to cryptocurrency trading prediction by:

1. **Market Regime Detection**: Uses HMM to identify different market states (trending, ranging, volatile)
2. **Technical Indicator Optimization**: Employs Optuna to find optimal parameters for technical indicators
3. **LSTM Hyperparameter Optimization**: Uses Optuna to optimize LSTM architecture and training parameters
4. **Regime-Aware Prediction**: Trains LSTM models that incorporate market regime information as features
5. **Comprehensive Validation**: Compares model performance against naive baselines with detailed reporting

## Architecture

```
┌────────────┐      ┌───────────────┐      ┌─────────────┐
│ Data Loader│──▶──▶│ Preprocessing │──▶──▶│ Train HMM   │
└────────────┘      └───────────────┘      └─────┬───────┘
                                                  │
                                                  ▼
                                          ┌─────────────┐
                                          │ Apply HMM   │
                                          └─────┬───────┘
                                                │
                                                ▼
   ┌───────────────────────┐     ┌─────────────────────────┐
   │ Optuna Indicators     │     │ Optuna LSTM              │
   │ (Feature parameters)  │     │ (Model parameters)       │
   └──────────┬────────────┘     └───────────────┬──────────┘
              │                                  │
              ▼                                  ▼
       ┌───────────────┐                 ┌───────────────┐
       │ Train LSTM    │                 │ Validate/Test │
       └───────────────┘                 └───────────────┘
```

## Pipeline Stages

### Stage 1: Data Loading (`x_01_data_loader.py`)
- Downloads OHLCV data for multiple symbols and timeframes
- Uses existing data downloader infrastructure
- Saves data with consistent naming convention
- Supports parallel downloads for efficiency

### Stage 2: Data Preprocessing (`x_02_preprocess.py`)
- Adds log returns and rolling statistics
- Calculates technical indicators with default parameters
- Handles missing values and outliers
- Normalizes features for model training
- Adds time-based features (cyclical encoding)

### Stage 3: HMM Training (`x_03_train_hmm.py`)
- Trains Hidden Markov Models for regime detection
- Uses selected features (log returns, volatility, momentum indicators)
- Generates regime visualizations
- Saves trained models with validation metrics

### Stage 4: HMM Application (`x_04_apply_hmm.py`)
- Applies trained HMM models to label data with regime states
- Adds regime features (confidence, duration, transitions)
- Validates regime quality and distribution
- Saves labeled datasets for LSTM training

### Stage 5: Indicator Optimization (`x_05_optuna_indicators.py`)
- Optimizes technical indicator parameters using Optuna
- Uses trading performance metrics as objectives (Sharpe ratio, profit factor)
- Implements simple trading strategies for evaluation
- Saves optimal parameters for feature generation

### Stage 6: LSTM Optimization (`x_06_optuna_lstm.py`)
- Optimizes LSTM hyperparameters using Optuna
- Searches optimal sequence length, hidden size, learning rate, etc.
- Uses regime information and optimized indicators as features
- Multi-objective optimization (MSE + directional accuracy)

### Stage 7: LSTM Training (`x_07_train_lstm.py`)
- Trains final LSTM models with optimized parameters
- Implements regime-aware architecture with one-hot encoding
- Uses early stopping and learning rate scheduling
- Saves trained models with comprehensive metadata

### Stage 8: Model Validation (`x_08_validate_lstm.py`)
- Validates models against naive baselines
- Calculates comprehensive performance metrics
- Analyzes performance by market regime
- Generates PDF reports with visualizations

## Configuration

The pipeline is configured via `config/pipeline/x01.yaml`:

```yaml
symbols: [BTCUSDT, ETHUSDT, LTCUSDT]
timeframes: [5m, 15m, 1h, 4h]

hmm:
  n_components: 3
  train_window_days: 730

lstm:
  sequence_length: 60
  hidden_size: 64
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  dropout: 0.2
  num_layers: 2

optuna:
  n_trials: 50
  sampler: tpe
  pruning: true
  timeout: 3600

evaluation:
  test_split: 0.1
  baseline_model: naive
  metrics: [mse, directional_accuracy, sharpe_ratio]
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- Required packages: pandas, numpy, scikit-learn, optuna, talib, matplotlib, seaborn, hmmlearn

### Setup
```bash
# Install required packages
pip install torch pandas numpy scikit-learn optuna TA-Lib matplotlib seaborn hmmlearn PyYAML

# Ensure the project structure exists
mkdir -p data/{raw,processed,labeled} results reports config/pipeline
```

## Usage

### Complete Pipeline
Run the entire pipeline from start to finish:
```bash
cd src/ml/pipeline/hmm_lstm_01
python run_pipeline.py
```

### Customized Execution
```bash
# Skip certain stages (e.g., if data already exists)
python run_pipeline.py --skip-stages 1,2

# Process specific symbols
python run_pipeline.py --symbols BTCUSDT,ETHUSDT

# Use custom configuration
python run_pipeline.py --config custom_config.yaml

# List all stages
python run_pipeline.py --list-stages

# Validate requirements only
python run_pipeline.py --validate-only
```

### Individual Stage Execution
Each stage can be run independently:
```bash
python x_01_data_loader.py      # Data loading
python x_02_preprocess.py       # Preprocessing
python x_03_train_hmm.py        # HMM training
python x_04_apply_hmm.py        # HMM application
python x_05_optuna_indicators.py # Indicator optimization
python x_06_optuna_lstm.py      # LSTM optimization
python x_07_train_lstm.py       # LSTM training
python x_08_validate_lstm.py    # Model validation
```

## Output Structure

```
project/
├── data/
│   ├── raw/                    # Downloaded OHLCV data
│   ├── processed/              # Preprocessed data with features
│   └── labeled/                # Data labeled with HMM regimes
├── src/ml/pipeline/hmm_lstm_01/
│   ├── models/
│   │   ├── hmm/               # Trained HMM models
│   │   └── lstm/              # Trained LSTM models
├── results/                    # Optimization results (JSON)
├── reports/                    # Validation reports (PDF)
└── config/pipeline/           # Configuration files
```

## Key Features

### Market Regime Detection
- **3-State HMM**: Identifies trending, ranging, and volatile market conditions
- **Feature Selection**: Uses log returns, volatility, and momentum indicators
- **Regime Persistence**: Tracks regime duration and transition patterns
- **Visualization**: Generates regime overlays on price charts

### Hyperparameter Optimization
- **Technical Indicators**: RSI, Bollinger Bands, MACD, EMA, ATR, Stochastic
- **LSTM Architecture**: Hidden size, layers, dropout, sequence length
- **Training Parameters**: Learning rate, batch size, epochs
- **Multi-Objective**: Balances prediction accuracy and trading performance

### Regime-Aware LSTM
- **Regime Conditioning**: Uses one-hot encoded regime states as additional inputs
- **Comprehensive Features**: Combines OHLC, technical indicators, and regime information
- **Temporal Modeling**: Captures sequential dependencies in financial data
- **Early Stopping**: Prevents overfitting with validation-based stopping

### Performance Evaluation
- **Baseline Comparison**: Compares against naive prediction models
- **Multiple Metrics**: MSE, MAE, directional accuracy, R², Sharpe ratio
- **Regime Analysis**: Performance breakdown by market regime
- **Visualization**: Comprehensive charts and error analysis

## Performance Metrics

The pipeline evaluates models using:

1. **Regression Metrics**:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R-squared (R²)

2. **Trading Metrics**:
   - Directional Accuracy (% correct direction predictions)
   - Hit Rate (% predictions within tolerance)
   - Sharpe-like Ratio (return/volatility)

3. **Improvement Metrics**:
   - MSE improvement over baseline
   - Directional accuracy improvement
   - Regime-specific performance

## Best Practices

### Data Quality
- Ensure sufficient historical data (2+ years for HMM training)
- Handle missing values and outliers appropriately
- Validate data integrity across all pipeline stages

### Hyperparameter Optimization
- Use sufficient trials (50-100) for Optuna optimization
- Set appropriate timeout limits for long-running optimizations
- Consider computational resources when setting trial numbers

### Model Training
- Monitor validation loss for overfitting
- Use early stopping to prevent overtraining
- Save models with comprehensive metadata for reproducibility

### Validation
- Reserve sufficient test data (10-20% of dataset)
- Compare against multiple baseline models
- Analyze performance across different market conditions

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or sequence length
2. **Slow Training**: Reduce number of Optuna trials or use GPU
3. **Poor Performance**: Check data quality and feature engineering
4. **Convergence Issues**: Adjust learning rate or model architecture

### Logging
- All stages provide comprehensive logging
- Check log files for detailed error messages
- Use `--validate-only` to check requirements before running

## Extensions

The pipeline can be extended with:

- **Additional Symbols**: Stocks, forex, commodities
- **More Indicators**: Custom technical indicators
- **Ensemble Models**: Combine multiple LSTM models
- **Real-time Inference**: Live trading integration
- **Advanced Regimes**: More complex HMM architectures
- **Alternative Models**: Transformer, GRU, or other architectures

## Contributing

When contributing to this pipeline:

1. Follow the existing code structure and naming conventions
2. Add comprehensive logging and error handling
3. Update configuration files for new parameters
4. Add validation and testing for new features
5. Update documentation for any changes

## License

This project is part of the e-trading platform and follows the same licensing terms.
