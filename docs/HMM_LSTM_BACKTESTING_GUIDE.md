# HMM-LSTM Backtesting System - Complete Implementation Guide

## Overview

This document provides a complete guide to the HMM-LSTM backtesting system I've implemented for your e-trading project. The system allows you to run comprehensive backtesting on your trained HMM and LSTM models using historical data.

## What I've Created

### 1. Configuration System
- **`config/optimizer/hmm_lstm_01.json`** - Main configuration file for HMM-LSTM backtesting
- Supports multiple symbols, timeframes, and strategy parameters
- Includes risk management and optimization settings

### 2. Core Backtesting Engine
- **`src/backtester/optimizer/hmm_lstm.py`** - Main backtesting script
- Integrates with your existing HMM-LSTM pipeline
- Loads trained models and processes OHLCV data in real-time
- Runs comprehensive backtesting with Backtrader using the original HMMLSTMStrategy

### 3. Execution Scripts
- **`bin/run_hmm_lstm_backtest.bat`** - Windows batch script
- **`bin/run_hmm_lstm_backtest.sh`** - Unix/Linux shell script

### 4. Documentation and Examples
- **`src/backtester/optimizer/README_HMM_LSTM.md`** - Detailed usage documentation
- **`examples/hmm_lstm_backtest_example.py`** - Interactive example script
- **`tests/test_hmm_lstm_backtest.py`** - Comprehensive test suite

## Key Features

### 🔄 **Seamless Pipeline Integration**
- Automatically discovers and loads trained HMM and LSTM models
- Uses labeled data from your existing pipeline
- Maintains consistency with your training configuration

### 📊 **Comprehensive Backtesting**
- Regime-aware trading decisions using HMM models
- Price prediction using LSTM models
- Technical indicator integration
- Risk management controls

### ⚙️ **Parameter Optimization**
- Optional Optuna-based parameter optimization
- Optimizes entry/exit thresholds and regime confidence
- Configurable optimization trials

### 📈 **Performance Analysis**
- Detailed performance metrics (Sharpe ratio, drawdown, returns)
- Trade analysis and statistics
- Automated report generation
- Performance visualization plots

### 🛡️ **Risk Management**
- Position sizing based on volatility and regime
- Dynamic stop-loss and take-profit levels
- Maximum drawdown protection
- Regime-aware position adjustments

## Quick Start

### Prerequisites
1. **Complete HMM-LSTM Pipeline Training**:
   ```bash
   cd src/ml/pipeline/hmm_lstm_01
   python run_pipeline.py
   ```

2. **Verify Models and Data**:
   - Models: `src/ml/pipeline/hmm_lstm_01/models/`
   - OHLCV data: `data/{symbol}_{timeframe}.csv`

### Basic Usage

#### Option 1: Using Batch Scripts
```bash
# Windows
bin\run_hmm_lstm_backtest.bat

# Unix/Linux
chmod +x bin/run_hmm_lstm_backtest.sh
bin/run_hmm_lstm_backtest.sh
```

#### Option 2: Direct Python Execution
```bash
python src/backtester/optimizer/hmm_lstm.py
```

#### Option 3: Interactive Example
```bash
python examples/hmm_lstm_backtest_example.py
```

### Advanced Usage

#### Custom Configuration
```bash
python src/backtester/optimizer/hmm_lstm.py --config config/optimizer/my_config.json
```

#### Automatic Discovery
The optimizer automatically discovers all available symbol-timeframe combinations from the data directory. No need to specify symbols or timeframes manually.

#### Enable Parameter Optimization
Edit `config/optimizer/hmm_lstm_01.json`:
```json
{
  "optimization": {
    "enabled": true,
    "n_trials": 100
  }
}
```

## Configuration Details

### ML Models Section
```json
{
  "ml_models": {
    "pipeline_dir": "src/ml/pipeline/hmm_lstm_01",
    "models_dir": "src/ml/pipeline/hmm_lstm_01/models",
    "config_file": "config/pipeline/x01.yaml"
  }
}
```

### Strategy Parameters
```json
{
  "strategy": {
    "entry_threshold": 0.6,        // LSTM prediction threshold for entry
    "regime_confidence_threshold": 0.7  // Minimum regime confidence
  }
}
```

### Optimization Configuration
```json
{
  "optimization": {
    "enabled": false,
    "n_trials": 50,
    "optimize_params": [
      "entry_threshold",
      "exit_threshold", 
      "regime_confidence_threshold"
    ],
    "parameter_ranges": {
      "entry_threshold": {
        "min": 0.3,
        "max": 0.8,
        "type": "float"
      },
      "exit_threshold": {
        "min": 0.2,
        "max": 0.7,
        "type": "float"
      },
      "regime_confidence_threshold": {
        "min": 0.5,
        "max": 0.9,
        "type": "float"
      }
    }
  }
}
```

**Parameter Types Supported**:
- `float`: Continuous values with min/max bounds
- `int`: Integer values with min/max bounds  
- `categorical`: Discrete choices from a list

### Data Configuration
```json
{
  "data": {
    "data_dir": "data",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01"
  }
}
```

**Automatic Discovery**:
- Automatically discovers all available symbol-timeframe combinations
- Scans `data/` directory for CSV files with format: `{symbol}_{timeframe}.csv`
- Only processes combinations that have both data files and trained models
- Gracefully skips missing combinations with informative warnings
- No need to specify symbols or timeframes in configuration

## Output and Results

### Generated Files
- **Individual Results**: `hmm_lstm_{symbol}_{timeframe}_{timestamp}.json`
- **Summary Report**: `hmm_lstm_summary_{timestamp}.csv`
- **Performance Plots**: `hmm_lstm_performance_{timestamp}.png`

### Key Metrics
- **Total Return**: Overall strategy performance
- **Annual Return**: Annualized return percentage
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Trade Statistics**: Win rate, average trade duration, etc.

## Strategy Logic

The HMMLSTMStrategy (the same one used for live trading) implements the following logic:

### Entry Conditions
1. **LSTM Prediction**: Price prediction exceeds entry threshold
2. **Regime Filtering**: Current market regime has sufficient confidence
3. **Technical Indicators**: Supporting technical analysis signals

### Exit Conditions
1. **LSTM Prediction**: Price prediction falls below exit threshold
2. **Stop Loss**: Position loss exceeds maximum allowed
3. **Take Profit**: Position gain reaches target level
4. **Regime Change**: Market regime shifts significantly

## Testing and Validation

### Run Tests
```bash
python tests/test_hmm_lstm_backtest.py
```

### Test Coverage
- Configuration loading and validation
- Model discovery and loading
- Data preparation
- Parameter optimization
- Results serialization
- Error handling

## Integration with Existing Systems

### Backtrader Compatibility
- Uses existing Backtrader analyzers
- Compatible with custom indicators
- Supports performance visualization

### Pipeline Integration
- Automatic model discovery
- Consistent data format handling
- Shared configuration management

### Results Integration
- JSON format for easy parsing
- CSV summary for spreadsheet analysis
- PNG plots for reporting

## Performance Optimization

### Memory Management
- Efficient data loading with chunking
- GPU memory optimization for PyTorch
- Automatic garbage collection

### Caching
- Model loading and preprocessing
- Technical indicator calculations
- Regime detection results

### Parallel Processing
- Support for multiple symbol-timeframe combinations
- Configurable parallel optimization trials

## Troubleshooting

### Common Issues

#### 1. Models Not Found
```
WARNING: No HMM model found for BTCUSDT 1h
```
**Solution**: Complete the HMM-LSTM pipeline training first.

#### 2. Labeled Data Missing
```
FileNotFoundError: Labeled data not found
```
**Solution**: Ensure the pipeline has generated labeled data files.

#### 3. Memory Issues
```
RuntimeError: CUDA out of memory
```
**Solution**: Use CPU instead of GPU or reduce batch size.

#### 4. Configuration Errors
```
KeyError: Missing required configuration section
```
**Solution**: Check the configuration file structure and required fields.

## Future Enhancements

### Planned Features
1. **Real-time Backtesting**: Live data integration
2. **Multi-asset Portfolio**: Cross-asset correlation analysis
3. **Advanced Risk Metrics**: VaR, CVaR, stress testing
4. **Machine Learning Integration**: Automated feature selection
5. **Cloud Deployment**: Distributed backtesting

### Extensibility
- Custom strategy implementations
- Additional ML model types
- New risk management rules
- Enhanced visualization options

## Support and Maintenance

### Documentation
- Comprehensive README with examples
- Configuration reference
- Troubleshooting guide
- API documentation

### Testing
- Unit tests for core functionality
- Integration tests for pipeline compatibility
- Performance benchmarks
- Error handling validation

### Monitoring
- Detailed logging throughout the process
- Performance metrics tracking
- Error reporting and debugging
- Resource usage monitoring

## Conclusion

The HMM-LSTM backtesting system provides a complete solution for evaluating your trained models on historical data. It integrates seamlessly with your existing pipeline while providing comprehensive analysis and optimization capabilities.

The system is designed to be:
- **Easy to use** with simple command-line interfaces
- **Highly configurable** for different trading scenarios
- **Robust and reliable** with comprehensive error handling
- **Extensible** for future enhancements

You can now run backtesting on your trained HMM and LSTM models to evaluate their performance and optimize trading strategies before deploying them in live trading.

## Next Steps

1. **Run the pipeline training** to generate models and labeled data
2. **Test the backtesting system** with the provided examples
3. **Configure your parameters** based on your trading requirements
4. **Run comprehensive backtesting** on your preferred symbols and timeframes
5. **Analyze results** and optimize strategy parameters
6. **Integrate with live trading** when ready

The system is ready to use and will help you validate and optimize your HMM-LSTM trading strategies effectively!
