# HMM-LSTM Backtesting System - Complete Implementation Guide

## Overview

This document provides a complete guide to the HMM-LSTM backtesting system for your e-trading project. The system allows you to run comprehensive backtesting on your trained HMM and LSTM models using historical data, with recent updates to handle log return predictions and optimized thresholds.

## Recent Updates (Latest Version)

### 🔄 **Log Return Prediction Support**
- **LSTM Model**: Now correctly handles log return predictions (log(price[t+1] / price[t]))
- **Prediction Scale**: Typical log returns range from -0.003 to +0.003 (-0.3% to +0.3%)
- **Threshold Adjustment**: Entry thresholds configured for log return scale (e.g., 0.001 = 0.1% predicted change)

### ⚙️ **Updated Strategy Parameters**
- **Entry Threshold**: `0.001` (0.1% log return) - more sensitive to small predictions
- **Regime Confidence**: `0.4` (40% confidence) - lower threshold for HMM models
- **Risk Management**: Adjusted stop-loss (0.5%) and take-profit (1%) for log return scale

### 🛠️ **Enhanced Debugging**
- **HMM Regime Debug**: Logs regime predictions and posteriors every 100 bars
- **Signal Check Debug**: Shows why trades aren't triggered every 50 bars
- **Better Error Handling**: More specific error messages and fallback mechanisms

## What I've Created

### 1. Configuration System
- **`config/optimizer/p01_hmm_lstm.json`** - Main configuration file for HMM-LSTM backtesting
- Supports multiple symbols, timeframes, and strategy parameters
- Includes risk management and optimization settings
- **Updated for log return predictions**

### 2. Core Backtesting Engine
- **`src/backtester/optimizer/hmm_lstm.py`** - Main backtesting script
- Integrates with your existing HMM-LSTM pipeline
- Loads trained models and processes OHLCV data in real-time
- Runs comprehensive backtesting with Backtrader using the enhanced HMMLSTMStrategy

### 3. Enhanced Strategy Implementation
- **`src/strategy/hmm_lstm_strategy.py`** - Updated strategy with log return support
- **Log Return Conversion**: Helper methods for log return ↔ percentage conversion
- **Enhanced LSTM Model**: Matches pipeline architecture with dense layers
- **Improved Debugging**: Detailed logging for troubleshooting

### 4. Execution Scripts
- **`bin/run_hmm_lstm_backtest.bat`** - Windows batch script
- **`bin/run_hmm_lstm_backtest.sh`** - Unix/Linux shell script

## Key Features

### 🔄 **Seamless Pipeline Integration**
- Automatically discovers and loads trained HMM and LSTM models
- Uses labeled data from your existing pipeline
- Maintains consistency with your training configuration

### 📊 **Comprehensive Backtesting**
- **Regime-aware trading decisions** using HMM models
- **Log return price prediction** using LSTM models
- **Technical indicator integration**
- **Risk management controls**

### ⚙️ **Parameter Optimization**
- Optional Optuna-based parameter optimization
- **Optimizes entry/exit thresholds** for log return scale
- **Optimizes regime confidence thresholds**
- Configurable optimization trials

### 📈 **Performance Analysis**
- Detailed performance metrics (Sharpe ratio, drawdown, returns)
- Trade analysis and statistics
- Automated report generation
- Performance visualization plots

### 🛡️ **Risk Management**
- Position sizing based on volatility and regime
- **Dynamic stop-loss and take-profit** levels (adjusted for log return scale)
- Maximum drawdown protection
- Regime-aware position adjustments

## Quick Start

### Prerequisites
1. **Complete HMM-LSTM Pipeline Training**:
   ```bash
   cd src/ml/pipeline/p01_hmm_lstm
   python run_pipeline.py
   ```

2. **Verify Models and Data**:
   - Models: `src/ml/pipeline/p01_hmm_lstm/models/`
   - OHLCV data: `data/raw/{source}_{symbol}_{timeframe}_{start_date}_{end_date}.csv`

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
   Edit `config/optimizer/p01_hmm_lstm.json`:
```json
{
  "optimization": {
    "enabled": true,
    "n_trials": 100
  }
}
```

**⚠️ Important**: Only enable optimization when the strategy is already showing promising results. Optimization is for fine-tuning, not fixing fundamentally broken strategies.

## Configuration Details

### ML Models Section
```json
{
  "ml_models": {
    "pipeline_dir": "src/ml/pipeline/p01_hmm_lstm",
    "models_dir": "src/ml/pipeline/p01_hmm_lstm/models",
    "config_file": "config/pipeline/p01.yaml"
  }
}
```

### Strategy Parameters (Updated for Log Returns)
```json
{
  "strategy": {
    "entry_threshold": 0.001,        // 0.1% log return threshold for entry
    "regime_confidence_threshold": 0.4  // 40% minimum regime confidence
  }
}
```

### Risk Management (Updated for Log Return Scale)
```json
{
  "risk_management": {
    "max_position_size": 0.2,
    "stop_loss_pct": 0.005,          // 0.5% stop loss
    "take_profit_pct": 0.01,         // 1% take profit
    "max_drawdown": 0.15
  }
}
```

### Optimization Configuration (Updated Ranges)
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
        "min": 0.0001,               // 0.01% log return
        "max": 0.002,                // 0.2% log return
        "type": "float"
      },
      "exit_threshold": {
        "min": 0.0002,               // 0.02% log return
        "max": 0.002,                // 0.2% log return
        "type": "float"
      },
      "regime_confidence_threshold": {
        "min": 0.3,                  // 30% confidence
        "max": 0.7,                  // 70% confidence
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
    "data_dir": "data/raw",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01"
  }
}
```

**Automatic Discovery**:
- Automatically discovers all available symbol-timeframe combinations
- Scans `data/raw/` directory for CSV files with format: `{source}_{symbol}_{timeframe}_{start_date}_{end_date}.csv`
- Only processes combinations that have both data files and trained models
- Gracefully skips missing combinations with informative warnings
- No need to specify symbols or timeframes in configuration

## Strategy Logic (Updated)

The enhanced HMMLSTMStrategy implements the following logic:

### **Log Return Prediction Understanding**
- **LSTM Output**: Predicts log returns: `log(price[t+1] / price[t])`
- **Positive Log Return**: Predicted price increase
- **Negative Log Return**: Predicted price decrease
- **Typical Scale**: -0.003 to +0.003 (-0.3% to +0.3%)

### Entry Conditions
1. **LSTM Prediction**: Log return prediction exceeds entry threshold (e.g., 0.001 = 0.1%)
2. **Regime Filtering**: Current market regime has sufficient confidence (e.g., 40%)
3. **Technical Indicators**: Supporting technical analysis signals

### Exit Conditions
1. **LSTM Prediction**: Log return prediction falls below exit threshold
2. **Stop Loss**: Position loss exceeds maximum allowed (0.5%)
3. **Take Profit**: Position gain reaches target level (1%)
4. **Regime Change**: Market regime shifts significantly

### Signal Strength Calculation
- **Log Return Conversion**: Converts log returns to percentages for signal strength
- **Confidence Scaling**: Scales confidence based on percentage change magnitude
- **Risk Multiplier**: Adjusts position size based on prediction strength

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

## Debugging and Troubleshooting

### Enhanced Debug Logging
The strategy now includes comprehensive debug logging:

#### HMM Regime Debug
```
DEBUG: HMM Regime Debug - Regime: 1, Posteriors: [0.2, 0.5, 0.3], Max Confidence: 0.500
```

#### Signal Check Debug
```
DEBUG: Signal Check - Prediction: 0.000112, Threshold: 0.001000, Regime Conf: 0.500, Threshold: 0.400
DEBUG: No trade: Prediction 0.000112 < threshold 0.001000
```

#### Trade Signal Logging
```
INFO: LONG signal - Log Return: 0.001234 (0.1234%), Regime: 1, Confidence: 0.600
```

### Common Issues and Solutions

#### 1. No Trading Signals Generated
**Symptoms**: Strategy runs but generates no trades
**Possible Causes**:
- HMM confidence too low (check if consistently 50%)
- LSTM predictions too small (check prediction magnitudes)
- Entry threshold too high

**Solutions**:
- Lower `regime_confidence_threshold` to 0.3-0.4
- Lower `entry_threshold` to 0.0005-0.001
- Check HMM model quality and training

#### 2. Models Not Found
```
WARNING: No HMM model found for BTCUSDT 1h
```
**Solution**: Complete the HMM-LSTM pipeline training first.

#### 3. Data Files Missing
```
FileNotFoundError: No data file found for BTCUSDT_1h in data/raw
```
**Solution**: Ensure data files follow the format: `{source}_{symbol}_{timeframe}_{start_date}_{end_date}.csv`

#### 4. Memory Issues
```
RuntimeError: CUDA out of memory
```
**Solution**: Use CPU instead of GPU or reduce batch size.

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

## Best Practices

### When to Enable Optimization
1. **Strategy is profitable** with default parameters
2. **Trading signals are being generated** consistently
3. **Models are working correctly** (HMM confidence > 30%, LSTM predictions reasonable)
4. **Basic performance metrics** are acceptable

### Optimization Workflow
1. **Start with manual parameters** and get strategy working
2. **Achieve basic profitability** before optimization
3. **Use optimization for fine-tuning** entry/exit thresholds
4. **Validate results** on out-of-sample data

### Debugging Workflow
1. **Check HMM confidence** - should vary, not stuck at 50%
2. **Check LSTM predictions** - should be in log return scale (-0.003 to +0.003)
3. **Verify thresholds** - entry threshold should be appropriate for log return scale
4. **Monitor debug logs** - use enhanced logging to understand strategy behavior

## Conclusion

The enhanced HMM-LSTM backtesting system provides a complete solution for evaluating your trained models on historical data. The recent updates include:

- **Log return prediction support** with appropriate thresholds
- **Enhanced debugging capabilities** for troubleshooting
- **Updated parameter ranges** for optimization
- **Improved error handling** and fallback mechanisms

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
