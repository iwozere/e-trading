# HMM-LSTM Backtesting System

This document explains how to use the HMM-LSTM backtesting system to evaluate the performance of trained HMM and LSTM models on historical data.

## Overview

The HMM-LSTM backtesting system integrates with the existing ML pipeline to:
1. Load trained HMM models for regime detection
2. Load trained LSTM models for price prediction
3. Run comprehensive backtesting using Backtrader
4. Generate performance reports and visualizations
5. Optionally optimize strategy parameters using Optuna

## Prerequisites

Before running HMM-LSTM backtesting, ensure you have:

1. **Trained Models**: Complete the HMM-LSTM pipeline training:
   ```bash
   cd src/ml/pipeline/p01_hmm_lstm
   python run_pipeline.py
   ```

2. **OHLCV Data**: Historical OHLCV data files in the format:
   ```
   data/{symbol}_{timeframe}.csv
   ```
   The data should contain columns: datetime, open, high, low, close, volume

3. **Model Files**: Trained models should be available in:
   ```
   src/ml/pipeline/p01_hmm_lstm/models/
   ├── hmm_{symbol}_{timeframe}_{timestamp}.pkl
   └── lstm_{symbol}_{timeframe}_{timestamp}.pkl
   ```

## Configuration

The backtesting system uses a JSON configuration file: `config/optimizer/p01_hmm_lstm.json`

### Key Configuration Sections

#### ML Models
```json
{
  "ml_models": {
    "pipeline_dir": "src/ml/pipeline/p01_hmm_lstm",
    "models_dir": "src/ml/pipeline/p01_hmm_lstm/models",
    "config_file": "config/pipeline/p01.yaml"
  }
}
```

#### Strategy Parameters
```json
{
  "strategy": {
    "name": "HMMLSTMStrategy",
    "entry_threshold": 0.6,
    "exit_threshold": 0.4,
    "prediction_horizon": 1,
    "use_regime_filtering": true,
    "regime_confidence_threshold": 0.7
  }
}
```

#### Data Configuration
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

#### Risk Management
```json
{
  "risk_management": {
    "max_position_size": 0.2,
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.1,
    "max_drawdown": 0.15
  }
}
```

#### Optimization Settings
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

## Usage

### Basic Usage

Run the backtesting with default configuration:

```bash
# Windows
bin\run_hmm_lstm_backtest.bat

# Unix/Linux
chmod +x bin/run_hmm_lstm_backtest.sh
bin/run_hmm_lstm_backtest.sh
```

### Advanced Usage

#### Custom Configuration
```bash
python src/backtester/optimizer/hmm_lstm.py --config config/optimizer/my_config.json
```

#### Automatic Discovery
The optimizer automatically discovers all available symbol-timeframe combinations from the data directory. No need to specify symbols or timeframes manually.

#### Enable Parameter Optimization
Edit the configuration file to enable optimization:
```json
{
  "optimization": {
    "enabled": true,
    "n_trials": 100
  }
}
```

## Output

The backtesting system generates several output files in the `results/` directory:

### Individual Results
- `hmm_lstm_{symbol}_{timeframe}_{timestamp}.json` - Detailed backtest results for each symbol-timeframe combination

### Summary Reports
- `hmm_lstm_summary_{timestamp}.csv` - Summary table with key metrics
- `hmm_lstm_performance_{timestamp}.png` - Performance visualization plots

### Key Metrics

Each backtest result includes:
- **Total Return**: Overall strategy return
- **Annual Return**: Annualized return percentage
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Trade Statistics**: Number of trades, win rate, average trade duration

## Strategy Logic

The HMMLSTMStrategy implements the following logic:

### Entry Conditions
1. **LSTM Prediction**: Price prediction exceeds entry threshold
2. **Regime Filtering**: Current market regime has sufficient confidence
3. **Technical Indicators**: Supporting technical analysis signals

### Exit Conditions
1. **LSTM Prediction**: Price prediction falls below exit threshold
2. **Stop Loss**: Position loss exceeds maximum allowed
3. **Take Profit**: Position gain reaches target level
4. **Regime Change**: Market regime shifts significantly

### Risk Management
- Position sizing based on volatility and regime
- Dynamic stop-loss and take-profit levels
- Maximum drawdown protection
- Regime-aware position adjustments

## Troubleshooting

### Common Issues

#### 1. Models Not Found
```
WARNING: No HMM model found for BTCUSDT 1h
WARNING: No LSTM model found for BTCUSDT 1h
```

**Solution**: Ensure the pipeline training has completed successfully and models are saved in the correct directory.

#### 2. OHLCV Data Missing
```
FileNotFoundError: Data file not found: data/BTCUSDT_1h.csv
```

**Solution**: Ensure you have historical OHLCV data files in the data directory with the correct format.

#### 3. Memory Issues
```
RuntimeError: CUDA out of memory
```

**Solution**: 
- Reduce batch size in configuration
- Use CPU instead of GPU: set `device = 'cpu'`
- Process fewer symbols/timeframes simultaneously

#### 4. Performance Issues
```
ValueError: Missing required columns: ['volume']
```

**Solution**: Ensure the OHLCV data includes all required columns (datetime, open, high, low, close, volume).

### Debug Mode

Enable detailed logging by modifying the logger configuration:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Existing Systems

### Backtrader Integration
The system uses Backtrader for backtesting, making it compatible with:
- Existing Backtrader analyzers
- Custom indicators and strategies
- Performance visualization tools

### Pipeline Integration
Seamless integration with the HMM-LSTM pipeline:
- Automatic model discovery and loading
- Consistent data format handling
- Shared configuration management

### Results Integration
Results can be integrated with:
- Portfolio management systems
- Risk monitoring tools
- Performance reporting dashboards

## Performance Optimization

### Parallel Processing
For multiple symbol-timeframe combinations:
```python
# Enable parallel processing in configuration
{
  "optimization": {
    "n_jobs": 4  # Number of parallel processes
  }
}
```

### Caching
The system implements caching for:
- Model loading and preprocessing
- Technical indicator calculations
- Regime detection results

### Memory Management
- Efficient data loading with chunking
- GPU memory optimization for PyTorch models
- Automatic garbage collection

## Future Enhancements

### Planned Features
1. **Real-time Backtesting**: Live data integration
2. **Multi-asset Portfolio**: Cross-asset correlation analysis
3. **Advanced Risk Metrics**: VaR, CVaR, and stress testing
4. **Machine Learning Integration**: Automated feature selection
5. **Cloud Deployment**: Distributed backtesting on cloud platforms

### Extensibility
The system is designed for easy extension:
- Custom strategy implementations
- Additional ML model types
- New risk management rules
- Enhanced visualization options

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the pipeline documentation
3. Examine the log files for detailed error messages
4. Consult the Backtrader documentation for strategy development

## Examples

### Example 1: Basic Backtesting
```bash
# Run with default settings
python src/backtester/optimizer/hmm_lstm.py
```

### Example 2: Custom Configuration
```bash
# Use custom config with specific symbols
python src/backtester/optimizer/hmm_lstm.py \
  --config config/optimizer/hmm_lstm_custom.json \
  --symbols BTCUSDT ETHUSDT ADAUSDT
```

### Example 3: Parameter Optimization
```bash
# Enable optimization for better parameters
python src/backtester/optimizer/hmm_lstm.py \
  --config config/optimizer/hmm_lstm_optimize.json
```

The configuration file should have optimization enabled:
```json
{
  "optimization": {
    "enabled": true,
    "n_trials": 100,
    "optimize_params": ["entry_threshold", "exit_threshold"]
  }
}
```
