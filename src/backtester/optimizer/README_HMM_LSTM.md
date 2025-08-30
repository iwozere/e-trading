# Updated HMM-LSTM Backtesting Optimizer

This document describes the updated HMM-LSTM backtesting optimizer that has been enhanced to match the quality and structure of `custom_optimizer.py`.

## 🎯 Key Improvements

The HMM-LSTM optimizer has been significantly enhanced to provide:

1. **Consistent Results Format** - Same structure as `custom_optimizer.py`
2. **Enhanced Analyzer Integration** - Uses all custom analyzers from `bt_analyzers.py`
3. **Improved Optimization Framework** - Better parameter optimization with Optuna
4. **Better Error Handling** - Robust error handling and logging
5. **Automatic Discovery** - Automatically finds available symbol-timeframe combinations
6. **Comprehensive Performance Analysis** - Detailed metrics and visualizations

## 📋 Prerequisites

Before running the HMM-LSTM optimizer, ensure you have:

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

## 🚀 Quick Start

### Basic Usage

```python
from src.backtester.optimizer.hmm_lstm import HMMLSTMOptimizer

# Initialize optimizer with default configuration
optimizer = HMMLSTMOptimizer("config/optimizer/p01_hmm_lstm.json")

# Run backtesting (automatically discovers available combinations)
optimizer.run()
```

### Command Line Usage

```bash
# Run with default configuration
python src/backtester/optimizer/hmm_lstm.py

# Run with custom configuration
python src/backtester/optimizer/hmm_lstm.py --config config/optimizer/p01_hmm_lstm.json
```

## ⚙️ Configuration

The optimizer uses a JSON configuration file: `config/optimizer/p01_hmm_lstm.json`

### Key Configuration Sections

#### Basic Settings
```json
{
  "optimizer_type": "hmm_lstm",
  "initial_capital": 10000.0,
  "commission": 0.001,
  "position_size": 0.1,
  "plot": true,
  "save_trades": true,
  "output_dir": "results"
}
```

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
      "regime_confidence_threshold"
    ],
    "parameter_ranges": {
      "entry_threshold": {
        "min": 0.3,
        "max": 0.8,
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

## 🔧 Features

### 1. Automatic Discovery

The optimizer automatically discovers available symbol-timeframe combinations:

- Scans `data/` directory for CSV files with format: `{symbol}_{timeframe}.csv`
- Checks for corresponding trained models in the models directory
- Only processes combinations that have both data files and trained models
- Gracefully skips missing combinations with informative warnings

### 2. Enhanced Analyzer Integration

Uses the same comprehensive analyzer suite as `custom_optimizer.py`:

- **Basic Analyzers**: Returns, DrawDown, SQN, TimeDrawDown, VWR, SharpeRatio
- **Custom Analyzers**: ProfitFactor, WinRate, CalmarRatio, CAGR, SortinoRatio, ConsecutiveWinsLosses, PortfolioVolatility
- **Trade Analysis**: Detailed trade-by-trade analysis

### 3. Consistent Results Format

Results are now in the same format as `custom_optimizer.py`:

```python
{
    'symbol': 'BTCUSDT',
    'timeframe': '1h',
    'best_params': {...},
    'total_profit': 1500.0,  # Gross profit (before commission)
    'total_profit_with_commission': 1450.0,  # Net profit (after commission)
    'total_commission': 50.0,  # Total commission paid
    'initial_capital': 10000.0,
    'final_capital': 11450.0,
    'total_return': 0.145,  # 14.5%
    'analyzers': {...},  # All analyzer results
    'trades': [...],  # Trade list
    'strategy_params': {...},
    'timestamp': '2024-01-01T12:00:00'
}
```

### 4. Enhanced Optimization

Improved parameter optimization with Optuna:

- **Faster Optimization**: Uses simplified metrics during optimization trials
- **Better Parameter Sampling**: Configurable parameter ranges and types
- **Robust Error Handling**: Graceful handling of failed trials
- **Final Detailed Analysis**: Runs final backtest with best parameters and full analyzers

### 5. Comprehensive Reporting

Generates detailed reports and visualizations:

- **Performance Summary**: CSV file with all results
- **Performance Plots**: Visual comparison of returns, Sharpe ratios, drawdowns
- **Risk-Return Analysis**: Scatter plots and heatmaps
- **Individual Results**: JSON files for each symbol-timeframe combination

## 📊 Performance Metrics

The optimizer provides comprehensive performance metrics:

### Basic Metrics
- **Total Return**: Overall strategy return
- **Total Profit**: Gross and net profit (with/without commission)
- **Final Capital**: Ending portfolio value

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return relative to maximum drawdown

### Trade Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Trade**: Mean profit/loss per trade
- **Consecutive Wins/Losses**: Streak analysis

### Advanced Metrics
- **CAGR**: Compound Annual Growth Rate
- **Portfolio Volatility**: Standard deviation of returns
- **VWR**: Variability-Weighted Return

## 🔄 Optimization Process

When optimization is enabled, the process follows these steps:

1. **Parameter Sampling**: Optuna suggests parameter values based on configured ranges
2. **Fast Backtesting**: Runs backtests without detailed analyzers for speed
3. **Metric Calculation**: Calculates optimization objective (e.g., total return)
4. **Parameter Update**: Updates strategy parameters for next trial
5. **Best Parameter Selection**: Identifies best performing parameter combination
6. **Final Analysis**: Runs detailed backtest with best parameters

## 📁 Output Structure

The optimizer generates the following output structure:

```
results/
├── hmm_lstm_summary_20240101_120000.csv          # Performance summary
├── hmm_lstm_performance_20240101_120000.png      # Performance plots
├── hmm_lstm_BTCUSDT_1h_20240101_120000.json     # Individual results
├── hmm_lstm_ETHUSDT_4h_20240101_120000.json     # Individual results
└── ...
```

## 🛠️ Usage Examples

### Example 1: Basic Backtesting

```python
from src.backtester.optimizer.hmm_lstm import HMMLSTMOptimizer

# Initialize optimizer
optimizer = HMMLSTMOptimizer("config/optimizer/p01_hmm_lstm.json")

# Run backtesting
optimizer.run()
```

### Example 2: With Optimization

```python
import json

# Load configuration
with open("config/optimizer/p01_hmm_lstm.json", 'r') as f:
    config = json.load(f)

# Enable optimization
config['optimization']['enabled'] = True
config['optimization']['n_trials'] = 50

# Save modified config
with open("config/optimizer/p01_hmm_lstm_opt.json", 'w') as f:
    json.dump(config, f, indent=2)

# Run optimization
optimizer = HMMLSTMOptimizer("config/optimizer/p01_hmm_lstm_opt.json")
optimizer.run()
```

### Example 3: Custom Configuration

```python
# Create custom configuration
custom_config = {
    "optimizer_type": "hmm_lstm",
    "initial_capital": 50000.0,
    "commission": 0.0005,
    "output_dir": "results/custom",
    "ml_models": {
        "pipeline_dir": "src/ml/pipeline/p01_hmm_lstm",
        "models_dir": "src/ml/pipeline/p01_hmm_lstm/models",
        "config_file": "config/pipeline/p01.yaml"
    },
    "strategy": {
        "entry_threshold": 0.5,
        "regime_confidence_threshold": 0.8
    },
    "data": {
        "data_dir": "data",
        "start_date": "2023-06-01",
        "end_date": "2023-12-31"
    },
    "optimization": {
        "enabled": False
    }
}

# Save and run
with open("config/optimizer/custom.json", 'w') as f:
    json.dump(custom_config, f, indent=2)

optimizer = HMMLSTMOptimizer("config/optimizer/custom.json")
optimizer.run()
```

## 🔍 Troubleshooting

### Common Issues

1. **No Models Found**
   - Ensure you've completed the HMM-LSTM pipeline training
   - Check that model files exist in the specified directory
   - Verify model file naming convention: `hmm/lstm_{symbol}_{timeframe}_{timestamp}.pkl`

2. **No Data Files Found**
   - Ensure OHLCV data files exist in the data directory
   - Check file naming convention: `{symbol}_{timeframe}.csv`
   - Verify data file format and required columns

3. **Optimization Errors**
   - Check parameter ranges in configuration
   - Ensure optimization parameters exist in strategy configuration
   - Verify Optuna is installed: `pip install optuna`

4. **Performance Issues**
   - Reduce number of optimization trials for faster execution
   - Use shorter date ranges for testing
   - Disable detailed analyzers during optimization for speed

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run optimizer
optimizer = HMMLSTMOptimizer("config/optimizer/p01_hmm_lstm.json")
optimizer.run()
```

## 📈 Performance Comparison

The updated HMM-LSTM optimizer now provides results in the same format as `custom_optimizer.py`, making it easy to compare performance across different strategies:

```python
# Compare HMM-LSTM vs Custom Strategy results
hmm_lstm_results = {
    'total_profit': 1500.0,
    'total_return': 0.15,
    'sharpe_ratio': 1.2,
    'max_drawdown': 0.08
}

custom_results = {
    'total_profit': 1200.0,
    'total_return': 0.12,
    'sharpe_ratio': 1.1,
    'max_drawdown': 0.10
}

# Easy comparison with consistent format
print(f"HMM-LSTM vs Custom Strategy:")
print(f"Profit: ${hmm_lstm_results['total_profit']} vs ${custom_results['total_profit']}")
print(f"Return: {hmm_lstm_results['total_return']:.1%} vs {custom_results['total_return']:.1%}")
print(f"Sharpe: {hmm_lstm_results['sharpe_ratio']:.2f} vs {custom_results['sharpe_ratio']:.2f}")
```

## 🎯 Next Steps

After running the HMM-LSTM optimizer:

1. **Review Results**: Analyze performance metrics and visualizations
2. **Parameter Tuning**: Use optimization results to fine-tune strategy parameters
3. **Risk Management**: Adjust risk management settings based on drawdown analysis
4. **Live Trading**: Use optimized parameters for live trading implementation
5. **Continuous Monitoring**: Regularly re-run optimization as market conditions change

## 📚 Dependencies

Required packages:
- `backtrader`
- `optuna`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `torch`
- `scikit-learn`
- `pyyaml`

Install with:
```bash
pip install backtrader optuna pandas numpy matplotlib seaborn torch scikit-learn pyyaml
```

## 🤝 Support

For issues and questions:
1. Check the logs for detailed error messages
2. Verify configuration file format
3. Ensure all dependencies are installed
4. Validate data and model files exist
5. Check the troubleshooting section above

The updated HMM-LSTM optimizer provides a robust, comprehensive, and user-friendly framework for evaluating HMM-LSTM trading strategies with the same quality and structure as the custom optimizer.
