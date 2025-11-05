# Backtester Test Framework - Setup Complete ✓

This document summarizes the backtester test framework that has been created.

## What Was Created

### 1. Configuration Files (JSON)

Located in: `config/backtester/`

- **custom_strategy_test.json** - RSI+BB entry with Fixed Ratio exit
- **rsi_volume_supertrend_test.json** - RSI+Volume+Supertrend entry with ATR exit
- **trailing_stop_test.json** - BB+Volume+Supertrend entry with Trailing Stop exit

### 2. Test Framework

Located in: `src/backtester/tests/`

- **backtester_test_framework.py** - Core framework for running backtests
  - `BacktesterTestFramework` class
  - Configuration loading and validation
  - Strategy and data setup
  - Backtest execution
  - Results validation against assertions
  - Report generation

- **test_custom_strategy.py** - Pytest test suite
  - Unit tests for framework components
  - Integration tests for full backtests
  - Multiple test cases for different strategies

- **__init__.py** - Module initialization
- **README.md** - Comprehensive documentation

### 3. Quick Start Script

Located in: `run_backtest.py` (project root)

Simple CLI tool for running backtests without pytest.

## How to Use

### Method 1: Quick Start Script (Easiest)

```bash
# List available configurations
python run_backtest.py --list-configs

# Run a specific test
python run_backtest.py config/backtester/custom_strategy_test.json

# Run without saving report file
python run_backtest.py config/backtester/custom_strategy_test.json --no-report
```

### Method 2: Direct Framework Usage

```bash
# Run directly with Python
python src/backtester/tests/backtester_test_framework.py config/backtester/custom_strategy_test.json
```

### Method 3: Pytest (For Testing)

```bash
# Run all tests
pytest src/backtester/tests/ -v

# Run specific test
pytest src/backtester/tests/test_custom_strategy.py::test_custom_strategy_rsi_bb_fixed_ratio -v

# Run integration tests
pytest src/backtester/tests/ -v -m integration

# Run with detailed output
pytest src/backtester/tests/ -v -s
```

### Method 4: Programmatic Usage

```python
from src.backtester.tests.backtester_test_framework import run_backtest_from_config

# Run backtest
results = run_backtest_from_config("config/backtester/custom_strategy_test.json")

# Check results
if results['success']:
    print("Test PASSED!")
    print(results['report'])
else:
    print("Test FAILED!")
    print("Failures:", results['validation']['failures'])
```

## JSON Configuration Format

### Minimal Configuration

```json
{
  "test_name": "My Test",
  "strategy": {
    "type": "CustomStrategy",
    "parameters": {
      "entry_logic": {
        "name": "RSIBBMixin",
        "params": {"rsi_period": 14, "rsi_oversold": 30}
      },
      "exit_logic": {
        "name": "FixedRatioExitMixin",
        "params": {"profit_ratio": 1.5, "stop_loss_ratio": 0.5}
      },
      "position_size": 0.1
    }
  },
  "data": {
    "file_path": "data/BTCUSDT_1h.csv",
    "symbol": "BTCUSDT",
    "datetime_col": "timestamp"
  },
  "broker": {
    "cash": 10000.0,
    "commission": 0.001
  }
}
```

### Full Configuration Options

See the example files in `config/backtester/` or the README at `src/backtester/tests/README.md`

## Available Mixins

### Entry Mixins

1. **RSIBBMixin** - RSI + Bollinger Bands
2. **RSIBBVolumeMixin** - RSI + BB + Volume
3. **RSIIchimokuMixin** - RSI + Ichimoku Cloud
4. **RSIOrBBMixin** - RSI OR BB logic
5. **RSIVolumeSuperTrendMixin** - RSI + Volume + Supertrend
6. **BBVolumeSuperTrendMixin** - BB + Volume + Supertrend
7. **HMMLSTMEntryMixin** - ML-based entry

### Exit Mixins

1. **FixedRatioExitMixin** - Fixed profit/stop loss ratios
2. **TrailingStopExitMixin** - Trailing stop loss
3. **ATRExitMixin** - ATR-based exits
4. **SimpleATRExitMixin** - Simplified ATR
5. **AdvancedATRExitMixin** - Enhanced ATR logic
6. **TimeBasedExitMixin** - Time-based exits
7. **MACrossoverExitMixin** - Moving average crossover
8. **RSIBBExitMixin** - RSI + BB exit signals
9. **RSIOrBBExitMixin** - RSI OR BB exit

## Creating Your Own Test

1. **Create JSON config** in `config/backtester/my_test.json`:

```json
{
  "test_name": "My Custom Strategy Test",
  "description": "Testing my strategy combination",
  "strategy": {
    "type": "CustomStrategy",
    "parameters": {
      "entry_logic": {
        "name": "RSIVolumeSuperTrendMixin",
        "params": {
          "rsi_period": 14,
          "rsi_threshold": 40,
          "volume_ma_period": 20,
          "volume_threshold": 1.5,
          "supertrend_period": 10,
          "supertrend_multiplier": 3.0
        }
      },
      "exit_logic": {
        "name": "TrailingStopExitMixin",
        "params": {
          "trail_percent": 5.0,
          "min_profit_percent": 2.0
        }
      },
      "position_size": 0.15
    }
  },
  "data": {
    "file_path": "data/BTCUSDT_1h.csv",
    "symbol": "BTCUSDT",
    "datetime_col": "timestamp",
    "open_col": "open",
    "high_col": "high",
    "low_col": "low",
    "close_col": "close",
    "volume_col": "volume",
    "fromdate": "2023-01-01",
    "todate": "2023-12-31"
  },
  "broker": {
    "cash": 50000.0,
    "commission": 0.001,
    "slippage": 0.0005
  },
  "analyzers": {
    "sharpe_ratio": {"enabled": true},
    "drawdown": {"enabled": true},
    "trades": {"enabled": true},
    "returns": {"enabled": true}
  },
  "assertions": {
    "min_trades": 3,
    "max_drawdown_pct": 40.0,
    "min_sharpe_ratio": 0.5,
    "final_value_greater_than_initial": false
  }
}
```

2. **Run the test**:

```bash
python run_backtest.py config/backtester/my_test.json
```

## Output and Results

### Console Output

```
================================================================================
RUNNING BACKTEST
================================================================================
Config: config/backtester/custom_strategy_test.json

================================================================================
BACKTESTER TEST REPORT
================================================================================

Test Name: CustomStrategy RSI+BB with Fixed Ratio Exit
Generated: 2025-11-05 10:30:45

Performance Metrics:
--------------------------------------------------------------------------------
  Initial Value:    $10,000.00
  Final Value:      $12,500.00
  Total P&L:        $2,500.00
  Total Return:     25.00%

  Sharpe Ratio:     1.45
  Max Drawdown:     12.50%

Trade Statistics:
--------------------------------------------------------------------------------
  Total Trades:     15
  Win Rate:         60.00%
  Profit Factor:    2.00

================================================================================
✓ TEST PASSED
================================================================================
```

### Report Files

Saved to: `results/backtester_tests/`

Format: `{test_name}_{timestamp}.txt`

## Validation and Assertions

Configure assertions in your JSON to automatically validate results:

```json
"assertions": {
  "min_trades": 5,                      // Must have at least 5 trades
  "max_drawdown_pct": 30.0,             // Max drawdown <= 30%
  "min_sharpe_ratio": 1.0,              // Sharpe ratio >= 1.0
  "final_value_greater_than_initial": true  // Must be profitable
}
```

Set any assertion to `null` to skip it.

## Troubleshooting

### Data File Not Found

```bash
# Check if data file exists
ls -la data/BTCUSDT_1h.csv

# Update the file_path in your JSON config
```

### Invalid Mixin Name

Check available mixins:
- `src/strategy/entry/entry_mixin_factory.py`
- `src/strategy/exit/exit_mixin_factory.py`

### No Trades Generated

Possible causes:
1. Entry logic too restrictive (adjust parameters)
2. Insufficient data or wrong date range
3. Position size too large

## Advanced Features

### Batch Testing Multiple Configurations

```python
from pathlib import Path
from src.backtester.tests.backtester_test_framework import run_backtest_from_config

config_dir = Path("config/backtester")
results = []

for config_file in config_dir.glob("*.json"):
    print(f"Testing: {config_file.name}")
    result = run_backtest_from_config(str(config_file))
    results.append({
        'config': config_file.name,
        'return': result['results']['total_return'],
        'trades': result['results']['total_trades'],
        'passed': result['success']
    })

# Analyze results
import pandas as pd
df = pd.DataFrame(results)
print(df.sort_values('return', ascending=False))
```

### Custom Validation Logic

```python
from src.backtester.tests.backtester_test_framework import BacktesterTestFramework

framework = BacktesterTestFramework("config/backtester/my_config.json")
framework.setup_backtest()
results = framework.run_backtest()

# Add custom assertions
assert results['win_rate'] > 0.55, "Win rate should be > 55%"
assert results['profit_factor'] > 2.0, "Profit factor should be > 2.0"
```

## Next Steps

1. **Create your own strategy configuration** by copying and modifying one of the example configs
2. **Run backtests** using the quick start script
3. **Analyze results** and refine your strategy parameters
4. **Add pytest tests** for automated testing in your CI/CD pipeline

## Documentation

For more detailed documentation, see:
- `src/backtester/tests/README.md` - Comprehensive framework documentation
- Example configs in `config/backtester/`
- Test examples in `src/backtester/tests/test_custom_strategy.py`

## Summary

You now have a complete backtester test framework that:
- ✓ Loads configurations from JSON files
- ✓ Dynamically sets up strategies with entry/exit mixins
- ✓ Runs backtests on provided data
- ✓ Validates results against assertions
- ✓ Generates detailed reports
- ✓ Integrates with pytest for automated testing
- ✓ Provides multiple ways to run tests (CLI, Python, pytest)

**Enjoy backtesting! 🚀**
