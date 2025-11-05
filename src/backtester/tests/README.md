# Backtester Test Framework

A comprehensive testing framework for backtesting trading strategies with JSON configuration support.

## Overview

This framework allows you to:
- Define backtest configurations in JSON files
- Test strategies with different entry/exit mixin combinations
- Validate results against assertions
- Generate detailed test reports
- Run automated tests with pytest

## Quick Start

### 1. Create a JSON Configuration

Create a configuration file in `config/backtester/` (e.g., `my_strategy_test.json`):

```json
{
  "test_name": "My Strategy Test",
  "description": "Testing CustomStrategy with specific mixins",

  "strategy": {
    "type": "CustomStrategy",
    "parameters": {
      "entry_logic": {
        "name": "RSIBBMixin",
        "params": {
          "rsi_period": 14,
          "rsi_oversold": 30,
          "rsi_overbought": 70,
          "bb_period": 20,
          "bb_dev": 2.0
        }
      },
      "exit_logic": {
        "name": "FixedRatioExitMixin",
        "params": {
          "profit_ratio": 1.5,
          "stop_loss_ratio": 0.5
        }
      },
      "position_size": 0.1
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
    "fromdate": null,
    "todate": null
  },

  "broker": {
    "cash": 10000.0,
    "commission": 0.001,
    "slippage": 0.0005
  },

  "analyzers": {
    "sharpe_ratio": {
      "enabled": true,
      "params": {
        "timeframe": "annual",
        "riskfreerate": 0.02
      }
    },
    "drawdown": {
      "enabled": true
    },
    "trades": {
      "enabled": true
    },
    "returns": {
      "enabled": true
    }
  },

  "assertions": {
    "min_trades": 1,
    "max_drawdown_pct": null,
    "min_sharpe_ratio": null,
    "final_value_greater_than_initial": false
  }
}
```

### 2. Run from Command Line

```bash
# Run a backtest directly
python src/backtester/tests/backtester_test_framework.py config/backtester/my_strategy_test.json

# Run all pytest tests
pytest src/backtester/tests/ -v

# Run a specific test
pytest src/backtester/tests/test_custom_strategy.py::test_custom_strategy_rsi_bb_fixed_ratio -v

# Run integration tests only
pytest src/backtester/tests/ -v -m integration
```

### 3. Run Programmatically

```python
from src.backtester.tests.backtester_test_framework import run_backtest_from_config

# Run backtest from config
results = run_backtest_from_config("config/backtester/my_strategy_test.json")

# Check if test passed
if results['success']:
    print("Test PASSED!")
else:
    print("Test FAILED!")
    print("Failures:", results['validation']['failures'])

# Print report
print(results['report'])
```

Or use the framework directly for more control:

```python
from src.backtester.tests.backtester_test_framework import BacktesterTestFramework

# Initialize framework
framework = BacktesterTestFramework("config/backtester/my_strategy_test.json")

# Setup backtest
framework.setup_backtest()

# Run backtest
results = framework.run_backtest()

# Validate results
validation = framework.validate_assertions()

# Generate report
report = framework.generate_report()
print(report)
```

## Configuration Reference

### Strategy Section

```json
"strategy": {
  "type": "CustomStrategy",           // Strategy class name
  "parameters": {
    "entry_logic": {
      "name": "RSIBBMixin",           // Entry mixin name
      "params": {                     // Mixin-specific parameters
        "rsi_period": 14,
        "rsi_oversold": 30
      }
    },
    "exit_logic": {
      "name": "FixedRatioExitMixin",  // Exit mixin name
      "params": {
        "profit_ratio": 1.5,
        "stop_loss_ratio": 0.5
      }
    },
    "position_size": 0.1              // Fraction of capital per trade
  }
}
```

### Available Entry Mixins

- `RSIBBMixin` - RSI + Bollinger Bands
- `RSIBBVolumeMixin` - RSI + BB + Volume
- `RSIIchimokuMixin` - RSI + Ichimoku Cloud
- `RSIOrBBMixin` - RSI OR BB logic
- `RSIVolumeSuperTrendMixin` - RSI + Volume + Supertrend
- `BBVolumeSuperTrendMixin` - BB + Volume + Supertrend
- `HMMLSTMEntryMixin` - ML-based entry

### Available Exit Mixins

- `FixedRatioExitMixin` - Fixed profit/stop loss ratios
- `TrailingStopExitMixin` - Trailing stop loss
- `ATRExitMixin` - ATR-based exits
- `SimpleATRExitMixin` - Simplified ATR
- `AdvancedATRExitMixin` - Enhanced ATR logic
- `TimeBasedExitMixin` - Time-based exits
- `MACrossoverExitMixin` - Moving average crossover
- `RSIBBExitMixin` - RSI + BB exit signals
- `RSIOrBBExitMixin` - RSI OR BB exit

### Data Section

```json
"data": {
  "file_path": "data/BTCUSDT_1h.csv",   // Path to CSV file
  "symbol": "BTCUSDT",                  // Symbol name
  "datetime_col": "timestamp",          // Datetime column name
  "open_col": "open",                   // OHLCV column names
  "high_col": "high",
  "low_col": "low",
  "close_col": "close",
  "volume_col": "volume",
  "fromdate": "2023-01-01",             // Optional: filter start date
  "todate": "2023-12-31"                // Optional: filter end date
}
```

### Broker Section

```json
"broker": {
  "cash": 10000.0,      // Initial cash
  "commission": 0.001,   // Commission rate (0.001 = 0.1%)
  "slippage": 0.0005    // Slippage (optional)
}
```

### Analyzers Section

```json
"analyzers": {
  "sharpe_ratio": {
    "enabled": true,
    "params": {
      "timeframe": "annual",
      "riskfreerate": 0.02
    }
  },
  "drawdown": {
    "enabled": true
  },
  "trades": {
    "enabled": true
  },
  "returns": {
    "enabled": true
  }
}
```

### Assertions Section

Define validation criteria for your backtest:

```json
"assertions": {
  "min_trades": 5,                      // Minimum number of trades
  "max_drawdown_pct": 30.0,             // Maximum drawdown percentage
  "min_sharpe_ratio": 1.0,              // Minimum Sharpe ratio
  "final_value_greater_than_initial": true  // Must be profitable
}
```

Set to `null` to skip a specific assertion.

## Output and Reports

### Console Output

When running backtests, you'll see:
- Test execution progress
- Performance metrics
- Trade statistics
- Validation results

### Report Files

Reports are saved to `results/backtester_tests/` with timestamps:
- Format: `{test_name}_{timestamp}.txt`
- Contains: Full performance metrics, trade statistics, validation results

### Example Report

```
================================================================================
BACKTESTER TEST REPORT
================================================================================

Test Name: CustomStrategy RSI+BB with Fixed Ratio Exit
Description: Backtester test for CustomStrategy using RSI+BB entry and Fixed Ratio exit
Generated: 2025-11-05 10:30:45

Strategy Configuration:
  Type: CustomStrategy
  Entry Logic: RSIBBMixin
    Params: {'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70}
  Exit Logic: FixedRatioExitMixin
    Params: {'profit_ratio': 1.5, 'stop_loss_ratio': 0.5}
  Position Size: 0.1

Performance Metrics:
--------------------------------------------------------------------------------
  Initial Value:    $10,000.00
  Final Value:      $12,500.00
  Total P&L:        $2,500.00
  Total Return:     25.00%

  Sharpe Ratio:     1.45
  Max Drawdown:     12.50%
  Max DD Period:    45 bars

Trade Statistics:
--------------------------------------------------------------------------------
  Total Trades:     15
  Winning Trades:   9
  Losing Trades:    6
  Win Rate:         60.00%
  Avg Win:          $450.00
  Avg Loss:         $-225.00
  Profit Factor:    2.00

Assertion Validation:
--------------------------------------------------------------------------------
  Overall Status:   PASSED
  Checks:
    ✓ min_trades: Expected >= 5, Got 15
    ✓ max_drawdown_pct: Expected <= 30.0%, Got 12.50%

================================================================================
```

## Testing with Pytest

### Test Structure

```python
import pytest
from src.backtester.tests.backtester_test_framework import BacktesterTestFramework

def test_my_strategy():
    """Test my custom strategy."""
    framework = BacktesterTestFramework("config/backtester/my_config.json")

    # Setup and run
    framework.setup_backtest()
    results = framework.run_backtest()

    # Validate
    validation = framework.validate_assertions()

    # Assertions
    assert results['total_trades'] > 0
    assert validation['passed']
```

### Pytest Markers

```bash
# Run all tests
pytest src/backtester/tests/ -v

# Run only integration tests
pytest src/backtester/tests/ -v -m integration

# Run with detailed output
pytest src/backtester/tests/ -v -s

# Run and stop on first failure
pytest src/backtester/tests/ -v -x
```

## Examples

### Example 1: RSI + Bollinger Bands Strategy

Config: `config/backtester/custom_strategy_test.json`

```bash
python src/backtester/tests/backtester_test_framework.py config/backtester/custom_strategy_test.json
```

### Example 2: RSI + Volume + Supertrend Strategy

Config: `config/backtester/rsi_volume_supertrend_test.json`

```bash
python src/backtester/tests/backtester_test_framework.py config/backtester/rsi_volume_supertrend_test.json
```

### Example 3: Custom Configuration

```python
import json
from src.backtester.tests.backtester_test_framework import run_backtest_from_config

# Create custom config
config = {
    "test_name": "My Custom Test",
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
        "datetime_col": "timestamp"
    },
    "broker": {
        "cash": 50000.0,
        "commission": 0.001
    },
    "assertions": {
        "min_trades": 3
    }
}

# Save config
with open("config/backtester/my_custom_test.json", "w") as f:
    json.dump(config, f, indent=2)

# Run test
results = run_backtest_from_config("config/backtester/my_custom_test.json")
print(results['report'])
```

## Troubleshooting

### Data File Not Found

Ensure your data file exists at the path specified in the config:
```bash
ls -la data/BTCUSDT_1h.csv
```

### Invalid Mixin Name

Check available mixins in:
- `src/strategy/entry/entry_mixin_factory.py` (ENTRY_MIXIN_REGISTRY)
- `src/strategy/exit/exit_mixin_factory.py` (EXIT_MIXIN_REGISTRY)

### Assertion Failures

Adjust assertions in your config or analyze why the strategy didn't meet expectations:
- Lower `min_trades` if not enough signals
- Increase `max_drawdown_pct` if drawdown is too high
- Lower `min_sharpe_ratio` for realistic expectations

### No Trades Generated ⚠️

**This is the most common issue!** If your backtest generates 0 trades, it means your entry conditions are too restrictive.

**Quick Diagnostic:**
```bash
# Run the debugger to analyze why no trades
python debug_no_trades.py config/backtester/your_config.json

# Or use the full debugger
python src/backtester/tests/backtest_debugger.py config/backtester/your_config.json --suggest
```

**Common Causes:**

1. **RSI threshold too low**
   - If RSI oversold is 30, price might never reach that level
   - Try increasing to 35-40 or check what RSI values actually occur in your data

2. **Volume threshold too high**
   - If requiring 1.5x average volume, this might be rare
   - Try lowering to 1.1x or 1.2x

3. **Multiple conditions with AND logic**
   - If requiring RSI oversold AND high volume AND BB touch, all three must happen simultaneously
   - Consider relaxing one condition or using OR logic where appropriate

4. **Date range too small**
   - Check if your date filter is limiting data too much
   - Ensure sufficient bars for indicators to warm up

5. **Bollinger Band deviation too high**
   - If BB deviation is 2.5 or 3.0, price might never touch lower band
   - Try reducing to 2.0 or 1.5

**Manual Check:**
```python
from src.backtester.tests.backtest_debugger import BacktestDebugger

debugger = BacktestDebugger("config/backtester/your_config.json")
debugger.load_data()
debugger.analyze_entry_conditions()
debugger.suggest_parameter_adjustments()
```

**Example Fix:**

If you have this config causing 0 trades:
```json
"entry_logic": {
  "name": "RSIBBVolumeMixin",
  "params": {
    "rsi_oversold": 25,           // Too low!
    "volume_threshold": 2.0,       // Too high!
    "bb_dev": 2.5                  // Too high!
  }
}
```

Try relaxing it:
```json
"entry_logic": {
  "name": "RSIBBVolumeMixin",
  "params": {
    "rsi_oversold": 40,            // More reasonable
    "volume_threshold": 1.2,       // Lower threshold
    "bb_dev": 2.0                  // Standard deviation
  }
}
```

## Advanced Usage

### Custom Assertions

You can add custom validation logic:

```python
from src.backtester.tests.backtester_test_framework import BacktesterTestFramework

framework = BacktesterTestFramework("config/backtester/my_config.json")
framework.setup_backtest()
results = framework.run_backtest()

# Custom validation
assert results['win_rate'] > 0.5, "Win rate should be > 50%"
assert results['profit_factor'] > 1.5, "Profit factor should be > 1.5"
```

### Batch Testing

Test multiple configurations:

```python
from pathlib import Path
from src.backtester.tests.backtester_test_framework import run_backtest_from_config

config_dir = Path("config/backtester")
all_results = []

for config_file in config_dir.glob("*.json"):
    print(f"\nTesting: {config_file.name}")
    try:
        result = run_backtest_from_config(str(config_file))
        all_results.append({
            'config': config_file.name,
            'return': result['results']['total_return'],
            'sharpe': result['results'].get('sharpe_ratio'),
            'trades': result['results'].get('total_trades'),
            'passed': result['success']
        })
    except Exception as e:
        print(f"Error: {e}")

# Analyze results
import pandas as pd
df = pd.DataFrame(all_results)
print("\nSummary:")
print(df.sort_values('return', ascending=False))
```

## Contributing

To add new strategies:

1. Add strategy class to `backtester_test_framework.py` in `_get_strategy_class()` method
2. Create JSON config in `config/backtester/`
3. Add pytest test in `test_custom_strategy.py`

## License

Part of the e-trading project.
