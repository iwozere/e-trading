# Backtester Quick Start 🚀

## TL;DR - Run Your First Backtest in 30 Seconds

### Option 1: IDE Debug Mode (Easiest!)

1. Open `src/backtester/tests/run_backtest.py`
2. Edit the top of the file:
   ```python
   DEBUG_MODE = True  # Already set!
   DEBUG_CONFIG = {
       'config_path': 'config/backtester/simple_test.json',  # Change this
       'generate_report': True,
       'verbose': True,
   }
   ```
3. Press **F5** (or Run button in your IDE)
4. Done! ✓

### Option 2: Command Line

```bash
python src/backtester/tests/run_backtest.py config/backtester/simple_test.json
```

## Available Test Configs

```bash
config/backtester/
├── simple_test.json                    # Guaranteed trades, good baseline
├── custom_strategy_test.json           # RSI+BB with Fixed Ratio exit
├── rsi_volume_supertrend_test.json     # RSI+Volume+Supertrend with ATR exit
└── trailing_stop_test.json             # BB+Volume+Supertrend with Trailing Stop
```

## Got 0 Trades? Debug It!

### Option 1: IDE Debug Mode

1. Open `src/backtester/tests/backtest_debugger.py`
2. Edit the top:
   ```python
   DEBUG_MODE = True
   DEBUG_CONFIG = {
       'config_path': 'config/backtester/YOUR_CONFIG.json',
       'suggest_adjustments': True,
       'report_path': None,
   }
   ```
3. Press **F5**
4. Follow the suggestions!

### Option 2: Command Line

```bash
python src/backtester/tests/backtest_debugger.py config/backtester/YOUR_CONFIG.json --suggest
```

## Create Your Own Config

Copy an existing config and modify:

```bash
cp config/backtester/simple_test.json config/backtester/my_test.json
# Edit my_test.json with your parameters
```

### Minimal Config Template

```json
{
  "test_name": "My Strategy Test",
  "strategy": {
    "type": "CustomStrategy",
    "parameters": {
      "entry_logic": {
        "name": "RSIBBMixin",
        "params": {"e_rsi_oversold": 40, "e_bb_dev": 2.0}
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

## Entry Mixins Cheat Sheet

```python
"RSIBBMixin"                  # RSI + Bollinger Bands
"RSIVolumeSuperTrendMixin"    # RSI + Volume + SuperTrend
"BBVolumeSuperTrendMixin"     # BB + Volume + SuperTrend
"RSIIchimokuMixin"            # RSI + Ichimoku Cloud
```

## Exit Mixins Cheat Sheet

```python
"FixedRatioExitMixin"         # Fixed profit/stop ratios
"TrailingStopExitMixin"       # Trailing stop loss
"ATRExitMixin"                # ATR-based stops
```

## Common Issues

### 0 Trades?
- **Quick Fix:** Use `simple_test.json` first to verify framework works
- **Then:** Run debugger to analyze your config
- **Usually:** Parameters too restrictive (RSI too low, volume too high)

### Data File Not Found?
- Check path in your config matches actual file location
- Default: `data/BTCUSDT_1h.csv`

### Mixin Not Found?
- Check spelling and capitalization
- Available mixins listed in cheat sheet above

## IDE Setup (One-Time)

### VS Code / Cursor

Just press F5 when `run_backtest.py` is open. It will work!

### PyCharm

Just press Shift+F10 when `run_backtest.py` is open. It will work!

## Next Steps

1. ✅ Run `simple_test.json` to verify setup
2. ✅ Copy and modify for your strategy
3. ✅ Use debugger if 0 trades
4. ✅ Iterate on parameters
5. ✅ Celebrate when it works! 🎉

## Full Documentation

- [README.md](README.md) - Complete framework documentation
- [DEBUG_MODE_GUIDE.md](DEBUG_MODE_GUIDE.md) - Detailed debug mode guide
- [BACKTESTER_SETUP.md](../../../BACKTESTER_SETUP.md) - Full setup guide

## Example: Complete Workflow

```python
# 1. Set your config in run_backtest.py
DEBUG_CONFIG = {
    'config_path': 'config/backtester/my_test.json',
    'generate_report': True,
    'verbose': True,
}

# 2. Press F5 - if 0 trades, go to step 3

# 3. Set same config in backtest_debugger.py
DEBUG_CONFIG = {
    'config_path': 'config/backtester/my_test.json',
    'suggest_adjustments': True,
    'report_path': None,
}

# 4. Press F5 - get suggestions

# 5. Update my_test.json with suggestions

# 6. Back to run_backtest.py, press F5

# 7. Profit! 💰
```

That's it! Start testing your strategies! 🚀
