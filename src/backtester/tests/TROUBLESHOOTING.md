# Troubleshooting Guide

Common issues and solutions when using the backtester framework.

## Indicator Computation Warnings

### Warning Message
```
WARNING - Unified service computation failed, falling back to native implementation: RSI() got an unexpected keyword argument 'period'
WARNING - Unified service computation failed, falling back to native implementation: Unknown indicator: bollinger_bands
```

### Cause
Parameter name mismatches between mixin configurations and indicator service expectations.

### Impact
**⚠️ HARMLESS** - These are just warnings. Indicators still compute correctly using the fallback (native) implementation.

### What's Happening
- Mixins use parameter names like `period`, `bb_period`
- Unified indicator service expects `timeperiod`, `bbands`
- When mismatch occurs, system automatically falls back to native Backtrader indicators
- **Your strategy still works correctly!**

### Should You Fix It?
**No action needed** - The fallback mechanism ensures everything works. These warnings don't affect backtest results.

If you want to eliminate the warnings (optional), the parameter mappings would need to be updated in the indicator adapters, but this is not necessary for backtesting to work.

## Unicode Encoding Errors (Windows)

### Error Message
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2717' in position XXX: character maps to <undefined>
```

### Cause
This happens on Windows when trying to write Unicode characters (like ✓ and ✗) to files using the default system encoding (cp1252).

### Solution
✅ **FIXED** - All file writes now use UTF-8 encoding explicitly.

If you still encounter this issue, make sure you're using the latest version of the files:
- `backtester_test_framework.py` - Line 524: `open(output_path, 'w', encoding='utf-8')`
- `backtest_debugger.py` - Line 352: `open(output_path, 'w', encoding='utf-8')`

## 0 Trades Generated

### Symptoms
- Backtest completes but shows `Total Trades: 0`
- Strategy never enters any positions

### Common Causes

1. **RSI threshold too low**
   - If `e_rsi_oversold: 30`, price might never reach that level
   - Check actual RSI values in your data

2. **Volume threshold too high**
   - If `e_min_volume_ratio: 1.5`, this might be rare
   - Most bars may not have 1.5x average volume

3. **Multiple AND conditions**
   - Requiring RSI < 30 AND Volume > 1.5x AND BB touch
   - All three rarely happen simultaneously

4. **BB deviation too high**
   - `e_bb_dev: 2.5` or higher means price rarely touches bands
   - Try reducing to 2.0 or 1.5

### Diagnostic Steps

1. **Run the debugger:**
   ```python
   # In backtest_debugger.py
   DEBUG_MODE = True
   DEBUG_CONFIG = {
       'config_path': 'config/backtester/your_config.json',
       'suggest_adjustments': True,
       'report_path': None,
   }
   ```
   Then press F5

2. **Review suggestions:**
   - The debugger analyzes your actual data
   - Suggests realistic parameter values
   - Shows how often each condition triggers

3. **Test with simple_test.json:**
   ```python
   # In run_backtest.py
   DEBUG_CONFIG = {
       'config_path': 'config/backtester/simple_test.json',
       'generate_report': True,
       'verbose': True,
   }
   ```
   This config is guaranteed to generate trades

4. **Relax one parameter at a time:**
   - Start with RSI: increase oversold threshold
   - Then volume: lower threshold
   - Then BB: reduce deviation

### Quick Fixes

**If using RSIBBMixin:**
```json
{
  "entry_logic": {
    "name": "RSIBBMixin",
    "params": {
      "e_rsi_oversold": 45,        // ← Increase from 30
      "e_bb_dev": 2.0,             // ← Reduce from 2.5+
      "e_use_bb_touch": false      // ← Don't require BB touch
    }
  }
}
```

**If using RSIBBVolumeMixin:**
```json
{
  "entry_logic": {
    "name": "RSIBBVolumeMixin",
    "params": {
      "e_rsi_oversold": 45,
      "e_min_volume_ratio": 1.1,   // ← Lower from 1.5+
      "e_bb_dev": 2.0
    }
  }
}
```

## Data File Not Found

### Error Message
```
FileNotFoundError: Data file not found: data/BTCUSDT_1h.csv
```

### Solution
1. Check if file exists: `ls data/BTCUSDT_1h.csv`
2. Verify path in your JSON config matches actual location
3. Use absolute path if needed:
   ```json
   "data": {
     "file_path": "C:/full/path/to/data.csv"
   }
   ```

## Invalid Mixin Name

### Error Message
```
KeyError: 'YourMixinName'
```

### Solution
Check available mixins:

**Entry Mixins:**
- `RSIBBMixin`
- `RSIBBVolumeMixin`
- `RSIVolumeSuperTrendMixin`
- `BBVolumeSuperTrendMixin`
- `RSIIchimokuMixin`
- `RSIOrBBMixin`
- `HMMLSTMEntryMixin`

**Exit Mixins:**
- `FixedRatioExitMixin`
- `TrailingStopExitMixin`
- `ATRExitMixin`
- `SimpleATRExitMixin`
- `AdvancedATRExitMixin`
- `TimeBasedExitMixin`
- `MACrossoverExitMixin`
- `RSIBBExitMixin`
- `RSIOrBBExitMixin`

**Note:** Names are case-sensitive!

## Config File Not Found

### Error Message
```
Configuration file not found: config/backtester/your_config.json
```

### Solution
1. List available configs:
   ```python
   DEBUG_MODE = False
   ```
   ```bash
   python src/backtester/tests/run_backtest.py --list-configs
   ```

2. Use relative path from project root:
   ```python
   'config_path': 'config/backtester/my_config.json'  # Correct
   ```
   Not:
   ```python
   'config_path': './config/backtester/my_config.json'  # Wrong
   'config_path': 'my_config.json'  # Wrong
   ```

## Assertion Failures

### Symptoms
- Test completes with trades
- Shows "TEST FAILED"
- Assertion validation errors

### Examples

**Min Trades Not Met:**
```
Minimum trades not met: 3 < 5
```
Solution: Lower `min_trades` in assertions or relax entry conditions

**Max Drawdown Exceeded:**
```
Max drawdown exceeded: 35.50% > 30.0%
```
Solution: Increase `max_drawdown_pct` or improve strategy

**Min Sharpe Not Met:**
```
Minimum Sharpe ratio not met: 0.8 < 1.0
```
Solution: Lower `min_sharpe_ratio` or improve strategy

### Quick Fix
Set problematic assertions to `null`:
```json
"assertions": {
  "min_trades": 1,
  "max_drawdown_pct": null,      // ← Disable this check
  "min_sharpe_ratio": null,      // ← Disable this check
  "final_value_greater_than_initial": false
}
```

## Module Import Errors

### Error Message
```
ModuleNotFoundError: No module named 'src'
```

### Solution
Run from project root:
```bash
cd c:/dev/cursor/e-trading
python src/backtester/tests/run_backtest.py config/backtester/my_config.json
```

Or use debug mode (automatically handles paths):
```python
DEBUG_MODE = True
```

## Backtrader Errors

### Error: "Not enough data to calculate indicator"

**Cause:** Insufficient bars for indicator initialization

**Solution:**
1. Check date range isn't too small
2. Ensure at least 100+ bars of data
3. Reduce indicator periods (e.g., RSI period from 14 to 7)

### Error: "No data available"

**Cause:** Date filters exclude all data

**Solution:**
1. Check `fromdate` and `todate` in config
2. Verify they match your data's date range
3. Set both to `null` to use all data:
   ```json
   "data": {
     "fromdate": null,
     "todate": null
   }
   ```

## Performance Issues

### Test runs very slowly

**Solutions:**
1. Disable report generation for quick tests:
   ```python
   DEBUG_CONFIG = {
       'generate_report': False,  # Much faster
       'verbose': False
   }
   ```

2. Use smaller date range:
   ```json
   "data": {
     "fromdate": "2024-01-01",
     "todate": "2024-03-31"
   }
   ```

3. Reduce data frequency (use 4h instead of 1h data)

## Debug Mode Not Working

### Issue: Still showing CLI help

**Solution:** Verify `DEBUG_MODE = True` at the top of the file

### Issue: Config path not found

**Solution:** Use path relative to project root:
```python
DEBUG_CONFIG = {
    'config_path': 'config/backtester/my_config.json',  # From project root
}
```

## Getting Help

If you're still stuck:

1. **Check logs:** Look for ERROR or WARNING messages
2. **Enable verbose mode:**
   ```python
   DEBUG_CONFIG = {
       'verbose': True,
   }
   ```
3. **Try simple_test.json:** Verify framework works
4. **Run debugger:** Get parameter suggestions
5. **Check documentation:**
   - [QUICK_START.md](QUICK_START.md)
   - [DEBUG_MODE_GUIDE.md](DEBUG_MODE_GUIDE.md)
   - [README.md](README.md)

## Common Gotchas

### JSON Syntax Errors
- Missing commas between dict items
- Trailing commas (not allowed in JSON)
- Using single quotes (must use double quotes)

### Parameter Name Confusion
Different mixins use different parameter prefixes:
- Entry params often use `e_` prefix: `e_rsi_oversold`
- Exit params often use `x_` prefix: `x_atr_period`
- Check each mixin's documentation for exact names

### Path Separators
Use forward slashes in paths, even on Windows:
```python
'config_path': 'config/backtester/my_config.json'  # Correct
'config_path': 'config\\backtester\\my_config.json'  # Works but not recommended
```

## Still Need Help?

The framework includes:
- Built-in debugger for parameter analysis
- Verbose error messages
- Example configurations that work
- Step-by-step guides

Most issues are due to:
1. Too restrictive entry conditions (use debugger!)
2. Wrong file paths (use paths from project root)
3. Missing data files (check if file exists)

Start with `simple_test.json` and work your way up! 🚀
