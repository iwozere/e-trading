# Debug Mode Guide for Backtester Tests

This guide explains how to run backtest scripts from your IDE debugger with parameters set directly in the code.

## Overview

Both `run_backtest.py` and `backtest_debugger.py` support two modes:
- **CLI Mode**: Traditional command-line usage with arguments
- **Debug Mode**: Set parameters at the top of the file and run directly from IDE

## Why Debug Mode?

When developing and debugging strategies, it's much easier to:
- Set a breakpoint and step through code
- Quickly change parameters without command-line arguments
- Run the same test repeatedly with consistent settings
- Use your IDE's debugger features

## Quick Start

### 1. Running Backtests in Debug Mode

**File:** `src/backtester/tests/run_backtest.py`

1. Open `run_backtest.py` in your IDE
2. Find the `DEBUG_CONFIG` section at the top:

```python
# =============================================================================
# DEBUG CONFIGURATION
# Set these parameters when running from IDE debugger
# =============================================================================
DEBUG_MODE = True  # Set to True to use DEBUG_CONFIG, False for CLI mode
DEBUG_CONFIG = {
    # Path to your config file (relative to project root or absolute)
    'config_path': 'config/backtester/custom_strategy_test.json',

    # Generate report file? (True/False)
    'generate_report': True,

    # Enable verbose output? (True/False)
    'verbose': True,
}
# =============================================================================
```

3. Update the parameters:
   - `config_path`: Path to your JSON config (relative to project root)
   - `generate_report`: Set to `True` to save report, `False` to skip
   - `verbose`: Set to `True` for detailed error messages

4. Run the script from your IDE (F5 in VSCode, Shift+F10 in PyCharm, etc.)

### 2. Debugging "No Trades" Issues

**File:** `src/backtester/tests/backtest_debugger.py`

1. Open `backtest_debugger.py` in your IDE
2. Find the `DEBUG_CONFIG` section at the top:

```python
# =============================================================================
# DEBUG CONFIGURATION
# Set these parameters when running from IDE debugger
# =============================================================================
DEBUG_MODE = True  # Set to True to use DEBUG_CONFIG, False for CLI mode
DEBUG_CONFIG = {
    # Path to your config file (relative to project root or absolute)
    'config_path': 'config/backtester/custom_strategy_test.json',

    # Suggest parameter adjustments? (True/False)
    'suggest_adjustments': True,

    # Save debug report to file? (None or path string)
    'report_path': None,  # e.g., 'results/debug_report.txt'
}
# =============================================================================
```

3. Update the parameters:
   - `config_path`: Your config file path
   - `suggest_adjustments`: Set to `True` to get parameter suggestions
   - `report_path`: Optional file path to save debug report

4. Run the script from your IDE

## Examples

### Example 1: Quick Backtest Run

```python
# In run_backtest.py
DEBUG_MODE = True
DEBUG_CONFIG = {
    'config_path': 'config/backtester/simple_test.json',
    'generate_report': False,  # Skip report for quick testing
    'verbose': True,
}
```

Then press F5 (or your IDE's run shortcut).

### Example 2: Full Analysis with Report

```python
# In run_backtest.py
DEBUG_MODE = True
DEBUG_CONFIG = {
    'config_path': 'config/backtester/rsi_volume_supertrend_test.json',
    'generate_report': True,  # Save detailed report
    'verbose': True,
}
```

### Example 3: Debug No Trades Issue

```python
# In backtest_debugger.py
DEBUG_MODE = True
DEBUG_CONFIG = {
    'config_path': 'config/backtester/my_failing_config.json',
    'suggest_adjustments': True,  # Get parameter suggestions
    'report_path': 'results/debug_analysis.txt',  # Save report
}
```

### Example 4: Using Absolute Paths

```python
# You can also use absolute paths
DEBUG_CONFIG = {
    'config_path': 'C:/dev/cursor/e-trading/config/backtester/custom_strategy_test.json',
    'generate_report': True,
    'verbose': True,
}
```

## Common Workflows

### Workflow 1: Iterative Strategy Development

1. Set your config in `run_backtest.py` debug mode
2. Run the backtest
3. If 0 trades, switch to `backtest_debugger.py`
4. Review suggestions
5. Update your JSON config
6. Return to step 2

### Workflow 2: Quick Parameter Testing

```python
# run_backtest.py - Test config A
DEBUG_CONFIG = {
    'config_path': 'config/backtester/test_a.json',
    'generate_report': False,
    'verbose': False,
}
# Run, note results

# Change config
DEBUG_CONFIG = {
    'config_path': 'config/backtester/test_b.json',
    'generate_report': False,
    'verbose': False,
}
# Run again, compare
```

### Workflow 3: Deep Debugging with Breakpoints

1. Set `DEBUG_MODE = True` in `run_backtest.py`
2. Set your config path
3. Set a breakpoint in `backtester_test_framework.py` (e.g., in `run_backtest()` method)
4. Run with debugger (F5)
5. Step through code to see exactly what's happening

## Switching Between Modes

### To Use CLI Mode

Set `DEBUG_MODE = False` at the top of the file:

```python
DEBUG_MODE = False  # Now uses command-line arguments
```

Then run from terminal:
```bash
python src/backtester/tests/run_backtest.py config/backtester/my_config.json
```

### To Use Debug Mode

Set `DEBUG_MODE = True`:

```python
DEBUG_MODE = True  # Now uses DEBUG_CONFIG
```

Then run from IDE (F5).

## IDE-Specific Instructions

### Visual Studio Code

1. Open `run_backtest.py`
2. Set `DEBUG_MODE = True` and configure `DEBUG_CONFIG`
3. Press `F5` or click "Run" → "Start Debugging"
4. Or press `Ctrl+F5` to run without debugging

**Setting Up Launch Configuration (Optional):**

Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run Backtest",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/backtester/tests/run_backtest.py",
            "console": "integratedTerminal"
        },
        {
            "name": "Debug No Trades",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/backtester/tests/backtest_debugger.py",
            "console": "integratedTerminal"
        }
    ]
}
```

### PyCharm

1. Open `run_backtest.py`
2. Set `DEBUG_MODE = True` and configure `DEBUG_CONFIG`
3. Right-click in the editor → "Run 'run_backtest'"
4. Or press `Shift+F10` to run, `Shift+F9` to debug

### Cursor

1. Open `run_backtest.py`
2. Set `DEBUG_MODE = True` and configure `DEBUG_CONFIG`
3. Use the same shortcuts as VS Code (F5 to debug)

## Tips and Best Practices

### Tip 1: Keep DEBUG_MODE = True During Development

Leave debug mode on while developing. It's faster and easier than switching to CLI.

### Tip 2: Use Simple Config First

If you're getting 0 trades, test with `simple_test.json` first:

```python
DEBUG_CONFIG = {
    'config_path': 'config/backtester/simple_test.json',  # Known working config
    'generate_report': False,
    'verbose': True,
}
```

### Tip 3: Use Verbose Mode for Errors

Always enable verbose when debugging errors:

```python
DEBUG_CONFIG = {
    'config_path': 'config/backtester/my_config.json',
    'generate_report': True,
    'verbose': True,  # Get full stack traces
}
```

### Tip 4: Comment Your Debug Configs

Keep notes about what you're testing:

```python
DEBUG_CONFIG = {
    # Testing RSI=50 with relaxed parameters
    'config_path': 'config/backtester/rsi_50_test.json',
    'generate_report': True,
    'verbose': True,
}
```

### Tip 5: Version Control Debug Settings

Add debug configs to `.gitignore` if they contain sensitive paths:

```gitignore
# .gitignore
*_debug_config.py
```

Or just remember to set `DEBUG_MODE = False` before committing.

## Troubleshooting

### Issue: "Config file not found"

**Solution:** Make sure your path is relative to project root:

```python
# ✓ Correct (relative to project root)
'config_path': 'config/backtester/my_config.json'

# ✗ Wrong
'config_path': 'my_config.json'  # Not in root
'config_path': './config/backtester/my_config.json'  # ./ is unnecessary
```

### Issue: Changes Not Taking Effect

**Solution:** Make sure `DEBUG_MODE = True` and you've saved the file.

### Issue: "No module named 'src'"

**Solution:** Run from project root or ensure the path setup is correct. The script automatically adds project root to path.

### Issue: Still Getting CLI Help

**Solution:** Verify `DEBUG_MODE = True` is actually set. Don't confuse CLI mode with debug mode.

## Example: Complete Debug Session

Here's a complete example of debugging a failing strategy:

### Step 1: Initial Run

```python
# run_backtest.py
DEBUG_MODE = True
DEBUG_CONFIG = {
    'config_path': 'config/backtester/my_strategy.json',
    'generate_report': True,
    'verbose': True,
}
```

**Output:** 0 trades, test failed

### Step 2: Analyze with Debugger

```python
# backtest_debugger.py
DEBUG_MODE = True
DEBUG_CONFIG = {
    'config_path': 'config/backtester/my_strategy.json',
    'suggest_adjustments': True,
    'report_path': None,
}
```

**Output:**
- RSI oversold=30, never reached (min was 35)
- Volume threshold=1.8, rarely exceeded
- Suggestion: Increase RSI to 40, lower volume to 1.2

### Step 3: Update Config

Edit `config/backtester/my_strategy.json`:
```json
{
  "entry_logic": {
    "params": {
      "e_rsi_oversold": 40,
      "e_min_volume_ratio": 1.2
    }
  }
}
```

### Step 4: Rerun

```python
# run_backtest.py (same config as Step 1)
DEBUG_MODE = True
DEBUG_CONFIG = {
    'config_path': 'config/backtester/my_strategy.json',
    'generate_report': True,
    'verbose': True,
}
```

**Output:** 8 trades, test passed!

## Summary

Debug mode makes backtesting development much faster:

✅ No need to type command-line arguments
✅ Easy to change parameters
✅ Works perfectly with IDE debuggers
✅ Set breakpoints and step through code
✅ Consistent test runs

Just set `DEBUG_MODE = True`, configure your parameters, and press F5!
