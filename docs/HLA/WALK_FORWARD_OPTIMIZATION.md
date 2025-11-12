# Walk-Forward Optimization System

**Document Version:** 1.0
**Last Updated:** 2025-11-12
**Author:** System Documentation
**Status:** Active

## Overview

The Walk-Forward Optimization system provides a robust framework for optimizing and validating trading strategies across multiple time periods while maintaining temporal integrity. The system now supports both manual window configuration and automatic data file discovery.

## Architecture

### Components

1. **Walk-Forward Optimizer** (`src/backtester/optimizer/walk_forward_optimizer.py`)
   - Orchestrates the optimization process
   - Manages training and testing windows
   - Handles multiple symbol/timeframe combinations
   - Supports auto-discovery of data files

2. **Custom Optimizer** (`src/backtester/optimizer/custom_optimizer.py`)
   - Executes individual optimization trials
   - Integrates with Optuna for hyperparameter optimization
   - Manages Backtrader cerebro instances

3. **Configuration System**
   - Auto-discovery mode: Automatically generates windows from data files
   - Manual mode: Explicit window definitions
   - Strategy filtering: Select specific entry/exit combinations

## Data-Driven Configuration (Auto-Discovery)

### Overview

The auto-discovery mode eliminates the need to manually configure training and testing windows. Simply place your data files in the `data/` directory, and the system automatically:

1. Scans for CSV files
2. Parses filenames to extract metadata
3. Groups by symbol and timeframe
4. Generates chronological window pairs

### File Naming Convention

Data files must follow this naming pattern:
```
SYMBOL_TIMEFRAME_STARTDATE_ENDDATE.csv
```

**Examples:**
- `BTCUSDT_4h_20200101_20201231.csv`
- `ETHUSDT_1h_20210101_20211231.csv`
- `LTCUSDT_15m_20220101_20221231.csv`

**Components:**
- `SYMBOL`: Trading pair (e.g., BTCUSDT, ETHUSDT)
- `TIMEFRAME`: Candle interval (e.g., 15m, 1h, 4h, 1d)
- `STARTDATE`: Start date in YYYYMMDD format
- `ENDDATE`: End date in YYYYMMDD format

### Configuration File

**Location:** `config/walk_forward/walk_forward_config.json`

**Simplified Structure:**
```json
{
  "window_type": "rolling",
  "auto_discover_data": true,
  "data_dir": "data",

  "symbols": ["BTCUSDT", "ETHUSDT", "LTCUSDT"],
  "timeframes": ["4h", "1h", "15m"],

  "entry_strategies": [
    "RSIOrBBEntryMixin",
    "RSIBBEntryMixin"
  ],
  "exit_strategies": [
    "SimpleATRExitMixin",
    "MACrossoverExitMixin"
  ],

  "optimizer_config_path": "config/optimizer/optimizer.json"
}
```

**Key Fields:**
- `auto_discover_data`: Enable/disable auto-discovery (true/false)
- `data_dir`: Directory containing data files (default: "data")
- `window_type`: Window generation strategy (see below)
- `symbols`: List of trading pairs to process
- `timeframes`: List of timeframes to process
- `entry_strategies`: Entry mixins to optimize
- `exit_strategies`: Exit mixins to optimize

## Window Types

### 1. Rolling Window (`"window_type": "rolling"`)

Each year trains on the previous year and tests on the current year.

**Example:**
```
Window 1: Train 2020 → Test 2021
Window 2: Train 2021 → Test 2022
Window 3: Train 2022 → Test 2023
```

**Use Case:** Best for detecting regime changes and ensuring strategies adapt to recent market conditions.

### 2. Expanding Window (`"window_type": "expanding"`)

Each year trains on ALL previous years and tests on the current year.

**Example:**
```
Window 1: Train 2020 → Test 2021
Window 2: Train 2020-2021 → Test 2022
Window 3: Train 2020-2022 → Test 2023
```

**Use Case:** Ideal for strategies that benefit from larger training datasets and long-term patterns.

### 3. Anchored Window (`"window_type": "anchored"`)

Trains on the first year only and tests on all subsequent years.

**Example:**
```
Window 1: Train 2020 → Test 2021
Window 2: Train 2020 → Test 2022
Window 3: Train 2020 → Test 2023
```

**Use Case:** Useful for testing strategy robustness across different market regimes with fixed training data.

## Auto-Discovery Workflow

### 1. Data File Scanning

```python
def auto_discover_windows(config: dict, data_dir: str = "data") -> list:
    """
    Scans data directory and generates window definitions.

    Process:
    1. Find all CSV files in data_dir
    2. Parse filenames using pattern: SYMBOL_TIMEFRAME_STARTDATE_ENDDATE.csv
    3. Filter by configured symbols and timeframes
    4. Group by (symbol, timeframe) pairs
    5. Sort chronologically by year
    6. Generate train/test windows based on window_type
    """
```

### 2. File Grouping

Files are grouped by `(symbol, timeframe)` combinations:

```
BTCUSDT/4h:
  - BTCUSDT_4h_20200101_20201231.csv (2020)
  - BTCUSDT_4h_20210101_20211231.csv (2021)
  - BTCUSDT_4h_20220101_20221231.csv (2022)

ETHUSDT/1h:
  - ETHUSDT_1h_20200101_20201231.csv (2020)
  - ETHUSDT_1h_20210101_20211231.csv (2021)
  - ETHUSDT_1h_20220101_20221231.csv (2022)
```

### 3. Window Generation

For each symbol/timeframe group, windows are generated according to the `window_type`:

**Generated Window Structure:**
```json
{
  "name": "2020_train_2021_test_BTCUSDT_4h",
  "train": ["BTCUSDT_4h_20200101_20201231.csv"],
  "test": ["BTCUSDT_4h_20210101_20211231.csv"],
  "train_year": "2020",
  "test_year": "2021",
  "symbol": "BTCUSDT",
  "timeframe": "4h"
}
```

## Running Walk-Forward Optimization

### Command

```bash
python src/backtester/optimizer/walk_forward_optimizer.py
```

### Execution Flow

1. **Load Configuration**
   - Read `walk_forward_config.json`
   - Check if `auto_discover_data` is enabled
   - Auto-generate windows or use manual configuration

2. **Validate Data**
   - Ensure all referenced files exist
   - Check temporal continuity
   - Validate symbol/timeframe combinations

3. **Optimize Windows**
   - For each window:
     - For each symbol/timeframe combination:
       - For each entry/exit strategy pair:
         - Run Optuna optimization (default: 100 trials)
         - Save best parameters and results

4. **Save Results**
   - Results saved to `results/walk_forward_reports/{year}/`
   - One JSON file per strategy combination
   - Includes best parameters, trades, and performance metrics

## Output Structure

### Results Directory

```
results/walk_forward_reports/
├── 2020/
│   ├── BTCUSDT_4h_RSIOrBBEntryMixin_SimpleATRExitMixin_20251112_143022.json
│   ├── BTCUSDT_4h_RSIOrBBEntryMixin_MACrossoverExitMixin_20251112_143045.json
│   └── ...
├── 2021/
│   └── ...
└── 2022/
    └── ...
```

### Result File Format

```json
{
  "data_file": "BTCUSDT_4h_20200101_20201231.csv",
  "window_name": "2020_train_2021_test_BTCUSDT_4h",
  "train_year": "2020",
  "symbol": "BTCUSDT",
  "timeframe": "4h",
  "total_trades": 45,
  "total_profit": 125.50,
  "total_profit_with_commission": 120.25,
  "total_commission": 5.25,
  "best_params": {
    "entry_logic": {
      "name": "RSIOrBBEntryMixin",
      "params": {
        "e_rsi_period": 18,
        "e_rsi_oversold": 28,
        "e_bb_period": 20,
        "e_bb_dev": 2.5
      }
    },
    "exit_logic": {
      "name": "SimpleATRExitMixin",
      "params": {
        "x_atr_period": 16,
        "x_atr_multiplier": 2.8
      }
    }
  },
  "analyzers": { ... },
  "trades": [ ... ]
}
```

## Strategy Configuration

### Entry Strategies

Entry strategy optimization parameters are defined in:
```
config/optimizer/entry/{MixinName}_{timeframe}.json
```

**Example:** `config/optimizer/entry/RSIOrBBEntryMixin_4h.json`

```json
{
  "name": "RSIOrBBEntryMixin",
  "timeframe": "4h",
  "params": {
    "e_rsi_period": {
      "type": "int",
      "low": 14,
      "high": 24,
      "default": 18
    },
    "e_rsi_oversold": {
      "type": "int",
      "low": 18,
      "high": 32,
      "default": 30
    }
  }
}
```

### Exit Strategies

Exit strategy parameters follow the same pattern:
```
config/optimizer/exit/{MixinName}_{timeframe}.json
```

## Performance Considerations

### Workload Calculation

**Formula:**
```
Total Backtests = Windows × Symbols × Timeframes × Entry Strategies × Exit Strategies × Trials per Combination

Example:
45 windows × 1 symbol × 1 timeframe × 6 entry × 9 exit × 100 trials = 243,000 backtests
```

**Estimated Time:**
- ~1 backtest per second
- 243,000 backtests ≈ 67-100 hours

### Optimization Strategies

1. **Limit Symbols/Timeframes:** Start with 1-2 combinations
2. **Reduce Trials:** Use 50 trials for initial testing
3. **Filter Strategies:** Test specific entry/exit pairs
4. **Parallel Processing:** Use `n_jobs=-1` in optimizer config
5. **Incremental Approach:** Run separate optimizations for each symbol

## Best Practices

### 1. Data Organization

- Use consistent file naming: `SYMBOL_TIMEFRAME_STARTDATE_ENDDATE.csv`
- Ensure contiguous yearly data (no gaps)
- Maintain separate directories for different data sources if needed

### 2. Configuration Management

- Start with auto-discovery for prototyping
- Use manual mode for production runs with specific windows
- Keep backup of configurations: `walk_forward_config.json.bkp`

### 3. Result Analysis

- Compare results across different window types
- Analyze strategy performance consistency across years
- Look for overfitting: high in-sample but poor out-of-sample performance

### 4. Computational Resources

- Start with small subsets (1 symbol, 1 timeframe)
- Monitor memory usage for large datasets
- Consider cloud compute for large-scale optimizations

## Troubleshooting

### Common Issues

**1. No windows generated**
- Verify data files exist in `data/` directory
- Check filename format matches pattern
- Ensure symbols/timeframes in config match file names

**2. Missing configuration files**
- Verify all entry/exit mixins have config files
- Check both timeframe-specific and generic configs exist
- Example: `RSIOrBBEntryMixin_4h.json` or `RSIOrBBEntryMixin.json`

**3. Out of memory errors**
- Reduce number of parallel jobs (`n_jobs`)
- Process fewer symbols/timeframes per run
- Use shorter time periods

**4. Slow optimization**
- Reduce number of trials (e.g., 50 instead of 100)
- Filter to specific strategy combinations
- Enable parallel processing if disabled

## Migration from Manual Mode

### Old Configuration (Manual)

```json
{
  "windows": [
    {
      "name": "2020_train_2021_test",
      "train": ["BTCUSDT_4h_20200101_20201231.csv"],
      "test": ["BTCUSDT_4h_20210101_20211231.csv"],
      "train_year": "2020",
      "test_year": "2021"
    }
  ],
  "symbols": ["BTCUSDT"],
  "timeframes": ["4h"]
}
```

### New Configuration (Auto-Discovery)

```json
{
  "window_type": "rolling",
  "auto_discover_data": true,
  "data_dir": "data",
  "symbols": ["BTCUSDT"],
  "timeframes": ["4h"]
}
```

**Benefits:**
- Automatically adapts to new data files
- No manual window configuration needed
- Supports multiple symbols/timeframes effortlessly
- Reduces configuration errors

## Future Enhancements

1. **Parallel Window Processing:** Optimize multiple windows simultaneously
2. **Result Aggregation:** Automatic summary reports across all windows
3. **Performance Tracking:** Time-series visualization of strategy evolution
4. **Smart Window Selection:** Skip windows with insufficient data quality
5. **Cloud Integration:** Support for distributed optimization

## Related Documentation

- [Optimizer Configuration](../config/optimizer/README.md)
- [Strategy Mixins](../src/strategy/README.md)
- [Backtest Analysis](./BACKTEST_ANALYSIS.md)
- [Performance Metrics](./PERFORMANCE_METRICS.md)

## Summary

The data-driven walk-forward optimization system provides a powerful, flexible framework for systematic strategy development. By automating window generation from data files, it significantly reduces configuration overhead while maintaining robust temporal validation.

**Key Advantages:**
✅ Zero manual window configuration
✅ Automatic adaptation to new data
✅ Support for multiple symbols and timeframes
✅ Flexible window strategies (rolling, expanding, anchored)
✅ Comprehensive result tracking and analysis

---

*For questions or issues, refer to the project's GitHub repository or contact the development team.*
