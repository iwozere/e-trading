# Walk-Forward Optimization & Validation Guide

## Overview

This guide explains how to use the walk-forward optimization and validation framework to test trading strategies robustly and avoid overfitting.

## What is Walk-Forward Optimization?

Walk-forward optimization is a technique that:
1. **Optimizes** strategy parameters on historical data (in-sample / IS)
2. **Tests** those parameters on future data (out-of-sample / OOS)
3. **Compares** IS vs OOS performance to identify overfitting
4. **Repeats** the process across multiple time windows

This prevents the common mistake of optimizing on all available data and expecting those results to hold in live trading.

## Directory Structure

```
config/walk_forward/
└── walk_forward_config.json     # Configuration for windows and symbols

data/_all/                        # Your yearly data files
├── BTCUSDT_1h_20220101_20221231.csv
├── BTCUSDT_1h_20230101_20231231.csv
└── ...

results/
├── optimization/                 # In-sample optimization results
│   ├── 2022/
│   ├── 2023/
│   └── 2024/
├── validation/                   # Out-of-sample validation results
│   ├── 2023/                    # Using 2022 params
│   ├── 2024/                    # Using 2023 params
│   └── 2025/                    # Using 2024 params
└── walk_forward_reports/        # Comparison reports
    ├── performance_comparison.csv
    ├── degradation_analysis.json
    └── robustness_summary.csv

src/backtester/
├── optimizer/
│   └── walk_forward_optimizer.py    # Step 1: IS optimization
├── validator/
│   ├── walk_forward_validator.py    # Step 2: OOS validation
│   └── performance_comparer.py      # Step 3: Comparison & reports
└── plotter/
    ├── run_plotter.py               # Plot all strategies (use sparingly)
    └── plot_top_strategies.py       # Step 4: Plot top performers (recommended)
```

## Quick Start

### Step 1: Configure Your Windows

Edit `config/walk_forward/walk_forward_config.json`:

```json
{
  "window_type": "rolling",
  "windows": [
    {
      "name": "2022_train_2023_test",
      "train": ["BTCUSDT_1h_20220101_20221231.csv"],
      "test": ["BTCUSDT_1h_20230101_20231231.csv"],
      "train_year": "2022",
      "test_year": "2023"
    }
  ],
  "symbols": ["BTCUSDT"],
  "timeframes": ["1h"],
  "optimizer_config_path": "config/optimizer/optimizer.json"
}
```

### Step 2: Run In-Sample Optimization

```bash
python src/backtester/optimizer/walk_forward_optimizer.py
```

This will:
- Load training data for each window
- Run optimization for all entry/exit strategy combinations
- Save best parameters to `results/optimization/{year}/`

**Expected Output:**
```
results/optimization/2022/
├── BTCUSDT_1h_20220101_20221231_RSIBBEntryMixin_TrailingStopExitMixin_20251109_143022.json
├── BTCUSDT_1h_20220101_20221231_MACDEntryMixin_ATRStopExitMixin_20251109_143145.json
└── ...
```

### Step 3: Run Out-of-Sample Validation

```bash
python src/backtester/validator/walk_forward_validator.py
```

This will:
- Load IS optimization results from Step 2
- Extract best parameters for each strategy
- Run backtests on OOS data **without re-optimization**
- Save OOS results to `results/walk_forward_reports/validation/{year}/`

**Expected Output:**
```
results/walk_forward_reports/validation/2023/
├── BTCUSDT_1h_20230101_20231231_RSIBBEntryMixin_TrailingStopExitMixin_OOS_20251109_150022.json
├── BTCUSDT_1h_20230101_20231231_MACDEntryMixin_ATRStopExitMixin_OOS_20251109_150145.json
└── ...
```

### Step 4: Generate Comparison Reports

```bash
python src/backtester/validator/performance_comparer.py
```

This will:
- Compare IS vs OOS performance for all strategies
- Calculate degradation metrics
- Generate 3 reports (CSV + JSON)

**Expected Output:**
```
results/walk_forward_reports/
├── performance_comparison.csv      # Detailed IS/OOS comparison
├── degradation_analysis.json       # Summary statistics
└── robustness_summary.csv          # Aggregated metrics
```

### Step 5: Visualize Top Strategies (Optional)

After generating comparison reports, you can create plots for the best performing strategies.

#### Option A: Plot Top Strategies Automatically

```bash
python src/backtester/plotter/plot_top_strategies.py
```

This will:
- Read `performance_comparison.csv`
- Select top 5 strategies by IS profit
- Select top 5 strategies by robustness score
- Create plots for each strategy with indicators, trades, and equity curve
- Generate a summary report

**Expected Output:**
```
results/plots/top_strategies/
├── top_is_profit/                  # Top 5 by in-sample profit
│   ├── BTCUSDT_4h_RSI_ATR_2021.png
│   ├── ETHUSDT_1h_BB_MACross_2020.png
│   └── ...
├── top_robustness/                 # Top 5 by robustness score
│   ├── LTCUSDT_4h_RSIVolume_FixedRatio_2022.png
│   └── ...
└── selection_summary.txt           # Details of selected strategies
```

#### Option B: Plot All Optimization Results

If you want to plot ALL optimization results (not just top performers):

```bash
python src/backtester/plotter/run_plotter.py
```

**Note:** This will create plots for EVERY strategy result, which can be hundreds of files. Use this only if you need comprehensive visualization.

**Recommended:** Use **Option A** (`plot_top_strategies.py`) to focus on the most promising strategies.

## Understanding the Reports

### 1. performance_comparison.csv

Open in Excel to see side-by-side IS/OOS comparison for every strategy.

**Key Columns:**
- `is_total_profit` vs `oos_total_profit`: Profit comparison
- `profit_degradation_ratio`: OOS_Profit / IS_Profit (ideally > 0.6)
- `profit_degradation_pct`: Percentage decrease from IS to OOS
- `overfitting_score`: 0-1 scale (higher = more overfitting)
- `robustness_score`: 0-1 scale (higher = more robust)

**How to Use:**
1. Sort by `oos_total_profit` (descending) to see best OOS performers
2. Filter for `robustness_score > 0.5` to exclude overfitted strategies
3. Check `profit_degradation_ratio` - values < 0.5 indicate poor generalization

### 2. degradation_analysis.json

Aggregate statistics and insights.

**Key Sections:**
- `summary`: Overall statistics across all strategies
- `by_window`: Performance per time window
- `top_strategies_by_oos_profit`: Best OOS performers
- `most_robust_strategies`: Strategies with highest robustness scores
- `warning_flags`: Strategies with severe overfitting or failures

**Example:**
```json
{
  "summary": {
    "total_strategies": 120,
    "avg_profit_degradation_ratio": 0.68,
    "strategies_with_positive_oos": 85,
    "high_robustness_strategies": 28
  },
  "top_strategies_by_oos_profit": [
    {
      "strategy_id": "BTCUSDT_1h_RSIBBEntryMixin_TrailingStopExitMixin",
      "oos_total_profit": 790.15,
      "robustness_score": 0.58
    }
  ]
}
```

### 3. robustness_summary.csv

Aggregated metrics per strategy **across all windows**.

**Key Columns:**
- `avg_oos_profit`: Average OOS profit across all windows
- `std_oos_profit`: Standard deviation (consistency measure)
- `avg_degradation_ratio`: Average IS→OOS degradation
- `consistency_score`: How stable profits are across windows
- `overall_robustness_score`: Combined robustness metric
- `recommendation`: "Excellent" / "Good" / "Fair" / "Poor" / "Reject"

**How to Use:**
1. Filter for `recommendation = "Excellent"` or `"Good"`
2. Sort by `overall_robustness_score` (descending)
3. Select top 3-5 strategies for live paper trading

## Interpreting Metrics

### Degradation Ratio (OOS/IS Profit)

| Range | Interpretation | Action |
|-------|----------------|--------|
| 0.8 - 1.2 | Excellent - strategy maintains performance | Use with confidence |
| 0.6 - 0.8 | Good - acceptable degradation | Consider using |
| 0.4 - 0.6 | Fair - significant degradation | Use cautiously |
| < 0.4 | Poor - likely overfit | Avoid |
| Negative | Failed - OOS losses | Reject |

### Robustness Score

| Range | Interpretation |
|-------|----------------|
| > 0.7 | Highly robust |
| 0.5 - 0.7 | Moderately robust |
| 0.3 - 0.5 | Marginally robust |
| < 0.3 | Not robust (overfit) |

### Warning Flags

The `degradation_analysis.json` includes automatic warnings for:
- **Profit degradation > 50%**: Strategy likely overfit
- **OOS losses when IS profitable**: Strategy completely failed
- **Large variance across windows**: Unstable/unreliable

## Advanced Configuration

### Rolling vs Expanding Windows

**Rolling Window** (default):
```json
{
  "window_type": "rolling",
  "windows": [
    {"train": ["2022"], "test": ["2023"]},
    {"train": ["2023"], "test": ["2024"]},
    {"train": ["2024"], "test": ["2025"]}
  ]
}
```
- Fixed 1-year training window
- Better for adapting to recent market conditions
- **Recommended for crypto** (markets change quickly)

**Expanding Window**:
```json
{
  "window_type": "expanding",
  "windows": [
    {"train": ["2022"], "test": ["2023"]},
    {"train": ["2022", "2023"], "test": ["2024"]},
    {"train": ["2022", "2023", "2024"], "test": ["2025"]}
  ]
}
```
- Cumulative training data
- More data = better statistical significance
- Better for stable markets

### Multiple Symbols and Timeframes

The framework processes all combinations:

```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", "LTCUSDT"],
  "timeframes": ["1h", "4h"]
}
```

This creates 6 combinations (3 symbols × 2 timeframes) per window.

## Workflow Examples

### Example 1: Initial Strategy Discovery

```bash
# 1. Configure 3 windows (2022→2023, 2023→2024, 2024→2025)
nano config/walk_forward/walk_forward_config.json

# 2. Run full pipeline
python src/backtester/optimizer/walk_forward_optimizer.py
python src/backtester/validator/walk_forward_validator.py
python src/backtester/validator/performance_comparer.py
python src/backtester/plotter/plot_top_strategies.py

# 3. Review top strategies
cat results/walk_forward_reports/degradation_analysis.json | grep -A 5 "top_strategies"
cat results/plots/top_strategies/selection_summary.txt

# 4. Open CSV in Excel and filter for robustness_score > 0.6

# 5. View plots in results/plots/top_strategies/
```

### Example 2: Test Single Strategy

To test just one strategy combination, edit the mixin registries or run with a smaller config:

```json
{
  "symbols": ["BTCUSDT"],
  "timeframes": ["1h"],
  "windows": [{"train": ["2022"], "test": ["2023"]}]
}
```

Then temporarily edit the entry/exit registries to include only your target strategy.

### Example 3: Periodic Re-validation

When new data arrives (e.g., 2025 data):

```json
{
  "windows": [
    ...existing windows...,
    {
      "name": "2025_train_2026_test",
      "train": ["BTCUSDT_1h_20250101_20251231.csv"],
      "test": ["BTCUSDT_1h_20260101_20261231.csv"],
      "train_year": "2025",
      "test_year": "2026"
    }
  ]
}
```

Run the pipeline again to see if previously robust strategies maintain performance.

## Best Practices

### 1. Start Small
- Begin with 1 symbol, 1 timeframe, 1 window
- Verify the pipeline works end-to-end
- Then scale up

### 2. Data Quality
- Ensure data files are clean and complete
- Check for gaps or anomalies
- Use consistent naming: `{SYMBOL}_{TIMEFRAME}_{STARTDATE}_{ENDDATE}.csv`

### 3. Sufficient Data
- Minimum 1 year of training data
- Minimum 6 months of test data
- More data = better statistical significance

### 4. Strategy Selection
- Don't just pick the highest OOS profit
- Prioritize **robustness_score** and **consistency**
- Diversify: Select 3-5 uncorrelated strategies

### 5. Overfitting Prevention
- Never re-optimize on OOS data
- Treat OOS results as final evaluation
- If results are poor, simplify parameters (not re-optimize)

### 6. Regular Re-validation
- Re-run validation quarterly as new data arrives
- Monitor if robustness degrades over time
- Retire strategies that fail new OOS tests

## Troubleshooting

### No results generated

**Check:**
- Data files exist in `data/_all/`
- Filenames match config exactly
- File naming follows pattern: `{SYMBOL}_{TIMEFRAME}_{STARTDATE}_{ENDDATE}.csv`

### Empty comparison reports

**Check:**
- IS optimization completed successfully (check `results/optimization/`)
- OOS validation completed (check `results/validation/`)
- `trained_on_year` field exists in OOS results
- Strategy keys match between IS and OOS

### Low robustness scores across all strategies

**Possible causes:**
- Parameter space too wide (overfitting during optimization)
- Market regime changed between IS and OOS periods
- Insufficient training data
- Strategy assumptions don't hold in OOS period

**Solutions:**
- Narrow parameter ranges in mixin configs
- Try expanding window (more training data)
- Simplify strategies (fewer parameters)

## Performance Considerations

### Optimization Speed

With default settings (n_jobs=-1), the optimizer uses all CPU cores:
- 1 window × 1 symbol × 1 timeframe × 40 strategies ≈ 2-4 hours
- 3 windows × 3 symbols × 2 timeframes × 40 strategies ≈ 24-48 hours

**To speed up:**
1. Reduce `n_trials` in `config/optimizer/optimizer.json`
2. Run windows in parallel (manually)
3. Use subset of strategies initially

### Disk Space

Each strategy result is ~50-500KB depending on trade count:
- 1 window × 100 strategies ≈ 5-50 MB
- 3 windows × 100 strategies ≈ 15-150 MB

## Next Steps

After identifying robust strategies:

1. **Paper Trading**: Test top 3-5 strategies in paper trading for 1-3 months
2. **Portfolio Construction**: Combine uncorrelated strategies
3. **Position Sizing**: Use Kelly criterion or risk parity
4. **Monitoring**: Track live performance vs OOS predictions
5. **Regular Re-validation**: Quarterly validation with new data

## Support

For detailed implementation specs, see:
- [.kiro/specs/walk-forward-optimization.md](.kiro/specs/walk-forward-optimization.md)

For questions or issues:
- Review logs in console output
- Check individual result JSON files for details
- Ensure all dependencies are installed

## Summary Commands

```bash
# Full pipeline (runs all steps sequentially)
python src/backtester/optimizer/walk_forward_optimizer.py && \
python src/backtester/validator/walk_forward_validator.py && \
python src/backtester/validator/performance_comparer.py && \
python src/backtester/plotter/plot_top_strategies.py

# Review results
cat results/walk_forward_reports/degradation_analysis.json
head -20 results/walk_forward_reports/robustness_summary.csv
cat results/plots/top_strategies/selection_summary.txt
```

## Plotter Scripts Guide

### When to Use Each Plotter

#### 1. `plot_top_strategies.py` ⭐ **RECOMMENDED**

**Use when:** You want to visualize only the best performing strategies

**Pros:**
- Automatically selects top performers
- Fast (plots only 5-10 strategies)
- Creates organized output with categories
- Generates summary report
- Perfect for decision-making

**Example:**
```bash
python src/backtester/plotter/plot_top_strategies.py
```

#### 2. `run_plotter.py` ⚠️ **USE WITH CAUTION**

**Use when:** You need to plot ALL optimization results

**Pros:**
- Comprehensive visualization
- Useful for debugging or detailed analysis

**Cons:**
- Very slow (can take hours for 100+ strategies)
- Creates hundreds of PNG files
- Requires significant disk space
- Overwhelms with too much data

**Example:**
```bash
# Plot all results from a specific directory
python src/backtester/plotter/run_plotter.py
```

**⚠️ Warning:** Only use this for small datasets or when you need complete visualization.

### Customizing Top Strategies Selection

Edit `plot_top_strategies.py` to change selection criteria:

```python
# Change number of top strategies (default: 5)
selections = select_top_strategies(df, n=10)

# Add custom selection criteria
# For example, top by OOS profit instead of IS profit:
top_by_oos_profit = df.nlargest(n, 'oos_total_profit')
```

---

**Happy Walk-Forward Testing!**
