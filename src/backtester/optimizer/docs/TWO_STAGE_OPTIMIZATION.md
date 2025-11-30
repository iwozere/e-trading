# Two-Stage Optimization System

## Overview

The optimizer now implements a **two-stage optimization system** that intelligently allocates computational resources by filtering out unpromising strategy combinations early in the process.

## How It Works

### Stage 1: Screening Phase
- Runs a moderate number of trials (default: 150) for every entry/exit combination
- Calculates comprehensive metrics including:
  - Best value
  - Median value (most reliable indicator)
  - Mean value
  - Standard deviation (stability)
  - Top 10 average
  - Top 20% average
- Evaluates combinations against selection criteria
- Filters out unpromising combinations

### Stage 2: Deep Optimization Phase
- Only runs for combinations that pass Stage 1 criteria
- Continues optimization from Stage 1 (warm restart)
- Runs additional trials to reach total target (default: 500 total)
- Produces fully optimized parameters for promising strategies

## Configuration

Configuration is in [config/optimizer/optimizer.json](../../../config/optimizer/optimizer.json):

```json
{
    "optimizer_settings": {
        "two_stage_optimization": true,     // Enable/disable two-stage mode
        "stage1_n_trials": 150,            // Trials for screening phase
        "stage2_n_trials": 500,            // Total trials including Stage 2
        "dry_run_mode": false,             // Test filtering without Stage 2
        "selection_criteria": {
            "threshold_median": 0.05,      // Minimum median return (5%)
            "threshold_best": 0.15,        // Minimum best return (15%)
            "threshold_std": 1.0,          // Maximum volatility
            "min_trials_for_evaluation": 100  // Min trials to evaluate
        }
    }
}
```

## Selection Criteria

A combination is considered **promising** if it meets ALL criteria:

1. **Median Return** > threshold_median (default: 0.05 = 5%)
   - Most reliable metric, less affected by outliers

2. **Best Return** > threshold_best (default: 0.15 = 15%)
   - Shows potential upside

3. **Standard Deviation** < threshold_std (default: 1.0)
   - Ensures stability, filters volatile strategies

4. **Minimum Trials** >= min_trials_for_evaluation (default: 100)
   - Ensures sufficient data for evaluation

### Adjusting Thresholds

You can adjust thresholds based on your requirements:

- **Conservative (fewer Stage 2 runs)**: Higher thresholds
  ```json
  "threshold_median": 0.10,
  "threshold_best": 0.30,
  "threshold_std": 0.5
  ```

- **Aggressive (more Stage 2 runs)**: Lower thresholds
  ```json
  "threshold_median": 0.03,
  "threshold_best": 0.10,
  "threshold_std": 1.5
  ```

## Features

### 1. Multi-Metric Evaluation

Instead of only looking at the best result, the system evaluates:
- **Median**: Most reliable indicator of typical performance
- **Mean**: Average performance across all trials
- **Std**: Volatility/stability of results
- **Top 10 Avg**: Performance of best 10 trials
- **Top 20% Avg**: Performance of top quintile

### 2. Warm Restart

Stage 2 continues from Stage 1's study object:
- No parameter space re-exploration
- Builds on Stage 1 knowledge
- More efficient than starting fresh

### 3. Dry Run Mode

Test filtering logic without running Stage 2:
```json
"dry_run_mode": true
```

This shows:
- Which combinations would pass/fail
- Why they pass/fail
- Expected time savings
- No actual Stage 2 execution

### 4. Statistical Validation

Automatically performs t-test to validate that promising combinations are statistically better than unpromising ones:
- T-statistic
- P-value
- Statistical significance (p < 0.05)

### 5. Comprehensive Reporting

Generates detailed summary report including:
- CSV export with all combination results
- Console output with formatted tables
- Statistics by stage
- Promising vs unpromising comparison
- Statistical validation results
- Overall optimization statistics

## Usage Examples

### Example 1: Standard Two-Stage Optimization

```json
{
    "two_stage_optimization": true,
    "stage1_n_trials": 150,
    "stage2_n_trials": 500,
    "dry_run_mode": false,
    "selection_criteria": {
        "threshold_median": 0.05,
        "threshold_best": 0.15,
        "threshold_std": 1.0,
        "min_trials_for_evaluation": 100
    }
}
```

**Expected behavior:**
- All combinations get 150 trials (Stage 1)
- ~30-40% pass to Stage 2 (depends on strategies)
- Passing combinations get 350 additional trials (500 total)
- Full backtest only for passing combinations

### Example 2: Dry Run Testing

```json
{
    "two_stage_optimization": true,
    "stage1_n_trials": 150,
    "dry_run_mode": true
}
```

**Expected behavior:**
- All combinations get 150 trials
- Filtering decisions logged
- No Stage 2 execution
- No full backtests
- Fast preview of which combinations would proceed

### Example 3: Conservative Filtering

```json
{
    "two_stage_optimization": true,
    "stage1_n_trials": 200,
    "stage2_n_trials": 1000,
    "selection_criteria": {
        "threshold_median": 0.10,
        "threshold_best": 0.30,
        "threshold_std": 0.5,
        "min_trials_for_evaluation": 150
    }
}
```

**Expected behavior:**
- More thorough Stage 1 (200 trials)
- Stricter filtering (fewer pass)
- Very deep Stage 2 optimization (1000 trials)
- Only best combinations get full treatment

### Example 4: Single-Stage Mode (Original Behavior)

```json
{
    "two_stage_optimization": false,
    "n_trials": 100
}
```

**Expected behavior:**
- All combinations get same treatment (100 trials)
- No filtering
- Full backtest for all
- Original optimizer behavior

## Output and Logs

### Console Output Examples

#### Stage 1 Logging
```
INFO: STAGE 1: Screening phase (150 trials)
INFO: Stage 1 Results - RSIBBEntryMixin + TrailingStopExitMixin: Best=0.2543, Median=0.0821, Mean=0.0654, Std=0.1234, Top10=0.1876
INFO: ✓ PROMISING - RSIBBEntryMixin + TrailingStopExitMixin: Passed all criteria
```

#### Stage 2 Logging
```
INFO: STAGE 2: Deep optimization phase (additional 350 trials, 500 total)
INFO: Stage 2 Results - RSIBBEntryMixin + TrailingStopExitMixin: Best=0.3124, Median=0.1054, Mean=0.0876, Std=0.1123, Top10=0.2234
```

#### Filtering Logging
```
INFO: ✗ FILTERED OUT - MACDEntryMixin + FixedTPExitMixin: Low median return (0.0234 <= 0.0500)
```

### Summary Report

Generated at end of optimization:

```
================================================================================
OPTIMIZATION SUMMARY - ALL COMBINATIONS
================================================================================
Entry Logic          Exit Logic           Best    Median   Mean    Std     ...
RSIBBEntryMixin      TrailingStopExit    0.3124  0.1054  0.0876  0.1123  ...
MACrossEntryMixin    DynamicTPExit       0.2876  0.0987  0.0765  0.0987  ...
...

================================================================================
TOP 10 COMBINATIONS BY MEDIAN RETURN
================================================================================
...

================================================================================
STATISTICS BY STAGE
================================================================================
                     Median                              Best
                     count  mean    std    min    max   mean   max
Stage
screening            20     0.0543  0.0234 0.0123 0.1234 0.0876 0.2345
deep_optimization    6      0.0987  0.0187 0.0654 0.1234 0.1456 0.3124

================================================================================
PROMISING vs UNPROMISING COMBINATIONS
================================================================================
Promising combinations: 6
Unpromising combinations: 14

Promising - Median stats: mean=0.0987, std=0.0187, min=0.0654, max=0.1234
Unpromising - Median stats: mean=0.0234, std=0.0123, min=0.0012, max=0.0487

================================================================================
STATISTICAL VALIDATION (T-TEST)
================================================================================
T-statistic: 8.5432
P-value: 0.000012
Statistically significant: True (p < 0.05)
Promising mean median: 0.0987
Unpromising mean median: 0.0234

================================================================================
OVERALL STATISTICS
================================================================================
Total combinations tested: 20
Stage 1 (screening): 20
Stage 2 (deep optimization): 6
Promising combinations: 6
Pass rate to Stage 2: 30.0%
Promising rate: 30.0%
```

### CSV Output

File: `results/optimization_summary_YYYYMMDD_HHMMSS.csv`

Columns:
- Entry Logic
- Exit Logic
- Best
- Median
- Mean
- Std
- Top 10 Avg
- Top 20% Avg
- Total Trials
- Stage
- Promising

## Performance Benefits

### Time Savings

Assuming:
- 20 entry/exit combinations
- 5 minutes per 100 trials
- 30% pass rate to Stage 2

**Original approach** (100 trials each):
- 20 combinations × 100 trials = 2000 trials
- Time: 20 × 5 = 100 minutes

**Two-stage approach** (150 + 350 for promising):
- Stage 1: 20 × 150 trials = 3000 trials (150 minutes)
- Stage 2: 6 × 350 trials = 2100 trials (105 minutes)
- Total time: 255 minutes

**But with better results!** The two-stage approach:
- Identifies best strategies more reliably (multi-metric evaluation)
- Optimizes promising strategies more deeply (500 vs 100 trials)
- Filters out noise from unpromising combinations
- Provides comprehensive comparison data

### Resource Allocation

Instead of:
- 100 trials for everything (including bad combinations)

You get:
- 150 trials for initial screening
- 500 trials for best combinations
- 0 additional trials for poor combinations

## Troubleshooting

### Issue: All combinations filtered out

**Symptoms:**
```
INFO: ✗ FILTERED OUT - All combinations
```

**Solutions:**
1. Lower thresholds in selection_criteria
2. Increase stage1_n_trials for better evaluation
3. Check if strategies are fundamentally unprofitable
4. Review data quality

### Issue: Too many combinations passing

**Symptoms:**
- Stage 2 running for 80%+ of combinations
- Little time savings

**Solutions:**
1. Raise thresholds (especially threshold_median)
2. Add stricter threshold_std requirement
3. Increase min_trials_for_evaluation

### Issue: Inconsistent results between stages

**Symptoms:**
- Stage 2 median worse than Stage 1

**Solutions:**
- This can happen due to randomness
- Increase stage1_n_trials for more stable estimates
- Check for overfitting in parameter ranges
- Review if warm restart is working correctly

## Best Practices

1. **Start with dry run mode** to calibrate thresholds
   ```json
   "dry_run_mode": true
   ```

2. **Use conservative thresholds initially**
   - Better to over-filter than under-filter
   - Can always lower thresholds if needed

3. **Monitor median, not just best**
   - Median is more reliable
   - Best can be a lucky outlier

4. **Review summary report carefully**
   - Check if filtering makes sense
   - Validate statistical significance
   - Look for patterns in promising combinations

5. **Adjust based on your data**
   - Different timeframes may need different thresholds
   - Different symbols may have different return profiles

6. **Save the CSV report**
   - Useful for comparing different threshold settings
   - Helps identify patterns across runs

## Technical Details

### Helper Functions

1. **calculate_optimization_metrics(study)**
   - Extracts all metrics from Optuna study
   - Returns dictionary with comprehensive statistics
   - Handles edge cases (no trials, failed trials)

2. **evaluate_combination_promise(metrics, thresholds)**
   - Applies selection criteria
   - Returns (is_promising, reason) tuple
   - Provides detailed reason for filtering decisions

3. **perform_statistical_validation(promising, unpromising)**
   - Performs independent samples t-test
   - Validates filtering effectiveness
   - Returns None if insufficient data

4. **generate_summary_report(all_results, validation)**
   - Creates pandas DataFrame
   - Generates CSV output
   - Prints formatted console tables
   - Shows statistical validation results

### Data Structures

**all_results**: Dictionary storing all combination results
```python
{
    (entry_name, exit_name): {
        'metrics': {...},
        'best_params': {...},
        'study': optuna.Study,
        'stage': 'screening' or 'deep_optimization',
        'is_promising': bool,
        'reason': str
    }
}
```

**metrics**: Dictionary with calculated metrics
```python
{
    'best_value': float,
    'median_value': float,
    'mean_value': float,
    'std_value': float,
    'top_10_avg': float,
    'top_20_percent_avg': float,
    'total_trials': int,
    'completed_trials': int,
    'failed_trials': int
}
```

## Migration Guide

### From Single-Stage to Two-Stage

No breaking changes! The system is backward compatible:

1. **Default behavior**: Two-stage enabled
2. **To disable**: Set `"two_stage_optimization": false`
3. **Existing configs**: Will use default thresholds

### Recommended Migration Steps

1. **Test with dry run**
   ```json
   "dry_run_mode": true
   ```

2. **Review filtering decisions**
   - Check console logs
   - Examine summary report

3. **Adjust thresholds if needed**

4. **Run full optimization**
   ```json
   "dry_run_mode": false
   ```

5. **Compare with previous runs**
   - Check if promising combinations make sense
   - Validate time savings
   - Review result quality

## References

- Implementation: [run_optimizer.py](run_optimizer.py)
- Configuration: [optimizer.json](../../../config/optimizer/optimizer.json)
- Original TODO: [TODO_EN.md](TODO_EN.md)
- Proposal: [IMPROVEMENT_PROPOSAL.md](IMPROVEMENT_PROPOSAL.md)
