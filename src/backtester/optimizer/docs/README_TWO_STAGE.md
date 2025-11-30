# Two-Stage Optimization - Quick Start Guide

## What Is This?

The optimizer now uses a **smart two-stage approach** that saves time by filtering out poor strategy combinations early.

## Quick Start

### 1. Use Default Settings (Recommended)

The optimizer is already configured with sensible defaults. Just run:

```bash
python src\backtester\optimizer\run_optimizer.py
```

It will:
- ✅ Test all combinations with 150 trials (Stage 1)
- ✅ Filter out poor performers automatically
- ✅ Deeply optimize promising ones with 500 trials (Stage 2)
- ✅ Generate a summary report at the end

### 2. Test Before Running (Dry Run)

To see what would be filtered without actually running Stage 2:

Edit `config/optimizer/optimizer.json`:
```json
{
    "optimizer_settings": {
        "dry_run_mode": true
    }
}
```

Run the optimizer. You'll see which combinations would pass/fail and why.

### 3. Adjust If Needed

If too many/few combinations are passing, adjust thresholds in `config/optimizer/optimizer.json`:

```json
{
    "optimizer_settings": {
        "selection_criteria": {
            "threshold_median": 0.05,    // 5% minimum median return
            "threshold_best": 0.15,      // 15% minimum best return
            "threshold_std": 1.0         // Maximum volatility
        }
    }
}
```

**More strict** (fewer pass): Increase threshold_median and threshold_best
**More lenient** (more pass): Decrease threshold_median and threshold_best

## Understanding the Output

### Console Logs

#### ✓ Promising Combination
```
INFO: Stage 1 Results - RSIBBEntry + TrailingStopExit: Best=0.25, Median=0.08
INFO: ✓ PROMISING - Will proceed to Stage 2
INFO: STAGE 2: Deep optimization (500 trials total)
```

#### ✗ Filtered Combination
```
INFO: Stage 1 Results - MACDEntry + FixedTPExit: Best=0.12, Median=0.02
INFO: ✗ FILTERED OUT - Low median return (0.02 <= 0.05)
```

### Summary Report

At the end, you'll get a comprehensive report:

**CSV File**: `results/optimization_summary_YYYYMMDD_HHMMSS.csv`
- All combinations ranked by performance
- Import into Excel for analysis

**Console Output**: Formatted tables showing:
- Top 10 combinations
- Statistics by stage
- Promising vs unpromising comparison
- Statistical validation

## Configuration Files

### optimizer.json Settings

| Setting | Default | What It Does |
|---------|---------|--------------|
| `two_stage_optimization` | true | Enable smart filtering |
| `stage1_n_trials` | 150 | Trials for screening |
| `stage2_n_trials` | 500 | Trials for deep optimization |
| `dry_run_mode` | false | Test without Stage 2 |
| `threshold_median` | 0.05 | Min median return (5%) |
| `threshold_best` | 0.15 | Min best return (15%) |
| `threshold_std` | 1.0 | Max volatility |

## Common Scenarios

### Scenario 1: "I want to test everything thoroughly"

```json
{
    "two_stage_optimization": false,
    "n_trials": 500
}
```

This runs 500 trials for **every** combination. Slow but thorough.

### Scenario 2: "I want to save time" (Default)

```json
{
    "two_stage_optimization": true,
    "stage1_n_trials": 150,
    "stage2_n_trials": 500
}
```

This screens with 150 trials, then deeply optimizes only the best.

### Scenario 3: "I'm not sure what thresholds to use"

```json
{
    "two_stage_optimization": true,
    "dry_run_mode": true
}
```

Run this first to see filtering decisions without commitment.

### Scenario 4: "I want very strict filtering"

```json
{
    "selection_criteria": {
        "threshold_median": 0.10,
        "threshold_best": 0.30,
        "threshold_std": 0.5
    }
}
```

Only the absolute best combinations will pass.

## Interpreting Results

### Median vs Best

**Median** = typical performance (50th percentile)
- More reliable indicator
- Less affected by lucky outliers
- **This is the primary filtering metric**

**Best** = single best trial result
- Shows potential upside
- Can be a lucky outlier
- Used as secondary filter

### Standard Deviation (Std)

**Low std** (e.g., 0.05) = stable, consistent strategy
**High std** (e.g., 1.5) = volatile, unpredictable strategy

Lower is generally better (more reliable).

### Pass Rate

**Typical**: 30-40% of combinations pass to Stage 2

**If 80%+ passing**: Thresholds too lenient, increase them
**If <10% passing**: Thresholds too strict, decrease them

## Troubleshooting

### "All combinations filtered out!"

**Problem**: Thresholds too strict or strategies genuinely unprofitable

**Solutions**:
1. Lower `threshold_median` to 0.03
2. Lower `threshold_best` to 0.10
3. Check if your strategies are fundamentally profitable
4. Review data quality

### "All combinations passing!"

**Problem**: Thresholds too lenient, no time savings

**Solutions**:
1. Raise `threshold_median` to 0.08
2. Raise `threshold_best` to 0.25
3. Add stricter `threshold_std` requirement (e.g., 0.5)

### "Results seem random"

**Problem**: Not enough trials for reliable evaluation

**Solutions**:
1. Increase `stage1_n_trials` to 200
2. Increase `min_trials_for_evaluation` to 150

## Advanced Features

### Statistical Validation

Automatically checks if promising combinations are **statistically** better than unpromising ones.

Look for in summary report:
```
P-value: 0.000012
Statistically significant: True (p < 0.05)
```

If not significant, thresholds may need adjustment.

### Warm Restart

Stage 2 continues from where Stage 1 left off:
- No wasted exploration
- Builds on Stage 1 knowledge
- More efficient than restarting

## Files and Documentation

- **[TWO_STAGE_OPTIMIZATION.md](TWO_STAGE_OPTIMIZATION.md)** - Complete user guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical details
- **[TODO_EN.md](TODO_EN.md)** - Background and rationale
- **[test_two_stage.py](test_two_stage.py)** - Unit tests

## Questions?

1. Read [TWO_STAGE_OPTIMIZATION.md](TWO_STAGE_OPTIMIZATION.md) for detailed guide
2. Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details
3. Review the summary CSV after a run to understand results

## Summary

**Default behavior**: Smart, efficient, saves time
**To disable**: Set `"two_stage_optimization": false`
**To test**: Set `"dry_run_mode": true`
**To adjust**: Modify `selection_criteria` thresholds

**That's it! The optimizer is ready to use.**
