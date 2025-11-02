# FINRA Calculated Fields Population Guide

This guide explains how to populate the missing `short_interest_pct` and `days_to_cover` fields in the `ss_finra_short_interest` table.

## Overview

The `ss_finra_short_interest` table has two calculated fields that need to be populated:

1. **`short_interest_pct`** = `(short_interest_shares / float_shares) * 100`
2. **`days_to_cover`** = `short_interest_shares / average_daily_volume`

## Prerequisites

- ✅ `edgar_floats.sql` has been applied (float_shares populated)
- ✅ FMP API access configured (for volume data)
- ✅ Database access with UPDATE permissions

## Available Methods

### Method 1: SQL Scripts (Fastest)

#### Step 1: Calculate Short Interest Percentage
```bash
psql -d your_database -f calculate_short_interest_pct.sql
```

**What it does:**
- Calculates `short_interest_pct` for all records with both `short_interest_shares` and `float_shares`
- Only updates NULL values (safe to run multiple times)
- Provides statistics and sample results

#### Step 2: Calculate Days to Cover (Limited)
```bash
psql -d your_database -f calculate_days_to_cover.sql
```

**What it does:**
- Uses volume data from `ss_snapshot` table (if available)
- Limited coverage - only works for tickers with recent snapshot data
- Safe fallback but may miss many records

### Method 2: Python Script (Complete)

#### Run Complete Population
```bash
python populate_finra_calculated_fields.py
```

**What it does:**
- Calculates `short_interest_pct` using SQL (fast)
- Fetches volume data from FMP API for `days_to_cover` calculation
- Processes in batches with rate limiting
- Provides comprehensive coverage for both fields

**Features:**
- ✅ Batch processing (50 tickers per batch)
- ✅ Rate limiting (0.2s between API calls)
- ✅ Error handling and retry logic
- ✅ Progress logging and statistics
- ✅ Safe to run multiple times (only updates NULL values)

## Verification

After running either method, verify the results:

```bash
psql -d your_database -f verify_finra_calculated_fields.sql
```

**Verification includes:**
- Coverage statistics (% of records populated)
- Data distribution analysis
- Top short squeeze candidates
- Data quality checks
- Sample of complete records

## Expected Results

### Short Interest Percentage
- **Range**: 0% to 100%+ (can exceed 100% in rare cases)
- **Typical values**: 1-30%
- **High interest**: >20%
- **Coverage**: Should be ~100% where float_shares is available

### Days to Cover
- **Range**: 0.1 to 50+ days
- **Typical values**: 1-10 days
- **High squeeze potential**: >5 days
- **Coverage**: Depends on volume data availability

## Troubleshooting

### Database Constraint Error (short_interest_pct > 100%)
**Problem**: `ERROR: new row violates check constraint "valid_percentage"`
**Cause**: Database constraint limits short_interest_pct to 100%, but real market values can exceed this

**Solutions:**
1. **Quick Fix**: Use `calculate_short_interest_pct.sql` (caps values at 100%)
2. **Proper Fix**: Run database migration first:
   ```bash
   # Fix the constraint to allow values up to 1000%
   psql -d your_database -f fix_short_interest_pct_constraint.sql
   
   # Then use uncapped calculation
   psql -d your_database -f calculate_short_interest_pct_uncapped.sql
   ```

### Low Coverage for Days to Cover
**Problem**: Many records still have NULL `days_to_cover`
**Solutions:**
1. Run the Python script (fetches fresh volume data)
2. Check FMP API connectivity
3. Verify tickers are valid and actively traded

### High Short Interest Percentages (>100%)
**Problem**: Some records show >100% short interest
**Explanation**: This can happen when:
- Float shares data is outdated
- Short interest includes synthetic shares
- Data timing mismatches between sources
**Action**: Review specific cases, but values >100% can be valid in real markets

### API Rate Limits
**Problem**: FMP API rate limiting during volume fetch
**Solutions:**
1. Increase `api_delay` in the Python script
2. Process in smaller batches
3. Run during off-peak hours

## Performance Notes

### SQL Method
- ⚡ **Speed**: Very fast (seconds)
- 📊 **Coverage**: Limited for days_to_cover
- 💰 **Cost**: Free (no API calls)

### Python Method
- ⚡ **Speed**: Moderate (minutes to hours depending on ticker count)
- 📊 **Coverage**: Complete for both fields
- 💰 **Cost**: Uses FMP API quota

## Recommended Workflow

1. **Start with SQL**: Run `calculate_short_interest_pct.sql` first
2. **Check coverage**: Run verification to see days_to_cover gaps
3. **Fill gaps**: Run Python script if significant gaps remain
4. **Verify results**: Run verification queries
5. **Schedule updates**: Set up periodic runs for new FINRA data

## File Summary

| File | Purpose | Method | Notes |
|------|---------|---------|-------|
| `fix_short_interest_pct_constraint.sql` | Fix DB constraint for >100% values | SQL | Run first if you get constraint errors |
| `calculate_short_interest_pct.sql` | Calculate short interest % (capped) | SQL | Caps at 100% to avoid constraint errors |
| `calculate_short_interest_pct_uncapped.sql` | Calculate short interest % (uncapped) | SQL | Requires constraint fix first |
| `calculate_days_to_cover.sql` | Calculate days to cover (limited) | SQL | Uses existing snapshot data |
| `populate_finra_calculated_fields.py` | Complete field population | Python + API | Handles constraint automatically |
| `verify_finra_calculated_fields.sql` | Verify and analyze results | SQL | Comprehensive analysis |

## Integration with Pipeline

These calculated fields are used by:
- Short squeeze detection algorithms
- Weekly screener scoring
- Alert generation thresholds
- Risk assessment calculations

Ensure fields are populated before running the short squeeze pipeline components.