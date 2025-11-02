# Volume Detector Migration to yfinance

## Overview
The volume detector script has been updated to use **yfinance** instead of the FMP API to avoid rate limiting issues (HTTP 429 errors).

## Changes Made

### 1. New Module: `volume_squeeze_detector_yf.py`
Created a new yfinance-based volume squeeze detector at:
- `src/ml/pipeline/p04_short_squeeze/core/volume_squeeze_detector_yf.py`

**Key Features:**
- Uses `yfinance.Ticker()` to fetch company info and OHLCV data
- No API key required
- No rate limits (within reasonable usage)
- Same analysis logic as the original FMP version

**Main Methods:**
- `get_company_info_yf(ticker)` - Fetches market cap, float shares, sector, etc.
- `get_ohlcv_yf(ticker, start_date, end_date)` - Fetches historical price/volume data
- `calculate_volume_metrics(ticker)` - Calculates volume spike ratio, trends, consistency
- `calculate_momentum_metrics(ticker)` - Calculates price momentum and volatility
- `calculate_squeeze_indicators(ticker)` - Combines all metrics into squeeze score
- `screen_universe(tickers, min_score)` - Screens multiple tickers for candidates

### 2. Updated Script: `run_volume_detector.py`
Modified the main volume detector script:

**Changes:**
- ✅ Removed FMP downloader initialization
- ✅ Updated to use `create_volume_squeeze_detector_yf()` instead of FMP version
- ✅ Updated `initialize_data_providers()` to work without FMP
- ✅ Updated `load_universe()` to fallback to S&P 500 list when database is empty
- ✅ Added script docstring noting yfinance usage

**No Changes to:**
- Database integration
- Progress tracking
- Result storage
- Performance reporting
- Command-line arguments

### 3. Test Script
Created a test script to verify functionality:
- `src/ml/pipeline/p04_short_squeeze/tests/test_volume_detector_yf.py`

## Usage

### Running the Volume Detector

```bash
# Basic run (uses database universe or S&P 500 fallback)
python src/ml/pipeline/p04_short_squeeze/scripts/run_volume_detector.py

# Test with limited universe
python src/ml/pipeline/p04_short_squeeze/scripts/run_volume_detector.py --max-universe 50

# Dry run (no database writes)
python src/ml/pipeline/p04_short_squeeze/scripts/run_volume_detector.py --dry-run --max-universe 10

# With progress tracking
python src/ml/pipeline/p04_short_squeeze/scripts/run_volume_detector.py --progress

# Custom volume spike threshold
python src/ml/pipeline/p04_short_squeeze/scripts/run_volume_detector.py --min-volume-spike 3.0
```

### Testing the Detector

```bash
# Run quick test with a few tickers
python src/ml/pipeline/p04_short_squeeze/scripts/test_volume_detector_yf.py
```

## Benefits of yfinance

✅ **No API Key Required** - No need for FMP API key  
✅ **No Rate Limits** - Can analyze hundreds of stocks without hitting limits  
✅ **Free & Open Source** - Completely free to use  
✅ **Reliable Data** - Fetches data directly from Yahoo Finance  
✅ **Easy Maintenance** - One less API dependency to manage  

## Data Comparison

| Metric | FMP API | yfinance |
|--------|---------|----------|
| Company Info | ✓ | ✓ |
| OHLCV Data | ✓ | ✓ |
| Volume Data | ✓ | ✓ |
| Market Cap | ✓ | ✓ |
| Float Shares | ✓ | ✓ |
| Short Interest | ✓ | ✗ |
| Rate Limits | Yes (429 errors) | No |
| API Key | Required | Not required |

**Note:** Short interest data is not available through yfinance, but the volume detector doesn't use short interest anyway (it focuses on volume and momentum patterns).

## Backward Compatibility

The original FMP-based detector is still available at:
- `src/ml/pipeline/p04_short_squeeze/core/volume_squeeze_detector.py`

If you need to switch back, just revert the imports in `run_volume_detector.py`.

## Performance

yfinance fetches data ticker-by-ticker, so screening large universes (1000+ stocks) may take longer than FMP batch requests. However:
- No rate limit interruptions
- Can run in the background without errors
- More reliable for daily automated runs

**Recommended Usage:**
- Use `--max-universe 100` for quick tests
- Use database universe (already filtered candidates) for production
- Enable `--progress` to monitor long-running screens

## Troubleshooting

### "No data returned for ticker"
- Ticker may be delisted or have no recent trading data
- yfinance may not recognize the ticker symbol
- Try on Yahoo Finance website to verify ticker exists

### Slow Performance
- yfinance fetches data sequentially
- Consider reducing universe size with `--max-universe`
- Use database universe instead of full S&P 500

### Import Error
- Ensure yfinance is installed: `pip install yfinance>=0.2.64`
- Already in requirements.txt

## Next Steps

1. ✅ Test with small universe (--max-universe 10)
2. ✅ Verify data quality matches FMP version
3. ✅ Run with database universe in production
4. Consider adding caching for frequently analyzed tickers
5. Monitor performance with larger universes

## Files Modified

- ✅ Created: `src/ml/pipeline/p04_short_squeeze/core/volume_squeeze_detector_yf.py` (new)
- ✅ Modified: `src/ml/pipeline/p04_short_squeeze/scripts/run_volume_detector.py`
- ✅ Created: `src/ml/pipeline/p04_short_squeeze/scripts/test_volume_detector_yf.py` (test)
- ✅ Created: `src/ml/pipeline/p04_short_squeeze/docs/VOLUME_DETECTOR_YFINANCE_MIGRATION.md` (this file)
