# Data Cache Refresh Guide

## Overview

This guide explains how to refresh your corrupted data cache using the proper unified architecture. You have two scripts available depending on your needs.

---

## Architecture

### Cache Structure

Data is cached in: `c:/data-cache/` (configured in `config/donotshare/donotshare.py`)

```
c:/data-cache/
└── ohlcv/
    ├── BTCUSDT/
    │   ├── 1h/
    │   │   ├── 2022.csv.gz           # Compressed OHLCV data
    │   │   ├── 2022.metadata.json    # Provider, date range, row count
    │   │   ├── 2023.csv.gz
    │   │   ├── 2023.metadata.json
    │   │   └── ...
    │   ├── 4h/
    │   │   ├── 2022.csv.gz
    │   │   └── 2022.metadata.json
    │   └── 5m/
    ├── ETHUSDT/
    │   └── 1h/
    ├── LTCUSDT/
    │   └── 4h/
    └── _metadata/
        ├── symbols.json           # List of cached symbols
        ├── providers.json         # Provider statistics
        └── cache_stats.json       # Cache size and metrics
```

### Why This Structure?

- **Year-based storage**: One file per year for efficient access
- **Gzip compression**: Saves ~90% disk space
- **Metadata tracking**: Provider info, date coverage, validation status
- **Provider-agnostic**: Can mix data from Binance, Yahoo, Alpaca, etc.

---

## Option 1: Simple Download Script (Recommended for Full Refresh)

**File**: `src/util/data_downloader.py` (now updated)

### When to Use

- You want to **completely refresh** all data for specific symbols
- You have corrupted files and want to re-download everything
- You want to force download even if cache exists

### Usage

```bash
python src/util/data_downloader.py
```

### What It Does

1. Downloads data for **LTCUSDT, BTCUSDT, ETHUSDT**
2. Intervals: **5m, 15m, 30m, 1h, 4h**
3. Date range: **2020-01-01 to 2025-11-11**
4. **Force refresh**: Downloads even if cached (overwrites corrupted data)
5. Automatically caches to `c:/data-cache/ohlcv/`

### Configuration

Edit the symbols/intervals/dates in the script:

```python
DOWNLOAD_SCENARIOS = {
    'symbols': ['LTCUSDT', 'BTCUSDT', 'ETHUSDT'],  # Your symbols
    'periods': [
        {'start_date': '20200101', 'end_date': '20251111'}  # Date range
    ],
    'intervals': ['5m', '15m', '30m', '1h', '4h']  # Timeframes
}
```

### Output

```
================================================================================
Starting data download using unified DataManager architecture
================================================================================
Cache directory: c:/data-cache
Total combinations: 15
================================================================================

--------------------------------------------------------------------------------
Processing 1/15: LTCUSDT 5m (2020-01-01 to 2025-11-11)
--------------------------------------------------------------------------------
✅ Successfully downloaded and cached LTCUSDT 5m: 543210 rows

...

================================================================================
DOWNLOAD SUMMARY
================================================================================
Total combinations: 15
Successful: 14
Failed: 1
Success rate: 93.3%
================================================================================
Data cached to: c:/data-cache
Cache structure: c:/data-cache/ohlcv/{SYMBOL}/{TIMEFRAME}/{YEAR}.csv.gz
================================================================================
```

---

## Option 2: Smart Cache Populator (Recommended for Incremental Updates)

**File**: `src/data/utils/populate_cache.py`

### When to Use

- You want to **intelligently fill gaps** in your cache
- You want to refresh only corrupted/invalid data
- You want to update existing cache with new data
- You want to validate existing cache files

### Features

✅ **Automatic corruption detection**:
- Detects mock/test data
- Validates date coverage (ensures full year)
- Checks minimum row counts
- Verifies metadata integrity

✅ **Smart gap filling**:
- Analyzes what's missing
- Only downloads what's needed
- Preserves valid cached data

✅ **Automatic symbol discovery**:
- Finds all symbols in your cache
- Ensures all intervals are cached

### Usage Examples

#### Refresh All Cached Symbols

Analyzes your cache and fills missing/corrupted data:

```bash
python src/data/utils/populate_cache.py --start-date 2020-01-01
```

#### Refresh Specific Symbols

```bash
python src/data/utils/populate_cache.py --tickers BTCUSDT,ETHUSDT,LTCUSDT --start-date 2020-01-01
```

#### Custom Intervals

```bash
python src/data/utils/populate_cache.py --tickers BTCUSDT --intervals 1h,4h,1d --start-date 2022-01-01
```

#### Full Date Range

```bash
python src/data/utils/populate_cache.py --tickers BTCUSDT --start-date 2020-01-01 --end-date 2024-12-31
```

### What It Does

1. **Discovery Phase**:
   - Scans `c:/data-cache/ohlcv/` for existing symbols
   - Lists all cached tickers

2. **Analysis Phase**:
   - Checks each symbol/interval/year combination
   - Validates metadata files
   - Detects corruption (mock data, incomplete coverage, too few rows)
   - Identifies missing years

3. **Population Phase**:
   - Downloads only missing/corrupted data
   - Uses DataManager (provider fallback, rate limiting)
   - Caches with compression and metadata

4. **Summary**:
   - Reports success/failure per symbol/interval
   - Shows overall completion rate

### Validation Logic

Data is considered **invalid** (will be redownloaded) if:

❌ Data file missing
❌ Metadata file missing
❌ Provider is "mock" (test data)
❌ Date coverage incomplete:
  - Daily data: Must start by Jan 10, end by Dec 20
  - Intraday data: More lenient (providers have limited history)
❌ Too few rows for the year:
  - 1h: Minimum ~175 hours
  - 4h: Minimum ~400 periods
  - 1d: Minimum ~250 trading days

### Output Example

```
================================================================================
CACHE POPULATION SUMMARY
================================================================================
✅ BTCUSDT: 5/5 intervals successful
✅ ETHUSDT: 5/5 intervals successful
⚠️ LTCUSDT: 4/5 intervals successful
------------------------------------------------------------
Overall: 14/15 operations successful (93.3%)
================================================================================
```

---

## Comparison: Which Script to Use?

| Feature | data_downloader.py | populate_cache.py |
|---------|-------------------|-------------------|
| **Force refresh all** | ✅ Yes | ❌ No (skips valid data) |
| **Detect corruption** | ❌ No | ✅ Yes |
| **Fill gaps only** | ❌ No | ✅ Yes |
| **Auto-discover symbols** | ❌ No (hardcoded) | ✅ Yes |
| **Validate metadata** | ❌ No | ✅ Yes |
| **Speed** | Slower (re-downloads everything) | Faster (only downloads needed data) |
| **Use case** | Complete cache rebuild | Incremental cache maintenance |

### Recommendation

**For your case (corrupted cache refresh)**:

1. **First**: Use `populate_cache.py` to intelligently refresh corrupted data:
   ```bash
   python src/data/utils/populate_cache.py --start-date 2020-01-01
   ```

2. **If issues persist**: Use `data_downloader.py` for full refresh:
   ```bash
   python src/util/data_downloader.py
   ```

---

## Workflow for Complete Cache Refresh

### Step 1: Backup Existing Cache (Optional)

```bash
# Windows
xcopy c:\data-cache c:\data-cache-backup /E /I

# Linux/Mac
cp -r /c/data-cache /c/data-cache-backup
```

### Step 2: Option A - Smart Refresh (Recommended)

```bash
# Detect and refresh corrupted files
python src/data/utils/populate_cache.py --start-date 2020-01-01

# If specific symbols are problematic, force refresh them
python src/data/utils/populate_cache.py --tickers BTCUSDT,ETHUSDT --start-date 2020-01-01
```

### Step 2: Option B - Nuclear Option (Full Re-download)

```bash
# Delete corrupted cache
rm -rf c:/data-cache/ohlcv/*

# Re-download everything
python src/util/data_downloader.py
```

### Step 3: Verify Cache

Check the cache structure:

```bash
# Windows
dir c:\data-cache\ohlcv /s

# Linux/Mac
ls -lR /c/data-cache/ohlcv
```

Expected structure:
```
c:/data-cache/ohlcv/
├── BTCUSDT/
│   ├── 1h/
│   │   ├── 2020.csv.gz  ✅
│   │   ├── 2020.metadata.json  ✅
│   │   ├── 2021.csv.gz  ✅
│   │   └── ...
```

---

## Verifying Data Quality

### Check Metadata

```python
import json
import gzip

# Read metadata
with open('c:/data-cache/ohlcv/BTCUSDT/1h/2024.metadata.json', 'r') as f:
    metadata = json.load(f)

print("Provider:", metadata['data_source'])
print("Date range:", metadata['start_date'], "to", metadata['end_date'])
print("Rows:", metadata['file_info']['rows'])
```

### Read Cached Data

```python
import pandas as pd

# Read compressed CSV
df = pd.read_csv('c:/data-cache/ohlcv/BTCUSDT/1h/2024.csv.gz', compression='gzip')
print(df.head())
print(f"Shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
```

### Using DataManager

```python
from src.data.data_manager import DataManager
from datetime import datetime

dm = DataManager()

# This will use cache if available
df = dm.get_ohlcv(
    symbol='BTCUSDT',
    timeframe='1h',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)

print(df.head())
```

---

## Troubleshooting

### "No data returned"

**Possible causes**:
- Provider doesn't support this symbol/interval
- Date range too far in the past (intraday data limited to ~1-2 years)
- API rate limits hit

**Solution**:
- Check provider capabilities in `config/data/provider_rules.yaml`
- Try shorter date ranges for intraday data
- Wait a few minutes for rate limits to reset

### "All providers failed"

**Possible causes**:
- Network issues
- All providers rate-limited
- Symbol not available on any provider

**Solution**:
- Check internet connection
- Verify symbol exists (e.g., BTCUSDT on Binance)
- Try again later

### "Metadata missing"

**Possible causes**:
- Previous download was interrupted
- File system permissions issue

**Solution**:
- Delete the corrupted year files:
  ```bash
  rm c:/data-cache/ohlcv/BTCUSDT/1h/2024.csv.gz
  rm c:/data-cache/ohlcv/BTCUSDT/1h/2024.metadata.json
  ```
- Re-run populate script

---

## Advanced: Cleaning Up Old Cache

### Remove All Cached Data

```bash
# Windows
rmdir /s /q c:\data-cache\ohlcv

# Linux/Mac
rm -rf /c/data-cache/ohlcv
```

### Remove Specific Symbol

```bash
# Windows
rmdir /s /q c:\data-cache\ohlcv\BTCUSDT

# Linux/Mac
rm -rf /c/data-cache/ohlcv/BTCUSDT
```

### Remove Specific Interval

```bash
# Windows
rmdir /s /q c:\data-cache\ohlcv\BTCUSDT\1h

# Linux/Mac
rm -rf /c/data-cache/ohlcv/BTCUSDT/1h
```

---

## Summary

**For corrupted cache refresh**:
1. Use `populate_cache.py` first (smart, faster)
2. Use `data_downloader.py` as fallback (complete refresh)

**Key benefits of new architecture**:
- ✅ Automatic caching to `c:/data-cache/ohlcv/`
- ✅ Gzip compression (~90% space savings)
- ✅ Metadata tracking (provider, dates, validation)
- ✅ Provider fallback (if Binance fails, tries Yahoo, etc.)
- ✅ Corruption detection and repair

**No more manual file management** - DataManager handles everything!
