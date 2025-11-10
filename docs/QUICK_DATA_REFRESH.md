# Quick Data Cache Refresh Reference

## Fixed Script: data_downloader.py

Your `src/util/data_downloader.py` has been fixed and now works properly!

### What Was Fixed

✅ Now uses **DataManager** (proper architecture)
✅ Automatic path setup (works from any directory)
✅ Automatic cache to `c:/data-cache/ohlcv/`
✅ Gzip compression + metadata
✅ Force refresh (overwrites corrupted files)

---

## Quick Usage

### Option 1: Simple Force Refresh

Completely re-download all data for BTCUSDT, ETHUSDT, LTCUSDT:

```bash
cd c:/dev/cursor/e-trading
python src/util/data_downloader.py
```

**Downloads**:
- Symbols: LTCUSDT, BTCUSDT, ETHUSDT
- Intervals: 5m, 15m, 30m, 1h, 4h
- Period: 2020-01-01 to 2025-11-11
- Total: 15 combinations

**Time**: ~30-60 minutes (depending on network/API limits)

---

### Option 2: Smart Corruption Detection

Only refresh corrupted/missing data:

```bash
cd c:/dev/cursor/e-trading
python src/data/utils/populate_cache.py --start-date 2020-01-01
```

**Features**:
- Detects corrupted files automatically
- Only downloads what's broken/missing
- Preserves valid cache
- Validates metadata

**Time**: ~5-30 minutes (only downloads needed data)

---

## Customizing data_downloader.py

Edit lines 51-57 to change what gets downloaded:

```python
DOWNLOAD_SCENARIOS = {
    'symbols': ['LTCUSDT', 'BTCUSDT', 'ETHUSDT'],  # Add/remove symbols
    'periods': [
        {'start_date': '20200101', 'end_date': '20251111'}  # Change dates
    ],
    'intervals': ['5m', '15m', '30m', '1h', '4h']  # Add/remove intervals
}
```

### Examples

**Download only 1h and 4h**:
```python
'intervals': ['1h', '4h']
```

**Download from 2022 only**:
```python
'periods': [{'start_date': '20220101', 'end_date': '20221231'}]
```

**Add more symbols**:
```python
'symbols': ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'BNBUSDT', 'ADAUSDT']
```

---

## Expected Output

```
2025-11-10 19:00:00,000 - INFO - Data Downloader Script
2025-11-10 19:00:00,000 - INFO - Project root: C:\dev\cursor\e-trading
2025-11-10 19:00:00,000 - INFO - Current directory: C:\dev\cursor\e-trading
2025-11-10 19:00:00,000 - INFO - Cache directory: c:/data-cache
2025-11-10 19:00:00,000 - INFO - ================================================================================
2025-11-10 19:00:00,000 - INFO - Starting data download using unified DataManager architecture
2025-11-10 19:00:00,000 - INFO - ================================================================================
2025-11-10 19:00:00,000 - INFO - Cache directory: c:/data-cache
2025-11-10 19:00:00,000 - INFO - Total combinations: 15
2025-11-10 19:00:00,000 - INFO - ================================================================================
2025-11-10 19:00:00,000 - INFO -
2025-11-10 19:00:00,000 - INFO - --------------------------------------------------------------------------------
2025-11-10 19:00:00,000 - INFO - Processing 1/15: LTCUSDT 5m (2020-01-01 to 2025-11-11)
2025-11-10 19:00:00,000 - INFO - --------------------------------------------------------------------------------
2025-11-10 19:00:05,000 - INFO - Successfully downloaded and cached LTCUSDT 5m: 543210 rows
...
2025-11-10 19:30:00,000 - INFO - ================================================================================
2025-11-10 19:30:00,000 - INFO - DOWNLOAD SUMMARY
2025-11-10 19:30:00,000 - INFO - ================================================================================
2025-11-10 19:30:00,000 - INFO - Total combinations: 15
2025-11-10 19:30:00,000 - INFO - Successful: 15
2025-11-10 19:30:00,000 - INFO - Failed: 0
2025-11-10 19:30:00,000 - INFO - Success rate: 100.0%
2025-11-10 19:30:00,000 - INFO - ================================================================================
2025-11-10 19:30:00,000 - INFO - Data cached to: c:/data-cache
2025-11-10 19:30:00,000 - INFO - Cache structure: c:/data-cache/ohlcv/{SYMBOL}/{TIMEFRAME}/{YEAR}.csv.gz
2025-11-10 19:30:00,000 - INFO - ================================================================================
```

---

## Verifying Cache

### Check Cache Structure

```bash
dir c:\data-cache\ohlcv /s
```

Expected output:
```
c:\data-cache\ohlcv\BTCUSDT\1h\
    2020.csv.gz
    2020.metadata.json
    2021.csv.gz
    2021.metadata.json
    ...
```

### Test Reading Cached Data

```python
from src.data.data_manager import DataManager
from datetime import datetime

dm = DataManager()
df = dm.get_ohlcv('BTCUSDT', '1h', datetime(2024, 1, 1), datetime(2024, 12, 31))
print(f"Loaded {len(df)} rows from cache")
```

---

## Troubleshooting

### ImportError: No module named 'config.donotshare'

**Solution**: The script now handles this automatically with `os.chdir(PROJECT_ROOT)`. Just make sure to run from project root:

```bash
cd c:/dev/cursor/e-trading
python src/util/data_downloader.py
```

### "No suitable provider found"

**Cause**: Symbol not available on configured providers

**Solution**: Check `config/data/provider_rules.yaml` or try different symbols

### Rate limit errors

**Cause**: Too many API requests

**Solution**:
- Wait 5-10 minutes
- Reduce number of symbols/intervals
- DataManager handles rate limiting automatically

### Slow download

**Normal**: Downloading 5+ years of 5m data takes time
- 5m data: ~500k rows = 2-5 minutes per symbol
- 1h data: ~40k rows = 30-60 seconds per symbol
- 4h data: ~10k rows = 10-20 seconds per symbol

---

## Where Data is Cached

All data goes to: `c:/data-cache/ohlcv/`

**Cache benefits**:
- Gzip compression (~90% space savings)
- Year-based organization (fast access)
- Metadata tracking (provider, dates, validation)
- Provider-agnostic (can mix Binance, Yahoo, etc.)

**Example cache entry**:

`c:/data-cache/ohlcv/BTCUSDT/1h/2024.csv.gz`:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,42150.00,42200.00,42100.00,42180.00,1234.56
2024-01-01 01:00:00,42180.00,42250.00,42150.00,42200.00,987.65
...
```

`c:/data-cache/ohlcv/BTCUSDT/1h/2024.metadata.json`:
```json
{
  "data_source": "binance",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "start_date": "2024-01-01T00:00:00",
  "end_date": "2024-12-31T23:00:00",
  "file_info": {
    "rows": 8760,
    "size_bytes": 456789,
    "compression": "gzip"
  },
  "provider_info": {
    "name": "binance",
    "version": "1.0"
  }
}
```

---

## Summary

**To refresh corrupted cache**:

1. **Quick way** (force re-download everything):
   ```bash
   cd c:/dev/cursor/e-trading
   python src/util/data_downloader.py
   ```

2. **Smart way** (only fix corrupted files):
   ```bash
   cd c:/dev/cursor/e-trading
   python src/data/utils/populate_cache.py --start-date 2020-01-01
   ```

Both scripts now use the **proper architecture** with automatic caching to `c:/data-cache/ohlcv/`!
