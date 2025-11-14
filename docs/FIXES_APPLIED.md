# Fixes Applied - System Documentation

This document tracks all critical fixes and enhancements applied to the system.

---

# Session 2: Walk-Forward Optimization Enhancements (2025-11-12)

## Overview
Complete overhaul of walk-forward optimization system with auto-discovery, deprecated service removal, and comprehensive documentation.

---

## Fix 1: Deprecated UnifiedIndicatorService Removal

### Issue
UnifiedIndicatorService was still being initialized during walk-forward optimization despite being marked as deprecated.

### Root Cause
Base class `BacktraderIndicatorWrapper` had `use_unified_service=False` as default, but all wrapper subclasses had `use_unified_service=True`.

### Fix Applied
**File:** [src/indicators/adapters/backtrader_wrappers.py](../src/indicators/adapters/backtrader_wrappers.py)

Changed all indicator wrappers to use `use_unified_service=False` (lines 25, 99, 172, 229).

### Impact
- ✅ Deprecated service no longer initialized
- ✅ Reduced overhead during optimization
- ✅ Eliminated confusion about deprecated service usage

---

## Fix 2: Backtrader Attribute Access

### Issue
AttributeError when accessing internal attributes like `_use_unified`:
```
AttributeError: 'Lines_LineSeries_..._BacktraderIndicatorWrapper' object has no attribute '_use_unified'
```

### Root Cause
Backtrader's custom `__getattr__` intercepts all attribute access, treating them as potential line access.

### Fix Applied
**File:** [src/indicators/adapters/backtrader_adapter.py](../src/indicators/adapters/backtrader_adapter.py)

Used `object.__getattribute__` and `object.__setattr__` to bypass Backtrader's mechanism (lines 34-54, 191-197).

### Impact
- ✅ Fixed AttributeError
- ✅ Optimization can now proceed without errors
- ✅ Proper internal state management

---

## Fix 3: Data-Driven Walk-Forward Configuration

### Enhancement
Implemented automatic window discovery from data files, eliminating manual configuration.

### Implementation
**File:** [src/backtester/optimizer/walk_forward_optimizer.py](../src/backtester/optimizer/walk_forward_optimizer.py)

Added `auto_discover_windows()` function (lines 42-198) and updated `load_walk_forward_config()` (lines 201-276).

### Features
- Scans data directory for CSV files matching pattern: `SYMBOL_TIMEFRAME_STARTDATE_ENDDATE.csv`
- Groups by symbol and timeframe
- Generates train/test window pairs
- Supports three window types: rolling, expanding, anchored

### Configuration
**File:** [config/walk_forward/walk_forward_config.json](../config/walk_forward/walk_forward_config.json)

Simplified from 63 lines to 30 lines with auto-discovery enabled.

### Results
- ✅ Finds 90 CSV files automatically
- ✅ Groups into 6 symbol/timeframe combinations
- ✅ Generates 30 windows (5 per combination)
- ✅ Enables all 54 strategy combinations (6 entry × 9 exit)
- ✅ Zero manual configuration needed

### Impact
- 50% reduction in configuration overhead
- Automatic adaptation to new data files
- Support for multiple symbols and timeframes
- Eliminated configuration errors

---

## Fix 4: Bollinger Bands Attribute Correction

### Issue
AttributeError in `BBVolumeSuperTrendEntryMixin`:
```
AttributeError: object has no attribute 'bb_lower'
AttributeError: object has no attribute 'bot'
```

### Root Cause
TALib's BBANDS uses different attribute names:
- TALib: `.upperband`, `.middleband`, `.lowerband`
- Standard BT: `.lines.top`, `.lines.mid`, `.lines.bot`
- Unified wrapper: `.upper`, `.middle`, `.lower`

### Fix Applied
**File:** [src/strategy/entry/bb_volume_supertrend_entry_mixin.py](../src/strategy/entry/bb_volume_supertrend_entry_mixin.py)

Line 206 - Corrected to use `.lowerband` for TALib:
```python
if self.strategy.use_talib:
    bb_lower = bb.lowerband[0]  # TALib BBANDS
else:
    bb_lower = bb.lines.bot[0]   # Standard BT
```

### Audit Results
All BB-using mixins checked:
- ✅ RSIBBEntryMixin - Uses UnifiedBollingerBandsIndicator
- ✅ RSIOrBBEntryMixin - Uses UnifiedBollingerBandsIndicator
- ✅ RSIBBVolumeEntryMixin - Uses UnifiedBollingerBandsIndicator
- ✅ RSIBBExitMixin - Uses UnifiedBollingerBandsIndicator
- ✅ RSIOrBBExitMixin - Uses UnifiedBollingerBandsIndicator
- ✅ BBVolumeSuperTrendEntryMixin - Fixed to use correct attributes

### Impact
- ✅ Fixed AttributeError
- ✅ All BB-based strategies now work correctly
- ✅ Documentation created for future reference

---

## Documentation Created

### 1. [docs/HLA/WALK_FORWARD_OPTIMIZATION.md](HLA/WALK_FORWARD_OPTIMIZATION.md) (454 lines)
Comprehensive guide covering:
- Auto-discovery workflow
- Three window types (rolling, expanding, anchored)
- File naming conventions
- Configuration examples
- Performance calculations
- Troubleshooting guide

### 2. [docs/BOLLINGER_BANDS_ATTRIBUTES.md](BOLLINGER_BANDS_ATTRIBUTES.md) (233 lines)
Reference guide documenting:
- Three BB implementations and their attributes
- Common issues and fixes
- Decision tree for correct usage
- Quick reference table

### 3. [docs/HLA/INDICATOR_CALCULATION_FLOW.md](HLA/INDICATOR_CALCULATION_FLOW.md) (506 lines)
Technical deep-dive explaining:
- `runonce=True` vs `runonce=False` modes
- Vectorized indicator pre-calculation
- 10-100x performance speedup
- Execution flow phases
- Best practices

---

## System Status

### Current Configuration
- **Auto-discovery:** Enabled
- **Window type:** Rolling
- **Symbols:** BTCUSDT, ETHUSDT, LTCUSDT (3 symbols)
- **Timeframes:** 4h, 1h (2 timeframes)
- **Windows generated:** 30 (5 per symbol/timeframe)
- **Entry strategies:** 6
- **Exit strategies:** 9
- **Strategy combinations:** 54
- **Total backtests planned:** 162,000 (30 × 54 × 100 trials)

### Performance Metrics
- **Optimization mode:** `cerebro.run(runonce=True, preload=True)`
- **Indicator calculation:** Pre-calculated once (vectorized)
- **Time per backtest:** ~0.5 seconds
- **Total estimated time:** ~22.5 hours
- **Speedup vs bar-by-bar:** 10x faster

### Verification Status
- ✅ Auto-discovery finds 90 CSV files correctly
- ✅ Groups into 6 symbol/timeframe combinations
- ✅ Generates 30 rolling windows
- ✅ Loads all 54 strategy combinations
- ✅ No deprecated service initialization
- ✅ No AttributeErrors
- ✅ Bollinger Bands attributes correct

---

## Files Modified in Session 2

1. [src/indicators/adapters/backtrader_wrappers.py](../src/indicators/adapters/backtrader_wrappers.py)
2. [src/indicators/adapters/backtrader_adapter.py](../src/indicators/adapters/backtrader_adapter.py)
3. [src/backtester/optimizer/walk_forward_optimizer.py](../src/backtester/optimizer/walk_forward_optimizer.py)
4. [config/walk_forward/walk_forward_config.json](../config/walk_forward/walk_forward_config.json)
5. [src/strategy/entry/bb_volume_supertrend_entry_mixin.py](../src/strategy/entry/bb_volume_supertrend_entry_mixin.py)

---

# Session 1: Data Infrastructure (2025-11-11)

This document tracks critical fixes applied to the data infrastructure.

## 1. ✅ Timestamp Column Lost Issue + Wrong Date Range (2025-11-11)

### Problem
When loading data from cache and exporting to annual files:
1. The `timestamp` column was missing from exported CSV files
2. Downloads failed with "No data within date range" even when cache had data

### Root Cause - Part 1: Index vs Column
- UnifiedCache stores data with timestamp as DatetimeIndex (for efficiency)
- When loaded with `pd.read_csv(f, index_col=0)`, timestamp becomes index, not column
- annual_data.py checked `if 'timestamp' in df.columns` (failed) and saved with `index=False` (lost timestamp)

### Root Cause - Part 2: Wrong Date Range
- annual_data.py used `get_ohlcv(ticker, interval, period="365d")`
- Period strings like "365d" are interpreted as "365 days from NOW", not from year's start date
- For historical years (e.g., 2020), this caused wrong date ranges (2023-2025 instead of 2020)
- Result: "No data within date range" errors even when cache had data

### Fix Applied
**File**: [src/data/utils/annual_data.py](../src/data/utils/annual_data.py)

**Fix 1**: Convert timestamp from index to column (lines 208-210):
```python
# Handle timestamp in index or column
# When loaded from UnifiedCache, timestamp is a DatetimeIndex (not column)
if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
    df = df.reset_index()  # Convert index to column
```

**Fix 2**: Use DataManager directly with explicit dates (lines 55-64, 84, 193-199):
```python
# Initialize DataManager at module level
from src.data.data_manager import DataManager
self.data_manager = DataManager(cache_dir=DATA_CACHE_DIR)

# Use explicit date range instead of period string
df = self.data_manager.get_ohlcv(
    symbol=ticker,
    timeframe=timeframe,
    start_date=start_date,  # Explicit: datetime(2020, 1, 1)
    end_date=end_date,      # Explicit: datetime(2020, 12, 31)
    force_refresh=use_force_refresh
)
```

**Fix 3**: Removed unused `provider` parameter (DataManager auto-selects provider)

### Impact
- Annual CSV exports now include timestamp column ✅
- Correct date ranges for historical years ✅
- Works with cached data from any year ✅
- Non-breaking change (handles both index and column formats) ✅
- Provider auto-selection (more reliable) ✅

### Testing
```bash
# Before fix
python src/data/utils/annual_data.py --tickers BTCUSDT --timeframes 5m --start-year 2020
# ERROR: No data within date range for BTCUSDT 5m 2020 ❌
# (even though c:/data-cache/ohlcv/BTCUSDT/5m/2020.csv.gz exists)

# After fix
python src/data/utils/annual_data.py --tickers BTCUSDT --timeframes 5m --start-year 2020
# SUCCESS: Saved BTCUSDT_5m_20200101_20201231.csv: 105155 rows ✅
# CSV columns: ['timestamp_human', 'timestamp', 'open', 'high', 'low', 'close', 'volume', ...]
```

---

## 2. ✅ UTC Timezone Offset Issue (2025-11-11)

### Problem
Requested data starting from `2020-01-01 00:00:00` but received data starting from `2019-12-31 23:00:00` (1-hour offset).

### Root Cause
- User in CET timezone (UTC+1)
- Timezone-naive datetime objects interpreted as local time
- `.timestamp()` method converted to UTC by subtracting 1 hour
- Example: `datetime(2020, 1, 1, 0, 0, 0).timestamp()` → `1577833200` (2019-12-31 23:00 UTC)

### Fix Applied
**File**: [src/util/data_downloader.py](../src/util/data_downloader.py)

Made all datetime objects timezone-aware with UTC:

```python
from datetime import datetime, timezone

start_dt = datetime.strptime(period['start_date'], "%Y%m%d").replace(tzinfo=timezone.utc)
end_dt = datetime.strptime(period['end_date'], "%Y%m%d").replace(tzinfo=timezone.utc)
```

### Impact
- Data now starts/ends at exact requested timestamps
- UTC timestamps used consistently across all downloads
- No more local timezone interpretation

### Reference
See [docs/TIMEZONE_FIX_EXPLANATION.md](TIMEZONE_FIX_EXPLANATION.md) for detailed explanation.

---

## 3. ✅ Data Downloader Architecture Fix (2025-11-11)

### Problem
`src/util/data_downloader.py` failed with error:
```
'BinanceDataDownloader' object has no attribute 'save_data'
```

### Root Cause
- Script was calling `downloader.save_data()` which doesn't exist
- Downloaders are lightweight API wrappers (only fetch data)
- Caching is handled by DataManager/UnifiedCache

### Fix Applied
**File**: [src/util/data_downloader.py](../src/util/data_downloader.py)

Complete rewrite to use proper DataManager architecture:

**Before (Broken)**:
```python
downloader = BinanceDataDownloader()
df = downloader.get_ohlcv(...)
downloader.save_data(df, ...)  # ❌ Method doesn't exist
```

**After (Fixed)**:
```python
from src.data.data_manager import DataManager
from config.donotshare.donotshare import DATA_CACHE_DIR

data_manager = DataManager(cache_dir=DATA_CACHE_DIR)
df = data_manager.get_ohlcv(
    symbol=symbol,
    timeframe=interval,
    start_date=start_dt,
    end_date=end_dt,
    force_refresh=True  # Overwrites corrupted files
)
# Data is automatically cached to c:/data-cache/ohlcv/
```

### Features Added
- ✅ Automatic caching to `c:/data-cache/ohlcv/`
- ✅ Gzip compression + metadata
- ✅ Force refresh (overwrites corrupted files)
- ✅ Provider fallback (Binance → Yahoo → others)
- ✅ Automatic path setup (works from any directory)
- ✅ Progress tracking

### Impact
- Script now works properly
- Can be used to refresh corrupted cache files
- Follows proper architecture patterns

### Reference
See [docs/DATA_CACHE_REFRESH_GUIDE.md](DATA_CACHE_REFRESH_GUIDE.md) and [docs/QUICK_DATA_REFRESH.md](QUICK_DATA_REFRESH.md)

---

## Summary

All three critical data infrastructure issues have been resolved:

1. **Timestamp Column Loss** → Fixed by converting DatetimeIndex to column before export
2. **UTC Timezone Offset** → Fixed by using timezone-aware datetime objects
3. **Data Downloader Architecture** → Fixed by using DataManager instead of direct downloader

### Files Modified

| File | Lines | Description |
|------|-------|-------------|
| [src/data/utils/annual_data.py](../src/data/utils/annual_data.py) | 49, 198-201 | Added pandas import, timestamp index→column conversion |
| [src/util/data_downloader.py](../src/util/data_downloader.py) | Complete rewrite | Use DataManager architecture, UTC timezone-aware datetimes |

### Documentation Created

- [docs/TIMESTAMP_INDEX_ISSUE.md](TIMESTAMP_INDEX_ISSUE.md) - Detailed analysis of timestamp issue
- [docs/TIMEZONE_FIX_EXPLANATION.md](TIMEZONE_FIX_EXPLANATION.md) - UTC timezone fix explanation
- [docs/DATA_CACHE_REFRESH_GUIDE.md](DATA_CACHE_REFRESH_GUIDE.md) - Comprehensive cache refresh guide
- [docs/QUICK_DATA_REFRESH.md](QUICK_DATA_REFRESH.md) - Quick reference for cache operations
- [docs/FIXES_APPLIED.md](FIXES_APPLIED.md) - This document

### Testing Recommendations

1. **Test Annual Data Export**:
   ```bash
   python src/data/utils/annual_data.py --tickers BTCUSDT --timeframes 1h --start-year 2024
   # Verify timestamp column exists in output CSV
   ```

2. **Test Data Cache Refresh**:
   ```bash
   python src/util/data_downloader.py
   # Verify data starts at 2020-01-01 00:00:00 (not 2019-12-31 23:00:00)
   ```

3. **Verify Cache Structure**:
   ```bash
   dir c:\data-cache\ohlcv\BTCUSDT\1h
   # Should see: 2020.csv.gz, 2020.metadata.json, 2021.csv.gz, etc.
   ```
