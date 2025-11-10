# Timezone Issue Fix - Data Downloader

## Problem

When requesting data starting from `2020-01-01`, the downloaded data was starting from `2019-12-31 23:00:00`.

## Root Cause

### The Issue

Your system is in **Central European Time (CET)**, which is **UTC+1**.

When you created datetime objects without timezone information:
```python
start_dt = datetime.strptime('20200101', '%Y%m%d')
# Result: 2020-01-01 00:00:00 (timezone-naive, assumes local timezone)
```

Python's `.timestamp()` method treats timezone-naive datetime objects as **local time**:
```python
start_dt.timestamp()
# Returns: 1577833200.0 (this is 2020-01-01 00:00:00 in CET)
# Which equals: 2019-12-31 23:00:00 in UTC
```

### Why This Happens

1. **Your request**: `2020-01-01 00:00:00` (no timezone specified)
2. **Python interprets as**: `2020-01-01 00:00:00 CET` (your local timezone)
3. **Converts to timestamp**: Subtracts 1 hour to get UTC
4. **Binance API receives**: Request for `2019-12-31 23:00:00 UTC`
5. **Data starts at**: `2019-12-31 23:00:00` (UTC)

## Solution

### The Fix

Make datetime objects **timezone-aware** by explicitly setting them to UTC:

```python
from datetime import datetime, timezone

start_dt = datetime.strptime('20200101', '%Y%m%d').replace(tzinfo=timezone.utc)
# Result: 2020-01-01 00:00:00+00:00 (timezone-aware UTC)
```

Now Python's `.timestamp()` knows this is already in UTC:
```python
start_dt.timestamp()
# Returns: 1577836800.0 (this is 2020-01-01 00:00:00 in UTC)
# No conversion needed!
```

### What Changed in data_downloader.py

**Before (broken)**:
```python
start_dt = datetime.strptime(period['start_date'], "%Y%m%d")
end_dt = datetime.strptime(period['end_date'], "%Y%m%d")
```

**After (fixed)**:
```python
start_dt = datetime.strptime(period['start_date'], "%Y%m%d").replace(tzinfo=timezone.utc)
end_dt = datetime.strptime(period['end_date'], "%Y%m%d").replace(tzinfo=timezone.utc)
```

## Verification

### Test the Fix

```python
from datetime import datetime, timezone

# Broken approach (timezone-naive)
naive_dt = datetime.strptime('20200101', '%Y%m%d')
print("Naive:", naive_dt)
print("Timestamp:", naive_dt.timestamp())
# Output:
# Naive: 2020-01-01 00:00:00
# Timestamp: 1577833200.0  (wrong - this is 2019-12-31 23:00:00 UTC)

# Fixed approach (timezone-aware UTC)
utc_dt = datetime.strptime('20200101', '%Y%m%d').replace(tzinfo=timezone.utc)
print("UTC:", utc_dt)
print("Timestamp:", utc_dt.timestamp())
# Output:
# UTC: 2020-01-01 00:00:00+00:00
# Timestamp: 1577836800.0  (correct - this is 2020-01-01 00:00:00 UTC)

# Difference
print("Difference:", (utc_dt.timestamp() - naive_dt.timestamp()) / 3600, "hours")
# Output: 1.0 hours
```

### Verify Data Now Starts Correctly

After running the fixed script:
```python
from src.data.data_manager import DataManager
from datetime import datetime, timezone

dm = DataManager()
df = dm.get_ohlcv(
    'BTCUSDT',
    '1h',
    datetime(2020, 1, 1, tzinfo=timezone.utc),
    datetime(2020, 1, 2, tzinfo=timezone.utc)
)

print("First timestamp:", df.index[0])
# Should now be: 2020-01-01 00:00:00 (not 2019-12-31 23:00:00)
```

## Impact

### Files Fixed
- ✅ `src/util/data_downloader.py` - Added timezone-aware datetime objects

### What This Fixes
- ✅ Data now starts exactly at requested date/time (in UTC)
- ✅ Consistent with Binance API expectations (UTC timezone)
- ✅ No more 1-hour offset for users in CET/UTC+1
- ✅ Works correctly regardless of your local timezone

### Who This Affects
- **CET/CEST users** (UTC+1/UTC+2): Had 1-2 hour offset
- **EST/PST users** (UTC-5/UTC-8): Had 5-8 hour offset
- **UTC users**: No offset, but now explicitly correct

## Best Practices

### Always Use Timezone-Aware Datetimes for APIs

**Wrong** (timezone-naive):
```python
start = datetime(2020, 1, 1)  # Ambiguous - what timezone?
```

**Correct** (timezone-aware):
```python
from datetime import timezone
start = datetime(2020, 1, 1, tzinfo=timezone.utc)  # Explicit UTC
```

### Alternative: Use pandas Timestamp

```python
import pandas as pd

start = pd.Timestamp('2020-01-01', tz='UTC')
end = pd.Timestamp('2020-12-31', tz='UTC')
```

### Why UTC for Crypto?

1. **Binance API uses UTC** - All timestamps are UTC
2. **24/7 markets** - No concept of "market close" tied to a timezone
3. **Global trading** - Traders in all timezones need consistent data
4. **Avoids DST issues** - UTC doesn't observe daylight saving time

## Additional Context

### How Binance API Works

Binance's `get_historical_klines` expects:
- **start_timestamp**: Milliseconds since Unix epoch (1970-01-01 00:00:00 UTC)
- **end_timestamp**: Milliseconds since Unix epoch (1970-01-01 00:00:00 UTC)

When you pass a timezone-naive datetime:
1. Python converts it using your **local timezone**
2. The timestamp sent to Binance is **offset by your timezone**
3. Binance returns data starting from that **offset time in UTC**

### Timestamp Conversion Flow

```
Your request:     2020-01-01 00:00:00 (naive)
                         ↓
Python interprets: 2020-01-01 00:00:00 CET
                         ↓
Convert to epoch:  1577833200 (seconds since 1970-01-01 UTC)
                         ↓
This equals:      2019-12-31 23:00:00 UTC
                         ↓
Binance API:      Returns data starting from 2019-12-31 23:00:00 UTC
                         ↓
Your data:        First row is 2019-12-31 23:00:00 ❌
```

**With timezone-aware UTC:**
```
Your request:     2020-01-01 00:00:00 UTC (aware)
                         ↓
Already UTC:      2020-01-01 00:00:00 UTC
                         ↓
Convert to epoch:  1577836800 (seconds since 1970-01-01 UTC)
                         ↓
This equals:      2020-01-01 00:00:00 UTC
                         ↓
Binance API:      Returns data starting from 2020-01-01 00:00:00 UTC
                         ↓
Your data:        First row is 2020-01-01 00:00:00 ✅
```

## Summary

**Problem**: Dates were interpreted as local timezone (CET), causing 1-hour offset

**Solution**: Explicitly set timezone to UTC using `.replace(tzinfo=timezone.utc)`

**Result**: Data now starts exactly at the requested UTC time

**Lesson**: Always use timezone-aware datetime objects when working with APIs!
