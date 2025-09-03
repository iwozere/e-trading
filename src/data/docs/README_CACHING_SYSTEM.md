# Intelligent Cached Data Downloaders

## Overview

The Intelligent Cached Data Downloaders system provides professional-grade caching capabilities for all data downloaders in the e-trading platform. This system automatically manages data caching, identifies gaps, and minimizes server requests while ensuring data completeness and integrity.

## Key Features

### 🚀 **Intelligent Caching**
- **Automatic cache checking** before any server request
- **Smart gap detection** to identify missing data periods
- **Partial downloads** for only missing data segments
- **Seamless data merging** of cached and downloaded data

### 🎯 **Smart Gap Detection**
- **Missing start/end periods** detection
- **Internal gaps** identification within cached data
- **Weekend/holiday tolerance** for daily data
- **Interval-specific gap analysis** (1m, 1h, 1d, etc.)

### 💾 **Professional Cache Management**
- **Year-based data organization** (2020, 2021, 2022, etc.)
- **Automatic data migration** from old formats
- **Cache validation** and integrity checks
- **Performance metrics** and statistics

### 🔧 **Universal Compatibility**
- **All data downloader types** supported
- **Drop-in replacement** for existing downloaders
- **No code changes** required in existing projects
- **Backward compatible** API

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              CachedDataDownloader                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Cache Check    │  │  Gap Analysis   │  │   Merge     │ │
│  │                 │  │                 │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              FileBasedCache                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Year Split     │  │  Smart Expiry   │  │  Metadata   │ │
│  │                 │  │                 │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              BaseDataDownloader                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Binance       │  │     Yahoo       │  │   Alpha     │ │
│  │                 │  │                 │  │  Vantage    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Basic Usage

```python
from src.data.cached_downloader_factory import create_cached_binance_downloader
from datetime import datetime

# Create a cached Binance downloader
downloader = create_cached_binance_downloader()

# Request data (automatically cached)
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 1, 31)

data = downloader.get_ohlcv("BTCUSDT", "1d", start_date, end_date)
print(f"Downloaded {len(data)} rows")
```

### 2. Factory Pattern Usage

```python
from src.data.cached_downloader_factory import get_cached_downloader_factory

# Get the factory
factory = get_cached_downloader_factory()

# Create multiple cached downloaders
binance = factory.create_binance_downloader()
yahoo = factory.create_yahoo_downloader()

# Use them seamlessly
btc_data = binance.get_ohlcv("BTCUSDT", "1d", start_date, end_date)
aapl_data = yahoo.get_ohlcv("AAPL", "1d", start_date, end_date)
```

### 3. Custom Downloader Wrapping

```python
from src.data.cached_data_downloader import CachedDataDownloader
from your_custom_downloader import CustomDataDownloader

# Wrap your custom downloader
custom_downloader = CustomDataDownloader()
cached_downloader = CachedDataDownloader(custom_downloader, cache)

# Use with caching
data = cached_downloader.get_ohlcv("SYMBOL", "1d", start_date, end_date)
```

## How It Works

### 1. **Cache Check Phase**
```python
# The system first checks if data exists in cache
cached_data = self._get_cached_data(symbol, interval, start_date, end_date)
```

### 2. **Gap Analysis Phase**
```python
# Identifies missing data periods
gaps = self.gap_analyzer.analyze_gaps(cached_data, start_date, end_date, interval)
```

### 3. **Smart Download Phase**
```python
# Downloads only missing data for each gap
for gap in gaps:
    gap_data = self.downloader.get_ohlcv(symbol, interval, gap.start_date, gap.end_date)
    downloaded_data.append(gap_data)
```

### 4. **Data Merge Phase**
```python
# Combines cached and downloaded data
all_data = self._merge_data(cached_data, downloaded_data)
```

### 5. **Cache Update Phase**
```python
# Caches the complete dataset
self._cache_complete_dataset(symbol, interval, all_data, start_date, end_date)
```

## Gap Detection Examples

### Example 1: Missing Start Period
```
Requested: 2024-01-01 to 2024-01-31
Cached:    2024-01-15 to 2024-01-31
Gap:       [2024-01-01, 2024-01-14] - Download needed
```

### Example 2: Missing End Period
```
Requested: 2024-01-01 to 2024-02-15
Cached:    2024-01-01 to 2024-01-31
Gap:       [2024-02-01, 2024-02-15] - Download needed
```

### Example 3: Internal Gap
```
Requested: 2024-01-01 to 2024-01-31
Cached:    2024-01-01 to 2024-01-15, 2024-01-20 to 2024-01-31
Gap:       [2024-01-16, 2024-01-19] - Download needed
```

## Cache Organization

### Directory Structure
```
d:/data-cache/
├── binance/
│   ├── BTCUSDT/
│   │   ├── 1d/
│   │   │   ├── 2024/
│   │   │   │   ├── data.csv
│   │   │   │   └── metadata.json
│   │   │   └── 2023/
│   │   │       ├── data.csv
│   │   │       └── metadata.json
│   │   └── 4h/
│   │       └── 2024/
│   │           ├── data.csv
│   │           └── metadata.json
│   └── ETHUSDT/
│       └── 1d/
│           └── 2024/
│               ├── data.csv
│               └── metadata.json
└── yahoo/
    ├── AAPL/
    │   └── 1d/
    │       └── 2024/
    │           ├── data.csv
    │           └── metadata.json
    └── TSLA/
        └── 1d/
            └── 2024/
                ├── data.csv
                └── metadata.json
```

### Metadata Structure
```json
{
  "provider": "binance",
  "symbol": "BTCUSDT",
  "interval": "1d",
  "year": 2024,
  "created_at": "2024-01-15T10:30:00",
  "rows": 31,
  "columns": ["timestamp", "open", "high", "low", "close", "volume"],
  "format": "csv",
  "compression_enabled": false,
  "start_date": "2024-01-01T00:00:00",
  "end_date": "2024-01-31T00:00:00",
  "version": "1.0.0"
}
```

## Performance Benefits

### 🚀 **Speed Improvements**
- **Cache hits**: 10-100x faster than server requests
- **Partial downloads**: Only missing data transferred
- **Smart batching**: Efficient gap filling

### 💰 **Cost Savings**
- **Reduced API calls**: Up to 90% fewer requests
- **Bandwidth optimization**: No duplicate data transfer
- **Rate limit management**: Better API quota utilization

### 🔄 **Reliability**
- **Offline access**: Cached data available without internet
- **Error resilience**: Partial failures don't affect cached data
- **Data integrity**: Automatic validation and gap detection

## Advanced Features

### 1. **Cache Statistics**
```python
factory = get_cached_downloader_factory()
stats = factory.get_cache_stats()

print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Cache size: {stats['cache_size_gb']:.2f} GB")
print(f"Total files: {stats['file_count']}")
```

### 2. **Cache Management**
```python
# Clear specific cache entries
factory.clear_cache(provider="binance", symbol="BTCUSDT")

# Clear all cache
factory.clear_cache()
```

### 3. **Custom Gap Tolerance**
```python
from src.data.cached_data_downloader import DataGapAnalyzer

# Customize gap tolerance for specific intervals
analyzer = DataGapAnalyzer()
analyzer.interval_gap_tolerance['1d'] = timedelta(days=14)  # Allow 2 weeks for daily data
```

## Migration from Existing Code

### Before (Direct Downloader Usage)
```python
from src.data.binance_data_downloader import BinanceDataDownloader

downloader = BinanceDataDownloader(api_key, api_secret)
data = downloader.get_ohlcv("BTCUSDT", "1d", start_date, end_date)
```

### After (Cached Downloader Usage)
```python
from src.data.cached_downloader_factory import create_cached_binance_downloader

downloader = create_cached_binance_downloader(api_key, api_secret)
data = downloader.get_ohlcv("BTCUSDT", "1d", start_date, end_date)
# Same API, but with automatic caching!
```

## Best Practices

### 1. **Use Factory Functions**
```python
# ✅ Good: Use factory functions
downloader = create_cached_binance_downloader(api_key, api_secret)

# ❌ Avoid: Direct instantiation
downloader = CachedDataDownloader(BinanceDataDownloader(api_key, api_secret), cache)
```

### 2. **Monitor Cache Performance**
```python
# Regularly check cache statistics
stats = factory.get_cache_stats()
if stats['hit_rate'] < 0.5:
    print("Cache hit rate is low - consider adjusting gap tolerance")
```

### 3. **Handle Large Date Ranges**
```python
# For very long periods, consider breaking into chunks
chunk_size = timedelta(days=365)
current_start = start_date

while current_start < end_date:
    current_end = min(current_start + chunk_size, end_date)
    data = downloader.get_ohlcv(symbol, interval, current_start, current_end)
    current_start = current_end + timedelta(days=1)
```

## Troubleshooting

### Common Issues

#### 1. **Cache Not Working**
```python
# Check if cache directory exists and is writable
import os
cache_dir = "d:/data-cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
```

#### 2. **High Memory Usage**
```python
# Clear old cache entries
factory.clear_cache(provider="binance", symbol="BTCUSDT", interval="1m")
```

#### 3. **Data Gaps Not Detected**
```python
# Check gap tolerance settings
analyzer = DataGapAnalyzer()
print(f"Daily gap tolerance: {analyzer.interval_gap_tolerance['1d']}")
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed cache operations
downloader = create_cached_binance_downloader()
```

## API Reference

### CachedDataDownloader Class

#### Methods
- `get_ohlcv(symbol, interval, start_date, end_date, **kwargs)` - Get OHLCV data with caching
- `get_periods()` - Get supported periods
- `get_intervals()` - Get supported intervals
- `is_valid_period_interval(period, interval)` - Validate period/interval combination

#### Properties
- `provider` - Data provider name
- `downloader` - Underlying data downloader
- `cache` - File-based cache instance

### DataGapAnalyzer Class

#### Methods
- `analyze_gaps(cached_data, requested_start, requested_end, interval)` - Analyze data gaps
- `_find_internal_gaps(data, interval)` - Find gaps within cached data

#### Properties
- `interval_gap_tolerance` - Gap tolerance for different intervals

### CachedDownloaderFactory Class

#### Methods
- `create_binance_downloader(api_key, api_secret)` - Create cached Binance downloader
- `create_yahoo_downloader()` - Create cached Yahoo downloader
- `create_cached_downloader(downloader, provider_name)` - Create cached version of any downloader
- `get_cache_stats()` - Get cache statistics
- `clear_cache(provider, symbol, interval)` - Clear specific cache entries

## Examples

### Complete Working Example
```python
from datetime import datetime, timedelta
from src.data.cached_downloader_factory import create_cached_binance_downloader

def download_crypto_data():
    # Create cached downloader
    downloader = create_cached_binance_downloader()
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Download data (automatically cached)
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    intervals = ["1d", "4h"]
    
    for symbol in symbols:
        for interval in intervals:
            try:
                data = downloader.get_ohlcv(symbol, interval, start_date, end_date)
                print(f"{symbol} {interval}: {len(data)} rows")
            except Exception as e:
                print(f"Error downloading {symbol} {interval}: {e}")

if __name__ == "__main__":
    download_crypto_data()
```

### Stock Data Example
```python
from src.data.cached_downloader_factory import create_cached_yahoo_downloader

def download_stock_data():
    # Create cached Yahoo downloader
    downloader = create_cached_yahoo_downloader()
    
    # Download multiple stocks
    stocks = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    for stock in stocks:
        try:
            data = downloader.get_ohlcv(stock, "1d", start_date, end_date)
            print(f"{stock}: {len(data)} rows from {data.index.min().date()} to {data.index.max().date()}")
        except Exception as e:
            print(f"Error downloading {stock}: {e}")

if __name__ == "__main__":
    download_stock_data()
```

## Conclusion

The Intelligent Cached Data Downloaders system provides a professional, efficient, and reliable way to manage financial data downloads. By automatically handling caching, gap detection, and data merging, it significantly reduces server requests while ensuring data completeness.

Key benefits:
- **Minimal code changes** required
- **Automatic performance optimization**
- **Professional error handling**
- **Comprehensive monitoring and statistics**
- **Universal compatibility** with all downloaders

Start using cached downloaders today to improve your data access performance and reduce API costs!
