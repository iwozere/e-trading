# Data Module

This module provides comprehensive data downloading, caching, and live feed capabilities for the e-trading platform. It supports multiple data sources with intelligent provider selection and a unified cache system for both historical data and real-time market feeds.

## Overview

The data module consists of:
- **Unified Cache System**: Intelligent file-based caching with gzip compression
- **Data Downloaders**: For fetching historical OHLCV data and fundamental information
- **Live Data Feeds**: For real-time market data streaming
- **Provider Selection**: Automatic selection of the best data provider based on symbol and timeframe
- **Data Validation**: Comprehensive data quality checks and validation

## Unified Cache System

### New Cache Structure

The unified cache system provides a simplified, efficient structure for storing historical data with proper year splitting:

```
data-cache/
├── BTCUSDT/
│   ├── 5m/
│   │   ├── 2025.csv.gz          # Data for 2025 only
│   │   ├── 2025.metadata.json   # Metadata for 2025
│   │   ├── 2024.csv.gz          # Data for 2024 only
│   │   └── 2024.metadata.json   # Metadata for 2024
│   └── 1h/
├── AAPL/
│   ├── 5m/
│   └── 1d/
└── _metadata/
    ├── symbols.json
    ├── providers.json
    └── quality_scores.json
```

### Key Features

- **Simplified Structure**: `symbol/timeframe/` instead of `provider/symbol/timeframe/year/`
- **Year Splitting**: Data is automatically split by year into separate files
- **Gzip Compression**: All CSV files are compressed for efficient storage
- **Provider Metadata**: Data source information embedded in metadata files
- **Intelligent Provider Selection**: Automatic selection of best provider based on symbol type and timeframe
- **Data Validation**: Built-in OHLCV data validation and quality scoring

### Usage

```python
from src.data.cache.unified_cache import configure_unified_cache

# Configure cache
cache = configure_unified_cache(cache_dir="./data-cache")

# Store data
cache.put(df, "BTCUSDT", "5m", start_date, end_date, provider="binance")

# Retrieve data
df = cache.get("BTCUSDT", "5m", start_date, end_date)
```

## Intelligent Provider Selection

The system automatically selects the best data provider based on symbol type and timeframe:

### Provider Selection Logic

- **Cryptocurrency Symbols**: Always use Binance
- **Stock Symbols (Daily)**: Use Yahoo Finance
- **Stock Symbols (Intraday)**: Use Alpha Vantage (full historical data, no 60-day limit)

### Usage

```python
from src.common.ticker_classifier import TickerClassifier

classifier = TickerClassifier()

# Get best provider for a symbol and timeframe
provider = classifier.get_provider_for_interval("BTCUSDT", "5m")  # Returns "binance"
provider = classifier.get_provider_for_interval("AAPL", "1d")     # Returns "yfinance"
provider = classifier.get_provider_for_interval("AAPL", "5m")     # Returns "alpha_vantage"

# Get detailed provider configuration
config = classifier.get_data_provider_config("AAPL", "5m")
print(f"Provider: {config['best_provider']}")
print(f"Reason: {config['reason']}")
```

## Data Downloaders

### Base Data Downloader

All data downloaders inherit from `BaseDataDownloader` which provides:
- Common file management operations
- Standardized data formats
- Error handling and logging
- Abstract `get_fundamentals()` method

### Available Data Sources

#### 1. Binance Data Downloader (`BinanceDataDownloader`)

**Best for:** Cryptocurrency data with comprehensive historical coverage

**Capabilities:**
- ✅ **Cryptocurrency OHLCV**: All major crypto pairs
- ✅ **Multiple Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
- ✅ **High Rate Limits**: 1200 requests per minute
- ✅ **Real-time Data**: WebSocket support for live feeds

**Data Quality:** Excellent - Direct from exchange
**Rate Limits:** 1200 requests/minute (free tier)
**Coverage:** Cryptocurrencies only

```python
from src.data.binance_data_downloader import BinanceDataDownloader

downloader = BinanceDataDownloader()
df = downloader.get_ohlcv("BTCUSDT", "5m", "2023-01-01", "2023-12-31")
```

#### 2. Yahoo Finance Data Downloader (`YahooDataDownloader`)

**Best for:** Stock fundamental data and daily historical data

**Capabilities:**
- ✅ **PE Ratio**: Trailing and forward PE ratios
- ✅ **Financial Ratios**: P/B, ROE, ROA, debt/equity, current ratio, quick ratio
- ✅ **Growth Metrics**: Revenue growth, net income growth
- ✅ **Company Information**: Name, sector, industry, country, exchange
- ✅ **Market Data**: Market cap, current price, shares outstanding
- ✅ **Daily Data**: Comprehensive historical daily data
- ⚠️ **Intraday Limitation**: Only 60 days of intraday data

**Data Quality:** High - Comprehensive fundamental data
**Rate Limits:** None for basic usage
**Coverage:** Global stocks and ETFs

```python
from src.data.yahoo_data_downloader import YahooDataDownloader

downloader = YahooDataDownloader()
fundamentals = downloader.get_fundamentals("AAPL")
df = downloader.get_ohlcv("AAPL", "1d", "2020-01-01", "2023-12-31")
```

#### 3. Alpha Vantage Data Downloader (`AlphaVantageDataDownloader`)

**Best for:** Full historical intraday stock data (no 60-day limit)

**Capabilities:**
- ✅ **Full Historical Data**: Complete intraday data history
- ✅ **Multiple Intervals**: 1m, 5m, 15m, 30m, 60m, 1h, 1d, 1w, 1M
- ✅ **Stock Data**: Comprehensive US and international stocks
- ✅ **Crypto Data**: Limited cryptocurrency support
- ✅ **Fundamental Data**: Company information and financial metrics

**Data Quality:** High - Professional-grade data
**Rate Limits:** 5 calls/minute, 500/day (free tier)
**Coverage:** Global stocks and limited crypto

```python
from src.data.alpha_vantage_data_downloader import AlphaVantageDataDownloader

downloader = AlphaVantageDataDownloader(api_key="YOUR_API_KEY")
df = downloader.get_ohlcv("AAPL", "5m", "2020-01-01", "2023-12-31")  # Full history!
```

## Cache Population Script

The `populate_cache.py` script provides an easy way to populate the cache with historical data:

### Basic Usage

```bash
# Populate cache with default symbols and timeframes
python src/data/cache/populate_cache.py

# Custom symbols and timeframes
python src/data/cache/populate_cache.py --symbols BTCUSDT,AAPL,GOOGL --intervals 5m,15m,1h,1d

# Specific date range
python src/data/cache/populate_cache.py --start-date 2020-01-01 --end-date 2023-12-31

# Custom cache directory
python src/data/cache/populate_cache.py --cache-dir ./my-cache
```

### Features

- **Intelligent Provider Selection**: Automatically chooses the best provider for each symbol/timeframe combination
- **Incremental Updates**: Only downloads missing data, skips existing recent data
- **Rate Limiting**: Built-in delays to respect API limits
- **Data Validation**: Validates all data before caching
- **Progress Tracking**: Shows detailed progress and statistics

### Example Output

```
🚀 Populating cache at: ./data-cache
📊 Symbols: BTCUSDT, AAPL, GOOGL
⏱️  Intervals: 5m, 15m, 1h, 1d
📅 Date range: 2020-01-01 to 2023-12-31
🔢 Total operations: 12

  ✅ Binance downloader initialized
  ✅ Yahoo downloader initialized
  ✅ Alpha Vantage downloader initialized
  ✅ Ticker classifier initialized
  📡 Available downloaders: binance, yahoo, alpha_vantage

🔄 Progress: 1/12 (8.3%)
📥 Downloading BTCUSDT 5m data...
  🔍 Ticker classification: BTCUSDT -> binance
  🎯 Best provider for 5m: binance
  💡 Reason: Crypto symbol detected
  📡 Using Binance data source for crypto symbol BTCUSDT
  ✅ Successfully downloaded from binance: 125,000 rows
  💾 Caching data to unified cache: BTCUSDT/5m/2020/...
  ✅ Downloaded and cached 125,000 rows
  📈 Quality score: 0.95
  📁 Cached to: BTCUSDT/5m/2020/
  🔗 Provider: binance
```

## Live Data Feeds

### Available Live Feeds

#### 1. Binance Live Feed (`BinanceLiveFeed`)

Real-time cryptocurrency data streaming from Binance.

**Features:**
- WebSocket-based real-time data
- Multiple symbol support
- Automatic reconnection
- Error handling

```python
from src.data.binance_live_feed import BinanceLiveFeed

feed = BinanceLiveFeed(["BTCUSDT", "ETHUSDT"])
feed.start()
```

#### 2. Yahoo Live Feed (`YahooLiveFeed`)

Real-time stock data streaming from Yahoo Finance.

**Features:**
- Real-time stock quotes
- Multiple symbol support
- WebSocket connection
- Automatic reconnection

```python
from src.data.yahoo_live_feed import YahooLiveFeed

feed = YahooLiveFeed(["AAPL", "GOOGL"])
feed.start()
```

## Data Validation

The system includes comprehensive data validation:

### OHLCV Data Validation

```python
from src.data.utils.validation import validate_ohlcv_data, get_data_quality_score

# Validate data
is_valid, errors = validate_ohlcv_data(df)
quality_score = get_data_quality_score(df)

print(f"Data valid: {is_valid}")
print(f"Quality score: {quality_score['quality_score']:.2f}")
```

### Validation Checks

- **Required Columns**: open, high, low, close, volume
- **Data Types**: Numeric validation for OHLCV columns
- **Logical Consistency**: high >= max(open, close), low <= min(open, close)
- **Timestamp Validation**: Proper datetime index, no duplicates
- **Data Gaps**: Detection of missing time periods
- **Quality Scoring**: Overall data quality assessment

## Cache Management

### Cache Statistics

```python
# Get cache statistics
stats = cache.get_stats()
print(f"Cache size: {stats['total_size_gb']:.2f} GB")
print(f"Files: {stats['files_count']}")
```

### Cache Operations

```python
# List available data
symbols = cache.list_symbols()
timeframes = cache.list_timeframes("BTCUSDT")
years = cache.list_years("BTCUSDT", "5m")

# Get data information
info = cache.get_data_info("BTCUSDT", "5m", 2023)
print(f"Provider: {info['data_source']}")
print(f"Quality: {info['data_quality']['score']}")
```

### Cache Cleanup

```python
# Remove old data (older than 365 days)
removed = cache.cleanup_old_data(max_age_days=365)
print(f"Removed {removed} old files")
```

## Migration Tools

### Migrate from Old Cache Structure

```bash
# Migrate from provider-based to unified cache
python src/data/cache/migrate_to_unified_cache.py --migrate

# Dry run to see what would be migrated
python src/data/cache/migrate_to_unified_cache.py --dry-run
```

### Cache Validation and Cleanup

```bash
# Validate cache files
python src/data/cache/cleanup_failed_cache.py --validate-only

# Clean up invalid files
python src/data/cache/cleanup_failed_cache.py --cleanup

# Validate and clean up
python src/data/cache/cleanup_failed_cache.py --validate-and-cleanup
```

## Error Handling

All data downloaders include comprehensive error handling:

```python
try:
    df = downloader.get_ohlcv("INVALID_SYMBOL", "1d", start_date, end_date)
except Exception as e:
    print(f"Error: {e}")
    # Returns None or empty DataFrame with error information
```

## Rate Limiting

The system respects API rate limits with built-in throttling:

- **Binance**: 1200 requests/minute (built-in throttling)
- **Yahoo Finance**: No limits for basic usage (built-in throttling)
- **Alpha Vantage**: 5 calls/minute, 500/day (built-in throttling)
- **Cache Population**: 0.2 second delay between operations

## Configuration

### Environment Variables

```bash
# Alpha Vantage API Key (for intraday stock data)
ALPHA_VANTAGE_API_KEY=your_api_key

# Binance API Credentials (optional, for higher rate limits)
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
```

### Cache Configuration

```python
# Configure cache with custom settings
cache = configure_unified_cache(
    cache_dir="./my-cache",
    max_size_gb=20.0
)
```

## Best Practices

1. **Use the cache population script** for initial data setup
2. **Let the system choose providers** automatically based on symbol type
3. **Validate data quality** before using in trading strategies
4. **Monitor cache size** and clean up old data regularly
5. **Use appropriate timeframes** for your trading strategy needs
6. **Handle rate limits** gracefully with built-in throttling

## Troubleshooting

### Common Issues

1. **Cache Directory Permissions**: Ensure write access to cache directory
2. **API Key Issues**: Verify environment variables are set correctly
3. **Data Validation Failures**: Check data quality and format
4. **Rate Limiting**: Built-in throttling should handle this automatically

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Documentation

For detailed information about the data module:

- **[Requirements.md](Requirements.md)** - Dependencies, API keys, and setup instructions
- **[Design.md](Design.md)** - Architecture, design decisions, and technical details  
- **[Tasks.md](Tasks.md)** - Development roadmap, known issues, and technical debt

## Contributing

When adding new data sources:
1. Inherit from `BaseDataDownloader`
2. Implement the `get_fundamentals()` method
3. Add comprehensive error handling
4. Include proper documentation
5. Add tests for the new implementation
6. Update the ticker classifier for provider selection
7. Update Requirements.md if new dependencies are needed
8. Document design decisions in Design.md
9. Add tasks to Tasks.md for future enhancements