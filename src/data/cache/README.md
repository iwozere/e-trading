# Cache Management Scripts

This directory contains scripts for managing the unified data cache system. These scripts handle cache population, validation, migration, and maintenance operations.

## Overview

The cache system stores historical market data in a unified structure: `symbol/timeframe/year.csv.gz` with metadata files for each year. This provides efficient storage, fast retrieval, and intelligent provider selection.

## Scripts Overview

### 1. `populate_cache.py` - Main Cache Population Script

**Purpose**: Downloads and caches historical market data from multiple providers.

**Key Features**:
- Intelligent provider selection (Binance for crypto, Yahoo Finance for daily stocks, Alpha Vantage for intraday stocks)
- Automatic timezone handling (UTC for Binance, naive for market-based providers)
- Rate limiting and error handling
- Year-based file splitting for efficient storage
- Comprehensive logging and progress tracking

**Usage**:
```bash
# Basic usage - populate cache with default settings
python src/data/cache/populate_cache.py --populate

# Specific symbols and timeframes
python src/data/cache/populate_cache.py --populate --symbols BTCUSDT,ETHUSDT,AAPL --intervals 5m,15m,1h,1d

# Custom date range
python src/data/cache/populate_cache.py --populate --start-date 2020-01-01 --end-date 2023-12-31

# Test system components
python src/data/cache/populate_cache.py --test-all

# Validate existing cache
python src/data/cache/populate_cache.py --validate-cache
```

**Provider Selection Logic**:
- **Crypto symbols** (BTCUSDT, ETHUSDT, etc.) → Binance
- **Stock symbols with daily intervals** (1d) → Yahoo Finance
- **Stock symbols with intraday intervals** (5m, 15m, 1h, 4h) → Alpha Vantage

### 2. `validate_gaps.py` - Data Validation and Gap Analysis

**Purpose**: Analyzes cached data for gaps and quality issues, generating comprehensive validation reports.

**Key Features**:
- Year-by-year data validation with detailed gap analysis
- OHLCV data validation with smart gap tolerance
- Asset-type aware validation (crypto vs stocks)
- Comprehensive JSON metadata generation
- Data quality scoring and reporting

**Usage**:
```bash
# Validate all cached data (default behavior)
python src/data/cache/validate_gaps.py

# Validate specific symbols and timeframes
python src/data/cache/validate_gaps.py --symbols BTCUSDT,ETHUSDT --intervals 5m,15m,1h

# Validate all cached data explicitly
python src/data/cache/validate_gaps.py --validate-all

# Use custom cache directory
python src/data/cache/validate_gaps.py --cache-dir /path/to/cache
```

**Output**: Generates `validate-metadata.json` in the cache directory with detailed gap analysis.

### 3. `fill_gaps.py` - Gap Filling with Alternative Providers

**Purpose**: Reads validation metadata and fills gaps using alternative data providers.

**Key Features**:
- Reads gap information from `validate-metadata.json`
- Alternative provider selection (since gaps exist in primary provider data):
  - **Crypto symbols**: CoinGecko (alternative to Binance, which is primary in populate_cache.py)
  - **Stock symbols**: Alpha Vantage (alternative to Yahoo Finance, which is primary in populate_cache.py)
- Configurable gap size limits
- Safe data merging with duplicate removal
- Comprehensive progress tracking

**Usage**:
```bash
# Fill gaps for all data in validate-metadata.json
python src/data/cache/fill_gaps.py

# Fill gaps for specific symbols only
python src/data/cache/fill_gaps.py --symbols BTCUSDT,ETHUSDT

# Fill gaps for specific intervals only
python src/data/cache/fill_gaps.py --intervals 5m,15m,1h

# Only fill gaps <= 12 hours
python src/data/cache/fill_gaps.py --max-gap-hours 12

# Use custom cache directory
python src/data/cache/fill_gaps.py --cache-dir /path/to/cache
```

**Prerequisites**: Run `validate_gaps.py` first to generate the validation metadata.

**API Limitations**: 
- CoinGecko free API is limited to 365 days of historical data
- Alpha Vantage free API has a 25 requests per day limit (no historical data range limit)
- For gaps older than 365 days, CoinGecko cannot fill them, but Alpha Vantage can (if within rate limits)

### 4. `migrate_to_unified_cache.py` - Cache Migration Utility

**Purpose**: Migrates data from the old provider-based cache structure to the new unified structure.

**Key Features**:
- Migrates from `provider/symbol/timeframe/year/` to `symbol/timeframe/year/`
- Preserves all metadata and data integrity
- Dry-run mode for safe testing
- Progress tracking and error handling
- Backup creation before migration

**Usage**:
```bash
# Dry run - see what would be migrated without making changes
python src/data/cache/migrate_to_unified_cache.py --dry-run

# Perform actual migration
python src/data/cache/migrate_to_unified_cache.py --migrate

# Custom source and destination paths
python src/data/cache/migrate_to_unified_cache.py --migrate --old-cache ./old-cache --new-cache ./new-cache
```

### 5. `cleanup_failed_cache.py` - Cache Cleanup and Validation

**Purpose**: Validates cache files and removes corrupted or invalid data.

**Key Features**:
- File-by-file validation of cached data
- Detection of corrupted or incomplete files
- Safe removal of invalid data
- Comprehensive reporting of cleanup actions
- Backup of removed files

**Usage**:
```bash
# Validate cache files without removing anything
python src/data/cache/cleanup_failed_cache.py --validate-only

# Remove invalid files
python src/data/cache/cleanup_failed_cache.py --cleanup

# Validate and clean up in one operation
python src/data/cache/cleanup_failed_cache.py --validate-and-cleanup
```

### 6. `unified_cache.py` - Core Cache Implementation

**Purpose**: Core implementation of the unified cache system (not a standalone script).

**Key Features**:
- Unified cache interface with `put()` and `get()` methods
- Year-based file splitting with gzip compression
- Metadata management for each year
- Provider information tracking
- Cache statistics and management functions

## Recommended Sequence of Operations

### Initial Setup (First Time)

1. **Configure API Keys**:
   ```bash
   # Ensure API keys are set in config/donotshare/donotshare.py
   # Required: ALPHA_VANTAGE_KEY for intraday stock data
   ```

2. **Test System Components**:
   ```bash
   python src/data/cache/populate_cache.py --test-all
   ```

3. **Populate Initial Cache**:
   ```bash
   # Start with a small test
   python src/data/cache/populate_cache.py --populate --symbols BTCUSDT,AAPL --intervals 1h,1d --start-date 2023-01-01 --end-date 2023-12-31
   
   # Then expand to full dataset
   python src/data/cache/populate_cache.py --populate --symbols BTCUSDT,ETHUSDT,AAPL,GOOGL,TSLA --intervals 5m,15m,1h,4h,1d --start-date 2020-01-01
   ```

### Regular Maintenance

1. **Validate Cache Data and Analyze Gaps**:
   ```bash
   python src/data/cache/validate_gaps.py
   ```

2. **Fill Data Gaps** (if gaps are found):
   ```bash
   # Fill all gaps found in validation
   python src/data/cache/fill_gaps.py
   
   # Fill gaps for specific symbols only
   python src/data/cache/fill_gaps.py --symbols BTCUSDT,ETHUSDT
   
   # Only fill small gaps (<= 12 hours)
   python src/data/cache/fill_gaps.py --max-gap-hours 12
   ```

3. **Clean Up Invalid Files** (if needed):
   ```bash
   python src/data/cache/cleanup_failed_cache.py --validate-and-cleanup
   ```

### Migration from Old Cache (if applicable)

1. **Test Migration**:
   ```bash
   python src/data/cache/migrate_to_unified_cache.py --dry-run
   ```

2. **Perform Migration**:
   ```bash
   python src/data/cache/migrate_to_unified_cache.py --migrate
   ```

3. **Validate Migrated Data**:
   ```bash
   python src/data/cache/populate_cache.py --validate-cache
   ```

## Cache Structure

The unified cache uses the following structure:

```
cache-dir/
├── symbol1/
│   ├── timeframe1/
│   │   ├── 2020.csv.gz
│   │   ├── 2020.metadata.json
│   │   ├── 2021.csv.gz
│   │   ├── 2021.metadata.json
│   │   └── ...
│   └── timeframe2/
│       └── ...
└── symbol2/
    └── ...
```

## Configuration

### Environment Variables
- `ALPHA_VANTAGE_KEY`: Required for intraday stock data from Alpha Vantage
- Cache directory can be specified via `--cache-dir` parameter (defaults to `c:/data-cache`)

### API Keys
API keys are managed through `config/donotshare/donotshare.py`:
- `ALPHA_VANTAGE_KEY`: For Alpha Vantage API access
- Binance and Yahoo Finance don't require API keys for basic historical data

## Troubleshooting

### Common Issues

1. **"No API key found"**: Ensure `ALPHA_VANTAGE_KEY` is set in `donotshare.py`
2. **Timezone errors**: Fixed in current version - ensure using latest scripts
3. **Cache re-downloading**: Check that cache directory path is consistent
4. **Provider selection issues**: Verify ticker classification is working correctly

### Debug Commands

```bash
# Check cache statistics
python -c "from src.data.cache.unified_cache import configure_unified_cache; cache = configure_unified_cache(); print(cache.get_stats())"

# List available symbols
python -c "from src.data.cache.unified_cache import configure_unified_cache; cache = configure_unified_cache(); print('Symbols:', cache.list_symbols())"

# Check specific symbol data
python -c "from src.data.cache.unified_cache import configure_unified_cache; cache = configure_unified_cache(); print('BTCUSDT timeframes:', cache.list_timeframes('BTCUSDT'))"
```

## Performance Tips

1. **Batch Operations**: Run multiple symbols/intervals in single commands for efficiency
2. **Date Ranges**: Use specific date ranges to avoid downloading unnecessary data
3. **Rate Limiting**: Built-in rate limiting prevents API throttling
4. **Compression**: Gzip compression reduces storage requirements by ~70%

## Logging

All scripts use the standard project logging system:
- Logs are written to both console and log files
- Log level can be controlled via environment variables
- Download completion reports are logged for audit trails

## Dependencies

- `pandas`: Data manipulation and analysis
- `yfinance`: Yahoo Finance data access
- `requests`: HTTP requests for API calls
- `gzip`: File compression
- `pathlib`: File system operations
- `datetime`: Date/time handling
- `typing`: Type hints
- `argparse`: Command-line argument parsing
