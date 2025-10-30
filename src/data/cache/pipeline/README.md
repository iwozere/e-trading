# Data Pipeline

This directory contains the complete data pipeline for downloading and processing market data from Alpaca Markets.

## Pipeline Overview

The pipeline consists of two main steps that work together to provide comprehensive market data:

### Step 1: Download 1-Minute Data (`step01_download_alpaca_1m.py`)
- Downloads 1-minute OHLCV data from Alpaca Markets
- Supports all US stocks and ETFs available on Alpaca
- Intelligent gap detection and filling
- Gzipped storage for 4x compression
- Comprehensive error handling and statistics

### Step 2: Calculate Higher Timeframes (`step02_calculate_timeframes.py`)
- Calculates 5m, 15m, 1h, 4h, and 1d timeframes from 1m data
- Intelligent handling of missing data and gaps
- Trading hours: 4:00 AM to 8:00 PM ET
- Yearly storage with JSON metadata
- Incremental processing

## Quick Start

### Run Complete Pipeline
```bash
# Download 1m data and calculate all timeframes for all tickers
python src/data/cache/pipeline/run_pipeline.py
```

### Run Individual Steps
```bash
# Step 1: Download 1-minute data only
python src/data/cache/pipeline/run_pipeline.py --steps 1

# Step 2: Calculate timeframes only (requires step 1 data)
python src/data/cache/pipeline/run_pipeline.py --steps 2
```

### Custom Parameters
```bash
# Specific tickers and timeframes
python src/data/cache/pipeline/run_pipeline.py --tickers AAPL,MSFT,GOOGL --timeframes 5m,15m,1h

# Custom date range
python src/data/cache/pipeline/run_pipeline.py --start-date 2023-01-01 --end-date 2023-12-31

# Force refresh all data
python src/data/cache/pipeline/run_pipeline.py --force-refresh
```

## Data Structure

The pipeline creates the following data structure:

```
DATA_CACHE_DIR/ohlcv/
├── AAPL/
│   ├── AAPL-1m.csv.gz          ← Step 1: 1-minute data (gzipped)
│   ├── 5m/
│   │   ├── 2020.csv.gz         ← Step 2: 5-minute data by year
│   │   ├── 2021.csv.gz
│   │   ├── 2022.csv.gz
│   │   └── metadata.json       ← Metadata for 5m timeframe
│   ├── 15m/
│   │   ├── 2020.csv.gz
│   │   └── metadata.json
│   ├── 1h/
│   │   ├── 2020.csv.gz
│   │   └── metadata.json
│   ├── 4h/
│   │   └── metadata.json
│   └── 1d/
│       └── metadata.json
├── MSFT/
│   └── ... (same structure)
└── ...
```

## Features

### Step 1 Features
- **Automatic Ticker Discovery**: Finds all tickers in cache directory
- **Gap Detection**: Only downloads missing data ranges
- **Stock Filtering**: Automatically identifies stock vs crypto/forex symbols
- **Gzip Compression**: 4x smaller file sizes
- **Rate Limiting**: Respects Alpaca's 200 requests/minute limit
- **10k Bar Limit**: Handles Alpaca's free tier limit with chunking
- **Error Categorization**: Identifies crypto/forex failures vs API errors

### Step 2 Features
- **Trading Hours**: 4:00 AM to 8:00 PM ET (16 hours)
- **Intelligent Gaps**: Preserves gaps as they exist in 1m data
- **Partial Data**: Accumulates available 1m bars into timeframe bars
- **Yearly Storage**: Efficient storage and retrieval by year
- **Metadata**: JSON metadata with statistics and configuration
- **Incremental Processing**: Only recalculates when needed

## Pipeline Intelligence

### Gap Handling
The pipeline intelligently handles data gaps:

1. **Preserves Natural Gaps**: Market holidays, weekends, after-hours
2. **No Artificial Filling**: Doesn't create fake data to fill gaps
3. **Partial Accumulation**: Uses available 1m bars even if incomplete

### Example Gap Scenarios
```
1m data: [09:30, 09:31, 09:33, 09:34, 09:35]  # Missing 09:32
5m bar:  [09:30-09:35] with 4 bars instead of 5  # Still creates bar

1m data: [09:30, 09:31]  # Only 2 bars available
5m bar:  [09:30-09:35] with 2 bars  # Creates partial bar

1m data: []  # No bars available
5m bar:  None  # Skips this timeframe bar
```

### Trading Hours
- **Start**: 4:00 AM ET (pre-market)
- **End**: 8:00 PM ET (after-hours)
- **Duration**: 16 hours of trading data per day
- **Daily Bars**: Calculated from 4 AM to 8 PM data

## Error Handling

### Step 1 Errors
- **Symbol Not Found**: Likely crypto/forex not supported by Alpaca
- **Rate Limiting**: Automatic delays and retry logic
- **Authentication**: API key/secret issues
- **Network**: Connection and timeout handling

### Step 2 Errors
- **No 1m Data**: Step 1 must be completed first
- **Calculation Errors**: Invalid data or processing issues
- **Storage Errors**: File system or permission issues

## Performance

### Step 1 Performance
- **Rate**: ~200 API calls per minute (Alpaca limit)
- **Chunking**: 10,000 bars per request
- **Compression**: 4x reduction in storage
- **Typical Speed**: 1-2 seconds per ticker for recent data

### Step 2 Performance
- **Processing**: Very fast (local calculations)
- **Storage**: Yearly files for efficient access
- **Incremental**: Only processes changed data
- **Typical Speed**: <1 second per ticker per timeframe

## Monitoring and Statistics

### Pipeline Statistics
Both steps provide comprehensive statistics:

```
📊 PROCESSING STATISTICS:
   Total tickers processed: 18
   ✅ Successfully updated: 15
   ⏭️  Already up to date: 2
   ❌ Failed: 1
   📈 Success rate: 94.4%

📈 DATA STATISTICS:
   Total rows downloaded: 1,234,567
   Total API calls made: 156
   Processing time: 45.2 seconds
```

### Failed Ticker Analysis
```
❌ FAILED TICKERS (3):
   BTCUSDT: Symbol not found (likely crypto/forex)
   ETHUSDT: Symbol not found (likely crypto/forex)
   INVALID: Authentication error

🔍 LIKELY CRYPTO/FOREX TICKERS (not supported by Alpaca):
   BTCUSDT
   ETHUSDT
```

## Configuration

### Environment Variables
```bash
# Required for Step 1
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Optional

# Optional
DATA_CACHE_DIR=c:/data-cache  # Default cache directory
```

### Pipeline Parameters
```bash
# Tickers (default: discover from cache)
--tickers AAPL,MSFT,GOOGL,TSLA

# Date range (default: 2020-01-01 to yesterday)
--start-date 2023-01-01
--end-date 2023-12-31

# Timeframes (default: all)
--timeframes 5m,15m,1h,4h,1d

# Force refresh
--force-refresh

# Custom cache directory
--cache-dir /path/to/cache
```

## Usage Examples

### Daily Update
```bash
# Run daily to update with latest data
python src/data/cache/pipeline/run_pipeline.py
```

### Historical Backfill
```bash
# Download historical data for specific period
python src/data/cache/pipeline/run_pipeline.py --start-date 2020-01-01 --end-date 2022-12-31 --force-refresh
```

### Specific Tickers
```bash
# Process only specific tickers
python src/data/cache/pipeline/run_pipeline.py --tickers AAPL,MSFT,GOOGL,TSLA
```

### Timeframe Subset
```bash
# Calculate only specific timeframes
python src/data/cache/pipeline/run_pipeline.py --timeframes 5m,15m,1h
```

### Development/Testing
```bash
# Test with single ticker
python src/data/cache/pipeline/run_pipeline.py --tickers AAPL --timeframes 5m,15m
```

## Automation

### Cron Job (Daily Updates)
```bash
# Add to crontab for daily updates at 6 AM
0 6 * * * cd /path/to/project && python src/data/cache/pipeline/run_pipeline.py
```

### Weekly Full Refresh
```bash
# Weekly full refresh on Sundays at 2 AM
0 2 * * 0 cd /path/to/project && python src/data/cache/pipeline/run_pipeline.py --force-refresh
```

## Troubleshooting

### Common Issues

1. **No tickers found**
   - Ensure cache directory exists with ticker folders
   - Check DATA_CACHE_DIR environment variable

2. **Alpaca authentication errors**
   - Verify ALPACA_API_KEY and ALPACA_SECRET_KEY
   - Check API key permissions

3. **Step 2 fails with "No 1m data"**
   - Run Step 1 first to download 1m data
   - Check if 1m files exist and are readable

4. **High failure rate**
   - Many failures likely crypto/forex tickers (not supported by Alpaca)
   - Check error messages for specific issues

### Debug Mode
```bash
# Enable debug logging
export PYTHONPATH=/path/to/project
python -c "
import logging
logging.getLogger('src.data.cache.pipeline').setLevel(logging.DEBUG)
"
```

## Integration

### With Trading Strategies
```python
# Read pipeline data in your trading code
import pandas as pd

# Read 1m data
df_1m = pd.read_csv('c:/data-cache/ohlcv/AAPL/AAPL-1m.csv.gz', compression='gzip')

# Read 15m data for 2023
df_15m = pd.read_csv('c:/data-cache/ohlcv/AAPL/15m/2023.csv.gz', compression='gzip')

# Read metadata
import json
with open('c:/data-cache/ohlcv/AAPL/15m/metadata.json', 'r') as f:
    metadata = json.load(f)
```

### With Data Manager
```python
# Use with existing DataManager
from src.data.data_manager import get_data_manager

dm = get_data_manager()
df = dm.get_ohlcv("AAPL", "15m", start_date, end_date)
```

## Support

For issues:
1. Check pipeline logs for detailed error messages
2. Verify Alpaca API credentials and permissions
3. Ensure sufficient disk space for data storage
4. Test with single ticker first

For Alpaca API issues:
- [Alpaca API Documentation](https://alpaca.markets/docs/)
- [alpaca-trade-api GitHub](https://github.com/alpacahq/alpaca-trade-api-python)