# Fundamentals Data Downloader

A comprehensive script for downloading fundamentals data for multiple tickers and storing it in cache with support for multiple data providers.

## Features

- **Multiple Data Providers**: Yahoo Finance, Financial Modeling Prep, Alpha Vantage, Finnhub, Polygon.io, Twelve Data, Tiingo
- **Intelligent Caching**: 7-day cache expiration with automatic cleanup
- **Batch Processing**: Download data for multiple tickers concurrently
- **Progress Tracking**: Real-time progress updates and detailed logging
- **Error Handling**: Robust error handling with retry logic
- **Data Validation**: Quality checks for downloaded data
- **Cache Management**: View cache statistics and cleanup expired data

## Quick Start

### Basic Usage

```bash
# Download fundamentals for specific tickers using Yahoo Finance
python src/data/utils/download_fundamentals.py --tickers AAPL,MSFT,GOOGL --provider yf

# Download from file using Financial Modeling Prep
python src/data/utils/download_fundamentals.py --tickers-file example_tickers.txt --provider fmp

# Force refresh existing cache
python src/data/utils/download_fundamentals.py --tickers AAPL --provider yf --force-refresh
```

### Cache Management

```bash
# Show cache information
python src/data/utils/download_fundamentals.py --show-cache

# Show cache info for specific ticker
python src/data/utils/download_fundamentals.py --show-cache --ticker AAPL

# Clean up expired cache
python src/data/utils/download_fundamentals.py --cleanup-cache
```

## Supported Providers

| Provider | Code | API Key Required | Rate Limits | Cost |
|----------|------|------------------|-------------|------|
| Yahoo Finance | `yf` | No | None | Free |
| Financial Modeling Prep | `fmp` | Yes | 3000/min (free) | Free tier |
| Alpha Vantage | `av` | Yes | 5/min, 500/day (free) | Free tier |
| Finnhub | `fh` | Yes | 60/min (free) | Free tier |
| Polygon.io | `pg` | Yes | 5/min (free) | Free tier |
| Twelve Data | `td` | Yes | 8/min, 800/day (free) | Free tier |
| Tiingo | `tiingo` | Yes | 1000/day (free) | Free tier |

## Command Line Options

### Input Options
- `--tickers`: Comma-separated list of ticker symbols
- `--tickers-file`: Path to file containing ticker symbols (one per line)
- `--show-cache`: Show cache information instead of downloading
- `--cleanup-cache`: Clean up expired cache data

### Provider Options
- `--provider`: Data provider code (default: yf)
- `--force-refresh`: Force refresh even if cache is valid

### Cache Options
- `--cache-dir`: Cache directory path (default: data-cache)

### Performance Options
- `--max-workers`: Maximum concurrent download threads (default: 5)

### Output Options
- `--ticker`: Specific ticker for cache info (use with --show-cache)
- `--quiet`: Suppress progress output

## Examples

### Download Fundamentals for Popular Stocks

```bash
python src/data/utils/download_fundamentals.py \
    --tickers AAPL,MSFT,GOOGL,AMZN,TSLA \
    --provider yf \
    --max-workers 3
```

### Batch Download from File

```bash
# Create tickers file
echo -e "AAPL\nMSFT\nGOOGL\nAMZN\nTSLA" > my_tickers.txt

# Download using FMP
python src/data/utils/download_fundamentals.py \
    --tickers-file my_tickers.txt \
    --provider fmp \
    --max-workers 5
```

### Force Refresh Cache

```bash
python src/data/utils/download_fundamentals.py \
    --tickers AAPL \
    --provider yf \
    --force-refresh
```

### Cache Management

```bash
# View cache statistics
python src/data/utils/download_fundamentals.py --show-cache

# View specific ticker cache
python src/data/utils/download_fundamentals.py --show-cache --ticker AAPL

# Clean up expired data
python src/data/utils/download_fundamentals.py --cleanup-cache
```

## Programmatic Usage

```python
from src.data.utils.download_fundamentals import FundamentalsDownloader

# Initialize downloader
downloader = FundamentalsDownloader()

# Download fundamentals
stats = downloader.download_fundamentals(
    tickers=["AAPL", "MSFT", "GOOGL"],
    provider="yf",
    show_progress=True
)

print(f"Downloaded: {stats['successful_downloads']} successful")

# Get cache information
cache_info = downloader.get_cache_info("AAPL")
print(f"AAPL cache files: {cache_info['files']}")
```

## Cache Structure

The cache stores data in the following structure:

```
data-cache/
└── fundamentals/
    ├── AAPL/
    │   ├── yf_20250109_143022.json
    │   └── fmp_20250109_143045.json
    ├── MSFT/
    │   └── yf_20250109_143030.json
    └── GOOGL/
        └── yf_20250109_143035.json
```

Each cache file contains:
- **Metadata**: Provider, timestamp, symbol, quality score
- **Fundamentals**: Actual financial data

## Data Quality Validation

The script validates downloaded data by checking:
- Essential fields presence (market_cap, pe_ratio, pb_ratio)
- Reasonable value ranges
- Data structure integrity

## Error Handling

- **Network Errors**: Automatic retry with exponential backoff
- **API Rate Limits**: Respects provider rate limits
- **Invalid Data**: Quality validation and error reporting
- **Cache Errors**: Graceful fallback to fresh downloads

## Performance Tips

1. **Use Yahoo Finance** for free, unlimited downloads
2. **Batch Processing**: Use `--max-workers` to control concurrency
3. **Cache Management**: Regular cleanup with `--cleanup-cache`
4. **Provider Selection**: Choose based on your API key availability

## Environment Variables

Set these environment variables for API key providers:

```bash
# Financial Modeling Prep
export FMP_API_KEY="your_fmp_api_key"

# Alpha Vantage
export ALPHA_VANTAGE_KEY="your_alpha_vantage_key"

# Finnhub
export FINNHUB_KEY="your_finnhub_key"

# Polygon.io
export POLYGON_KEY="your_polygon_key"

# Twelve Data
export TWELVE_DATA_KEY="your_twelve_data_key"

# Tiingo
export TIINGO_API_KEY="your_tiingo_key"
```

## Troubleshooting

### Common Issues

1. **"No fundamentals data returned"**
   - Check if ticker symbol is valid
   - Verify provider supports the ticker
   - Check API key for paid providers

2. **"Failed to create downloader"**
   - Verify provider code is correct
   - Check API key environment variables
   - Ensure provider is supported

3. **"Rate limit exceeded"**
   - Reduce `--max-workers` value
   - Wait before retrying
   - Consider using different provider

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration

The downloaded data can be used with:

- **Trading Strategies**: Access fundamentals in strategy development
- **Backtesting**: Use historical fundamentals data
- **Portfolio Analysis**: Analyze company fundamentals
- **Risk Management**: Incorporate fundamental metrics

## License

This script is part of the e-trading system and follows the same license terms.
