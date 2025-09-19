# Alpaca Data Downloader

## Overview

The AlpacaDataDownloader provides access to historical market data and fundamental information for US stocks and ETFs through the Alpaca Markets API. It integrates seamlessly with the existing data downloader framework and supports multiple timeframes.

## Features

- **Historical OHLCV Data**: Download historical price and volume data
- **Multiple Timeframes**: Support for 1m, 5m, 15m, 30m, 1h, and 1d intervals
- **Fundamental Data**: Basic company information and asset details
- **Rate Limiting**: Built-in handling of API rate limits
- **Error Handling**: Comprehensive error handling and logging
- **Factory Integration**: Works with DataDownloaderFactory for easy instantiation

## Installation

1. Install the required package:
```bash
pip install alpaca-trade-api
```

2. If you encounter websockets compatibility issues:
```bash
pip install "websockets>=9.0,<11"
```

## Configuration

Set your Alpaca API credentials in `config/donotshare/.env`:

```env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Optional, defaults to paper trading
```

## Usage

### Direct Usage

```python
from src.data.downloader.alpaca_data_downloader import AlpacaDataDownloader
from datetime import datetime, timedelta

# Initialize downloader
downloader = AlpacaDataDownloader()

# Download daily data
end_date = datetime.now() - timedelta(days=1)
start_date = end_date - timedelta(days=30)

df = downloader.get_ohlcv("AAPL", "1d", start_date, end_date)
print(f"Downloaded {len(df)} rows of data")
```

### Factory Usage

```python
from src.data.downloader.data_downloader_factory import DataDownloaderFactory

# Create downloader via factory
downloader = DataDownloaderFactory.create_downloader("alpaca")

# Or use short code
downloader = DataDownloaderFactory.create_downloader("alp")
```

### Supported Intervals

- `1m` - 1 minute bars
- `5m` - 5 minute bars  
- `15m` - 15 minute bars
- `30m` - 30 minute bars
- `1h` - 1 hour bars
- `1d` - Daily bars

### Bar Limits

The free tier is limited to 10,000 bars per request. The downloader automatically enforces this limit:

```python
# This will automatically limit to 10,000 bars
df = downloader.get_ohlcv("AAPL", "1m", very_old_date, end_date)
print(f"Received {len(df)} bars (max 10,000)")

# You can also specify a custom limit (up to 10,000)
df = downloader.get_ohlcv("AAPL", "1d", start_date, end_date, limit=5000)
```

### Example: Multiple Symbols

```python
symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
results = {}

for symbol in symbols:
    # Note: Each request is limited to 10,000 bars (free tier)
    df = downloader.get_ohlcv(symbol, "1d", start_date, end_date)
    if not df.empty:
        results[symbol] = {
            'rows': len(df),
            'latest_price': df['close'].iloc[-1],
            'weekly_return': (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        }
```

### Fundamental Data

```python
# Get basic company information
fundamentals = downloader.get_fundamentals("AAPL")
if fundamentals:
    print(f"Company: {fundamentals.company_name}")
    print(f"Sector: {fundamentals.sector}")
    print(f"Industry: {fundamentals.industry}")
```

## API Limits

- **Rate Limits**: 200 requests per minute (free tier)
- **Bar Limits**: 10,000 bars per request (free tier)
- **Data Coverage**: US stocks and ETFs
- **Historical Data**: Extensive historical coverage
- **Real-time Data**: Available with appropriate subscription

## Data Quality

- **Source**: Professional-grade market data from Alpaca Markets
- **Accuracy**: High-quality, exchange-sourced data
- **Coverage**: Comprehensive US equity markets
- **Updates**: Real-time during market hours

## Error Handling

The downloader includes comprehensive error handling:

- **API Errors**: Graceful handling of API failures
- **Rate Limiting**: Automatic retry with backoff
- **Data Validation**: Basic validation of returned data
- **Logging**: Detailed logging for debugging

## Integration with Data Manager

The Alpaca downloader integrates seamlessly with the DataManager:

```python
from src.data.data_manager import get_data_manager

# Get data manager with Alpaca as provider
dm = get_data_manager()
df = dm.get_ohlcv("AAPL", "1d", start_date, end_date, provider="alpaca")
```

## Testing

Run the test scripts to verify your setup:

```bash
# Minimal test (no dependencies)
python src/data/downloader/test_alpaca_minimal.py

# Simple test (requires alpaca-trade-api)
python src/data/downloader/test_alpaca_simple.py

# Full example usage
python src/data/downloader/example_alpaca_usage.py
```

## Troubleshooting

### Common Issues

1. **Import Error**: Install `alpaca-trade-api` package
2. **Websockets Error**: Install compatible websockets version
3. **Authentication Error**: Check API credentials in config
4. **No Data**: Verify symbol exists and date range is valid

### Debug Mode

Enable debug logging to see detailed API interactions:

```python
import logging
logging.getLogger('src.data.downloader.alpaca_data_downloader').setLevel(logging.DEBUG)
```

## Provider Information

- **Name**: Alpaca Markets
- **Codes**: `alpaca`, `alp`
- **API Key Required**: Yes (both key and secret)
- **Cost**: Free tier available
- **Coverage**: US stocks and ETFs
- **Fundamental Data**: Basic company information

## Related Files

- `alpaca_data_downloader.py` - Main implementation
- `example_alpaca_usage.py` - Usage examples
- `test_alpaca_minimal.py` - Basic tests
- `test_alpaca_simple.py` - Simple functionality tests

## Support

For issues with the Alpaca API itself, refer to:
- [Alpaca API Documentation](https://alpaca.markets/docs/)
- [alpaca-trade-api GitHub](https://github.com/alpacahq/alpaca-trade-api-python)

For issues with this downloader implementation, check the logs and ensure your credentials are properly configured.