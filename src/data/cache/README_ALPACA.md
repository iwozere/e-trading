# Alpaca 1-Minute Data Downloader

This script downloads 1-minute OHLCV data from Alpaca for all tickers in your cache directory.

## Installation

1. Install the Alpaca Trade API package:
```bash
pip install alpaca-trade-api
```

2. Set up your Alpaca API credentials as environment variables:
```bash
# For Windows (Command Prompt)
set ALPACA_API_KEY=your_api_key_here
set ALPACA_SECRET_KEY=your_secret_key_here
set ALPACA_BASE_URL=https://paper-api.alpaca.markets

# For Windows (PowerShell)
$env:ALPACA_API_KEY="your_api_key_here"
$env:ALPACA_SECRET_KEY="your_secret_key_here"
$env:ALPACA_BASE_URL="https://paper-api.alpaca.markets"

# For Linux/Mac
export ALPACA_API_KEY=your_api_key_here
export ALPACA_SECRET_KEY=your_secret_key_here
export ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## Getting Alpaca API Credentials

1. Sign up for a free Alpaca account at https://alpaca.markets/
2. Go to the Alpaca dashboard
3. Navigate to "API Keys" section
4. Generate new API keys
5. Use the paper trading URL for testing: `https://paper-api.alpaca.markets`
6. Use the live trading URL for real data: `https://api.alpaca.markets`

## Usage

### Download for all cached tickers (from 2020-01-01 to yesterday)
```bash
python src/data/cache/download_alpaca_1m.py
```

### Download for specific tickers
```bash
python src/data/cache/download_alpaca_1m.py --tickers AAPL,MSFT,GOOGL
```

### Custom date range
```bash
python src/data/cache/download_alpaca_1m.py --start-date 2022-01-01 --end-date 2023-12-31
```

### Download for specific tickers with custom date range
```bash
python src/data/cache/download_alpaca_1m.py --tickers AAPL,VT,INTC --start-date 2020-01-01 --end-date 2024-12-31
```

## Output

The script will create CSV files in the following structure:
```
c:/data-cache/ohlcv/
├── AAPL/
│   └── AAPL-1m.csv
├── VT/
│   └── VT-1m.csv
├── INTC/
│   └── INTC-1m.csv
└── ...
```

Each CSV file contains columns:
- `timestamp`: Date and time (timezone-naive)
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

## Features

- **Automatic ticker discovery**: Finds all tickers in your existing cache directory
- **Rate limiting**: Respects Alpaca API rate limits (200 requests/minute)
- **Error handling**: Continues processing other tickers if one fails
- **Progress tracking**: Shows progress and detailed logging
- **Comprehensive summary**: Reports success/failure for each ticker

## Limitations

- **US stocks only**: Alpaca primarily provides US stock market data
- **Crypto tickers**: Will be skipped (BTCUSDT, ETHUSDT, etc. are not available on Alpaca)
- **Market hours**: Data is only available during market trading hours
- **Historical limits**: Free tier may have limitations on historical data depth

## Expected Results

Based on your current cache tickers, the script will:

✅ **Download successfully**:
- AAPL, GOOG, INTC, VT (US stocks)
- IONQ, LULU, MASI, MRNA, NFLX, NVDA, PFE, PSNY, QBTS, RPD, SMCI, TSLA (US stocks)

⚠️ **Skip or fail**:
- ADAUSDT, BTCUSDT, ETHUSDT, LTCUSDT, QTUM, VUSD (crypto tickers - not available on Alpaca)

## Troubleshooting

### "alpaca-trade-api not installed"
```bash
pip install alpaca-trade-api
```

### "Alpaca API credentials required"
Make sure you've set the environment variables:
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`

### "No data available for ticker"
- Check if the ticker is a valid US stock symbol
- Crypto tickers are not supported by Alpaca
- Some tickers might not have 1-minute data available

### Rate limiting errors
The script includes built-in rate limiting, but if you encounter issues:
- Reduce the number of tickers processed at once
- Increase the delay between requests in the script

## Integration with Your System

After downloading, you can use the 1-minute data for:
- High-frequency alert monitoring
- Detailed backtesting
- Intraday analysis
- Pattern recognition

The CSV files can be easily loaded into pandas:
```python
import pandas as pd
df = pd.read_csv('c:/data-cache/ohlcv/AAPL/AAPL-1m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
```