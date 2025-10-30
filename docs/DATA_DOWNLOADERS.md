# Data Downloaders Overview

This document describes all data downloaders implemented in the `src/data` folder. Each downloader provides a unified interface for fetching historical OHLCV (Open, High, Low, Close, Volume) data from various providers, returning data in a consistent format for analysis and backtesting.

## Common Interface

All downloaders inherit from `BaseDataDownloader` and provide:
- `get_ohlcv(symbol, interval, start_date, end_date) -> pd.DataFrame`: Download historical data for a symbol.
- `get_periods() -> list`: List of valid period strings.
- `get_intervals() -> list`: List of valid interval strings.
- `is_valid_period_interval(period, interval) -> bool`: Validate a period/interval combination.

**Returned DataFrame columns:**
- `timestamp` (datetime)
- `open` (float)
- `high` (float)
- `low` (float)
- `close` (float)
- `volume` (float)

All columns are lowercase. Timestamps are pandas `datetime` objects.

---

## Downloaders Summary Table

| Provider      | Class Name                    | API Key Required | Intervals Supported                | Periods Supported                | Free Tier Limits / Notes |
|---------------|------------------------------|------------------|------------------------------------|----------------------------------|-------------------------|
| Binance       | BinanceDataDownloader        | No               | 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M | 1d, 7d, 1w, 1mo, 3mo, 6mo, 1y, 2y | 1000 bars/request, public API |
| CoinGecko     | CoinGeckoDataDownloader      | No               | 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1mo (resampled) | 1d, 7d, 1w, 1mo, 3mo, 6mo, 1y, 2y | Public API, rate limits apply |
| Yahoo         | YahooDataDownloader          | No               | 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo | 1d, 7d, 1mo, 3mo, 6mo, 1y, 2y | Public API, subject to Yahoo limits |
| Alpha Vantage | AlphaVantageDataDownloader   | Yes              | 1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo | 1d, 7d, 1w, 1mo, 3mo, 6mo, 1y, 2y | 5 requests/min, 500/day |
| Polygon.io    | PolygonDataDownloader        | Yes              | 1m, 5m, 15m, 1h, 1d (resampled)    | 1d, 7d, 1mo, 3mo, 6mo, 1y, 2y    | 5 req/min, 2y daily, 2mo 1m |
| Finnhub       | FinnhubDataDownloader        | Yes              | 1m, 5m, 15m, 1h, 1d (resampled)    | 1d, 7d, 1mo, 3mo, 6mo, 1y, 2y, 5y| 60 req/min, 30d 1m, 5y 1d |
| Twelve Data   | TwelveDataDataDownloader     | Yes              | 1m, 5m, 15m, 1h, 1d (resampled)    | 1d, 7d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y | 8 req/min, 800/day, 1mo 1m, 10y 1d |

---

## Downloader Details

### BinanceDataDownloader
- **API Key:** Not required
- **Intervals:** '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'
- **Periods:** '1d', '7d', '1w', '1mo', '3mo', '6mo', '1y', '2y'
- **Limits:** 1000 candles/request, public API, may need batching for long periods
- **Example:**
```python
from src.data.binance_data_downloader import BinanceDataDownloader
downloader = BinanceDataDownloader()
df = downloader.get_ohlcv('BTCUSDT', '1h', '2023-01-01', '2023-02-01')
```

### CoinGeckoDataDownloader
- **API Key:** Not required
- **Intervals:** '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo' (resampled)
- **Periods:** '1d', '7d', '1w', '1mo', '3mo', '6mo', '1y', '2y'
- **Limits:** Public API, rate limits apply, OHLCV is approximated from price/volume
- **Example:**
```python
from src.data.coingecko_data_downloader import CoinGeckoDataDownloader
downloader = CoinGeckoDataDownloader()
df = downloader.get_ohlcv('bitcoin', '1d', '2023-01-01', '2023-02-01')
```

### YahooDataDownloader
- **API Key:** Not required
- **Intervals:** '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
- **Periods:** '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y'
- **Limits:** Public API, subject to Yahoo's own limits
- **Example:**
```python
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
downloader = YahooDataDownloader()
df = downloader.get_ohlcv('AAPL', '1d', '2023-01-01', '2023-02-01')
```

### AlphaVantageDataDownloader
- **API Key:** Required
- **Intervals:** '1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo'
- **Periods:** '1d', '7d', '1w', '1mo', '3mo', '6mo', '1y', '2y'
- **Limits:** 5 requests/min, 500 requests/day (free tier)
- **Example:**
```python
from src.data.alpha_vantage_data_downloader import AlphaVantageDataDownloader
downloader = AlphaVantageDataDownloader(api_key='YOUR_API_KEY')
df = downloader.get_ohlcv('AAPL', '1d', '2023-01-01', '2023-02-01')
```

### PolygonDataDownloader
- **API Key:** Required
- **Intervals:** '1m', '5m', '15m', '1h', '1d' (resampled)
- **Periods:** '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y'
- **Limits:** 5 requests/min, 2 years daily, 2 months minute data (free tier)
- **Example:**
```python
from src.data.polygon_data_downloader import PolygonDataDownloader
downloader = PolygonDataDownloader(api_key='YOUR_API_KEY')
df = downloader.get_ohlcv('AAPL', '1d', '2023-01-01', '2023-02-01')
```

### FinnhubDataDownloader
- **API Key:** Required
- **Intervals:** '1m', '5m', '15m', '1h', '1d' (resampled)
- **Periods:** '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y'
- **Limits:** 60 requests/min, 30 days 1m, 5 years 1d (free tier)
- **Example:**
```python
from src.data.finnhub_data_downloader import FinnhubDataDownloader
downloader = FinnhubDataDownloader(api_key='YOUR_API_KEY')
df = downloader.get_ohlcv('AAPL', '1d', '2023-01-01', '2023-02-01')
```

### TwelveDataDataDownloader
- **API Key:** Required
- **Intervals:** '1m', '5m', '15m', '1h', '1d' (resampled)
- **Periods:** '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'
- **Limits:** 8 requests/min, 800/day, 1 month 1m, 10 years 1d (free tier)
- **Example:**
```python
from src.data.twelvedata_data_downloader import TwelveDataDataDownloader
downloader = TwelveDataDataDownloader(api_key='YOUR_API_KEY')
df = downloader.get_ohlcv('AAPL', '1d', '2023-01-01', '2023-02-01')
```

---

## Notes
- All downloaders return data in the same format for easy downstream processing.
- Some providers require batching for long periods due to API limits.
- Always check provider documentation for the latest limits and supported features.
- For free API keys, be mindful of rate limits and data range restrictions. 