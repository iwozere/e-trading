# Data Module

This module provides comprehensive data downloading and live feed capabilities for the e-trading platform. It supports multiple data sources for both historical data and real-time market feeds.

## Overview

The data module consists of:
- **Data Downloaders**: For fetching historical OHLCV data and fundamental information
- **Live Data Feeds**: For real-time market data streaming
- **Data Factory**: For creating and managing data sources
- **Database Integration**: For storing and retrieving market data

## Data Downloaders

### Base Data Downloader

All data downloaders inherit from `BaseDataDownloader` which provides:
- Common file management operations
- Standardized data formats
- Error handling and logging
- Abstract `get_fundamentals()` method

### Available Data Sources

#### 1. Yahoo Finance Data Downloader (`YahooDataDownloader`)

**Best for:** Comprehensive fundamental analysis and global stock coverage

**Capabilities:**
- ✅ **PE Ratio**: Trailing and forward PE ratios
- ✅ **Financial Ratios**: P/B, ROE, ROA, debt/equity, current ratio, quick ratio
- ✅ **Growth Metrics**: Revenue growth, net income growth
- ✅ **Company Information**: Name, sector, industry, country, exchange
- ✅ **Market Data**: Market cap, current price, shares outstanding
- ✅ **Profitability Metrics**: Operating margin, profit margin, free cash flow
- ✅ **Valuation Metrics**: Beta, PEG ratio, price-to-sales, enterprise value

**Data Quality:** High - Comprehensive fundamental data
**Rate Limits:** None for basic usage
**Coverage:** Global stocks and ETFs

```python
from src.data.yahoo_data_downloader import YahooDataDownloader

downloader = YahooDataDownloader()
fundamentals = downloader.get_fundamentals("AAPL")
print(f"PE Ratio: {fundamentals.pe_ratio}")
print(f"Market Cap: ${fundamentals.market_cap:,.0f}")
```

#### 2. Alpha Vantage Data Downloader (`AlphaVantageDataDownloader`)

**Best for:** High-quality fundamental data with API key

**Capabilities:**
- ✅ **PE Ratio**: Trailing and forward PE ratios
- ✅ **Financial Ratios**: P/B, ROE, ROA, debt/equity, current ratio, quick ratio
- ✅ **Growth Metrics**: Revenue growth, net income growth
- ✅ **Company Information**: Name, sector, industry, country, exchange
- ✅ **Market Data**: Market cap, current price, shares outstanding
- ✅ **Profitability Metrics**: Operating margin, profit margin, free cash flow
- ✅ **Valuation Metrics**: Beta, PEG ratio, price-to-sales, enterprise value

**Data Quality:** High - Comprehensive fundamental data
**Rate Limits:** 5 API calls per minute (free tier), 500 per day (free tier)
**Coverage:** Global stocks and ETFs

```python
from src.data.alpha_vantage_data_downloader import AlphaVantageDataDownloader

downloader = AlphaVantageDataDownloader("YOUR_API_KEY")
fundamentals = downloader.get_fundamentals("AAPL")
```

#### 3. Finnhub Data Downloader (`FinnhubDataDownloader`)

**Best for:** Real-time data and comprehensive fundamentals

**Capabilities:**
- ✅ **PE Ratio**: Trailing and forward PE ratios
- ✅ **Financial Ratios**: P/B, ROE, ROA, debt/equity, current ratio, quick ratio
- ✅ **Growth Metrics**: Revenue growth, net income growth
- ✅ **Company Information**: Name, sector, industry, country, exchange
- ✅ **Market Data**: Market cap, current price, shares outstanding
- ✅ **Profitability Metrics**: Operating margin, profit margin, free cash flow
- ✅ **Valuation Metrics**: Beta, PEG ratio, price-to-sales, enterprise value

**Data Quality:** High - Comprehensive fundamental data
**Rate Limits:** 60 API calls per minute (free tier)
**Coverage:** Global stocks and ETFs

```python
from src.data.finnhub_data_downloader import FinnhubDataDownloader

downloader = FinnhubDataDownloader("YOUR_API_KEY")
fundamentals = downloader.get_fundamentals("AAPL")
```

#### 4. Polygon.io Data Downloader (`PolygonDataDownloader`)

**Best for:** US market data with basic fundamentals (free tier)

**Capabilities:**
- ❌ **PE Ratio**: Requires paid tier
- ❌ **Financial Ratios**: Requires paid tier
- ❌ **Growth Metrics**: Requires paid tier
- ✅ **Company Information**: Name, sector, industry, country, exchange
- ✅ **Market Data**: Market cap, current price, shares outstanding
- ❌ **Profitability Metrics**: Requires paid tier
- ❌ **Valuation Metrics**: Requires paid tier

**Data Quality:** Basic (free tier) - Limited fundamental data
**Rate Limits:** 5 API calls per minute (free tier)
**Coverage:** US stocks and ETFs (free tier)

```python
from src.data.polygon_data_downloader import PolygonDataDownloader

downloader = PolygonDataDownloader("YOUR_API_KEY")
fundamentals = downloader.get_fundamentals("AAPL")
```

#### 5. Twelve Data Downloader (`TwelveDataDataDownloader`)

**Best for:** Global coverage with basic fundamental data

**Capabilities:**
- ✅ **PE Ratio**: Basic PE ratio
- ✅ **Price-to-Book Ratio**: Available
- ❌ **Financial Ratios**: ROE, ROA, debt/equity, etc. not available
- ❌ **Growth Metrics**: Revenue growth, net income growth not available
- ✅ **Company Information**: Name, sector, industry, country, exchange
- ✅ **Market Data**: Market cap, current price
- ✅ **Earnings Data**: EPS, revenue from earnings reports
- ✅ **Beta**: Volatility measure

**Data Quality:** Basic - Limited fundamental data
**Rate Limits:** 8 API calls per minute (free tier), 800 per day (free tier)
**Coverage:** Global stocks and ETFs

```python
from src.data.twelvedata_data_downloader import TwelveDataDataDownloader

downloader = TwelveDataDataDownloader("YOUR_API_KEY")
fundamentals = downloader.get_fundamentals("AAPL")
```

#### 6. Binance Data Downloader (`BinanceDataDownloader`)

**Best for:** Cryptocurrency data only

**Capabilities:**
- ❌ **Fundamental Data**: Not applicable for cryptocurrencies
- ✅ **Cryptocurrency OHLCV**: Comprehensive crypto market data

**Data Quality:** N/A - Cryptocurrency exchange
**Rate Limits:** 1200 requests per minute (free tier)
**Coverage:** Cryptocurrencies only

```python
from src.data.binance_data_downloader import BinanceDataDownloader

downloader = BinanceDataDownloader("YOUR_API_KEY", "YOUR_SECRET_KEY")
# Get cryptocurrency data
df = downloader.get_ohlcv("BTCUSDT", "1d", "2023-01-01", "2023-12-31")
```

#### 7. CoinGecko Data Downloader (`CoinGeckoDataDownloader`)

**Best for:** Cryptocurrency data with no API key required

**Capabilities:**
- ❌ **Fundamental Data**: Not applicable for cryptocurrencies
- ✅ **Cryptocurrency OHLCV**: Comprehensive crypto market data

**Data Quality:** N/A - Cryptocurrency exchange
**Rate Limits:** 50 calls per minute (free tier)
**Coverage:** Cryptocurrencies only

```python
from src.data.coingecko_data_downloader import CoinGeckoDataDownloader

downloader = CoinGeckoDataDownloader()
# Get cryptocurrency data
df = downloader.get_ohlcv("bitcoin", "1d", "2023-01-01", "2023-12-31")
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

#### 3. IBKR Live Feed (`IBKRLiveFeed`)

Interactive Brokers real-time data feed.

**Features:**
- Professional-grade data
- Multiple asset classes
- Real-time streaming
- Advanced order management

```python
from src.data.ibkr_live_feed import IBKRLiveFeed

feed = IBKRLiveFeed()
feed.connect()
```

#### 4. CoinGecko Live Feed (`CoinGeckoLiveDataFeed`)

Real-time cryptocurrency data streaming from CoinGecko.

**Features:**
- Polling-based real-time data (no WebSocket available)
- Multiple cryptocurrency support
- Automatic rate limiting (50 calls/minute)
- Error handling and reconnection
- No API key required

**Note:** CoinGecko doesn't provide WebSocket API, so this implementation uses polling to simulate real-time updates.

```python
from src.data.coingecko_live_feed import CoinGeckoLiveDataFeed

feed = CoinGeckoLiveDataFeed("bitcoin", "1h", polling_interval=60)
feed.start()
```

## Data Factories

### Data Downloader Factory

The `DataDownloaderFactory` provides a unified interface for creating data downloaders using short provider codes:

```python
from src.data.data_downloader_factory import DataDownloaderFactory

# Create downloaders using short provider codes
yahoo_downloader = DataDownloaderFactory.create_downloader("yf")  # Yahoo Finance
alpha_vantage_downloader = DataDownloaderFactory.create_downloader("av", api_key="your_key")
binance_downloader = DataDownloaderFactory.create_downloader("bnc", api_key="your_key", secret_key="your_secret")

# Get fundamental data
fundamentals = yahoo_downloader.get_fundamentals("AAPL")
```

**Supported Provider Codes:**
- `"yf"` or `"yahoo"` → Yahoo Finance
- `"av"` or `"alphavantage"` → Alpha Vantage
- `"fh"` or `"finnhub"` → Finnhub
- `"pg"` or `"polygon"` → Polygon.io
- `"td"` or `"twelvedata"` → Twelve Data
- `"bnc"` or `"binance"` → Binance
- `"cg"` or `"coingecko"` → CoinGecko

**Environment Variable Support:**
The factory automatically reads API keys from environment variables:
- `ALPHA_VANTAGE_API_KEY`
- `FINNHUB_API_KEY`
- `POLYGON_API_KEY`
- `TWELVEDATA_API_KEY`
- `BINANCE_API_KEY`
- `BINANCE_SECRET_KEY`

### Data Feed Factory

The `DataFeedFactory` provides a unified interface for creating live data feeds:

```python
from src.data.data_feed_factory import DataFeedFactory

# Create a live feed
feed = DataFeedFactory.create_live_feed("binance", symbols=["BTCUSDT"])

# Create a CoinGecko live feed
coingecko_config = {
    "data_source": "coingecko",
    "symbol": "bitcoin",
    "interval": "1h",
    "polling_interval": 60
}
feed = DataFeedFactory.create_data_feed(coingecko_config)
```

### Data Downloader Factory Usage Examples

The `DataDownloaderFactory` makes it easy to create downloaders using short provider codes:

```python
from src.data.data_downloader_factory import DataDownloaderFactory

# Yahoo Finance (no API key required)
yahoo_downloader = DataDownloaderFactory.create_downloader("yf")
fundamentals = yahoo_downloader.get_fundamentals("AAPL")

# Alpha Vantage (API key from environment variable)
alpha_vantage_downloader = DataDownloaderFactory.create_downloader("av")
fundamentals = alpha_vantage_downloader.get_fundamentals("AAPL")

# Binance (API credentials from environment variables)
binance_downloader = DataDownloaderFactory.create_downloader("bnc")
df = binance_downloader.get_ohlcv("BTCUSDT", "1d", "2023-01-01", "2023-12-31")

# List all available providers
DataDownloaderFactory.list_providers()

# Get provider information
provider_info = DataDownloaderFactory.get_provider_info()
```

**Quick Provider Reference:**
- `"yf"` - Yahoo Finance (no API key)
- `"av"` - Alpha Vantage (API key required)
- `"fh"` - Finnhub (API key required)
- `"pg"` - Polygon.io (API key required)
- `"td"` - Twelve Data (API key required)
- `"bnc"` - Binance (API key + secret required)
- `"cg"` - CoinGecko (no API key)

## Fundamentals Data Model

All data downloaders return fundamental data using the `Fundamentals` dataclass:

```python
@dataclass
class Fundamentals:
    ticker: str
    company_name: str
    current_price: float
    market_cap: float
    pe_ratio: float
    forward_pe: float
    dividend_yield: float
    earnings_per_share: float
    # Additional fields for comprehensive analysis
    price_to_book: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    revenue: Optional[float] = None
    revenue_growth: Optional[float] = None
    net_income: Optional[float] = None
    net_income_growth: Optional[float] = None
    free_cash_flow: Optional[float] = None
    operating_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    beta: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    shares_outstanding: Optional[float] = None
    float_shares: Optional[float] = None
    short_ratio: Optional[float] = None
    payout_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_sales: Optional[float] = None
    enterprise_value: Optional[float] = None
    enterprise_value_to_ebitda: Optional[float] = None
    data_source: str = "Unknown"
    last_updated: str = ""
```

## Usage Examples

### Basic Data Download

```python
from src.data.yahoo_data_downloader import YahooDataDownloader

# Initialize downloader
downloader = YahooDataDownloader()

# Download historical data
df = downloader.get_ohlcv("AAPL", "1d", "2023-01-01", "2023-12-31")

# Get fundamental data
fundamentals = downloader.get_fundamentals("AAPL")
print(f"Company: {fundamentals.company_name}")
print(f"PE Ratio: {fundamentals.pe_ratio}")
print(f"Market Cap: ${fundamentals.market_cap:,.0f}")
```

### Batch Data Download

```python
from src.data.yahoo_data_downloader import YahooDataDownloader

downloader = YahooDataDownloader()
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

# Download data for multiple symbols
for symbol in symbols:
    df = downloader.get_ohlcv(symbol, "1d", "2023-01-01", "2023-12-31")
    fundamentals = downloader.get_fundamentals(symbol)
    print(f"{symbol}: PE={fundamentals.pe_ratio}, Market Cap=${fundamentals.market_cap:,.0f}")
```

### Live Data Streaming

```python
from src.data.binance_live_feed import BinanceLiveFeed

def on_data(data):
    print(f"Received: {data}")

feed = BinanceLiveFeed(["BTCUSDT", "ETHUSDT"])
feed.set_callback(on_data)
feed.start()
```

```python
from src.data.coingecko_live_feed import CoinGeckoLiveDataFeed

def on_new_bar(symbol, timestamp, data):
    print(f"New {symbol} data at {timestamp}: {data}")

feed = CoinGeckoLiveDataFeed("bitcoin", "1h", polling_interval=60)
feed.on_new_bar = on_new_bar
feed.start()
```

## Error Handling

All data downloaders include comprehensive error handling:

```python
try:
    fundamentals = downloader.get_fundamentals("INVALID_SYMBOL")
except Exception as e:
    print(f"Error: {e}")
    # Returns default Fundamentals object with error information
```

## Rate Limiting

Be aware of API rate limits when using multiple data sources:

- **Yahoo Finance**: No limits for basic usage
- **Alpha Vantage**: 5 calls/minute, 500/day (free tier)
- **Finnhub**: 60 calls/minute (free tier)
- **Polygon.io**: 5 calls/minute (free tier)
- **Twelve Data**: 8 calls/minute, 800/day (free tier)
- **Binance**: 1200 requests/minute (free tier)
- **CoinGecko**: 50 calls/minute (free tier)

## Configuration

Data downloaders can be configured through environment variables or direct initialization:

```python
# Using environment variables
import os
os.environ["ALPHA_VANTAGE_API_KEY"] = "your_api_key"
os.environ["FINNHUB_API_KEY"] = "your_api_key"

# Direct initialization
downloader = AlphaVantageDataDownloader(api_key="your_api_key")
```

## Best Practices

1. **Choose the right data source** based on your needs:
   - For comprehensive fundamental analysis: Yahoo Finance, Alpha Vantage, or Finnhub
   - For basic data: Polygon.io or Twelve Data
   - For cryptocurrencies: Binance or CoinGecko

2. **Handle rate limits** by implementing appropriate delays between requests

3. **Cache data** when possible to avoid repeated API calls

4. **Use error handling** to gracefully handle API failures

5. **Monitor data quality** and validate results

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
6. Update Requirements.md if new dependencies are needed
7. Document design decisions in Design.md
8. Add tasks to Tasks.md for future enhancements 