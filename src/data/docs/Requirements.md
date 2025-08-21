# Requirements

## Python Dependencies

### Core Data Processing
- `pandas >= 2.0.0` - DataFrame operations and time series data manipulation
- `numpy >= 1.24.0` - Numerical computations and array operations

### Financial Data APIs
- `yfinance >= 0.2.18` - Yahoo Finance API for stock and ETF data (includes VIX data)
- `alpha_vantage >= 2.3.1` - Alpha Vantage API wrapper for financial data
- `finnhub-python >= 2.4.15` - Finnhub API client for real-time market data
- `polygon-api-client >= 1.12.5` - Polygon.io API for US market data
- `twelvedata >= 1.2.14` - Twelve Data API for global market coverage
- `python-binance >= 1.0.19` - Binance API for cryptocurrency data
- `pycoingecko >= 3.1.0` - CoinGecko API for cryptocurrency market data

### Trading and Backtesting Framework
- `backtrader >= 1.9.78.123` - Backtesting and live trading framework
- `ib_insync >= 0.9.86` - Interactive Brokers API integration

### Real-time Data and WebSockets
- `websockets >= 11.0.3` - WebSocket client for real-time data feeds
- `websocket-client >= 1.6.1` - Alternative WebSocket implementation
- `aiohttp >= 3.8.5` - Async HTTP client for REST API calls

### Database and Storage
- `sqlalchemy >= 2.0.0` - SQL toolkit and ORM for database operations
- `sqlite3` - Built-in SQLite database support (Python standard library)
- `psycopg2-binary >= 2.9.7` - PostgreSQL adapter (optional for production)

### Date and Time Handling
- `python-dateutil >= 2.8.2` - Date parsing and manipulation utilities
- `pytz >= 2023.3` - Timezone handling for global markets

### Configuration and Environment
- `python-dotenv >= 1.0.0` - Environment variable management
- `pyyaml >= 6.0.1` - YAML configuration file support

### Logging and Monitoring
- `colorlog >= 6.7.0` - Colored logging output for better debugging

## External Services

### Required API Keys (Production)
- **Alpha Vantage API Key** - For comprehensive fundamental data
  - Free tier: 5 calls/minute, 500 calls/day
  - Environment variable: `ALPHA_VANTAGE_API_KEY`
  - Registration: [Alpha Vantage](https://www.alphavantage.co/support/#api-key)

- **Finnhub API Key** - For real-time market data and fundamentals
  - Free tier: 60 calls/minute
  - Environment variable: `FINNHUB_API_KEY`
  - Registration: [Finnhub](https://finnhub.io/)

- **Polygon.io API Key** - For US market data
  - Free tier: 5 calls/minute
  - Environment variable: `POLYGON_API_KEY`
  - Registration: [Polygon.io](https://polygon.io/)

- **Twelve Data API Key** - For global market coverage
  - Free tier: 8 calls/minute, 800 calls/day
  - Environment variable: `TWELVEDATA_API_KEY`
  - Registration: [Twelve Data](https://twelvedata.com/)

- **Binance API Credentials** - For cryptocurrency data
  - Rate limit: 1200 requests/minute
  - Environment variables: `BINANCE_API_KEY`, `BINANCE_SECRET_KEY`
  - Registration: [Binance](https://www.binance.com/en/my/settings/api-management)

### Optional API Keys
- **Interactive Brokers Account** - For professional trading data
  - Requires TWS or IB Gateway installation
  - Market data subscriptions may be required
  - Configuration: Host, Port, Client ID

### No API Key Required
- **Yahoo Finance** - Basic stock and ETF data (rate limited)
- **CoinGecko** - Cryptocurrency data (50 calls/minute free tier)
- **VIX Data** - CBOE VIX volatility index data via Yahoo Finance

## System Requirements

### Network Requirements
- **Internet Connection**: Stable broadband for real-time data feeds
- **Latency**: Low latency preferred for high-frequency trading
- **Bandwidth**: Moderate (streaming data can be bandwidth intensive)

### Hardware Requirements
- **Memory**: Minimum 4GB RAM (8GB+ recommended for live trading)
- **Storage**: 10GB+ available space for historical data storage
  - VIX data: ~50MB for complete historical dataset (1990-present)
- **CPU**: Multi-core processor recommended for parallel data processing

### Operating System Compatibility
- **Windows**: Windows 10+ (PowerShell support)
- **macOS**: macOS 10.15+ (Catalina or later)
- **Linux**: Ubuntu 20.04+ or equivalent distributions

## Development Dependencies (Optional)

### Testing Framework
- `pytest >= 7.4.0` - Testing framework
- `pytest-asyncio >= 0.21.1` - Async testing support
- `pytest-mock >= 3.11.1` - Mocking utilities for tests

### Code Quality
- `black >= 23.7.0` - Code formatting
- `flake8 >= 6.0.0` - Code linting
- `isort >= 5.12.0` - Import sorting

### Documentation
- `sphinx >= 7.1.0` - Documentation generation
- `sphinx-rtd-theme >= 1.3.0` - ReadTheDocs theme

## Installation Instructions

### Using pip
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Using conda
```bash
# Create conda environment
conda create -n e-trading python=3.11
conda activate e-trading

# Install dependencies
conda install pandas numpy sqlalchemy pyyaml
pip install -r requirements.txt
```

### Environment Setup
1. Copy `.env.example` to `.env`
2. Add your API keys to the `.env` file:
```bash
# Financial Data APIs
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key
POLYGON_API_KEY=your_polygon_key
TWELVEDATA_API_KEY=your_twelvedata_key

# Cryptocurrency APIs
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
```

## Security Considerations

### API Key Management
- **Never commit API keys to version control**
- Use environment variables or secure key management systems
- Rotate API keys regularly
- Use different keys for development and production

### Network Security
- Use HTTPS for all API communications (enforced by libraries)
- Consider VPN for production deployments
- Monitor API usage to detect unauthorized access

### Data Privacy
- Comply with financial data regulations (SEC, GDPR, etc.)
- Implement proper data retention policies
- Secure storage of historical trading data

## Rate Limiting and Fair Usage

### API Rate Limits (Free Tiers)
| Provider | Calls/Minute | Calls/Day | Notes |
|----------|--------------|-----------|-------|
| Yahoo Finance | Unlimited* | Unlimited* | *Rate limited but not specified |
| Alpha Vantage | 5 | 500 | Upgrade available |
| Finnhub | 60 | - | Real-time data |
| Polygon.io | 5 | - | US markets only (free) |
| Twelve Data | 8 | 800 | Global coverage |
| Binance | 1200 | - | Crypto only |
| CoinGecko | 50 | - | Crypto only |

### Best Practices
- Implement exponential backoff for rate limiting
- Cache data locally to reduce API calls
- Use multiple providers for redundancy
- Monitor usage to avoid hitting limits

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **API Key Errors**: Verify environment variables are set correctly
3. **Rate Limiting**: Implement proper delays between API calls
4. **Data Quality**: Validate data from multiple sources when possible
5. **Network Issues**: Implement retry logic with exponential backoff

### Debug Mode
Enable debug logging to troubleshoot issues:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Support Resources
- Check provider documentation for API changes
- Monitor provider status pages for outages
- Join community forums for troubleshooting tips
