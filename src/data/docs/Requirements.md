# Requirements

## Python Dependencies

### Core Data Processing
- `pandas >= 2.0.0` - DataFrame operations and time series data manipulation
- `numpy >= 1.24.0` - Numerical computations and array operations

### Financial Data APIs
- `yfinance >= 0.2.18` - Yahoo Finance API for stock and ETF data (includes VIX data)
- `alpha_vantage >= 2.3.1` - Alpha Vantage API wrapper for financial data
- `python-binance >= 1.0.19` - Binance API for cryptocurrency data
- `alpaca-trade-api >= 3.0.0` - Alpaca API for US stock market data
- `requests >= 2.31.0` - HTTP library for API requests
- `websocket-client >= 1.6.0` - WebSocket client for real-time data feeds

### Database and Storage
- `sqlalchemy >= 2.0.0` - ORM and database abstraction layer
- `sqlite3` - Built-in Python SQLite database support (no additional dependency)
- `gzip` - Built-in Python compression for cache files (no additional dependency)
- `pathlib` - Built-in Python path handling (no additional dependency)

### Date and Time Handling
- `python-dateutil >= 2.8.2` - Date parsing and manipulation utilities
- `pytz >= 2023.3` - Timezone handling for global markets

### Configuration and Environment
- `python-dotenv >= 1.0.0` - Environment variable management
- `pyyaml >= 6.0` - YAML configuration file support

### Logging and Monitoring
- `colorlog >= 6.7.0` - Colored logging output for better debugging

### Data Processing and Utilities
- `scipy >= 1.11.0` - Scientific computing for data analysis
- `ta-lib >= 0.4.25` - Technical analysis library (optional)
- `ccxt >= 4.0.0` - Cryptocurrency exchange library (optional)

## External Services

### Required API Keys (Production)

#### Alpha Vantage API Key
- **Purpose**: Full historical intraday stock data (no 60-day limit)
- **Free tier**: 5 calls/minute, 500 calls/day
- **Environment variable**: `ALPHA_VANTAGE_API_KEY`
- **Registration**: [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
- **Usage**: Stock symbols with intraday timeframes (5m, 15m, 30m, 1h)

#### FMP API Key (Recommended)
- **Purpose**: Professional-grade financial data for stocks (primary provider for intraday data)
- **Free tier**: Limited requests per day
- **Paid tier**: 3000 requests/minute
- **Environment variable**: `FMP_API_KEY`
- **Registration**: [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs)
- **Usage**: Stock symbols with intraday timeframes (5m, 15m, 30m, 1h)

#### Alpaca API Credentials (Optional)
- **Purpose**: Professional-grade US market data and trading integration
- **Rate limit**: 200 requests/minute (free tier)
- **Environment variables**: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- **Registration**: [Alpaca](https://alpaca.markets/)
- **Usage**: US stocks and ETFs with high-quality data

#### Binance API Credentials (Optional)
- **Purpose**: Higher rate limits for cryptocurrency data
- **Rate limit**: 1200 requests/minute (free tier)
- **Environment variables**: `BINANCE_KEY`, `BINANCE_SECRET`
- **Registration**: [Binance](https://www.binance.com/en/my/settings/api-management)
- **Usage**: Cryptocurrency symbols (BTCUSDT, ETHUSDT, etc.)

### No API Key Required

#### Yahoo Finance
- **Purpose**: Stock fundamental data and daily historical data
- **Rate limits**: None for basic usage
- **Usage**: Stock symbols with daily timeframes (1d)

#### Binance Public API
- **Purpose**: Cryptocurrency data (public endpoints)
- **Rate limits**: 1200 requests/minute
- **Usage**: Cryptocurrency symbols (no API key needed for basic data)

## Database Requirements

### Database System
- **SQLite**: Built-in Python support, no additional installation required
- **Single Database**: Unified database for all operations (trading + telegram)
- **File Location**: `db/trading.db` (automatically created)
- **Schema Management**: Automatic table creation via SQLAlchemy metadata

### Database Features
- **ACID Compliance**: Full transaction support for data integrity
- **Concurrent Access**: SQLite supports multiple readers, single writer
- **Backup Support**: Standard SQLite backup tools and file-based backups
- **Migration Support**: SQLAlchemy-based schema migrations

### Database Models
- **Trading Models**: Trade lifecycle, bot instances, performance metrics
- **Telegram Models**: User management, alerts, schedules, audit logs
- **Shared Base**: Single SQLAlchemy Base for unified metadata management

### Database Performance
- **Indexes**: Optimized indexes for common query patterns
- **Connection Pooling**: SQLAlchemy connection pooling for efficiency
- **Session Management**: Automatic session cleanup via context managers
- **Query Optimization**: Repository pattern with optimized queries

## System Requirements

### Network Requirements
- **Internet Connection**: Stable broadband for real-time data feeds
- **Latency**: Low latency preferred for high-frequency trading
- **Bandwidth**: Moderate (streaming data can be bandwidth intensive)

### Hardware Requirements
- **Memory**: Minimum 4GB RAM (8GB+ recommended for live trading)
- **Storage**: 10GB+ available space for historical data cache and database
  - Database file: 10-100MB typical size for trading and telegram data
  - Cache files are gzip compressed for efficient storage
  - Typical cache size: 1-5GB for comprehensive historical data
- **CPU**: Multi-core processor recommended for parallel data processing

### Operating System Compatibility
- **Windows**: Windows 10+ (PowerShell support)
- **macOS**: macOS 10.15+ (Catalina or later)
- **Linux**: Ubuntu 20.04+ or equivalent distributions

## Cache System Requirements

### Unified Cache Structure
- **Directory Structure**: `symbol/timeframe/` (e.g., `BTCUSDT/5m/`)
- **File Format**: Gzip-compressed CSV files (`.csv.gz`)
- **Metadata**: JSON files for data source and quality information
- **Storage**: Local file system (no database required)

### Fundamentals Cache Structure
- **Directory Structure**: `fundamentals/{symbol}/` (e.g., `fundamentals/AAPL/`)
- **File Format**: JSON files with provider and timestamp naming
- **Naming Convention**: `{provider}_{timestamp}.json` (e.g., `yfinance_20250106_143022.json`)
- **Cache Duration**: 7-day cache-first rule for all stock providers
- **Multi-Provider Support**: Combine snapshots from multiple providers (yfinance, IBKR, FMP, Alpha Vantage)
- **Stale Data Cleanup**: Automatic removal of outdated fundamentals data

### Cache Directory Permissions
- **Write Access**: Full write permissions to cache directory
- **Read Access**: Read access for data retrieval
- **Default Location**: `./data-cache` (project root)
- **Custom Location**: Configurable via `--cache-dir` parameter

### Cache Size Management
- **Default Limit**: 10GB maximum cache size
- **Configurable**: Adjustable via `max_size_gb` parameter
- **Cleanup**: Automatic cleanup of old data (configurable age limit)
- **Compression**: All data files are gzip compressed for efficiency

### Fundamentals Cache Management
- **Cache Duration**: 7-day maximum age for fundamentals data
- **Provider Priority**: FMP > Yahoo Finance > Alpha Vantage > IBKR > others
- **Data Combination**: Merge snapshots from multiple providers for comprehensive data
- **Stale Data Cleanup**: Automatic removal of outdated fundamentals when new data is downloaded
- **Storage Efficiency**: JSON format with provider-specific field optimization

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
conda install pandas numpy python-dateutil pytz
pip install -r requirements.txt
```

### Environment Setup
1. Copy `.env.example` to `.env`
2. Add your API keys to the `.env` file:
```bash
# Financial Data APIs
ALPHA_VANTAGE_API_KEY=your_ALPHA_VANTAGE_API_KEY

# Cryptocurrency APIs (optional)
BINANCE_KEY=your_BINANCE_KEY
BINANCE_SECRET=your_BINANCE_SECRET
```

## Cache System Setup

### Initial Cache Population
```bash
# Populate cache with default symbols and timeframes
python src/data/cache/populate_cache.py

# Custom cache directory
python src/data/cache/populate_cache.py --cache-dir /path/to/cache

# Specific symbols and timeframes
python src/data/cache/populate_cache.py --symbols BTCUSDT,AAPL,GOOGL --intervals 5m,15m,1h,1d
```

### Cache Migration (if upgrading from old system)
```bash
# Migrate from old provider-based cache structure
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

### Cache Security
- **Local Storage**: Cache data is stored locally on your system
- **No Sensitive Data**: Cache contains only market data, no personal information
- **Access Control**: Ensure proper file system permissions for cache directory

## Rate Limiting and Fair Usage

### API Rate Limits (Free Tiers)
| Provider | Calls/Minute | Calls/Day | Notes |
|----------|--------------|-----------|-------|
| Yahoo Finance | Unlimited* | Unlimited* | *Rate limited but not specified |
| Alpha Vantage | 5 | 500 | Upgrade available |
| Binance | 1200 | - | Crypto only |

### Built-in Rate Limiting
- **Cache Population**: 0.2 second delay between operations
- **Downloader Throttling**: Built-in rate limiting in each downloader
- **Automatic Retry**: Exponential backoff for failed requests

### Best Practices
- Implement exponential backoff for rate limiting
- Cache data locally to reduce API calls
- Use multiple providers for redundancy
- Monitor usage to avoid hitting limits

## Troubleshooting

### Common Issues

#### Cache Issues
1. **Cache Directory Permissions**: Ensure write access to cache directory
2. **Disk Space**: Monitor available disk space for cache growth
3. **File Corruption**: Use cache validation tools to check data integrity

#### API Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **API Key Errors**: Verify environment variables are set correctly
3. **Rate Limiting**: Built-in throttling should handle this automatically
4. **Data Quality**: Validate data from multiple sources when possible
5. **Network Issues**: Implement retry logic with exponential backoff

#### Provider Selection Issues
1. **Symbol Classification**: Check if symbol is properly classified as crypto or stock
2. **Provider Availability**: Verify that the selected provider is available
3. **Fallback Behavior**: System falls back to mock data if no provider is available

### Debug Mode
Enable debug logging to troubleshoot issues:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Cache Debugging
```bash
# Check cache structure
ls -la ./data-cache/

# Validate specific cache files
python src/data/cache/cleanup_failed_cache.py --validate-only

# Check cache statistics
python -c "from src.data.cache.unified_cache import configure_unified_cache; cache = configure_unified_cache(); print(cache.get_stats())"
```

### Support Resources
- Check provider documentation for API changes
- Monitor provider status pages for outages
- Join community forums for troubleshooting tips
- Review cache validation logs for data quality issues

## Performance Considerations

### Cache Performance
- **Gzip Compression**: Reduces storage space by 60-80%
- **Local Access**: Fast local file system access
- **Incremental Updates**: Only download missing data
- **Parallel Processing**: Support for concurrent operations

### Memory Usage
- **DataFrames**: Efficient pandas DataFrame operations
- **Lazy Loading**: Load data on-demand
- **Memory Management**: Automatic cleanup of large datasets

### Network Optimization
- **Connection Pooling**: Reuse HTTP connections
- **Compression**: Gzip compression for data transfer
- **Rate Limiting**: Respect API limits with built-in throttling

## Monitoring and Maintenance

### Cache Monitoring
- **Size Monitoring**: Track cache growth over time
- **Quality Monitoring**: Monitor data quality scores
- **Provider Performance**: Track provider success rates

### Regular Maintenance
- **Cache Cleanup**: Remove old data periodically
- **Validation**: Regular cache validation checks
- **Backup**: Backup important cache data
- **Updates**: Keep dependencies updated

### Health Checks
```bash
# Check cache health
python src/data/cache/populate_cache.py --test-all

# Validate cache integrity
python src/data/cache/cleanup_failed_cache.py --validate-and-cleanup
```