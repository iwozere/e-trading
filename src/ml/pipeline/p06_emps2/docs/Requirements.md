# Requirements

## Python Dependencies

### Core Libraries
- `pandas >= 1.3.0` - Data manipulation and analysis
- `numpy >= 1.21.0` - Numerical operations
- `requests >= 2.26.0` - HTTP requests for NASDAQ Trader API

### Data Providers
- `yfinance >= 0.2.0` - Yahoo Finance data (no API key required)
- Finnhub API key - Required for fundamental data (free tier: 60 calls/min)

### Technical Indicators
- `TA-Lib >= 0.4.0` - Technical analysis library for ATR calculation

### Project Dependencies
- `src.data.downloader.finnhub_data_downloader` - Finnhub integration
- `src.data.downloader.yahoo_data_downloader` - Yahoo Finance integration
- `src.notification.logger` - Logging framework
- `src.model.schemas` - Data schemas

## External Services

### NASDAQ Trader FTP
- **URL:** ftp://ftp.nasdaqtrader.com/SymbolDirectory/
- **Authentication:** None required (public FTP)
- **Rate Limits:** None
- **Data:** Complete US stock universe (NASDAQ, NYSE, AMEX)

### Finnhub API
- **Purpose:** Fundamental data (market cap, float, sector, volume)
- **Rate Limits:** 60 calls per minute (free tier)
- **API Key:** Required (set in `config/donotshare/donotshare.py`)
- **Endpoint:** https://finnhub.io/api/v1/
- **Free Tier Limitations:**
  - 60 API calls/minute
  - Basic fundamental data
  - Real-time quotes

### Yahoo Finance
- **Purpose:** Intraday OHLCV data (15-minute bars)
- **Rate Limits:** Reasonable (supports batch operations)
- **API Key:** Not required
- **Data Availability:** Last 60 days of intraday data
- **Intervals Supported:** 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d

## System Requirements

### Minimum Requirements
- **RAM:** 2 GB
- **Disk Space:** 100 MB (for results and cache)
- **Network:** Broadband internet connection
- **Python:** 3.8+

### Recommended Requirements
- **RAM:** 4 GB (for processing large batches)
- **Disk Space:** 500 MB (for historical results)
- **Network:** Stable broadband (for API calls)

## Installation

### 1. Install Python Dependencies

```bash
# Core dependencies
pip install pandas numpy requests yfinance

# TA-Lib (platform-specific)
# Windows: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.XX‑cpXX‑cpXX‑win_amd64.whl

# Linux/Mac:
# First install TA-Lib C library, then:
pip install TA-Lib
```

### 2. Configure API Keys

Add Finnhub API key to `config/donotshare/donotshare.py`:

```python
FINNHUB_KEY = "your_api_key_here"
```

Get free API key at: https://finnhub.io/register

### 3. Verify Installation

```bash
python -c "import talib; print('TA-Lib version:', talib.__version__)"
python -c "from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader; print('Finnhub OK')"
python -c "from src.data.downloader.yahoo_data_downloader import YahooDataDownloader; print('Yahoo OK')"
```

## Environment Variables

No environment variables required. All configuration is in code.

## Cross-Module Dependencies

### Depends On
- `src.data.downloader` - Data downloaders
- `src.notification.logger` - Logging
- `src.model.schemas` - Data schemas

### Used By
- Can be used standalone
- Can provide input to P05 EMPS pipeline
- Results can be consumed by any trading strategy

## Performance Considerations

### API Call Estimates
- **NASDAQ Universe:** 2 API calls (NASDAQ + Other exchanges)
- **Fundamental Filter:** ~8000 tickers × 2 calls = ~16,000 calls (~4.5 hours at 60 calls/min)
- **Volatility Filter:** ~500-1000 tickers × 1 batch call = ~1 batch operation
- **Total Time:** ~4-5 hours for full pipeline (due to Finnhub rate limits)

### Optimization Strategies
1. **Cache universe:** Reduce NASDAQ downloads (24h TTL)
2. **Batch processing:** Use Yahoo Finance batch downloads
3. **Rate limiting:** Respect Finnhub limits (1.1s between calls)
4. **Parallel processing:** Future enhancement for faster processing

## Security Requirements

### API Key Management
- Store Finnhub API key in `config/donotshare/donotshare.py`
- Never commit API keys to version control
- `.gitignore` should include `config/donotshare/`

### Data Privacy
- All data is public market data
- No PII (Personally Identifiable Information)
- Results stored locally only

## Known Limitations

1. **Finnhub Rate Limits:** Free tier limited to 60 calls/minute
2. **Yahoo Finance Data:** Only last 60 days of intraday data
3. **Processing Time:** ~4-5 hours for full 8000-ticker scan
4. **Float Data:** Not all tickers have float data available
5. **Market Hours:** Best results during market hours for real-time data

## Future Requirements

### Planned Enhancements
- Premium Finnhub subscription for faster processing
- Alternative fundamental data providers (FMP, Alpha Vantage)
- Database storage for historical results
- Parallel processing for faster execution
- Real-time updates during market hours

---

**Last Updated:** 2025-11-27
