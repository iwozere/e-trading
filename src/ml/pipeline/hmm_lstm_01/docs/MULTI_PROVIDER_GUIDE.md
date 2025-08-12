# Multi-Provider Data Configuration Guide

## Overview

The HMM-LSTM pipeline now supports multiple data providers with provider-specific configurations. This allows you to process different types of financial data (cryptocurrencies, stocks, forex) from various sources in a single pipeline run.

## Configuration Format

### New Multi-Provider Configuration (Recommended)

```yaml
# Multi-provider data configuration
data_sources:
  binance:
    symbols: [LTCUSDT, BTCUSDT]
    timeframes: [5m, 4h]
  yfinance:
    symbols: [AAPL, MSFT]
    timeframes: [4h, 1d]
  alphavantage:
    symbols: [EURUSD, GBPUSD]
    timeframes: [1h, 1d]

# Legacy support - will be deprecated
symbols: [LTCUSDT]
timeframes: [5m, 15m, 1h, 4h]
```

### Legacy Configuration (Deprecated)

```yaml
symbols: [LTCUSDT]
timeframes: [5m, 15m, 1h, 4h]
data:
  provider: "bnc"  # Default provider
```

## Supported Data Providers

| Provider | Code | Asset Types | API Key Required | Rate Limits | Best For |
|----------|------|-------------|------------------|-------------|----------|
| **Binance** | `binance` | Cryptocurrencies | No | 1200 req/min | Crypto trading |
| **Yahoo Finance** | `yfinance` | Stocks, ETFs | No | None | Stock analysis |
| **Alpha Vantage** | `alphavantage` | Stocks, Forex | Yes | 5 req/min (free) | Professional data |
| **Finnhub** | `finnhub` | Stocks | Yes | 60 req/min (free) | Real-time data |
| **Polygon.io** | `polygon` | US Stocks | Yes | 5 req/min (free) | US markets |
| **Twelve Data** | `twelvedata` | Global markets | Yes | 8 req/min (free) | Global coverage |
| **CoinGecko** | `coingecko` | Cryptocurrencies | No | 50 req/min | Crypto research |

## Provider-Specific Considerations

### Binance (Cryptocurrencies)
- **Symbols**: Use trading pairs like `BTCUSDT`, `ETHUSDT`, `LTCUSDT`
- **Timeframes**: `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `3d`, `1w`, `1M`
- **Best for**: High-frequency crypto trading, short-term analysis

### Yahoo Finance (Stocks)
- **Symbols**: Use stock symbols like `AAPL`, `MSFT`, `GOOGL`
- **Timeframes**: `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo`
- **Best for**: Stock analysis, fundamental research

### Alpha Vantage (Professional)
- **Symbols**: Stocks (`AAPL`), Forex (`EURUSD`), Crypto (`BTCUSD`)
- **Timeframes**: `1min`, `5min`, `15min`, `30min`, `60min`, `daily`, `weekly`, `monthly`
- **API Key**: Required (free tier available)
- **Best for**: Professional applications, comprehensive data

## File Naming Convention

### Multi-Provider Files
Files are saved with provider prefix to avoid conflicts:

```
data/raw/
├── binance_BTCUSDT_5m_20230101_20231231.csv
├── binance_LTCUSDT_4h_20230101_20231231.csv
├── yfinance_AAPL_4h_20230101_20231231.csv
├── yfinance_MSFT_1d_20230101_20231231.csv
└── alphavantage_EURUSD_1h_20230101_20231231.csv
```

### Legacy Files
```
data/raw/
├── BTCUSDT_5m_20230101_20231231.csv
└── LTCUSDT_4h_20230101_20231231.csv
```

## Configuration Examples

### Crypto + Stocks Mix
```yaml
data_sources:
  binance:
    symbols: [BTCUSDT, ETHUSDT, LTCUSDT]
    timeframes: [5m, 15m, 1h, 4h]
  yfinance:
    symbols: [AAPL, MSFT, GOOGL, TSLA]
    timeframes: [1h, 4h, 1d]
```

### Multi-Asset Analysis
```yaml
data_sources:
  binance:
    symbols: [BTCUSDT, ETHUSDT]
    timeframes: [5m, 1h, 4h]
  yfinance:
    symbols: [SPY, QQQ, IWM]  # ETFs
    timeframes: [1h, 1d]
  alphavantage:
    symbols: [EURUSD, GBPUSD, USDJPY]  # Forex
    timeframes: [1h, 1d]
```

### Research Configuration
```yaml
data_sources:
  yfinance:
    symbols: [AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, AMD]
    timeframes: [1d]
  coingecko:
    symbols: [bitcoin, ethereum, litecoin, cardano, polkadot]
    timeframes: [1d]
```

## Pipeline Behavior

### Data Loading Stage
- **Multi-provider mode**: Downloads from all configured providers in parallel
- **Provider-specific error handling**: Each provider's failures are tracked separately
- **Progress tracking**: Shows progress per provider

### HMM Training Stage
- **Provider-aware file detection**: Looks for provider-prefixed files first
- **Fallback to legacy**: Falls back to legacy naming if provider-prefixed files not found
- **Provider-specific logging**: Shows which provider each model is trained on

### Pipeline Runner
- **Configuration detection**: Automatically detects multi-provider vs legacy configuration
- **Provider summary**: Shows statistics per provider in pipeline summary
- **Backward compatibility**: Legacy configuration still works

## Migration Guide

### From Legacy to Multi-Provider

1. **Update Configuration**
   ```yaml
   # OLD
   symbols: [BTCUSDT, ETHUSDT]
   timeframes: [5m, 1h, 4h]
   data:
     provider: "bnc"
   
   # NEW
   data_sources:
     binance:
       symbols: [BTCUSDT, ETHUSDT]
       timeframes: [5m, 1h, 4h]
   ```

2. **Re-download Data** (if needed)
   ```bash
   python run_pipeline.py --skip-stages 2,3,4,5,6,7,8
   ```

3. **Verify File Names**
   - New files will have provider prefix: `binance_BTCUSDT_5m_*.csv`
   - Old files remain: `BTCUSDT_5m_*.csv`

### Adding New Providers

1. **Add Provider Configuration**
   ```yaml
   data_sources:
     binance:
       symbols: [BTCUSDT]
       timeframes: [5m]
     yfinance:  # New provider
       symbols: [AAPL]
       timeframes: [1d]
   ```

2. **Set API Keys** (if required)
   ```bash
   export ALPHA_VANTAGE_KEY="your_key_here"
   export FINNHUB_KEY="your_key_here"
   ```

3. **Run Pipeline**
   ```bash
   python run_pipeline.py --skip-stages 2,3,4,5,6,7,8
   ```

## Best Practices

### Provider Selection
- **Crypto**: Use Binance for active trading, CoinGecko for research
- **Stocks**: Use Yahoo Finance for free access, Alpha Vantage for professional use
- **Forex**: Use Alpha Vantage or Twelve Data
- **Mixed**: Combine providers based on asset type

### Timeframe Selection
- **High-frequency**: 5m, 15m (crypto), 1m, 5m (stocks)
- **Swing trading**: 1h, 4h, 1d
- **Long-term**: 1d, 1w, 1mo

### Error Handling
- **API Limits**: Monitor rate limits, especially for free tiers
- **Data Quality**: Verify downloaded data quality before training
- **Fallbacks**: Use multiple providers for critical symbols

### Performance Optimization
- **Parallel Downloads**: Use multiple workers for faster downloads
- **Selective Downloads**: Only download timeframes you need
- **Caching**: Reuse downloaded data when possible

## Troubleshooting

### Common Issues

1. **"No data returned" errors**
   - Check symbol format (e.g., `BTCUSDT` vs `BTC-USD`)
   - Verify timeframe availability for the provider
   - Check API key validity (if required)

2. **Rate limit errors**
   - Reduce parallel workers: `--max-workers 2`
   - Add delays between requests
   - Use paid API tiers for higher limits

3. **File naming conflicts**
   - Ensure provider prefixes are used
   - Check for duplicate symbols across providers
   - Use different timeframes to avoid conflicts

### Debug Commands

```bash
# List downloaded files
python x_01_data_loader.py --list-files

# Validate data quality
python x_01_data_loader.py --validate-only

# Test specific provider
python x_01_data_loader.py --provider binance --symbols BTCUSDT

# Check configuration
python run_pipeline.py --validate-only
```

## Future Enhancements

### Planned Features
- **Provider-specific preprocessing**: Different feature engineering per provider
- **Cross-provider validation**: Compare data quality across providers
- **Automatic provider selection**: Choose best provider based on symbol
- **Provider performance metrics**: Track download success rates per provider

### Configuration Extensions
- **Provider-specific parameters**: Custom settings per provider
- **Conditional downloads**: Download based on market hours
- **Data validation rules**: Provider-specific quality checks
- **Backup providers**: Fallback providers for critical symbols

---

*This guide covers the multi-provider data configuration feature. For additional help, refer to the main README.md or create an issue in the project repository.*
