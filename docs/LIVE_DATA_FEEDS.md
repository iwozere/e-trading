# Live Data Feeds Documentation

This document describes the live data feeds system that provides real-time market data to the trading platform.

## Overview

The live data feeds system provides real-time market data from multiple sources:
- **Binance**: Cryptocurrency data with WebSocket support
- **Yahoo Finance**: Stock and ETF data via polling
- **Interactive Brokers (IBKR)**: Professional trading data via native API

All data feeds inherit from `BaseLiveDataFeed` and integrate seamlessly with Backtrader.

## Architecture

### Base Class: `BaseLiveDataFeed`

The base class provides:
- Common interface for all data feeds
- Historical data loading with configurable lookback
- Real-time data updates
- Automatic error handling and reconnection
- Backtrader integration
- Status monitoring

### Data Feed Implementations

#### 1. Binance Live Data Feed (`BinanceLiveDataFeed`)

**Features:**
- WebSocket-based real-time updates
- Historical data via REST API
- Automatic reconnection on connection loss
- Support for testnet and mainnet

**Configuration:**
```json
{
    "data_source": "binance",
    "symbol": "BTCUSDT",
    "interval": "1m",
    "lookback_bars": 1000,
    "retry_interval": 60,
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "testnet": false
}
```

**Supported Intervals:** 1m, 5m, 15m, 30m, 1h, 4h, 1d

#### 2. Yahoo Finance Live Data Feed (`YahooLiveDataFeed`)

**Features:**
- Polling-based real-time updates
- Historical data via yfinance
- Configurable polling intervals
- No authentication required

**Configuration:**
```json
{
    "data_source": "yahoo",
    "symbol": "AAPL",
    "interval": "5m",
    "lookback_bars": 500,
    "retry_interval": 60,
    "polling_interval": 60
}
```

**Supported Intervals:** 1m, 5m, 15m, 30m, 1h, 4h, 1d

#### 3. IBKR Live Data Feed (`IBKRLiveDataFeed`)

**Features:**
- Native API real-time updates
- Historical data via IBKR API
- Professional-grade data quality
- Requires IBKR TWS or Gateway

**Configuration:**
```json
{
    "data_source": "ibkr",
    "symbol": "SPY",
    "interval": "1m",
    "lookback_bars": 1000,
    "retry_interval": 60,
    "host": "127.0.0.1",
    "port": 7497,
    "client_id": 1
}
```

**Supported Intervals:** 1m, 5m, 15m, 30m, 1h, 4h, 1d

## Usage

### Basic Usage

```python
from src.data.data_feed_factory import DataFeedFactory
import backtrader as bt

# Create configuration
config = {
    "data_source": "binance",
    "symbol": "BTCUSDT",
    "interval": "1m",
    "lookback_bars": 1000,
    "retry_interval": 60
}

# Create data feed
data_feed = DataFeedFactory.create_data_feed(config)

# Use with Backtrader
cerebro = bt.Cerebro()
cerebro.adddata(data_feed)
cerebro.run()
```

### With Callback Function

```python
def on_new_bar(symbol, timestamp, data):
    print(f"New {symbol} bar: Close={data['close']}")

config = {
    "data_source": "binance",
    "symbol": "BTCUSDT",
    "interval": "1m",
    "on_new_bar": on_new_bar
}

data_feed = DataFeedFactory.create_data_feed(config)
```

### Using Factory Pattern

```python
from src.data.data_feed_factory import DataFeedFactory

# Load configuration from file
with open('config/trading/live_data_example.json', 'r') as f:
    config = json.load(f)

# Create data feed
feed_config = config['data_feeds']['binance_example']
data_feed = DataFeedFactory.create_data_feed(feed_config)
```

## Configuration

### Common Parameters

All data feeds support these common parameters:

- `data_source`: Data source identifier ("binance", "yahoo", "ibkr")
- `symbol`: Trading symbol (e.g., "BTCUSDT", "AAPL", "SPY")
- `interval`: Data interval (e.g., "1m", "5m", "1h", "1d")
- `lookback_bars`: Number of historical bars to load initially
- `retry_interval`: Seconds to wait before retrying on connection failure
- `on_new_bar`: Optional callback function for new data notifications

### Source-Specific Parameters

#### Binance
- `api_key`: Binance API key (optional for public data)
- `api_secret`: Binance API secret (optional for public data)
- `testnet`: Use Binance testnet (default: False)

#### Yahoo Finance
- `polling_interval`: Seconds between polling attempts (default: 60)

#### IBKR
- `host`: IBKR TWS/Gateway host (default: "127.0.0.1")
- `port`: IBKR TWS/Gateway port (default: 7497 for TWS, 4001 for Gateway)
- `client_id`: IBKR client ID (default: 1)

## Data Format

All data feeds return data in the same format:

```python
DataFrame with columns:
- datetime: Timestamp index
- open: Opening price
- high: High price
- low: Low price
- close: Closing price
- volume: Trading volume
```

## Error Handling

### Automatic Reconnection

All data feeds implement automatic reconnection:
- Detects connection loss
- Waits for specified retry interval
- Attempts to reconnect automatically
- Logs connection status

### Error Logging

Errors are logged with appropriate levels:
- `INFO`: Normal operations
- `WARNING`: Non-critical issues
- `ERROR`: Critical errors requiring attention

## Performance Considerations

### Rate Limits

- **Binance**: High frequency, WebSocket-based
- **Yahoo Finance**: Moderate frequency, polling-based
- **IBKR**: High frequency, native API

### Memory Usage

- Historical data is loaded once at startup
- Real-time data is appended incrementally
- Old data can be trimmed to manage memory

### Network Usage

- **Binance**: WebSocket connection (low overhead)
- **Yahoo Finance**: HTTP polling (moderate overhead)
- **IBKR**: Native API connection (low overhead)

## Integration with Trading Bots

### Using with run_bot.py

```python
# In your trading bot configuration
{
    "data_feed": {
        "data_source": "binance",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "lookback_bars": 1000
    },
    "strategy": {
        "type": "rsi_bb_volume",
        "params": {...}
    }
}
```

### Using with Backtrader Strategies

```python
class MyStrategy(bt.Strategy):
    def __init__(self):
        # Access data lines
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.dataopen = self.datas[0].open
        self.datavolume = self.datas[0].volume
    
    def next(self):
        # Process new data
        if not self.position:
            if self.dataclose[0] > self.dataclose[-1]:
                self.buy()
        else:
            if self.dataclose[0] < self.dataclose[-1]:
                self.sell()
```

## Testing

### Test Script

Use the provided test script to verify data feeds:

```bash
# Test Binance data feed
python test_live_data_feeds.py binance BTCUSDT 1m

# Test Yahoo Finance data feed
python test_live_data_feeds.py yahoo AAPL 5m

# Test IBKR data feed
python test_live_data_feeds.py ibkr SPY 1m
```

### Manual Testing

```python
from src.data.data_feed_factory import DataFeedFactory

# Create and test data feed
config = {
    "data_source": "binance",
    "symbol": "BTCUSDT",
    "interval": "1m",
    "lookback_bars": 10
}

data_feed = DataFeedFactory.create_data_feed(config)
print(data_feed.get_status())
```

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check network connectivity
   - Verify API keys (if required)
   - Check firewall settings

2. **No Data Received**
   - Verify symbol is valid
   - Check interval format
   - Ensure data source is available

3. **High Memory Usage**
   - Reduce `lookback_bars`
   - Implement data trimming
   - Monitor memory usage

4. **Rate Limit Exceeded**
   - Increase polling intervals
   - Use appropriate API keys
   - Implement rate limiting

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check data feed status:

```python
status = data_feed.get_status()
print(json.dumps(status, indent=2, default=str))
```

## Security Considerations

### API Keys

- Store API keys in environment variables
- Never commit keys to version control
- Use testnet for development

### Network Security

- Use secure connections (HTTPS/WSS)
- Implement proper authentication
- Monitor for suspicious activity

## Future Enhancements

### Planned Features

1. **Additional Data Sources**
   - Alpha Vantage
   - Polygon.io
   - Coinbase Pro

2. **Advanced Features**
   - Data caching
   - Multiple symbol support
   - Custom indicators

3. **Performance Improvements**
   - Async data processing
   - Connection pooling
   - Data compression

### Contributing

To add a new data source:

1. Create a new class inheriting from `BaseLiveDataFeed`
2. Implement required abstract methods
3. Add to `DataFeedFactory`
4. Update documentation
5. Add tests

## Support

For issues and questions:
- Check the troubleshooting section
- Review error logs
- Test with different configurations
- Consult the API documentation for each data source 