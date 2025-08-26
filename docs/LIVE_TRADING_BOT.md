# Live Trading Bot Documentation

This document describes the live trading bot system that orchestrates real-time trading with data feeds, strategies, brokers, and notifications.

## Overview

The live trading bot is a comprehensive system that:
- Reads configuration from JSON files
- Creates and manages live data feeds
- Executes trading strategies with Backtrader
- Manages broker connections and orders
- Handles position tracking and persistence
- Provides error recovery and notifications
- Integrates with web interface and monitoring

## Architecture

### Core Components

1. **LiveTradingBot**: Main orchestrator class
2. **ConfigValidator**: Configuration validation and error checking
3. **Data Feed Integration**: Live data from Binance, Yahoo, IBKR
4. **Strategy Execution**: CustomStrategy with entry/exit mixins
5. **Broker Integration**: Order execution and position management
6. **Notification System**: Telegram and email notifications
7. **Error Handling**: Automatic recovery and restart mechanisms

### Configuration Structure

The bot reads a single JSON configuration file with the following structure:

```json
{
    "description": "Strategy description",
    "version": "1.0",
    
    "broker": {
        "type": "binance_paper",
        "initial_balance": 1000.0,
        "commission": 0.001
    },
    
    "trading": {
        "symbol": "BTCUSDT",
        "position_size": 0.1,
        "max_positions": 1,
        "max_drawdown_pct": 20.0,
        "max_exposure": 1.0
    },
    
    "data": {
        "data_source": "binance",
        "symbol": "BTCUSDT",
        "interval": "1h",
        "lookback_bars": 1000,
        "retry_interval": 60
    },
    
    "strategy": {
        "type": "custom",
        "entry_logic": {
            "name": "RSIBBVolumeEntryMixin",
            "params": {...}
        },
        "exit_logic": {
            "name": "RSIBBExitMixin",
            "params": {...}
        },
        "position_size": 0.1
    },
    
    "notifications": {
        "enabled": true,
        "telegram": {
            "enabled": true,
            "notify_on": ["trade_entry", "trade_exit", "error", "status"]
        },
        "email": {
            "enabled": false,
            "notify_on": ["trade_entry", "trade_exit", "error"]
        }
    },
    
    "risk_management": {
        "stop_loss_pct": 5.0,
        "take_profit_pct": 10.0,
        "trailing_stop": {
            "enabled": false,
            "activation_pct": 3.0,
            "trailing_pct": 2.0
        },
        "max_daily_trades": 10,
        "max_daily_loss": 50.0
    },
    
    "logging": {
        "level": "INFO",
        "save_trades": true,
        "save_equity_curve": true,
        "log_file": "logs/live/trading_bot_0001.log"
    }
}
```

## Usage

### Basic Usage

```bash
# Run with configuration file
python src/trading/run_bot.py 0001.json

# Test configuration
python test_live_bot_config.py 0001.json
```

### Configuration Validation

The bot validates configuration before starting:

```bash
# Validate configuration
python src/trading/config_validator.py config/trading/0001.json
```

### Testing Components

Test individual components:

```bash
# Test data feeds
python test_live_data_feeds.py binance BTCUSDT 1m

# Test configuration
python test_live_bot_config.py 0001.json
```

## Configuration Sections

### Broker Configuration

```json
{
    "broker": {
        "type": "binance_paper|binance|ibkr|mock",
        "initial_balance": 1000.0,
        "commission": 0.001
    }
}
```

**Supported Brokers:**
- `binance_paper`: Binance paper trading (recommended for testing)
- `binance`: Live Binance trading (requires API keys)
- `ibkr`: Interactive Brokers (requires TWS/Gateway)
- `mock`: Mock broker for testing

### Trading Configuration

```json
{
    "trading": {
        "symbol": "BTCUSDT",
        "position_size": 0.1,
        "max_positions": 1,
        "max_drawdown_pct": 20.0,
        "max_exposure": 1.0
    }
}
```

**Parameters:**
- `symbol`: Trading symbol (e.g., "BTCUSDT", "AAPL")
- `position_size`: Position size as fraction of capital (0.1 = 10%)
- `max_positions`: Maximum concurrent positions
- `max_drawdown_pct`: Maximum drawdown percentage
- `max_exposure`: Maximum portfolio exposure (1.0 = 100%)

### Data Configuration

```json
{
    "data": {
        "data_source": "binance|yahoo|ibkr",
        "symbol": "BTCUSDT",
        "interval": "1m|5m|15m|30m|1h|4h|1d",
        "lookback_bars": 1000,
        "retry_interval": 60
    }
}
```

**Data Sources:**
- `binance`: Cryptocurrency data with WebSocket
- `yahoo`: Stock data with polling
- `ibkr`: Professional data with native API

### Strategy Configuration

```json
{
    "strategy": {
        "type": "custom",
        "entry_logic": {
            "name": "RSIBBVolumeEntryMixin",
            "params": {
                "e_rsi_period": 14,
                "e_rsi_oversold": 30,
                "e_bb_period": 20,
                "e_bb_dev": 2.0
            }
        },
        "exit_logic": {
            "name": "RSIBBExitMixin",
            "params": {
                "x_rsi_period": 14,
                "x_rsi_overbought": 70
            }
        },
        "position_size": 0.1
    }
}
```

**Available Entry Mixins:**
- `RSIBBVolumeEntryMixin`: RSI + Bollinger Bands + Volume
- `RSIBBEntryMixin`: RSI + Bollinger Bands
- `RSIIchimokuEntryMixin`: RSI + Ichimoku
- `RSIVolumeSupertrendEntryMixin`: RSI + Volume + SuperTrend

**Available Exit Mixins:**
- `RSIBBExitMixin`: RSI + Bollinger Bands
- `ATRExitMixin`: Average True Range
- `TrailingStopExitMixin`: Trailing stop loss
- `TimeBasedExitMixin`: Time-based exit
- `FixedRatioExitMixin`: Fixed profit/loss ratio

### Notifications Configuration

```json
{
    "notifications": {
        "enabled": true,
        "telegram": {
            "enabled": true,
            "notify_on": ["trade_entry", "trade_exit", "error", "status"]
        },
        "email": {
            "enabled": false,
            "notify_on": ["trade_entry", "trade_exit", "error"]
        }
    }
}
```

**Notification Events:**
- `trade_entry`: When a new position is opened
- `trade_exit`: When a position is closed
- `error`: When an error occurs
- `status`: Bot status updates
- `daily_summary`: Daily performance summary

## Bot Lifecycle

### 1. Initialization
- Load and validate configuration
- Setup notification systems
- Create data feed, broker, and strategy
- Load open positions from database

### 2. Data Feed Setup
- Initialize live data feed
- Load historical data for indicators
- Start real-time data updates
- Monitor connection health

### 3. Strategy Execution
- Initialize Backtrader engine
- Setup strategy with entry/exit mixins
- Start trading loop
- Process real-time signals

### 4. Position Management
- Execute buy/sell orders
- Track open positions
- Apply risk management rules
- Save position state

### 5. Monitoring
- Monitor data feed health
- Check broker connection
- Track performance metrics
- Send status notifications

### 6. Error Handling
- Detect connection issues
- Attempt automatic recovery
- Restart components if needed
- Send error notifications

## Error Recovery

### Automatic Recovery

The bot implements automatic recovery for common issues:

1. **Data Feed Disconnection**
   - Detects connection loss
   - Attempts reconnection every 60 seconds
   - Notifies via Telegram/email

2. **Broker API Errors**
   - Retries failed orders
   - Restarts broker connection
   - Falls back to paper trading if needed

3. **Strategy Errors**
   - Logs error details
   - Restarts strategy execution
   - Maintains open positions

### Manual Recovery

For persistent issues:

```bash
# Stop the bot
Ctrl+C

# Check logs
tail -f logs/live/trading_bot_0001.log

# Restart the bot
python src/trading/run_bot.py 0001.json
```

## Monitoring and Logging

### Log Files

- **Main Log**: `logs/live/trading_bot_0001.log`
- **Trade Log**: `logs/json/trades.json`
- **Order Log**: `logs/json/orders.json`
- **Error Log**: `logs/errors.log`

### Status Monitoring

Get bot status via Python:

```python
from src.trading.live_trading_bot import LiveTradingBot

bot = LiveTradingBot("0001.json")
status = bot.get_status()
print(status)
```

### Web Interface

Access bot management via web interface:

```bash
python src/management/webgui/app.py
```

Then visit: `http://localhost:5000`

## Performance Considerations

### Memory Usage
- Historical data loaded once at startup
- Real-time data appended incrementally
- Position data cached in memory

### Network Usage
- WebSocket connections (Binance, IBKR)
- HTTP polling (Yahoo Finance)
- API calls for order execution

### CPU Usage
- Indicator calculations on each bar
- Strategy signal processing
- Position management logic

## Security

### API Keys
- Store keys in environment variables
- Never commit keys to version control
- Use paper trading for testing

### Network Security
- Use secure connections (HTTPS/WSS)
- Implement proper authentication
- Monitor for suspicious activity

## Troubleshooting

### Common Issues

1. **Configuration Errors**
   ```bash
   # Validate configuration
   python src/trading/config_validator.py config/trading/0001.json
   ```

2. **Data Feed Issues**
   ```bash
   # Test data feed
   python test_live_data_feeds.py binance BTCUSDT 1m
   ```

3. **Broker Connection Issues**
   - Check API keys and permissions
   - Verify network connectivity
   - Test with paper trading first

4. **Strategy Errors**
   - Check indicator parameters
   - Verify data availability
   - Review strategy logic

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Monitor bot performance:

```python
# Get performance metrics
status = bot.get_status()
print(f"Data points: {status['data_feed_status']['data_points']}")
print(f"Cash: {status['broker_status']['cash']}")
```

## Extending the Bot

### Adding New Data Sources

1. Create new data feed class
2. Inherit from `BaseLiveDataFeed`
3. Add to `DataFeedFactory`
4. Update configuration validation

### Adding New Strategies

1. Create new strategy class
2. Inherit from `bt.Strategy`
3. Add entry/exit mixins
4. Update configuration validation

### Adding New Brokers

1. Create new broker class
2. Inherit from base broker
3. Add to `broker_factory.py`
4. Update configuration validation

## Best Practices

### Configuration Management
- Use descriptive configuration names
- Version control configuration files
- Test configurations before deployment
- Document parameter meanings

### Risk Management
- Start with paper trading
- Use small position sizes
- Set appropriate stop losses
- Monitor drawdown limits

### Monitoring
- Set up notifications for all events
- Monitor logs regularly
- Track performance metrics
- Set up alerts for errors

### Testing
- Test with historical data first
- Use paper trading for live testing
- Validate all components
- Monitor for edge cases

## Support

For issues and questions:
- Check the troubleshooting section
- Review error logs
- Test with different configurations
- Consult the API documentation 