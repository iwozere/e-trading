# Test Bot Configurations

This directory contains JSON configurations for test trading bots that use mock brokers and file data feeds. These configurations are designed for testing, development, and demonstration purposes.

## Available Configurations

### 1. `mock_file_test_btc_5m.json`
- **Symbol**: BTCUSDT
- **Data**: 5-minute CSV data from `data/BTCUSDT_5m_20220101_20250707.csv`
- **Strategy**: RSI + Bollinger Bands + Volume entry, ATR exit
- **Simulation**: Realistic paper trading with slippage and latency
- **Balance**: $10,000
- **Features**: Real-time simulation, notifications enabled

### 2. `mock_file_test_eth_15m.json`
- **Symbol**: ETHUSDT  
- **Data**: 15-minute CSV data from `data/ETHUSDT_15m_20220101_20250707.csv`
- **Strategy**: MACD entry, Fixed percentage exit
- **Simulation**: Realistic paper trading
- **Balance**: $5,000
- **Features**: Real-time simulation, weekly performance summaries

### 3. `mock_file_test_ltc_simple.json`
- **Symbol**: LTCUSDT
- **Data**: 15-minute CSV data from `data/LTCUSDT_15m_20220101_20250707.csv`
- **Strategy**: Simple SMA crossover
- **Simulation**: Basic paper trading (no slippage/latency)
- **Balance**: $2,000
- **Features**: Static backtesting mode, minimal logging

## Configuration Structure

Each configuration follows this structure:

```json
{
  "id": "unique_bot_identifier",
  "name": "Human-readable bot name",
  "enabled": true,
  "symbol": "TRADING_SYMBOL",
  "broker": {
    "type": "mock",
    "trading_mode": "paper",
    "cash": 10000.0,
    "paper_trading_config": {
      "mode": "realistic|basic|advanced",
      "commission_rate": 0.001,
      "slippage_model": "linear|sqrt|fixed",
      "latency_simulation": true,
      "realistic_fills": true
    }
  },
  "strategy": {
    "type": "CustomStrategy",
    "parameters": {
      "entry_logic": { "name": "...", "params": {...} },
      "exit_logic": { "name": "...", "params": {...} }
    }
  },
  "data": {
    "data_source": "file",
    "file_path": "data/SYMBOL_INTERVAL_DATES.csv",
    "interval": "5m|15m|1h|1d",
    "simulate_realtime": true|false,
    "fromdate": "2024-01-01",
    "todate": "2024-12-31"
  },
  "risk_management": {
    "max_position_size": 1000.0,
    "stop_loss_pct": 3.0,
    "take_profit_pct": 6.0,
    "max_daily_loss": 200.0
  },
  "notifications": {
    "position_opened": true,
    "position_closed": true,
    "telegram_enabled": true
  }
}
```

## Key Features

### Mock Broker
- **No real money**: All trades are simulated
- **Realistic execution**: Configurable slippage, latency, and market impact
- **Order types**: Market, limit, stop orders
- **Commission simulation**: Realistic trading costs
- **Partial fills**: Configurable probability of partial order execution

### File Data Feed
- **CSV data source**: Uses historical data from CSV files
- **Real-time simulation**: Can simulate real-time data delivery
- **Date filtering**: Specify date ranges for backtesting
- **Multiple timeframes**: Support for various intervals (5m, 15m, 1h, etc.)
- **Data validation**: Automatic OHLCV data validation

### Paper Trading Modes

#### Basic Mode
- Simple order execution at market prices
- Fixed commission rates
- No slippage or latency simulation
- Perfect fills

#### Realistic Mode (Recommended)
- Linear slippage model based on order size
- Latency simulation (10-100ms)
- Market impact modeling
- Partial fill probability
- Order rejection probability
- Execution quality metrics

#### Advanced Mode
- Complex market impact models
- Volume-based slippage
- Time-of-day effects
- Advanced order routing simulation

## Usage

### 1. Database Integration
Insert configurations into the `trading_bots` table:

```sql
INSERT INTO trading_bots (user_id, type, status, config, description)
VALUES (
  1, 
  'paper', 
  'stopped', 
  '{"id": "mock_file_test_btc_5m", ...}',
  'Mock file test bot for BTC 5m data'
);
```

### 2. Enhanced Trading Service
Use with the enhanced trading service:

```python
from src.data.db.services import trading_service

# Get bot configuration from database
bot_config = trading_service.get_bot_by_id(1)

# Validate configuration
is_valid, errors, warnings = trading_service.validate_bot_configuration(1)

if is_valid:
    # Create and run bot
    # (Implementation depends on enhanced trading runner)
    pass
```

### 3. Direct Usage
Use configurations directly:

```python
import json
from src.trading.broker.broker_factory import get_broker
from src.data.feed.file_data_feed import FileDataFeed

# Load configuration
with open('config/test_bot_configurations/mock_file_test_btc_5m.json') as f:
    config = json.load(f)

# Create broker
broker = get_broker(config['broker'])

# Create data feed
data_feed = FileDataFeed(
    dataname=config['data']['file_path'],
    symbol=config['symbol'],
    simulate_realtime=config['data']['simulate_realtime']
)
```

## Testing and Validation

### Configuration Validation
```python
from src.trading.services.bot_config_validator import validate_database_bot_record

# Validate configuration
is_valid, errors, warnings = validate_database_bot_record(bot_record)
```

### Demo Script
Run the demo script to test all configurations:

```bash
python src/trading/docs/demo_mock_file_trading.py
```

## Data Requirements

### CSV File Format
CSV files should have these columns:
- `datetime`: Timestamp (ISO format or parseable date string)
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price  
- `close`: Closing price
- `volume`: Trading volume

### Available Data Files
- `data/BTCUSDT_5m_20220101_20250707.csv` - Bitcoin 5-minute data
- `data/BTCUSDT_15m_20220101_20250707.csv` - Bitcoin 15-minute data
- `data/ETHUSDT_5m_20220101_20250707.csv` - Ethereum 5-minute data
- `data/ETHUSDT_15m_20220101_20250707.csv` - Ethereum 15-minute data
- `data/LTCUSDT_15m_20220101_20250707.csv` - Litecoin 15-minute data

## Customization

### Creating New Configurations
1. Copy an existing configuration file
2. Modify the parameters:
   - Change `id` and `name`
   - Update `symbol` and `file_path`
   - Adjust strategy parameters
   - Set appropriate risk management rules
3. Validate the configuration
4. Test with the demo script

### Strategy Parameters
Each strategy type has specific parameters:

#### RSI + Bollinger Bands Entry
```json
"entry_logic": {
  "name": "RSIBBVolumeEntryMixin",
  "params": {
    "e_rsi_period": 14,
    "e_rsi_oversold": 30,
    "e_bb_period": 20,
    "e_bb_dev": 2.0
  }
}
```

#### MACD Entry
```json
"entry_logic": {
  "name": "MACDEntryMixin", 
  "params": {
    "e_macd_fast": 12,
    "e_macd_slow": 26,
    "e_macd_signal": 9
  }
}
```

#### ATR Exit
```json
"exit_logic": {
  "name": "ATRExitMixin",
  "params": {
    "x_atr_period": 14,
    "x_sl_multiplier": 1.5
  }
}
```

## Best Practices

1. **Start Simple**: Use basic paper trading mode for initial testing
2. **Validate Data**: Always check CSV data quality before running
3. **Test Configurations**: Use the demo script to validate setups
4. **Monitor Performance**: Enable notifications and performance tracking
5. **Risk Management**: Always set appropriate risk limits
6. **Gradual Complexity**: Move from basic to realistic to advanced simulation modes

## Troubleshooting

### Common Issues

1. **File Not Found**: Check that CSV files exist in the `data/` directory
2. **Invalid Configuration**: Use the validator to check configuration syntax
3. **Date Range Issues**: Ensure `fromdate`/`todate` are within the CSV data range
4. **Strategy Parameters**: Verify that strategy mixin names and parameters are correct
5. **Memory Issues**: Large CSV files may require date filtering

### Debug Mode
Enable debug logging in configurations:
```json
"testing": {
  "log_level": "DEBUG",
  "save_trades": true
}
```

## Integration with Enhanced Trading Service

These configurations are designed to work with the enhanced database-driven trading service:

1. **Database Storage**: Configurations are stored in `trading_bots.config` JSONB field
2. **Validation**: Automatic validation using `bot_config_validator`
3. **Multi-Bot Management**: Run multiple test bots simultaneously
4. **Performance Tracking**: Integrated performance metrics and reporting
5. **Notifications**: Real-time trade and error notifications

This provides a complete testing environment that behaves like real trading but uses historical data and simulated execution.