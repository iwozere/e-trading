# Trading Bot Design Documentation

## Architecture Overview

The trading bot system is built with a modular, component-based architecture that separates concerns and enables easy testing and maintenance.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Feeds    │    │   Strategies    │    │    Brokers      │
│                 │    │                 │    │                 │
│ • Binance API   │───▶│ • CustomStrategy│───▶│ • Paper Trading │
│ • Historical    │    │ • Entry Mixins  │    │ • Live Trading  │
│ • Real-time     │    │ • Exit Mixins   │    │ • Risk Mgmt     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │   Core Engine   │
                    │                 │
                    │ • Backtrader    │
                    │ • Event Loop    │
                    │ • Order Mgmt    │
                    │ • Position Mgmt │
                    └─────────────────┘
```

## Component Design

### 1. Strategy Architecture

#### CustomStrategy
The main strategy class that orchestrates entry and exit mixins:

```python
class CustomStrategy(BaseStrategy):
    def __init__(self, config):
        # Initialize entry and exit mixins
        self.entry_mixin = get_entry_mixin(config['entry_logic'])
        self.exit_mixin = get_exit_mixin(config['exit_logic'])
    
    def next(self):
        # Check entry conditions
        if self.entry_mixin.should_enter():
            self._enter_position()
        
        # Check exit conditions
        if self.exit_mixin.should_exit():
            self._exit_position()
```

#### Entry Mixin: RSIOrBBEntryMixin
Combines RSI and Bollinger Bands for entry signals:

**Parameters:**
- `e_rsi_period`: RSI calculation period (default: 14)
- `e_rsi_oversold`: RSI oversold threshold (default: 30)
- `e_bb_period`: Bollinger Bands period (default: 20)
- `e_bb_dev`: Bollinger Bands standard deviation (default: 2.0)
- `e_rsi_cross`: Use RSI cross-over logic (default: false)
- `e_bb_reentry`: Allow re-entry on BB touch (default: false)
- `e_cooldown_bars`: Cooldown period between entries (default: 5)

**Entry Logic:**
```python
def should_enter(self) -> bool:
    # RSI oversold condition
    rsi_condition = self.rsi[0] < self.e_rsi_oversold
    
    # Bollinger Band lower touch
    bb_condition = self.data.close[0] <= self.bb.lines.bot[0]
    
    # Cooldown check
    cooldown_ok = self._check_cooldown()
    
    return (rsi_condition or bb_condition) and cooldown_ok
```

#### Exit Mixin: ATRExitMixin
Uses Average True Range for trailing stop loss:

**Parameters:**
- `x_atr_period`: ATR calculation period (default: 14)
- `x_sl_multiplier`: Stop loss multiplier (default: 2.0)

**Exit Logic:**
```python
def should_exit(self) -> bool:
    # Track highest price since entry
    self.highest_price = max(self.highest_price, self.data.high[0])
    
    # Calculate trailing stop loss
    stop_loss = self.highest_price - (self.atr[0] * self.x_sl_multiplier)
    
    # Update trailing stop (only moves up)
    self.stop_loss = max(self.stop_loss, stop_loss)
    
    # Check exit condition
    return self.data.close[0] <= self.stop_loss
```

### 2. Broker Architecture

#### BinancePaperBroker
Paper trading implementation using Binance testnet:

```python
class BinancePaperBroker(BaseBinanceBroker):
    def __init__(self, api_key, api_secret, cash=10000.0):
        super().__init__(api_key, api_secret, cash, testnet=True)
        self.broker_name = "Binance Paper"
        self.client.API_URL = "https://testnet.binance.vision/api"
```

**Features:**
- Real-time price data from Binance testnet
- Simulated order execution
- Position tracking
- Balance management
- Commission calculation

### 3. Configuration Design

#### Trading Configuration Structure
```json
{
  "bot_id": "paper_trading_rsi_atr",
  "symbol": "LTCUSDT",
  "timeframe": "4h",
  "broker_type": "binance_paper",
  "strategy_type": "custom",
  "strategy_params": {
    "entry_logic": {
      "name": "RSIOrBBEntryMixin",
      "params": { /* entry parameters */ }
    },
    "exit_logic": {
      "name": "ATRExitMixin", 
      "params": { /* exit parameters */ }
    }
  },
  "risk_management": {
    "position_size": 0.1,
    "max_open_trades": 3,
    "stop_loss_pct": 5.0
  }
}
```

#### Configuration Validation
- Pydantic models for type safety
- Required field validation
- Parameter range checking
- Environment-specific overrides

### 4. Data Flow Design

#### Real-time Data Flow
```
Binance API → Data Feed → Strategy → Broker → Database
     ↓           ↓          ↓         ↓         ↓
  Price Data  Indicators  Signals  Orders   Trade Log
```

#### Event Processing
1. **Data Update**: New price bar arrives
2. **Indicator Update**: Calculate RSI, BB, ATR
3. **Signal Generation**: Entry/exit conditions checked
4. **Order Execution**: Buy/sell orders placed
5. **Position Update**: Track positions and P&L
6. **Logging**: Record trades and performance

### 5. Risk Management Design

#### Position Sizing
```python
def calculate_position_size(self, signal_strength: float) -> float:
    base_size = self.config['position_size']
    risk_adjusted = base_size * signal_strength
    return min(risk_adjusted, self.max_position_size)
```

#### Risk Controls
- **Maximum Open Trades**: Limit concurrent positions
- **Stop Loss**: ATR-based trailing stop
- **Daily Loss Limit**: Stop trading if daily loss exceeded
- **Drawdown Control**: Reduce position size during drawdowns

### 6. Database Design

#### Trade Storage
```sql
CREATE TABLE trades (
    id VARCHAR(36) PRIMARY KEY,
    bot_id VARCHAR(255) NOT NULL,
    trade_type VARCHAR(10) NOT NULL, -- 'paper', 'live', 'optimization'
    symbol VARCHAR(20) NOT NULL,
    entry_time DATETIME,
    exit_time DATETIME,
    entry_price DECIMAL(20,8),
    exit_price DECIMAL(20,8),
    size DECIMAL(20,8),
    direction VARCHAR(10), -- 'long', 'short'
    gross_pnl DECIMAL(20,8),
    net_pnl DECIMAL(20,8),
    exit_reason VARCHAR(50),
    status VARCHAR(20)
);
```

#### Performance Metrics
- Trade-by-trade P&L tracking
- Win/loss ratios
- Maximum drawdown
- Sharpe ratio
- Sortino ratio

### 7. Error Handling Design

#### Error Categories
1. **Network Errors**: API connection issues
2. **Data Errors**: Invalid price data
3. **Strategy Errors**: Calculation failures
4. **Broker Errors**: Order execution failures

#### Recovery Mechanisms
```python
def handle_error(self, error_type: str, error: Exception):
    if error_type == "network":
        self.retry_connection()
    elif error_type == "data":
        self.skip_invalid_bar()
    elif error_type == "strategy":
        self.reset_indicators()
    elif error_type == "broker":
        self.cancel_pending_orders()
```

### 8. Monitoring Design

#### Real-time Monitoring
- Trade execution logs
- Performance metrics
- Error tracking
- System health checks

#### Alerting System
- Trade notifications
- Error alerts
- Performance warnings
- System status updates

## Design Principles

### 1. Modularity
- Separate entry and exit logic
- Pluggable broker implementations
- Configurable risk management

### 2. Testability
- Paper trading for safe testing
- Unit tests for individual components
- Integration tests for full workflows

### 3. Scalability
- Support for multiple symbols
- Concurrent strategy execution
- Distributed deployment options

### 4. Maintainability
- Clear separation of concerns
- Comprehensive logging
- Configuration-driven behavior

### 5. Safety
- Paper trading by default
- Risk management controls
- Error recovery mechanisms

## Performance Considerations

### 1. Latency Optimization
- Efficient indicator calculations
- Minimal API calls
- Cached data when possible

### 2. Memory Management
- Limited historical data retention
- Efficient data structures
- Garbage collection optimization

### 3. Network Efficiency
- Batch API requests
- Connection pooling
- Retry mechanisms

## Security Considerations

### 1. API Security
- Secure key storage
- Rate limiting
- Request signing

### 2. Data Security
- Encrypted storage
- Access controls
- Audit logging

### 3. Operational Security
- Paper trading by default
- Manual approval for live trading
- Regular security audits
