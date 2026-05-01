# Trading Engine Module

## Purpose & Responsibilities

The Trading Engine module serves as the core execution and strategy management system for the Advanced Trading Framework. It orchestrates trade execution, strategy implementation, risk management, and broker integration to enable both paper and live trading operations.

## 🔗 Quick Navigation
- **[📖 Documentation Index](../INDEX.md)** - Complete documentation guide
- **[🏗️ System Architecture](../README.md)** - Overall system overview
- **[📊 Data Management](data-management.md)** - Market data and feeds
- **[🧠 ML & Analytics](ml-analytics.md)** - Performance analytics and ML integration
- **[🤖 Communication](communication.md)** - Notifications and user interfaces
- **[⚙️ Configuration](configuration.md)** - Strategy and bot configuration

## 🔄 Related Modules
| Module | Relationship | Integration Points |
|--------|--------------|-------------------|
| **[Data Management](data-management.md)** | Data Provider | Market data feeds, historical data for backtesting |
| **[ML & Analytics](ml-analytics.md)** | Analytics Consumer | Trade results, performance metrics, ML signals |
| **[Communication](communication.md)** | Notification Target | Trade alerts, bot status, performance reports |
| **[Infrastructure](infrastructure.md)** | Service Provider | Database persistence, job scheduling, error handling |
| **[Configuration](configuration.md)** | Configuration Source | Strategy parameters, risk settings, broker configuration |

**Core Responsibilities:**
- **Strategy Framework**: Modular strategy system with entry/exit mixins for flexible trading logic
- **Trade Execution**: Comprehensive trade lifecycle management from signal generation to position closure
- **Broker Integration**: Unified interface supporting multiple brokers (Binance, IBKR) with seamless paper-to-live switching
- **Risk Management**: Multi-layered risk controls including position sizing, stop losses, and exposure limits
- **Position Management**: Real-time position tracking with support for partial exits and complex order types
- **Performance Analytics**: Comprehensive trade tracking, P&L calculation, and performance metrics
- **Notification System**: Real-time alerts for trade events via Telegram and email

## Key Components

### 1. BaseTradingBot (Core Engine)

The `BaseTradingBot` serves as the foundation for all trading bot implementations, providing essential infrastructure for trade execution and management.

```python
from src.trading.base_trading_bot import BaseTradingBot

# Initialize trading bot
bot = BaseTradingBot(
    config=config,
    strategy_class=CustomStrategy,
    parameters=strategy_params,
    broker=broker,
    paper_trading=True
)

# Start trading
await bot.start_trading()
```

**Key Features:**
- **Signal Processing**: Converts strategy signals into executable trades
- **Position Management**: Tracks active positions and manages trade lifecycle
- **Balance Management**: Monitors account balance and available capital
- **Trade History**: Maintains comprehensive trade records with database integration
- **Notification Integration**: Real-time trade notifications via multiple channels
- **Risk Controls**: Implements position sizing and risk management rules

### 2. Strategy Framework (Modular Architecture)

The strategy framework implements a modular mixin-based architecture that separates entry logic, exit logic, and core strategy management.

#### BaseStrategy (Foundation)

```python
from src.strategy.base_strategy import BaseStrategy

class CustomStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        # Strategy initialization
        
    def next(self):
        # Strategy execution logic
        if self.should_enter():
            self.enter_position()
        elif self.should_exit():
            self.exit_position()
```

**Core Capabilities:**
- **Trade Tracking**: Complete trade lifecycle management with partial exit support
- **Position Sizing**: Intelligent position sizing with asset type validation
- **Performance Monitoring**: Real-time P&L, win rate, and drawdown tracking
- **Database Integration**: Persistent storage of all trade data
- **Asset Type Detection**: Automatic detection of crypto vs stock symbols
- **Risk Management**: Built-in position and exposure controls

#### Entry Mixins (Signal Generation)

Entry mixins provide modular, reusable entry logic that can be combined and configured:

**RSIOrBBEntryMixin** - RSI and Bollinger Bands Entry Logic:
```python
# Configuration
{
    "e_rsi_period": 14,
    "e_rsi_oversold": 30,
    "e_bb_period": 20,
    "e_bb_dev": 2.0,
    "e_cooldown_bars": 5
}

# Entry Logic
def should_enter(self) -> bool:
    rsi_condition = self.rsi[0] < self.e_rsi_oversold
    bb_condition = self.data.close[0] <= self.bb.lines.bot[0]
    cooldown_ok = self._check_cooldown()
    return (rsi_condition or bb_condition) and cooldown_ok
```

**Available Entry Mixins:**
- **RSIOrBBEntryMixin**: RSI oversold + Bollinger Band lower touch
- **RSIBBVolumeEntryMixin**: Volume-confirmed RSI/BB signals
- **HMMLSTMEntryMixin**: Machine learning-based entry signals
- **RSIIchimokuEntryMixin**: RSI + Ichimoku cloud confirmation
- **RSIVolumeSuperTrendEntryMixin**: Multi-indicator confirmation system

#### Exit Mixins (Position Management)

Exit mixins handle position closure logic with sophisticated risk management:

**AdvancedATRExitMixin** - Advanced ATR Trailing Stop:
```python
# Configuration
{
    "x_atr_period": 14,
    "x_sl_multiplier": 2.0,
    "x_tp_multiplier": 3.0,
    "x_trailing_enabled": true,
    "x_breakeven_enabled": true
}

# Exit Logic
def should_exit(self) -> bool:
    # Update trailing stop
    self._update_trailing_stop()
    
    # Check exit conditions
    return (self.data.close[0] <= self.stop_loss or 
            self.data.close[0] >= self.take_profit)
```

**Available Exit Mixins:**
- **AdvancedATRExitMixin**: Advanced ATR-based trailing stops with breakeven
- **ATRExitMixin**: Basic ATR trailing stop loss
- **TrailingStopExitMixin**: Simple percentage-based trailing stop
- **TimeBasedExitMixin**: Time-based position closure
- **FixedRatioExitMixin**: Fixed risk/reward ratio exits
- **MACrossoverExitMixin**: Moving average crossover exits

### 3. Broker System (Execution Layer)

The broker system provides unified access to multiple trading venues with seamless paper-to-live trading capabilities.

#### BaseBroker (Unified Interface)

```python
from src.trading.broker.base_broker import BaseBroker

# Broker supports both paper and live trading
broker = BrokerFactory.create_broker({
    "type": "binance",
    "trading_mode": "paper",  # or "live"
    "cash": 10000.0,
    "paper_trading_config": {
        "mode": "realistic",
        "slippage_model": "linear",
        "market_impact_enabled": True
    }
})
```

**Key Features:**
- **Unified Interface**: Consistent API across all broker implementations
- **Paper Trading**: Realistic simulation with slippage, latency, and market impact
- **Live Trading**: Direct integration with real broker APIs
- **Order Management**: Support for market, limit, stop, and OCO orders
- **Position Tracking**: Real-time position and balance monitoring
- **Risk Controls**: Built-in risk management and validation

#### Supported Brokers

**Binance Broker** - Cryptocurrency Trading:
```python
# Automatic credential selection based on mode
binance_config = {
    "type": "binance",
    "trading_mode": "paper",  # Uses testnet credentials
    "commission_rate": 0.001,
    "paper_trading_config": {
        "realistic_fills": True,
        "market_impact_enabled": True
    }
}
```

**Features:**
- **Dual Account Support**: Separate testnet/live credentials
- **WebSocket Integration**: Real-time order and position updates
- **Advanced Orders**: OCO, stop-loss, take-profit support
- **Market Data**: Integrated real-time and historical data
- **Rate Limiting**: Built-in API rate limit management

**IBKR Broker** - Multi-Asset Trading:
```python
# Automatic port selection based on mode
ibkr_config = {
    "type": "ibkr",
    "trading_mode": "paper",  # Uses paper trading port
    "host": "127.0.0.1",
    "commission_per_share": 0.005,
    "paper_trading_config": {
        "mode": "advanced",
        "partial_fill_probability": 0.1
    }
}
```

**Features:**
- **Multi-Asset Support**: Stocks, options, futures, forex
- **Dual Port Support**: Separate paper/live trading ports
- **Professional Features**: Advanced order types and risk management
- **Market Data**: Real-time Level II data and market depth
- **Portfolio Management**: Multi-currency and multi-asset portfolios

#### Paper Trading Simulation

The paper trading system provides realistic simulation with configurable parameters:

```python
paper_config = {
    "mode": "realistic",           # basic, realistic, advanced
    "slippage_model": "linear",    # linear, sqrt, fixed
    "base_slippage": 0.0005,       # 5 basis points
    "market_impact_enabled": True,
    "market_impact_factor": 0.0001,
    "latency_simulation": True,
    "min_latency_ms": 10,
    "max_latency_ms": 100,
    "partial_fill_probability": 0.1,
    "reject_probability": 0.01
}
```

**Simulation Features:**
- **Realistic Slippage**: Multiple slippage models (linear, square root, fixed)
- **Market Impact**: Order size-based market impact simulation
- **Execution Latency**: Configurable execution delays (10-200ms)
- **Partial Fills**: Realistic partial order execution
- **Order Rejections**: Configurable rejection probability
- **Execution Quality**: Comprehensive execution analytics

### 4. Risk Management System

The risk management system provides multi-layered protection with configurable controls:

```python
from src.trading.risk.controller import RiskController

risk_config = {
    "max_position_size": 0.2,      # 20% of portfolio per position
    "max_daily_loss": 0.05,        # 5% daily loss limit
    "max_drawdown": 0.15,          # 15% maximum drawdown
    "position_concentration": 0.3,  # 30% sector concentration limit
    "leverage_limit": 2.0,         # 2x maximum leverage
    "stop_loss_required": True,    # Require stop loss on all positions
    "risk_per_trade": 0.02         # 2% risk per trade
}

risk_controller = RiskController(risk_config)
```

**Risk Controls:**
- **Position Sizing**: Dynamic position sizing based on volatility and risk
- **Exposure Limits**: Maximum position size and concentration limits
- **Drawdown Protection**: Automatic trading halt on excessive losses
- **Stop Loss Enforcement**: Mandatory stop losses on all positions
- **Leverage Controls**: Maximum leverage and margin requirements
- **Correlation Limits**: Prevent over-concentration in correlated assets

### 5. Performance Analytics System

Comprehensive performance tracking and analytics for strategy optimization:

```python
# Real-time performance metrics
performance = {
    "total_trades": 150,
    "winning_trades": 95,
    "losing_trades": 55,
    "win_rate": 0.633,
    "total_pnl": 2450.75,
    "max_drawdown": -8.2,
    "sharpe_ratio": 1.85,
    "profit_factor": 1.42,
    "avg_win": 45.30,
    "avg_loss": -28.15,
    "largest_win": 125.50,
    "largest_loss": -85.20
}
```

**Analytics Features:**
- **Trade Statistics**: Win rate, profit factor, average win/loss
- **Risk Metrics**: Maximum drawdown, Sharpe ratio, Sortino ratio
- **Execution Quality**: Slippage analysis, fill rates, execution scores
- **Performance Attribution**: Strategy component performance analysis
- **Benchmark Comparison**: Performance vs market benchmarks
- **Real-time Monitoring**: Live performance dashboard and alerts

### 6. Notification System

Real-time notifications for trade events and system status:

```python
notification_config = {
    "position_opened": True,
    "position_closed": True,
    "email_enabled": True,
    "telegram_enabled": True,
    "error_notifications": True,
    "performance_reports": True
}
```

**Notification Types:**
- **Position Events**: Trade entry/exit notifications with P&L details
- **Risk Alerts**: Drawdown warnings, exposure limit breaches
- **System Status**: Bot start/stop, connection issues, errors
- **Performance Reports**: Daily/weekly performance summaries
- **Market Events**: Significant price movements, volatility spikes

## Architecture Patterns

### 1. Strategy Pattern (Entry/Exit Mixins)
The mixin system implements the strategy pattern, allowing dynamic composition of entry and exit logic without inheritance complexity.

### 2. Factory Pattern (Broker Creation)
The broker factory creates appropriate broker instances based on configuration, handling credential selection and mode switching automatically.

### 3. Observer Pattern (Notifications)
The notification system uses the observer pattern to decouple trade events from notification delivery, supporting multiple channels.

### 4. State Machine (Trade Lifecycle)
Trade management implements a state machine pattern to handle complex trade states including partial exits and order modifications.

### 5. Template Method (Base Strategy)
The BaseStrategy uses the template method pattern to define the strategy execution framework while allowing customization of specific steps.

## Integration Points

### With Data Management
- **Historical Data**: Retrieves OHLCV data for strategy backtesting and analysis
- **Real-time Feeds**: Consumes live market data for trading signal generation
- **Fundamentals**: Incorporates fundamental data for strategy decisions

### With ML & Analytics
- **Feature Engineering**: Provides market data for ML model training
- **Signal Generation**: Integrates ML-based entry/exit signals
- **Performance Analysis**: Supplies trade data for strategy optimization

### With Notification System
- **Trade Alerts**: Sends real-time trade notifications
- **Performance Reports**: Delivers periodic performance summaries
- **Error Reporting**: Notifies of system errors and issues

### With Database System
- **Trade Persistence**: Stores all trade data for analysis
- **Performance Tracking**: Maintains historical performance records
- **Bot Management**: Tracks multiple bot instances and configurations

## Data Models

### Trade Data Model
```python
{
    "trade_id": "uuid",
    "bot_instance_id": "uuid",
    "symbol": "BTCUSDT",
    "side": "buy",
    "entry_price": 45000.0,
    "exit_price": 46500.0,
    "quantity": 0.1,
    "entry_time": "2025-01-15T10:30:00Z",
    "exit_time": "2025-01-15T12:45:00Z",
    "pnl": 150.0,
    "pnl_percentage": 3.33,
    "exit_reason": "take_profit",
    "strategy_name": "RSI_BB_ATR",
    "trade_type": "paper"
}
```

### Position Data Model
```python
{
    "position_id": "uuid",
    "symbol": "AAPL",
    "side": "long",
    "quantity": 100,
    "avg_price": 150.25,
    "current_price": 152.80,
    "unrealized_pnl": 255.0,
    "realized_pnl": 0.0,
    "stop_loss": 147.50,
    "take_profit": 155.00,
    "entry_time": "2025-01-15T09:30:00Z",
    "last_update": "2025-01-15T15:45:00Z"
}
```

### Strategy Configuration Model
```python
{
    "strategy_name": "CustomStrategy",
    "entry_mixins": ["RSIOrBBEntryMixin"],
    "exit_mixins": ["AdvancedATRExitMixin"],
    "parameters": {
        "e_rsi_period": 14,
        "e_rsi_oversold": 30,
        "x_atr_period": 14,
        "x_sl_multiplier": 2.0
    },
    "position_size": 0.1,
    "risk_management": {
        "max_position_size": 0.2,
        "stop_loss_required": True
    }
}
```

## Roadmap & Feature Status

### ✅ Implemented Features (Q3-Q4 2024)
- **Modular Strategy Framework**: Complete mixin-based architecture with 10+ entry/exit mixins
- **Multi-Broker Support**: Binance and IBKR brokers with paper/live trading
- **Comprehensive Risk Management**: Position sizing, stop losses, exposure limits
- **Advanced Paper Trading**: Realistic simulation with slippage, latency, market impact
- **Performance Analytics**: Real-time trade tracking and performance metrics
- **Notification System**: Email and Telegram notifications for trade events
- **Database Integration**: Complete trade persistence and historical analysis
- **Order Management**: Market, limit, stop, and OCO order support

### 🔄 In Progress (Q1 2025)
- **Advanced Order Types**: Bracket orders, trailing stops, conditional orders (Target: Feb 2025)
- **Multi-Timeframe Strategies**: Strategies operating on multiple timeframes (Target: Mar 2025)
- **Portfolio Management**: Multi-asset portfolio optimization and rebalancing (Target: Mar 2025)
- **Machine Learning Integration**: Enhanced ML-based signal generation (Target: Feb 2025)

### 📋 Planned Enhancements

#### Q2 2025 - Advanced Trading Features
- **Options Trading**: Options strategies and risk management
  - Timeline: April-June 2025
  - Benefits: Expanded trading capabilities, advanced hedging strategies
  - Dependencies: New broker integrations, options data feeds
  - Complexity: High - requires new risk models and pricing engines

- **Algorithmic Execution**: TWAP, VWAP, and other execution algorithms
  - Timeline: May-July 2025
  - Benefits: Improved execution quality, reduced market impact
  - Dependencies: Enhanced broker APIs, real-time market data
  - Complexity: Medium - algorithmic execution logic

#### Q3 2025 - Multi-Exchange & Social Features
- **Cross-Exchange Arbitrage**: Multi-exchange trading opportunities
  - Timeline: July-September 2025
  - Benefits: Additional alpha generation, risk diversification
  - Dependencies: Multiple broker integrations, latency optimization
  - Complexity: High - requires sophisticated timing and risk management

- **Social Trading**: Copy trading and signal sharing capabilities
  - Timeline: August-October 2025
  - Benefits: Community-driven trading, strategy monetization
  - Dependencies: User management system, real-time communication
  - Complexity: Medium - social features and signal distribution

#### Q4 2025 - Advanced Analytics & AI
- **Advanced Analytics**: Factor analysis, attribution, and risk decomposition
  - Timeline: October-December 2025
  - Benefits: Deeper performance insights, risk understanding
  - Dependencies: ML & Analytics module enhancements
  - Complexity: Medium - statistical analysis and visualization

- **AI-Powered Strategy Generation**: Automated strategy discovery and optimization
  - Timeline: November 2025-Q1 2026
  - Benefits: Automated alpha discovery, reduced development time
  - Dependencies: Advanced ML infrastructure, large datasets
  - Complexity: Very High - requires sophisticated AI/ML capabilities

### Migration & Evolution Strategy

#### Phase 1: Enhanced Execution (Q1-Q2 2025)
- **Current State**: Basic order types with single-timeframe strategies
- **Target State**: Advanced order types with multi-timeframe capabilities
- **Migration Path**:
  - Implement advanced order types as optional features
  - Extend strategy framework to support multiple timeframes
  - Maintain backward compatibility with existing strategies
- **Backward Compatibility**: All existing strategies continue to work

#### Phase 2: Multi-Asset & Portfolio (Q2-Q3 2025)
- **Current State**: Single-asset trading with basic risk management
- **Target State**: Multi-asset portfolio management with advanced risk controls
- **Migration Path**:
  - Implement portfolio-level risk management alongside position-level controls
  - Add multi-asset strategy support as new strategy type
  - Gradual migration of risk controls to portfolio level
- **Backward Compatibility**: Single-asset strategies remain supported

#### Phase 3: Advanced Features (Q3-Q4 2025)
- **Current State**: Traditional trading with basic analytics
- **Target State**: Advanced trading with AI-powered features
- **Migration Path**:
  - Implement advanced features as optional modules
  - Provide migration tools for existing configurations
  - Maintain simple trading interface for basic users
- **Backward Compatibility**: Core trading functionality unchanged

### Version History & Updates

| Version | Release Date | Key Features | Breaking Changes |
|---------|--------------|--------------|------------------|
| **1.0.0** | Sep 2024 | Basic trading engine with strategy framework | N/A |
| **1.1.0** | Oct 2024 | Multi-broker support, paper trading | None |
| **1.2.0** | Nov 2024 | Advanced risk management, performance analytics | Strategy interface updates |
| **1.3.0** | Dec 2024 | Enhanced notifications, database integration | None |
| **1.4.0** | Jan 2025 | Advanced order types, ML integration | None (planned) |
| **2.0.0** | Q2 2025 | Portfolio management, options trading | Configuration changes (planned) |
| **3.0.0** | Q4 2025 | AI-powered features, social trading | API changes (planned) |

### Deprecation Timeline

#### Deprecated Features
- **Legacy Strategy Interface** (Deprecated: Nov 2024, Removed: May 2025)
  - Reason: Enhanced mixin architecture provides better flexibility
  - Migration: Automatic conversion tools provided
  - Impact: Minimal - most strategies auto-convert

#### Future Deprecations
- **Single-Asset Risk Management** (Deprecation: Q3 2025, Removal: Q1 2026)
  - Reason: Portfolio-level risk management is more comprehensive
  - Migration: Gradual migration to portfolio-based controls
  - Impact: Configuration updates required

- **Basic Paper Trading** (Deprecation: Q4 2025, Removal: Q2 2026)
  - Reason: Advanced simulation provides more realistic results
  - Migration: Automatic upgrade to advanced simulation
  - Impact: Minimal - improved accuracy

### Performance Targets & Benchmarks

#### Current Performance (Q4 2024)
- **Strategy Execution**: <1ms signal processing
- **Order Processing**: 10-100ms execution latency
- **Risk Checks**: <5ms validation time
- **Database Operations**: <50ms trade persistence

#### Target Performance (Q4 2025)
- **Strategy Execution**: <0.5ms signal processing (50% improvement)
- **Order Processing**: 5-50ms execution latency (50% improvement)
- **Portfolio Risk**: <10ms portfolio-level validation
- **AI Signal Generation**: <100ms ML-based signals

## Configuration

### Strategy Configuration
```yaml
# Strategy configuration
strategy:
  name: "CustomStrategy"
  entry_mixins:
    - name: "RSIOrBBEntryMixin"
      parameters:
        e_rsi_period: 14
        e_rsi_oversold: 30
        e_bb_period: 20
        e_bb_dev: 2.0
  exit_mixins:
    - name: "AdvancedATRExitMixin"
      parameters:
        x_atr_period: 14
        x_sl_multiplier: 2.0
        x_tp_multiplier: 3.0
```

### Broker Configuration
```yaml
# Broker configuration
broker:
  type: "binance"
  trading_mode: "paper"
  cash: 10000.0
  commission_rate: 0.001
  paper_trading_config:
    mode: "realistic"
    slippage_model: "linear"
    base_slippage: 0.0005
    market_impact_enabled: true
    latency_simulation: true
```

### Risk Management Configuration
```yaml
# Risk management
risk_management:
  max_position_size: 0.2
  max_daily_loss: 0.05
  max_drawdown: 0.15
  stop_loss_required: true
  risk_per_trade: 0.02
  position_concentration: 0.3
```

## Performance Characteristics

### Strategy Execution
- **Signal Generation**: Sub-millisecond indicator calculations
- **Order Processing**: 10-100ms order execution (depending on broker)
- **Risk Checks**: Real-time risk validation with microsecond latency
- **Database Operations**: Asynchronous trade persistence

### Broker Performance
- **Binance**: 1200 requests/minute, WebSocket real-time updates
- **IBKR**: Professional-grade execution with Level II data
- **Paper Trading**: Zero-latency simulation with realistic modeling

### Memory Usage
- **Strategy Framework**: Efficient indicator caching and data management
- **Trade Tracking**: Optimized data structures for large trade histories
- **Real-time Processing**: Stream processing with bounded memory usage

## Error Handling & Resilience

### Connection Management
- **Automatic Reconnection**: Seamless reconnection on network failures
- **Failover Support**: Backup connection endpoints and credentials
- **Circuit Breakers**: Temporary halt on repeated failures
- **Health Monitoring**: Continuous connection and API health checks

### Trade Execution Resilience
- **Order Validation**: Pre-trade validation and risk checks
- **Execution Monitoring**: Real-time order status tracking
- **Error Recovery**: Automatic retry with exponential backoff
- **Position Reconciliation**: Regular position and balance reconciliation

### Data Integrity
- **Trade Validation**: Comprehensive trade data validation
- **Database Transactions**: ACID compliance for trade persistence
- **Backup Systems**: Automated backup and recovery procedures
- **Audit Trails**: Complete audit trail for all trading activities

## Testing Strategy

### Unit Tests
- **Strategy Components**: Individual mixin and strategy logic testing
- **Broker Operations**: Order execution and position management tests
- **Risk Management**: Risk control validation and edge case testing
- **Performance Metrics**: Calculation accuracy and edge case handling

### Integration Tests
- **End-to-End Trading**: Complete trading workflow validation
- **Broker Integration**: Real broker API integration testing
- **Database Integration**: Trade persistence and retrieval testing
- **Notification System**: Multi-channel notification delivery testing

### Performance Tests
- **Strategy Backtesting**: Historical performance validation
- **Load Testing**: High-frequency trading simulation
- **Stress Testing**: System behavior under extreme conditions
- **Memory Profiling**: Memory usage optimization and leak detection

## Monitoring & Observability

### Real-time Metrics
- **Trade Metrics**: Real-time P&L, win rate, drawdown tracking
- **System Metrics**: CPU, memory, network usage monitoring
- **Broker Metrics**: API latency, error rates, connection status
- **Risk Metrics**: Position exposure, leverage, correlation monitoring

### Logging Strategy
- **Structured Logging**: JSON-formatted logs with consistent schema
- **Trade Logging**: Complete trade lifecycle logging
- **Error Logging**: Comprehensive error tracking and analysis
- **Performance Logging**: Execution timing and performance metrics

### Alerting System
- **Performance Alerts**: Drawdown warnings, performance degradation
- **System Alerts**: Connection failures, API errors, system issues
- **Risk Alerts**: Position limit breaches, exposure warnings
- **Trade Alerts**: Significant wins/losses, unusual trading patterns

## AdvancedATRExitMixin — Deep Dive

The `AdvancedATRExitMixin` is the primary production exit strategy. It implements a volatility-adaptive trailing stop with a full state machine, structural ratcheting, and time-based tightening.

### State Machine

| State | Description |
|-------|-------------|
| **INIT** | Initial stop placement at entry |
| **ARMED** | Break-even trigger reached |
| **PHASE1** | Running trail at standard `k_run` multiplier |
| **PHASE2** | Tighter trail after profit target (`phase2_at_R`) |
| **LOCKED** | Parabolic-style tightening near peak |
| **EXIT** | Position closed |

### Multi-Timeframe ATR

```python
# Three ATR layers are aggregated into one effective ATR
effective_atr = (alpha_fast * fast_atr) + (alpha_slow * slow_atr) + (alpha_htf * htf_atr)
```

- **Fast ATR** (`p_fast=7`): short-term noise filter
- **Slow ATR** (`p_slow=21`): trend-level volatility
- **HTF ATR** (`htf="4h"`): regime-level volatility (optional)

### Complete Parameter Reference

```json
{
  "anchor": "high",             // Price reference: high | close | mid
  "k_init": 2.5,                // Initial ATR multiplier (1.0–6.0)
  "k_run": 2.0,                 // Running trail multiplier (1.0–5.0)
  "k_phase2": 1.5,              // Phase 2 multiplier (0.8–3.0)
  "p_fast": 7,                  // Fast ATR period (5–21)
  "p_slow": 21,                 // Slow ATR period (14–100)
  "use_htf_atr": true,          // Enable higher-timeframe ATR
  "htf": "4h",                  // Higher timeframe
  "alpha_fast": 1.0,            // Fast ATR weight (0.25–2.0)
  "alpha_slow": 1.0,            // Slow ATR weight
  "alpha_htf": 1.0,             // HTF ATR weight
  "arm_at_R": 1.0,              // Break-even trigger in R multiples (0.5–2.0)
  "breakeven_offset_atr": 0.0,  // Extra cushion at break-even (ATR units)
  "phase2_at_R": 2.0,           // Phase 2 trigger (1.0–4.0)
  "use_swing_ratchet": true,    // Enable structural ratcheting
  "swing_lookback": 10,         // Swing detection lookback (5–30)
  "struct_buffer_atr": 0.25,    // Buffer beyond swing level (0.0–0.6)
  "tighten_if_stagnant_bars": 20, // Stagnation threshold (10–60)
  "tighten_k_factor": 0.8,      // Tightening factor (0.6–0.95)
  "min_bars_between_tighten": 5, // Tightening cooldown (1–10)
  "min_stop_step": 0.0,         // Minimum stop movement
  "noise_filter_atr": 0.0,      // Noise filter threshold (0.0–0.4)
  "pt_levels_R": [1.0, 2.0],    // Partial TP levels in R multiples
  "pt_sizes": [0.33, 0.33],     // Fraction of position to exit at each level
  "retune_after_pt": true       // Retune multipliers after partial TP
}
```

### Optimization Sequence

Optimize in three stages to avoid overfitting:

- **Stage A** — Core: `k_init`, `k_run`, `p_fast`, `p_slow`, `use_htf_atr`
- **Stage B** — Risk: `arm_at_R`, `phase2_at_R`, `k_phase2`, `use_swing_ratchet`, `struct_buffer_atr`
- **Stage C** — Tuning: `tighten_if_stagnant_bars`, `tighten_k_factor`, `noise_filter_atr`, alpha weights

### Market-Condition Guidance

| Condition | Recommendation |
|-----------|---------------|
| Strong trend | Higher `k_init` / `k_run` |
| Ranging market | Enable time-based tightening |
| High volatility (crypto) | Raise `atr_floor`, `noise_filter_atr`, enable HTF ATR |
| Intraday | Shorter ATR periods (5–14) |
| Daily / swing | Longer ATR periods (14–50) |

---

## Risk Management — Three-Stage Framework

Risk controls operate in three sequential stages around each trade.

### Module Layout

```
src/trading/risk/
├── pre_trade/
│   ├── position_sizing.py     # Kelly Criterion, Fixed Fractional
│   ├── exposure_limits.py     # Position and portfolio concentration limits
│   └── correlation_check.py  # Prevents over-exposure to correlated assets
├── real_time/
│   ├── stop_loss_manager.py  # Dynamic trailing stop adjustment
│   ├── drawdown_control.py   # Circuit breakers on drawdown thresholds
│   └── volatility_scaling.py # Vol-adjusted position sizing
└── post_trade/
    ├── pnl_attribution.py    # P&L by symbol / strategy
    ├── trade_analysis.py     # Win rate, expectancy, avg win/loss
    └── risk_reporting.py     # Risk dashboards and compliance reports
```

### Integration Pattern

```python
from src.trading.risk.pre_trade import position_sizing, exposure_limits, correlation_check
from src.trading.risk.real_time import stop_loss_manager, drawdown_control, volatility_scaling
from src.trading.risk.post_trade import pnl_attribution, trade_analysis, risk_reporting

# 1. Pre-trade: size the position
size = position_sizing.kelly_criterion(win_rate, avg_win, avg_loss, capital)
exposure_limits.check(symbol, size, portfolio)
correlation_check.check(symbol, portfolio)

# 2. Real-time: protect the position
stop_loss_manager.dynamic_stop_loss(position, atr)
drawdown_control.check_drawdown(portfolio)          # triggers circuit breaker
volatility_scaling.volatility_scaled_position(size, realized_vol)

# 3. Post-trade: analyse the result
pnl_attribution.pnl_attribution(trades, by="symbol")
trade_analysis.trade_quality_metrics(trades)
risk_reporting.generate_risk_report(portfolio)
```

---

## Live Trading Bot — Configuration Reference

The `LiveTradingBot` reads a single JSON manifest. Below is the canonical full-config template.

```json
{
    "description": "Strategy description",
    "version": "1.0",

    "broker": {
        "type": "binance_paper",   // binance_paper | binance | ibkr | mock
        "initial_balance": 1000.0,
        "commission": 0.001
    },

    "trading": {
        "symbol": "BTCUSDT",
        "position_size": 0.1,      // fraction of capital
        "max_positions": 1,
        "max_drawdown_pct": 20.0,
        "max_exposure": 1.0
    },

    "data": {
        "data_source": "binance",  // binance | yahoo | ibkr
        "symbol": "BTCUSDT",
        "interval": "1h",          // 1m 5m 15m 30m 1h 4h 1d
        "lookback_bars": 1000,
        "retry_interval": 60
    },

    "strategy": {
        "type": "custom",
        "entry_logic": {
            "name": "RSIBBVolumeEntryMixin",
            "params": { "e_rsi_period": 14, "e_rsi_oversold": 30, "e_bb_period": 20 }
        },
        "exit_logic": {
            "name": "AdvancedATRExitMixin",
            "params": { "k_init": 2.5, "k_run": 2.0, "use_swing_ratchet": true }
        },
        "position_size": 0.1
    },

    "risk_management": {
        "stop_loss_pct": 5.0,
        "take_profit_pct": 10.0,
        "trailing_stop": { "enabled": false, "activation_pct": 3.0, "trailing_pct": 2.0 },
        "max_daily_trades": 10,
        "max_daily_loss": 50.0
    },

    "notifications": {
        "enabled": true,
        "telegram": { "enabled": true, "notify_on": ["trade_entry", "trade_exit", "error", "status"] },
        "email":    { "enabled": false, "notify_on": ["trade_entry", "trade_exit", "error"] }
    },

    "logging": {
        "level": "INFO",
        "save_trades": true,
        "save_equity_curve": true,
        "log_file": "logs/live/trading_bot_0001.log"
    }
}
```

### CLI Usage

```bash
# Run bot
python src/trading/trading_bot.py config/trading/0001.json

# Validate config only
python src/trading/config_validator.py config/trading/0001.json

# Test data feed
python test_live_data_feeds.py binance BTCUSDT 1m
```

### Bot Lifecycle

1. Load and validate config → fail fast on errors
2. Setup notification client
3. Create broker, data feed, strategy instances
4. Load open positions from database (resume after restart)
5. Start Backtrader engine + real-time data loop
6. On bar close: evaluate signals → execute orders → persist trades
7. Monitor connection health; auto-reconnect every 60 s on failure
8. Send trade, error, and status notifications

### Extending the Bot

| Extension Point | Steps |
|-----------------|-------|
| New data source | Subclass `BaseLiveDataFeed` → register in `DataFeedFactory` |
| New strategy | Subclass `bt.Strategy` + entry/exit mixins → update config validation |
| New broker | Subclass `BaseBroker` → register in `BrokerFactory` |

---

## Bollinger Bands Implementation Reference

Three distinct BB implementations are used across the codebase. Use the correct attribute names for each.

### Quick Reference Table

| Implementation | Upper | Middle | Lower |
|----------------|-------|--------|-------|
| `UnifiedBollingerBandsIndicator` | `bb.upper[0]` | `bb.middle[0]` | `bb.lower[0]` |
| `bt.talib.BBANDS` | `bb.upperband[0]` | `bb.middleband[0]` | `bb.lowerband[0]` |
| `bt.indicators.BollingerBands` | `bb.lines.top[0]` | `bb.lines.mid[0]` | `bb.lines.bot[0]` |

### Decision Tree

```
Using UnifiedBollingerBandsIndicator?
├─ YES → bb.lower / bb.middle / bb.upper
└─ NO → Using bt.talib.BBANDS directly?
    ├─ YES → bb.lowerband / bb.middleband / bb.upperband
    └─ NO (standard BT) → bb.lines.bot / bb.lines.mid / bb.lines.top
```

### Best Practice — Prefer the Unified Wrapper

```python
from src.indicators.adapters.backtrader_wrappers import UnifiedBollingerBandsIndicator

bb = UnifiedBollingerBandsIndicator(
    self.data, period=20, devfactor=2.0,
    backend="bt-talib" if self.use_talib else "bt"
)
# Same attributes regardless of backend:
bb_lower  = bb.lower[0]
bb_middle = bb.middle[0]
bb_upper  = bb.upper[0]
```

Mixins using the unified wrapper: `RSIBBEntryMixin`, `RSIOrBBEntryMixin`, `RSIBBVolumeEntryMixin`, `RSIBBExitMixin`, `RSIOrBBExitMixin`.

---

**Module Version**: 1.4.0
**Last Updated**: 2026-04-30
**Owner**: Trading Team
**Dependencies**: [Data Management](data-management.md), [Infrastructure](infrastructure.md), [Configuration](configuration.md)
**Used By**: [ML & Analytics](ml-analytics.md), [Communication](communication.md)