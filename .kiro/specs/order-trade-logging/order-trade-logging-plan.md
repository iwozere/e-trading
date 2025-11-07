# Order and Trade Logging Plan

## Executive Summary

This document outlines the comprehensive logging strategy for orders and trades in the e-trading platform. The plan establishes clear responsibilities between strategy and broker layers, defines log formats, and ensures full traceability of all trading activity.

## Current State Analysis

### Existing Logging Infrastructure

1. **Logger Module**: [`src/notification/logger.py`](../../src/notification/logger.py)
   - Centralized logging configuration
   - Multiple log files: `app.log`, `trades.log`, specialized bot logs
   - UTF-8 encoding support (critical for special characters)
   - Rotating file handlers (500MB per file, 99 backups)
   - Context-aware logging for module hierarchies

2. **Existing Trade Logging**:
   - **Location**: [`base_strategy.py:658`](../../src/strategy/base_strategy.py#L658) - `notify_trade()` method
   - **Current Behavior**: Logs trade notifications when trades close
   - **Log File**: Mixed into `app.log` (no dedicated strategy log)
   - **Format**: Basic INFO level messages
   ```python
   _logger.info(
       "Trade notification - Status: %s, Size: %.6f, PnL: %s, Price: %s",
       'CLOSED' if trade.isclosed else 'OPEN',
       actual_size, trade_pnl, trade_price
   )
   ```

3. **Existing Order Logging**:
   - **STATUS**: ❌ NOT IMPLEMENTED
   - No `notify_order()` method found in base_strategy.py
   - Orders are placed via `self.buy()` and `self.sell()` but not logged
   - Only basic entry/exit messages at position level

## Responsibility Architecture

### Strategy Layer Responsibilities
**File**: [`src/strategy/base_strategy.py`](../../src/strategy/base_strategy.py)

**What to Log**:
1. ✅ **Trade Lifecycle** (already exists, needs enhancement)
   - Trade opened/closed events
   - Trade P&L, duration, metrics
   - Partial vs full exits
   - Exit reasons (stop loss, take profit, signal, etc.)

2. ⚠️ **Order Execution** (MISSING - needs implementation)
   - Order submission (buy/sell)
   - Order acceptance/rejection
   - Order execution/cancellation
   - Order price, size, type

3. ✅ **Position Management** (partially exists)
   - Entry signals and reasons
   - Exit signals and reasons
   - Position sizing calculations

**Why Strategy Layer**:
- Has full context of trading logic and decision-making
- Knows WHY an order was placed (signal, indicator values, conditions)
- Can provide high-level semantic meaning to trades
- Has access to strategy-specific metadata (entry/exit reasons, mixin info)

### Broker Layer Responsibilities
**File**: Backtrader broker (external library)

**What It Already Logs**:
- Internal order state transitions
- Fill prices and execution details
- Commission calculations
- Margin requirements

**What We Don't Control**:
- Backtrader's internal logging is not customizable
- We receive notifications via `notify_order()` and `notify_trade()`
- We must log at the strategy level to capture our information

## Proposed Logging Architecture

### Log File Structure

```
logs/log/
├── app.log                    # General application logs
├── trades.log                 # Dedicated trade logs (existing)
├── orders.log                 # NEW: Dedicated order logs
└── strategies/                # NEW: Per-strategy logs
    ├── CustomStrategy_{timestamp}.log
    ├── HMMStrategy_{timestamp}.log
    └── ...
```

### Log Levels

- **DEBUG**: Indicator values, internal state, calculation details
- **INFO**: Order submissions, trade opens/closes, position changes
- **WARNING**: Invalid signals, rejected orders, unusual conditions
- **ERROR**: Exceptions, failures, critical issues

### Detailed Logging Specification

#### 1. Order Logging (NEW)

**Implementation Location**: `base_strategy.py` - new `notify_order()` method

**Log to**:
- `logs/log/orders.log` (dedicated orders file)
- `logs/log/strategies/{StrategyName}_{timestamp}.log` (per-strategy file)

**Format**:
```
2025-11-05 20:45:27,895 - INFO - ORDER - {symbol} | Type: {BUY/SELL} | Status: {Submitted/Accepted/Completed/Canceled/Rejected} | Size: {size} | Price: {price} | Order ID: {order.ref} | Reason: {reason}
```

**Events to Log**:

```python
def notify_order(self, order):
    """Log all order status changes"""

    # Order submitted
    if order.status in [order.Submitted]:
        logger.info(
            "ORDER - %s | Type: %s | Status: Submitted | Size: %.6f | Price: %.4f | Order ID: %d | Reason: %s",
            self.symbol,
            "BUY" if order.isbuy() else "SELL",
            order.size,
            order.price or self.data.close[0],
            order.ref,
            self.current_entry_reason or "unknown"
        )

    # Order accepted by broker
    elif order.status in [order.Accepted]:
        logger.info(
            "ORDER - %s | Type: %s | Status: Accepted | Order ID: %d",
            self.symbol,
            "BUY" if order.isbuy() else "SELL",
            order.ref
        )

    # Order completed (filled)
    elif order.status in [order.Completed]:
        logger.info(
            "ORDER - %s | Type: %s | Status: Completed | Size: %.6f | Executed Price: %.4f | Order ID: %d | Commission: %.4f",
            self.symbol,
            "BUY" if order.isbuy() else "SELL",
            order.executed.size,
            order.executed.price,
            order.ref,
            order.executed.comm or 0.0
        )

    # Order canceled
    elif order.status in [order.Canceled]:
        logger.warning(
            "ORDER - %s | Type: %s | Status: Canceled | Order ID: %d",
            self.symbol,
            "BUY" if order.isbuy() else "SELL",
            order.ref
        )

    # Order rejected
    elif order.status in [order.Rejected]:
        logger.error(
            "ORDER - %s | Type: %s | Status: Rejected | Order ID: %d | Reason: Insufficient margin or invalid parameters",
            self.symbol,
            "BUY" if order.isbuy() else "SELL",
            order.ref
        )
```

#### 2. Trade Logging (ENHANCED)

**Implementation Location**: `base_strategy.py:658` - enhance existing `notify_trade()` method

**Log to**:
- `logs/log/trades.log` (existing trades file)
- `logs/log/strategies/{StrategyName}_{timestamp}.log` (per-strategy file)

**Current Format** (needs enhancement):
```python
_logger.info(
    "Trade notification - Status: %s, Size: %.6f, PnL: %s, Price: %s",
    'CLOSED' if trade.isclosed else 'OPEN',
    actual_size, trade_pnl, trade_price
)
```

**Enhanced Format**:
```
2025-11-05 20:45:27,895 - INFO - TRADE - {symbol} | Status: {OPEN/CLOSED} | Direction: {LONG/SHORT} | Entry Price: {price} | Exit Price: {price} | Size: {size} | PnL: {pnl} | PnL%: {pnl_pct} | Duration: {duration} | Reason: {exit_reason} | Trade ID: {trade_ref}
```

**Enhanced Logging**:

```python
def notify_trade(self, trade):
    """Enhanced trade logging with full context"""

    actual_size = self._calculate_actual_trade_size(trade)

    if trade.isclosed:
        # Calculate metrics
        duration_bars = len(self.data) - trade.baropen
        pnl = trade.pnl or 0.0
        pnl_pct = (pnl / (trade.price * actual_size) * 100) if trade.price else 0.0

        # Get exit reason (set by _exit_position)
        exit_reason = self.current_exit_reason or "unknown"

        logger.info(
            "TRADE - %s | Status: CLOSED | Direction: %s | Entry: %.4f | Exit: %.4f | Size: %.6f | PnL: %.4f | PnL%%: %.2f%% | Duration: %d bars | Reason: %s | Trade ID: %d",
            self.symbol,
            "LONG" if trade.size > 0 else "SHORT",
            trade.price,  # entry price
            self.data.close[0],  # exit price
            actual_size,
            pnl,
            pnl_pct,
            duration_bars,
            exit_reason,
            trade.ref
        )

        # Also log detailed trade record to trades.log
        trade_logger = logging.getLogger('trades')
        trade_logger.info(
            "TRADE_RECORD | Symbol: %s | Entry_Time: %s | Exit_Time: %s | Entry_Price: %.4f | Exit_Price: %.4f | Size: %.6f | PnL: %.4f | Commission: %.4f | Exit_Reason: %s",
            self.symbol,
            self.data.num2date(trade.baropen),
            self.data.datetime.datetime(),
            trade.price,
            self.data.close[0],
            actual_size,
            pnl,
            trade.commission or 0.0,
            exit_reason
        )
    else:
        # Trade opened
        logger.info(
            "TRADE - %s | Status: OPEN | Direction: %s | Entry: %.4f | Size: %.6f | Trade ID: %d",
            self.symbol,
            "LONG" if trade.size > 0 else "SHORT",
            trade.price,
            actual_size,
            trade.ref
        )
```

#### 3. Position Entry/Exit Logging (ENHANCED)

**Implementation Location**: `base_strategy.py:346` (`_enter_position`) and `base_strategy.py:436` (`_exit_position`)

**Current State**: Basic INFO logging exists

**Enhancement**: Add more context

```python
def _enter_position(self, direction: str, confidence: float = 1.0,
                   risk_multiplier: float = 1.0, reason: str = ""):
    """Enhanced position entry logging"""

    position_size = self._calculate_position_size(confidence, risk_multiplier)
    shares = self._calculate_shares(position_size)

    if direction.lower() == 'long':
        order = self.buy(size=shares)

        # Enhanced logging with full context
        logger.info(
            "POSITION_ENTRY - %s | Direction: LONG | Size: %.6f shares (%.2f%% of capital) | Price: %.4f | Confidence: %.2f | Risk Multiplier: %.2f | Reason: %s | Entry Conditions: %s",
            self.symbol,
            shares,
            position_size * 100,
            self.data.close[0],
            confidence,
            risk_multiplier,
            reason,
            self._get_entry_conditions()  # New method to capture indicator states
        )

        # Store reason for notify_order
        self.current_entry_reason = reason

def _exit_position(self, reason: str = ""):
    """Enhanced position exit logging"""

    self.exit_size = abs(self.position.size)
    self.current_exit_reason = reason

    # Calculate current P&L
    current_pnl = (self.data.close[0] - self.entry_price) * self.exit_size
    current_pnl_pct = (current_pnl / (self.entry_price * self.exit_size)) * 100 if self.entry_price else 0

    logger.info(
        "POSITION_EXIT - %s | Size: %.6f | Exit Price: %.4f | Unrealized PnL: %.4f (%.2f%%) | Reason: %s | Exit Conditions: %s",
        self.symbol,
        self.exit_size,
        self.data.close[0],
        current_pnl,
        current_pnl_pct,
        reason,
        self._get_exit_conditions()  # New method to capture indicator states
    )

    self.close()
```

#### 4. Per-Strategy Logging (NEW)

**Purpose**: Isolate each strategy's logs for easier debugging and analysis

**Implementation**: New logger configuration in `logger.py`

```python
def setup_strategy_logger(strategy_name: str, instance_id: str = None) -> logging.Logger:
    """
    Create a dedicated logger for a strategy instance.

    Args:
        strategy_name: Name of the strategy class
        instance_id: Optional unique identifier for this strategy instance

    Returns:
        Logger that writes to strategy-specific file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instance_suffix = f"_{instance_id}" if instance_id else ""
    log_filename = f"{strategy_name}{instance_suffix}_{timestamp}.log"

    strategy_log_dir = PROJECT_ROOT / "logs" / "log" / "strategies"
    strategy_log_dir.mkdir(parents=True, exist_ok=True)

    log_file = strategy_log_dir / log_filename

    return setup_logger(f"strategy.{strategy_name}", str(log_file))
```

**Usage in BaseStrategy**:

```python
class BaseStrategy(bt.Strategy):
    def __init__(self):
        super().__init__()

        # Create strategy-specific logger
        strategy_name = self.__class__.__name__
        self.strategy_logger = setup_strategy_logger(
            strategy_name,
            instance_id=self.p.symbol  # Use symbol as instance ID
        )
```

### Log Output Examples

#### Order Log Example (`orders.log`):
```
2025-11-05 20:45:27,895 - INFO - ORDER - BTCUSDT | Type: BUY | Status: Submitted | Size: 0.050000 | Price: 46296.95 | Order ID: 1 | Reason: RSI oversold + BB touch
2025-11-05 20:45:27,896 - INFO - ORDER - BTCUSDT | Type: BUY | Status: Accepted | Order ID: 1
2025-11-05 20:45:27,897 - INFO - ORDER - BTCUSDT | Type: BUY | Status: Completed | Size: 0.050000 | Executed Price: 46296.95 | Order ID: 1 | Commission: 0.0463
2025-11-05 20:48:15,234 - INFO - ORDER - BTCUSDT | Type: SELL | Status: Submitted | Size: 0.050000 | Price: 47125.80 | Order ID: 2 | Reason: ATR take profit
2025-11-05 20:48:15,235 - INFO - ORDER - BTCUSDT | Type: SELL | Status: Accepted | Order ID: 2
2025-11-05 20:48:15,236 - INFO - ORDER - BTCUSDT | Type: SELL | Status: Completed | Size: 0.050000 | Executed Price: 47125.80 | Order ID: 2 | Commission: 0.0471
```

#### Trade Log Example (`trades.log`):
```
2025-11-05 20:45:27,897 - INFO - TRADE - BTCUSDT | Status: OPEN | Direction: LONG | Entry: 46296.95 | Size: 0.050000 | Trade ID: 1
2025-11-05 20:48:15,236 - INFO - TRADE - BTCUSDT | Status: CLOSED | Direction: LONG | Entry: 46296.95 | Exit: 47125.80 | Size: 0.050000 | PnL: 41.44 | PnL%: 1.79% | Duration: 168 bars | Reason: ATR take profit | Trade ID: 1
2025-11-05 20:48:15,237 - INFO - TRADE_RECORD | Symbol: BTCUSDT | Entry_Time: 2024-03-15 14:00:00 | Exit_Time: 2024-03-22 14:00:00 | Entry_Price: 46296.95 | Exit_Price: 47125.80 | Size: 0.050000 | PnL: 41.44 | Commission: 0.0934 | Exit_Reason: ATR take profit
```

#### Strategy Log Example (`strategies/CustomStrategy_BTCUSDT_20251105_204527.log`):
```
2025-11-05 20:45:27,890 - INFO - STRATEGY_INIT - CustomStrategy initialized for BTCUSDT | Entry: RSIBBEntryMixin | Exit: ATRExitMixin | Position Size: 0.1
2025-11-05 20:45:27,895 - INFO - POSITION_ENTRY - BTCUSDT | Direction: LONG | Size: 0.050000 shares (10.00% of capital) | Price: 46296.95 | Confidence: 1.00 | Risk Multiplier: 1.00 | Reason: Entry mixin signal | Entry Conditions: RSI=28.5, BB_Lower=45800.20, Price=46296.95
2025-11-05 20:48:15,234 - INFO - POSITION_EXIT - BTCUSDT | Size: 0.050000 | Exit Price: 47125.80 | Unrealized PnL: 41.44 (1.79%) | Reason: ATR take profit | Exit Conditions: ATR=1250.50, TP_Level=47100.00, SL_Level=45200.00
```

## Implementation Priority

### Phase 1: Order Logging (HIGH PRIORITY)
**Files to Modify**:
- `src/strategy/base_strategy.py` - Add `notify_order()` method
- `src/notification/logger.py` - Add orders.log handler

**Estimated Effort**: 2-3 hours

### Phase 2: Enhanced Trade Logging (MEDIUM PRIORITY)
**Files to Modify**:
- `src/strategy/base_strategy.py` - Enhance `notify_trade()` method
- Add structured trade records to trades.log

**Estimated Effort**: 1-2 hours

### Phase 3: Per-Strategy Logging (MEDIUM PRIORITY)
**Files to Modify**:
- `src/notification/logger.py` - Add `setup_strategy_logger()` function
- `src/strategy/base_strategy.py` - Initialize strategy logger

**Estimated Effort**: 2-3 hours

### Phase 4: Enhanced Position Logging (LOW PRIORITY)
**Files to Modify**:
- `src/strategy/base_strategy.py` - Enhance `_enter_position()` and `_exit_position()`
- Add `_get_entry_conditions()` and `_get_exit_conditions()` helper methods

**Estimated Effort**: 3-4 hours

## Testing Plan

1. **Unit Tests**: Test each logging method independently
2. **Integration Tests**: Run backtest with logging enabled
3. **Log Parsing Tests**: Verify log format is machine-readable
4. **Performance Tests**: Ensure logging doesn't impact backtest speed

## Benefits

1. **Full Traceability**: Every order and trade is logged with complete context
2. **Debugging**: Easy to identify why trades were entered/exited
3. **Analysis**: Machine-readable logs for post-backtest analysis
4. **Compliance**: Audit trail for all trading activity
5. **Optimization**: Understand strategy behavior across different market conditions

## Conclusion

**Answer to Your Question**:
> "Is it going to be the responsibility of the trading bot? Or strategy?"

**Answer**: **Strategy Layer** is responsible for logging orders and trades.

**Rationale**:
1. Strategy has full context of WHY decisions were made
2. Strategy knows entry/exit reasons, indicator values, conditions
3. Backtrader broker is external library - we get notifications via callbacks
4. Strategy layer is where we implement `notify_order()` and `notify_trade()`
5. All strategy-level events (position entry/exit, signals) are already logged there

The broker (Backtrader) handles execution internally, but we capture and log all events through the notification callbacks in the strategy layer.
