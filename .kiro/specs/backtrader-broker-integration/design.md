# Design Document

## Overview

This design enables BaseBroker to work seamlessly within the backtrader framework while maintaining backward compatibility with existing implementations. The solution uses conditional inheritance and adapter patterns to provide backtrader compatibility when needed while preserving all existing functionality.

## Architecture

### High-Level Architecture

The design implements a flexible inheritance strategy where BaseBroker can inherit from either `bt.Broker` or `ABC` depending on the context and availability of backtrader. This is achieved through:

1. **Conditional Import and Inheritance**: Dynamically determine the base class at runtime
2. **Adapter Pattern**: Bridge between backtrader interfaces and existing broker methods
3. **Compatibility Layer**: Ensure data structures are compatible between both usage modes

### Component Design

#### 1. Dynamic Base Class Selection

```python
# Conditional import and base class selection
try:
    import backtrader as bt
    BACKTRADER_AVAILABLE = True
    BaseBrokerClass = bt.Broker
except ImportError:
    BACKTRADER_AVAILABLE = False
    from abc import ABC
    BaseBrokerClass = ABC
```

#### 2. Enhanced Base Broker with Flexible Inheritance

```python
class BaseBroker(BaseBrokerClass):
    def __init__(self, config: Dict[str, Any]):
        # Initialize based on base class type
        if BACKTRADER_AVAILABLE and isinstance(self, bt.Broker):
            super().__init__()  # Initialize bt.Broker
            self._backtrader_mode = True
        else:
            # ABC doesn't need super() call
            self._backtrader_mode = False
        
        # Continue with existing initialization
        self.config = config
        # ... rest of existing init code
```

#### 3. Backtrader Interface Adapter

A set of methods that bridge backtrader's expected interface with our existing broker methods:

```python
# Backtrader-specific methods (only when inheriting from bt.Broker)
def buy(self, owner, data, size, price=None, plimit=None, exectype=None, valid=None, tradeid=0, oco=None, trailamount=None, trailpercent=None, parent=None, transmit=True, **kwargs):
    """Backtrader buy method - adapts to our place_order"""
    
def sell(self, owner, data, size, price=None, plimit=None, exectype=None, valid=None, tradeid=0, oco=None, trailamount=None, trailpercent=None, parent=None, transmit=True, **kwargs):
    """Backtrader sell method - adapts to our place_order"""

def cancel(self, order):
    """Backtrader cancel method - adapts to our cancel_order"""

def get_notification(self):
    """Backtrader notification method"""

def next(self):
    """Backtrader next method for processing"""
```

#### 4. Data Structure Compatibility

Ensure that our internal data structures (Order, Position, Portfolio) are compatible with backtrader's expectations:

```python
class BacktraderOrderAdapter:
    """Adapter to make our Order compatible with backtrader"""
    
class BacktraderPositionAdapter:
    """Adapter to make our Position compatible with backtrader"""
```

## Data Flow

### Backtrader Integration Flow

1. **Initialization**: 
   - Check if backtrader is available
   - Select appropriate base class
   - Initialize broker with backtrader compatibility if needed

2. **Order Execution**:
   - Backtrader calls `buy()` or `sell()`
   - Adapter converts backtrader parameters to our Order format
   - Existing `place_order()` method handles execution
   - Results are converted back to backtrader format

3. **Position Management**:
   - Backtrader queries positions through standard interface
   - Our `get_positions()` method provides data
   - Adapter ensures backtrader-compatible format

4. **Notifications and Metrics**:
   - All existing notification and metrics functionality continues to work
   - Additional backtrader-specific notifications can be added

### Non-Backtrader Flow

When backtrader is not available or not needed:
- Inherits from ABC as before
- All existing functionality works unchanged
- No backtrader-specific methods are available

## Design Decisions

### 1. Conditional Inheritance vs. Composition

**Decision**: Use conditional inheritance rather than composition.

**Rationale**: 
- Backtrader expects brokers to inherit from `bt.Broker`
- Inheritance provides seamless integration with backtrader's internal mechanisms
- Conditional inheritance maintains backward compatibility

### 2. Runtime vs. Compile-time Base Class Selection

**Decision**: Use runtime base class selection with try/except import.

**Rationale**:
- Allows the same codebase to work with or without backtrader installed
- Graceful degradation when backtrader is not available
- No need for separate broker implementations

### 3. Adapter Pattern for Interface Compatibility

**Decision**: Use adapter methods to bridge backtrader interface with existing methods.

**Rationale**:
- Preserves existing broker API for non-backtrader usage
- Provides clean separation between backtrader-specific and general functionality
- Allows for easy maintenance and testing

### 4. Backward Compatibility Strategy

**Decision**: Maintain 100% backward compatibility with existing broker usage.

**Rationale**:
- Existing trading bots should continue to work without modification
- Reduces risk of breaking changes
- Allows gradual adoption of backtrader features

## Integration Patterns

### With Backtrader Strategies

```python
class MyBacktraderStrategy(bt.Strategy):
    def __init__(self):
        self.broker = BaseBroker(config)
        # Broker automatically works with backtrader
    
    def next(self):
        # Use standard backtrader broker interface
        order = self.buy(size=100)
        # Enhanced features (notifications, metrics) work automatically
```

### With Existing Trading Bots

```python
# Existing code continues to work unchanged
broker = BaseBroker(config)
order = await broker.place_order(order_obj)
positions = await broker.get_positions()
```

## Error Handling

### Import Error Handling

- Graceful handling when backtrader is not installed
- Clear error messages if backtrader features are requested but not available
- Fallback to ABC-based implementation

### Interface Compatibility

- Validation that backtrader methods receive compatible parameters
- Error handling for unsupported backtrader features
- Logging of backtrader-specific operations

## Testing Strategy

### Unit Tests

1. **Conditional Inheritance Tests**:
   - Test with backtrader available
   - Test with backtrader unavailable
   - Verify correct base class selection

2. **Interface Compatibility Tests**:
   - Test backtrader method calls
   - Verify parameter conversion
   - Test data structure compatibility

3. **Backward Compatibility Tests**:
   - Ensure existing broker functionality works
   - Test all existing methods and features
   - Verify configuration compatibility

### Integration Tests

1. **Backtrader Integration Tests**:
   - Test with actual backtrader strategies
   - Verify order execution flow
   - Test position and portfolio queries

2. **Paper Trading Tests**:
   - Test paper trading within backtrader
   - Verify execution metrics collection
   - Test notification system integration

### Performance Tests

1. **Overhead Assessment**:
   - Measure performance impact of conditional inheritance
   - Test adapter method overhead
   - Compare with direct backtrader broker usage