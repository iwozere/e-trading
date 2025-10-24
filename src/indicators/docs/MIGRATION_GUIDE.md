# Unified Indicator Service Migration Guide

This guide provides comprehensive instructions for migrating from the legacy indicator services to the unified indicator service.

## Overview

The unified indicator service consolidates multiple fragmented indicator implementations into a single, cohesive system. This migration eliminates redundancy, improves maintainability, and provides a consistent API for all indicator-related operations.

## Key Changes

### 1. Simplified Interface
- **Before**: Multiple services with different interfaces
- **After**: Single unified service with consistent API
- **Impact**: Cleaner code, easier maintenance

### 2. Removed Backward Compatibility Parameters
- **Before**: Complex parameter structures with legacy support
- **After**: Streamlined parameters focused on core functionality
- **Impact**: Simpler configuration, reduced complexity

### 3. Direct Unified Service Integration
- **Before**: Wrapper-based approach with fallback mechanisms
- **After**: Direct integration with unified service
- **Impact**: Better performance, cleaner architecture

## Migration Steps

### Step 1: Update Strategy Entry/Exit Mixins

#### Before (Old Wrapper Approach)
```python
from src.strategy.indicator.wrappers import create_indicator_wrapper  # Legacy - removed

# In _init_indicators method
if self.strategy.use_talib:
    rsi_raw = bt.talib.RSI(self.strategy.data.close, timeperiod=rsi_period)
    bb_raw = bt.talib.BBANDS(
        self.strategy.data.close,
        timeperiod=bb_period,
        nbdevup=bb_dev_factor,
        nbdevdn=bb_dev_factor,
    )
else:
    rsi_raw = bt.indicators.RSI(self.strategy.data.close, period=rsi_period)
    bb_raw = bt.indicators.BollingerBands(
        self.strategy.data.close, period=bb_period, devfactor=bb_dev_factor
    )

# Create wrapped indicators
self.rsi = create_indicator_wrapper(rsi_raw, 'rsi', self.strategy.use_talib)
self.bb = create_indicator_wrapper(bb_raw, 'bb', self.strategy.use_talib)

# Access values
rsi_value = rsi[0]
bb_lower = bb.bot[0]  # or bb.lowerband[0] for TALib
```

#### After (Unified Service Approach)
```python
from src.indicators.adapters.backtrader_wrappers import UnifiedRSIIndicator, UnifiedBollingerBandsIndicator

# In _init_indicators method
backend = "bt-talib" if self.strategy.use_talib else "bt"

self.rsi = UnifiedRSIIndicator(
    self.strategy.data,
    period=rsi_period,
    backend=backend
)

self.bb = UnifiedBollingerBandsIndicator(
    self.strategy.data,
    period=bb_period,
    devfactor=bb_dev_factor,
    backend=backend
)

# Access values (consistent interface)
rsi_value = rsi.rsi[0]
bb_lower = bb.lower[0]  # Consistent naming
```

### Step 2: Update Indicator Factory Usage

#### Before
```python
from src.strategy.indicator.indicator_factory import IndicatorFactory

factory = IndicatorFactory()
rsi_indicator = factory.create_backtrader_rsi(
    data, period=14, backend="bt", use_unified_service=True
)
```

#### After
```python
from src.indicators.indicator_factory import IndicatorFactory

factory = IndicatorFactory()
rsi_indicator = factory.create_backtrader_rsi(
    data, period=14, backend="bt"
)
# Note: use_unified_service parameter removed - always uses unified service
```

### Step 3: Update Direct Indicator Imports

#### Before
```python
# Multiple different import paths
from src.common.indicator_service import IndicatorService  # Legacy - no longer exists
from src.strategy.indicator.rsi import RSIIndicator  # Legacy individual files - removed
```

#### After
```python
# Single unified import path
from src.indicators.service import UnifiedIndicatorService
from src.indicators.adapters.backtrader_wrappers import UnifiedRSIIndicator as RsiIndicator  # Unified implementation
```

## Parameter Changes

### Bollinger Bands
| Old Parameter | New Parameter | Notes |
|---------------|---------------|-------|
| `bb.bot[0]` or `bb.lowerband[0]` | `bb.lower[0]` | Consistent naming |
| `bb.mid[0]` or `bb.middleband[0]` | `bb.middle[0]` | Consistent naming |
| `bb.top[0]` or `bb.upperband[0]` | `bb.upper[0]` | Consistent naming |

### RSI
| Old Parameter | New Parameter | Notes |
|---------------|---------------|-------|
| `rsi[0]` | `rsi.rsi[0]` | Explicit line access |

### ATR
| Old Parameter | New Parameter | Notes |
|---------------|---------------|-------|
| `atr[0]` | `atr.atr[0]` | Explicit line access |

### MACD
| Old Parameter | New Parameter | Notes |
|---------------|---------------|-------|
| `macd.macd[0]` | `macd.macd[0]` | No change |
| `macd.signal[0]` | `macd.signal[0]` | No change |
| `macd.histogram[0]` | `macd.histogram[0]` | No change |

## Configuration Changes

### Removed Parameters
- `use_unified_service`: Always true, parameter removed
- `fallback_enabled`: No fallback mechanisms, parameter removed
- `legacy_compatibility`: No legacy support, parameter removed

### Simplified Backend Selection
- **Before**: Complex backend selection with fallbacks
- **After**: Simple backend parameter: `"bt"`, `"bt-talib"`, or `"talib"`

## Common Migration Issues

### Issue 1: AttributeError on Indicator Access
**Problem**: `AttributeError: 'UnifiedRSIIndicator' object has no attribute '__getitem__'`

**Solution**: Use explicit line access
```python
# Wrong
value = rsi[0]

# Correct
value = rsi.rsi[0]
```

### Issue 2: Bollinger Bands Naming
**Problem**: `AttributeError: 'UnifiedBollingerBandsIndicator' object has no attribute 'bot'`

**Solution**: Use consistent naming
```python
# Wrong
lower_band = bb.bot[0]

# Correct
lower_band = bb.lower[0]
```

### Issue 3: Backend Parameter
**Problem**: Invalid backend specification

**Solution**: Use correct backend values
```python
# Wrong
backend = "talib" if use_talib else "backtrader"

# Correct
backend = "bt-talib" if use_talib else "bt"
```

## Testing Your Migration

### 1. Unit Tests
Create unit tests to verify indicator calculations:

```python
import unittest
from src.indicators.adapters.backtrader_wrappers import UnifiedRSIIndicator

class TestUnifiedIndicators(unittest.TestCase):
    def test_rsi_calculation(self):
        # Test RSI calculation matches expected values
        pass
    
    def test_bollinger_bands_calculation(self):
        # Test Bollinger Bands calculation matches expected values
        pass
```

### 2. Integration Tests
Test with real strategy code:

```python
def test_strategy_with_unified_indicators():
    # Test complete strategy using unified indicators
    pass
```

### 3. Performance Tests
Compare performance before and after migration:

```python
def benchmark_indicator_performance():
    # Compare calculation times
    pass
```

## Migration Checklist

### Pre-Migration
- [ ] Identify all files using old indicator services
- [ ] Create backup of current implementation
- [ ] Review parameter usage in existing code
- [ ] Plan testing strategy

### During Migration
- [ ] Update import statements
- [ ] Replace wrapper creation with direct unified indicators
- [ ] Update parameter access patterns
- [ ] Remove deprecated parameters
- [ ] Update configuration files

### Post-Migration
- [ ] Run comprehensive tests
- [ ] Verify performance is maintained or improved
- [ ] Update documentation
- [ ] Remove old indicator service files
- [ ] Clean up unused imports

## Rollback Plan

If issues arise during migration:

1. **Immediate Rollback**: Restore from backup
2. **Partial Rollback**: Revert specific components while keeping others
3. **Gradual Migration**: Migrate components one at a time

## Support and Troubleshooting

### Common Error Messages

#### "No module named 'src.strategy.indicator.wrappers'"
**Cause**: Old import path used
**Solution**: Update to use unified service imports

#### "Backend 'backtrader' not supported"
**Cause**: Invalid backend parameter
**Solution**: Use "bt" instead of "backtrader"

#### "Indicator not ready"
**Cause**: Insufficient data or initialization issue
**Solution**: Check data availability and indicator initialization

### Getting Help

1. Check this migration guide first
2. Review the unified service documentation
3. Check existing test cases for examples
4. Contact the development team for complex issues

## Performance Considerations

### Expected Improvements
- **Memory Usage**: Reduced due to elimination of wrapper layers
- **Calculation Speed**: Improved due to direct service integration
- **Initialization Time**: Faster due to simplified architecture

### Monitoring
Monitor these metrics after migration:
- Indicator calculation times
- Memory usage during backtesting
- Strategy initialization time
- Overall system performance

## Future Considerations

### Deprecated Features
The following features are deprecated and will be removed:
- Legacy wrapper system
- Backward compatibility parameters
- Fallback mechanisms

### New Features
The unified service enables:
- Better error handling
- Improved logging
- Enhanced performance monitoring
- Easier testing and debugging

## Conclusion

The migration to the unified indicator service provides significant benefits in terms of maintainability, performance, and developer experience. While the migration requires updating existing code, the simplified interface and improved architecture make it worthwhile.

Follow this guide step by step, test thoroughly, and don't hesitate to reach out for support during the migration process.