# Unified Indicator Service - Quick Migration Reference

This is a quick reference guide for the most common migration patterns when updating to the unified indicator service.

## Import Changes

### Before
```python
from src.strategy.indicator.wrappers import create_indicator_wrapper  # Legacy - removed
```

### After
```python
from src.indicators.adapters.backtrader_wrappers import (
    UnifiedRSIIndicator,
    UnifiedBollingerBandsIndicator,
    UnifiedATRIndicator,
    UnifiedMACDIndicator
)
```

## Indicator Creation Patterns

### RSI Indicator

#### Before
```python
if self.strategy.use_talib:
    rsi_raw = bt.talib.RSI(self.strategy.data.close, timeperiod=period)
else:
    rsi_raw = bt.indicators.RSI(self.strategy.data.close, period=period)

self.rsi = create_indicator_wrapper(rsi_raw, 'rsi', self.strategy.use_talib)
```

#### After
```python
backend = "bt-talib" if self.strategy.use_talib else "bt"
self.rsi = UnifiedRSIIndicator(self.strategy.data, period=period, backend=backend)
```

### Bollinger Bands Indicator

#### Before
```python
if self.strategy.use_talib:
    bb_raw = bt.talib.BBANDS(
        self.strategy.data.close,
        timeperiod=period,
        nbdevup=devfactor,
        nbdevdn=devfactor,
    )
else:
    bb_raw = bt.indicators.BollingerBands(
        self.strategy.data.close, period=period, devfactor=devfactor
    )

self.bb = create_indicator_wrapper(bb_raw, 'bb', self.strategy.use_talib)
```

#### After
```python
backend = "bt-talib" if self.strategy.use_talib else "bt"
self.bb = UnifiedBollingerBandsIndicator(
    self.strategy.data, period=period, devfactor=devfactor, backend=backend
)
```

### ATR Indicator

#### Before
```python
if self.strategy.use_talib:
    atr_raw = bt.talib.ATR(
        self.strategy.data.high,
        self.strategy.data.low,
        self.strategy.data.close,
        timeperiod=period
    )
else:
    atr_raw = bt.indicators.ATR(self.strategy.data, period=period)

self.atr = create_indicator_wrapper(atr_raw, 'atr', self.strategy.use_talib)
```

#### After
```python
backend = "bt-talib" if self.strategy.use_talib else "bt"
self.atr = UnifiedATRIndicator(self.strategy.data, period=period, backend=backend)
```

## Value Access Patterns

### RSI Values

#### Before
```python
rsi_value = self.rsi[0]
rsi_previous = self.rsi[-1]
```

#### After
```python
rsi_value = self.rsi.rsi[0]
rsi_previous = self.rsi.rsi[-1]
```

### Bollinger Bands Values

#### Before (TALib)
```python
upper_band = self.bb.upperband[0]
middle_band = self.bb.middleband[0]
lower_band = self.bb.lowerband[0]
```

#### Before (Backtrader)
```python
upper_band = self.bb.top[0]
middle_band = self.bb.mid[0]
lower_band = self.bb.bot[0]
```

#### After (Unified)
```python
upper_band = self.bb.upper[0]
middle_band = self.bb.middle[0]
lower_band = self.bb.lower[0]
```

### ATR Values

#### Before
```python
atr_value = self.atr[0]
```

#### After
```python
atr_value = self.atr.atr[0]
```

## Backend Parameters

### Before
```python
# Complex backend handling
if use_talib:
    backend = "talib"
    use_unified_service = True
else:
    backend = "backtrader"
    use_unified_service = True
```

### After
```python
# Simplified backend selection
backend = "bt-talib" if use_talib else "bt"
# Note: use_unified_service parameter removed
```

## Common Validation Patterns

### Indicator Ready Check

#### Before
```python
def are_indicators_ready(self) -> bool:
    try:
        _ = self.rsi[0]
        _ = self.bb.bot[0]  # Different for TALib vs Backtrader
        return True
    except (IndexError, AttributeError):
        return False
```

#### After
```python
def are_indicators_ready(self) -> bool:
    try:
        _ = self.rsi.rsi[0]
        _ = self.bb.lower[0]  # Consistent naming
        return True
    except (IndexError, AttributeError):
        return False
```

## Error Handling Updates

### Before
```python
# Different error handling for different backends
if self.strategy.use_talib:
    try:
        value = self.bb.lowerband[0]
    except AttributeError:
        value = None
else:
    try:
        value = self.bb.bot[0]
    except AttributeError:
        value = None
```

### After
```python
# Consistent error handling
try:
    value = self.bb.lower[0]
except (AttributeError, IndexError):
    value = None
```

## Factory Method Updates

### Before
```python
factory = IndicatorFactory()
rsi = factory.create_backtrader_rsi(
    data, period=14, backend="bt", use_unified_service=True
)
```

### After
```python
factory = IndicatorFactory()
rsi = factory.create_backtrader_rsi(data, period=14, backend="bt")
# Note: use_unified_service parameter removed
```

## Testing Pattern Updates

### Before
```python
def test_indicator_creation(self):
    # Test both TALib and Backtrader versions
    for use_talib in [True, False]:
        if use_talib:
            rsi_raw = bt.talib.RSI(data.close, timeperiod=14)
        else:
            rsi_raw = bt.indicators.RSI(data.close, period=14)
        
        rsi = create_indicator_wrapper(rsi_raw, 'rsi', use_talib)
        self.assertIsNotNone(rsi[0])
```

### After
```python
def test_indicator_creation(self):
    # Test both backends with unified interface
    for backend in ["bt", "bt-talib"]:
        rsi = UnifiedRSIIndicator(data, period=14, backend=backend)
        self.assertIsNotNone(rsi.rsi[0])
```

## Search and Replace Patterns

Use these regex patterns for bulk updates:

### Update Imports
```regex
# Find
from src\.strategy\.indicator\.wrappers import create_indicator_wrapper

# Replace
from src.indicators.adapters.backtrader_wrappers import UnifiedRSIIndicator, UnifiedBollingerBandsIndicator, UnifiedATRIndicator
```

### Update RSI Access
```regex
# Find
\.rsi\[([0-9-]+)\]

# Replace
.rsi.rsi[$1]
```

### Update Bollinger Bands Access
```regex
# Find
\.bb\.bot\[([0-9-]+)\]

# Replace
.bb.lower[$1]
```

```regex
# Find
\.bb\.mid\[([0-9-]+)\]

# Replace
.bb.middle[$1]
```

```regex
# Find
\.bb\.top\[([0-9-]+)\]

# Replace
.bb.upper[$1]
```

### Update ATR Access
```regex
# Find
\.atr\[([0-9-]+)\]

# Replace
.atr.atr[$1]
```

## Validation Commands

Run these commands to validate your migration:

```bash
# Check for old imports
grep -r "from src.strategy.indicator.wrappers" src/

# Check for old wrapper usage
grep -r "create_indicator_wrapper" src/

# Check for old parameter access patterns
grep -r "\.bot\[" src/
grep -r "\.mid\[" src/
grep -r "\.top\[" src/

# Check for deprecated parameters
grep -r "use_unified_service" src/
```

## Quick Checklist

- [ ] Updated all imports from wrappers to unified indicators
- [ ] Replaced `create_indicator_wrapper` calls
- [ ] Updated RSI access from `rsi[0]` to `rsi.rsi[0]`
- [ ] Updated Bollinger Bands access to consistent naming
- [ ] Updated ATR access from `atr[0]` to `atr.atr[0]`
- [ ] Simplified backend parameter usage
- [ ] Removed deprecated parameters
- [ ] Updated test cases
- [ ] Validated with real data

This quick reference should cover 90% of common migration scenarios. For complex cases, refer to the full migration guide.