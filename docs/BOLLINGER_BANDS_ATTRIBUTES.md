# Bollinger Bands Indicator Attributes - Reference Guide

**Last Updated:** 2025-11-12

## Summary

Different Bollinger Bands implementations use different attribute names for accessing band values. This document clarifies the correct usage for each implementation.

## Implementations

### 1. UnifiedBollingerBandsIndicator (Unified Wrapper)

**Location:** `src/indicators/adapters/backtrader_wrappers.py`

**Lines Defined:**
```python
lines = ("upper", "middle", "lower")
```

**Usage:**
```python
bb = UnifiedBollingerBandsIndicator(data, period=20, devfactor=2.0)
upper_band = bb.upper[0]
middle_band = bb.middle[0]
lower_band = bb.lower[0]
```

**Used By:**
- `RSIBBEntryMixin`
- `RSIOrBBEntryMixin`
- `RSIBBVolumeEntryMixin`
- `RSIBBExitMixin`
- `RSIOrBBExitMixin`

**Key Feature:** Provides unified interface regardless of backend (TALib or standard Backtrader)

---

### 2. Backtrader TALib BBANDS

**Direct TALib Usage:** `bt.talib.BBANDS()`

**Attributes:**
```python
bb = bt.talib.BBANDS(data.close, timeperiod=20, nbdevup=2, nbdevdn=2)
upper_band = bb.upperband[0]
middle_band = bb.middleband[0]
lower_band = bb.lowerband[0]
```

**Used By:**
- `BBVolumeSuperTrendEntryMixin` (legacy architecture)

**Note:** TALib's BBANDS returns three separate lines: `upperband`, `middleband`, `lowerband`

---

### 3. Standard Backtrader BollingerBands

**Standard Indicator:** `bt.indicators.BollingerBands()`

**Attributes:**
```python
bb = bt.indicators.BollingerBands(data, period=20, devfactor=2.0)
upper_band = bb.lines.top[0]
middle_band = bb.lines.mid[0]
lower_band = bb.lines.bot[0]
```

**Note:** Access through `.lines` property

---

## Common Issues and Fixes

### Issue 1: AttributeError 'bb_lower' does not exist

**Problem:**
```python
bb_lower = bb.bb_lower[0]  # ❌ WRONG
```

**Solution - Using UnifiedBollingerBandsIndicator:**
```python
bb_lower = bb.lower[0]  # ✅ CORRECT
```

**Solution - Using TALib BBANDS:**
```python
bb_lower = bb.lowerband[0]  # ✅ CORRECT
```

---

### Issue 2: Mixing TALib attributes with wrapper

**Problem:**
```python
# When using UnifiedBollingerBandsIndicator
if strategy.use_talib:
    bb_lower = bb.bot[0]  # ❌ WRONG - wrapper always uses .lower
else:
    bb_lower = bb.lower[0]
```

**Solution:**
```python
# UnifiedBollingerBandsIndicator always uses same interface
bb_lower = bb.lower[0]  # ✅ CORRECT - works regardless of backend
```

---

### Issue 3: Accessing TALib BBANDS container object

**Problem:**
```python
# When TALib BBANDS returns a container object
bb = bt.talib.BBANDS(...)
bb_lower = bb.bot[0]  # ❌ WRONG - TALib uses .lowerband not .bot
```

**Solution:**
```python
bb = bt.talib.BBANDS(...)
bb_lower = bb.lowerband[0]  # ✅ CORRECT for TALib BBANDS
```

---


---

## Decision Tree

```
Are you using UnifiedBollingerBandsIndicator wrapper?
├─ YES → Use: bb.lower, bb.middle, bb.upper
└─ NO → Are you using direct bt.talib.BBANDS?
    ├─ YES → Use: bb.lowerband, bb.middleband, bb.upperband
    └─ NO → Standard BT → Use: bb.lines.bot, bb.lines.mid, bb.lines.top
```

---

## Recent Fixes

### BBVolumeSuperTrendEntryMixin (2025-11-12)

**File:** `src/strategy/entry/bb_volume_supertrend_entry_mixin.py:206`

**Before:**
```python
if self.strategy.use_talib:
    bb_lower = bb.bb_lower[0]  # ❌ AttributeError
```

**After (Attempt 1 - Still Wrong):**
```python
if self.strategy.use_talib:
    bb_lower = bb.bot[0]  # ❌ Still AttributeError - TALib uses .lowerband
```

**After (Attempt 2 - Correct):**
```python
if self.strategy.use_talib:
    bb_lower = bb.lowerband[0]  # ✅ CORRECT - TALib uses .lowerband
```

---

## Best Practices

### 1. Prefer Unified Wrappers

Use `UnifiedBollingerBandsIndicator` when possible for consistent interface:

```python
from src.indicators.adapters.backtrader_wrappers import UnifiedBollingerBandsIndicator

# Works with both TALib and standard backends
bb = UnifiedBollingerBandsIndicator(
    strategy.data,
    period=20,
    devfactor=2.0,
    backend="bt-talib" if strategy.use_talib else "bt"
)

# Always use the same attributes
bb_lower = bb.lower[0]
bb_middle = bb.middle[0]
bb_upper = bb.upper[0]
```

### 2. Document Direct TALib Usage

If using TALib directly, add clear comments:

```python
if self.strategy.use_talib:
    # TALib BBANDS uses: upperband, middleband, lowerband
    bb = bt.talib.BBANDS(...)
    bb_lower = bb.lowerband[0]
else:
    bb = bt.indicators.BollingerBands(...)
    bb_lower = bb.lines.bot[0]
```

### 3. Test with Both Backends

When creating new mixins that use Bollinger Bands:
- Test with `use_talib=True`
- Test with `use_talib=False`
- Ensure both paths work correctly

---

## Quick Reference Table

| Implementation | Upper | Middle | Lower |
|---|---|---|---|
| **UnifiedBollingerBandsIndicator** | `bb.upper[0]` | `bb.middle[0]` | `bb.lower[0]` |
| **bt.talib.BBANDS** | `bb.upperband[0]` | `bb.middleband[0]` | `bb.lowerband[0]` |
| **bt.indicators.BollingerBands** | `bb.lines.top[0]` | `bb.lines.mid[0]` | `bb.lines.bot[0]` |

---

## See Also

- [Indicator Service Architecture](./HLA/indicator-service-architecture.md)
- [Strategy Mixin Development](../src/strategy/README.md)
- [Backtrader TALib Integration](https://backtrader.com/docu/talib/talib/)
