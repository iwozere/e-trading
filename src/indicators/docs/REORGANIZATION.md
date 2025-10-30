# Indicators Module Reorganization

## Overview

All indicator-related models and logic have been moved into the `src/indicators` folder to isolate the indicator service logic and resolve circular import issues.

## Changes Made

### 1. Model Migration
- Moved all indicator models from `src/model/indicators.py` to `src/indicators/models.py`
- Updated all internal imports within the indicators module to use local imports
- Created backward compatibility layer in `src/model/indicators.py`

### 2. Import Updates
- Updated `src/indicators/constants.py` to import from local models
- Updated all files within `src/indicators/` to use local imports
- Updated `src/common/` files to import directly from `src/indicators/models`

### 3. Backward Compatibility
- `src/model/indicators.py` now serves as a backward compatibility layer
- External modules can still import from `src/model/indicators` 
- All existing APIs remain unchanged

## File Structure

```
src/indicators/
├── models.py           # All indicator data models (moved from src/model/)
├── constants.py        # Constants and utilities
├── service.py          # Main indicator service
├── types.py           # Type definitions
├── registry.py        # Indicator registry
├── recommendation_engine.py
├── utils.py
├── config_manager.py
├── adapters/          # Data adapters
├── tests/             # Test suite
└── docs/              # Documentation
    └── REORGANIZATION.md  # This file
```

## Import Guidelines

### For Code Within `src/indicators/`
Use local imports:
```python
from src.indicators.models import RecommendationType, IndicatorResult
from src.indicators.constants import validate_indicator_name
```

### For Code Outside `src/indicators/`
Use either approach:
```python
# Option 1: Direct import (recommended for new code)
from src.indicators.models import RecommendationType, IndicatorResult

# Option 2: Backward compatibility (for existing code)
from src.indicators.models import RecommendationType, IndicatorResult
```

## Benefits

1. **Isolation**: All indicator logic is now contained within the `src/indicators` module
2. **No Circular Imports**: Resolved circular dependency issues
3. **Backward Compatibility**: Existing code continues to work without changes
4. **Clear Structure**: Better separation of concerns and module boundaries
5. **Maintainability**: Easier to maintain and extend indicator functionality

## Migration Notes

- No breaking changes for external consumers
- All existing APIs remain functional
- Tests continue to pass without modification
- Documentation examples work with both import styles