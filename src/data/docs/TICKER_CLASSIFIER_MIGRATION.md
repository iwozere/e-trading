# TickerClassifier Migration Guide

## Overview

The `TickerClassifier` from `src/common/ticker_classifier.py` has been replaced by the enhanced `ProviderSelector` in the new data architecture. This migration provides better maintainability, configuration-driven rules, and integration with the unified data management system.

## What Changed

### Before (TickerClassifier)
```python
from src.common.ticker_classifier import TickerClassifier, DataProvider

classifier = TickerClassifier()
ticker_info = classifier.classify_ticker("BTCUSDT")
config = classifier.get_data_provider_config("BTCUSDT", "1h")
```

### After (ProviderSelector)
```python
from src.data.data_manager import ProviderSelector

selector = ProviderSelector()
ticker_info = selector.get_ticker_info("BTCUSDT")
config = selector.get_data_provider_config("BTCUSDT", "1h")
```

## Migration Steps

### 1. Update Imports
Replace all imports of `TickerClassifier` with `ProviderSelector`:

```python
# Old
from src.common.ticker_classifier import TickerClassifier, DataProvider

# New
from src.data.data_manager import ProviderSelector
```

### 2. Update Class Instantiation
```python
# Old
classifier = TickerClassifier()

# New
selector = ProviderSelector()
```

### 3. Update Method Calls
The new `ProviderSelector` provides equivalent functionality with slightly different method names:

| Old Method | New Method | Notes |
|------------|------------|-------|
| `classify_ticker(symbol)` | `get_ticker_info(symbol)` | Returns dict instead of TickerInfo object |
| `get_data_provider_config(symbol, interval)` | `get_data_provider_config(symbol, interval)` | Same method name, enhanced functionality |
| `validate_ticker(symbol)` | `validate_ticker(symbol)` | Same method name, enhanced validation |
| `get_provider_for_interval(symbol, interval)` | `get_best_provider(symbol, interval)` | Simplified method name |

### 4. Update Data Access
The new methods return dictionaries instead of dataclass objects:

```python
# Old
ticker_info = classifier.classify_ticker("BTCUSDT")
provider = ticker_info.provider.value
formatted_ticker = ticker_info.formatted_ticker

# New
ticker_info = selector.get_ticker_info("BTCUSDT")
symbol_type = ticker_info['symbol_type']
formatted_ticker = ticker_info['formatted_ticker']
```

## Configuration-Driven Rules

The new system uses `config/data/provider_rules.yaml` for all classification rules instead of hardcoded logic. This provides:

- **Maintainability**: Rules can be updated without code changes
- **Extensibility**: Easy to add new symbol types and patterns
- **Consistency**: Single source of truth for all classification rules

### Key Configuration Sections

1. **Symbol Classification Rules**: Define patterns for crypto and stock symbols
2. **Provider Rules**: Define which providers to use for different symbol types and timeframes
3. **Exchange Mappings**: Comprehensive list of stock exchange suffixes

## Benefits of Migration

1. **Unified Architecture**: Integrated with the new DataManager system
2. **Configuration-Driven**: Rules defined in YAML instead of hardcoded
3. **Better Performance**: Optimized regex patterns and caching
4. **Enhanced Validation**: More comprehensive ticker validation
5. **Failover Support**: Built-in provider failover mechanisms
6. **Maintainability**: Easier to update and extend

## Backward Compatibility

The new `ProviderSelector` maintains API compatibility for the most common use cases:

- `get_data_provider_config()` method signature is identical
- `validate_ticker()` method signature is identical
- Return values contain the same information (in dict format)

## Testing the Migration

To test that the migration works correctly:

```python
from src.data.data_manager import ProviderSelector

selector = ProviderSelector()

# Test crypto symbols
crypto_info = selector.get_ticker_info("BTCUSDT")
assert crypto_info['symbol_type'] == 'crypto'
assert crypto_info['base_asset'] == 'BTC'
assert crypto_info['quote_asset'] == 'USDT'

# Test stock symbols
stock_info = selector.get_ticker_info("AAPL")
assert stock_info['symbol_type'] == 'stock'
assert stock_info['exchange'] == "US Markets (NASDAQ/NYSE)"

# Test international stocks
intl_info = selector.get_ticker_info("SAP.DE")
assert intl_info['symbol_type'] == 'stock'
assert intl_info['exchange'] == "XETRA (Germany)"
```

## Files Updated

The following files have been updated to use the new ProviderSelector:

- `src/data/cache/populate_cache.py`
- `src/data/data_manager.py` (enhanced ProviderSelector)
- `config/data/provider_rules.yaml` (added symbol classification rules)

## Deprecation Notice

The `src/common/ticker_classifier.py` file can now be safely removed after all references have been migrated to use the new `ProviderSelector`. The old file will be deprecated in a future release.
