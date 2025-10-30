# Unified Indicator Configuration Guide

## Overview

The unified indicator service uses a single configuration file `config/indicators.json` that consolidates all indicator-related settings, including parameters, presets, legacy mappings, and plotter configurations.

## Configuration Structure

### Version 2.0 Format

```json
{
  "version": "2.0",
  "description": "Unified indicator service configuration",
  "default_parameters": { ... },
  "presets": { ... },
  "legacy_mappings": { ... },
  "plotter_config": { ... }
}
```

## Sections

### 1. Default Parameters

Contains default parameters for each indicator using canonical names:

```json
"default_parameters": {
  "rsi": {
    "timeperiod": 14
  },
  "macd": {
    "fastperiod": 12,
    "slowperiod": 26,
    "signalperiod": 9
  },
  "bbands": {
    "timeperiod": 20,
    "nbdevup": 2,
    "nbdevdn": 2
  }
}
```

### 2. Presets

Predefined parameter sets for different trading styles:

- **default**: Balanced parameters for general use
- **conservative**: Lower risk, longer periods
- **aggressive**: Higher frequency, shorter periods  
- **day_trading**: Optimized for intraday trading

```json
"presets": {
  "conservative": {
    "description": "Conservative parameters for lower risk trading",
    "parameters": {
      "rsi": {"timeperiod": 21},
      "macd": {"fastperiod": 15, "slowperiod": 30, "signalperiod": 12}
    }
  }
}
```

### 3. Legacy Mappings

Maps old indicator names to canonical names for backward compatibility:

```json
"legacy_mappings": {
  "RSI": "rsi",
  "MACD": "macd",
  "Bollinger_Bands": "bbands",
  "BB_UPPER": "bbands",
  "SMA_FAST": "sma",
  "SMA_50": "sma"
}
```

### 4. Plotter Configuration

Defines how indicators are displayed in plots for different strategy mixins:

```json
"plotter_config": {
  "entry_mixins": {
    "RSIBBEntryMixin": {
      "indicators": ["rsi", "bbands"],
      "subplot_type": {
        "rsi": "separate",
        "bbands": "overlay"
      }
    }
  },
  "exit_mixins": {
    "ATRExitMixin": {
      "indicators": ["atr"],
      "subplot_type": {
        "atr": "separate"
      }
    }
  }
}
```

## Usage Examples

### Loading Configuration

```python
from src.indicators.config_manager import get_config_manager

# Get the global config manager
config_manager = get_config_manager()

# Get parameters for an indicator
rsi_params = config_manager.get_parameters('rsi')
macd_params = config_manager.get_parameters('MACD')  # Legacy name works too
```

### Using Presets

```python
# Set a preset
config_manager.set_preset('aggressive')

# Get parameters with preset applied
rsi_params = config_manager.get_parameters('rsi')  # Uses aggressive preset
```

### Runtime Overrides

```python
# Override a parameter at runtime
config_manager.set_parameter_override('rsi', 'timeperiod', 10)

# Get parameters with override applied
rsi_params = config_manager.get_parameters('rsi')  # timeperiod will be 10
```

## Migration from Legacy Configuration

### Old Format (v1.0)
```json
{
  "default_parameters": {
    "RSI": {"timeperiod": 14}
  },
  "indicator_mapping": {
    "SMA_50": "SMA_FAST"
  },
  "custom_presets": {
    "conservative": {
      "RSI": {"timeperiod": 20}
    }
  }
}
```

### New Format (v2.0)
```json
{
  "version": "2.0",
  "default_parameters": {
    "rsi": {"timeperiod": 14}
  },
  "legacy_mappings": {
    "RSI": "rsi",
    "SMA_50": "sma"
  },
  "presets": {
    "conservative": {
      "description": "Conservative parameters",
      "parameters": {
        "rsi": {"timeperiod": 20}
      }
    }
  }
}
```

## Key Changes

1. **Canonical Names**: All indicators now use lowercase canonical names (e.g., `rsi` instead of `RSI`)
2. **Unified Presets**: `custom_presets` renamed to `presets` with structured format
3. **Legacy Support**: `indicator_mapping` renamed to `legacy_mappings` for clarity
4. **Plotter Integration**: Plotter configuration moved from separate file to main config
5. **Version Tracking**: Added version field for future migrations

## Best Practices

1. **Use Canonical Names**: Always use lowercase canonical names in new code
2. **Leverage Presets**: Use presets instead of hardcoding parameters
3. **Test Legacy Names**: Ensure backward compatibility when updating existing code
4. **Document Changes**: Update documentation when adding new indicators or presets
5. **Validate Parameters**: Use the config manager's validation methods

## Troubleshooting

### Common Issues

1. **Unknown Indicator**: Check if the indicator name is in the registry
2. **Invalid Parameters**: Use `validate_parameters()` to check parameter validity
3. **Preset Not Found**: Check available presets with `get_available_presets()`
4. **Legacy Name Issues**: Verify legacy mappings are correct

### Debug Commands

```python
# Get configuration info
info = config_manager.get_config_info()
print(info)

# Check available presets
presets = config_manager.get_available_presets()
print(presets)

# Validate parameters
errors = config_manager.validate_parameters('rsi', {'timeperiod': 14})
print(errors)
```