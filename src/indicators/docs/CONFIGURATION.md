# Unified Indicator Service Configuration Guide

## Overview

The Unified Indicator Service uses a centralized configuration system that manages parameters, presets, and mappings for all indicators. This guide covers all configuration options, parameter customization, and the simplified parameter structure introduced in the unified service.

## Configuration File Structure

The main configuration file is located at `config/indicators.json` and follows this structure:

```json
{
  "version": "2.0",
  "default_parameters": {
    "rsi": {"timeperiod": 14},
    "macd": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
    "bbands": {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0}
  },
  "presets": {
    "default": {
      "description": "Default parameters from registry",
      "parameters": {}
    },
    "conservative": {
      "description": "Conservative trading parameters",
      "parameters": {
        "rsi": {"timeperiod": 21},
        "macd": {"fastperiod": 15, "slowperiod": 30, "signalperiod": 12}
      }
    }
  },
  "legacy_mappings": {
    "RSI": "rsi",
    "MACD": "macd",
    "BB_UPPER": "bbands"
  }
}
```

## Parameter Presets

The unified service includes four built-in parameter presets optimized for different trading styles:

### 1. Default Preset

Standard parameters based on common technical analysis practices.

```json
{
  "rsi": {"timeperiod": 14},
  "macd": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
  "bbands": {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
  "stoch": {"fastk_period": 14, "slowk_period": 3, "slowd_period": 3},
  "adx": {"timeperiod": 14},
  "sma": {"timeperiod": 20},
  "ema": {"timeperiod": 20},
  "cci": {"timeperiod": 14},
  "roc": {"timeperiod": 10},
  "mfi": {"timeperiod": 14},
  "williams_r": {"timeperiod": 14},
  "atr": {"timeperiod": 14}
}
```

### 2. Conservative Preset

Longer periods for smoother signals, suitable for long-term investing.

```json
{
  "rsi": {"timeperiod": 21},
  "macd": {"fastperiod": 15, "slowperiod": 30, "signalperiod": 12},
  "bbands": {"timeperiod": 25, "nbdevup": 2.5, "nbdevdn": 2.5},
  "stoch": {"fastk_period": 21, "slowk_period": 5, "slowd_period": 5},
  "adx": {"timeperiod": 21},
  "sma": {"timeperiod": 50},
  "ema": {"timeperiod": 50},
  "cci": {"timeperiod": 21},
  "roc": {"timeperiod": 15},
  "mfi": {"timeperiod": 21},
  "williams_r": {"timeperiod": 21},
  "atr": {"timeperiod": 21}
}
```

### 3. Aggressive Preset

Shorter periods for faster signals, suitable for active trading.

```json
{
  "rsi": {"timeperiod": 7},
  "macd": {"fastperiod": 8, "slowperiod": 17, "signalperiod": 5},
  "bbands": {"timeperiod": 15, "nbdevup": 1.5, "nbdevdn": 1.5},
  "stoch": {"fastk_period": 7, "slowk_period": 2, "slowd_period": 2},
  "adx": {"timeperiod": 7},
  "sma": {"timeperiod": 10},
  "ema": {"timeperiod": 10},
  "cci": {"timeperiod": 7},
  "roc": {"timeperiod": 5},
  "mfi": {"timeperiod": 7},
  "williams_r": {"timeperiod": 7},
  "atr": {"timeperiod": 7}
}
```

### 4. Day Trading Preset

Very short periods optimized for intraday trading.

```json
{
  "rsi": {"timeperiod": 5},
  "macd": {"fastperiod": 5, "slowperiod": 13, "signalperiod": 3},
  "bbands": {"timeperiod": 10, "nbdevup": 1.8, "nbdevdn": 1.8},
  "stoch": {"fastk_period": 5, "slowk_period": 1, "slowd_period": 1},
  "adx": {"timeperiod": 5},
  "sma": {"timeperiod": 5},
  "ema": {"timeperiod": 5},
  "cci": {"timeperiod": 5},
  "roc": {"timeperiod": 3},
  "mfi": {"timeperiod": 5},
  "williams_r": {"timeperiod": 5},
  "atr": {"timeperiod": 5}
}
```

## Configuration Management API

### Basic Usage

```python
from src.indicators.config_manager import get_config_manager

# Get the global configuration manager
config = get_config_manager()

# Get current configuration info
info = config.get_config_info()
print(f"Current preset: {info['current_preset']}")
print(f"Available presets: {info['available_presets']}")
```

### Working with Presets

```python
# List available presets
presets = config.get_available_presets()
print(f"Available presets: {presets}")

# Get preset information
preset_info = config.get_preset_info("conservative")
print(f"Description: {preset_info.description}")
print(f"Parameters: {preset_info.parameters}")

# Set active preset
success = config.set_preset("aggressive")
if success:
    print("Preset changed to aggressive")
else:
    print("Failed to change preset")

# Get current preset
current = config.get_current_preset()
print(f"Current preset: {current}")
```

### Parameter Retrieval

```python
# Get parameters for specific indicator
rsi_params = config.get_parameters("rsi")
print(f"RSI parameters: {rsi_params}")

# Get parameters with specific preset
conservative_rsi = config.get_parameters("rsi", "conservative")
print(f"Conservative RSI: {conservative_rsi}")

# Get parameters for legacy indicator name
legacy_params = config.get_parameters("RSI")  # Same as "rsi"
print(f"Legacy RSI parameters: {legacy_params}")
```

### Runtime Parameter Overrides

```python
# Override specific parameter
config.set_parameter_override("rsi", "timeperiod", 21)

# Override multiple parameters
config.set_parameter_override("macd", "fastperiod", 8)
config.set_parameter_override("macd", "slowperiod", 17)
config.set_parameter_override("macd", "signalperiod", 5)

# Get parameters with overrides applied
params = config.get_parameters("rsi")
print(f"RSI with override: {params}")  # {'timeperiod': 21}

# Clear specific override
config.clear_parameter_overrides("rsi")

# Clear all overrides
config.clear_parameter_overrides()
```

### Creating Custom Presets

```python
# Define custom parameters
custom_params = {
    "rsi": {"timeperiod": 18},
    "macd": {"fastperiod": 10, "slowperiod": 22, "signalperiod": 8},
    "bbands": {"timeperiod": 18, "nbdevup": 2.2, "nbdevdn": 2.2}
}

# Create custom preset
success = config.create_preset(
    name="custom_swing",
    description="Custom parameters for swing trading",
    parameters=custom_params
)

if success:
    print("Custom preset created successfully")
    
    # Use the custom preset
    config.set_preset("custom_swing")
```

### Parameter Validation

```python
# Validate parameters before setting
test_params = {"timeperiod": 25, "invalid_param": "bad_value"}
errors = config.validate_parameters("rsi", test_params)

if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Parameters are valid")
```

## Indicator-Specific Parameters

### Technical Indicators

#### RSI (Relative Strength Index)
```json
{
  "timeperiod": 14  // Period for RSI calculation (1-200)
}
```

#### MACD (Moving Average Convergence Divergence)
```json
{
  "fastperiod": 12,    // Fast EMA period (1-100)
  "slowperiod": 26,    // Slow EMA period (1-200, must be > fastperiod)
  "signalperiod": 9    // Signal line EMA period (1-50)
}
```

#### Bollinger Bands
```json
{
  "timeperiod": 20,  // Moving average period (1-200)
  "nbdevup": 2.0,    // Upper band standard deviations (0.1-5.0)
  "nbdevdn": 2.0     // Lower band standard deviations (0.1-5.0)
}
```

#### Stochastic Oscillator
```json
{
  "fastk_period": 14,  // %K period (1-100)
  "slowk_period": 3,   // %K slowing period (1-50)
  "slowd_period": 3    // %D period (1-50)
}
```

#### ADX (Average Directional Index)
```json
{
  "timeperiod": 14  // Period for ADX calculation (1-100)
}
```

#### Moving Averages (SMA/EMA)
```json
{
  "timeperiod": 20  // Period for moving average (1-200)
}
```

#### CCI (Commodity Channel Index)
```json
{
  "timeperiod": 14  // Period for CCI calculation (1-100)
}
```

#### ROC (Rate of Change)
```json
{
  "timeperiod": 10  // Period for ROC calculation (1-100)
}
```

#### MFI (Money Flow Index)
```json
{
  "timeperiod": 14  // Period for MFI calculation (1-100)
}
```

#### Williams %R
```json
{
  "timeperiod": 14  // Period for Williams %R calculation (1-100)
}
```

#### ATR (Average True Range)
```json
{
  "timeperiod": 14  // Period for ATR calculation (1-100)
}
```

#### Aroon Oscillator
```json
{
  "timeperiod": 14  // Period for Aroon calculation (1-100)
}
```

#### Parabolic SAR
```json
{
  "acceleration": 0.02,  // Acceleration factor (0.01-1.0)
  "maximum": 0.20        // Maximum acceleration (0.01-1.0)
}
```

#### Super Trend
```json
{
  "length": 10,      // ATR period (1-100)
  "multiplier": 3.0  // ATR multiplier (1.0-10.0)
}
```

### Fundamental Indicators

Fundamental indicators typically don't require parameters as they use the latest available financial data. However, some may accept optional parameters:

#### Growth Indicators
```json
{
  "period": "annual"  // "annual" or "quarterly" for growth calculations
}
```

## Parameter Priority System

The configuration system uses a priority hierarchy for parameter resolution:

1. **Runtime Overrides** (Highest Priority)
   - Set via `set_parameter_override()`
   - Temporary, cleared on service restart

2. **Active Preset Parameters**
   - Parameters from the currently active preset
   - Set via `set_preset()`

3. **Global Default Parameters**
   - Parameters from `default_parameters` section in config file

4. **Registry Defaults** (Lowest Priority)
   - Built-in defaults from indicator metadata

### Example Priority Resolution

```python
# Registry default: RSI timeperiod = 14
# Config file default: RSI timeperiod = 16
# Conservative preset: RSI timeperiod = 21
# Runtime override: RSI timeperiod = 25

config.set_preset("conservative")
config.set_parameter_override("rsi", "timeperiod", 25)

params = config.get_parameters("rsi")
print(params)  # {'timeperiod': 25} - Runtime override wins
```

## Legacy Name Mappings

The unified service maintains backward compatibility through legacy name mappings:

### Technical Indicator Mappings

```json
{
  "RSI": "rsi",
  "MACD": "macd",
  "MACD_SIGNAL": "macd",
  "MACD_HISTOGRAM": "macd",
  "BB_UPPER": "bbands",
  "BB_MIDDLE": "bbands",
  "BB_LOWER": "bbands",
  "SMA_FAST": "sma",
  "SMA_SLOW": "sma",
  "SMA_50": "sma",
  "SMA_200": "sma",
  "EMA_FAST": "ema",
  "EMA_SLOW": "ema",
  "EMA_12": "ema",
  "EMA_26": "ema",
  "ADX": "adx",
  "PLUS_DI": "plus_di",
  "MINUS_DI": "minus_di",
  "STOCH_K": "stoch",
  "STOCH_D": "stoch",
  "WILLIAMS_R": "williams_r",
  "CCI": "cci",
  "ROC": "roc",
  "MFI": "mfi",
  "OBV": "obv",
  "ADR": "adr",
  "ATR": "atr"
}
```

### Fundamental Indicator Mappings

```json
{
  "PE_RATIO": "pe_ratio",
  "FORWARD_PE": "forward_pe",
  "PB_RATIO": "pb_ratio",
  "PS_RATIO": "ps_ratio",
  "PEG_RATIO": "peg_ratio",
  "ROE": "roe",
  "ROA": "roa",
  "DEBT_TO_EQUITY": "debt_to_equity",
  "CURRENT_RATIO": "current_ratio",
  "QUICK_RATIO": "quick_ratio",
  "OPERATING_MARGIN": "operating_margin",
  "PROFIT_MARGIN": "profit_margin",
  "REVENUE_GROWTH": "revenue_growth",
  "NET_INCOME_GROWTH": "net_income_growth",
  "FREE_CASH_FLOW": "free_cash_flow",
  "DIVIDEND_YIELD": "dividend_yield",
  "PAYOUT_RATIO": "payout_ratio",
  "BETA": "beta",
  "MARKET_CAP": "market_cap",
  "ENTERPRISE_VALUE": "enterprise_value",
  "EV_TO_EBITDA": "ev_to_ebitda"
}
```

## Configuration File Management

### Loading Configuration

```python
# Reload configuration from file
config.reload_config()

# Get configuration file path
info = config.get_config_info()
print(f"Config file: {info['config_path']}")
```

### Saving Configuration

```python
# Save current configuration to file
success = config.save_config()
if success:
    print("Configuration saved successfully")

# Save to custom location
success = config.save_config("custom/path/indicators.json")
```

### Configuration Validation

The configuration system validates all parameters:

```python
# Validate specific indicator parameters
errors = config.validate_parameters("rsi", {"timeperiod": -5})
print(errors)  # ['Parameter timeperiod for rsi must be positive']

# Validate preset parameters
test_preset = {
    "rsi": {"timeperiod": 0},  # Invalid
    "macd": {"fastperiod": 20, "slowperiod": 10}  # Invalid: slow < fast
}

for indicator, params in test_preset.items():
    errors = config.validate_parameters(indicator, params)
    if errors:
        print(f"{indicator} errors: {errors}")
```

## Environment-Based Configuration

The service supports environment-based configuration overrides:

### Environment Variables

```bash
# Override default preset
export INDICATOR_DEFAULT_PRESET=aggressive

# Override specific parameters
export INDICATOR_RSI_TIMEPERIOD=21
export INDICATOR_MACD_FASTPERIOD=8

# Override configuration file path
export INDICATOR_CONFIG_PATH=/custom/path/indicators.json
```

### Loading Environment Overrides

```python
import os
from src.indicators.config_manager import get_config_manager

# Environment variables are automatically loaded
config = get_config_manager()

# Check if environment overrides are active
if os.getenv('INDICATOR_DEFAULT_PRESET'):
    print(f"Using environment preset: {os.getenv('INDICATOR_DEFAULT_PRESET')}")
```

## Advanced Configuration

### Custom Configuration Manager

```python
from src.indicators.config_manager import UnifiedConfigManager

# Create custom configuration manager with different file
custom_config = UnifiedConfigManager("custom/indicators.json")

# Use custom configuration
from src.indicators.service import UnifiedIndicatorService

service = UnifiedIndicatorService()
service.config_manager = custom_config
```

### Configuration Inheritance

```python
# Create preset that inherits from another
base_preset = config.get_preset_info("conservative")
custom_params = base_preset.parameters.copy()

# Modify specific parameters
custom_params["rsi"]["timeperiod"] = 18
custom_params["macd"]["fastperiod"] = 10

# Create new preset
config.create_preset(
    name="custom_conservative",
    description="Modified conservative preset",
    parameters=custom_params
)
```

### Batch Parameter Updates

```python
# Update multiple parameters at once
parameter_updates = {
    "rsi": {"timeperiod": 21},
    "macd": {"fastperiod": 8, "slowperiod": 17},
    "bbands": {"timeperiod": 18, "nbdevup": 2.2}
}

for indicator, params in parameter_updates.items():
    for param, value in params.items():
        config.set_parameter_override(indicator, param, value)
```

## Migration from Legacy Configuration

### Parameter Name Changes

The unified service uses simplified parameter names:

| Legacy Parameter | Unified Parameter | Notes |
|------------------|-------------------|-------|
| `rsi_period` | `timeperiod` | Standardized name |
| `macd_fast` | `fastperiod` | Standardized name |
| `macd_slow` | `slowperiod` | Standardized name |
| `macd_signal` | `signalperiod` | Standardized name |
| `bb_period` | `timeperiod` | Standardized name |
| `bb_std_dev` | `nbdevup`/`nbdevdn` | Split into up/down |

### Configuration File Migration

```python
# Migrate legacy configuration
def migrate_legacy_config(legacy_config):
    """Migrate legacy configuration to unified format."""
    unified_config = {
        "version": "2.0",
        "default_parameters": {},
        "presets": {"default": {"description": "Migrated", "parameters": {}}},
        "legacy_mappings": {}
    }
    
    # Map legacy parameters
    for indicator, params in legacy_config.items():
        canonical_name = get_canonical_name(indicator)
        unified_params = {}
        
        # Map parameter names
        for param, value in params.items():
            if param == "rsi_period":
                unified_params["timeperiod"] = value
            elif param == "macd_fast":
                unified_params["fastperiod"] = value
            # ... more mappings
            
        unified_config["default_parameters"][canonical_name] = unified_params
    
    return unified_config
```

## Best Practices

### 1. Use Appropriate Presets

Choose presets based on your trading style:

```python
# Long-term investing
config.set_preset("conservative")

# Active trading
config.set_preset("aggressive")

# Day trading
config.set_preset("day_trading")
```

### 2. Validate Parameters

Always validate parameters before applying:

```python
def safe_parameter_override(indicator, parameter, value):
    """Safely set parameter override with validation."""
    test_params = {parameter: value}
    errors = config.validate_parameters(indicator, test_params)
    
    if errors:
        print(f"Validation failed: {errors}")
        return False
    
    config.set_parameter_override(indicator, parameter, value)
    return True
```

### 3. Document Custom Presets

```python
# Create well-documented custom presets
config.create_preset(
    name="swing_trading",
    description="Optimized for 2-5 day swing trades with moderate risk",
    parameters={
        "rsi": {"timeperiod": 18},  # Slightly faster than default
        "macd": {"fastperiod": 10, "slowperiod": 22, "signalperiod": 8},
        "bbands": {"timeperiod": 18, "nbdevup": 2.2, "nbdevdn": 2.2}
    }
)
```

### 4. Use Environment Variables for Production

```bash
# Production environment
export INDICATOR_DEFAULT_PRESET=conservative
export INDICATOR_CONFIG_PATH=/etc/trading/indicators.json
```

### 5. Monitor Configuration Changes

```python
# Log configuration changes
import logging
logger = logging.getLogger(__name__)

def log_preset_change(old_preset, new_preset):
    logger.info("Preset changed from %s to %s", old_preset, new_preset)

old_preset = config.get_current_preset()
config.set_preset("aggressive")
log_preset_change(old_preset, "aggressive")
```

## Troubleshooting

### Common Configuration Issues

1. **Invalid Parameter Values**
   ```python
   # Check parameter ranges
   errors = config.validate_parameters("rsi", {"timeperiod": 0})
   print(errors)  # Shows validation error
   ```

2. **Preset Not Found**
   ```python
   # Check available presets
   presets = config.get_available_presets()
   print(f"Available: {presets}")
   ```

3. **Configuration File Not Found**
   ```python
   # Check configuration path
   info = config.get_config_info()
   print(f"Config path: {info['config_path']}")
   
   # Create default configuration
   config.save_config()
   ```

4. **Legacy Name Issues**
   ```python
   # Check legacy mapping
   canonical = config.get_legacy_mapping("OLD_NAME")
   if canonical:
       print(f"OLD_NAME maps to {canonical}")
   else:
       print("No mapping found")
   ```

### Debug Configuration

```python
# Enable debug logging
import logging
logging.getLogger("src.indicators.config_manager").setLevel(logging.DEBUG)

# Get detailed configuration info
info = config.get_config_info()
for key, value in info.items():
    print(f"{key}: {value}")
```

## Configuration Examples

### Complete Configuration File

```json
{
  "version": "2.0",
  "default_parameters": {
    "rsi": {"timeperiod": 14},
    "macd": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
    "bbands": {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
    "stoch": {"fastk_period": 14, "slowk_period": 3, "slowd_period": 3},
    "adx": {"timeperiod": 14},
    "sma": {"timeperiod": 20},
    "ema": {"timeperiod": 20},
    "cci": {"timeperiod": 14},
    "roc": {"timeperiod": 10},
    "mfi": {"timeperiod": 14},
    "williams_r": {"timeperiod": 14},
    "atr": {"timeperiod": 14},
    "aroon": {"timeperiod": 14},
    "sar": {"acceleration": 0.02, "maximum": 0.20},
    "super_trend": {"length": 10, "multiplier": 3.0}
  },
  "presets": {
    "default": {
      "description": "Default parameters from registry",
      "parameters": {}
    },
    "conservative": {
      "description": "Conservative trading parameters with longer periods",
      "parameters": {
        "rsi": {"timeperiod": 21},
        "macd": {"fastperiod": 15, "slowperiod": 30, "signalperiod": 12},
        "bbands": {"timeperiod": 25, "nbdevup": 2.5, "nbdevdn": 2.5},
        "stoch": {"fastk_period": 21, "slowk_period": 5, "slowd_period": 5}
      }
    },
    "aggressive": {
      "description": "Aggressive trading parameters with shorter periods",
      "parameters": {
        "rsi": {"timeperiod": 7},
        "macd": {"fastperiod": 8, "slowperiod": 17, "signalperiod": 5},
        "bbands": {"timeperiod": 15, "nbdevup": 1.5, "nbdevdn": 1.5},
        "stoch": {"fastk_period": 7, "slowk_period": 2, "slowd_period": 2}
      }
    },
    "day_trading": {
      "description": "Day trading optimized parameters",
      "parameters": {
        "rsi": {"timeperiod": 5},
        "macd": {"fastperiod": 5, "slowperiod": 13, "signalperiod": 3},
        "bbands": {"timeperiod": 10, "nbdevup": 1.8, "nbdevdn": 1.8},
        "stoch": {"fastk_period": 5, "slowk_period": 1, "slowd_period": 1}
      }
    },
    "swing_trading": {
      "description": "Custom swing trading parameters",
      "parameters": {
        "rsi": {"timeperiod": 18},
        "macd": {"fastperiod": 10, "slowperiod": 22, "signalperiod": 8},
        "bbands": {"timeperiod": 18, "nbdevup": 2.2, "nbdevdn": 2.2}
      }
    }
  },
  "legacy_mappings": {
    "RSI": "rsi",
    "MACD": "macd",
    "MACD_SIGNAL": "macd",
    "MACD_HISTOGRAM": "macd",
    "BB_UPPER": "bbands",
    "BB_MIDDLE": "bbands",
    "BB_LOWER": "bbands",
    "SMA_FAST": "sma",
    "SMA_SLOW": "sma",
    "EMA_FAST": "ema",
    "EMA_SLOW": "ema",
    "ADX": "adx",
    "STOCH_K": "stoch",
    "STOCH_D": "stoch"
  }
}
```

### Python Configuration Example

```python
from src.indicators.config_manager import get_config_manager

# Get configuration manager
config = get_config_manager()

# Set up for swing trading
config.set_preset("conservative")

# Fine-tune specific indicators
config.set_parameter_override("rsi", "timeperiod", 18)
config.set_parameter_override("macd", "fastperiod", 10)
config.set_parameter_override("macd", "slowperiod", 22)

# Verify configuration
rsi_params = config.get_parameters("rsi")
macd_params = config.get_parameters("macd")

print(f"RSI parameters: {rsi_params}")
print(f"MACD parameters: {macd_params}")

# Save custom configuration
config.save_config("custom_swing_config.json")
```

This configuration system provides maximum flexibility while maintaining simplicity and backward compatibility. The preset system allows quick switching between trading styles, while runtime overrides enable fine-tuning for specific strategies.