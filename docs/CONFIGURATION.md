# Configuration Management System

## Overview

The crypto trading platform implements a comprehensive **Centralized Configuration Management System** that provides unified configuration schemas, environment-specific configs, schema validation, and configuration templates. This system addresses the issues of scattered configurations, lack of validation, and difficulty managing environment-specific settings.

## 🎯 **Problem Statement**

The original configuration system had three major issues:

1. **Configuration scattered across multiple files** - Configs were spread across `config/trading/`, `config/optimizer/`, `config/plotter/`, etc.
2. **No schema validation** - Only basic JSON validation existed, no comprehensive schema validation
3. **Hard to manage environment-specific configs** - No clear separation between dev/staging/prod environments

## ✅ **Solution Overview**

Implemented a comprehensive **Centralized Configuration Management System** that addresses all three issues:

### Core Features

1. **Unified Configuration Schema** - All configurations use Pydantic-based schemas with validation
2. **Environment-Specific Configs** - Separate configurations for dev/staging/prod environments
3. **Configuration Registry** - Centralized tracking and discovery of all configurations
4. **Schema Validation** - Comprehensive validation with detailed error messages
5. **Configuration Templates** - Pre-defined templates for common use cases
6. **Hot-reload Support** - Automatic reloading during development

---

## Table of Contents

1. [Architecture](#architecture)
2. [Quick Start](#quick-start)
3. [Configuration Types](#configuration-types)
4. [Schema Validation](#schema-validation)
5. [Environment Management](#environment-management)
6. [Configuration Templates](#configuration-templates)
7. [Configuration Registry](#configuration-registry)
8. [Hot-Reload Support](#hot-reload-support)
9. [Migration Support](#migration-support)
10. [Best Practices](#best-practices)
11. [API Reference](#api-reference)
12. [Troubleshooting](#troubleshooting)

---

## Architecture

```
config/
├── development/          # Development environment configs
│   ├── trading/
│   ├── optimizer/
│   └── data/
├── staging/             # Staging environment configs
├── production/          # Production environment configs
├── templates/           # Configuration templates
└── schemas/            # Schema definitions
```

### Key Components

- `src/config/schemas.py` - Pydantic-based schemas for all config types
- `src/config/config_manager.py` - Main configuration manager
- `src/config/registry.py` - Configuration registry and discovery
- `src/config/templates.py` - Pre-defined configuration templates

---

## Quick Start

### 1. Initialize Configuration Manager

```python
from src.config import ConfigManager

# Initialize with environment
config_manager = ConfigManager(
    config_dir="config",
    environment="development"
)
```

### 2. Create Configuration from Template

```python
# Create a paper trading bot configuration
trading_config = config_manager.create_config(
    "trading_paper",
    bot_id="my_paper_bot",
    symbol="BTCUSDT",
    initial_balance=5000.0
)

# Save the configuration
config_path = config_manager.save_config(trading_config, "my_paper_bot.json")
```

### 3. Load and Use Configuration

```python
# Load a specific configuration
config = config_manager.get_config("my_paper_bot")

# Get all trading configurations
trading_configs = config_manager.get_trading_configs()

# Search configurations
results = config_manager.search_configs("BTCUSDT", "trading")
```

---

## Configuration Types

### 1. Trading Configuration

Trading configurations define bot behavior, risk management, and strategy parameters.

```json
{
    "environment": "development",
    "version": "1.0.0",
    "description": "Paper trading bot for strategy testing",
    
    "bot_id": "paper_bot_001",
    "broker": {
        "type": "binance_paper",
        "initial_balance": 10000.0,
        "commission": 0.001
    },
    "trading": {
        "symbol": "BTCUSDT",
        "position_size": 0.1,
        "max_positions": 1,
        "max_drawdown_pct": 20.0,
        "max_exposure": 1.0
    },
    "data": {
        "data_source": "binance",
        "symbol": "BTCUSDT",
        "interval": "1h",
        "lookback_bars": 1000,
        "retry_interval": 60,
        "testnet": true
    },
    "strategy": {
        "type": "custom",
        "entry_logic": {
            "name": "RSIBBVolumeEntryMixin",
            "params": {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "bb_period": 20,
                "bb_dev": 2.0
            }
        },
        "exit_logic": {
            "name": "RSIBBExitMixin",
            "params": {
                "rsi_period": 14,
                "rsi_overbought": 70
            }
        }
    },
    "risk_management": {
        "stop_loss_pct": 5.0,
        "take_profit_pct": 10.0,
        "max_daily_trades": 10,
        "max_daily_loss": 50.0,
        "max_drawdown_pct": 20.0,
        "max_exposure": 1.0,
        "trailing_stop": {
            "enabled": false,
            "activation_pct": 3.0,
            "trailing_pct": 2.0
        }
    },
    "logging": {
        "level": "INFO",
        "save_trades": true,
        "save_equity_curve": true,
        "log_file": "logs/paper/trading_bot_paper.log"
    },
    "scheduling": {
        "enabled": false,
        "start_time": "09:00",
        "end_time": "17:00",
        "timezone": "UTC",
        "trading_days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
    },
    "performance": {
        "target_sharpe_ratio": 1.0,
        "target_win_rate": 60.0,
        "target_profit_factor": 1.5,
        "max_consecutive_losses": 5,
        "performance_check_interval": 24
    },
    "notifications": {
        "enabled": true,
        "telegram": {
            "enabled": false,
            "notify_on": ["trade_entry", "trade_exit", "error"]
        },
        "email": {
            "enabled": false,
            "notify_on": ["trade_entry", "trade_exit", "error"]
        }
    }
}
```

### 2. Optimizer Configuration

Optimizer configurations define parameter optimization settings.

```json
{
    "environment": "development",
    "version": "1.0.0",
    "description": "Basic optimization configuration",
    
    "optimizer_type": "optuna",
    "initial_capital": 1000.0,
    "commission": 0.001,
    "risk_free_rate": 0.01,
    "n_trials": 100,
    "n_jobs": 1,
    "position_size": 0.1,
    
    "entry_strategies": [
        {
            "name": "RSIBBVolumeEntryMixin",
            "params": {
                "rsi_period": {"type": "int", "low": 5, "high": 30, "default": 14},
                "bb_period": {"type": "int", "low": 10, "high": 50, "default": 20},
                "bb_dev": {"type": "float", "low": 1.0, "high": 3.0, "default": 2.0}
            }
        }
    ],
    "exit_strategies": [
        {
            "name": "RSIBBExitMixin",
            "params": {
                "rsi_period": {"type": "int", "low": 5, "high": 30, "default": 14},
                "rsi_overbought": {"type": "int", "low": 60, "high": 90, "default": 70}
            }
        }
    ],
    
    "plot": true,
    "save_trades": true,
    "output_dir": "results"
}
```

### 3. Data Configuration

Data configurations define data feed settings.

```json
{
    "environment": "development",
    "version": "1.0.0",
    "description": "Binance data feed configuration",
    
    "data_source": "binance",
    "symbol": "BTCUSDT",
    "interval": "1h",
    "lookback_bars": 1000,
    "retry_interval": 60,
    "testnet": true
}
```

---

## Schema Validation

### Comprehensive Validation System

**Problem**: No schema validation led to runtime errors and inconsistent configurations.

**Solution**:
- Pydantic-based validation with detailed error messages
- Type checking, range validation, and cross-field validation
- Automatic validation on configuration load and save

### Validation Features

- **Type Validation**: Ensures correct data types for all fields
- **Range Validation**: Numeric values must be within valid ranges
- **Enum Validation**: String fields must match allowed values
- **Cross-Field Validation**: Related fields must be consistent
- **Custom Validation**: Business logic validation rules

### Example Validation

```python
@validator('take_profit_pct')
def validate_take_profit(cls, v, values):
    """Ensure take profit is greater than stop loss"""
    if 'stop_loss_pct' in values and v <= values['stop_loss_pct']:
        raise ValueError("Take profit must be greater than stop loss")
    return v

@validator('position_size')
def validate_position_size(cls, v):
    """Ensure position size is between 0 and 1"""
    if not 0 < v <= 1:
        raise ValueError("Position size must be between 0 and 1")
    return v
```

### Validation Error Messages

```python
# Clear, actionable error messages
try:
    config = TradingConfig(**invalid_data)
except ValidationError as e:
    print(e.json())
    # Output: Detailed error information with field names and suggestions
```

---

## Environment Management

### Environment-Specific Configuration Management

**Problem**: No clear separation between development, staging, and production environments.

**Solution**:
- Environment-specific configuration directories
- Environment variable support
- Different default settings per environment
- Automatic environment detection

### Environment Structure

```
config/
├── development/          # Development environment configs
│   ├── trading/
│   ├── optimizer/
│   └── data/
├── staging/             # Staging environment configs
├── production/          # Production environment configs
└── templates/           # Configuration templates
```

### Environment Features

- **Development**: Debug logging, hot-reload, testnet APIs
- **Staging**: Info logging, production-like settings, test data
- **Production**: Warning logging, full security, live APIs

### Environment Configuration

```python
# Set environment via environment variable
export TRADING_ENV=production

# Or specify in code
config_manager = ConfigManager(environment="staging")

# Get environment-specific configuration
env_config = config_manager.get_environment_config("database_url")
```

---

## Configuration Templates

### Pre-defined Templates

Pre-defined templates for common use cases:

```python
# Create from template
config = config_manager.create_config(
    "trading_paper",
    bot_id="my_bot",
    symbol="BTCUSDT",
    initial_balance=5000.0
)
```

### Available Templates

- **`trading_paper`** - Paper trading with safe defaults
- **`trading_live`** - Live trading with full risk management
- **`trading_dev`** - Development trading bot with debugging enabled
- **`optimizer_basic`** - Basic optimization configuration
- **`optimizer_advanced`** - Advanced optimization with extended parameters
- **`data_binance`** - Binance data feed configuration
- **`data_yahoo`** - Yahoo Finance data feed configuration
- **`data_ibkr`** - IBKR data feed configuration
- **`env_development`** - Development environment settings
- **`env_staging`** - Staging environment settings
- **`env_production`** - Production environment settings

### Template Usage

```python
# List available templates
templates = config_manager.templates.list_templates()
for template in templates:
    desc = config_manager.templates.get_template_description(template)
    print(f"{template}: {desc}")

# Create configuration from template
config = config_manager.create_config(
    "trading_live",
    bot_id="live_bot_001",
    symbol="ETHUSDT",
    initial_balance=1000.0,
    api_key="YOUR_API_KEY",
    api_secret="YOUR_API_SECRET"
)
```

---

## Configuration Registry

### Centralized Tracking and Discovery

Centralized tracking and discovery of all configurations:

```python
# Get all trading configurations
trading_configs = config_manager.get_trading_configs()

# Search configurations
results = config_manager.search_configs("BTCUSDT")

# Get configuration metadata
metadata = config_manager.registry.get_config_metadata("my_bot")

# List all configurations by type
configs_by_type = config_manager.list_configs()
```

### Registry Features

- **Configuration Discovery**: Automatically find and load configurations
- **Metadata Tracking**: Track configuration creation, updates, and usage
- **Type-based Organization**: Group configurations by type (trading, optimizer, data)
- **Search and Filter**: Find configurations by criteria

### Registry Operations

```python
# Get configuration summary
summary = config_manager.get_config_summary()
print(f"Total configs: {summary['total_configs']}")
print(f"Configs by type: {summary['configs_by_type']}")

# Validate configuration file
is_valid, errors = config_manager.validate_config_file("config.json")
if not is_valid:
    print(f"Validation errors: {errors}")
```

---

## Hot-Reload Support

### Automatic Configuration Reloading

Automatic configuration reloading during development:

```python
# Enable hot-reload
config_manager.enable_hot_reload(True)

# Configurations automatically reload when files change
```

### Hot-Reload Features

- **File System Monitoring**: Watch for configuration file changes
- **Automatic Reloading**: Reload configurations when files are modified
- **Development Friendly**: Perfect for development and testing
- **Configurable**: Enable/disable as needed

### Hot-Reload Usage

```python
# Enable hot-reload for development
if environment == "development":
    config_manager.enable_hot_reload(True)

# Disable for production
if environment == "production":
    config_manager.enable_hot_reload(False)

# Manual reload
config_manager.reload_config("specific_config.json")
```

---

## Migration Support

### Automatic Migration from Old System

Automatic migration from old scattered configuration system:

```bash
python src/config/migrate_configs.py --old-dir config --new-dir config_new --backup
```

### Migration Features

- **Automatic Discovery**: Find and migrate existing configurations
- **Schema Validation**: Validate migrated configurations
- **Backup Creation**: Create backups before migration
- **Incremental Migration**: Migrate configurations incrementally

### Migration Process

```python
from src.config.migrate_configs import migrate_configs

# Migrate configurations
migrate_configs(
    old_dir="config",
    new_dir="config_new",
    environment="development",
    backup=True
)
```

---

## Best Practices

### 1. Configuration Organization

- **Use Environment-Specific Directories**: Separate dev/staging/prod configs
- **Use Descriptive Names**: Clear, descriptive configuration names
- **Group Related Configs**: Keep related configurations together
- **Version Control**: Track configuration changes in version control

### 2. Schema Design

- **Use Strong Types**: Define specific types for all fields
- **Add Validation**: Include business logic validation
- **Provide Defaults**: Set sensible default values
- **Document Fields**: Add clear field descriptions

### 3. Template Usage

- **Start with Templates**: Use templates for common configurations
- **Customize Carefully**: Modify templates only when necessary
- **Test Changes**: Validate all configuration changes
- **Document Customizations**: Document any template customizations

### 4. Environment Management

- **Use Environment Variables**: For sensitive configuration
- **Separate Environments**: Clear separation between environments
- **Test Configurations**: Test configurations in staging first
- **Monitor Changes**: Track configuration changes across environments

### 5. Security Considerations

- **Secure Sensitive Data**: Never commit API keys or secrets
- **Use Environment Variables**: For production secrets
- **Validate Inputs**: Validate all configuration inputs
- **Access Control**: Control access to configuration files

---

## API Reference

### ConfigManager

```python
class ConfigManager:
    def __init__(self, config_dir: str = "config", environment: str = None)
    def get_config(self, config_id: str) -> Optional[Any]
    def get_config_by_type(self, config_type: str) -> List[Any]
    def create_config(self, config_type: str, **kwargs) -> Any
    def save_config(self, config: Any, filename: str = None) -> str
    def delete_config(self, config_id: str) -> bool
    def reload_config(self, config_path: str = None)
    def enable_hot_reload(self, enabled: bool = True)
    def get_environment_config(self, key: str, default: Any = None) -> Any
    def list_configs(self) -> Dict[str, List[str]]
    def validate_config_file(self, config_path: str) -> tuple[bool, List[str]]
    def get_config_summary(self) -> Dict[str, Any]
```

### Configuration Schemas

```python
class TradingConfig(BaseModel):
    environment: Environment
    version: str
    bot_id: str
    broker: BrokerConfig
    trading: TradingParams
    data: DataConfig
    strategy: StrategyConfig
    risk_management: RiskManagementConfig
    logging: LoggingConfig
    notifications: NotificationConfig

class OptimizerConfig(BaseModel):
    environment: Environment
    version: str
    optimizer_type: str
    initial_capital: float
    commission: float
    n_trials: int
    entry_strategies: List[StrategyConfig]
    exit_strategies: List[StrategyConfig]

class DataConfig(BaseModel):
    environment: Environment
    version: str
    data_source: DataSourceType
    symbol: str
    interval: str
    lookback_bars: int
    retry_interval: int
```

---

## Troubleshooting

### Common Issues

1. **Configuration Not Found**
   - Check file path and environment
   - Verify configuration file exists
   - Check file permissions

2. **Validation Errors**
   - Review error messages carefully
   - Check field types and ranges
   - Verify required fields are present

3. **Environment Issues**
   - Check environment variable settings
   - Verify environment directory exists
   - Check environment-specific defaults

4. **Hot-Reload Not Working**
   - Verify hot-reload is enabled
   - Check file system permissions
   - Monitor file change events

### Debug Mode

```python
import logging
logging.getLogger('src.config').setLevel(logging.DEBUG)

# Enable detailed logging for configuration operations
```

### Validation Debugging

```python
# Validate configuration with detailed errors
is_valid, errors = config_manager.validate_config_file("config.json")
if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

---

## Integration Examples

### Trading Bot Integration

```python
class LiveTradingBot:
    def __init__(self, config_id: str):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config(config_id)
        
        # Initialize components with configuration
        self.broker = self._create_broker(self.config.broker)
        self.strategy = self._create_strategy(self.config.strategy)
        self.data_feed = self._create_data_feed(self.config.data)
    
    def _create_broker(self, broker_config):
        # Create broker based on configuration
        pass
    
    def _create_strategy(self, strategy_config):
        # Create strategy based on configuration
        pass
```

### Optimizer Integration

```python
def run_optimization(config_id: str):
    config_manager = ConfigManager()
    config = config_manager.get_config(config_id)
    
    # Run optimization with configuration
    optimizer = Optimizer(config)
    results = optimizer.optimize()
    
    return results
```

### Data Feed Integration

```python
def create_data_feed(config_id: str):
    config_manager = ConfigManager()
    config = config_manager.get_config(config_id)
    
    # Create data feed based on configuration
    data_feed = DataFeedFactory.create(config)
    
    return data_feed
```

---

## Benefits Delivered

### Reliability
- Comprehensive schema validation prevents configuration errors
- Environment-specific configurations ensure proper deployment
- Configuration templates reduce setup errors

### Maintainability
- Centralized configuration management
- Clear separation of concerns
- Easy configuration discovery and updates

### Developer Experience
- Hot-reload support for development
- Clear error messages for validation issues
- Pre-defined templates for common use cases

### Production Readiness
- Environment-specific configurations
- Secure handling of sensitive data
- Comprehensive validation and error handling

---

*Last Updated: December 2024*
*Version: 1.0* 