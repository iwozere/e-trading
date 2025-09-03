"""
Centralized Configuration Management System
==========================================

This module provides a unified configuration management system with:
- Pydantic-based schema validation
- Environment-specific configurations (dev/staging/prod)
- Configuration registry and discovery
- Hot-reload support for development
- Template-based configuration generation

Classes:
- ConfigManager: Main configuration manager (legacy)
- ConfigSchema: Base schema for all configurations (legacy)
- TradingConfig: Trading bot configuration schema (legacy)
- OptimizerConfig: Optimization configuration schema (legacy)
- DataConfig: Data feed configuration schema (legacy)

New Simplified Models:
- TradingBotConfig: Simplified trading bot configuration
- OptimizerConfig: Simplified optimizer configuration
- DataConfig: Simplified data configuration
"""

# Legacy imports (for backward compatibility)
from src.config.config_manager import ConfigManager
from src.model.schemas import (
    ConfigSchema,
    TradingConfig,
    OptimizerConfig as LegacyOptimizerConfig,
    DataConfig as LegacyDataConfig,
    NotificationConfig,
    RiskManagementConfig,
    LoggingConfig,
    SchedulingConfig,
    PerformanceConfig
)
from src.config.registry import ConfigRegistry
from src.config.templates import ConfigTemplates

# New simplified imports
from src.model.config_models import (
    TradingBotConfig,
    OptimizerConfig,
    DataConfig,
    Environment,
    BrokerType,
    DataSourceType,
    StrategyType
)
from src.config.config_loader import (
    load_config,
    load_optimizer_config,
    load_data_config,
    save_config,
    validate_config_file,
    create_sample_config,
    convert_old_config
)

__all__ = [
    # Legacy exports (for backward compatibility)
    'ConfigManager',
    'ConfigSchema',
    'TradingConfig',
    'LegacyOptimizerConfig',
    'LegacyDataConfig',
    'NotificationConfig',
    'RiskManagementConfig',
    'LoggingConfig',
    'SchedulingConfig',
    'PerformanceConfig',
    'ConfigRegistry',
    'ConfigTemplates',

    # New simplified exports
    'TradingBotConfig',
    'OptimizerConfig',
    'DataConfig',
    'Environment',
    'BrokerType',
    'DataSourceType',
    'StrategyType',
    'load_config',
    'load_optimizer_config',
    'load_data_config',
    'save_config',
    'validate_config_file',
    'create_sample_config',
    'convert_old_config'
]
