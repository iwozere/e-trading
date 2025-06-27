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
- ConfigManager: Main configuration manager
- ConfigSchema: Base schema for all configurations
- TradingConfig: Trading bot configuration schema
- OptimizerConfig: Optimization configuration schema
- DataConfig: Data feed configuration schema
- NotificationConfig: Notification configuration schema
"""

from .config_manager import ConfigManager
from .schemas import (
    ConfigSchema,
    TradingConfig,
    OptimizerConfig,
    DataConfig,
    NotificationConfig,
    RiskManagementConfig,
    LoggingConfig,
    SchedulingConfig,
    PerformanceConfig
)
from .registry import ConfigRegistry
from .templates import ConfigTemplates

__all__ = [
    'ConfigManager',
    'ConfigSchema',
    'TradingConfig',
    'OptimizerConfig',
    'DataConfig',
    'NotificationConfig',
    'RiskManagementConfig',
    'LoggingConfig',
    'SchedulingConfig',
    'PerformanceConfig',
    'ConfigRegistry',
    'ConfigTemplates'
] 