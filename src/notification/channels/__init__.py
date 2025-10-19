"""
Notification Channel Plugins Package

This package provides the plugin system for notification channels.
It includes base classes, plugin loading, and configuration validation.
"""

from .base import (
    NotificationChannel,
    DeliveryResult,
    ChannelHealth,
    MessageContent,
    DeliveryStatus,
    ChannelHealthStatus,
    ChannelRegistry,
    channel_registry
)

from .loader import (
    PluginLoader,
    plugin_loader,
    load_all_channels,
    register_external_plugin,
    get_available_channels
)

from .config import (
    ConfigValidator,
    ValidationRule,
    ConfigValidationError,
    CommonValidationRules
)

__all__ = [
    # Base classes and enums
    'NotificationChannel',
    'DeliveryResult',
    'ChannelHealth',
    'MessageContent',
    'DeliveryStatus',
    'ChannelHealthStatus',

    # Registry
    'ChannelRegistry',
    'channel_registry',

    # Plugin loading
    'PluginLoader',
    'plugin_loader',
    'load_all_channels',
    'register_external_plugin',
    'get_available_channels',

    # Configuration validation
    'ConfigValidator',
    'ValidationRule',
    'ConfigValidationError',
    'CommonValidationRules'
]