"""
Notification Channel Plugins Package

This package provides the plugin system for notification channels.
It includes base classes, plugin loading, and configuration validation.
"""

from src.notification.channels.config import ConfigValidationError
from src.notification.channels.base import MessageContent, channel_registry
from src.notification.channels.loader import load_all_channels

__all__ = [
    "ConfigValidationError",
    "MessageContent",
    "channel_registry",
    "load_all_channels",
]
