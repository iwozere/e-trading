from .base import SignalPlugin
from .registry import registry

# Automatically discover and register all plugins in this directory
registry.discover_plugins()

__all__ = ['SignalPlugin', 'registry']
