import os
import importlib
import inspect
from typing import Dict, Optional
from src.notification.logger import setup_logger
from .base import SignalPlugin

_logger = setup_logger(__name__)

class PluginRegistry:
    """
    Registry for loading and retrieving SignalPlugins for the alert configuration system.
    """

    def __init__(self):
        self._plugins: Dict[str, SignalPlugin] = {}

    def discover_plugins(self, package_path: str = "src.common.alerts.plugins") -> None:
        """
        Dynamically discover and instantiate all SignalPlugin instances in a specific package.
        """
        try:
            # We assume plugins are in the same directory as this file
            dir_path = os.path.dirname(os.path.abspath(__file__))
            
            for file in os.listdir(dir_path):
                if file.endswith('.py') and not file.startswith('__') and file not in ('base.py', 'registry.py'):
                    module_name = file[:-3]
                    full_module_name = f"{package_path}.{module_name}"
                    
                    try:
                        module = importlib.import_module(full_module_name)
                        for _, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and issubclass(obj, SignalPlugin) and obj is not SignalPlugin:
                                plugin_instance = obj()
                                self.register(plugin_instance)
                    except Exception as e:
                        _logger.error("Failed to load plugin module %s: %s", full_module_name, e)

        except Exception as e:
            _logger.error("Failed to discover plugins: %s", e)

    def register(self, plugin: SignalPlugin) -> None:
        """Register a plugin instance manually."""
        if plugin.name in self._plugins:
            _logger.warning("Plugin with name %s is already registered, overwriting.", plugin.name)
        
        self._plugins[plugin.name] = plugin
        _logger.info("Successfully registered SignalPlugin: %s", plugin.name)

    def get_plugin(self, name: str) -> Optional[SignalPlugin]:
        """Retrieve a specific plugin instance."""
        return self._plugins.get(name)

    def get_all_schemas(self) -> Dict[str, Dict]:
        """
        Returns an aggregated dictionary of { plugin_name: schema_dict }
        which can be injected into the main AlertSchemaValidator.
        """
        return {name: plugin.schema() for name, plugin in self._plugins.items()}

# Global singleton registry instance
registry = PluginRegistry()
