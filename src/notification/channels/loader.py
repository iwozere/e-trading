"""
Channel Plugin Discovery and Loading

Automatic discovery and loading of notification channel plugins.
Supports both built-in and external plugins.
"""

import importlib
import importlib.util
import inspect
import pkgutil
import sys
from typing import Dict, List, Optional, Type
from pathlib import Path

from src.notification.channels.base import NotificationChannel, channel_registry
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class PluginLoader:
    """
    Plugin loader for notification channels.

    Discovers and loads channel plugins from various sources.
    """

    def __init__(self):
        """Initialize the plugin loader."""
        self._loaded_plugins: Dict[str, Type[NotificationChannel]] = {}
        self._plugin_paths: List[str] = []

    def add_plugin_path(self, path: str) -> None:
        """
        Add a path to search for plugins.

        Args:
            path: Path to search for plugin modules
        """
        if path not in self._plugin_paths:
            self._plugin_paths.append(path)
            _logger.info("Added plugin search path: %s", path)

    def discover_builtin_plugins(self) -> List[str]:
        """
        Discover built-in channel plugins.

        Returns:
            List of discovered plugin module names
        """
        plugins = []

        # Look for plugins in the channels package
        try:
            import src.notification.channels as channels_package
            package_path = Path(channels_package.__file__).parent

            for module_info in pkgutil.iter_modules([str(package_path)]):
                module_name = module_info.name

                # Skip base modules and private modules
                if module_name in ['base', 'loader', '__init__'] or module_name.startswith('_'):
                    continue

                plugins.append(f"src.notification.channels.{module_name}")

        except Exception:
            _logger.exception("Error discovering built-in plugins:")

        return plugins

    def load_plugin_module(self, module_name: str) -> Optional[Type[NotificationChannel]]:
        """
        Load a plugin from a module.

        Args:
            module_name: Full module name to load

        Returns:
            Channel class if found, None otherwise
        """
        try:
            module = importlib.import_module(module_name)

            # Look for classes that inherit from NotificationChannel
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, NotificationChannel) and
                    obj is not NotificationChannel and
                    not name.startswith('_')):

                    _logger.info("Found channel plugin: %s in module %s", name, module_name)
                    return obj

            _logger.warning("No channel plugin found in module: %s", module_name)
            return None

        except ImportError as e:
            _logger.error("Failed to import plugin module %s: %s", module_name, e)
            return None
        except Exception as e:
            _logger.error("Error loading plugin from module %s: %s", module_name, e)
            return None

    def load_plugin_from_file(self, file_path: str, channel_name: str) -> Optional[Type[NotificationChannel]]:
        """
        Load a plugin from a Python file.

        Args:
            file_path: Path to the Python file
            channel_name: Name to use for the channel

        Returns:
            Channel class if loaded successfully, None otherwise
        """
        try:
            spec = importlib.util.spec_from_file_location(channel_name, file_path)
            if spec is None or spec.loader is None:
                _logger.error("Could not create spec for file: %s", file_path)
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for channel classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, NotificationChannel) and
                    obj is not NotificationChannel):

                    _logger.info("Loaded channel plugin: %s from file %s", name, file_path)
                    return obj

            _logger.warning("No channel plugin found in file: %s", file_path)
            return None

        except Exception as e:
            _logger.error("Error loading plugin from file %s: %s", file_path, e)
            return None

    def register_plugin(self, channel_name: str, channel_class: Type[NotificationChannel]) -> bool:
        """
        Register a loaded plugin with the channel registry.

        Args:
            channel_name: Name for the channel
            channel_class: Channel class to register

        Returns:
            True if registered successfully, False otherwise
        """
        try:
            channel_registry.register_channel(channel_name, channel_class)
            self._loaded_plugins[channel_name] = channel_class
            return True
        except Exception as e:
            _logger.error("Failed to register plugin %s: %s", channel_name, e)
            return False

    def load_all_plugins(self) -> Dict[str, Type[NotificationChannel]]:
        """
        Load all available plugins.

        Returns:
            Dictionary mapping channel names to channel classes
        """
        loaded = {}

        # Load built-in plugins
        builtin_plugins = self.discover_builtin_plugins()
        for module_name in builtin_plugins:
            channel_class = self.load_plugin_module(module_name)
            if channel_class:
                # Extract channel name from module name
                channel_name = module_name.split('.')[-1]
                if self.register_plugin(channel_name, channel_class):
                    loaded[channel_name] = channel_class

        # Load plugins from additional paths
        for plugin_path in self._plugin_paths:
            try:
                path_obj = Path(plugin_path)
                if path_obj.is_file() and path_obj.suffix == '.py':
                    # Single file
                    channel_name = path_obj.stem
                    channel_class = self.load_plugin_from_file(str(path_obj), channel_name)
                    if channel_class and self.register_plugin(channel_name, channel_class):
                        loaded[channel_name] = channel_class

                elif path_obj.is_dir():
                    # Directory with multiple plugins
                    for py_file in path_obj.glob('*.py'):
                        if py_file.name.startswith('_'):
                            continue

                        channel_name = py_file.stem
                        channel_class = self.load_plugin_from_file(str(py_file), channel_name)
                        if channel_class and self.register_plugin(channel_name, channel_class):
                            loaded[channel_name] = channel_class

            except Exception as e:
                _logger.error("Error loading plugins from path %s: %s", plugin_path, e)

        _logger.info("Loaded %s channel plugins: %s", len(loaded), list(loaded.keys()))
        return loaded

    def reload_plugin(self, channel_name: str) -> bool:
        """
        Reload a specific plugin.

        Args:
            channel_name: Name of the channel to reload

        Returns:
            True if reloaded successfully, False otherwise
        """
        if channel_name not in self._loaded_plugins:
            _logger.warning("Plugin %s is not currently loaded", channel_name)
            return False

        try:
            # Unregister current plugin
            channel_registry.unregister_channel(channel_name)

            # Try to reload from built-in plugins first
            builtin_plugins = self.discover_builtin_plugins()
            target_module = f"src.notification.channels.{channel_name}"

            if target_module in builtin_plugins:
                # Reload the module
                if target_module in sys.modules:
                    importlib.reload(sys.modules[target_module])

                channel_class = self.load_plugin_module(target_module)
                if channel_class and self.register_plugin(channel_name, channel_class):
                    _logger.info("Reloaded built-in plugin: %s", channel_name)
                    return True

            # Try to reload from plugin paths
            for plugin_path in self._plugin_paths:
                path_obj = Path(plugin_path)

                if path_obj.is_file() and path_obj.stem == channel_name:
                    channel_class = self.load_plugin_from_file(str(path_obj), channel_name)
                    if channel_class and self.register_plugin(channel_name, channel_class):
                        _logger.info("Reloaded plugin from file: %s", channel_name)
                        return True

                elif path_obj.is_dir():
                    plugin_file = path_obj / f"{channel_name}.py"
                    if plugin_file.exists():
                        channel_class = self.load_plugin_from_file(str(plugin_file), channel_name)
                        if channel_class and self.register_plugin(channel_name, channel_class):
                            _logger.info("Reloaded plugin from directory: %s", channel_name)
                            return True

            _logger.error("Failed to reload plugin: %s", channel_name)
            return False

        except Exception as e:
            _logger.error("Error reloading plugin %s: %s", channel_name, e)
            return False

    def get_loaded_plugins(self) -> Dict[str, Type[NotificationChannel]]:
        """
        Get all currently loaded plugins.

        Returns:
            Dictionary mapping channel names to channel classes
        """
        return self._loaded_plugins.copy()

    def validate_plugin(self, channel_class: Type[NotificationChannel]) -> List[str]:
        """
        Validate a channel plugin implementation.

        Args:
            channel_class: Channel class to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check if it's a proper subclass
        if not issubclass(channel_class, NotificationChannel):
            errors.append("Class must inherit from NotificationChannel")
            return errors

        # Check required methods are implemented
        required_methods = [
            'validate_config',
            'send_message',
            'check_health',
            'get_rate_limit',
            'supports_feature'
        ]

        for method_name in required_methods:
            if not hasattr(channel_class, method_name):
                errors.append(f"Missing required method: {method_name}")
                continue

            method = getattr(channel_class, method_name)
            if not callable(method):
                errors.append(f"Method {method_name} is not callable")
                continue

            # Check if method is properly implemented (not just abstract)
            if hasattr(method, '__isabstractmethod__') and method.__isabstractmethod__:
                errors.append(f"Method {method_name} is not implemented (still abstract)")

        # Check constructor signature
        try:
            sig = inspect.signature(channel_class.__init__)
            params = list(sig.parameters.keys())

            # Should have self, channel_name, config
            if len(params) < 3:
                errors.append("Constructor must accept (self, channel_name, config) parameters")
            elif params[1] != 'channel_name' or params[2] != 'config':
                errors.append("Constructor parameters should be (self, channel_name, config)")

        except Exception as e:
            errors.append(f"Error inspecting constructor: {e}")

        return errors


# Global plugin loader instance
plugin_loader = PluginLoader()


def load_all_channels() -> Dict[str, Type[NotificationChannel]]:
    """
    Convenience function to load all available channel plugins.

    Returns:
        Dictionary mapping channel names to channel classes
    """
    return plugin_loader.load_all_plugins()


def register_external_plugin(channel_name: str, channel_class: Type[NotificationChannel]) -> bool:
    """
    Register an external channel plugin.

    Args:
        channel_name: Name for the channel
        channel_class: Channel class to register

    Returns:
        True if registered successfully, False otherwise
    """
    # Validate the plugin first
    errors = plugin_loader.validate_plugin(channel_class)
    if errors:
        _logger.error("Plugin validation failed for %s: %s", channel_name, errors)
        return False

    return plugin_loader.register_plugin(channel_name, channel_class)


def get_available_channels() -> List[str]:
    """
    Get list of all available channel names.

    Returns:
        List of available channel names
    """
    return channel_registry.list_channels()