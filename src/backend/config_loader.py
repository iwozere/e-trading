"""
Screener Sets Configuration Loader

Loads and manages screener sets configuration from YAML files.
Provides helper functions for accessing screener sets and their tickers.
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

_logger = logging.getLogger(__name__)


class ScreenerConfigLoader:
    """Loads and manages screener sets configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config loader.

        Args:
            config_path: Path to the screener_sets.yml file. If None, uses default path.
        """
        if config_path is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "screener_sets.yml"

        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None
        self._last_modified: Optional[float] = None

        # Load configuration on initialization
        self.reload()

    def reload(self) -> None:
        """Reload the configuration from file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Screener config file not found: {self.config_path}")

            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)

            self._last_modified = self.config_path.stat().st_mtime
            _logger.info(f"Loaded screener config from {self.config_path}")

        except Exception as e:
            _logger.error(f"Failed to load screener config: {e}")
            raise

    def _check_reload(self) -> None:
        """Check if config file has been modified and reload if necessary."""
        if self._last_modified is None:
            return

        try:
            current_mtime = self.config_path.stat().st_mtime
            if current_mtime > self._last_modified:
                _logger.info("Screener config file modified, reloading...")
                self.reload()
        except OSError:
            # File might have been deleted or moved
            pass

    def get_tickers(self, set_name: str) -> List[str]:
        """
        Get tickers for a specific screener set.

        Args:
            set_name: Name of the screener set

        Returns:
            List of ticker symbols

        Raises:
            KeyError: If screener set not found
        """
        self._check_reload()

        if not self._config or 'screener_sets' not in self._config:
            raise KeyError("No screener sets found in configuration")

        screener_sets = self._config['screener_sets']
        if set_name not in screener_sets:
            available_sets = list(screener_sets.keys())
            raise KeyError(f"Screener set '{set_name}' not found. Available sets: {available_sets}")

        return screener_sets[set_name]['tickers']

    def get_set_info(self, set_name: str) -> Dict[str, Any]:
        """
        Get complete information for a specific screener set.

        Args:
            set_name: Name of the screener set

        Returns:
            Dictionary with set information (name, description, categories, tickers)

        Raises:
            KeyError: If screener set not found
        """
        self._check_reload()

        if not self._config or 'screener_sets' not in self._config:
            raise KeyError("No screener sets found in configuration")

        screener_sets = self._config['screener_sets']
        if set_name not in screener_sets:
            available_sets = list(screener_sets.keys())
            raise KeyError(f"Screener set '{set_name}' not found. Available sets: {available_sets}")

        set_info = screener_sets[set_name].copy()
        set_info['ticker_count'] = len(set_info['tickers'])
        return set_info

    def list_available_sets(self) -> List[str]:
        """
        List all available screener set names.

        Returns:
            List of screener set names
        """
        self._check_reload()

        if not self._config or 'screener_sets' not in self._config:
            return []

        return list(self._config['screener_sets'].keys())

    def list_sets_by_category(self, category: str) -> List[str]:
        """
        List screener sets that contain a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of screener set names that contain the category
        """
        self._check_reload()

        if not self._config or 'screener_sets' not in self._config:
            return []

        matching_sets = []
        for set_name, set_info in self._config['screener_sets'].items():
            if 'categories' in set_info and category in set_info['categories']:
                matching_sets.append(set_name)

        return matching_sets

    def get_all_sets_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information for all screener sets.

        Returns:
            Dictionary mapping set names to their information
        """
        self._check_reload()

        if not self._config or 'screener_sets' not in self._config:
            return {}

        result = {}
        for set_name, set_info in self._config['screener_sets'].items():
            result[set_name] = set_info.copy()
            result[set_name]['ticker_count'] = len(set_info['tickers'])

        return result

    def get_default_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get a default screener configuration.

        Args:
            config_name: Name of the default configuration

        Returns:
            Dictionary with configuration parameters

        Raises:
            KeyError: If configuration not found
        """
        self._check_reload()

        if not self._config or 'default_configs' not in self._config:
            raise KeyError("No default configurations found")

        default_configs = self._config['default_configs']
        if config_name not in default_configs:
            available_configs = list(default_configs.keys())
            raise KeyError(f"Default config '{config_name}' not found. Available configs: {available_configs}")

        return default_configs[config_name]

    def list_default_configs(self) -> List[str]:
        """
        List all available default configuration names.

        Returns:
            List of default configuration names
        """
        self._check_reload()

        if not self._config or 'default_configs' not in self._config:
            return []

        return list(self._config['default_configs'].keys())

    def validate_set_name(self, set_name: str) -> bool:
        """
        Validate if a screener set name exists.

        Args:
            set_name: Name to validate

        Returns:
            True if set exists, False otherwise
        """
        try:
            self.get_tickers(set_name)
            return True
        except KeyError:
            return False

    def get_set_categories(self, set_name: str) -> List[str]:
        """
        Get categories for a specific screener set.

        Args:
            set_name: Name of the screener set

        Returns:
            List of categories

        Raises:
            KeyError: If screener set not found
        """
        set_info = self.get_set_info(set_name)
        return set_info.get('categories', [])

    def search_sets(self, query: str) -> List[str]:
        """
        Search screener sets by name or description.

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching screener set names
        """
        self._check_reload()

        if not self._config or 'screener_sets' not in self._config:
            return []

        query_lower = query.lower()
        matching_sets = []

        for set_name, set_info in self._config['screener_sets'].items():
            # Search in name
            if query_lower in set_name.lower():
                matching_sets.append(set_name)
                continue

            # Search in description
            if 'description' in set_info and query_lower in set_info['description'].lower():
                matching_sets.append(set_name)
                continue

            # Search in categories
            if 'categories' in set_info:
                for category in set_info['categories']:
                    if query_lower in category.lower():
                        matching_sets.append(set_name)
                        break

        return matching_sets


# Global instance for easy access
_screener_config: Optional[ScreenerConfigLoader] = None


def get_screener_config() -> ScreenerConfigLoader:
    """
    Get the global screener config instance.

    Returns:
        ScreenerConfigLoader instance
    """
    global _screener_config
    if _screener_config is None:
        _screener_config = ScreenerConfigLoader()
    return _screener_config


def get_tickers(set_name: str) -> List[str]:
    """
    Convenience function to get tickers for a screener set.

    Args:
        set_name: Name of the screener set

    Returns:
        List of ticker symbols
    """
    return get_screener_config().get_tickers(set_name)


def list_available_sets() -> List[str]:
    """
    Convenience function to list all available screener sets.

    Returns:
        List of screener set names
    """
    return get_screener_config().list_available_sets()


def validate_set_name(set_name: str) -> bool:
    """
    Convenience function to validate a screener set name.

    Args:
        set_name: Name to validate

    Returns:
        True if set exists, False otherwise
    """
    return get_screener_config().validate_set_name(set_name)

