"""
Screener Sets Configuration Loader
-----------------------------------

Loads and provides access to predefined screener sets from YAML configuration.
Used by the job scheduler system for batch screening operations.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import lru_cache

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class ScreenerConfigLoader:
    """
    Loads and manages screener sets configuration.

    Provides access to predefined ticker sets and screening criteria
    from YAML configuration files.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the screener config loader.

        Args:
            config_path: Path to screener_sets.yml. If None, uses default location.
        """
        if config_path is None:
            # Default to config/schemas/screener_sets.yml
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "schemas" / "screener_sets.yml"

        self.config_path = config_path
        self._config = None

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self._config is not None:
            return self._config

        if not self.config_path.exists():
            _logger.error("Screener config file not found: %s", self.config_path)
            raise FileNotFoundError(f"Screener config file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)

            _logger.debug("Loaded screener config from: %s", self.config_path)
            return self._config

        except yaml.YAMLError as e:
            _logger.error("Invalid YAML in screener config: %s", e)
            raise ValueError(f"Invalid YAML in screener config: {e}")

    def list_available_sets(self) -> List[str]:
        """
        List all available screener set names.

        Returns:
            List of screener set names
        """
        config = self._load_config()
        screener_sets = config.get("screener_sets", {})
        return list(screener_sets.keys())

    def list_sets_by_category(self, category: str) -> List[str]:
        """
        List screener sets filtered by category.

        Args:
            category: Category to filter by (e.g., "us", "technology", "equity")

        Returns:
            List of screener set names matching the category
        """
        config = self._load_config()
        screener_sets = config.get("screener_sets", {})

        matching_sets = []
        for set_name, set_data in screener_sets.items():
            categories = set_data.get("categories", [])
            if category.lower() in [c.lower() for c in categories]:
                matching_sets.append(set_name)

        return matching_sets

    def search_sets(self, search_term: str) -> List[str]:
        """
        Search for screener sets by name or description.

        Args:
            search_term: Term to search for (case-insensitive)

        Returns:
            List of screener set names matching the search term
        """
        config = self._load_config()
        screener_sets = config.get("screener_sets", {})

        search_lower = search_term.lower()
        matching_sets = []

        for set_name, set_data in screener_sets.items():
            name_lower = set_name.lower()
            desc_lower = set_data.get("description", "").lower()
            display_name_lower = set_data.get("name", "").lower()

            if (search_lower in name_lower or
                search_lower in desc_lower or
                search_lower in display_name_lower):
                matching_sets.append(set_name)

        return matching_sets

    def get_set_info(self, set_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a screener set.

        Args:
            set_name: Name of the screener set

        Returns:
            Dictionary with set information including:
            - name: Display name
            - description: Set description
            - categories: List of categories
            - tickers: List of ticker symbols
            - ticker_count: Number of tickers

        Raises:
            KeyError: If set_name is not found
        """
        config = self._load_config()
        screener_sets = config.get("screener_sets", {})

        if set_name not in screener_sets:
            available = ", ".join(list(screener_sets.keys())[:5])
            _logger.error("Screener set not found: %s (available: %s...)", set_name, available)
            raise KeyError(f"Screener set not found: {set_name}")

        set_data = screener_sets[set_name]
        tickers = set_data.get("tickers", [])

        return {
            "name": set_data.get("name", set_name),
            "description": set_data.get("description", ""),
            "categories": set_data.get("categories", []),
            "tickers": tickers,
            "ticker_count": len(tickers)
        }

    def get_tickers(self, set_name: str) -> List[str]:
        """
        Get the list of tickers for a screener set.

        Args:
            set_name: Name of the screener set

        Returns:
            List of ticker symbols

        Raises:
            KeyError: If set_name is not found
        """
        set_info = self.get_set_info(set_name)
        return set_info["tickers"]

    def get_default_config(self, config_type: str) -> Dict[str, Any]:
        """
        Get a default screener configuration.

        Args:
            config_type: Type of configuration (e.g., "fundamental", "technical", "growth")

        Returns:
            Dictionary with default configuration parameters

        Raises:
            KeyError: If config_type is not found
        """
        config = self._load_config()
        default_configs = config.get("default_configs", {})

        if config_type not in default_configs:
            available = ", ".join(list(default_configs.keys()))
            raise KeyError(f"Default config not found: {config_type} (available: {available})")

        return default_configs[config_type]

    def list_default_configs(self) -> List[str]:
        """
        List all available default configuration types.

        Returns:
            List of default configuration names
        """
        config = self._load_config()
        default_configs = config.get("default_configs", {})
        return list(default_configs.keys())


# Global singleton instance
_screener_config = None


@lru_cache(maxsize=1)
def get_screener_config() -> ScreenerConfigLoader:
    """
    Get the global screener config loader instance.

    Returns:
        ScreenerConfigLoader instance
    """
    global _screener_config
    if _screener_config is None:
        _screener_config = ScreenerConfigLoader()
    return _screener_config
