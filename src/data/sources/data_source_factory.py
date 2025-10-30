"""
Data source factory for creating and managing data sources.

This module provides a centralized factory for creating data sources with
configuration-based initialization and lifecycle management.
"""

import logging
from typing import Dict, Type, Optional, List, Any
from pathlib import Path
import yaml

from src.data.sources.base_data_source import BaseDataSource
from src.data.utils import configure_cache

# Import cache directory setting
try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

_logger = logging.getLogger(__name__)


class DataSourceFactory:
    """
    Factory for creating and managing data sources.

    Provides:
    - Configuration-based data source creation
    - Lifecycle management
    - Health monitoring
    - Centralized configuration
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize data source factory.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or Path("src/config/data/config.yaml")
        self._data_sources: Dict[str, BaseDataSource] = {}
        self._data_source_classes: Dict[str, Type[BaseDataSource]] = {}
        self._config = self._load_config()

        # Configure global cache
        if self._config.get('caching', {}).get('enabled', True):
            cache_config = self._config['caching']
            configure_cache(
                cache_dir=cache_config.get('directory', DATA_CACHE_DIR),
                max_size_gb=cache_config.get('max_size_gb', 10.0),
                retention_days=cache_config.get('retention_days', 30)
            )

        _logger.info("Data source factory initialized")

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                _logger.info("Loaded configuration from %s", self.config_path)
                return config or {}
            else:
                _logger.warning("Configuration file not found: %s", self.config_path)
                return {}
        except Exception as e:
            _logger.exception("Failed to load configuration:")
            return {}

    def register_data_source(
        self,
        provider_name: str,
        data_source_class: Type[BaseDataSource]
    ) -> None:
        """
        Register a data source class for a provider.

        Args:
            provider_name: Name of the data provider
            data_source_class: Class implementing BaseDataSource
        """
        if not issubclass(data_source_class, BaseDataSource):
            raise ValueError(f"Class {data_source_class.__name__} must inherit from BaseDataSource")

        self._data_source_classes[provider_name] = data_source_class
        _logger.info("Registered data source class for %s", provider_name)

    def create_data_source(
        self,
        provider_name: str,
        **kwargs
    ) -> Optional[BaseDataSource]:
        """
        Create a data source instance.

        Args:
            provider_name: Name of the data provider
            **kwargs: Additional arguments for data source initialization

        Returns:
            Data source instance or None if creation failed
        """
        try:
            # Check if provider is registered
            if provider_name not in self._data_source_classes:
                _logger.error("No data source class registered for %s", provider_name)
                return None

            # Get provider configuration
            provider_config = self._config.get(provider_name, {})

            # Merge configuration with kwargs
            config = {**provider_config, **kwargs}

            # Set defaults from global config
            global_config = self._config.get('global', {})
            config.setdefault('cache_enabled', global_config.get('cache_enabled', True))
            config.setdefault('rate_limit_enabled', global_config.get('rate_limit_enabled', True))
            config.setdefault('validation_enabled', global_config.get('validation_enabled', True))

            # Create data source instance
            data_source_class = self._data_source_classes[provider_name]
            data_source = data_source_class(**config)

            # Store instance
            self._data_sources[provider_name] = data_source

            _logger.info("Created data source for %s", provider_name)
            return data_source

        except Exception as e:
            _logger.exception("Failed to create data source for %s:", provider_name)
            return None

    def get_data_source(self, provider_name: str) -> Optional[BaseDataSource]:
        """
        Get existing data source instance.

        Args:
            provider_name: Name of the data provider

        Returns:
            Data source instance or None if not found
        """
        return self._data_sources.get(provider_name)

    def get_or_create_data_source(
        self,
        provider_name: str,
        **kwargs
    ) -> Optional[BaseDataSource]:
        """
        Get existing data source or create new one.

        Args:
            provider_name: Name of the data provider
            **kwargs: Additional arguments for data source initialization

        Returns:
            Data source instance or None if creation failed
        """
        # Try to get existing instance
        data_source = self.get_data_source(provider_name)
        if data_source is not None:
            return data_source

        # Create new instance
        return self.create_data_source(provider_name, **kwargs)

    def get_all_data_sources(self) -> Dict[str, BaseDataSource]:
        """
        Get all created data sources.

        Returns:
            Dictionary mapping provider names to data source instances
        """
        return self._data_sources.copy()

    def get_available_providers(self) -> List[str]:
        """
        Get list of available data providers.

        Returns:
            List of provider names
        """
        return list(self._data_source_classes.keys())

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all data sources.

        Returns:
            Dictionary with health information for each provider
        """
        health_status = {}

        for provider_name, data_source in self._data_sources.items():
            try:
                health_status[provider_name] = {
                    'status': data_source.get_connection_status(),
                    'is_healthy': data_source.is_healthy()
                }
            except Exception as e:
                health_status[provider_name] = {
                    'status': {'error': str(e)},
                    'is_healthy': False
                }

        return health_status

    def get_data_quality_reports(
        self,
        symbols: List[str],
        interval: str,
        start_date=None,
        end_date=None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get data quality reports for multiple symbols across all providers.

        Args:
            symbols: List of trading symbols
            interval: Data interval
            start_date: Start date for data range
            end_date: End date for data range

        Returns:
            Dictionary mapping provider names to quality reports
        """
        quality_reports = {}

        for provider_name, data_source in self._data_sources.items():
            try:
                provider_reports = {}
                for symbol in symbols:
                    try:
                        report = data_source.get_data_quality_report(
                            symbol, interval, start_date, end_date
                        )
                        provider_reports[symbol] = report
                    except Exception as e:
                        _logger.warning("Failed to get quality report for %s from %s: %s", symbol, provider_name, e)
                        provider_reports[symbol] = {'error': str(e)}

                quality_reports[provider_name] = provider_reports

            except Exception as e:
                _logger.exception("Failed to get quality reports from %s:", provider_name)
                quality_reports[provider_name] = {'error': str(e)}

        return quality_reports

    def cleanup_all(self) -> None:
        """Clean up all data sources."""
        for provider_name, data_source in list(self._data_sources.items()):
            try:
                data_source.cleanup()
                del self._data_sources[provider_name]
                _logger.info("Cleaned up data source for %s", provider_name)
            except Exception as e:
                _logger.exception("Error cleaning up data source for %s:", provider_name)

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._config = self._load_config()
        _logger.info("Configuration reloaded")

    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.

        Args:
            provider_name: Name of the data provider

        Returns:
            Provider configuration dictionary
        """
        return self._config.get(provider_name, {})

    def update_provider_config(
        self,
        provider_name: str,
        config_updates: Dict[str, Any]
    ) -> bool:
        """
        Update configuration for a specific provider.

        Args:
            provider_name: Name of the data provider
            config_updates: Configuration updates to apply

        Returns:
            True if update successful, False otherwise
        """
        try:
            if provider_name not in self._config:
                self._config[provider_name] = {}

            self._config[provider_name].update(config_updates)

            # Save updated configuration
            with open(self.config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)

            _logger.info("Updated configuration for %s", provider_name)
            return True

        except Exception as e:
            _logger.exception("Failed to update configuration for %s:", provider_name)
            return False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup_all()


# Global factory instance
_factory_instance: Optional[DataSourceFactory] = None


def get_data_source_factory(config_path: Optional[Path] = None) -> DataSourceFactory:
    """
    Get or create global data source factory instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Global DataSourceFactory instance
    """
    global _factory_instance

    if _factory_instance is None:
        _factory_instance = DataSourceFactory(config_path)

    return _factory_instance


def register_data_source(
    provider_name: str,
    data_source_class: Type[BaseDataSource]
) -> None:
    """
    Register a data source class with the global factory.

    Args:
        provider_name: Name of the data provider
        data_source_class: Class implementing BaseDataSource
    """
    factory = get_data_source_factory()
    factory.register_data_source(provider_name, data_source_class)


def create_data_source(provider_name: str, **kwargs) -> Optional[BaseDataSource]:
    """
    Create a data source using the global factory.

    Args:
        provider_name: Name of the data provider
        **kwargs: Additional arguments for data source initialization

    Returns:
        Data source instance or None if creation failed
    """
    factory = get_data_source_factory()
    return factory.create_data_source(provider_name, **kwargs)


def get_data_source(provider_name: str) -> Optional[BaseDataSource]:
    """
    Get existing data source from the global factory.

    Args:
        provider_name: Name of the data provider

    Returns:
        Data source instance or None if not found
    """
    factory = get_data_source_factory()
    return factory.get_data_source(provider_name)
