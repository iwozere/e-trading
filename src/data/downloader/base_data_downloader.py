"""
Abstract base class for data downloaders, defining the interface for downloading historical market data from various sources.

Base Data Downloader Module
--------------------------

This module provides the BaseDataDownloader class, which defines the interface for downloading historical market data.
In the new unified architecture, downloaders focus solely on data fetching - caching, validation, and rate limiting
are handled by the DataManager.

Main Features:
- Clean interface for data fetching from various providers
- No caching logic (handled by DataManager)
- No rate limiting (handled by DataManager)
- Focus on single responsibility: fetching raw data from APIs

Classes:
- BaseDataDownloader: Abstract base class for data downloaders
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
import os

import pandas as pd

from src.model.schemas import OptionalFundamentals
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class BaseDataDownloader(ABC):
    """
    Abstract base class for data downloaders. Focuses solely on fetching raw data from APIs.

    In the new unified architecture:
    - Caching is handled by DataManager
    - Rate limiting is handled by DataManager
    - Validation is handled by DataManager
    - This class only handles API communication and data fetching
    """

    def __init__(self):
        """Initialize the data downloader."""
        pass

    @staticmethod
    def _get_config_value(config_key: str, env_var: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
        """
        Get config value from environment variables or config module.

        This method provides centralized configuration access that:
        1. First tries environment variables (if env_var is provided)
        2. Falls back to config.donotshare.donotshare module
        3. Returns default value if not found

        Args:
            config_key: Name of the config variable in config.donotshare.donotshare
            env_var: Environment variable name (optional, defaults to config_key)
            default: Default value to return if not found (optional)

        Returns:
            Config value as string, or None/default if not found

        Example:
            >>> api_key = BaseDataDownloader._get_config_value('POLYGON_API_KEY', 'POLYGON_API_KEY')
            >>> # Tries POLYGON_API_KEY env var first, then config.donotshare.donotshare.POLYGON_API_KEY
        """
        # Use env_var if provided, otherwise use config_key
        env_var_name = env_var or config_key

        # Try environment variable first
        value = os.getenv(env_var_name)
        if value:
            return value

        # Fallback to config module
        try:
            config_module = __import__('config.donotshare.donotshare', fromlist=[config_key])
            value = getattr(config_module, config_key, None)
            if value:
                return value
        except (ImportError, AttributeError):
            pass

        # Return default if nothing found
        return default


    @abstractmethod
    def get_supported_intervals(self) -> List[str]:
        """Return the list of supported intervals for this data downloader."""
        pass

    @abstractmethod
    def get_ohlcv(self, symbol: str, interval: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """
        Download historical OHLCV data for a given symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'AAPL')
            interval: Data interval (e.g., '1m', '1h', '1d')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            **kwargs: Additional provider-specific parameters

        Returns:
            DataFrame with OHLCV data (columns: open, high, low, close, volume)
            Index should be datetime
        """
        pass

    def get_fundamentals(self, symbol: str) -> OptionalFundamentals:
        """
        Return fundamentals if available for this provider; otherwise None.

        Args:
            symbol: Trading symbol

        Returns:
            Fundamentals data or None if not supported
        """
        return None

    def get_periods(self) -> List[str]:
        """Return the list of supported periods for this data downloader."""
        return ['1d', '7d', '1mo', '3mo', '6mo', '1y', '2y']

    def get_intervals(self) -> List[str]:
        """Return the list of supported intervals for this data downloader."""
        return self.get_supported_intervals()

    def is_valid_period_interval(self, period: str, interval: str) -> bool:
        """Check if the given period and interval combination is valid."""
        return interval in self.get_supported_intervals() and period in self.get_periods()