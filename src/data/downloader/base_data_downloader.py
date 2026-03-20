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
        Get config value for a provider API key.

        Priority: environment variable → config.donotshare.donotshare → default.
        Delegates to the central ``src.config.provider_config`` service so the
        lookup happens at most once per key across the entire process.

        Args:
            config_key: Attribute name in config.donotshare.donotshare (also used
                        as the env-var name when ``env_var`` is not supplied).
            env_var: Override for the environment variable name (optional).
            default: Value to return if nothing is found.

        Returns:
            Config value as string, or ``default`` if not found.
        """
        from src.config.provider_config import get_api_key

        # env_var overrides the lookup key for the environment-variable check
        if env_var and env_var != config_key:
            value = get_api_key(env_var) or get_api_key(config_key)
        else:
            value = get_api_key(config_key)

        return value if value is not None else default


    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the canonical provider name for this downloader."""
        pass

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