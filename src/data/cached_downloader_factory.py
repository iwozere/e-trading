"""
Cached Downloader Factory

This module provides factory functions to create cached versions of any data downloader.
It automatically wraps existing downloaders with intelligent caching capabilities.
"""

from typing import Optional, Dict, Any
from src.data.base_data_downloader import BaseDataDownloader
from src.data.cached_data_downloader import CachedDataDownloader, create_cached_downloader
from src.data.utils.file_based_cache import configure_file_cache

# Import all available downloaders
from src.data.binance_data_downloader import BinanceDataDownloader
from src.data.yahoo_data_downloader import YahooDataDownloader
from src.data.alpha_vantage_data_downloader import AlphaVantageDataDownloader
from src.data.polygon_data_downloader import PolygonDataDownloader
from src.data.fmp_data_downloader import FMPDataDownloader
from src.data.finnhub_data_downloader import FinnhubDataDownloader
from src.data.coingecko_data_downloader import CoinGeckoDataDownloader
from src.data.twelvedata_data_downloader import TwelveDataDataDownloader


class CachedDownloaderFactory:
    """
    Factory for creating cached versions of data downloaders.

    This factory automatically wraps any data downloader with intelligent caching
    capabilities, providing seamless data access with minimal server requests.
    """

    def __init__(self, cache_dir: str = "d:/data-cache"):
        """
        Initialize the factory.

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = cache_dir
        self.cache = configure_file_cache(cache_dir=cache_dir)
        self._cached_downloaders: Dict[str, CachedDataDownloader] = {}

    def create_binance_downloader(self, api_key: Optional[str] = None,
                                 api_secret: Optional[str] = None) -> CachedDataDownloader:
        """Create a cached Binance downloader."""
        key = f"binance_{api_key or 'public'}"
        if key not in self._cached_downloaders:
            downloader = BinanceDataDownloader(api_key, api_secret)
            self._cached_downloaders[key] = CachedDataDownloader(downloader, self.cache)
        return self._cached_downloaders[key]

    def create_yahoo_downloader(self) -> CachedDataDownloader:
        """Create a cached Yahoo Finance downloader."""
        key = "yahoo"
        if key not in self._cached_downloaders:
            downloader = YahooDataDownloader()
            self._cached_downloaders[key] = CachedDataDownloader(downloader, self.cache)
        return self._cached_downloaders[key]

    def create_alpha_vantage_downloader(self, api_key: str) -> CachedDataDownloader:
        """Create a cached Alpha Vantage downloader."""
        key = f"alpha_vantage_{api_key[:8]}"
        if key not in self._cached_downloaders:
            downloader = AlphaVantageDataDownloader(api_key)
            self._cached_downloaders[key] = CachedDataDownloader(downloader, self.cache)
        return self._cached_downloaders[key]

    def create_polygon_downloader(self, api_key: str) -> CachedDataDownloader:
        """Create a cached Polygon downloader."""
        key = f"polygon_{api_key[:8]}"
        if key not in self._cached_downloaders:
            downloader = PolygonDataDownloader(api_key)
            self._cached_downloaders[key] = CachedDataDownloader(downloader, self.cache)
        return self._cached_downloaders[key]

    def create_fmp_downloader(self, api_key: str) -> CachedDataDownloader:
        """Create a cached FMP downloader."""
        key = f"fmp_{api_key[:8]}"
        if key not in self._cached_downloaders:
            downloader = FMPDataDownloader(api_key)
            self._cached_downloaders[key] = CachedDataDownloader(downloader, self.cache)
        return self._cached_downloaders[key]

    def create_finnhub_downloader(self, api_key: str) -> CachedDataDownloader:
        """Create a cached Finnhub downloader."""
        key = f"finnhub_{api_key[:8]}"
        if key not in self._cached_downloaders:
            downloader = FinnhubDataDownloader(api_key)
            self._cached_downloaders[key] = CachedDataDownloader(downloader, self.cache)
        return self._cached_downloaders[key]

    def create_coingecko_downloader(self) -> CachedDataDownloader:
        """Create a cached CoinGecko downloader."""
        key = "coingecko"
        if key not in self._cached_downloaders:
            downloader = CoinGeckoDataDownloader()
            self._cached_downloaders[key] = CachedDataDownloader(downloader, self.cache)
        return self._cached_downloaders[key]

    def create_twelvedata_downloader(self, api_key: str) -> CachedDataDownloader:
        """Create a cached Twelve Data downloader."""
        key = f"twelvedata_{api_key[:8]}"
        if key not in self._cached_downloaders:
            downloader = TwelveDataDataDownloader(api_key)
            self._cached_downloaders[key] = CachedDataDownloader(downloader, self.cache)
        return self._cached_downloaders[key]

    def create_cached_downloader(self, downloader: BaseDataDownloader,
                                provider_name: Optional[str] = None) -> CachedDataDownloader:
        """
        Create a cached version of any data downloader.

        Args:
            downloader: The underlying data downloader
            provider_name: Optional provider name override

        Returns:
            CachedDataDownloader instance
        """
        if provider_name:
            key = provider_name
        else:
            # Extract provider name from class
            class_name = downloader.__class__.__name__.lower()
            if 'binance' in class_name:
                key = "binance"
            elif 'yahoo' in class_name:
                key = "yahoo"
            elif 'alpha' in class_name:
                key = "alpha_vantage"
            elif 'polygon' in class_name:
                key = "polygon"
            elif 'fmp' in class_name:
                key = "fmp"
            elif 'finnhub' in class_name:
                key = "finnhub"
            elif 'coingecko' in class_name:
                key = "coingecko"
            elif 'twelvedata' in class_name:
                key = "twelvedata"
            else:
                key = "custom"

        if key not in self._cached_downloaders:
            self._cached_downloaders[key] = CachedDataDownloader(downloader, self.cache)

        return self._cached_downloaders[key]

    def get_all_cached_downloaders(self) -> Dict[str, CachedDataDownloader]:
        """Get all created cached downloaders."""
        return self._cached_downloaders.copy()

    def clear_cache(self, provider: Optional[str] = None, symbol: Optional[str] = None,
                   interval: Optional[str] = None):
        """Clear cache for specific criteria."""
        self.cache.clear(provider, symbol, interval)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


# Global factory instance
_global_factory: Optional[CachedDownloaderFactory] = None


def get_cached_downloader_factory(cache_dir: str = "d:/data-cache") -> CachedDownloaderFactory:
    """
    Get or create the global cached downloader factory.

    Args:
        cache_dir: Directory for cache storage

    Returns:
        CachedDownloaderFactory instance
    """
    global _global_factory

    if _global_factory is None:
        _global_factory = CachedDownloaderFactory(cache_dir)

    return _global_factory


# Convenience functions for common use cases
def create_cached_binance_downloader(api_key: Optional[str] = None,
                                    api_secret: Optional[str] = None,
                                    cache_dir: str = "d:/data-cache") -> CachedDataDownloader:
    """Create a cached Binance downloader."""
    factory = get_cached_downloader_factory(cache_dir)
    return factory.create_binance_downloader(api_key, api_secret)


def create_cached_yahoo_downloader(cache_dir: str = "d:/data-cache") -> CachedDataDownloader:
    """Create a cached Yahoo Finance downloader."""
    factory = get_cached_downloader_factory(cache_dir)
    return factory.create_yahoo_downloader()


def create_cached_alpha_vantage_downloader(api_key: str,
                                          cache_dir: str = "d:/data-cache") -> CachedDataDownloader:
    """Create a cached Alpha Vantage downloader."""
    factory = get_cached_downloader_factory(cache_dir)
    return factory.create_alpha_vantage_downloader(api_key)


def create_cached_polygon_downloader(api_key: str,
                                    cache_dir: str = "d:/data-cache") -> CachedDataDownloader:
    """Create a cached Polygon downloader."""
    factory = get_cached_downloader_factory(cache_dir)
    return factory.create_polygon_downloader(api_key)


def create_cached_fmp_downloader(api_key: str,
                                cache_dir: str = "d:/data-cache") -> CachedDataDownloader:
    """Create a cached FMP downloader."""
    factory = get_cached_downloader_factory(cache_dir)
    return factory.create_fmp_downloader(api_key)


def create_cached_finnhub_downloader(api_key: str,
                                    cache_dir: str = "d:/data-cache") -> CachedDataDownloader:
    """Create a cached Finnhub downloader."""
    factory = get_cached_downloader_factory(cache_dir)
    return factory.create_finnhub_downloader(api_key)


def create_cached_coingecko_downloader(cache_dir: str = "d:/data-cache") -> CachedDataDownloader:
    """Create a cached CoinGecko downloader."""
    factory = get_cached_downloader_factory(cache_dir)
    return factory.create_coingecko_downloader()


def create_cached_twelvedata_downloader(api_key: str,
                                       cache_dir: str = "d:/data-cache") -> CachedDataDownloader:
    """Create a cached Twelve Data downloader."""
    factory = get_cached_downloader_factory(cache_dir)
    return factory.create_twelvedata_downloader(api_key)
