"""
Universe Loader for Short Squeeze Detection Pipeline

This module provides functionality to load and filter the universe of stocks
for short squeeze analysis using FMP data provider.
"""

from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import os

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.ml.pipeline.p04_short_squeeze.config.data_classes import UniverseConfig

_logger = setup_logger(__name__)


class UniverseLoader:
    """
    Universe loader for short squeeze detection pipeline.

    Loads and filters stocks from FMP based on market cap, volume, and exchange criteria.
    Includes caching functionality for performance optimization.
    """

    def __init__(self, fmp_downloader: FMPDataDownloader, config: UniverseConfig):
        """
        Initialize Universe Loader.

        Args:
            fmp_downloader: FMP data downloader instance
            config: Universe configuration with filtering criteria
        """
        self.fmp_downloader = fmp_downloader
        self.config = config
        self._cache_dir = Path("data/cache/universe")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_hours = 24  # Cache universe for 24 hours

        _logger.info("Universe Loader initialized with config: min_cap=%s, max_cap=%s, min_volume=%s",
                    self.config.min_market_cap, self.config.max_market_cap, self.config.min_avg_volume)

    def load_universe(self) -> List[str]:
        """
        Load the complete universe of stocks for short squeeze analysis.

        Returns:
            List of ticker symbols that meet all filtering criteria
        """
        try:
            _logger.info("Loading universe with filtering criteria")

            # Check cache first
            cached_universe = self._load_from_cache()
            if cached_universe:
                _logger.info("Loaded %d tickers from cache", len(cached_universe))
                return cached_universe

            # Load from FMP screener
            universe = self._load_from_screener()

            # Apply additional filters
            filtered_universe = self._apply_filters(universe)

            # Cache the results
            self._save_to_cache(filtered_universe)

            _logger.info("Successfully loaded universe of %d tickers", len(filtered_universe))
            return filtered_universe

        except Exception as e:
            _logger.error("Error loading universe: %s", e)
            return []

    def _load_from_screener(self) -> List[str]:
        """
        Load universe from FMP stock screener.

        Returns:
            List of ticker symbols from screener
        """
        try:
            # Build screener criteria
            criteria = {
                'marketCapMoreThan': self.config.min_market_cap,
                'marketCapLowerThan': self.config.max_market_cap,
                'volumeMoreThan': self.config.min_avg_volume,
                'exchange': ','.join(self.config.exchanges),
                'limit': 1000  # Maximum results from FMP
            }

            _logger.info("Loading universe from FMP screener with criteria: %s", criteria)

            # Use the existing method from FMP downloader
            tickers = self.fmp_downloader.load_universe_from_screener(criteria)

            _logger.info("FMP screener returned %d tickers", len(tickers))
            return tickers

        except Exception as e:
            _logger.error("Error loading from screener: %s", e)
            return []

    def _apply_filters(self, tickers: List[str]) -> List[str]:
        """
        Apply additional filters to the ticker list.

        Args:
            tickers: List of ticker symbols to filter

        Returns:
            Filtered list of ticker symbols
        """
        try:
            if not tickers:
                return []

            _logger.info("Applying additional filters to %d tickers", len(tickers))

            # For now, we rely on the FMP screener filters
            # Additional filters can be added here if needed
            filtered_tickers = []

            for ticker in tickers:
                # Basic ticker validation
                if self._is_valid_ticker(ticker):
                    filtered_tickers.append(ticker)

            _logger.info("Filtered universe to %d tickers", len(filtered_tickers))
            return filtered_tickers

        except Exception as e:
            _logger.error("Error applying filters: %s", e)
            return tickers  # Return original list on error

    def _is_valid_ticker(self, ticker: str) -> bool:
        """
        Validate if a ticker is suitable for short squeeze analysis.

        Args:
            ticker: Ticker symbol to validate

        Returns:
            True if ticker is valid, False otherwise
        """
        try:
            # Basic validation
            if not ticker or len(ticker.strip()) == 0:
                return False

            ticker = ticker.strip().upper()

            # Length check (typical US tickers are 1-5 characters)
            if len(ticker) > 5:
                return False

            # Exclude certain ticker patterns that are typically not suitable
            # for short squeeze analysis (ETFs, preferred shares, etc.)
            exclude_patterns = [
                '.', '-', '/', ' ',  # Special characters
                'WARR', 'WS', 'WT',  # Warrants
                'PR',  # Preferred shares (though this might be too broad)
            ]

            for pattern in exclude_patterns:
                if pattern in ticker:
                    return False

            return True

        except Exception as e:
            _logger.warning("Error validating ticker %s: %s", ticker, e)
            return False

    def filter_by_market_cap(self, tickers: List[str], min_cap: float, max_cap: float) -> List[str]:
        """
        Filter tickers by market capitalization range.

        Args:
            tickers: List of ticker symbols to filter
            min_cap: Minimum market cap
            max_cap: Maximum market cap

        Returns:
            Filtered list of ticker symbols
        """
        try:
            if not tickers:
                return []

            _logger.info("Filtering %d tickers by market cap range: $%s - $%s",
                        len(tickers), min_cap, max_cap)

            filtered_tickers = []

            # Get market cap data in batches to avoid overwhelming the API
            batch_size = 50
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]

                for ticker in batch:
                    try:
                        market_cap_data = self.fmp_downloader.get_market_cap_data(ticker)
                        if market_cap_data:
                            market_cap = market_cap_data.get('marketCap')
                            if market_cap and min_cap <= market_cap <= max_cap:
                                filtered_tickers.append(ticker)
                    except Exception as e:
                        _logger.warning("Error getting market cap for %s: %s", ticker, e)
                        continue

            _logger.info("Market cap filter resulted in %d tickers", len(filtered_tickers))
            return filtered_tickers

        except Exception as e:
            _logger.error("Error filtering by market cap: %s", e)
            return tickers

    def filter_by_volume(self, tickers: List[str], min_volume: int) -> List[str]:
        """
        Filter tickers by minimum average volume.

        Args:
            tickers: List of ticker symbols to filter
            min_volume: Minimum average volume

        Returns:
            Filtered list of ticker symbols
        """
        try:
            if not tickers:
                return []

            _logger.info("Filtering %d tickers by minimum volume: %s", len(tickers), min_volume)

            filtered_tickers = []

            # Get recent volume data to calculate average
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # 30-day average

            for ticker in tickers:
                try:
                    # Get recent OHLCV data
                    df = self.fmp_downloader.get_ohlcv(ticker, '1d', start_date, end_date)
                    if df is not None and not df.empty:
                        avg_volume = df['volume'].mean()
                        if avg_volume >= min_volume:
                            filtered_tickers.append(ticker)
                except Exception as e:
                    _logger.warning("Error getting volume data for %s: %s", ticker, e)
                    continue

            _logger.info("Volume filter resulted in %d tickers", len(filtered_tickers))
            return filtered_tickers

        except Exception as e:
            _logger.error("Error filtering by volume: %s", e)
            return tickers

    def _load_from_cache(self) -> Optional[List[str]]:
        """
        Load universe from cache if available and not expired.

        Returns:
            Cached ticker list or None if cache is invalid/expired
        """
        try:
            cache_file = self._cache_dir / "universe_cache.json"

            if not cache_file.exists():
                return None

            # Check cache age
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.total_seconds() > (self._cache_ttl_hours * 3600):
                _logger.info("Universe cache expired (age: %s hours)", cache_age.total_seconds() / 3600)
                return None

            # Load cache
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            # Validate cache structure
            if not isinstance(cache_data, dict) or 'tickers' not in cache_data:
                _logger.warning("Invalid cache structure, ignoring cache")
                return None

            tickers = cache_data['tickers']
            if not isinstance(tickers, list):
                _logger.warning("Invalid ticker list in cache, ignoring cache")
                return None

            _logger.info("Loaded %d tickers from cache (created: %s)",
                        len(tickers), cache_data.get('created_at', 'unknown'))
            return tickers

        except Exception as e:
            _logger.warning("Error loading from cache: %s", e)
            return None

    def _save_to_cache(self, tickers: List[str]) -> None:
        """
        Save universe to cache.

        Args:
            tickers: List of ticker symbols to cache
        """
        try:
            cache_file = self._cache_dir / "universe_cache.json"

            cache_data = {
                'tickers': tickers,
                'created_at': datetime.now().isoformat(),
                'config': {
                    'min_market_cap': self.config.min_market_cap,
                    'max_market_cap': self.config.max_market_cap,
                    'min_avg_volume': self.config.min_avg_volume,
                    'exchanges': self.config.exchanges
                }
            }

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            _logger.info("Saved %d tickers to cache", len(tickers))

        except Exception as e:
            _logger.warning("Error saving to cache: %s", e)

    def clear_cache(self) -> None:
        """Clear the universe cache."""
        try:
            cache_file = self._cache_dir / "universe_cache.json"
            if cache_file.exists():
                cache_file.unlink()
                _logger.info("Universe cache cleared")
        except Exception as e:
            _logger.warning("Error clearing cache: %s", e)

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache.

        Returns:
            Dictionary with cache information
        """
        try:
            cache_file = self._cache_dir / "universe_cache.json"

            if not cache_file.exists():
                return {'exists': False}

            stat = cache_file.stat()
            cache_age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)

            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            return {
                'exists': True,
                'size': len(cache_data.get('tickers', [])),
                'created_at': cache_data.get('created_at'),
                'age_hours': cache_age.total_seconds() / 3600,
                'expired': cache_age.total_seconds() > (self._cache_ttl_hours * 3600),
                'config': cache_data.get('config', {})
            }

        except Exception as e:
            _logger.warning("Error getting cache info: %s", e)
            return {'exists': False, 'error': str(e)}


def create_universe_loader(fmp_downloader: FMPDataDownloader,
                          config: UniverseConfig) -> UniverseLoader:
    """
    Factory function to create Universe Loader.

    Args:
        fmp_downloader: FMP data downloader instance
        config: Universe configuration

    Returns:
        Configured Universe Loader instance
    """
    return UniverseLoader(fmp_downloader, config)


# Example usage
if __name__ == "__main__":
    from src.ml.pipeline.p04_short_squeeze.config.data_classes import UniverseConfig

    # Create FMP downloader
    fmp_downloader = FMPDataDownloader()

    # Create universe config
    universe_config = UniverseConfig(
        min_market_cap=100_000_000,  # $100M
        max_market_cap=10_000_000_000,  # $10B
        min_avg_volume=200_000,
        exchanges=['NYSE', 'NASDAQ']
    )

    # Create universe loader
    loader = create_universe_loader(fmp_downloader, universe_config)

    # Test connection
    if fmp_downloader.test_connection():
        print("✅ FMP API connection successful")

        # Load universe
        universe = loader.load_universe()
        if universe:
            print(f"✅ Loaded universe of {len(universe)} stocks")
            print(f"Sample tickers: {universe[:10]}")

            # Show cache info
            cache_info = loader.get_cache_info()
            print(f"Cache info: {cache_info}")
        else:
            print("❌ Failed to load universe")
    else:
        print("❌ FMP API connection failed")