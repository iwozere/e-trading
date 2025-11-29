"""
Fundamental Data Cache

Persistent cache for Finnhub profile2 endpoint data to reduce API calls
and improve pipeline resilience. Implements TTL-based expiration.
"""

from pathlib import Path
import sys
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class FundamentalCache:
    """
    Cache for fundamental data with TTL-based expiration.

    Stores Finnhub profile2 responses as JSON files in:
    results/emps2/cache/fundamentals/TICKER.json

    Each cache entry includes:
    - Fundamental data (market_cap, float, sector, etc.)
    - Timestamp of cache creation
    - TTL expiration check
    """

    def __init__(self, cache_ttl_days: int = 3, negative_cache_ttl_days: int = 2):
        """
        Initialize fundamental data cache.

        Args:
            cache_ttl_days: Cache time-to-live in days for positive results (default: 3)
            negative_cache_ttl_days: Cache TTL for negative results (empty/failed) (default: 2)
        """
        self.cache_ttl_days = cache_ttl_days
        self.negative_cache_ttl_days = negative_cache_ttl_days

        # Cache directory structure: results/emps2/cache/fundamentals/
        self._cache_dir = Path("results") / "emps2" / "cache" / "fundamentals"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("Fundamental cache initialized (TTL: %d days, negative TTL: %d days, dir: %s)",
                    cache_ttl_days, negative_cache_ttl_days, self._cache_dir)

    def get(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get cached fundamental data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Cached fundamental data dict, or None if not cached/expired.
            For negative cache hits (empty response), returns a special dict:
            {'is_negative_cache': True} to signal "known bad ticker, skip API call"
        """
        try:
            cache_file = self._get_cache_path(ticker)

            if not cache_file.exists():
                return None

            # Read cache file
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            # Check if cache is expired
            if self._is_expired(cache_data):
                _logger.debug("Cache expired for %s", ticker)
                return None

            # Check if this is a negative cache entry (empty/failed response)
            if cache_data.get('is_empty_response', False):
                _logger.debug("Negative cache hit for %s (known empty ticker)", ticker)
                return {'is_negative_cache': True}

            _logger.debug("Cache hit for %s", ticker)
            return cache_data.get('data')

        except Exception:
            _logger.debug("Error reading cache for %s", ticker)
            return None

    def set(self, ticker: str, data: Optional[Dict[str, Any]]) -> None:
        """
        Cache fundamental data for a ticker.

        Supports both positive caching (valid data) and negative caching (empty/failed responses).

        Args:
            ticker: Stock ticker symbol
            data: Fundamental data dictionary to cache, or None/empty dict for negative cache
        """
        try:
            cache_file = self._get_cache_path(ticker)

            # Determine if this is a negative cache entry (empty/failed response)
            is_empty = not data or data == {}

            cache_entry = {
                'ticker': ticker.upper(),
                'data': data if data else None,
                'cached_at': datetime.now(timezone.utc).isoformat(),
                'ttl_days': self.negative_cache_ttl_days if is_empty else self.cache_ttl_days,
                'is_empty_response': is_empty
            }

            with open(cache_file, 'w') as f:
                json.dump(cache_entry, f, indent=2)

            if is_empty:
                _logger.debug("Cached negative result for %s (TTL: %d days)",
                            ticker, self.negative_cache_ttl_days)
            else:
                _logger.debug("Cached data for %s", ticker)

        except Exception:
            _logger.debug("Error caching data for %s", ticker)

    def _get_cache_path(self, ticker: str) -> Path:
        """
        Get cache file path for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Path to cache file
        """
        return self._cache_dir / f"{ticker.upper()}.json"

    def _is_expired(self, cache_data: Dict[str, Any]) -> bool:
        """
        Check if cached data is expired.

        Args:
            cache_data: Cache entry dictionary

        Returns:
            True if expired, False otherwise
        """
        try:
            cached_at_str = cache_data.get('cached_at')
            if not cached_at_str:
                return True

            cached_at = datetime.fromisoformat(cached_at_str)
            now = datetime.now(timezone.utc)

            # Make cached_at timezone-aware if it isn't
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=timezone.utc)

            age = now - cached_at
            max_age = timedelta(days=self.cache_ttl_days)

            return age > max_age

        except Exception:
            return True

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (total, valid, expired, positive, negative)
        """
        try:
            cache_files = list(self._cache_dir.glob("*.json"))
            total = len(cache_files)

            valid = 0
            expired = 0
            positive = 0  # Valid data
            negative = 0  # Empty/failed responses

            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)

                    if self._is_expired(cache_data):
                        expired += 1
                    else:
                        valid += 1
                        # Count positive vs negative cache entries
                        if cache_data.get('is_empty_response', False):
                            negative += 1
                        else:
                            positive += 1
                except Exception:
                    expired += 1

            return {
                'total': total,
                'valid': valid,
                'expired': expired,
                'positive': positive,
                'negative': negative
            }

        except Exception:
            return {'total': 0, 'valid': 0, 'expired': 0, 'positive': 0, 'negative': 0}

    def clear_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        removed = 0

        try:
            cache_files = list(self._cache_dir.glob("*.json"))

            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)

                    if self._is_expired(cache_data):
                        cache_file.unlink()
                        removed += 1

                except Exception:
                    continue

            if removed > 0:
                _logger.info("Cleared %d expired cache entries", removed)

        except Exception:
            _logger.exception("Error clearing expired cache:")

        return removed

    def clear_all(self) -> None:
        """Clear entire cache."""
        try:
            cache_files = list(self._cache_dir.glob("*.json"))
            for cache_file in cache_files:
                cache_file.unlink()

            _logger.info("Cleared entire cache (%d entries)", len(cache_files))

        except Exception:
            _logger.exception("Error clearing cache:")


def create_fundamental_cache(
    cache_ttl_days: int = 3,
    negative_cache_ttl_days: int = 2
) -> FundamentalCache:
    """
    Factory function to create fundamental cache.

    Args:
        cache_ttl_days: Cache time-to-live in days for positive results
        negative_cache_ttl_days: Cache TTL for negative results (empty/failed)

    Returns:
        FundamentalCache instance
    """
    return FundamentalCache(cache_ttl_days, negative_cache_ttl_days)
