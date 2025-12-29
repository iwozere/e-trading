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
        self.profile_ttl_days = 14
        self.metrics_ttl_days = 1
        self.quote_ttl_days = 1
        self.negative_cache_ttl_days = negative_cache_ttl_days

        # Cache directory structure: results/emps2/cache/fundamentals/
        self._cache_dir = Path("results") / "emps2" / "cache" / "fundamentals"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("Fundamental cache initialized (Profile: 14d, Metrics: 1d, Quote: 1d, Neg: %dd)",
                    negative_cache_ttl_days)

    def get_full_entry(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get the entire cache entry for a ticker."""
        try:
            cache_file = self._get_cache_path(ticker)
            if not cache_file.exists():
                return None
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def get_data(self, ticker: str, data_type: str) -> Optional[Dict[str, Any]]:
        """
        Get specific data type (profile, metrics, quote) from cache if not expired.
        """
        entry = self.get_full_entry(ticker)
        if not entry:
            return None

        if entry.get('is_empty_response'):
            # Check negative cache expiration
            if self._is_expired(entry.get('cached_at'), self.negative_cache_ttl_days):
                return None
            return {'is_negative_cache': True}

        type_data = entry.get(data_type)
        if not type_data:
            return None

        # Determine TTL for this type
        ttl_map = {
            'profile': self.profile_ttl_days,
            'metrics': self.metrics_ttl_days,
            'quote': self.quote_ttl_days
        }
        ttl = ttl_map.get(data_type, 1)

        if self._is_expired(type_data.get('updated_at'), ttl):
            return None

        return type_data.get('data')

    def set_data(self, ticker: str, data_type: str, data: Optional[Dict[str, Any]]) -> None:
        """Set specific data type in cache."""
        try:
            cache_file = self._get_cache_path(ticker)
            entry = self.get_full_entry(ticker) or {'ticker': ticker.upper()}

            is_empty = not data or data == {}

            if is_empty:
                entry['is_empty_response'] = True
                entry['cached_at'] = datetime.now(timezone.utc).isoformat()
            else:
                entry['is_empty_response'] = False
                entry[data_type] = {
                    'data': data,
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }

            with open(cache_file, 'w') as f:
                json.dump(entry, f, indent=2)

        except Exception:
            _logger.debug("Error caching %s for %s", data_type, ticker)

    def get(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Backward compatible get() - returns a combined dict if all types are valid.
        """
        profile = self.get_data(ticker, 'profile')
        if profile and profile.get('is_negative_cache'):
            return profile

        metrics = self.get_data(ticker, 'metrics')
        quote = self.get_data(ticker, 'quote')

        if not profile or not metrics or not quote:
            return None

        # Combine into expected format for FundamentalFilter
        combined = profile.copy()
        combined.update(metrics)
        combined.update(quote)
        return combined

    def set(self, ticker: str, data: Optional[Dict[str, Any]]) -> None:
        """
        Backward compatible set() - sets all types at once.
        """
        if not data:
            self.set_data(ticker, 'profile', None)
            return

        # Map combined keys back to types
        # This is a bit hacky but maintains compatibility during transition
        profile_keys = {'ticker', 'company_name', 'sector', 'industry', 'shares_outstanding', 'marketChar'}
        metrics_keys = {'avg_volume', 'pe_ratio', 'forward_pe', 'eps'} # etc

        # Since we use Fundamentals dataclass mostly, we'll just split by known fields
        self.set_data(ticker, 'profile', {k: v for k, v in data.items() if k in profile_keys or k == 'market_cap'})
        self.set_data(ticker, 'metrics', {k: v for k, v in data.items() if k == 'avg_volume'})
        self.set_data(ticker, 'quote', {k: v for k, v in data.items() if k == 'current_price'})

    def _get_cache_path(self, ticker: str) -> Path:
        """
        Get cache file path for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Path to cache file
        """
        return self._cache_dir / f"{ticker.upper()}.json"

    def _is_expired(self, timestamp_str: Optional[str], ttl_days: int) -> bool:
        """Check if a specific timestamp is expired."""
        try:
            if not timestamp_str:
                return True

            updated_at = datetime.fromisoformat(timestamp_str)
            now = datetime.now(timezone.utc)

            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)

            return (now - updated_at) > timedelta(days=ttl_days)

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
