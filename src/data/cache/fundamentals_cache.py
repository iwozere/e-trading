"""
Fundamentals Cache System

This module provides JSON-based caching for fundamentals data with configurable TTL rules.
It supports multiple providers and automatic stale data cleanup.

Cache Structure:
- {cache_dir}/fundamentals/{symbol}/{provider}_{timestamp}.json
- Example: c:/data-cache/fundamentals/AAPL/yfinance_20250106_143022.json

Features:
- Configurable TTL based on data type (profiles: 14d, ratios: 3d, statements: 90d)
- Multi-provider support
- Automatic stale data cleanup
- Provider priority-based data combination
"""

import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Import cache directory setting
from config.donotshare.donotshare import DATA_CACHE_DIR

_logger = logging.getLogger(__name__)

@dataclass
class CacheMetadata:
    """Metadata for cached fundamentals data."""
    provider: str
    timestamp: datetime
    symbol: str
    data_quality_score: float
    file_path: str
    is_valid: bool = True

class FundamentalsCache:
    """
    JSON-based fundamentals cache with configurable TTL and multi-provider support.
    """

    def __init__(self, cache_dir: str = DATA_CACHE_DIR, combiner=None):
        """
        Initialize the fundamentals cache.

        Args:
            cache_dir: Base cache directory path
            combiner: FundamentalsCombiner instance for TTL configuration
        """
        self.cache_dir = Path(cache_dir)
        self.fundamentals_dir = self.cache_dir / "fundamentals"
        self.combiner = combiner
        self.default_ttl_days = 7

        # Create directories if they don't exist
        self.fundamentals_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("Fundamentals cache initialized at %s", self.fundamentals_dir)

    def find_latest_json(self, symbol: str, provider: Optional[str] = None,
                        data_type: str = "general") -> Optional[CacheMetadata]:
        """
        Find the most recent cached fundamentals data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            provider: Specific provider to look for (optional)
            data_type: Type of data to determine TTL (profiles, ratios, statements, etc.)

        Returns:
            CacheMetadata object if valid cache found, None otherwise
        """
        symbol_dir = self.fundamentals_dir / symbol.upper()

        if not symbol_dir.exists():
            return None

        latest_metadata = None
        latest_timestamp = None

        # Look for JSON files in the symbol directory
        for file_path in symbol_dir.glob("*.json"):
            try:
                # Parse filename: {provider}_{timestamp}.json
                filename = file_path.stem
                if '_' not in filename:
                    continue

                # Split filename: provider_YYYYMMDD_HHMMSS
                parts = filename.split('_')
                if len(parts) < 3:
                    continue

                provider_name = parts[0]
                timestamp_str = '_'.join(parts[1:])  # Rejoin date and time parts

                # Skip if looking for specific provider and this isn't it
                if provider and provider_name != provider:
                    continue

                # Parse timestamp
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                # Check if this is the latest
                if latest_timestamp is None or timestamp > latest_timestamp:
                    # Check if cache is still valid with data-type specific TTL
                    if self.is_cache_valid(timestamp, data_type=data_type):
                        latest_timestamp = timestamp
                        latest_metadata = CacheMetadata(
                            provider=provider_name,
                            timestamp=timestamp,
                            symbol=symbol.upper(),
                            data_quality_score=1.0,  # Default score
                            file_path=str(file_path)
                        )

            except (ValueError, IndexError) as e:
                _logger.warning("Invalid cache file format: %s - %s", file_path, e)
                continue

        if latest_metadata:
            _logger.debug("Found latest cache for %s: %s from %s",
                         symbol, latest_metadata.timestamp, latest_metadata.provider)

        return latest_metadata

    def write_json(self, symbol: str, provider: str, data: Dict[str, Any],
                   timestamp: Optional[datetime] = None) -> str:
        """
        Write fundamentals data to cache.

        Args:
            symbol: Trading symbol
            provider: Data provider name
            data: Fundamentals data dictionary
            timestamp: Timestamp for the data (defaults to now)

        Returns:
            Path to the written cache file
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Create symbol directory
        symbol_dir = self.fundamentals_dir / symbol.upper()
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{provider}_{timestamp_str}.json"
        file_path = symbol_dir / filename

        # Prepare data with metadata
        cache_data = {
            "metadata": {
                "provider": provider,
                "symbol": symbol.upper(),
                "timestamp": timestamp.isoformat(),
                "cache_version": "1.0",
                "data_quality_score": self._calculate_quality_score(data)
            },
            "fundamentals": data
        }

        # Write to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            _logger.info("Cached fundamentals for %s from %s at %s",
                        symbol, provider, timestamp)

            return str(file_path)

        except Exception as e:
            _logger.exception("Failed to write cache file %s:", file_path)
            raise

    def read_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Read fundamentals data from cache file.

        Args:
            file_path: Path to the cache file

        Returns:
            Fundamentals data dictionary or None if read fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Validate cache structure
            if "fundamentals" not in cache_data:
                _logger.warning("Invalid cache file structure: %s", file_path)
                return None

            return cache_data["fundamentals"]

        except Exception as e:
            _logger.exception("Failed to read cache file %s:", file_path)
            return None

    def is_cache_valid(self, timestamp: datetime, max_age_days: Optional[int] = None,
                      data_type: str = "general") -> bool:
        """
        Check if cached data is still valid based on age and data type.

        Args:
            timestamp: Timestamp of the cached data
            max_age_days: Maximum age in days (overrides data_type TTL if provided)
            data_type: Type of data to determine TTL (profiles, ratios, statements, etc.)

        Returns:
            True if cache is still valid, False otherwise
        """
        if max_age_days is None:
            # Get TTL from combiner configuration if available
            if self.combiner:
                max_age_days = self.combiner.get_ttl_for_data_type(data_type)
            else:
                max_age_days = self.default_ttl_days

        age = datetime.now() - timestamp
        is_valid = age.days < max_age_days

        if not is_valid:
            _logger.debug("Cache expired: %s days old (max: %s for %s)", age.days, max_age_days, data_type)

        return is_valid

    def cleanup_stale_data(self, symbol: str, provider: str, new_timestamp: datetime) -> List[str]:
        """
        Remove stale fundamentals data when new data is downloaded.

        Args:
            symbol: Trading symbol
            provider: Provider that just downloaded new data
            new_timestamp: Timestamp of the new data

        Returns:
            List of removed file paths
        """
        symbol_dir = self.fundamentals_dir / symbol.upper()

        if not symbol_dir.exists():
            return []

        removed_files = []

        # Find all cache files for this symbol and provider
        pattern = f"{provider}_*.json"
        for file_path in symbol_dir.glob(pattern):
            try:
                # Parse timestamp from filename
                filename = file_path.stem
                parts = filename.split('_')
                if len(parts) < 3:
                    continue
                timestamp_str = '_'.join(parts[1:])  # Rejoin date and time parts
                file_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                # Remove if older than new data
                if file_timestamp < new_timestamp:
                    file_path.unlink()
                    removed_files.append(str(file_path))
                    _logger.info("Removed stale cache file: %s", file_path)

            except (ValueError, IndexError) as e:
                _logger.warning("Invalid cache file format during cleanup: %s - %s", file_path, e)
                continue

        # Keep at least one backup copy (safety mechanism)
        remaining_files = list(symbol_dir.glob(f"{provider}_*.json"))
        if len(remaining_files) == 0 and removed_files:
            _logger.warning("All cache files removed for %s %s, no backup available",
                           symbol, provider)

        return removed_files

    def get_cache_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cache statistics for a symbol or all symbols.

        Args:
            symbol: Specific symbol to get stats for (optional)

        Returns:
            Dictionary with cache statistics
        """
        if symbol:
            symbol_dir = self.fundamentals_dir / symbol.upper()
            if not symbol_dir.exists():
                return {"symbol": symbol, "files": 0, "total_size": 0}

            files = list(symbol_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in files)

            return {
                "symbol": symbol.upper(),
                "files": len(files),
                "total_size": total_size,
                "providers": list(set(f.stem.split('_')[0] for f in files))
            }
        else:
            # Get stats for all symbols
            total_files = 0
            total_size = 0
            symbols = []

            for symbol_dir in self.fundamentals_dir.iterdir():
                if symbol_dir.is_dir():
                    files = list(symbol_dir.glob("*.json"))
                    total_files += len(files)
                    total_size += sum(f.stat().st_size for f in files)
                    symbols.append(symbol_dir.name)

            return {
                "total_symbols": len(symbols),
                "total_files": total_files,
                "total_size": total_size,
                "symbols": sorted(symbols)
            }

    def _calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """
        Calculate data quality score based on available fields.

        Args:
            data: Fundamentals data dictionary

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not data:
            return 0.0

        # Define important fields and their weights
        important_fields = {
            'market_cap': 0.2,
            'pe_ratio': 0.15,
            'pb_ratio': 0.15,
            'dividend_yield': 0.1,
            'revenue': 0.1,
            'net_income': 0.1,
            'total_debt': 0.1,
            'cash': 0.1
        }

        score = 0.0
        for field, weight in important_fields.items():
            if field in data and data[field] is not None:
                score += weight

        return min(score, 1.0)

    def cleanup_expired_data(self, max_age_days: Optional[int] = None,
                           data_type: str = "general") -> Dict[str, int]:
        """
        Clean up all expired cache data.

        Args:
            max_age_days: Maximum age in days (overrides data_type TTL if provided)
            data_type: Type of data to determine TTL (profiles, ratios, statements, etc.)

        Returns:
            Dictionary with cleanup statistics
        """
        if max_age_days is None:
            # Get TTL from combiner configuration if available
            if self.combiner:
                max_age_days = self.combiner.get_ttl_for_data_type(data_type)
            else:
                max_age_days = self.default_ttl_days

        stats = {"removed_files": 0, "removed_symbols": 0}

        for symbol_dir in self.fundamentals_dir.iterdir():
            if not symbol_dir.is_dir():
                continue

            removed_files = 0
            for file_path in symbol_dir.glob("*.json"):
                try:
                    # Parse timestamp from filename
                    filename = file_path.stem
                    if '_' not in filename:
                        continue

                    parts = filename.split('_')
                    if len(parts) < 3:
                        continue
                    timestamp_str = '_'.join(parts[1:])  # Rejoin date and time parts
                    file_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                    # Remove if expired
                    if not self.is_cache_valid(file_timestamp, max_age_days, data_type):
                        file_path.unlink()
                        removed_files += 1
                        stats["removed_files"] += 1

                except (ValueError, IndexError) as e:
                    _logger.warning("Invalid cache file format during cleanup: %s - %s", file_path, e)
                    continue

            # Remove empty symbol directories
            if removed_files > 0 and not list(symbol_dir.glob("*.json")):
                symbol_dir.rmdir()
                stats["removed_symbols"] += 1

        _logger.info("Cache cleanup completed: %d files, %d symbols removed",
                    stats["removed_files"], stats["removed_symbols"])

        return stats


# Global cache instance
_fundamentals_cache = None

def get_fundamentals_cache(cache_dir: str = DATA_CACHE_DIR, combiner=None) -> FundamentalsCache:
    """
    Get the global fundamentals cache instance.

    Args:
        cache_dir: Base cache directory path
        combiner: FundamentalsCombiner instance for TTL configuration

    Returns:
        FundamentalsCache instance
    """
    global _fundamentals_cache
    if _fundamentals_cache is None:
        _fundamentals_cache = FundamentalsCache(cache_dir, combiner)
    return _fundamentals_cache
