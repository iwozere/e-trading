"""
Data caching utilities.

This module provides file-based caching functionality for market data,
supporting multiple formats and configurable cache paths.
"""

import shutil
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging
import json

# Import cache directory setting
try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

_logger = logging.getLogger(__name__)


class DataCache:
    """
    File-based cache for market data with configurable storage.

    Supports multiple data formats (CSV, Parquet) and provides
    automatic cache management and cleanup.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = DATA_CACHE_DIR,
        max_size_gb: float = 10.0,
        compression: str = "snappy",
        partition_by: Optional[List[str]] = None,
        retention_days: int = 365,
        cleanup_interval_hours: int = 24
    ):
        """
        Initialize data cache.

        Args:
            cache_dir: Base directory for cache storage
            max_size_gb: Maximum cache size in GB
            compression: Compression format for Parquet files
            partition_by: List of partition keys (e.g., ['provider', 'symbol', 'interval'])
            retention_days: Days to keep cached data
            cleanup_interval_hours: Hours between cleanup operations
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_gb = max_size_gb
        self.compression = compression
        self.partition_by = partition_by or ["provider", "symbol", "interval"]
        self.retention_days = retention_days
        self.cleanup_interval_hours = cleanup_interval_hours

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

        # Last cleanup time
        self.last_cleanup = self.metadata.get('last_cleanup', 0)

        # Perform cleanup if needed
        self._maybe_cleanup()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                _logger.warning("Failed to load cache metadata: %s", e)

        return {
            'files': {},
            'last_cleanup': 0,
            'total_size_bytes': 0
        }

    def _save_metadata(self):
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except IOError:
            _logger.exception("Failed to save cache metadata:")

    def get_cache_path(
        self,
        provider: str,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        file_format: str = "parquet"
    ) -> Path:
        """
        Get cache file path for specific data.

        Args:
            provider: Data provider name
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date for data range
            end_date: End date for data range
            file_format: File format (csv, parquet)

        Returns:
            Full path to cache file
        """
        # Create partition path
        partition_path = self.cache_dir
        for key in self.partition_by:
            if key == "provider":
                partition_path = partition_path / provider
            elif key == "symbol":
                partition_path = partition_path / symbol
            elif key == "interval":
                partition_path = partition_path / interval

        partition_path.mkdir(parents=True, exist_ok=True)

        # Create filename
        if start_date and end_date:
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            filename = f"{symbol}_{interval}_{start_str}_{end_str}.{file_format}"
        else:
            filename = f"{symbol}_{interval}.{file_format}"

        return partition_path / filename

    def get(
        self,
        provider: str,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        file_format: str = "parquet"
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache.

        Args:
            provider: Data provider name
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date for data range
            end_date: End date for data range
            file_format: File format (csv, parquet)

        Returns:
            DataFrame if found and valid, None otherwise
        """
        cache_path = self.get_cache_path(provider, symbol, interval, start_date, end_date, file_format)

        if not cache_path.exists():
            return None

        # Check if file is in metadata
        file_key = str(cache_path.relative_to(self.cache_dir))
        if file_key not in self.metadata['files']:
            return None

        # Check if file is expired
        file_info = self.metadata['files'][file_key]
        if datetime.now().timestamp() - file_info['created_at'] > self.retention_days * 24 * 3600:
            self._remove_file(cache_path, file_key)
            return None

        try:
            # Load data
            if file_format == "parquet":
                df = pd.read_parquet(cache_path)
            elif file_format == "csv":
                df = pd.read_csv(cache_path)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                _logger.error("Unsupported file format: %s", file_format)
                return None

            # Validate data
            if df.empty:
                _logger.warning("Cached file is empty: %s", cache_path)
                self._remove_file(cache_path, file_key)
                return None

            # Update access time
            self.metadata['files'][file_key]['last_accessed'] = datetime.now().timestamp()
            self._save_metadata()

            _logger.debug("Retrieved data from cache: %s", cache_path)
            return df

        except Exception as e:
            _logger.error("Failed to load cached data from %s: %s", cache_path, e)
            self._remove_file(cache_path, file_key)
            return None

    def put(
        self,
        df: pd.DataFrame,
        provider: str,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        file_format: str = "parquet",
        overwrite: bool = True
    ) -> bool:
        """
        Store data in cache.

        Args:
            df: DataFrame to cache
            provider: Data provider name
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date for data range
            end_date: End date for data range
            file_format: File format (csv, parquet)
            overwrite: Whether to overwrite existing files

        Returns:
            True if successful, False otherwise
        """
        if df.empty:
            _logger.warning("Cannot cache empty DataFrame")
            return False

        cache_path = self.get_cache_path(provider, symbol, interval, start_date, end_date, file_format)

        # Check if file exists and overwrite is disabled
        if cache_path.exists() and not overwrite:
            _logger.debug("Cache file exists and overwrite disabled: %s", cache_path)
            return False

        try:
            # Ensure directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Save data
            if file_format == "parquet":
                df.to_parquet(cache_path, compression=self.compression, index=False)
            elif file_format == "csv":
                df.to_csv(cache_path, index=False)
            else:
                _logger.error("Unsupported file format: %s", file_format)
                return False

            # Update metadata
            file_key = str(cache_path.relative_to(self.cache_dir))
            file_size = cache_path.stat().st_size

            self.metadata['files'][file_key] = {
                'provider': provider,
                'symbol': symbol,
                'interval': interval,
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None,
                'file_format': file_format,
                'file_size': file_size,
                'created_at': datetime.now().timestamp(),
                'last_accessed': datetime.now().timestamp(),
                'rows': len(df),
                'columns': list(df.columns)
            }

            # Update total size
            if file_key in self.metadata['files']:
                old_size = self.metadata['files'][file_key]['file_size']
                self.metadata['total_size_bytes'] -= old_size

            self.metadata['total_size_bytes'] += file_size
            self._save_metadata()

            # Check if we need to cleanup due to size
            if self.metadata['total_size_bytes'] > self.max_size_gb * 1024**3:
                self._cleanup_by_size()

            _logger.debug("Cached data: %s (%d rows)", cache_path, len(df))
            return True

        except Exception as e:
            _logger.error("Failed to cache data to %s: %s", cache_path, e)
            return False

    def exists(
        self,
        provider: str,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        file_format: str = "parquet"
    ) -> bool:
        """Check if data exists in cache."""
        cache_path = self.get_cache_path(provider, symbol, interval, start_date, end_date, file_format)
        return cache_path.exists()

    def remove(
        self,
        provider: str,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        file_format: str = "parquet"
    ) -> bool:
        """Remove data from cache."""
        cache_path = self.get_cache_path(provider, symbol, interval, start_date, end_date, file_format)
        file_key = str(cache_path.relative_to(self.cache_dir))
        return self._remove_file(cache_path, file_key)

    def _remove_file(self, cache_path: Path, file_key: str) -> bool:
        """Remove a file from cache and update metadata."""
        try:
            if cache_path.exists():
                # Update total size
                if file_key in self.metadata['files']:
                    file_size = self.metadata['files'][file_key]['file_size']
                    self.metadata['total_size_bytes'] -= file_size

                # Remove file
                cache_path.unlink()

                # Remove from metadata
                if file_key in self.metadata['files']:
                    del self.metadata['files'][file_key]

                self._save_metadata()
                return True
        except Exception as e:
            _logger.error("Failed to remove cache file %s: %s", cache_path, e)

        return False

    def clear(self, provider: Optional[str] = None, symbol: Optional[str] = None):
        """Clear cache for specific provider/symbol or entire cache."""
        if provider is None and symbol is None:
            # Clear entire cache
            try:
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.metadata = {'files': {}, 'last_cleanup': 0, 'total_size_bytes': 0}
                self._save_metadata()
                _logger.info("Cleared entire cache")
            except Exception:
                _logger.exception("Failed to clear cache:")
        else:
            # Clear specific provider/symbol
            for file_key, file_info in list(self.metadata['files'].items()):
                if provider and file_info['provider'] != provider:
                    continue
                if symbol and file_info['symbol'] != symbol:
                    continue

                cache_path = self.cache_dir / file_key
                self._remove_file(cache_path, file_key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_files = len(self.metadata['files'])
        total_size_gb = self.metadata['total_size_bytes'] / (1024**3)

        # Count by provider
        providers = {}
        for file_info in self.metadata['files'].values():
            provider = file_info['provider']
            if provider not in providers:
                providers[provider] = {'files': 0, 'size_bytes': 0}
            providers[provider]['files'] += 1
            providers[provider]['size_bytes'] += file_info['file_size']

        # Convert provider sizes to GB
        for provider_info in providers.values():
            provider_info['size_gb'] = provider_info['size_bytes'] / (1024**3)

        return {
            'total_files': total_files,
            'total_size_gb': total_size_gb,
            'max_size_gb': self.max_size_gb,
            'providers': providers,
            'last_cleanup': datetime.fromtimestamp(self.metadata['last_cleanup']).isoformat() if self.metadata['last_cleanup'] > 0 else None
        }

    def _maybe_cleanup(self):
        """Perform cleanup if enough time has passed."""
        now = datetime.now().timestamp()
        if now - self.last_cleanup > self.cleanup_interval_hours * 3600:
            self._cleanup()

    def _cleanup(self):
        """Clean up expired and oversized cache files."""
        _logger.info("Starting cache cleanup")

        # Cleanup expired files
        now = datetime.now().timestamp()
        expired_files = []

        for file_key, file_info in self.metadata['files'].items():
            if now - file_info['created_at'] > self.retention_days * 24 * 3600:
                expired_files.append(file_key)

        for file_key in expired_files:
            cache_path = self.cache_dir / file_key
            self._remove_file(cache_path, file_key)

        if expired_files:
            _logger.info("Removed %d expired files", len(expired_files))

        # Cleanup by size if needed
        if self.metadata['total_size_bytes'] > self.max_size_gb * 1024**3:
            self._cleanup_by_size()

        # Update cleanup time
        self.last_cleanup = now
        self.metadata['last_cleanup'] = now
        self._save_metadata()

        _logger.info("Cache cleanup completed")

    def _cleanup_by_size(self):
        """Cleanup cache files to reduce size below limit."""
        target_size = self.max_size_gb * 1024**3 * 0.8  # Target 80% of max size

        if self.metadata['total_size_bytes'] <= target_size:
            return

        # Sort files by last access time (oldest first)
        files_by_access = sorted(
            self.metadata['files'].items(),
            key=lambda x: x[1]['last_accessed']
        )

        # Remove files until we're under target size
        for file_key, file_info in files_by_access:
            if self.metadata['total_size_bytes'] <= target_size:
                break

            cache_path = self.cache_dir / file_key
            self._remove_file(cache_path, file_key)

        _logger.info("Cache size reduced to %.2f GB", self.metadata['total_size_bytes'] / (1024**3))


# Global cache instance
_cache = None


def get_cache() -> DataCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = DataCache()
    return _cache


def configure_cache(
    cache_dir: Union[str, Path] = DATA_CACHE_DIR,
    max_size_gb: float = 10.0,
    **kwargs
):
    """Configure global cache settings."""
    global _cache
    _cache = DataCache(cache_dir=cache_dir, max_size_gb=max_size_gb, **kwargs)
