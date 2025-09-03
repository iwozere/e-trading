"""
File-based caching system with hierarchical structure.

This module provides a file-based caching system that organizes data in a
hierarchical structure: provider/symbol/timframe/year for efficient data
management and retrieval.

Features:
- Hierarchical file organization (provider/symbol/timframe/year)
- Support for CSV and Parquet formats
- Automatic data compression
- Cache invalidation strategies
- Performance metrics
- Year-based data partitioning
"""

import os
import time
import hashlib
import json
import pickle
import gzip
from typing import Any, Optional, Dict, List, Union, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import threading
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

_logger = logging.getLogger(__name__)


@dataclass
class FileCacheMetrics:
    """File cache performance metrics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    compression_ratio: float = 1.0
    avg_response_time_ms: float = 0.0
    files_created: int = 0
    files_deleted: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_operations(self) -> int:
        """Total cache operations."""
        return self.hits + self.misses + self.sets + self.deletes


class FileCacheCompressor:
    """Handles data compression for file-based caching."""

    def __init__(self, compression_level: int = 3):
        """
        Initialize compressor.

        Args:
            compression_level: Compression level (1-22 for zstd, 1-9 for gzip)
        """
        self.compression_level = compression_level
        self.use_zstd = ZSTD_AVAILABLE

    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        if self.use_zstd:
            return zstd.compress(data, level=self.compression_level)
        else:
            return gzip.compress(data, compresslevel=self.compression_level)

    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        if self.use_zstd:
            return zstd.decompress(data)
        else:
            return gzip.decompress(data)

    def get_compression_ratio(self, original: bytes, compressed: bytes) -> float:
        """Calculate compression ratio."""
        return len(compressed) / len(original) if original else 1.0


class FileCacheInvalidationStrategy:
    """Base class for cache invalidation strategies."""

    def should_invalidate(self, file_path: Path, metadata: Dict[str, Any]) -> bool:
        """Determine if a cache file should be invalidated."""
        raise NotImplementedError


class TimeBasedInvalidation(FileCacheInvalidationStrategy):
    """Time-based cache invalidation."""

    def __init__(self, max_age_hours: int = 24):
        """
        Initialize time-based invalidation.

        Args:
            max_age_hours: Maximum age of cache files in hours
        """
        self.max_age_hours = max_age_hours

    def should_invalidate(self, file_path: Path, metadata: Dict[str, Any]) -> bool:
        """Check if file is too old."""
        if not file_path.exists():
            return True

        file_age = time.time() - file_path.stat().st_mtime
        return file_age > (self.max_age_hours * 3600)


class SmartExpirationInvalidation(FileCacheInvalidationStrategy):
    """Smart expiration invalidation based on data year and current date."""

    def __init__(self, current_year_update_days: int = 30):
        """
        Initialize smart expiration invalidation.

        Args:
            current_year_update_days: Days to wait before updating current year data
        """
        self.current_year_update_days = current_year_update_days

    def should_invalidate(self, file_path: Path, metadata: Dict[str, Any]) -> bool:
        """
        Smart expiration logic:
        - Previous years data: NEVER expire (keep forever)
        - Current year data: Update every 30 days (configurable)
        """
        if not file_path.exists():
            return True

        # Get the year from metadata
        data_year = metadata.get('year')
        if data_year is None:
            # If we can't determine year, use default behavior
            return False

        current_year = datetime.now().year

        # Previous years data: NEVER expire
        if data_year < current_year:
            return False

        # Current year data: Check if it needs updating
        if data_year == current_year:
            # Check if the file is older than the update interval
            file_age_days = (time.time() - file_path.stat().st_mtime) / (24 * 3600)
            return file_age_days > self.current_year_update_days

        # Future years data: This shouldn't happen, but treat as current year
        return False


class VersionBasedInvalidation(FileCacheInvalidationStrategy):
    """Version-based cache invalidation."""

    def __init__(self, current_version: str = "1.0.0"):
        """
        Initialize version-based invalidation.

        Args:
            current_version: Current data version
        """
        self.current_version = current_version

    def should_invalidate(self, file_path: Path, metadata: Dict[str, Any]) -> bool:
        """Check if file version is outdated."""
        cached_version = metadata.get('version', '0.0.0')
        return cached_version != self.current_version


class FileBasedCache:
    """
    File-based caching system with hierarchical structure.

    Organizes data as: provider/symbol/timframe/year/
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "d:/data-cache",
        max_size_gb: float = 10.0,
        retention_days: int = 30,
        invalidation_strategies: Optional[List[FileCacheInvalidationStrategy]] = None,
        compression_enabled: bool = True,
        compression_level: int = 3,
                        default_format: str = "csv"
    ):
        """
        Initialize file-based cache.

        Args:
            cache_dir: Base directory for cache files
            max_size_gb: Maximum cache size in GB
            retention_days: Days to retain cache files
            invalidation_strategies: List of invalidation strategies
            compression_enabled: Enable data compression
            compression_level: Compression level
            default_format: Default file format (csv, parquet)
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_gb = max_size_gb
        self.retention_days = retention_days
        self.compression_enabled = compression_enabled
        self.default_format = default_format

        # Initialize components
        self.compressor = FileCacheCompressor(compression_level) if compression_enabled else None
        self.metrics = FileCacheMetrics()
        self.lock = threading.RLock()

        # Set up invalidation strategies
        self.invalidation_strategies = invalidation_strategies or [
            SmartExpirationInvalidation(current_year_update_days=30)
        ]

        # Create cache directory structure
        self._ensure_cache_dir()

        _logger.info(f"File-based cache initialized at {self.cache_dir}")

    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, provider: str, symbol: str, interval: str, year: int) -> Path:
        """
        Get cache file path for the hierarchical structure.

        Args:
            provider: Data provider name
            symbol: Trading symbol
            interval: Time interval
            year: Year for data

        Returns:
            Path to cache file
        """
        # Create hierarchical path: provider/symbol/interval/year/
        cache_path = self.cache_dir / provider / symbol / interval / str(year)
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def _get_metadata_path(self, cache_path: Path) -> Path:
        """Get metadata file path."""
        return cache_path / "metadata.json"

    def _get_data_path(self, cache_path: Path, format: str = "csv") -> Path:
        """Get data file path."""
        if format == "parquet":
            return cache_path / "data.parquet"
        else:
            return cache_path / "data.csv"

    def _generate_cache_key(self, provider: str, symbol: str, interval: str,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> str:
        """Generate cache key for data."""
        key_parts = [provider, symbol, interval]

        if start_date:
            key_parts.append(start_date.strftime("%Y%m%d"))
        if end_date:
            key_parts.append(end_date.strftime("%Y%m%d"))

        return "_".join(key_parts)

    def _save_metadata(self, cache_path: Path, metadata: Dict[str, Any]):
        """Save metadata to file."""
        metadata_path = self._get_metadata_path(cache_path)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def _load_metadata(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """Load metadata from file."""
        metadata_path = self._get_metadata_path(cache_path)
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            _logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
            return None

    def _should_invalidate(self, cache_path: Path, metadata: Dict[str, Any]) -> bool:
        """Check if cache should be invalidated."""
        for strategy in self.invalidation_strategies:
            if strategy.should_invalidate(cache_path, metadata):
                return True
        return False

    def _split_data_by_years(self, df: pd.DataFrame, start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> Dict[int, pd.DataFrame]:
        """
        Split DataFrame data by years for proper caching.

        Args:
            df: DataFrame with datetime index
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering

        Returns:
            Dictionary mapping year to DataFrame subset for that year
        """
        if df.empty:
            return {}

        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            _logger.warning("DataFrame index is not DatetimeIndex, cannot split by years")
            return {datetime.now().year: df}

        # Filter by date range if specified
        filtered_df = df.copy()
        if start_date:
            filtered_df = filtered_df[filtered_df.index >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df.index <= end_date]

        if filtered_df.empty:
            return {}

        # Split by years
        years_data = {}
        for year in filtered_df.index.year.unique():
            year_mask = filtered_df.index.year == year
            year_df = filtered_df[year_mask].copy()

            if not year_df.empty:
                years_data[year] = year_df
                _logger.debug(f"Split data for year {year}: {len(year_df)} rows")

        return years_data

    def _get_years_to_check(self, start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[int]:
        """
        Determine which years to check in cache based on the requested date range.

        Args:
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            List of years to check in cache
        """
        if start_date and end_date:
            # If both dates specified, check all years in between
            start_year = start_date.year
            end_year = end_date.year
            return list(range(start_year, end_year + 1))
        elif start_date:
            # If only start date, check from start year to current year
            start_year = start_date.year
            current_year = datetime.now().year
            return list(range(start_year, current_year + 1))
        elif end_date:
            # If only end date, check from a reasonable start year to end year
            end_year = end_date.year
            start_year = max(1990, end_year - 10)  # Go back up to 10 years
            return list(range(start_year, end_year + 1))
        else:
            # If no dates specified, check current year
            return [datetime.now().year]

    def migrate_existing_data(self) -> Dict[str, Any]:
        """
        Migrate existing cache data to the new year-split structure.
        This function detects data files that span multiple years and splits them.

        Returns:
            Dictionary with migration results
        """
        migration_results = {
            'files_migrated': 0,
            'years_created': 0,
            'errors': 0,
            'details': []
        }

        try:
            _logger.info("Starting cache data migration to year-split structure...")

            # Walk through all cache directories
            for provider_dir in self.cache_dir.iterdir():
                if not provider_dir.is_dir():
                    continue

                provider = provider_dir.name

                for symbol_dir in provider_dir.iterdir():
                    if not symbol_dir.is_dir():
                        continue

                    symbol = symbol_dir.name

                    for interval_dir in symbol_dir.iterdir():
                        if not interval_dir.is_dir():
                            continue

                        interval = interval_dir.name

                        for year_dir in interval_dir.iterdir():
                            if not year_dir.is_dir() or not year_dir.name.isdigit():
                                continue

                            year = int(year_dir.name)
                            data_path = self._get_data_path(year_dir, 'csv')  # Check CSV first

                            if not data_path.exists():
                                data_path = self._get_data_path(year_dir, 'parquet')  # Check Parquet

                            if not data_path.exists():
                                continue

                            # Load the data to check if it spans multiple years
                            try:
                                if data_path.suffix == '.parquet':
                                    df = pd.read_parquet(data_path)
                                else:
                                    # Try to load CSV with timestamp column first
                                    try:
                                        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
                                    except:
                                        try:
                                            df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
                                        except:
                                            df = pd.read_csv(data_path)

                                if df.empty or not isinstance(df.index, pd.DatetimeIndex):
                                    continue

                                # Check if this file contains data from multiple years
                                years_in_data = df.index.year.unique()
                                if len(years_in_data) > 1:
                                    _logger.info(f"Found multi-year data in {provider}/{symbol}/{interval}/{year}: {years_in_data}")

                                    # Split and migrate the data
                                    success = self._migrate_multi_year_file(
                                        df, provider, symbol, interval, year_dir, years_in_data
                                    )

                                    if success:
                                        migration_results['files_migrated'] += 1
                                        migration_results['years_created'] += len(years_in_data)
                                        migration_results['details'].append(f"{provider}/{symbol}/{interval}/{year} -> {years_in_data}")
                                    else:
                                        migration_results['errors'] += 1

                            except Exception as e:
                                _logger.warning(f"Error processing {provider}/{symbol}/{interval}/{year}: {e}")
                                migration_results['errors'] += 1
                                continue

            _logger.info(f"Cache migration completed: {migration_results['files_migrated']} files migrated, {migration_results['years_created']} years created")
            return migration_results

        except Exception as e:
            _logger.error(f"Error during cache migration: {e}")
            migration_results['errors'] += 1
            return migration_results

    def _migrate_multi_year_file(self, df: pd.DataFrame, provider: str, symbol: str,
                                interval: str, original_year_dir: Path, years_in_data: List[int]) -> bool:
        """
        Migrate a single multi-year file to year-split structure.

        Args:
            df: DataFrame containing multi-year data
            provider: Data provider name
            symbol: Trading symbol
            interval: Time interval
            original_year_dir: Original directory containing the multi-year data
            years_in_data: List of years found in the data

        Returns:
            True if migration successful, False otherwise
        """
        try:
            # Split data by years
            years_data = {}
            for year in years_in_data:
                year_mask = df.index.year == year
                year_df = df[year_mask].copy()

                if not year_df.empty:
                    years_data[year] = year_df

            # Save each year's data to its own directory
            for year, year_df in years_data.items():
                if year == int(original_year_dir.name):
                    # This year stays in the original directory, just update the data
                    data_path = self._get_data_path(original_year_dir, 'csv')
                    year_df.to_csv(data_path, compression='gzip' if self.compression_enabled else None)

                    # Update metadata
                    cache_metadata = {
                        'provider': provider,
                        'symbol': symbol,
                        'interval': interval,
                        'year': year,
                        'created_at': datetime.now().isoformat(),
                        'rows': len(year_df),
                        'columns': list(year_df.columns),
                        'format': 'csv',
                        'compression_enabled': self.compression_enabled,
                        'start_date': year_df.index.min().isoformat() if not year_df.index.empty else None,
                        'end_date': year_df.index.max().isoformat() if not year_df.index.empty else None,
                        'version': '1.0.0',
                        'migrated': True
                    }
                    self._save_metadata(original_year_dir, cache_metadata)

                else:
                    # Create new directory for this year
                    new_cache_path = self._get_cache_path(provider, symbol, interval, year)
                    data_path = self._get_data_path(new_cache_path, 'csv')

                    # Save data
                    year_df.to_csv(data_path, compression='gzip' if self.compression_enabled else None)

                    # Save metadata
                    cache_metadata = {
                        'provider': provider,
                        'symbol': symbol,
                        'interval': interval,
                        'year': year,
                        'created_at': datetime.now().isoformat(),
                        'rows': len(year_df),
                        'columns': list(year_df.columns),
                        'format': 'csv',
                        'compression_enabled': self.compression_enabled,
                        'start_date': year_df.index.min().isoformat() if not year_df.index.empty else None,
                        'end_date': year_df.index.max().isoformat() if not year_df.index.empty else None,
                        'version': '1.0.0',
                        'migrated': True
                    }
                    self._save_metadata(new_cache_path, cache_metadata)

                    _logger.debug(f"Created new year directory {year} with {len(year_df)} rows")

            return True

        except Exception as e:
            _logger.error(f"Error migrating multi-year file: {e}")
            return False

    def get(self, provider: str, symbol: str, interval: str,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            format: str = "parquet") -> Optional[pd.DataFrame]:
        """
        Get data from cache.

        Args:
            provider: Data provider name
            symbol: Trading symbol
            interval: Time interval
            start_date: Start date for data
            end_date: End date for data
            format: File format (csv, parquet)

        Returns:
            DataFrame if found and valid, None otherwise
        """
        start_time = time.time()

        try:
            with self.lock:
                # Determine which years we need to check based on the date range
                years_to_check = self._get_years_to_check(start_date, end_date)

                all_data = []
                found_data = False

                for year in years_to_check:
                    cache_path = self._get_cache_path(provider, symbol, interval, year)
                    data_path = self._get_data_path(cache_path, format)

                    if not data_path.exists():
                        continue

                    # Load and check metadata
                    metadata = self._load_metadata(cache_path)
                    if not metadata:
                        continue

                    # Check invalidation
                    if self._should_invalidate(data_path, metadata):
                        _logger.info(f"Cache invalidated for {provider}/{symbol}/{interval}/{year}")
                        self.delete(provider, symbol, interval, year)
                        continue

                    # Load data for this year
                    try:
                        if format == "parquet":
                            year_df = pd.read_parquet(data_path)
                        else:
                            # Try to load CSV with timestamp column first, fallback to datetime
                            try:
                                year_df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
                            except:
                                try:
                                    year_df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
                                except:
                                    # If neither column exists, load without setting index
                                    year_df = pd.read_csv(data_path)

                        if not year_df.empty:
                            all_data.append(year_df)
                            found_data = True
                            _logger.debug(f"Loaded data for year {year}: {len(year_df)} rows")
                    except Exception as e:
                        _logger.warning(f"Error loading data for year {year}: {e}")
                        continue

                if not found_data:
                    self.metrics.misses += 1
                    return None

                # Combine all years' data
                if len(all_data) == 1:
                    df = all_data[0]
                else:
                    df = pd.concat(all_data, axis=0)
                    df = df.sort_index()  # Ensure chronological order
                    # Remove duplicates if any
                    df = df[~df.index.duplicated(keep='last')]

                # Filter by date range if specified
                if start_date or end_date:
                    if start_date:
                        df = df[df.index >= start_date]
                    if end_date:
                        df = df[df.index <= end_date]

                self.metrics.hits += 1
                response_time = (time.time() - start_time) * 1000
                self.metrics.avg_response_time_ms = (
                    (self.metrics.avg_response_time_ms * (self.metrics.hits - 1) + response_time) /
                    self.metrics.hits
                )

                _logger.debug(f"Cache hit for {provider}/{symbol}/{interval} across {len(all_data)} years: {len(df)} total rows")
                return df

        except Exception as e:
            self.metrics.errors += 1
            _logger.error(f"Error reading from cache: {e}")
            return None

    def put(self, df: pd.DataFrame, provider: str, symbol: str, interval: str,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            format: str = "parquet", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store data in cache.

        Args:
            df: DataFrame to cache
            provider: Data provider name
            symbol: Trading symbol
            interval: Time interval
            start_date: Start date for data
            end_date: End date for data
            format: File format (csv, parquet)
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.lock:
                if df.empty:
                    _logger.warning(f"Empty DataFrame provided for {provider}/{symbol}/{interval}")
                    return False

                # Split data by years and cache each year separately
                years_data = self._split_data_by_years(df, start_date, end_date)

                total_rows_cached = 0
                files_created = 0

                for year, year_df in years_data.items():
                    if year_df.empty:
                        continue

                    cache_path = self._get_cache_path(provider, symbol, interval, year)
                    data_path = self._get_data_path(cache_path, format)

                    # Prepare metadata for this year
                    year_start = year_df.index.min() if not year_df.index.empty else None
                    year_end = year_df.index.max() if not year_df.index.empty else None

                    cache_metadata = {
                        'provider': provider,
                        'symbol': symbol,
                        'interval': interval,
                        'year': year,
                        'created_at': datetime.now().isoformat(),
                        'rows': len(year_df),
                        'columns': list(year_df.columns),
                        'format': format,
                        'compression_enabled': self.compression_enabled,
                        'start_date': year_start.isoformat() if year_start else None,
                        'end_date': year_end.isoformat() if year_end else None,
                        'version': '1.0.0'
                    }

                    if metadata:
                        cache_metadata.update(metadata)

                    # Save data for this year
                    if format == "parquet":
                        year_df.to_parquet(data_path, compression='snappy' if self.compression_enabled else None)
                    else:
                        year_df.to_csv(data_path, compression='gzip' if self.compression_enabled else None)

                    # Save metadata
                    self._save_metadata(cache_path, cache_metadata)

                    # Update metrics
                    total_rows_cached += len(year_df)
                    files_created += 1
                    file_size = data_path.stat().st_size
                    self.metrics.total_size_bytes += file_size

                    _logger.debug(f"Cached data for {provider}/{symbol}/{interval}/{year}: {len(year_df)} rows")

                # Update overall metrics
                self.metrics.sets += 1
                self.metrics.files_created += files_created

                _logger.info(f"Successfully cached {total_rows_cached} total rows across {files_created} year files for {provider}/{symbol}/{interval}")
                return True

        except Exception as e:
            self.metrics.errors += 1
            _logger.error(f"Error writing to cache: {e}")
            return False

    def delete(self, provider: str, symbol: str, interval: str, year: int) -> bool:
        """
        Delete cached data for a specific year.

        Args:
            provider: Data provider name
            symbol: Trading symbol
            interval: Time interval
            year: Year to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.lock:
                cache_path = self._get_cache_path(provider, symbol, interval, year)

                if cache_path.exists():
                    # Remove all files in the directory
                    for file_path in cache_path.iterdir():
                        if file_path.is_file():
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            self.metrics.total_size_bytes -= file_size

                    # Remove directory
                    cache_path.rmdir()
                    self.metrics.deletes += 1
                    self.metrics.files_deleted += 1

                    _logger.debug(f"Deleted cache for {provider}/{symbol}/{interval}/{year}")
                    return True

                return False

        except Exception as e:
            self.metrics.errors += 1
            _logger.error(f"Error deleting cache: {e}")
            return False

    def clear(self, provider: Optional[str] = None, symbol: Optional[str] = None,
              interval: Optional[str] = None) -> bool:
        """
        Clear cache for specific criteria.

        Args:
            provider: Data provider name (None for all)
            symbol: Trading symbol (None for all)
            interval: Time interval (None for all)

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.lock:
                if provider is None:
                    # Clear entire cache
                    for item in self.cache_dir.iterdir():
                        if item.is_dir():
                            self._clear_directory(item)
                else:
                    provider_path = self.cache_dir / provider
                    if symbol is None:
                        if provider_path.exists():
                            self._clear_directory(provider_path)
                    else:
                        symbol_path = provider_path / symbol
                        if interval is None:
                            if symbol_path.exists():
                                self._clear_directory(symbol_path)
                        else:
                            interval_path = symbol_path / interval
                            if interval_path.exists():
                                self._clear_directory(interval_path)

                return True

        except Exception as e:
            _logger.error(f"Error clearing cache: {e}")
            return False

    def _clear_directory(self, directory: Path):
        """Recursively clear directory."""
        for item in directory.iterdir():
            if item.is_file():
                file_size = item.stat().st_size
                item.unlink()
                self.metrics.total_size_bytes -= file_size
                self.metrics.files_deleted += 1
            elif item.is_dir():
                self._clear_directory(item)
        directory.rmdir()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            stats = asdict(self.metrics)

            # Add computed properties
            stats['hit_rate'] = self.metrics.hit_rate
            stats['total_operations'] = self.metrics.total_operations

            stats['cache_dir'] = str(self.cache_dir)
            stats['cache_size_gb'] = self.metrics.total_size_bytes / (1024**3)
            stats['max_size_gb'] = self.max_size_gb
            stats['compression_enabled'] = self.compression_enabled
            stats['default_format'] = self.default_format

            # Count files and directories
            file_count = 0
            dir_count = 0
            for root, dirs, files in os.walk(self.cache_dir):
                dir_count += len(dirs)
                file_count += len(files)

            stats['file_count'] = file_count
            stats['directory_count'] = dir_count

            return stats

    def cleanup_old_files(self) -> int:
        """
        Clean up old cache files based on retention policy.

        Returns:
            Number of files deleted
        """
        deleted_count = 0
        cutoff_time = time.time() - (self.retention_days * 24 * 3600)

        try:
            with self.lock:
                for root, dirs, files in os.walk(self.cache_dir):
                    for file in files:
                        file_path = Path(root) / file
                        if file_path.stat().st_mtime < cutoff_time:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            self.metrics.total_size_bytes -= file_size
                            self.metrics.files_deleted += 1
                            deleted_count += 1

                            # Remove empty directories
                            try:
                                Path(root).rmdir()
                            except OSError:
                                pass  # Directory not empty

                _logger.info(f"Cleaned up {deleted_count} old cache files")
                return deleted_count

        except Exception as e:
            _logger.error(f"Error during cleanup: {e}")
            return deleted_count

    def get_cache_info(self, provider: str, symbol: str, interval: str) -> Dict[str, Any]:
        """
        Get information about cached data for a specific symbol.

        Args:
            provider: Data provider name
            symbol: Trading symbol
            interval: Time interval

        Returns:
            Dictionary with cache information
        """
        info = {
            'provider': provider,
            'symbol': symbol,
            'interval': interval,
            'years_available': [],
            'total_rows': 0,
            'total_size_bytes': 0,
            'last_updated': None
        }

        try:
            symbol_path = self.cache_dir / provider / symbol / interval

            if symbol_path.exists():
                for year_dir in symbol_path.iterdir():
                    if year_dir.is_dir() and year_dir.name.isdigit():
                        year = int(year_dir.name)
                        metadata = self._load_metadata(year_dir)

                        if metadata:
                            info['years_available'].append(year)
                            info['total_rows'] += metadata.get('rows', 0)

                            # Get file size
                            data_path = self._get_data_path(year_dir, metadata.get('format', 'parquet'))
                            if data_path.exists():
                                info['total_size_bytes'] += data_path.stat().st_size

                            # Update last updated
                            created_at = metadata.get('created_at')
                            if created_at:
                                try:
                                    created_dt = datetime.fromisoformat(created_at)
                                    if info['last_updated'] is None or created_dt > info['last_updated']:
                                        info['last_updated'] = created_dt
                                except:
                                    pass

                info['years_available'].sort()

        except Exception as e:
            _logger.error(f"Error getting cache info: {e}")

        return info


# Global cache instance
_file_cache_instance: Optional[FileBasedCache] = None


def get_file_cache(
    cache_dir: Union[str, Path] = "d:/data-cache",
    max_size_gb: float = 10.0,
    retention_days: int = 30,
    compression_enabled: bool = True
) -> FileBasedCache:
    """
    Get or create global file cache instance.

    Args:
        cache_dir: Base directory for cache files
        max_size_gb: Maximum cache size in GB
        retention_days: Days to retain cache files
        compression_enabled: Enable data compression

    Returns:
        FileBasedCache instance
    """
    global _file_cache_instance

    if _file_cache_instance is None:
        _file_cache_instance = FileBasedCache(
            cache_dir=cache_dir,
            max_size_gb=max_size_gb,
            retention_days=retention_days,
            compression_enabled=compression_enabled
        )

    return _file_cache_instance


def configure_file_cache(
    cache_dir: Union[str, Path] = "d:/data-cache",
    max_size_gb: float = 10.0,
    retention_days: int = 100,
    compression_enabled: bool = False,
    invalidation_strategies: Optional[List[FileCacheInvalidationStrategy]] = None,
    auto_migrate: bool = True
) -> FileBasedCache:
    """
    Configure and return file cache instance.

    Args:
        cache_dir: Base directory for cache files
        max_size_gb: Maximum cache size in GB
        retention_days: Days to retain cache files
        compression_enabled: Enable data compression
        invalidation_strategies: List of invalidation strategies
        auto_migrate: Automatically migrate existing data to year-split structure

    Returns:
        Configured FileBasedCache instance
    """
    global _file_cache_instance

    _file_cache_instance = FileBasedCache(
        cache_dir=cache_dir,
        max_size_gb=max_size_gb,
        retention_days=retention_days,
        compression_enabled=compression_enabled,
        invalidation_strategies=invalidation_strategies
    )

    # Auto-migrate existing data if requested
    if auto_migrate:
        try:
            migration_results = _file_cache_instance.migrate_existing_data()
            if migration_results['files_migrated'] > 0:
                _logger.info(f"Auto-migration completed: {migration_results['files_migrated']} files migrated")
        except Exception as e:
            _logger.warning(f"Auto-migration failed: {e}")

    return _file_cache_instance
