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
import json
import gzip
from typing import Any, Optional, Dict, List, Union
from datetime import datetime
from pathlib import Path
import logging
import threading
from dataclasses import dataclass, asdict
import pandas as pd
from io import StringIO

# Import cache directory setting
from config.donotshare.donotshare import DATA_CACHE_DIR

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

_logger = logging.getLogger(__name__)


class CSVFormatConventions:
    """Standardized CSV format for financial data."""

    REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    OPTIONAL_COLUMNS = ['provider_download_ts']
    ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """Validate DataFrame follows CSV conventions."""
        if df.empty:
            return False

        # Check if we have timestamp column or datetime index
        has_timestamp_col = 'timestamp' in df.columns
        has_datetime_index = isinstance(df.index, pd.DatetimeIndex)

        if not has_timestamp_col and not has_datetime_index:
            return False

        # Check required columns (excluding timestamp if it's in index)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            return False

        # Check timestamp is datetime (either column or index)
        if has_timestamp_col:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                return False
        elif has_datetime_index:
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                return False

        # Check numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False

        return True

    @staticmethod
    def standardize_dataframe(df: pd.DataFrame, provider: str = None) -> pd.DataFrame:
        """Standardize DataFrame to CSV conventions."""
        # Ensure timestamp column exists and is datetime
        if 'timestamp' not in df.columns:
            if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            else:
                raise ValueError("DataFrame must have timestamp column or datetime index")

        # Convert timestamp to UTC if not already
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

        # Ensure required columns exist
        for col in CSVFormatConventions.REQUIRED_COLUMNS:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0.0  # Default volume
                else:
                    raise ValueError(f"Missing required column: {col}")

        # Add provider download timestamp if not present
        if 'provider_download_ts' not in df.columns and provider:
            df['provider_download_ts'] = pd.Timestamp.now(tz='UTC')

        # Sort by timestamp and remove duplicates
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')

        # Select only standard columns in correct order
        final_cols = CSVFormatConventions.ALL_COLUMNS
        available_cols = [col for col in final_cols if col in df.columns]

        return df[available_cols]


class SafeCSVAppender:
    """Safely append data to CSV files using atomic operations."""

    @staticmethod
    def append_to_csv(file_path: Path, new_data: pd.DataFrame,
                     backup_existing: bool = True) -> bool:
        """
        Safely append new data to existing CSV file.

        Args:
            file_path: Path to CSV file
            new_data: New data to append
            backup_existing: Whether to backup existing file

        Returns:
            True if successful, False otherwise
        """
        temp_path = None
        try:
            # Read existing data if file exists
            existing_data = None
            if file_path.exists():
                if backup_existing:
                    backup_path = file_path.with_suffix('.csv.backup')
                    backup_path.write_text(file_path.read_text())

                # Handle compressed files
                if file_path.suffix == '.gz':
                    existing_data = pd.read_csv(file_path, parse_dates=['timestamp'], compression='gzip')
                else:
                    existing_data = pd.read_csv(file_path, parse_dates=['timestamp'])

            # Combine data
            if existing_data is not None:
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                # Remove duplicates and sort
                combined_data = combined_data.sort_values('timestamp').drop_duplicates(
                    subset=['timestamp'], keep='last'
                )
            else:
                combined_data = new_data

            # Write to temporary file first
            temp_path = file_path.with_suffix('.tmp')
            combined_data.to_csv(temp_path, index=False)

            # Atomic replace
            os.replace(temp_path, file_path)

            return True

        except Exception:
            _logger.exception("Error appending to CSV %s:", file_path)
            # Clean up temp file if it exists
            if temp_path and temp_path.exists():
                temp_path.unlink()
            return False


class SmartDataAppender:
    """Intelligent data appending with overlap detection."""

    @staticmethod
    def append_new_data(file_path: Path, new_data: pd.DataFrame,
                       metadata: 'CacheMetadata') -> tuple[bool, int]:
        """
        Append only new data to existing CSV file.

        Returns:
            (success, rows_added)
        """
        try:
            if not file_path.exists():
                # File doesn't exist, create new
                new_data.to_csv(file_path, index=False)
                metadata.update_integrity_info(new_data, file_path)
                return True, len(new_data)

            # Read existing data
            existing_data = pd.read_csv(file_path, parse_dates=['timestamp'])

            # Find new data (after last existing timestamp)
            if not existing_data.empty:
                last_existing_ts = existing_data['timestamp'].max()
                new_data_filtered = new_data[new_data['timestamp'] > last_existing_ts]
            else:
                new_data_filtered = new_data

            if new_data_filtered.empty:
                return True, 0  # No new data to add

            # Append new data
            combined_data = pd.concat([existing_data, new_data_filtered], ignore_index=True)
            combined_data = combined_data.sort_values('timestamp').drop_duplicates(
                subset=['timestamp'], keep='last'
            )

            # Write using safe append
            success = SafeCSVAppender.append_to_csv(file_path, combined_data)
            if success:
                metadata.update_integrity_info(combined_data, file_path)
                return True, len(new_data_filtered)

            return False, 0

        except Exception:
            _logger.exception("Error in smart data append:")
            return False, 0


@dataclass
class CacheMetadata:
    """Enhanced cache metadata with integrity checks."""

    # Basic info
    provider: str
    symbol: str
    interval: str
    year: int
    format: str = "csv"
    version: str = "1.0.0"

    # Data integrity
    first_timestamp: Optional[str] = None
    last_timestamp: Optional[str] = None
    row_count: int = 0
    file_size_bytes: int = 0

    # Provider sync info
    last_sync_timestamp: Optional[str] = None
    provider_last_update: Optional[str] = None

    # Schema info
    columns: List[str] = None
    compression_enabled: bool = False

    # Timestamps
    created_at: str = None
    last_modified: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_modified is None:
            self.last_modified = self.created_at
        if self.columns is None:
            self.columns = CSVFormatConventions.REQUIRED_COLUMNS

    def update_integrity_info(self, df: pd.DataFrame, file_path: Path):
        """Update integrity information from DataFrame and file."""
        if not df.empty:
            self.first_timestamp = df['timestamp'].min().isoformat()
            self.last_timestamp = df['timestamp'].max().isoformat()
            self.row_count = len(df)

        if file_path.exists():
            self.file_size_bytes = file_path.stat().st_size

        self.last_modified = datetime.now().isoformat()

    def validate_integrity(self, file_path: Path) -> bool:
        """Validate file integrity against metadata."""
        if not file_path.exists():
            return False

        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'])

            # Check row count
            if len(df) != self.row_count:
                return False

            # Check timestamp range
            if not df.empty:
                if (df['timestamp'].min().isoformat() != self.first_timestamp or
                    df['timestamp'].max().isoformat() != self.last_timestamp):
                    return False

            return True

        except Exception:
            return False


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

    def save_compressed_csv(self, df: pd.DataFrame, file_path: Path,
                           compression: str = 'gzip') -> bool:
        """Save DataFrame as compressed CSV."""
        try:
            if compression == 'gzip':
                df.to_csv(file_path.with_suffix('.csv.gz'), index=False, compression='gzip')
            elif compression == 'zstandard' and ZSTD_AVAILABLE:
                # Custom zstandard compression
                csv_content = df.to_csv(index=False)
                compressed = zstd.compress(csv_content.encode('utf-8'), level=self.compression_level)
                file_path.with_suffix('.csv.zst').write_bytes(compressed)
            else:
                df.to_csv(file_path, index=False)
            return True
        except Exception:
            _logger.exception("Error saving compressed CSV:")
            return False

    def load_compressed_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load compressed CSV file."""
        try:
            if file_path.suffix == '.gz':
                return pd.read_csv(file_path, parse_dates=['timestamp'], compression='gzip')
            elif file_path.suffix == '.zst':
                compressed = file_path.read_bytes()
                csv_content = zstd.decompress(compressed).decode('utf-8')
                return pd.read_csv(StringIO(csv_content), parse_dates=['timestamp'])
            else:
                return pd.read_csv(file_path, parse_dates=['timestamp'])
        except Exception:
            _logger.exception("Error loading compressed CSV:")
            return None


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

        # Get the year from metadata - handle both old and new metadata formats
        data_year = None

        # Try to get year from the metadata directly (old format)
        if 'year' in metadata:
            data_year = metadata.get('year')
        # Try to get year from the file path (new format)
        elif 'years' in metadata and file_path.exists():
            # Extract year from filename
            try:
                filename = file_path.stem
                if filename.endswith('.csv'):
                    filename = filename[:-4]  # Remove .csv from stem
                data_year = int(filename)
            except (ValueError, TypeError):
                pass

        if data_year is None:
            # If we can't determine year, use default behavior
            return False

        # Ensure data_year is an integer (JSON loads strings)
        try:
            data_year = int(data_year)
        except (ValueError, TypeError):
            # If we can't convert to int, use default behavior
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
        cache_dir: Union[str, Path] = DATA_CACHE_DIR,
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

        _logger.info("File-based cache initialized at %s", self.cache_dir)

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
        # Create hierarchical path: provider/symbol/interval/
        cache_path = self.cache_dir / provider / symbol / interval
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def _get_metadata_path(self, cache_path: Path) -> Path:
        """Get metadata file path."""
        return cache_path / "metadata.json"

    def _get_data_path(self, cache_path: Path, format: str = "csv", year: Optional[int] = None) -> Path:
        """Get data file path."""
        if format == "parquet":
            if year:
                return cache_path / f"{year}.parquet"
            else:
                return cache_path / "data.parquet"
        else:
            if year:
                # Check for compressed files first if compression is enabled
                if hasattr(self, 'compression_enabled') and self.compression_enabled:
                    compressed_path = cache_path / f"{year}.csv.gz"
                    if compressed_path.exists():
                        return compressed_path
                return cache_path / f"{year}.csv"
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
            _logger.warning("Failed to load metadata from %s: %s", metadata_path, e)
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
            df: DataFrame with timestamp column or datetime index
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering

        Returns:
            Dictionary mapping year to DataFrame subset for that year
        """
        if df.empty:
            return {}

        # Handle both timestamp column and datetime index
        if 'timestamp' in df.columns:
            # Use timestamp column
            timestamps = df['timestamp']
        elif isinstance(df.index, pd.DatetimeIndex):
            # Use datetime index
            timestamps = df.index
        else:
            _logger.warning("DataFrame has neither timestamp column nor DatetimeIndex, cannot split by years")
            return {datetime.now().year: df}

        # Filter by date range if specified
        filtered_df = df.copy()
        if start_date:
            filtered_df = filtered_df[timestamps >= start_date]
        if end_date:
            filtered_df = filtered_df[timestamps <= end_date]

        if filtered_df.empty:
            return {}

        # Split by years
        years_data = {}
        # Handle both Series (with .dt accessor) and DatetimeIndex (direct .year access)
        if hasattr(timestamps, 'dt'):
            # Series with .dt accessor
            years = timestamps.dt.year.unique()
            for year in years:
                year_mask = timestamps.dt.year == year
                year_df = filtered_df[year_mask].copy()
                if not year_df.empty:
                    years_data[year] = year_df
                    _logger.debug("Split data for year %s: %d rows", year, len(year_df))
        else:
            # DatetimeIndex with direct .year access
            years = timestamps.year.unique()
            for year in years:
                year_mask = timestamps.year == year
                year_df = filtered_df[year_mask].copy()
                if not year_df.empty:
                    years_data[year] = year_df
                    _logger.debug("Split data for year %s: %d rows", year, len(year_df))

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
            _logger.debug("Both dates specified: start_year=%s (%s), end_year=%s (%s)", start_year, type(start_year), end_year, type(end_year))
            return list(range(start_year, end_year + 1))
        elif start_date:
            # If only start date, check from start year to current year
            start_year = start_date.year
            current_year = datetime.now().year
            _logger.debug("Only start date: start_year=%s (%s), current_year=%s (%s)", start_year, type(start_year), current_year, type(current_year))
            return list(range(start_year, current_year + 1))
        elif end_date:
            # If only end date, check from a reasonable start year to end year
            end_year = end_date.year
            start_year = max(1990, end_year - 10)  # Go back up to 10 years
            _logger.debug("Only end date: start_year=%s (%s), end_year=%s (%s)", start_year, type(start_year), end_year, type(end_year))
            return list(range(start_year, end_year + 1))
        else:
            # If no dates specified, check all available years in the cache
            # This is more robust for testing and general use
            available_years = []
            if self.cache_dir.exists():
                for provider_dir in self.cache_dir.iterdir():
                    if not provider_dir.is_dir():
                        continue
                    for symbol_dir in provider_dir.iterdir():
                        if not symbol_dir.is_dir():
                            continue
                        for interval_dir in symbol_dir.iterdir():
                            if not interval_dir.is_dir():
                                continue
                            # Look for year files in the interval directory
                            for file_path in interval_dir.iterdir():
                                if file_path.is_file():
                                    # Extract year from filename (e.g., "2023.csv", "2024.csv.gz")
                                    filename = file_path.stem  # Remove extension
                                    _logger.debug("Checking file: %s, stem: %s, isdigit: %s", file_path, filename, filename.isdigit())
                                    if filename.isdigit():
                                        year_int = int(filename)
                                        _logger.debug("Extracted year: %s (%s)", year_int, type(year_int))
                                        available_years.append(year_int)

            _logger.debug("Available years found: %s", available_years)
            if available_years:
                return sorted(list(set(available_years)))
            else:
                # Fallback to current year if no cache data found
                fallback_year = datetime.now().year
                _logger.debug("No years found, using fallback: %s (%s)", fallback_year, type(fallback_year))
                return [fallback_year]

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

                        # Look for year-based files in the interval directory
                        for file_path in interval_dir.iterdir():
                            if not file_path.is_file() or file_path.suffix not in ['.csv', '.parquet']:
                                continue

                            # Extract year from filename
                            try:
                                year = int(file_path.stem)
                            except ValueError:
                                continue

                            data_path = file_path

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
                                    _logger.info("Found multi-year data in %s/%s/%s/%s: %s", provider, symbol, interval, year, years_in_data)

                                    # Split and migrate the data
                                    success = self._migrate_multi_year_file(
                                        df, provider, symbol, interval, file_path, years_in_data
                                    )

                                    if success:
                                        migration_results['files_migrated'] += 1
                                        migration_results['years_created'] += len(years_in_data)
                                        migration_results['details'].append(f"{provider}/{symbol}/{interval}/{year} -> {years_in_data}")
                                    else:
                                        migration_results['errors'] += 1

                            except Exception as e:
                                _logger.warning("Error processing %s/%s/%s/%s: %s", provider, symbol, interval, year, e)
                                migration_results['errors'] += 1
                                continue

            _logger.info("Cache migration completed: %d files migrated, %d years created", migration_results['files_migrated'], migration_results['years_created'])
            return migration_results

        except Exception:
            _logger.exception("Error during cache migration:")
            migration_results['errors'] += 1
            return migration_results

    def _migrate_multi_year_file(self, df: pd.DataFrame, provider: str, symbol: str,
                                interval: str, original_file_path: Path, years_in_data: List[int]) -> bool:
        """
        Migrate a single multi-year file to year-split structure.

        Args:
            df: DataFrame containing multi-year data
            provider: Data provider name
            symbol: Trading symbol
            interval: Time interval
            original_file_path: Original file containing the multi-year data
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

            # Get the base cache path for this symbol/interval
            base_cache_path = self._get_cache_path(provider, symbol, interval, 0)  # year doesn't matter for base path

            # Save each year's data to its own file
            for year, year_df in years_data.items():
                # Create the year-specific file path
                data_path = self._get_data_path(base_cache_path, 'csv', year)

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
                self._save_metadata(base_cache_path, cache_metadata)

                _logger.debug("Created year file %s.csv with %d rows", year, len(year_df))

            # Remove the original multi-year file
            if original_file_path.exists():
                original_file_path.unlink()
                _logger.debug("Removed original multi-year file: %s", original_file_path)

            return True

        except Exception:
            _logger.exception("Error migrating multi-year file:")
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

                    # Try to find the year file in any available format
                    data_path = None
                    if format == "parquet":
                        # Try parquet first, then CSV
                        parquet_path = self._get_data_path(cache_path, "parquet", year)
                        csv_path = self._get_data_path(cache_path, "csv", year)
                        if parquet_path.exists():
                            data_path = parquet_path
                        elif csv_path.exists():
                            data_path = csv_path
                    else:
                        # Try CSV first (including compressed), then parquet
                        csv_path = self._get_data_path(cache_path, "csv", year)
                        parquet_path = self._get_data_path(cache_path, "parquet", year)

                        # Check for compressed CSV files
                        if self.compression_enabled:
                            compressed_csv_path = cache_path / f"{year}.csv.gz"
                            if compressed_csv_path.exists():
                                data_path = compressed_csv_path
                            elif csv_path.exists():
                                data_path = csv_path
                        else:
                            if csv_path.exists():
                                data_path = csv_path

                        # Fallback to parquet if CSV not found
                        if not data_path and parquet_path.exists():
                            data_path = parquet_path

                    if not data_path or not data_path.exists():
                        continue

                    # Load and check metadata
                    metadata = self._load_metadata(cache_path)
                    if not metadata:
                        continue

                    # Check invalidation
                    if self._should_invalidate(data_path, metadata):
                        _logger.info("Cache invalidated for %s/%s/%s/%s", provider, symbol, interval, year)
                        self.delete(provider, symbol, interval, year)
                        continue

                    # Load data for this year
                    try:
                        if data_path.suffix == '.parquet':
                            year_df = pd.read_parquet(data_path)
                        else:
                            # Load CSV with compression support
                            if self.compressor and data_path.suffix in ['.gz', '.zst']:
                                year_df = self.compressor.load_compressed_csv(data_path)
                            else:
                                year_df = pd.read_csv(data_path, parse_dates=['timestamp'])

                            if year_df is not None:
                                # Ensure timestamp column is properly parsed as datetime
                                if 'timestamp' in year_df.columns:
                                    year_df['timestamp'] = pd.to_datetime(year_df['timestamp'])
                                    # Set timestamp as index for consistency
                                    year_df.set_index('timestamp', inplace=True)
                                else:
                                    _logger.warning("No timestamp column found in %s", data_path)
                                    continue
                            else:
                                _logger.warning("Failed to load CSV from %s", data_path)
                                continue

                        if not year_df.empty:
                            all_data.append(year_df)
                            found_data = True
                            _logger.debug("Loaded data for year %s: %d rows", year, len(year_df))
                        else:
                            _logger.warning("Empty DataFrame loaded for year %s", year)
                    except Exception as e:
                        _logger.warning("Error loading data for year %s: %s", year, e)
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
                    # Ensure index is datetime for comparison
                    if not isinstance(df.index, pd.DatetimeIndex):
                        _logger.warning("Index is not DatetimeIndex, type: %s", type(df.index))
                        if df.index.dtype == 'object':
                            # Try to convert string timestamps to datetime
                            try:
                                df.index = pd.to_datetime(df.index)
                                _logger.info("Successfully converted string index to DatetimeIndex")
                            except Exception:
                                _logger.exception("Failed to convert index to datetime:")
                                # Skip date filtering if we can't convert
                                pass

                    if start_date and isinstance(df.index, pd.DatetimeIndex):
                        df = df[df.index >= start_date]
                    if end_date and isinstance(df.index, pd.DatetimeIndex):
                        df = df[df.index <= end_date]

                self.metrics.hits += 1
                response_time = (time.time() - start_time) * 1000
                self.metrics.avg_response_time_ms = (
                    (self.metrics.avg_response_time_ms * (self.metrics.hits - 1) + response_time) /
                    self.metrics.hits
                )

                _logger.debug("Cache hit for %s/%s/%s across %d years: %d total rows", provider, symbol, interval, len(all_data), len(df))
                return df

        except Exception:
            self.metrics.errors += 1
            _logger.exception("Error reading from cache:")
            return None

    def put(self, df: pd.DataFrame, provider: str, symbol: str, interval: str,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            format: str = "csv", metadata: Optional[Dict[str, Any]] = None,
            append_mode: bool = True) -> bool:
        """
        Store data in cache with enhanced CSV conventions.

        Args:
            df: DataFrame to cache
            provider: Data provider name
            symbol: Trading symbol
            interval: Time interval
            start_date: Start date for data
            end_date: End date for data
            format: File format (csv, parquet)
            metadata: Additional metadata
            append_mode: Whether to append to existing data or overwrite

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.lock:
                if df.empty:
                    _logger.warning("Empty DataFrame provided for %s/%s/%s", provider, symbol, interval)
                    return False

                # Standardize DataFrame format for CSV (validation removed)
                if format == "csv":
                    try:
                        df_standardized = CSVFormatConventions.standardize_dataframe(df, provider)
                    except ValueError:
                        _logger.exception("DataFrame standardization failed:")
                        return False
                else:
                    df_standardized = df

                # Split data by years and cache each year separately
                years_data = self._split_data_by_years(df_standardized, start_date, end_date)

                total_rows_cached = 0
                files_created = 0

                for year, year_df in years_data.items():
                    if year_df.empty:
                        continue

                    cache_path = self._get_cache_path(provider, symbol, interval, year)
                    data_path = self._get_data_path(cache_path, format, year)

                    # Create enhanced metadata
                    cache_metadata = CacheMetadata(
                        provider=provider,
                        symbol=symbol,
                        interval=interval,
                        year=year,
                        format=format,
                        compression_enabled=self.compression_enabled
                    )

                    if metadata:
                        for key, value in metadata.items():
                            if hasattr(cache_metadata, key):
                                setattr(cache_metadata, key, value)

                    # Handle data storage
                    if format == "csv":
                        if append_mode and data_path.exists():
                            # Append mode - use smart appending
                            success, rows_added = SmartDataAppender.append_new_data(
                                data_path, year_df, cache_metadata
                            )
                            if success:
                                total_rows_cached += rows_added
                                files_created += 1
                            else:
                                _logger.error("Failed to append data for %s/%s/%s/%s", provider, symbol, interval, year)
                                continue
                        else:
                            # Overwrite mode or new file
                            if self.compression_enabled:
                                # Use compression
                                success = self.compressor.save_compressed_csv(year_df, data_path, 'gzip')
                                if success:
                                    # Update data_path to point to compressed file
                                    data_path = data_path.with_suffix('.csv.gz')
                            else:
                                year_df.to_csv(data_path, index=False)

                            cache_metadata.update_integrity_info(year_df, data_path)
                            total_rows_cached += len(year_df)
                            files_created += 1
                    else:
                        # Parquet format
                        year_df.to_parquet(data_path, compression='snappy' if self.compression_enabled else None)
                        cache_metadata.update_integrity_info(year_df, data_path)
                        total_rows_cached += len(year_df)
                        files_created += 1

                    # Save metadata (aggregate with existing metadata if any)
                    existing_metadata = self._load_metadata(cache_path) or {}

                    # Create aggregated metadata
                    if 'years' not in existing_metadata:
                        existing_metadata['years'] = {}

                    # Store year-specific metadata (ensure datetime objects are converted to strings)
                    metadata_dict = asdict(cache_metadata)
                    # Convert any remaining datetime objects to ISO format strings
                    for key, value in metadata_dict.items():
                        if isinstance(value, datetime):
                            metadata_dict[key] = value.isoformat()
                    existing_metadata['years'][str(year)] = metadata_dict

                    # Update overall metadata
                    existing_metadata['provider'] = provider
                    existing_metadata['symbol'] = symbol
                    existing_metadata['interval'] = interval
                    existing_metadata['last_updated'] = datetime.now().isoformat()

                    self._save_metadata(cache_path, existing_metadata)

                    # Update metrics
                    file_size = data_path.stat().st_size
                    self.metrics.total_size_bytes += file_size

                    _logger.debug("Cached data for %s/%s/%s/%s: %d rows", provider, symbol, interval, year, len(year_df))

                # Update overall metrics
                self.metrics.sets += 1
                self.metrics.files_created += files_created

                _logger.info("Successfully cached %d total rows across %d year files for %s/%s/%s", total_rows_cached, files_created, provider, symbol, interval)
                return True

        except Exception:
            self.metrics.errors += 1
            _logger.exception("Error writing to cache:")
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
                                # Try to find and delete the year file (check CSV, compressed CSV, and parquet)
                csv_path = self._get_data_path(cache_path, format="csv", year=year)
                parquet_path = self._get_data_path(cache_path, format="parquet", year=year)

                data_path = None
                if csv_path.exists():
                    data_path = csv_path
                elif parquet_path.exists():
                    data_path = parquet_path

                # Also check for compressed CSV files
                if not data_path and self.compression_enabled:
                    compressed_csv_path = cache_path / f"{year}.csv.gz"
                    if compressed_csv_path.exists():
                        data_path = compressed_csv_path

                if data_path and data_path.exists():
                    # Remove the specific year file
                    file_size = data_path.stat().st_size
                    data_path.unlink()
                    self.metrics.total_size_bytes -= file_size
                    self.metrics.deletes += 1
                    self.metrics.files_deleted += 1

                    # Also remove metadata if it exists
                    metadata_path = self._get_metadata_path(cache_path)
                    if metadata_path.exists():
                        metadata_path.unlink()

                    _logger.debug("Deleted cache for %s/%s/%s/%s", provider, symbol, interval, year)
                    return True

                return False

        except Exception:
            self.metrics.errors += 1
            _logger.exception("Error deleting cache:")
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

        except Exception:
            _logger.exception("Error clearing cache:")
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

                _logger.info("Cleaned up %d old cache files", deleted_count)
                return deleted_count

        except Exception:
            _logger.exception("Error during cleanup:")
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
                for file_path in symbol_path.iterdir():
                    if file_path.is_file() and file_path.suffix in ['.csv', '.parquet', '.gz', '.zst']:
                        # Extract year from filename (e.g., "2024.csv" -> 2024, "2024.csv.gz" -> 2024)
                        try:
                            # Handle compressed files by removing compression extension first
                            stem = file_path.stem
                            if stem.endswith('.csv'):
                                stem = stem[:-4]  # Remove .csv from stem
                            year = int(stem)
                            info['years_available'].append(year)

                            # Get file size
                            info['total_size_bytes'] += file_path.stat().st_size

                            # Try to load metadata for this year
                            metadata = self._load_metadata(symbol_path)
                            if metadata and 'years' in metadata:
                                year_metadata = metadata['years'].get(str(year))
                                if year_metadata:
                                    info['total_rows'] += year_metadata.get('row_count', 0)

                                # Update last updated
                                created_at = metadata.get('created_at')
                                if created_at:
                                    try:
                                        created_dt = datetime.fromisoformat(created_at)
                                        if info['last_updated'] is None or created_dt > info['last_updated']:
                                            info['last_updated'] = created_dt
                                    except:
                                        pass
                        except ValueError:
                            # Skip files that don't have year as filename
                            continue

                info['years_available'].sort()

        except Exception:
            _logger.exception("Error getting cache info:")

        return info


# Global cache instance
_file_cache_instance: Optional[FileBasedCache] = None


def get_file_cache(
    cache_dir: Union[str, Path] = DATA_CACHE_DIR,
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
    cache_dir: Union[str, Path] = DATA_CACHE_DIR,
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
                _logger.info("Auto-migration completed: %d files migrated", migration_results['files_migrated'])
        except Exception as e:
            _logger.warning("Auto-migration failed: %s", e)

    return _file_cache_instance
