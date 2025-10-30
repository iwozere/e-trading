#!/usr/bin/env python3
"""
Unified Cache System

This module provides a simplified cache structure: symbol/timeframe/year/
- No provider-based folders
- Gzip compression for CSV files
- Intelligent provider selection
- Unified data access interface
"""

import json
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd

# Import cache directory setting
from config.donotshare.donotshare import DATA_CACHE_DIR

# Validation removed - data is cached as-is without validation
# Validation will be handled by validate_and_fill_gaps.py script


class UnifiedCache:
    """
    Unified cache system with simplified structure: ohlcv/symbol/timeframe/

    Structure:
    cache_dir/
    ├── ohlcv/
    │   ├── BTCUSDT/
    │   │   ├── 5m/
    │   │   │   ├── 2025.csv.gz
    │   │   │   ├── 2025.metadata.json
    │   │   │   ├── 2024.metadata.json
    │   │   │   └── 2024.csv.gz
    │   │   └── 1h/
    │   ├── AAPL/
    │   │   ├── 5m/
    │   │   └── 1d/
    │   └── _metadata/
    │       ├── symbols.json
    │       ├── providers.json
    │       └── quality_scores.json
    └── fundamentals/
        └── [fundamentals cache structure]
    """

    def __init__(self, cache_dir: str = DATA_CACHE_DIR, max_size_gb: float = 10.0):
        """
        Initialize unified cache.

        Args:
            cache_dir: Cache directory path
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir)
        self.ohlcv_dir = self.cache_dir / "ohlcv"
        self.max_size_gb = max_size_gb
        self.metadata_dir = self.ohlcv_dir / "_metadata"

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ohlcv_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        # Initialize metadata
        self._init_metadata()

    def _init_metadata(self):
        """Initialize cache metadata files."""
        metadata_files = {
            'symbols.json': {},
            'providers.json': {},
            'quality_scores.json': {},
            'cache_stats.json': {
                'total_size_gb': 0.0,
                'files_count': 0,
                'last_updated': datetime.now().isoformat()
            }
        }

        for filename, default_data in metadata_files.items():
            filepath = self.metadata_dir / filename
            if not filepath.exists():
                with open(filepath, 'w') as f:
                    json.dump(default_data, f, indent=2)

    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get cache path for symbol/timeframe."""
        return self.ohlcv_dir / symbol / timeframe

    def _get_data_file_path(self, symbol: str, timeframe: str, year: int) -> Path:
        """Get data file path (compressed CSV)."""
        cache_path = self._get_cache_path(symbol, timeframe)
        return cache_path / f"{year}.csv.gz"

    def _get_metadata_file_path(self, symbol: str, timeframe: str, year: int) -> Path:
        """Get metadata file path."""
        cache_path = self._get_cache_path(symbol, timeframe)
        return cache_path / f"{year}.metadata.json"

    def put(self, df: pd.DataFrame, symbol: str, timeframe: str,
            start_date: datetime, end_date: datetime,
            provider: str = "unknown", **kwargs) -> bool:
        """
        Store data in unified cache.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name
            timeframe: Time interval
            start_date: Start date
            end_date: End date
            provider: Data provider name
            **kwargs: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            if df is None or df.empty:
                return False

            # Skip validation - data is cached as-is without validation
            # Validation will be handled by validate_and_fill_gaps.py script
            is_valid = True
            quality_score = {'quality_score': 1.0}

            # Create cache directory
            cache_path = self._get_cache_path(symbol, timeframe)
            cache_path.mkdir(parents=True, exist_ok=True)

            # Split data by year and save each year separately
            years_in_data = df.index.year.unique()
            saved_files = []

            for year in years_in_data:
                # Filter data for this year
                year_data = df[df.index.year == year]

                if not year_data.empty:
                    # Save compressed CSV for this year
                    data_file = self._get_data_file_path(symbol, timeframe, year)
                    self._save_compressed_csv(year_data, data_file)
                    saved_files.append(year)

                    # Create metadata for this year
                    year_metadata = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'year': year,
                        'data_source': provider,
                        'created_at': datetime.now().isoformat(),
                        'last_updated': datetime.now().isoformat(),
                        'start_date': year_data.index.min().isoformat(),
                        'end_date': year_data.index.max().isoformat(),
                        'data_quality': {
                            'score': quality_score['quality_score'],
                            'validation_errors': [],  # No validation performed
                            'gaps': 0,  # Gap analysis handled by validate_and_fill_gaps.py
                            'duplicates': 0  # Duplicate analysis handled by validate_and_fill_gaps.py
                        },
                        'file_info': {
                            'format': 'csv.gz',
                            'size_bytes': data_file.stat().st_size if data_file.exists() else 0,
                            'rows': len(year_data),
                            'columns': list(year_data.columns)
                        },
                        'provider_info': {
                            'name': provider,
                            'reliability': 0.95,  # Default reliability score
                            'rate_limit': 'unknown'
                        }
                    }

                    # Save metadata for this year
                    metadata_file = self._get_metadata_file_path(symbol, timeframe, year)
                    with open(metadata_file, 'w') as f:
                        json.dump(year_metadata, f, indent=2)

                    # Update global metadata for this year
                    self._update_symbol_metadata(symbol, timeframe, year, year_metadata)

            print(f"✅ Cached {len(df)} rows for {symbol} {timeframe} across {len(saved_files)} years from {provider}")
            print(f"   Years saved: {sorted(saved_files)}")
            return True

        except Exception as e:
            print(f"❌ Error caching data for {symbol} {timeframe}: {str(e)}")
            return False

    def get(self, symbol: str, timeframe: str,
            start_date: datetime = None, end_date: datetime = None,
            format: str = 'csv') -> Optional[pd.DataFrame]:
        """
        Retrieve data from unified cache.

        Args:
            symbol: Symbol name
            timeframe: Time interval
            start_date: Start date (optional)
            end_date: End date (optional)
            format: Data format (only 'csv' supported for now)

        Returns:
            DataFrame with data or None if not found
        """
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()

            # Get all available years
            available_years = self._get_available_years(symbol, timeframe)
            if not available_years:
                return None

            # Filter years within date range
            relevant_years = [
                year for year in available_years
                if year >= start_date.year and year <= end_date.year
            ]

            if not relevant_years:
                return None

            # Load and combine data from relevant years
            combined_data = []
            for year in relevant_years:
                year_data = self._load_year_data(symbol, timeframe, year)
                if year_data is not None and not year_data.empty:
                    combined_data.append(year_data)

            if not combined_data:
                return None

            # Combine all years
            if len(combined_data) == 1:
                df = combined_data[0]
            else:
                df = pd.concat(combined_data, axis=0)
                df = df.sort_index()
                df = df[~df.index.duplicated(keep='first')]  # Remove duplicates

            # Filter by date range (ensure timezone compatibility)
            # Convert timezone-aware dates to naive for comparison with cached data
            start_date_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
            end_date_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
            df = df[(df.index >= start_date_naive) & (df.index <= end_date_naive)]

            if df.empty:
                return None

            return df

        except Exception as e:
            print(f"❌ Error retrieving data for {symbol} {timeframe}: {str(e)}")
            return None

    def _save_compressed_csv(self, df: pd.DataFrame, filepath: Path):
        """Save DataFrame as compressed CSV."""
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            df.to_csv(f, index=True, lineterminator='\n')

    def _load_year_data(self, symbol: str, timeframe: str, year: int) -> Optional[pd.DataFrame]:
        """Load data for a specific year."""
        data_file = self._get_data_file_path(symbol, timeframe, year)
        if not data_file.exists():
            return None

        try:
            with gzip.open(data_file, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            print(f"❌ Error loading data from {data_file}: {str(e)}")
            return None

    def _get_available_years(self, symbol: str, timeframe: str) -> List[int]:
        """Get list of available years for symbol/timeframe."""
        timeframe_dir = self.ohlcv_dir / symbol / timeframe
        if not timeframe_dir.exists():
            return []

        years = []
        for item in timeframe_dir.iterdir():
            if item.is_file() and item.name.endswith('.csv.gz'):
                # Extract year from filename like "2020.csv.gz"
                year_str = item.name.replace('.csv.gz', '')
                if year_str.isdigit():
                    years.append(int(year_str))

        return sorted(years)

    def _update_symbol_metadata(self, symbol: str, timeframe: str, year: int, metadata: Dict):
        """Update global symbol metadata."""
        symbols_file = self.metadata_dir / 'symbols.json'

        try:
            with open(symbols_file, 'r') as f:
                symbols_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            symbols_data = {}

        # Initialize symbol data if not exists
        if symbol not in symbols_data:
            symbols_data[symbol] = {}

        if timeframe not in symbols_data[symbol]:
            symbols_data[symbol][timeframe] = {}

        # Update year data
        symbols_data[symbol][timeframe][str(year)] = {
            'provider': metadata['data_source'],
            'last_updated': metadata['last_updated'],
            'quality_score': metadata['data_quality']['score'],
            'rows': metadata['file_info']['rows']
        }

        # Save updated metadata
        with open(symbols_file, 'w') as f:
            json.dump(symbols_data, f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            stats_file = self.metadata_dir / 'cache_stats.json'
            with open(stats_file, 'r') as f:
                stats = json.load(f)

            # Calculate current stats
            total_size = 0
            files_count = 0

            for item in self.ohlcv_dir.rglob('*.csv.gz'):
                if item.is_file():
                    total_size += item.stat().st_size
                    files_count += 1

            stats['total_size_gb'] = total_size / (1024**3)
            stats['files_count'] = files_count
            stats['last_updated'] = datetime.now().isoformat()

            # Save updated stats
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

            return stats

        except Exception as e:
            print(f"❌ Error getting cache stats: {str(e)}")
            return {}

    def list_symbols(self) -> List[str]:
        """List all symbols in cache."""
        symbols = []
        for item in self.ohlcv_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                symbols.append(item.name)
        return sorted(symbols)

    def list_timeframes(self, symbol: str) -> List[str]:
        """List all timeframes for a symbol."""
        symbol_dir = self.ohlcv_dir / symbol
        if not symbol_dir.exists():
            return []

        timeframes = []
        for item in symbol_dir.iterdir():
            if item.is_dir():
                timeframes.append(item.name)
        return sorted(timeframes)

    def list_years(self, symbol: str, timeframe: str) -> List[int]:
        """List all years for a symbol/timeframe."""
        return self._get_available_years(symbol, timeframe)

    def get_data_info(self, symbol: str, timeframe: str, year: int) -> Optional[Dict]:
        """Get detailed information about cached data."""
        metadata_file = self._get_metadata_file_path(symbol, timeframe, year)
        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def cleanup_old_data(self, max_age_days: int = 365) -> int:
        """Remove data older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed_files = 0

        for item in self.ohlcv_dir.rglob('*.csv.gz'):
            if item.is_file():
                try:
                    # Get file modification time
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    if mtime < cutoff_date:
                        item.unlink()
                        removed_files += 1
                except Exception:
                    continue

        print(f"🧹 Cleaned up {removed_files} old files")
        return removed_files


def configure_unified_cache(cache_dir: str = DATA_CACHE_DIR, max_size_gb: float = 10.0) -> UnifiedCache:
    """Configure and return unified cache instance."""
    return UnifiedCache(cache_dir=cache_dir, max_size_gb=max_size_gb)


def get_unified_cache() -> UnifiedCache:
    """Get default unified cache instance."""
    return configure_unified_cache()
