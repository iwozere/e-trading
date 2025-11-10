#!/usr/bin/env python3
"""
Data Downloader Script - Proper Architecture

This script downloads and caches OHLCV data using the unified DataManager architecture.
Data is automatically cached to c:/data-cache/ohlcv/ with proper compression and metadata.

Usage:
    python src/util/data_downloader.py

Features:
    - Automatic caching to unified cache directory
    - Gzip compression
    - Metadata tracking
    - Provider fallback
    - Rate limiting
    - Duplicate detection
"""

import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Change to project root directory for imports to work
os.chdir(PROJECT_ROOT)

from src.data.data_manager import DataManager
from src.notification.logger import setup_logger

# Import DATA_CACHE_DIR with fallback
from config.donotshare.donotshare import DATA_CACHE_DIR

_logger = setup_logger(__name__)

# Log startup info
_logger.info("Data Downloader Script")
_logger.info("Project root: %s", PROJECT_ROOT)
_logger.info("Current directory: %s", os.getcwd())
_logger.info("Cache directory: %s", DATA_CACHE_DIR)

# Define download scenarios
# NOTE: All dates are interpreted as UTC timezone to ensure consistency with Binance API
# If dates were timezone-naive, they would be interpreted as local timezone (e.g., CET)
# which would cause data to start at 2019-12-31 23:00:00 UTC instead of 2020-01-01 00:00:00 UTC
DOWNLOAD_SCENARIOS = {
    'symbols': ['LTCUSDT', 'BTCUSDT', 'ETHUSDT'],
    'periods': [
        {'start_date': '20200101', 'end_date': '20251111'}
    ],
    'intervals': ['5m', '15m', '30m', '1h', '4h']
}


def download_all_scenarios():
    """
    Download data for all combinations of symbols, periods, and intervals.

    This uses DataManager which automatically:
    - Caches to c:/data-cache/ohlcv/
    - Compresses with gzip
    - Creates metadata files
    - Validates data
    - Handles provider selection and fallback
    """
    # Initialize DataManager (uses unified cache architecture)
    data_manager = DataManager(cache_dir=DATA_CACHE_DIR)

    total_combinations = (
        len(DOWNLOAD_SCENARIOS['symbols']) *
        len(DOWNLOAD_SCENARIOS['periods']) *
        len(DOWNLOAD_SCENARIOS['intervals'])
    )
    completed = 0
    successful = 0
    failed = 0

    _logger.info("=" * 80)
    _logger.info("Starting data download using unified DataManager architecture")
    _logger.info("=" * 80)
    _logger.info("Cache directory: %s", DATA_CACHE_DIR)
    _logger.info("Total combinations: %d", total_combinations)
    _logger.info("=" * 80)

    for symbol in DOWNLOAD_SCENARIOS['symbols']:
        for period in DOWNLOAD_SCENARIOS['periods']:
            for interval in DOWNLOAD_SCENARIOS['intervals']:
                completed += 1

                # Convert date strings to datetime objects (UTC timezone-aware)
                try:
                    start_dt = datetime.strptime(period['start_date'], "%Y%m%d").replace(tzinfo=timezone.utc)
                    end_dt = datetime.strptime(period['end_date'], "%Y%m%d").replace(tzinfo=timezone.utc)
                except ValueError as e:
                    _logger.error("Invalid date format: %s", e)
                    failed += 1
                    continue

                _logger.info("")
                _logger.info("-" * 80)
                _logger.info("Processing %d/%d: %s %s (%s to %s)",
                           completed, total_combinations,
                           symbol, interval,
                           start_dt.date(), end_dt.date())
                _logger.info("-" * 80)

                try:
                    # Use DataManager to get/cache data
                    # This automatically:
                    # 1. Checks cache first
                    # 2. Downloads if not cached (with provider selection)
                    # 3. Validates data
                    # 4. Saves to cache with compression
                    # 5. Creates metadata
                    df = data_manager.get_ohlcv(
                        symbol=symbol,
                        timeframe=interval,
                        start_date=start_dt,
                        end_date=end_dt,
                        force_refresh=True  # Force download even if cached (to refresh corrupted data)
                    )

                    if df is not None and not df.empty:
                        _logger.info("✅ Successfully downloaded and cached %s %s: %d rows",
                                   symbol, interval, len(df))
                        successful += 1
                    else:
                        _logger.warning("⚠️ No data returned for %s %s", symbol, interval)
                        failed += 1

                except Exception as e:
                    _logger.error("❌ Error downloading %s %s: %s", symbol, interval, e)
                    failed += 1
                    continue

    # Print summary
    _logger.info("")
    _logger.info("=" * 80)
    _logger.info("DOWNLOAD SUMMARY")
    _logger.info("=" * 80)
    _logger.info("Total combinations: %d", total_combinations)
    _logger.info("Successful: %d", successful)
    _logger.info("Failed: %d", failed)
    _logger.info("Success rate: %.1f%%", (successful / total_combinations * 100) if total_combinations > 0 else 0)
    _logger.info("=" * 80)
    _logger.info("Data cached to: %s", DATA_CACHE_DIR)
    _logger.info("Cache structure: %s/ohlcv/{SYMBOL}/{TIMEFRAME}/{YEAR}.csv.gz", DATA_CACHE_DIR)
    _logger.info("=" * 80)


if __name__ == "__main__":
    try:
        download_all_scenarios()
    except KeyboardInterrupt:
        _logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception:
        _logger.exception("Fatal error:")
        sys.exit(1)
