#!/usr/bin/env python3
"""
FINRA Data Collector Script for Short Squeeze Detection Pipeline

This script runs bi-weekly to download official FINRA short interest data
and store it in the ss_finra_short_interest table for use in squeeze detection.
Enhanced with float shares and volume data from yfinance to calculate:
- float_shares (from yfinance)
- short_interest_pct (calculated)
- days_to_cover (calculated using 30-day avg volume)

Usage:
    python run_finra_collector.py [options]

Examples:
    # Run with default configuration
    python run_finra_collector.py

    # Run with custom configuration file
    python run_finra_collector.py --config /path/to/config.yaml

    # Run for specific date
    python run_finra_collector.py --date 2024-01-15

    # Run in dry-run mode (no database writes)
    python run_finra_collector.py --dry-run

    # Skip enrichment (faster, but no float/volume data)
    python run_finra_collector.py --skip-enrichment

    # Run with verbose logging
    python run_finra_collector.py --verbose

    # Test FINRA connection only
    python run_finra_collector.py --test-connection
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.finra_data_downloader import create_finra_downloader
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.data.db.core.database import session_scope
from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.ml.pipeline.p04_short_squeeze.config.config_manager import ConfigManager

_logger = setup_logger(__name__)


class FINRADataEnricher:
    """
    Enriches FINRA data with additional metrics from yfinance.

    Calculates:
    - float_shares: From yfinance ticker info
    - short_interest_pct: 100 * short_interest_shares / float_shares
    - days_to_cover: short_interest_shares / avg_daily_volume_30d
    """

    def __init__(self, yf_downloader: YahooDataDownloader):
        """
        Initialize the FINRA data enricher.

        Args:
            yf_downloader: Yahoo Finance downloader instance
        """
        self.yf_downloader = yf_downloader
        self._float_cache: Dict[str, Optional[int]] = {}
        self._volume_cache: Dict[str, Optional[float]] = {}

    def enrich_finra_records(self, finra_records: List[Dict[str, Any]],
                            settlement_date: date,
                            batch_size: int = 50) -> List[Dict[str, Any]]:
        """
        Enrich FINRA records with float shares and volume data.

        Args:
            finra_records: List of FINRA data records
            settlement_date: Settlement date for volume calculation
            batch_size: Number of tickers to process in each batch

        Returns:
            Enriched FINRA records with float_shares, short_interest_pct, days_to_cover
        """
        if not finra_records:
            return finra_records

        _logger.info("Enriching %d FINRA records with float and volume data", len(finra_records))

        # Extract unique tickers
        tickers = list(set(record['ticker'] for record in finra_records))
        _logger.info("Processing %d unique tickers", len(tickers))

        # Enrich float shares in batches
        self._enrich_float_shares_batch(tickers, batch_size)

        # Enrich volume data in batches
        self._enrich_volume_data_batch(tickers, settlement_date, batch_size)

        # Apply enrichments to records
        enriched_records = []
        skipped_count = 0

        for record in finra_records:
            ticker = record['ticker']
            short_interest_shares = record.get('short_interest_shares', 0)

            # Get float shares from cache
            float_shares = self._float_cache.get(ticker)

            # Get average volume from cache
            avg_volume_30d = self._volume_cache.get(ticker)

            # Calculate short_interest_pct
            short_interest_pct = None
            if float_shares and float_shares > 0 and short_interest_shares > 0:
                short_interest_pct = 100.0 * short_interest_shares / float_shares

            # Calculate days_to_cover
            days_to_cover = None
            if avg_volume_30d and avg_volume_30d > 0 and short_interest_shares > 0:
                days_to_cover = short_interest_shares / avg_volume_30d

            # Add enriched fields to record
            enriched_record = record.copy()
            enriched_record['float_shares'] = float_shares
            enriched_record['short_interest_pct'] = short_interest_pct
            enriched_record['days_to_cover'] = days_to_cover

            # Track records with missing data
            if float_shares is None or avg_volume_30d is None:
                skipped_count += 1
                _logger.debug("Incomplete data for %s: float=%s, volume=%s",
                            ticker, float_shares, avg_volume_30d)

            enriched_records.append(enriched_record)

        _logger.info("Enrichment complete: %d records enriched, %d with incomplete data",
                    len(enriched_records), skipped_count)

        return enriched_records

    def _enrich_float_shares_batch(self, tickers: List[str], batch_size: int) -> None:
        """
        Enrich float shares for tickers in batches.

        Args:
            tickers: List of tickers to enrich
            batch_size: Number of tickers per batch
        """
        _logger.info("Fetching float shares for %d tickers (batch size: %d)",
                    len(tickers), batch_size)

        # Process in batches to avoid overwhelming yfinance
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            _logger.debug("Processing float shares batch %d/%d (%d tickers)",
                         i // batch_size + 1,
                         (len(tickers) + batch_size - 1) // batch_size,
                         len(batch))

            try:
                # Use batch fundamentals to get float shares efficiently
                fundamentals_batch = self.yf_downloader.get_fundamentals_batch(batch)

                for ticker in batch:
                    if ticker in fundamentals_batch:
                        fundamentals = fundamentals_batch[ticker]
                        float_shares = fundamentals.float_shares
                        self._float_cache[ticker] = float_shares

                        if float_shares:
                            _logger.debug("Float shares for %s: %s", ticker,
                                        f"{float_shares:,}")
                        else:
                            _logger.debug("No float shares data for %s", ticker)
                    else:
                        _logger.warning("No fundamentals data for %s", ticker)
                        self._float_cache[ticker] = None

                # Small delay between batches to respect rate limits
                if i + batch_size < len(tickers):
                    time.sleep(0.5)

            except Exception as e:
                _logger.exception("Error fetching float shares for batch:")
                # Mark all tickers in failed batch as None
                for ticker in batch:
                    if ticker not in self._float_cache:
                        self._float_cache[ticker] = None

        success_count = sum(1 for v in self._float_cache.values() if v is not None)
        _logger.info("Float shares enrichment: %d/%d tickers successful",
                    success_count, len(tickers))

    def _enrich_volume_data_batch(self, tickers: List[str], settlement_date: date,
                                  batch_size: int) -> None:
        """
        Enrich volume data for tickers in batches.

        Args:
            tickers: List of tickers to enrich
            settlement_date: Settlement date (end date for volume calculation)
            batch_size: Number of tickers per batch
        """
        _logger.info("Fetching 30-day average volume for %d tickers (batch size: %d)",
                    len(tickers), batch_size)

        # Calculate date range: 30 trading days before settlement_date
        # Use 45 calendar days to ensure we get 30 trading days
        end_date = datetime.combine(settlement_date, datetime.min.time())
        start_date = end_date - timedelta(days=45)

        _logger.debug("Volume date range: %s to %s", start_date.date(), end_date.date())

        # Process in batches
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            _logger.debug("Processing volume batch %d/%d (%d tickers)",
                         i // batch_size + 1,
                         (len(tickers) + batch_size - 1) // batch_size,
                         len(batch))

            try:
                # Use batch OHLCV download for efficiency
                ohlcv_batch = self.yf_downloader.get_ohlcv_batch(
                    batch,
                    interval='1d',
                    start_date=start_date,
                    end_date=end_date
                )

                for ticker in batch:
                    if ticker in ohlcv_batch:
                        df = ohlcv_batch[ticker]

                        if not df.empty and 'volume' in df.columns:
                            # Calculate average volume over the last 30 trading days
                            volumes = df['volume'].tail(30)

                            if len(volumes) > 0:
                                avg_volume = float(volumes.mean())
                                self._volume_cache[ticker] = avg_volume
                                _logger.debug("30-day avg volume for %s: %s (from %d days)",
                                            ticker, f"{avg_volume:,.0f}", len(volumes))
                            else:
                                _logger.debug("No volume data for %s", ticker)
                                self._volume_cache[ticker] = None
                        else:
                            _logger.debug("Empty OHLCV data for %s", ticker)
                            self._volume_cache[ticker] = None
                    else:
                        _logger.warning("No OHLCV data for %s", ticker)
                        self._volume_cache[ticker] = None

                # Small delay between batches
                if i + batch_size < len(tickers):
                    time.sleep(0.5)

            except Exception as e:
                _logger.exception("Error fetching volume data for batch:")
                # Mark all tickers in failed batch as None
                for ticker in batch:
                    if ticker not in self._volume_cache:
                        self._volume_cache[ticker] = None

        success_count = sum(1 for v in self._volume_cache.values() if v is not None)
        _logger.info("Volume enrichment: %d/%d tickers successful",
                    success_count, len(tickers))

    def get_enrichment_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the enrichment process.

        Returns:
            Dictionary with enrichment statistics
        """
        total_tickers = len(self._float_cache)
        float_success = sum(1 for v in self._float_cache.values() if v is not None)
        volume_success = sum(1 for v in self._volume_cache.values() if v is not None)

        both_success = sum(
            1 for ticker in self._float_cache.keys()
            if self._float_cache.get(ticker) is not None
            and self._volume_cache.get(ticker) is not None
        )

        return {
            'total_tickers': total_tickers,
            'float_success_count': float_success,
            'float_success_rate': float_success / total_tickers if total_tickers > 0 else 0,
            'volume_success_count': volume_success,
            'volume_success_rate': volume_success / total_tickers if total_tickers > 0 else 0,
            'both_success_count': both_success,
            'both_success_rate': both_success / total_tickers if total_tickers > 0 else 0
        }


class FINRACollectorRunner:
    """
    Runner class for the FINRA data collector script.

    Handles command-line arguments, configuration loading, and orchestrates
    the FINRA data collection process with comprehensive error handling and metrics.
    """

    def __init__(self):
        """Initialize the FINRA collector runner."""
        self.config_manager: Optional[ConfigManager] = None
        self.finra_downloader = None
        self.yf_downloader: Optional[YahooDataDownloader] = None
        self.enricher: Optional[FINRADataEnricher] = None
        self.start_time: Optional[datetime] = None
        self.run_id: Optional[str] = None

    def parse_arguments(self) -> argparse.Namespace:
        """
        Parse command-line arguments.

        Returns:
            Parsed arguments namespace
        """
        parser = argparse.ArgumentParser(
            description="Run FINRA data collector for short squeeze detection",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ""
        )

        parser.add_argument(
            '--config', '-c',
            type=str,
            help='Path to configuration file (default: uses default config location)'
        )

        parser.add_argument(
            '--date',
            type=str,
            help='Specific date to collect data for (YYYY-MM-DD format, default: auto-detect latest)'
        )

        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Run without writing results to database'
        )

        parser.add_argument(
            '--skip-enrichment',
            action='store_true',
            help='Skip float shares and volume enrichment (faster but incomplete data)'
        )

        parser.add_argument(
            '--enrichment-batch-size',
            type=int,
            default=50,
            help='Batch size for enrichment operations (default: 50)'
        )

        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )

        parser.add_argument(
            '--test-connection',
            action='store_true',
            help='Test FINRA connection and exit'
        )

        parser.add_argument(
            '--run-id',
            type=str,
            help='Custom run ID (default: auto-generated timestamp)'
        )

        parser.add_argument(
            '--output-dir',
            type=str,
            help='Directory to save output reports (optional)'
        )

        parser.add_argument(
            '--force-download',
            action='store_true',
            help='Force download even if data already exists for the date'
        )

        parser.add_argument(
            '--max-retries',
            type=int,
            default=3,
            help='Maximum number of retry attempts for failed downloads'
        )

        return parser.parse_args()

    def setup_logging(self, verbose: bool) -> None:
        """
        Setup logging configuration.

        Args:
            verbose: Enable verbose logging if True
        """
        if verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
            _logger.info("Verbose logging enabled")

    def load_configuration(self, config_path: Optional[str]) -> bool:
        """
        Load and validate configuration.

        Args:
            config_path: Optional path to configuration file

        Returns:
            True if configuration loaded successfully, False otherwise
        """
        try:
            _logger.info("Loading configuration...")
            self.config_manager = ConfigManager(config_path)
            config = self.config_manager.load_config()

            _logger.info("Configuration loaded successfully")
            _logger.info("Run ID: %s", config.run_id)

            # Get FINRA-specific configuration if available
            finra_config = getattr(config, 'finra_collector', None)
            if finra_config:
                _logger.info("FINRA config: max_file_age=%d days, retry_attempts=%d",
                           finra_config.max_file_age_days, finra_config.retry_attempts)

            return True

        except Exception as e:
            _logger.exception("Failed to load configuration:")
            return False

    def initialize_downloaders(self, skip_enrichment: bool) -> bool:
        """
        Initialize FINRA and Yahoo Finance downloaders.

        Args:
            skip_enrichment: If True, skip Yahoo Finance downloader initialization

        Returns:
            True if downloaders initialized successfully, False otherwise
        """
        try:
            _logger.info("Initializing FINRA downloader...")

            # Initialize FINRA downloader
            self.finra_downloader = create_finra_downloader()

            # Test connection if available
            if hasattr(self.finra_downloader, 'test_connection'):
                if not self.finra_downloader.test_connection():
                    _logger.error("FINRA connection test failed")
                    return False
                _logger.info("FINRA connection successful")
            else:
                _logger.info("FINRA downloader initialized (no connection test available)")

            # Initialize Yahoo Finance downloader for enrichment
            if not skip_enrichment:
                _logger.info("Initializing Yahoo Finance downloader for enrichment...")
                self.yf_downloader = YahooDataDownloader()
                self.enricher = FINRADataEnricher(self.yf_downloader)
                _logger.info("Yahoo Finance downloader initialized")
            else:
                _logger.info("Skipping Yahoo Finance downloader (enrichment disabled)")

            return True

        except Exception as e:
            _logger.exception("Failed to initialize downloaders:")
            return False

    def determine_collection_dates(self, date_str: Optional[str]) -> List[date]:
        """
        Determine the dates for FINRA data collection.

        Args:
            date_str: Optional date string in YYYY-MM-DD format for single date

        Returns:
            List of dates to collect (single date if specified, all missing dates if auto-detect)
        """
        if date_str:
            try:
                single_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                return [single_date]
            except ValueError:
                _logger.error("Invalid date format '%s', expected YYYY-MM-DD", date_str)
                return []

        # Auto-detect missing FINRA dates
        try:
            with session_scope() as session:
                service = ShortSqueezeService(session)
                missing_dates = service.get_missing_finra_dates()

                if missing_dates:
                    _logger.info("Found %d missing FINRA dates to download:", len(missing_dates))
                    for i, missing_date in enumerate(sorted(missing_dates), 1):
                        _logger.info("  %d. %s", i, missing_date.strftime('%Y-%m-%d'))
                    return sorted(missing_dates)
                else:
                    _logger.info("No missing FINRA dates found - all available data is up to date")
                    return []

        except Exception as e:
            _logger.warning("Error checking missing dates, falling back to auto-detection: %s", e)

        # Fallback to auto-detect latest FINRA reporting date
        # FINRA reports are published twice monthly (around 15th and end of month)
        today = date.today()

        # Check if we're past the 15th - if so, use 15th, otherwise use previous month end
        if today.day >= 15:
            collection_date = date(today.year, today.month, 15)
        else:
            # Go to previous month end
            first_of_month = date(today.year, today.month, 1)
            last_month = first_of_month - timedelta(days=1)
            collection_date = date(last_month.year, last_month.month,
                                 self._get_last_day_of_month(last_month.year, last_month.month))

        _logger.info("Auto-detected FINRA collection date: %s", collection_date)
        return [collection_date]

    def _get_last_day_of_month(self, year: int, month: int) -> int:
        """Get the last day of a given month."""
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        last_day = next_month - timedelta(days=1)
        return last_day.day

    def filter_existing_dates(self, collection_dates: List[date], force_download: bool) -> List[date]:
        """
        Filter out dates that already have data in the database.

        Args:
            collection_dates: List of dates to check
            force_download: If True, skip the check and return all dates

        Returns:
            List of dates that need to be downloaded
        """
        if force_download:
            _logger.info("Force download enabled - will download all %d dates", len(collection_dates))
            return collection_dates

        try:
            with session_scope() as session:
                service = ShortSqueezeService(session)

                dates_to_download = []
                existing_dates = []

                for collection_date in collection_dates:
                    existing_count = service.get_finra_data_count_for_date(collection_date)

                    if existing_count > 0:
                        existing_dates.append(collection_date)
                        _logger.debug("FINRA data already exists for %s (%d records)",
                                    collection_date, existing_count)
                    else:
                        dates_to_download.append(collection_date)

                if existing_dates:
                    _logger.info("Skipping %d dates that already have data: %s",
                               len(existing_dates),
                               [d.strftime('%Y-%m-%d') for d in existing_dates])

                if dates_to_download:
                    _logger.info("Will download %d missing dates: %s",
                               len(dates_to_download),
                               [d.strftime('%Y-%m-%d') for d in dates_to_download])
                else:
                    _logger.info("All requested dates already have data. Use --force-download to override")

                return dates_to_download

        except Exception as e:
            _logger.warning("Error checking existing data: %s", e)
            # If we can't check, proceed with all downloads
            return collection_dates

    def download_finra_data(self, collection_date: date, max_retries: int) -> Optional[List[Dict[str, Any]]]:
        """
        Download FINRA data for the specified date.

        Args:
            collection_date: Date to download data for
            max_retries: Maximum number of retry attempts

        Returns:
            List of FINRA data records or None if failed
        """
        try:
            _logger.info("Downloading FINRA data for %s", collection_date)

            for attempt in range(max_retries + 1):
                try:
                    # Download FINRA data
                    finra_data = self.finra_downloader.get_short_interest_data(collection_date)

                    if finra_data is None:
                        _logger.warning("No FINRA data returned for %s", collection_date)
                        return None

                    if isinstance(finra_data, list) and len(finra_data) == 0:
                        _logger.warning("Empty FINRA data returned for %s", collection_date)
                        return None

                    # Convert DataFrame to list of dictionaries if needed
                    if hasattr(finra_data, 'to_dict'):
                        # It's a DataFrame
                        records = []
                        for _, row in finra_data.iterrows():
                            # Handle Symbol field which might be NaN/float
                            symbol = row.get('Symbol', '')
                            if pd.isna(symbol) or not isinstance(symbol, str):
                                continue  # Skip invalid symbols

                            ticker = symbol.strip().upper()
                            if not ticker:
                                continue  # Skip empty tickers

                            # Convert raw data to JSON-serializable format
                            raw_data = row.to_dict()
                            for key, value in raw_data.items():
                                if pd.isna(value):
                                    raw_data[key] = None
                                elif isinstance(value, (pd.Timestamp, datetime, date)):
                                    raw_data[key] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
                                elif isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
                                    raw_data[key] = None if pd.isna(value) else value

                            record = {
                                'ticker': ticker,
                                'settlement_date': collection_date,
                                'short_interest_shares': int(row.get('ShortVolume', 0)),
                                'total_shares_outstanding': int(row.get('TotalVolume', 0)) if 'TotalVolume' in row else None,
                                'raw_data': raw_data
                            }

                            # Skip invalid records
                            if record['short_interest_shares'] < 0:
                                continue

                            records.append(record)

                        finra_data = records

                    _logger.info("Successfully downloaded %d FINRA records", len(finra_data))
                    return finra_data

                except Exception as e:
                    if attempt < max_retries:
                        wait_time = (2 ** attempt) * 5  # Exponential backoff: 5, 10, 20 seconds
                        _logger.warning("Download attempt %d failed: %s. Retrying in %d seconds...",
                                      attempt + 1, e, wait_time)
                        time.sleep(wait_time)
                    else:
                        _logger.exception("All download attempts failed:")
                        raise

            return None

        except Exception as e:
            _logger.exception("Failed to download FINRA data:")
            return None

    def enrich_finra_data(self, finra_data: List[Dict[str, Any]],
                         settlement_date: date,
                         batch_size: int) -> List[Dict[str, Any]]:
        """
        Enrich FINRA data with float shares and volume metrics.

        Args:
            finra_data: List of FINRA data records
            settlement_date: Settlement date for the data
            batch_size: Batch size for enrichment operations

        Returns:
            Enriched FINRA data records
        """
        if not self.enricher:
            _logger.warning("Enricher not initialized - skipping enrichment")
            return finra_data

        try:
            _logger.info("Starting enrichment process for %d records", len(finra_data))
            enriched_data = self.enricher.enrich_finra_records(
                finra_data,
                settlement_date,
                batch_size
            )

            # Log enrichment statistics
            stats = self.enricher.get_enrichment_stats()
            _logger.info("Enrichment statistics:")
            _logger.info("  Float shares: %d/%d tickers (%.1f%%)",
                        stats['float_success_count'],
                        stats['total_tickers'],
                        stats['float_success_rate'] * 100)
            _logger.info("  Volume data: %d/%d tickers (%.1f%%)",
                        stats['volume_success_count'],
                        stats['total_tickers'],
                        stats['volume_success_rate'] * 100)
            _logger.info("  Complete data: %d/%d tickers (%.1f%%)",
                        stats['both_success_count'],
                        stats['total_tickers'],
                        stats['both_success_rate'] * 100)

            return enriched_data

        except Exception as e:
            _logger.exception("Error during enrichment - returning original data:")
            return finra_data

    def store_finra_data(self, finra_data: List[Dict[str, Any]], dry_run: bool = False) -> int:
        """
        Store FINRA data in the database.

        Args:
            finra_data: List of FINRA data records
            dry_run: If True, don't actually write to database

        Returns:
            Number of records stored
        """
        try:
            if dry_run:
                _logger.info("DRY RUN MODE: Would store %d FINRA records", len(finra_data))
                # Log sample of what would be stored
                if finra_data:
                    sample = finra_data[0]
                    _logger.debug("Sample record: ticker=%s, si_shares=%s, float=%s, si_pct=%s, dtc=%s",
                                sample.get('ticker'),
                                sample.get('short_interest_shares'),
                                sample.get('float_shares'),
                                sample.get('short_interest_pct'),
                                sample.get('days_to_cover'))
                return len(finra_data)

            _logger.info("Storing %d FINRA records in database...", len(finra_data))

            with session_scope() as session:
                service = ShortSqueezeService(session)
                records_stored = service.store_finra_data(finra_data)

            _logger.info("Successfully stored %d FINRA records", records_stored)
            return records_stored

        except Exception as e:
            _logger.exception("Failed to store FINRA data:")
            return 0

    def generate_performance_report(self, results: Dict[str, Any]) -> None:
        """
        Generate and log performance metrics report.

        Args:
            results: Collection results dictionary
        """
        try:
            _logger.info("=== FINRA COLLECTOR PERFORMANCE REPORT ===")
            _logger.info("Run ID: %s", results.get('run_id'))

            # Multiple dates summary
            total_dates = results.get('total_dates_processed', 0)
            successful = results.get('successful_downloads', 0)
            failed = results.get('failed_downloads', 0)

            _logger.info("Dates Processed: %d total", total_dates)
            _logger.info("Success Rate: %d/%d (%.1f%%)", successful, total_dates,
                        results.get('data_quality_metrics', {}).get('success_rate', 0))

            # Runtime metrics
            duration = results.get('duration_seconds', 0)
            _logger.info("Runtime: %.2f seconds", duration)

            # Collection results
            _logger.info("Total Records Downloaded: %d", results.get('total_records_downloaded', 0))
            _logger.info("Total Records Stored: %d", results.get('total_records_stored', 0))
            _logger.info("Download Success: %s", "✅" if results.get('download_success') else "❌")
            _logger.info("Storage Success: %s", "✅" if results.get('storage_success') else "❌")

            # Enrichment statistics
            enrichment_stats = results.get('enrichment_stats')
            if enrichment_stats:
                _logger.info("Enrichment Statistics:")
                _logger.info("  Float Shares Success: %d/%d (%.1f%%)",
                            enrichment_stats.get('float_success_count', 0),
                            enrichment_stats.get('total_tickers', 0),
                            enrichment_stats.get('float_success_rate', 0) * 100)
                _logger.info("  Volume Data Success: %d/%d (%.1f%%)",
                            enrichment_stats.get('volume_success_count', 0),
                            enrichment_stats.get('total_tickers', 0),
                            enrichment_stats.get('volume_success_rate', 0) * 100)
                _logger.info("  Complete Enrichment: %d/%d (%.1f%%)",
                            enrichment_stats.get('both_success_count', 0),
                            enrichment_stats.get('total_tickers', 0),
                            enrichment_stats.get('both_success_rate', 0) * 100)

            # Per-date breakdown
            download_results = results.get('download_results', [])
            if download_results:
                _logger.info("Per-Date Results:")
                for result in download_results:
                    status = "✅" if result['success'] else "❌"
                    _logger.info("  %s %s: %d records downloaded, %d stored",
                               status, result['date'],
                               result['records_downloaded'], result['records_stored'])

            # Data quality metrics
            if results.get('data_quality_metrics'):
                quality = results['data_quality_metrics']
                avg_records = quality.get('avg_records_per_date', 0)
                if avg_records > 0:
                    _logger.info("Average Records per Date: %.0f", avg_records)

            _logger.info("=== END PERFORMANCE REPORT ===")

        except Exception as e:
            _logger.warning("Failed to generate performance report: %s", e)

    def save_output_report(self, results: Dict[str, Any], output_dir: Optional[str]) -> None:
        """
        Save output report to file if output directory is specified.

        Args:
            results: Collection results dictionary
            output_dir: Directory to save output files
        """
        if not output_dir:
            return

        try:
            import json
            from pathlib import Path

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save JSON report
            run_id = results.get('run_id', 'unknown')
            json_file = output_path / f"finra_collector_{run_id}.json"

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)

            _logger.info("Output report saved to: %s", json_file)

        except Exception as e:
            _logger.warning("Failed to save output report: %s", e)

    def run(self) -> int:
        """
        Main execution method.

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            self.start_time = datetime.now()
            args = self.parse_arguments()

            # Set run ID
            self.run_id = args.run_id or datetime.now().strftime("finra_%Y%m%d_%H%M%S")

            _logger.info("Starting FINRA Data Collector Script")
            _logger.info("Run ID: %s", self.run_id)
            _logger.info("Arguments: %s", vars(args))

            # Setup logging
            self.setup_logging(args.verbose)

            # Load configuration
            if not self.load_configuration(args.config):
                return 1

            # Initialize downloaders (FINRA + optionally Yahoo Finance)
            if not self.initialize_downloaders(args.skip_enrichment):
                return 1

            # Test connection mode
            if args.test_connection:
                _logger.info("FINRA connection test successful - exiting")
                return 0

            # Determine collection dates
            collection_dates = self.determine_collection_dates(args.date)
            if not collection_dates:
                _logger.info("No dates to collect")
                return 0

            _logger.info("Found %d dates to process", len(collection_dates))

            # Filter out existing dates
            dates_to_download = self.filter_existing_dates(collection_dates, args.force_download)
            if not dates_to_download:
                _logger.info("No new dates to download - all data is up to date")
                return 0

            # Download and store FINRA data for all missing dates
            total_records_downloaded = 0
            total_records_stored = 0
            successful_downloads = 0
            failed_downloads = 0
            download_results = []
            all_enrichment_stats = []

            for i, collection_date in enumerate(dates_to_download, 1):
                _logger.info("Processing date %d/%d: %s", i, len(dates_to_download), collection_date)

                try:
                    # Download FINRA data for this date
                    finra_data = self.download_finra_data(collection_date, args.max_retries)

                    if finra_data:
                        # Enrich with float shares and volume data (unless skipped)
                        if not args.skip_enrichment:
                            _logger.info("Enriching %d records with float and volume data...",
                                       len(finra_data))
                            finra_data = self.enrich_finra_data(
                                finra_data,
                                collection_date,
                                args.enrichment_batch_size
                            )

                            # Collect enrichment stats
                            if self.enricher:
                                all_enrichment_stats.append(self.enricher.get_enrichment_stats())
                        else:
                            _logger.info("Skipping enrichment as requested")

                        # Store FINRA data
                        records_stored = self.store_finra_data(finra_data, args.dry_run)

                        total_records_downloaded += len(finra_data)
                        total_records_stored += records_stored
                        successful_downloads += 1

                        download_results.append({
                            'date': collection_date,
                            'records_downloaded': len(finra_data),
                            'records_stored': records_stored,
                            'success': True
                        })

                        _logger.info("✅ %s: Downloaded %d records, stored %d",
                                   collection_date, len(finra_data), records_stored)
                    else:
                        failed_downloads += 1
                        download_results.append({
                            'date': collection_date,
                            'records_downloaded': 0,
                            'records_stored': 0,
                            'success': False
                        })
                        _logger.error("❌ %s: Failed to download data", collection_date)

                except Exception as e:
                    failed_downloads += 1
                    download_results.append({
                        'date': collection_date,
                        'records_downloaded': 0,
                        'records_stored': 0,
                        'success': False,
                        'error': str(e)
                    })
                    _logger.error("❌ %s: Error during processing: %s", collection_date, e)

            # Calculate total runtime
            end_time = datetime.now()
            total_runtime = (end_time - self.start_time).total_seconds()

            # Aggregate enrichment stats
            aggregated_enrichment = None
            if all_enrichment_stats:
                total_tickers = sum(s['total_tickers'] for s in all_enrichment_stats)
                float_success = sum(s['float_success_count'] for s in all_enrichment_stats)
                volume_success = sum(s['volume_success_count'] for s in all_enrichment_stats)
                both_success = sum(s['both_success_count'] for s in all_enrichment_stats)

                aggregated_enrichment = {
                    'total_tickers': total_tickers,
                    'float_success_count': float_success,
                    'float_success_rate': float_success / total_tickers if total_tickers > 0 else 0,
                    'volume_success_count': volume_success,
                    'volume_success_rate': volume_success / total_tickers if total_tickers > 0 else 0,
                    'both_success_count': both_success,
                    'both_success_rate': both_success / total_tickers if total_tickers > 0 else 0
                }

            # Prepare results
            results = {
                'run_id': self.run_id,
                'collection_dates': [d.isoformat() for d in dates_to_download],
                'total_dates_processed': len(dates_to_download),
                'successful_downloads': successful_downloads,
                'failed_downloads': failed_downloads,
                'total_records_downloaded': total_records_downloaded,
                'total_records_stored': total_records_stored,
                'download_success': successful_downloads > 0,
                'storage_success': total_records_stored > 0,
                'duration_seconds': total_runtime,
                'download_results': download_results,
                'enrichment_enabled': not args.skip_enrichment,
                'enrichment_stats': aggregated_enrichment,
                'data_quality_metrics': {
                    'success_rate': (successful_downloads / len(dates_to_download) * 100) if dates_to_download else 0,
                    'avg_records_per_date': (total_records_downloaded / successful_downloads) if successful_downloads > 0 else 0
                }
            }

            # Generate performance report
            self.generate_performance_report(results)

            # Save output report if requested
            self.save_output_report(results, args.output_dir)

            _logger.info("FINRA collector script completed successfully in %.2f seconds", total_runtime)
            _logger.info("Summary: %d/%d dates successful, %d total records stored",
                        successful_downloads, len(dates_to_download), total_records_stored)

            if aggregated_enrichment:
                _logger.info("Enrichment: %.1f%% float data, %.1f%% volume data, %.1f%% complete",
                           aggregated_enrichment['float_success_rate'] * 100,
                           aggregated_enrichment['volume_success_rate'] * 100,
                           aggregated_enrichment['both_success_rate'] * 100)

            return 0 if successful_downloads > 0 else 1

        except KeyboardInterrupt:
            _logger.warning("Script interrupted by user")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            _logger.error("Unexpected error in FINRA collector script: %s", e, exc_info=True)
            return 1


def main() -> int:
    """
    Main entry point for the FINRA collector script.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    runner = FINRACollectorRunner()
    return runner.run()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)