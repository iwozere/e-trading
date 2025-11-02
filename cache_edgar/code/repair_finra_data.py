#!/usr/bin/env python3
"""
FINRA Data Enrichment Utility Script

This utility script enriches existing FINRA data in the ss_finra_short_interest table
by populating missing fields:
- float_shares (from yfinance)
- short_interest_pct (calculated: 100 * short_interest_shares / float_shares)
- days_to_cover (calculated: short_interest_shares / avg_daily_volume_30d)

This script is useful for backfilling data or updating existing records.

Usage:
    python enrich_finra_data.py [options]

Examples:
    # Enrich all records with missing data
    python enrich_finra_data.py

    # Enrich specific settlement date
    python enrich_finra_data.py --date 2024-01-15

    # Enrich date range
    python enrich_finra_data.py --start-date 2024-01-01 --end-date 2024-03-31

    # Enrich specific tickers
    python enrich_finra_data.py --tickers AAPL,TSLA,GME,AMC

    # Dry run (no database updates)
    python enrich_finra_data.py --dry-run

    # Force re-enrichment of already enriched records
    python enrich_finra_data.py --force

    # Use smaller batch size for slower connections
    python enrich_finra_data.py --batch-size 25
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.data.db.core.database import session_scope
from src.data.db.models.model_short_squeeze import FINRAShortInterest
from sqlalchemy import and_, or_

_logger = setup_logger(__name__)


class FINRADataEnrichmentUtility:
    """
    Utility for enriching existing FINRA data with float shares and volume metrics.
    """

    def __init__(self, yf_downloader: YahooDataDownloader, batch_size: int = 50):
        """
        Initialize the enrichment utility.

        Args:
            yf_downloader: Yahoo Finance downloader instance
            batch_size: Number of tickers to process in each batch
        """
        self.yf_downloader = yf_downloader
        self.batch_size = batch_size
        self._float_cache: Dict[str, Optional[int]] = {}
        self._volume_cache: Dict[Tuple[str, date], Optional[float]] = {}

    def get_records_to_enrich(self,
                             settlement_date: Optional[date] = None,
                             start_date: Optional[date] = None,
                             end_date: Optional[date] = None,
                             tickers: Optional[List[str]] = None,
                             force: bool = False) -> List[Dict[str, Any]]:
        """
        Get FINRA records that need enrichment.

        Args:
            settlement_date: Specific settlement date to enrich
            start_date: Start date for date range
            end_date: End date for date range
            tickers: Specific tickers to enrich
            force: If True, include already enriched records

        Returns:
            List of record dictionaries with id, ticker, settlement_date, short_interest_shares
        """
        try:
            with session_scope() as session:
                query = session.query(FINRAShortInterest)

                # Apply filters
                conditions = []

                if settlement_date:
                    conditions.append(FINRAShortInterest.settlement_date == settlement_date)
                elif start_date and end_date:
                    conditions.append(FINRAShortInterest.settlement_date >= start_date)
                    conditions.append(FINRAShortInterest.settlement_date <= end_date)
                elif start_date:
                    conditions.append(FINRAShortInterest.settlement_date >= start_date)
                elif end_date:
                    conditions.append(FINRAShortInterest.settlement_date <= end_date)

                if tickers:
                    conditions.append(FINRAShortInterest.ticker.in_(tickers))

                # If not forcing, only get records with missing enrichment data
                if not force:
                    missing_data = or_(
                        FINRAShortInterest.float_shares.is_(None),
                        FINRAShortInterest.short_interest_pct.is_(None),
                        FINRAShortInterest.days_to_cover.is_(None)
                    )
                    conditions.append(missing_data)

                if conditions:
                    query = query.filter(and_(*conditions))

                records = query.all()
                _logger.info("Found %d records to enrich", len(records))

                # Convert to dictionaries to avoid detached instance issues
                record_dicts = []
                for record in records:
                    record_dicts.append({
                        'id': record.id,
                        'ticker': record.ticker,
                        'settlement_date': record.settlement_date,
                        'short_interest_shares': record.short_interest_shares
                    })

                return record_dicts

        except Exception as e:
            _logger.exception("Error getting records to enrich:")
            return []

    def enrich_records(self, records: List[Dict[str, Any]], dry_run: bool = False) -> Dict[str, Any]:
        """
        Enrich FINRA records with float shares and volume data.

        Args:
            records: List of record dictionaries (id, ticker, settlement_date, short_interest_shares)
            dry_run: If True, don't update database

        Returns:
            Dictionary with enrichment statistics
        """
        if not records:
            _logger.info("No records to enrich")
            return {
                'total_records': 0,
                'records_updated': 0,
                'float_success': 0,
                'volume_success': 0,
                'complete_success': 0
            }

        _logger.info("Starting enrichment of %d records", len(records))

        # Group records by ticker and settlement date
        records_by_ticker = {}
        records_by_date = {}

        for record in records:
            ticker = record['ticker']
            settlement_date = record['settlement_date']

            if ticker not in records_by_ticker:
                records_by_ticker[ticker] = []
            records_by_ticker[ticker].append(record)

            if settlement_date not in records_by_date:
                records_by_date[settlement_date] = []
            records_by_date[settlement_date].append(record)

        _logger.info("Processing %d unique tickers across %d settlement dates",
                    len(records_by_ticker), len(records_by_date))

        # Step 1: Fetch float shares for all unique tickers
        unique_tickers = list(records_by_ticker.keys())
        self._fetch_float_shares(unique_tickers)

        # Step 2: Fetch volume data grouped by settlement date
        for settlement_date, date_records in records_by_date.items():
            date_tickers = list(set(r['ticker'] for r in date_records))
            self._fetch_volume_data(date_tickers, settlement_date)

        # Step 3: Update records with enriched data
        stats = {
            'total_records': len(records),
            'records_updated': 0,
            'float_success': 0,
            'volume_success': 0,
            'complete_success': 0,
            'float_missing': 0,
            'volume_missing': 0
        }

        try:
            with session_scope() as session:
                for record in records:
                    record_id = record['id']
                    ticker = record['ticker']
                    settlement_date = record['settlement_date']
                    short_interest_shares = record['short_interest_shares']

                    # Get float shares
                    float_shares = self._float_cache.get(ticker)

                    # Get volume data
                    avg_volume_30d = self._volume_cache.get((ticker, settlement_date))

                    # Calculate short_interest_pct
                    short_interest_pct = None
                    if float_shares and float_shares > 0 and short_interest_shares > 0:
                        short_interest_pct = 100.0 * short_interest_shares / float_shares
                        stats['float_success'] += 1
                    else:
                        stats['float_missing'] += 1

                    # Calculate days_to_cover
                    days_to_cover = None
                    if avg_volume_30d and avg_volume_30d > 0 and short_interest_shares > 0:
                        days_to_cover = short_interest_shares / avg_volume_30d
                        stats['volume_success'] += 1
                    else:
                        stats['volume_missing'] += 1

                    # Track complete enrichment
                    if float_shares and avg_volume_30d:
                        stats['complete_success'] += 1

                    # Update record
                    if not dry_run:
                        # Query the record within this session
                        db_record = session.query(FINRAShortInterest).filter(
                            FINRAShortInterest.id == record_id
                        ).first()

                        if db_record:
                            # Log before update
                            _logger.debug("Updating record id=%d, ticker=%s: float=%s, si_pct=%s, dtc=%s",
                                        record_id, ticker,
                                        float_shares,
                                        f"{short_interest_pct:.2f}%" if short_interest_pct else None,
                                        f"{days_to_cover:.2f}" if days_to_cover else None)

                            db_record.float_shares = float_shares
                            db_record.short_interest_pct = short_interest_pct
                            db_record.days_to_cover = days_to_cover
                            stats['records_updated'] += 1

                            # Flush periodically to avoid memory issues with large batches
                            if stats['records_updated'] % 2 == 0:
                                _logger.info("Progress: Updated %d/%d records, flushing...",
                                           stats['records_updated'], len(records))
                                #session.flush()
                                session.commit()
                        else:
                            _logger.warning("Record with id=%d not found in database", record_id)
                    else:
                        # Dry run - just count what would be updated
                        _logger.debug("DRY RUN: Would update %s (date: %s) - float=%s, si_pct=%.2f%%, dtc=%.2f",
                                    ticker, settlement_date,
                                    float_shares if float_shares else "None",
                                    short_interest_pct if short_interest_pct else 0,
                                    days_to_cover if days_to_cover else 0)
                        stats['records_updated'] += 1

                # Commit all updates
                if not dry_run:
                    _logger.info("Committing %d record updates to database...", stats['records_updated'])
                    session.commit()
                    _logger.info("✅ Successfully committed %d record updates", stats['records_updated'])

                    # Verify a sample of updates
                    if records:
                        sample_record = records[0]
                        verification = session.query(FINRAShortInterest).filter(
                            FINRAShortInterest.id == sample_record['id']
                        ).first()

                        if verification:
                            _logger.info("Verification - Sample record %s:", sample_record['ticker'])
                            _logger.info("  float_shares: %s", verification.float_shares)
                            _logger.info("  short_interest_pct: %s", verification.short_interest_pct)
                            _logger.info("  days_to_cover: %s", verification.days_to_cover)
                        else:
                            _logger.warning("Could not verify sample record")
                else:
                    _logger.info("DRY RUN: Would update %d records", stats['records_updated'])

        except Exception as e:
            _logger.exception("Error updating records:")
            if not dry_run:
                _logger.error("Rolling back transaction due to error")
                session.rollback()
            raise  # Re-raise to see full error in main()

        return stats

    def _fetch_float_shares(self, tickers: List[str]) -> None:
        """
        Fetch float shares for all tickers in batches.

        Args:
            tickers: List of tickers to fetch
        """
        _logger.info("Fetching float shares for %d tickers (batch size: %d)",
                    len(tickers), self.batch_size)

        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(tickers) + self.batch_size - 1) // self.batch_size

            _logger.info("Processing float shares batch %d/%d (%d tickers)",
                        batch_num, total_batches, len(batch))

            try:
                # Use batch fundamentals to get float shares efficiently
                fundamentals_batch = self.yf_downloader.get_fundamentals_batch(batch)

                for ticker in batch:
                    if ticker in fundamentals_batch:
                        fundamentals = fundamentals_batch[ticker]
                        float_shares = fundamentals.float_shares
                        self._float_cache[ticker] = float_shares

                        if float_shares:
                            _logger.debug("Float shares for %s: %s", ticker, f"{float_shares:,}")
                        else:
                            _logger.debug("No float shares data for %s", ticker)
                    else:
                        _logger.warning("No fundamentals data for %s", ticker)
                        self._float_cache[ticker] = None

                # Small delay between batches
                if i + self.batch_size < len(tickers):
                    import time
                    time.sleep(0.5)

            except Exception as e:
                _logger.exception("Error fetching float shares for batch %d:", batch_num)
                # Mark all tickers in failed batch as None
                for ticker in batch:
                    if ticker not in self._float_cache:
                        self._float_cache[ticker] = None

        success_count = sum(1 for v in self._float_cache.values() if v is not None)
        _logger.info("Float shares fetch complete: %d/%d tickers successful (%.1f%%)",
                    success_count, len(tickers), success_count / len(tickers) * 100)

    def _fetch_volume_data(self, tickers: List[str], settlement_date: date) -> None:
        """
        Fetch volume data for tickers for a specific settlement date.

        Args:
            tickers: List of tickers to fetch
            settlement_date: Settlement date (end date for volume calculation)
        """
        _logger.info("Fetching 30-day average volume for %d tickers (settlement: %s)",
                    len(tickers), settlement_date)

        # Calculate date range: 30 trading days before settlement_date
        # Use 45 calendar days to ensure we get 30 trading days
        end_date = datetime.combine(settlement_date, datetime.min.time())
        start_date = end_date - timedelta(days=45)

        _logger.debug("Volume date range: %s to %s", start_date.date(), end_date.date())

        # Process in batches
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(tickers) + self.batch_size - 1) // self.batch_size

            _logger.info("Processing volume batch %d/%d (%d tickers) for %s",
                        batch_num, total_batches, len(batch), settlement_date)

            try:
                # Use batch OHLCV download for efficiency
                ohlcv_batch = self.yf_downloader.get_ohlcv_batch(
                    batch,
                    interval='1d',
                    start_date=start_date,
                    end_date=end_date
                )

                for ticker in batch:
                    cache_key = (ticker, settlement_date)

                    if ticker in ohlcv_batch:
                        df = ohlcv_batch[ticker]

                        if not df.empty and 'volume' in df.columns:
                            # Calculate average volume over the last 30 trading days
                            volumes = df['volume'].tail(30)

                            if len(volumes) > 0:
                                avg_volume = float(volumes.mean())
                                self._volume_cache[cache_key] = avg_volume
                                _logger.debug("30-day avg volume for %s: %s (from %d days)",
                                            ticker, f"{avg_volume:,.0f}", len(volumes))
                            else:
                                _logger.debug("No volume data for %s", ticker)
                                self._volume_cache[cache_key] = None
                        else:
                            _logger.debug("Empty OHLCV data for %s", ticker)
                            self._volume_cache[cache_key] = None
                    else:
                        _logger.warning("No OHLCV data for %s", ticker)
                        self._volume_cache[cache_key] = None

                # Small delay between batches
                if i + self.batch_size < len(tickers):
                    import time
                    time.sleep(0.5)

            except Exception as e:
                _logger.exception("Error fetching volume data for batch %d:", batch_num)
                # Mark all tickers in failed batch as None
                for ticker in batch:
                    cache_key = (ticker, settlement_date)
                    if cache_key not in self._volume_cache:
                        self._volume_cache[cache_key] = None

        # Count successes for this settlement date
        success_count = sum(
            1 for (t, d), v in self._volume_cache.items()
            if d == settlement_date and v is not None
        )
        _logger.info("Volume fetch complete for %s: %d/%d tickers successful (%.1f%%)",
                    settlement_date, success_count, len(tickers),
                    success_count / len(tickers) * 100 if tickers else 0)

    def get_enrichment_summary(self) -> Dict[str, Any]:
        """
        Get summary of cached enrichment data.

        Returns:
            Dictionary with cache statistics
        """
        total_tickers_float = len(self._float_cache)
        float_success = sum(1 for v in self._float_cache.values() if v is not None)

        total_entries_volume = len(self._volume_cache)
        volume_success = sum(1 for v in self._volume_cache.values() if v is not None)

        return {
            'float_cache_size': total_tickers_float,
            'float_success_count': float_success,
            'float_success_rate': float_success / total_tickers_float if total_tickers_float > 0 else 0,
            'volume_cache_size': total_entries_volume,
            'volume_success_count': volume_success,
            'volume_success_rate': volume_success / total_entries_volume if total_entries_volume > 0 else 0
        }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Enrich existing FINRA data with float shares and volume metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ""
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Specific settlement date to enrich (YYYY-MM-DD format)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for date range (YYYY-MM-DD format)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for date range (YYYY-MM-DD format)'
    )

    parser.add_argument(
        '--tickers',
        type=str,
        help='Comma-separated list of tickers to enrich (e.g., AAPL,TSLA,GME)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of tickers to process in each batch (default: 50)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without updating database (preview mode)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-enrichment of already enriched records'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of records to process (for testing)'
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the enrichment utility.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        args = parse_arguments()

        # Setup logging
        if args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)

        _logger.info("Starting FINRA Data Enrichment Utility")
        _logger.info("Arguments: %s", vars(args))

        # Parse dates
        settlement_date = None
        start_date = None
        end_date = None

        if args.date:
            try:
                settlement_date = datetime.strptime(args.date, '%Y-%m-%d').date()
                _logger.info("Enriching specific date: %s", settlement_date)
            except ValueError:
                _logger.error("Invalid date format: %s (expected YYYY-MM-DD)", args.date)
                return 1

        if args.start_date:
            try:
                start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
                _logger.info("Start date: %s", start_date)
            except ValueError:
                _logger.error("Invalid start date format: %s (expected YYYY-MM-DD)", args.start_date)
                return 1

        if args.end_date:
            try:
                end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
                _logger.info("End date: %s", end_date)
            except ValueError:
                _logger.error("Invalid end date format: %s (expected YYYY-MM-DD)", args.end_date)
                return 1

        # Parse tickers
        tickers = None
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(',')]
            _logger.info("Enriching specific tickers: %s", tickers)

        # Initialize Yahoo Finance downloader
        _logger.info("Initializing Yahoo Finance downloader...")
        yf_downloader = YahooDataDownloader()

        # Initialize enrichment utility
        utility = FINRADataEnrichmentUtility(yf_downloader, args.batch_size)

        # Get records to enrich
        _logger.info("Fetching records to enrich...")
        records = utility.get_records_to_enrich(
            settlement_date=settlement_date,
            start_date=start_date,
            end_date=end_date,
            tickers=tickers,
            force=args.force
        )

        if not records:
            _logger.info("No records found to enrich")
            return 0

        # Apply limit if specified
        if args.limit and len(records) > args.limit:
            _logger.info("Limiting to %d records (from %d total)", args.limit, len(records))
            records = records[:args.limit]

        # Enrich records
        _logger.info("Starting enrichment process...")
        start_time = datetime.now()

        stats = utility.enrich_records(records, dry_run=args.dry_run)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Print summary
        _logger.info("=" * 60)
        _logger.info("ENRICHMENT SUMMARY")
        _logger.info("=" * 60)
        _logger.info("Mode: %s", "DRY RUN" if args.dry_run else "LIVE UPDATE")
        _logger.info("Total Records Processed: %d", stats['total_records'])
        _logger.info("Records Updated: %d", stats['records_updated'])
        _logger.info("")
        _logger.info("Float Shares Success: %d/%d (%.1f%%)",
                    stats['float_success'], stats['total_records'],
                    stats['float_success'] / stats['total_records'] * 100 if stats['total_records'] > 0 else 0)
        _logger.info("Volume Data Success: %d/%d (%.1f%%)",
                    stats['volume_success'], stats['total_records'],
                    stats['volume_success'] / stats['total_records'] * 100 if stats['total_records'] > 0 else 0)
        _logger.info("Complete Enrichment: %d/%d (%.1f%%)",
                    stats['complete_success'], stats['total_records'],
                    stats['complete_success'] / stats['total_records'] * 100 if stats['total_records'] > 0 else 0)
        _logger.info("")
        _logger.info("Missing Float Data: %d", stats['float_missing'])
        _logger.info("Missing Volume Data: %d", stats['volume_missing'])
        _logger.info("")
        _logger.info("Duration: %.2f seconds", duration)
        _logger.info("=" * 60)

        # Get cache summary
        cache_summary = utility.get_enrichment_summary()
        _logger.info("")
        _logger.info("Cache Statistics:")
        _logger.info("  Float cache: %d tickers, %d successful (%.1f%%)",
                    cache_summary['float_cache_size'],
                    cache_summary['float_success_count'],
                    cache_summary['float_success_rate'] * 100)
        _logger.info("  Volume cache: %d entries, %d successful (%.1f%%)",
                    cache_summary['volume_cache_size'],
                    cache_summary['volume_success_count'],
                    cache_summary['volume_success_rate'] * 100)

        if args.dry_run:
            _logger.info("")
            _logger.info("⚠️  DRY RUN - No database changes were made")
            _logger.info("Run without --dry-run to apply changes")
        else:
            # Verify database updates
            _logger.info("")
            _logger.info("Verifying database updates...")
            verify_stats = verify_database_updates(records, args.limit if args.limit else 5)
            _logger.info("Verification Results:")
            _logger.info("  Records checked: %d", verify_stats['checked'])
            _logger.info("  Float populated: %d", verify_stats['float_populated'])
            _logger.info("  SI% populated: %d", verify_stats['si_pct_populated'])
            _logger.info("  DTC populated: %d", verify_stats['dtc_populated'])
            _logger.info("  Fully enriched: %d", verify_stats['fully_enriched'])

        _logger.info("")
        _logger.info("Enrichment utility completed successfully")
        return 0

    except KeyboardInterrupt:
        _logger.warning("Enrichment interrupted by user")
        return 130
    except Exception as e:
        _logger.exception("Unexpected error in enrichment utility:")
        return 1


def verify_database_updates(records, sample_size=5):
    """
    Minimal verification stub to avoid NameError.
    When present, the script will log a short message and return a basic dict.
    If you want a stronger verification later, replace this stub with a full check.
    """
    _logger.info(
        "verify_database_updates: stub called (records=%d, sample_size=%d). Skipping deep verification.",
        len(records) if records is not None else 0,
        sample_size
    )
    return {"checked": 0, "missing": 0}


if __name__ == "__main__":
    sys.exit(main())