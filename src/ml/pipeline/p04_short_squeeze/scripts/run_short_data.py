#!/usr/bin/env python3
"""
Short Interest Data Collector from ss_snapshot

Collects current short interest data for all tickers in the ss_snapshot table
and upserts them into ss_finra_short_interest table using yfinance data.

This script replaces run_finra_collector.py for getting short interest data
from Yahoo Finance instead of FINRA official data.

Usage:
    python run_short_data.py
    python run_short_data.py --batch-size 25 --dry-run
    python run_short_data.py --settlement-date 2024-01-15
"""

import argparse
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.data.db.core.database import session_scope
from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.data.db.models.model_short_squeeze import ScreenerSnapshot
from src.notification.logger import setup_logger
from sqlalchemy import select, func

_logger = setup_logger(__name__)


class ShortDataCollector:
    """
    Collects short interest data for tickers from ss_snapshot table using Yahoo Finance.

    Fetches all tickers from ss_snapshot table and collects current short interest data
    from Yahoo Finance, then upserts into ss_finra_short_interest table.
    """

    def __init__(self, batch_size: int = 50, settlement_date: Optional[date] = None):
        """
        Initialize the collector.

        Args:
            batch_size: Number of tickers to process per batch
            settlement_date: Settlement date to use (defaults to today)
        """
        self.yf_downloader = YahooDataDownloader()
        self.batch_size = batch_size
        self.settlement_date = settlement_date or date.today()

    def get_tickers_from_snapshot(self) -> List[str]:
        """
        Get all distinct tickers from ss_snapshot table.

        Returns:
            List of unique ticker symbols
        """
        _logger.info("Fetching tickers from ss_snapshot table...")

        try:
            with session_scope() as session:
                # Get all distinct tickers from ss_snapshot table
                result = session.execute(
                    select(func.distinct(ScreenerSnapshot.ticker))
                    .order_by(ScreenerSnapshot.ticker)
                )
                tickers = [row[0] for row in result.fetchall()]

            _logger.info("Found %d unique tickers in ss_snapshot table", len(tickers))
            return tickers

        except Exception:
            _logger.exception("Error fetching tickers from ss_snapshot table:")
            return []

    def collect_short_interest(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Collect short interest data for a list of tickers using Yahoo Finance.

        Args:
            tickers: List of ticker symbols

        Returns:
            List of dictionaries with short interest data formatted for ss_finra_short_interest table
        """
        _logger.info("Collecting short interest data for %d tickers using settlement_date %s",
                    len(tickers), self.settlement_date)

        all_records = []

        # Process in batches to respect rate limits
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(tickers) + self.batch_size - 1) // self.batch_size

            _logger.info("Processing batch %d/%d (%d tickers)",
                        batch_num, total_batches, len(batch))

            try:
                # Get fundamentals for this batch (includes short interest data)
                fundamentals_batch = self.yf_downloader.get_fundamentals_batch(batch)

                # Extract short interest data
                for ticker in batch:
                    if ticker not in fundamentals_batch:
                        _logger.warning("No data for %s", ticker)
                        continue

                    fund = fundamentals_batch[ticker]

                    # Skip if no company name (invalid ticker)
                    if fund.company_name == "Unknown":
                        _logger.warning("Invalid ticker or no data: %s", ticker)
                        continue

                    # Get shares short from individual ticker info (required for accurate data)
                    shares_short = self._get_shares_short(ticker)

                    # Calculate short interest percentage
                    short_interest_pct = None
                    if shares_short and shares_short > 0:
                        # Try to use total shares outstanding first (more reliable for percentage calculation)
                        if fund.shares_outstanding and fund.shares_outstanding > 0:
                            short_interest_pct = (shares_short / fund.shares_outstanding) * 100.0
                        # Fallback to float shares if outstanding shares not available
                        elif fund.float_shares and fund.float_shares > 0:
                            short_interest_pct = (shares_short / fund.float_shares) * 100.0

                        # Data validation: short interest percentage should not exceed 100%
                        # If it does, there's likely a data quality issue
                        if short_interest_pct and short_interest_pct > 100.0:
                            _logger.warning("Invalid short interest percentage for %s: %.2f%% (shares_short=%d, outstanding=%s, float=%s). Capping at 100%%",
                                          ticker, short_interest_pct, shares_short,
                                          fund.shares_outstanding, fund.float_shares)
                            short_interest_pct = min(short_interest_pct, 100.0)

                    # Get days to cover from short_ratio (if available)
                    days_to_cover = fund.short_ratio

                    # Create raw data payload for audit trail
                    raw_data = {
                        'company_name': fund.company_name,
                        'current_price': fund.current_price,
                        'market_cap': fund.market_cap,
                        'sector': fund.sector,
                        'industry': fund.industry,
                        'data_source': 'Yahoo Finance',
                        'collection_timestamp': datetime.now().isoformat(),
                        'short_ratio': fund.short_ratio,
                        'shares_short_raw': shares_short
                    }

                    # Format record for ss_finra_short_interest table
                    record = {
                        'ticker': ticker.upper(),
                        'settlement_date': self.settlement_date,
                        'short_interest_shares': shares_short or 0,
                        'total_shares_outstanding': fund.shares_outstanding,
                        'float_shares': fund.float_shares,
                        'short_interest_pct': short_interest_pct,
                        'days_to_cover': days_to_cover,
                        'data_source': 'Yahoo Finance',
                        'data_quality_score': 0.8,  # Yahoo Finance data quality score
                        'raw_data': raw_data
                    }

                    all_records.append(record)

                    _logger.debug("✓ %s: SI_shares=%s, SI%%=%.2f%%, DTC=%.2f, Float=%s",
                                ticker,
                                shares_short or 0,
                                short_interest_pct if short_interest_pct else 0,
                                days_to_cover if days_to_cover else 0,
                                f"{fund.float_shares:,}" if fund.float_shares else "N/A")

                # Rate limiting between batches
                if i + self.batch_size < len(tickers):
                    time.sleep(1)

            except Exception:
                _logger.exception("Error processing batch %d:", batch_num)
                continue

        _logger.info("Collected data for %d/%d tickers", len(all_records), len(tickers))
        return all_records

    def _get_shares_short(self, ticker: str) -> Optional[int]:
        """
        Get shares short for a single ticker.
        This requires an individual API call as it's not in the batch fundamentals.

        Args:
            ticker: Ticker symbol

        Returns:
            Number of shares short, or None if unavailable
        """
        try:
            import yfinance as yf
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            shares_short = info.get('sharesShort', None)

            # Validate the data
            if shares_short is not None and shares_short < 0:
                _logger.warning("Invalid shares short value for %s: %s", ticker, shares_short)
                return None

            return shares_short
        except Exception as e:
            _logger.debug("Could not get shares short for %s: %s", ticker, e)
            return None

    def filter_by_short_interest_pct(self, records: List[Dict[str, Any]],
                                     min_pct: float = 15.0) -> List[Dict[str, Any]]:
        """
        Filter records by minimum short interest percentage.

        Args:
            records: List of short interest records
            min_pct: Minimum short interest percentage threshold

        Returns:
            Filtered list of records
        """
        filtered = [
            r for r in records
            if r.get('short_interest_pct') is not None
            and r['short_interest_pct'] >= min_pct
        ]

        _logger.info("Filtered to %d/%d tickers with SI%% >= %.1f%%",
                    len(filtered), len(records), min_pct)

        return filtered

    def save_to_csv(self, records: List[Dict[str, Any]], filepath: str) -> None:
        """
        Save records to CSV file.

        Args:
            records: List of short interest records
            filepath: Output CSV file path
        """
        try:
            # Flatten the records for CSV export
            csv_records = []
            for record in records:
                raw_data = record.get('raw_data', {})
                csv_record = {
                    'ticker': record['ticker'],
                    'settlement_date': record['settlement_date'],
                    'short_interest_shares': record.get('short_interest_shares', 0),
                    'short_interest_pct': record.get('short_interest_pct'),
                    'days_to_cover': record.get('days_to_cover'),
                    'float_shares': record.get('float_shares'),
                    'total_shares_outstanding': record.get('total_shares_outstanding'),
                    'data_source': record.get('data_source', 'Yahoo Finance'),
                    'data_quality_score': record.get('data_quality_score', 0.8),
                    'company_name': raw_data.get('company_name'),
                    'current_price': raw_data.get('current_price'),
                    'market_cap': raw_data.get('market_cap'),
                    'sector': raw_data.get('sector'),
                    'industry': raw_data.get('industry'),
                    'collection_timestamp': raw_data.get('collection_timestamp')
                }
                csv_records.append(csv_record)

            df = pd.DataFrame(csv_records)

            # Sort by short interest percentage (descending)
            if 'short_interest_pct' in df.columns:
                df = df.sort_values('short_interest_pct', ascending=False, na_last=True)

            df.to_csv(filepath, index=False)
            _logger.info("Saved %d records to %s", len(records), filepath)

        except Exception:
            _logger.exception("Error saving to CSV:")
            raise

    def save_to_database(self, records: List[Dict[str, Any]],
                        dry_run: bool = False) -> int:
        """
        Save records to ss_finra_short_interest table using upsert logic.

        Args:
            records: List of short interest records (already formatted for database)
            dry_run: If True, don't actually write to database

        Returns:
            Number of records saved
        """
        if dry_run:
            _logger.info("DRY RUN: Would upsert %d records to ss_finra_short_interest table", len(records))
            if records:
                sample = records[0]
                _logger.info("Sample record: ticker=%s, settlement_date=%s, short_interest_shares=%s, short_interest_pct=%s",
                           sample.get('ticker'), sample.get('settlement_date'),
                           sample.get('short_interest_shares'), sample.get('short_interest_pct'))
            return len(records)

        try:
            # Use the service layer for proper data handling
            service = ShortSqueezeService()
            count = service.store_finra_data(records)

            _logger.info("Successfully upserted %d records to ss_finra_short_interest table", count)
            return count

        except Exception:
            _logger.exception("Error saving to database:")
            return 0

    def generate_report(self, records: List[Dict[str, Any]]) -> None:
        """
        Generate and log a summary report.

        Args:
            records: List of short interest records
        """
        if not records:
            _logger.info("No records to report")
            return

        _logger.info("=" * 60)
        _logger.info("SHORT INTEREST DATA COLLECTION REPORT")
        _logger.info("=" * 60)
        _logger.info("Settlement Date: %s", self.settlement_date)
        _logger.info("Data Source: Yahoo Finance")
        _logger.info("Total Tickers Processed: %d", len(records))

        # Calculate statistics
        valid_si = [r['short_interest_pct'] for r in records
                   if r.get('short_interest_pct') is not None and r['short_interest_pct'] > 0]

        valid_shares = [r['short_interest_shares'] for r in records
                       if r.get('short_interest_shares') is not None and r['short_interest_shares'] > 0]

        _logger.info("Records with Short Interest Data: %d/%d (%.1f%%)",
                    len(valid_si), len(records),
                    (len(valid_si) / len(records) * 100) if records else 0)

        if valid_si:
            _logger.info("Short Interest %% Stats:")
            _logger.info("  Min:    %.2f%%", min(valid_si))
            _logger.info("  Max:    %.2f%%", max(valid_si))
            _logger.info("  Avg:    %.2f%%", sum(valid_si) / len(valid_si))
            _logger.info("  Median: %.2f%%", sorted(valid_si)[len(valid_si)//2])

        if valid_shares:
            _logger.info("Short Interest Shares Stats:")
            _logger.info("  Min:    %s", f"{min(valid_shares):,}")
            _logger.info("  Max:    %s", f"{max(valid_shares):,}")
            _logger.info("  Avg:    %s", f"{int(sum(valid_shares) / len(valid_shares)):,}")

        # Top 10 by short interest percentage
        sorted_records = sorted(
            [r for r in records if r.get('short_interest_pct') is not None and r['short_interest_pct'] > 0],
            key=lambda x: x['short_interest_pct'],
            reverse=True
        )[:10]

        if sorted_records:
            _logger.info("\nTop 10 Tickers by Short Interest %%:")
            _logger.info("-" * 60)
            for i, record in enumerate(sorted_records, 1):
                company_name = record.get('raw_data', {}).get('company_name', 'Unknown')
                _logger.info("%2d. %-6s %6.2f%%  DTC: %5.1f  SI_Shares: %s  %s",
                           i,
                           record['ticker'],
                           record['short_interest_pct'],
                           record.get('days_to_cover', 0) or 0,
                           f"{record.get('short_interest_shares', 0):,}",
                           company_name[:25])

        _logger.info("=" * 60)





def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Short Interest Data Collector from ss_snapshot table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect for all tickers in ss_snapshot table
  python run_short_data.py

  # Collect with custom batch size and dry run
  python run_short_data.py --batch-size 25 --dry-run

  # Collect with specific settlement date
  python run_short_data.py --settlement-date 2024-01-15

  # Filter by minimum short interest percentage
  python run_short_data.py --min-si-pct 10.0

  # Save to CSV file
  python run_short_data.py --output short_interest_data.csv

Note: This script fetches all tickers from ss_snapshot table and collects
      current short interest data from Yahoo Finance, then upserts into
      ss_finra_short_interest table with ticker/settlement_date as primary key.
        """
    )

    # Settlement date option
    parser.add_argument(
        '--settlement-date',
        type=str,
        help='Settlement date to use (YYYY-MM-DD format, default: today)'
    )

    # Filter options
    parser.add_argument(
        '--min-si-pct',
        type=float,
        default=0.0,
        help='Minimum short interest percentage threshold (default: 0.0)'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output CSV file path'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode (no database writes)'
    )

    # Processing options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for processing (default: 50)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Parse settlement date if provided
        settlement_date = None
        if args.settlement_date:
            try:
                settlement_date = datetime.strptime(args.settlement_date, '%Y-%m-%d').date()
            except ValueError:
                _logger.error("Invalid settlement date format '%s', expected YYYY-MM-DD", args.settlement_date)
                return 1

        _logger.info("Starting short interest data collection from ss_snapshot table")
        _logger.info("Settlement date: %s", settlement_date or date.today())

        # Initialize collector
        collector = ShortDataCollector(
            batch_size=args.batch_size,
            settlement_date=settlement_date
        )

        # Get tickers from ss_snapshot table
        tickers = collector.get_tickers_from_snapshot()
        if not tickers:
            _logger.error("No tickers found in ss_snapshot table")
            return 1

        _logger.info("Processing %d tickers from ss_snapshot table", len(tickers))

        # Collect short interest data
        records = collector.collect_short_interest(tickers)

        if not records:
            _logger.warning("No short interest data collected")
            return 1

        # Filter by short interest percentage if requested
        if args.min_si_pct > 0:
            records = collector.filter_by_short_interest_pct(records, args.min_si_pct)

        # Generate report
        collector.generate_report(records)

        # Save to CSV if requested
        if args.output:
            collector.save_to_csv(records, args.output)

        # Save to database (always enabled, use --dry-run to prevent writes)
        count = collector.save_to_database(records, dry_run=args.dry_run)
        if count > 0:
            _logger.info("✓ Successfully upserted %d records to ss_finra_short_interest table", count)
        elif not args.dry_run:
            _logger.warning("No records were saved to database")

        _logger.info("Short interest data collection completed successfully")
        return 0

    except KeyboardInterrupt:
        _logger.warning("Interrupted by user")
        return 130
    except Exception:
        _logger.exception("Error during collection:")
        return 1


if __name__ == "__main__":
    sys.exit(main())