"""
Backfill TRF Data for Historical Dates

Downloads FINRA TRF data for multiple historical dates and places them
in the appropriate results/emps2/YYYY-MM-DD/ folders.

Usage:
    # Download last 7 days
    python backfill_trf_data.py --days 7

    # Download specific date range
    python backfill_trf_data.py --start 2025-11-25 --end 2025-12-01

    # Download specific dates
    python backfill_trf_data.py --dates 2025-11-25 2025-11-26 2025-11-27
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Also try adding current directory
import os
os.chdir(PROJECT_ROOT)

from src.data.downloader.finra_data_downloader import FinraDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def is_weekend(date: datetime) -> bool:
    """Check if date is a weekend."""
    return date.weekday() >= 5  # 5=Saturday, 6=Sunday


def backfill_trf_data(dates: list[datetime]) -> dict:
    """
    Download TRF data for multiple dates.

    Args:
        dates: List of datetime objects to download

    Returns:
        Dictionary with results: {date: status}
    """
    results = {}

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")

        # Skip weekends
        if is_weekend(date):
            _logger.info("Skipping %s (weekend)", date_str)
            results[date_str] = "skipped_weekend"
            continue

        # Check if already exists
        output_dir = Path("results") / "emps2" / date_str
        trf_file = output_dir / "trf.csv"

        if trf_file.exists():
            _logger.info("TRF data already exists for %s: %s", date_str, trf_file)
            results[date_str] = "already_exists"
            continue

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download TRF data
        _logger.info("="*70)
        _logger.info("Downloading TRF data for %s", date_str)
        _logger.info("="*70)

        try:
            downloader = FinraDataDownloader(
                date=date_str,
                output_dir=output_dir,
                output_filename="trf.csv",
                fetch_yfinance_data=True
            )

            result_df = downloader.run()

            if result_df.empty:
                _logger.warning("No TRF data for %s (market may be closed)", date_str)
                results[date_str] = "no_data"
            else:
                ticker_count = len(result_df["ticker"].unique())
                _logger.info("Successfully downloaded TRF data for %s: %d tickers",
                           date_str, ticker_count)
                results[date_str] = f"success_{ticker_count}_tickers"

        except Exception as e:
            _logger.error("Failed to download TRF data for %s: %s", date_str, str(e))
            results[date_str] = f"error_{str(e)}"

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill FINRA TRF data for historical dates"
    )

    # Option 1: Last N days
    parser.add_argument(
        "--days",
        type=int,
        default=8,
        help="Download TRF data for last N days (default: 7)"
    )

    # Option 2: Date range
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)"
    )

    # Option 3: Specific dates
    parser.add_argument(
        "--dates",
        nargs="+",
        help="Specific dates to download (YYYY-MM-DD format)"
    )

    args = parser.parse_args()

    # Determine dates to download
    dates = []

    if args.dates:
        # Specific dates provided
        for date_str in args.dates:
            try:
                dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
            except ValueError:
                _logger.error("Invalid date format: %s (use YYYY-MM-DD)", date_str)
                return 1

    elif args.start and args.end:
        # Date range provided
        try:
            start = datetime.strptime(args.start, "%Y-%m-%d")
            end = datetime.strptime(args.end, "%Y-%m-%d")

            current = start
            while current <= end:
                dates.append(current)
                current += timedelta(days=1)

        except ValueError as e:
            _logger.error("Invalid date format: %s", str(e))
            return 1

    else:
        # Default: Last N days (default 7)
        days = args.days if args.days else 7
        today = datetime.now()

        for i in range(days, 0, -1):
            date = today - timedelta(days=i)
            dates.append(date)

    # Download TRF data
    _logger.info("Starting TRF backfill for %d dates", len(dates))
    _logger.info("Date range: %s to %s",
                dates[0].strftime("%Y-%m-%d"),
                dates[-1].strftime("%Y-%m-%d"))

    results = backfill_trf_data(dates)

    # Print summary
    _logger.info("="*70)
    _logger.info("TRF Backfill Summary")
    _logger.info("="*70)

    success_count = sum(1 for v in results.values() if v.startswith("success"))
    already_exists = sum(1 for v in results.values() if v == "already_exists")
    skipped = sum(1 for v in results.values() if v == "skipped_weekend")
    no_data = sum(1 for v in results.values() if v == "no_data")
    errors = sum(1 for v in results.values() if v.startswith("error"))

    _logger.info("Total dates processed: %d", len(dates))
    _logger.info("  - Successfully downloaded: %d", success_count)
    _logger.info("  - Already existed: %d", already_exists)
    _logger.info("  - Skipped (weekend): %d", skipped)
    _logger.info("  - No data (holiday): %d", no_data)
    _logger.info("  - Errors: %d", errors)

    if errors > 0:
        _logger.info("")
        _logger.info("Failed dates:")
        for date_str, status in results.items():
            if status.startswith("error"):
                _logger.info("  - %s: %s", date_str, status)

    _logger.info("="*70)

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
