"""
FINRA TRF Downloader Wrapper

Wrapper script for backward compatibility with p06_emps2 pipeline.
This script calls the main FINRA TRF downloader from src/data/downloader.

For direct usage of the downloader class, import from:
    from src.data.downloader.finra_trf_downloader import FinraTRFDownloader
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.finra_trf_downloader import FinraTRFDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def main():
    """
    Main entry point for command-line usage.

    Wrapper that calls FinraTRFDownloader with p06_emps2-specific output paths.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Download FINRA TRF data for EMPS2 pipeline")
    parser.add_argument(
        "--date",
        type=str,
        help="Date in YYYY-MM-DD format (default: yesterday)"
    )
    parser.add_argument(
        "--no-yfinance",
        action="store_true",
        help="Skip fetching yfinance volume data"
    )

    args = parser.parse_args()

    try:
        # Parse date for output directory
        if args.date:
            date_obj = datetime.strptime(args.date, "%Y-%m-%d")
        else:
            from datetime import timedelta
            date_obj = datetime.now() - timedelta(days=1)

        # Use p06_emps2 results directory structure
        output_dir = Path("results") / "emps2" / date_obj.strftime("%Y-%m-%d")

        _logger.info("Running FINRA TRF downloader for EMPS2 pipeline")
        _logger.info("Output directory: %s", output_dir)

        # Create downloader with EMPS2-specific settings
        downloader = FinraTRFDownloader(
            date=args.date,
            output_dir=output_dir,
            output_filename="trf.csv",  # Keep original filename for pipeline compatibility
            fetch_yfinance_data=not args.no_yfinance
        )

        # Run download
        result_df = downloader.run()

        if result_df.empty:
            _logger.warning("No TRF data downloaded (market may be closed)")
            return 0

        _logger.info("Successfully downloaded TRF data for %d tickers", len(result_df["ticker"].unique()))
        return 0

    except Exception as e:
        _logger.critical("Fatal error in TRF downloader: %s", str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
