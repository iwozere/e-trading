"""
FINRA TRF Downloader Wrapper

Wrapper script for backward compatibility with p06_emps2 pipeline.
This script calls the main FINRA TRF downloader from src/data/downloader.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.finra_data_downloader import FinraDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def get_previous_trading_day(date: Optional[datetime] = None) -> datetime:
    """Get the previous trading day (Monday-Friday)"""
    if date is None:
        date = datetime.now()
    # If it's Monday, go back to Friday
    if date.weekday() == 0:  # Monday
        return date - timedelta(days=3)
    # If it's Sunday, go back to Friday
    elif date.weekday() == 6:  # Sunday
        return date - timedelta(days=2)
    # Otherwise just go back one day
    return date - timedelta(days=1)

def download_trf(target_date: Optional[datetime] = None, force_download: bool = False) -> Path:
    """
    Download TRF data for the specified date.

    Args:
        target_date: Date to download TRF data for. If None, uses previous trading day.
        force_download: If True, forces re-download even if file exists

    Returns:
        Path to the downloaded TRF file
    """
    # Get the actual date of the TRF data
    if target_date is None or (isinstance(target_date, datetime) and target_date.date() == date.today()):
        trf_date = get_previous_trading_day()
    else:
        trf_date = target_date

    date_str = trf_date.strftime('%Y-%m-%d')
    # Default to a shared location if no parent context
    trf_dir = Path("results") / "trf_data" / date_str
    output_file = trf_dir / "trf.csv"

    # Check if file already exists
    if output_file.exists() and not force_download:
        _logger.info(f"TRF file already exists: {output_file}")
        return output_file

    # Create output directory
    trf_dir.mkdir(parents=True, exist_ok=True)

    # Download the TRF data
    # NOTE: fetch_yfinance_data=False is important — the downstream consumer
    # (get_trf_correction_factor below, plus volatility_filter / accumulation_analyzer)
    # only reads FINRA's own total_volume / short_volume columns. The default
    # yfinance merge would download ~9k tickers of daily bars (serial, batched)
    # which takes ~10 minutes on a Raspberry Pi and is entirely unused.
    _logger.info(f"Downloading TRF data for {date_str}")
    downloader = FinraDataDownloader(
        date=date_str,
        output_dir=trf_dir,
        output_filename="trf.csv",
        fetch_yfinance_data=False,
    )

    try:
        downloader.run()
        if output_file.exists():
            _logger.info("Successfully downloaded TRF data to %s", output_file)
        else:
            # FINRA returned no data (market closed / holiday) — write an empty sentinel
            # so the cache check on subsequent calls doesn't trigger re-downloads.
            output_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"ticker": [], "short_volume": [], "total_volume": []}).to_csv(output_file, index=False)
            _logger.info("No TRF data for %s (market closed?) — wrote empty sentinel: %s", date_str, output_file)
        return output_file
    except Exception as e:
        _logger.error("Failed to download TRF data: %s", str(e))
        raise

def get_trf_correction_factor(ticker: str, date: datetime) -> float:
    """
    Get TRF correction factor for a ticker on a specific date.

    Args:
        ticker: Stock ticker symbol
        date: Date to get correction factor for

    Returns:
        Correction factor (1.0 if no data available)
    """
    date_str = date.strftime('%Y-%m-%d')
    # Check multiple possible locations (shared or pipeline specific)
    possible_paths = [
        Path("results") / "trf_data" / date_str / "trf.csv",
        Path("results") / "p06_emps2" / date_str / "trf.csv",
        Path("results") / "p10_emps3" / date_str / "trf.csv",
        Path("results") / "emps2" / date_str / "trf.csv" # Legacy
    ]
    
    trf_file = possible_paths[0] # Default for download
    for p in possible_paths:
        if p.exists():
            trf_file = p
            break

    # If file doesn't exist, try to download it
    if not trf_file.exists():
        try:
            download_trf(date)
        except Exception as e:
            _logger.warning(f"Failed to download TRF data for {date_str}: {str(e)}")

    # If file still doesn't exist, try previous days (up to 5 days back)
    max_days_back = 5
    current_date = date
    while not trf_file.exists() and max_days_back > 0:
        current_date = current_date - timedelta(days=1)
        date_str = current_date.strftime('%Y-%m-%d')
        # Check all possible paths for previous day too
        for p_base in ["trf_data", "p06_emps2", "p10_emps3", "emps2"]:
            p = Path("results") / p_base / date_str / "trf.csv"
            if p.exists():
                trf_file = p
                break
        max_days_back -= 1

    # If we found a file, read it and get the correction factor
    if trf_file.exists():
        try:
            df = pd.read_csv(trf_file)
            ticker_data = df[df['ticker'] == ticker.upper()]
            if not ticker_data.empty:
                # Calculate correction factor (example: total_volume / (total_volume - short_volume))
                total_volume = ticker_data['total_volume'].iloc[0]
                short_volume = ticker_data['short_volume'].iloc[0]
                if total_volume > 0 and short_volume < total_volume:
                    return total_volume / (total_volume - short_volume)
        except Exception as e:
            _logger.error(f"Error reading TRF file {trf_file}: {str(e)}")

    # Return 1.0 (no correction) if we couldn't find or read the file
    return 1.0

def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Download FINRA TRF data')
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format (default: previous trading day)')
    parser.add_argument('--force', action='store_true', help='Force re-download even if file exists')

    args = parser.parse_args()

    try:
        target_date = datetime.strptime(args.date, '%Y-%m-%d') if args.date else None
        download_trf(target_date, args.force)
    except Exception as e:
        _logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()