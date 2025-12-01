"""
TRF and Volume Data Downloader

Downloads TRF (Trade Reporting Facility) data from FINRA and volume data from yfinance.
Saves results in results/emps2/YYYY-MM-DD/trf.csv.

This script is designed to be run daily at 6 AM via cron.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
import json
import time

import pandas as pd
import requests
import yfinance as yf
from dateutil.relativedelta import relativedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

# Set up logger
_logger = setup_logger(__name__)


class TRFDownloader:
    """
    Downloads and processes TRF (Trade Reporting Facility) data from FINRA.
    Also fetches volume data from yfinance for the same tickers.
    """

    FINRA_TRF_URL = "https://api.finra.org/data/trf/OTCMarketVolume"
    RESULTS_DIR = Path("results") / "emps2"
    OUTPUT_FILENAME = "trf.csv"

    def __init__(self, date: Optional[str] = None):
        """
        Initialize TRF downloader.

        Args:
            date: Date string in 'YYYY-MM-DD' format. If None, uses yesterday's date.
        """
        self.date = self._parse_date(date)
        self.output_dir = self.RESULTS_DIR / self.date.strftime("%Y-%m-%d")
        self.output_file = self.output_dir / self.OUTPUT_FILENAME

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_weekend_or_holiday(date: datetime) -> bool:
        """Check if the date is a weekend or public holiday."""
        # Check for weekend
        if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return True

        # Add public holidays check here if needed
        # Example: if (date.month, date.day) in [(1, 1), (12, 25)]:  # New Year's Day, Christmas
        #     return True

        return False

    @staticmethod
    def _parse_date(date_str: Optional[str] = None) -> datetime:
        """Parse date string or return yesterday's date if None."""
        if date_str:
            return datetime.strptime(date_str, "%Y-%m-%d")
        return datetime.now() - timedelta(days=1)

    def _download_trf_data(self) -> pd.DataFrame:
        """Download TRF data from FINRA API."""
        date_str = self.date.strftime("%Y-%m-%d")
        _logger.info("Downloading TRF data for %s", date_str)

        try:
            response = requests.get(
                self.FINRA_TRF_URL,
                params={"tradeDate": date_str},
                timeout=30
            )

            # Handle 404 specifically
            if response.status_code == 404:
                if self._is_weekend_or_holiday(self.date):
                    _logger.info("Market was closed on %s (weekend or holiday)", date_str)
                    return pd.DataFrame()
                else:
                    response.raise_for_status()  # Will raise HTTPError for 404 on trading day

            # Handle other errors
            response.raise_for_status()

            data = response.json()
            df = pd.DataFrame(data)

            if df.empty:
                _logger.warning("No TRF data found for %s", date_str)
                return df

            # Standardize column names
            df = df.rename(columns={
                "issueSymbol": "ticker",
                "totalVolume": "offex_volume",
                "totalTrades": "offex_trades",
                "tradeDate": "date"
            })

            # Convert date column
            df["date"] = pd.to_datetime(df["date"]).dt.date

            # Keep only relevant columns
            keep_columns = ["date", "ticker", "offex_volume", "offex_trades", "lastPrice", "marketCategory"]
            df = df[[col for col in keep_columns if col in df.columns]]

            return df

        except requests.RequestException as e:
            _logger.error("Failed to download TRF data: %s", str(e))
            raise

    def _get_volume_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch volume data from yfinance for the given tickers."""
        if not tickers:
            return pd.DataFrame()

        _logger.info("Fetching volume data for %d tickers", len(tickers))

        try:
            # Download data for all tickers
            data = yf.download(
                tickers,
                start=self.date - timedelta(days=1),  # Include previous day in case of timezone issues
                end=self.date + timedelta(days=1),    # Include next day in case of timezone issues
                group_by='ticker',
                progress=False
            )

            # Process the data
            volume_data = []
            for ticker in tickers:
                try:
                    if ticker in data:
                        # Get the data for this ticker
                        ticker_data = data[ticker] if len(tickers) > 1 else data

                        # Get the row for our target date
                        target_date = self.date.strftime("%Y-%m-%d")
                        if target_date in ticker_data.index:
                            row = ticker_data.loc[target_date]
                            volume_data.append({
                                "ticker": ticker,
                                "date": self.date.date(),
                                "volume": row["Volume"],
                                "open": row["Open"],
                                "high": row["High"],
                                "low": row["Low"],
                                "close": row["Close"],
                                "adj_close": row.get("Adj Close", row["Close"])
                            })
                except Exception as e:
                    _logger.warning("Failed to process ticker %s: %s", ticker, str(e))

            return pd.DataFrame(volume_data)

        except Exception as e:
            _logger.error("Failed to fetch volume data: %s", str(e))
            return pd.DataFrame()

    def _load_previous_universe(self) -> List[str]:
        """Load ticker universe from previous day's results if available."""
        # Get yesterday's date
        prev_date = self.date - timedelta(days=1)
        prev_dir = self.RESULTS_DIR / prev_date.strftime("%Y-%m-%d")

        # Look for universe files
        for filename in ["universe.csv", "tickers.csv"]:
            filepath = prev_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    if "ticker" in df.columns:
                        return df["ticker"].tolist()
                    elif "symbol" in df.columns:
                        return df["symbol"].tolist()
                except Exception as e:
                    _logger.warning("Failed to load previous universe from %s: %s", filepath, str(e))

        return []

    def run(self) -> None:
        """Run the TRF and volume data download process."""
        _logger.info("Starting TRF and volume data download for %s", self.date.strftime("%Y-%m-%d"))

        try:
            # Step 1: Download TRF data
            trf_df = self._download_trf_data()

            # If no data and it's a weekend/holiday, just return
            if trf_df.empty:
                _logger.info("No data available (market closed or no data)")
                return

            # Get list of unique tickers
            tickers = trf_df["ticker"].unique().tolist()
            _logger.info("Found %d unique tickers in TRF data", len(tickers))

            # Step 2: Get volume data for these tickers
            if not tickers:
                _logger.warning("No tickers found for volume data download")
                volume_df = pd.DataFrame()
            else:
                volume_df = self._get_volume_data(tickers)

            # Step 3: Merge TRF and volume data
            if not volume_df.empty:
                # Merge on ticker and date
                result_df = pd.merge(
                    trf_df,
                    volume_df,
                    on=["ticker", "date"],
                    how="left"
                )

                # Calculate off-exchange ratio
                result_df["offex_ratio"] = result_df["offex_volume"] / result_df["volume"]
            else:
                result_df = trf_df
                _logger.warning("No volume data available for any tickers")

            # Save results
            result_df.to_csv(self.output_file, index=False)
            _logger.info("Saved TRF and volume data to %s", self.output_file)

        except Exception as e:
            _logger.error("Error in TRF download process: %s", str(e), exc_info=True)
            raise


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Download TRF and volume data")
    parser.add_argument(
        "--date",
        type=str,
        help="Date in YYYY-MM-DD format (default: yesterday)"
    )

    args = parser.parse_args()

    try:
        downloader = TRFDownloader(args.date)
        downloader.run()
    except Exception as e:
        _logger.critical("Fatal error in TRF downloader: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()