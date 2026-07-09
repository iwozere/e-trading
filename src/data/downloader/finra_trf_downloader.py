"""
FINRA TRF (Trade Reporting Facility) Data Downloader

Downloads daily TRF data from FINRA API (Reg SHO Daily Short Sale Volume).
Uses OAuth 2.0 authentication to access FINRA's API.

Data includes:
- Short sale volume by ticker
- Total trading volume
- Reporting facility information
- Market codes

Can optionally merge with yfinance volume data for validation.
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import requests
import yfinance as yf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from config.donotshare.donotshare import DATA_CACHE_DIR, FINRA_API_CLIENT, FINRA_API_SECRET
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class FinraTRFDownloader:
    """
    Downloads and processes TRF (Trade Reporting Facility) data from FINRA API.
    Optionally fetches volume data from yfinance for the same tickers.
    """

    FINRA_API_BASE = "https://api.finra.org/data"
    FINRA_AUTH_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"
    FINRA_GROUP = "otcmarket"
    FINRA_DATASET = "regShoDaily"  # Using daily short sale volume dataset

    @property
    def finra_url(self) -> str:
        """Construct the FINRA API URL."""
        return f"{self.FINRA_API_BASE}/group/{self.FINRA_GROUP}/name/{self.FINRA_DATASET}"

    def __init__(
        self,
        date: str | None = None,
        output_dir: Path | None = None,
        output_filename: str = "finra_trf.csv",
        fetch_yfinance_data: bool = True,
    ):
        """
        Initialize FINRA TRF downloader.

        Args:
            date: Date string in 'YYYY-MM-DD' format. If None, uses yesterday's date.
            output_dir: Directory to save output file. If None, creates output_dir based on date.
            output_filename: Name of the output CSV file.
            fetch_yfinance_data: Whether to fetch and merge yfinance volume data.
        """
        self.date = self._parse_date(date)
        self.fetch_yfinance_data = fetch_yfinance_data

        # Set output paths
        if output_dir is None:
            self.output_dir = Path("results") / "finra_trf" / self.date.strftime("%Y-%m-%d")
        else:
            self.output_dir = Path(output_dir)

        self.output_file = self.output_dir / output_filename
        self._cache_dir = Path(DATA_CACHE_DIR) / "trf"

        # OAuth token management
        self._access_token: str | None = None
        self._token_expires_at: datetime | None = None

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
    def _parse_date(date_str: str | None = None) -> datetime:
        """Parse date string or return yesterday's date if None."""
        if date_str:
            return datetime.strptime(date_str, "%Y-%m-%d")
        return datetime.now() - timedelta(days=1)

    def get_provider_name(self) -> str:
        """Return the canonical provider name for this downloader."""
        return "finra"

    def _get_access_token(self) -> str:
        """
        Obtain OAuth 2.0 access token from FINRA Identity Platform.

        Returns:
            Access token string.

        Raises:
            requests.RequestException: If token retrieval fails.
        """
        # Check if we have a valid cached token
        if self._access_token and self._token_expires_at:
            if datetime.now(UTC) < self._token_expires_at:
                _logger.debug("Using cached access token")
                return self._access_token

        _logger.info("Requesting new access token from FINRA")

        try:
            # Use Basic Auth with Client ID and Secret
            response = requests.post(self.FINRA_AUTH_URL, auth=(FINRA_API_CLIENT or "", FINRA_API_SECRET or ""), timeout=30)
            response.raise_for_status()

            token_data = response.json()
            self._access_token = token_data["access_token"]

            # Cache token for 30 minutes (or use expires_in from response)
            expires_in = int(token_data.get("expires_in", 1800))  # Default 30 minutes
            self._token_expires_at = datetime.now(UTC) + timedelta(seconds=expires_in - 60)

            _logger.info("Successfully obtained access token (expires in %s seconds)", expires_in)
            return self._access_token

        except requests.RequestException as e:
            _logger.error("Failed to obtain access token: %s", str(e))
            raise

    def download_trf_data(self) -> pd.DataFrame:
        """
        Download TRF data from FINRA API.

        Returns:
            DataFrame with TRF data containing columns:
                - date: Trading date
                - ticker: Stock symbol
                - short_volume: Short sale volume
                - short_exempt_volume: Short exempt volume
                - total_volume: Total volume
                - short_ratio: Ratio of short volume to total volume
                - market_code: Market identifier
                - facility_code: Reporting facility code
        """
        date_str = self.date.strftime("%Y-%m-%d")
        _logger.info("Downloading TRF data for %s", date_str)

        try:
            # Get OAuth access token
            access_token = self._get_access_token()

            # Prepare the request headers with Bearer token
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
            }

            # Filter for the specific trade date
            filters = {
                "compareFilters": [{"compareType": "EQUAL", "fieldName": "tradeReportDate", "fieldValue": date_str}],
                "limit": 10000,  # Adjust based on expected data volume
            }

            # Make POST request to FINRA API
            response = requests.post(self.finra_url, headers=headers, json=filters, timeout=30)

            # Log response details for debugging
            _logger.debug("Response status: %s", response.status_code)
            if response.status_code >= 400:
                _logger.error("Response body: %s", response.text)

            # 204 No Content = no data for this date (holiday / non-trading day)
            if response.status_code == 204:
                _logger.info("No TRF data for %s (FINRA returned 204 No Content — likely holiday)", date_str)
                return pd.DataFrame()

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
            if not data:
                _logger.warning("No data returned from FINRA API for %s", date_str)
                return pd.DataFrame()

            df = pd.DataFrame(data)

            if df.empty:
                _logger.warning("No TRF data found for %s", date_str)
                return df

            # Standardize column names based on actual API response
            df = df.rename(
                columns={
                    "securitiesInformationProcessorSymbolIdentifier": "ticker",
                    "tradeReportDate": "date",
                    "shortParQuantity": "short_volume",
                    "shortExemptParQuantity": "short_exempt_volume",
                    "totalParQuantity": "total_volume",
                    "marketCode": "market_code",
                    "reportingFacilityCode": "facility_code",
                }
            )

            # Convert date column
            df["date"] = pd.to_datetime(df["date"]).dt.date

            # Calculate short ratio
            df["short_ratio"] = df["short_volume"] / df["total_volume"]

            # Keep only relevant columns
            keep_columns = [
                "date",
                "ticker",
                "short_volume",
                "short_exempt_volume",
                "total_volume",
                "short_ratio",
                "market_code",
                "facility_code",
            ]
            df = df[[col for col in keep_columns if col in df.columns]]

            return df

        except requests.RequestException as e:
            _logger.error("Failed to download TRF data: %s", str(e))
            raise

    def get_volume_data(self, tickers: List[str], batch_size: int = 100) -> pd.DataFrame:
        """
        Fetch volume data from yfinance for the given tickers.

        Args:
            tickers: List of ticker symbols to fetch.
            batch_size: Number of tickers to process in each batch (default: 100).

        Returns:
            DataFrame with volume and OHLC data from yfinance.
        """
        if not tickers:
            return pd.DataFrame()

        _logger.info("Fetching volume data for %d tickers in batches of %d", len(tickers), batch_size)

        volume_data = []
        failed_tickers = []

        # Process tickers in batches to balance speed and error handling
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(tickers) + batch_size - 1) // batch_size

            _logger.debug("Processing batch %d/%d (%d tickers)", batch_num, total_batches, len(batch))

            try:
                # Download data for batch with error suppression
                # Use threads=False to avoid thread-related errors with delisted tickers
                data = yf.download(
                    batch,
                    start=self.date - timedelta(days=1),
                    end=self.date + timedelta(days=1),
                    group_by="ticker",
                    progress=False,
                    threads=False,  # Disable threading to avoid error propagation
                )

                if data.empty:
                    _logger.debug("No data returned for batch %d", batch_num)
                    failed_tickers.extend(batch)
                    continue

                # Process each ticker in the batch
                for ticker in batch:
                    try:
                        # Handle single ticker vs multi-ticker data structure
                        if len(batch) == 1:
                            ticker_data = data
                        else:
                            # Check if ticker exists in multi-ticker response
                            if ticker not in data.columns.get_level_values(0):
                                _logger.debug("Ticker %s not in response (possibly delisted)", ticker)
                                failed_tickers.append(ticker)
                                continue
                            ticker_data = data[ticker]

                        # Check if ticker_data is valid
                        if ticker_data.empty:
                            _logger.debug("No data for ticker %s", ticker)
                            failed_tickers.append(ticker)
                            continue

                        # Get the row for our target date
                        target_date = self.date.strftime("%Y-%m-%d")
                        if target_date in ticker_data.index:
                            row = ticker_data.loc[target_date]

                            # Check if Volume is valid (not NaN and > 0)
                            volume = row.get("Volume") if isinstance(row, pd.Series) else None
                            if volume is not None and pd.notna(volume) and volume > 0:
                                volume_data.append(
                                    {
                                        "ticker": ticker,
                                        "date": self.date.date(),
                                        "volume": volume,
                                        "open": row.get("Open"),
                                        "high": row.get("High"),
                                        "low": row.get("Low"),
                                        "close": row.get("Close"),
                                        "adj_close": row.get("Adj Close", row.get("Close")),
                                    }
                                )
                            else:
                                _logger.debug("No volume data for ticker %s on %s", ticker, target_date)
                                failed_tickers.append(ticker)
                        else:
                            _logger.debug("Date %s not found for ticker %s", target_date, ticker)
                            failed_tickers.append(ticker)

                    except Exception as e:
                        _logger.debug("Failed to process ticker %s: %s", ticker, str(e))
                        failed_tickers.append(ticker)

            except Exception as e:
                _logger.warning("Batch %d failed: %s (marking %d tickers as failed)", batch_num, str(e), len(batch))
                failed_tickers.extend(batch)

        # Log summary
        if volume_data:
            _logger.info("Successfully fetched volume data for %d/%d tickers", len(volume_data), len(tickers))
        else:
            _logger.warning("No volume data fetched for any tickers")

        if failed_tickers:
            _logger.info("Failed to fetch %d tickers (delisted or no data)", len(failed_tickers))
            if len(failed_tickers) <= 50:
                _logger.debug("Failed tickers: %s", ", ".join(failed_tickers))
            else:
                _logger.debug("Failed tickers (first 50): %s", ", ".join(failed_tickers[:50]))

        return pd.DataFrame(volume_data)

    def run(self) -> pd.DataFrame:
        """
        Run the TRF and volume data download process.

        Returns:
            DataFrame with merged TRF and volume data.
        """
        _logger.info("Starting TRF data download for %s", self.date.strftime("%Y-%m-%d"))

        try:
            # Step 1: Download TRF data
            trf_df = self.download_trf_data()

            # If no data and it's a weekend/holiday, just return
            if trf_df.empty:
                _logger.info("No data available (market closed or no data)")
                return trf_df

            # Get list of unique tickers
            tickers = trf_df["ticker"].unique().tolist()
            _logger.info("Found %d unique tickers in TRF data", len(tickers))

            # Step 2: Get volume data for these tickers (if requested)
            if self.fetch_yfinance_data and tickers:
                volume_df = self.get_volume_data(tickers)

                # Step 3: Merge TRF and volume data
                if not volume_df.empty:
                    # Merge on ticker and date
                    result_df = pd.merge(
                        trf_df, volume_df, on=["ticker", "date"], how="left", suffixes=("_finra", "_yf")
                    )

                    # Calculate ratio of FINRA total volume to yfinance volume (should be ~1.0)
                    result_df["volume_ratio"] = result_df["total_volume"] / result_df["volume"]
                else:
                    result_df = trf_df
                    _logger.warning("No volume data available for any tickers")
            else:
                result_df = trf_df
                if not self.fetch_yfinance_data:
                    _logger.info("Skipping yfinance data fetch (fetch_yfinance_data=False)")

            # Save to per-day CSV.gz: DATA_CACHE_DIR/trf/YYYY-MM-DD.csv.gz
            save_df = result_df.copy()
            save_df["date"] = pd.to_datetime(save_df["date"])
            date_str = self.date.strftime("%Y-%m-%d")
            out_path = self._cache_dir / f"{date_str}.csv.gz"
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            save_df.set_index("date").to_csv(out_path, compression="gzip")
            _logger.info("Saved TRF data to %s", out_path)

            return result_df

        except Exception as e:
            _logger.error("Error in TRF download process: %s", str(e), exc_info=True)
            raise


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Download FINRA TRF and volume data")
    parser.add_argument("--date", default=None, type=str, help="Date in YYYY-MM-DD format (default: yesterday)")
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Output directory path (default: results/finra_trf/<date>)",
    )
    parser.add_argument("--output-filename", type=str, default="trf.csv", help="Output filename (default: trf.csv)")
    parser.add_argument("--no-yfinance", action="store_true", help="Skip fetching yfinance volume data")

    args = parser.parse_args()

    try:
        downloader = FinraTRFDownloader(
            date=args.date,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            output_filename=args.output_filename,
            fetch_yfinance_data=not args.no_yfinance,
        )
        downloader.run()
    except Exception as e:
        _logger.critical("Fatal error in TRF downloader: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
