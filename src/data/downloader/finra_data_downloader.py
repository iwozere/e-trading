#!/usr/bin/env python3
"""
FINRA Data Downloader

This module provides data downloading capabilities from FINRA (Financial Industry Regulatory Authority).
Consolidates both FINRA short interest data and TRF (Trade Reporting Facility) data downloaders.

FINRA provides:
- Official short interest data (updated bi-weekly)
- TRF (Trade Reporting Facility) daily short sale volume data

FINRA Short Interest Data: https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, date, timezone
from typing import List, Optional, Dict, Any, Union

import pandas as pd
import requests
import io
import time
import yfinance as yf

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.notification.logger import setup_logger
from src.data.downloader.base_data_downloader import BaseDataDownloader

_logger = setup_logger(__name__)


class FinraDataDownloader(BaseDataDownloader):
    """
    Consolidated FINRA data downloader for short interest and TRF data.

    This class combines functionality from:
    - FINRADataDownloader: Short interest data (bi-weekly)
    - FinraTRFDownloader: TRF daily short sale volume data

    Inherits from BaseDataDownloader for consistency with other downloaders.
    """

    # TRF API constants
    FINRA_API_BASE = "https://api.finra.org/data"
    FINRA_AUTH_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"
    FINRA_GROUP = "otcmarket"
    FINRA_DATASET = "regShoDaily"  # Using daily short sale volume dataset

    # Short interest API constants
    SHORT_INTEREST_BASE_URL = "https://cdn.finra.org/equity/regsho/daily"

    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        date: Optional[str] = None,
        output_dir: Optional[Path] = None,
        output_filename: str = "finra_trf.csv",
        fetch_yfinance_data: bool = True
    ):
        """
        Initialize FINRA data downloader.

        Args:
            rate_limit_delay: Delay between requests in seconds (be respectful to FINRA)
            date: Date string in 'YYYY-MM-DD' format for TRF data. If None, uses yesterday's date.
            output_dir: Directory to save TRF output file. If None, creates output_dir based on date.
            output_filename: Name of the TRF output CSV file.
            fetch_yfinance_data: Whether to fetch and merge yfinance volume data for TRF.
        """
        super().__init__()

        self.rate_limit_delay = rate_limit_delay
        self.base_url = self.SHORT_INTEREST_BASE_URL

        # TRF-specific initialization
        self.date = self._parse_date(date) if date else None
        self.fetch_yfinance_data = fetch_yfinance_data

        # Set output paths for TRF data
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        elif self.date:
            self.output_dir = Path("results") / "finra_trf" / self.date.strftime("%Y-%m-%d")
        else:
            self.output_dir = None

        if self.output_dir:
            self.output_file = self.output_dir / output_filename
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_file = None

        # OAuth token management for TRF API
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

        # Store API credentials for OAuth (loaded via centralized config)
        self._finra_api_client = self._get_config_value('FINRA_API_CLIENT', 'FINRA_API_CLIENT')
        self._finra_api_secret = self._get_config_value('FINRA_API_SECRET', 'FINRA_API_SECRET')

        _logger.info("FinraDataDownloader initialized")

    # ============================================================================
    # BaseDataDownloader abstract method implementations
    # ============================================================================

    def get_supported_intervals(self) -> List[str]:
        """
        Return the list of supported intervals for this data downloader.

        Note: FINRA doesn't provide traditional OHLCV data, so this returns
        an empty list as FINRA data is not interval-based.
        """
        return []  # FINRA doesn't provide interval-based OHLCV data

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Download historical OHLCV data for a given symbol.

        Note: FINRA doesn't provide traditional OHLCV data. This method
        returns an empty DataFrame as FINRA focuses on short interest and TRF data.

        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            interval: Data interval (not used for FINRA)
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            **kwargs: Additional provider-specific parameters

        Returns:
            Empty DataFrame (FINRA doesn't provide OHLCV data)
        """
        _logger.warning(
            "FINRA doesn't provide OHLCV data. Use get_short_interest_data() "
            "or download_trf_data() for FINRA-specific data."
        )
        return pd.DataFrame()

    # ============================================================================
    # Short Interest Data Methods (from FINRADataDownloader)
    # ============================================================================

    def get_available_dates(self, start_date: datetime = None, end_date: datetime = None) -> List[datetime]:
        """
        Get list of available FINRA short interest report dates.

        FINRA publishes twice monthly on:
        - 15th of month (or last business day before if 15th is weekend)
        - Last day of month (or last business day before if last day is weekend)

        Args:
            start_date: Start date for search (default: 1 year ago)
            end_date: End date for search (default: today)

        Returns:
            List of available report dates
        """
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=365)  # 1 year ago
            if not end_date:
                end_date = datetime.now()

            available_dates = []

            # Generate expected FINRA settlement dates with business day logic
            current_date = start_date.replace(day=1)  # Start from beginning of month

            while current_date <= end_date:
                # Mid-month settlement (15th or last business day before)
                mid_month_date = self._get_finra_settlement_date(current_date.year, current_date.month, 15)
                if mid_month_date >= start_date.date() and mid_month_date <= end_date.date():
                    mid_month_datetime = datetime.combine(mid_month_date, datetime.min.time())
                    if self._check_date_availability(mid_month_datetime):
                        available_dates.append(mid_month_datetime)

                # End-of-month settlement (last day or last business day before)
                end_month_date = self._get_finra_settlement_date(current_date.year, current_date.month, -1)
                if end_month_date >= start_date.date() and end_month_date <= end_date.date():
                    end_month_datetime = datetime.combine(end_month_date, datetime.min.time())
                    if self._check_date_availability(end_month_datetime):
                        available_dates.append(end_month_datetime)

                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)

            _logger.info("Found %d available FINRA report dates", len(available_dates))
            return sorted(available_dates)

        except Exception:
            _logger.exception("Error getting available dates:")
            return []

    def _get_finra_settlement_date(self, year: int, month: int, target_day: int) -> date:
        """
        Get FINRA settlement date with business day adjustment.

        Args:
            year: Year
            month: Month
            target_day: Target day (15 for mid-month, -1 for end-of-month)

        Returns:
            Actual settlement date (adjusted to business day if needed)
        """
        if target_day == -1:
            # End of month - get last day
            if month == 12:
                next_month = date(year + 1, 1, 1)
            else:
                next_month = date(year, month + 1, 1)
            target_date = next_month - timedelta(days=1)
        else:
            # Specific day (15th)
            target_date = date(year, month, target_day)

        # Adjust to business day if weekend
        # Monday=0, Tuesday=1, ..., Saturday=5, Sunday=6
        weekday = target_date.weekday()

        if weekday == 5:  # Saturday
            # Move to Friday
            adjusted_date = target_date - timedelta(days=1)
        elif weekday == 6:  # Sunday
            # Move to Friday
            adjusted_date = target_date - timedelta(days=2)
        else:
            # Weekday - no adjustment needed
            adjusted_date = target_date

        return adjusted_date

    def _check_date_availability(self, date: datetime) -> bool:
        """
        Check if FINRA data is available for a specific date.

        Args:
            date: Date to check

        Returns:
            True if data is available, False otherwise
        """
        try:
            # FINRA file naming convention: CNMSshvol20241015.txt
            date_str = date.strftime("%Y%m%d")
            url = f"{self.base_url}/CNMSshvol{date_str}.txt"

            time.sleep(self.rate_limit_delay)
            response = requests.head(url, timeout=10)
            return response.status_code == 200

        except Exception:
            return False

    def get_short_interest_data(self, date: Union[datetime, date] = None) -> Optional[pd.DataFrame]:
        """
        Download FINRA short interest data for a specific date.

        Args:
            date: Date to download (default: most recent available)

        Returns:
            DataFrame with short interest data or None if failed
        """
        try:
            if not date:
                # Find most recent available date
                available_dates = self.get_available_dates()
                if not available_dates:
                    _logger.error("No FINRA data available")
                    return None
                date = available_dates[-1]

            # Handle both datetime and date objects for formatting
            if isinstance(date, datetime):
                date_for_format = date
            else:
                date_for_format = datetime.combine(date, datetime.min.time())

            _logger.info("Downloading FINRA short interest data for %s", date_for_format.strftime("%Y-%m-%d"))

            # FINRA file naming convention
            date_str = date_for_format.strftime("%Y%m%d")
            url = f"{self.base_url}/CNMSshvol{date_str}.txt"

            time.sleep(self.rate_limit_delay)
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse the pipe-delimited file
            df = pd.read_csv(io.StringIO(response.text), sep='|')

            # Clean and standardize column names
            df.columns = df.columns.str.strip()

            # Add metadata
            # Handle both datetime and date objects
            if isinstance(date, datetime):
                report_date = date.date()
            else:
                report_date = date  # Already a date object

            df['report_date'] = report_date
            df['data_source'] = 'FINRA'
            df['downloaded_at'] = datetime.now()

            _logger.info("Downloaded FINRA data: %d records for %s", len(df), date_for_format.strftime("%Y-%m-%d"))
            return df

        except Exception as e:
            _logger.error("Error downloading FINRA data for %s: %s",
                         date_for_format.strftime("%Y-%m-%d") if date else "unknown", e)
            return None

    def get_short_interest_for_symbol(self, symbol: str, date: Union[datetime, date] = None) -> Optional[Dict[str, Any]]:
        """
        Get short interest data for a specific symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            date: Date to get data for (default: most recent)

        Returns:
            Dictionary with short interest data or None if not found
        """
        try:
            df = self.get_short_interest_data(date)
            if df is None or df.empty:
                return None

            # Filter for the specific symbol
            symbol_data = df[df['Symbol'] == symbol.upper()]

            if symbol_data.empty:
                _logger.warning("No FINRA data found for symbol %s", symbol)
                return None

            # Return the most recent record for the symbol
            record = symbol_data.iloc[0].to_dict()

            _logger.debug("Found FINRA data for %s: %s shares short", symbol, record.get('ShortVolume', 'N/A'))
            return record

        except Exception as e:
            _logger.error("Error getting FINRA data for symbol %s: %s", symbol, e)
            return None

    def get_bulk_short_interest(self, symbols: List[str], date: datetime = None) -> Dict[str, Dict[str, Any]]:
        """
        Get short interest data for multiple symbols efficiently.

        Args:
            symbols: List of stock symbols
            date: Date to get data for (default: most recent)

        Returns:
            Dictionary mapping symbols to their short interest data
        """
        try:
            _logger.info("Getting bulk FINRA data for %d symbols", len(symbols))

            # Download the full dataset once
            df = self.get_short_interest_data(date)
            if df is None or df.empty:
                return {}

            # Filter for requested symbols
            symbols_upper = [s.upper() for s in symbols]
            filtered_df = df[df['Symbol'].isin(symbols_upper)]

            # Convert to dictionary
            result = {}
            for _, row in filtered_df.iterrows():
                symbol = row['Symbol']
                result[symbol] = row.to_dict()

            _logger.info("Found FINRA data for %d/%d requested symbols", len(result), len(symbols))
            return result

        except Exception:
            _logger.exception("Error getting bulk FINRA data:")
            return {}

    def calculate_short_interest_metrics(self, finra_data: Dict[str, Any],
                                       shares_outstanding: int) -> Dict[str, float]:
        """
        Calculate short interest metrics from FINRA data.

        Args:
            finra_data: FINRA data dictionary for a symbol
            shares_outstanding: Total shares outstanding

        Returns:
            Dictionary with calculated metrics
        """
        try:
            short_volume = finra_data.get('ShortVolume', 0)
            total_volume = finra_data.get('TotalVolume', 0)

            # Calculate short interest percentage (approximation)
            # Note: FINRA provides daily short volume, not total short interest
            # This is an approximation - real short interest would need accumulation over time
            short_volume_ratio = short_volume / max(total_volume, 1)

            # Estimate short interest as percentage of float
            # This is a rough approximation
            estimated_short_interest_pct = short_volume_ratio * 0.1  # Conservative estimate

            # Calculate days to cover (would need average volume data)
            # This would be calculated elsewhere with volume data

            metrics = {
                'short_volume': short_volume,
                'total_volume': total_volume,
                'short_volume_ratio': short_volume_ratio,
                'estimated_short_interest_pct': estimated_short_interest_pct,
                'report_date': finra_data.get('report_date'),
                'data_source': 'FINRA'
            }

            return metrics

        except Exception:
            _logger.exception("Error calculating short interest metrics:")
            return {}

    def get_historical_short_data(self, symbol: str, days_back: int = 30) -> Optional[pd.DataFrame]:
        """
        Get historical short interest data for a symbol.

        Args:
            symbol: Stock symbol
            days_back: Number of days to look back

        Returns:
            DataFrame with historical data or None if failed
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            available_dates = self.get_available_dates(start_date, end_date)

            if not available_dates:
                return None

            historical_data = []

            for date in available_dates:
                symbol_data = self.get_short_interest_for_symbol(symbol, date)
                if symbol_data:
                    historical_data.append(symbol_data)

            if not historical_data:
                return None

            df = pd.DataFrame(historical_data)
            df['report_date'] = pd.to_datetime(df['report_date'])
            df = df.sort_values('report_date')

            _logger.info("Retrieved %d historical records for %s", len(df), symbol)
            return df

        except Exception as e:
            _logger.error("Error getting historical data for %s: %s", symbol, e)
            return None

    def test_connection(self) -> bool:
        """
        Test connection to FINRA data source.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            _logger.info("Testing FINRA connection...")

            # Try to get available dates
            available_dates = self.get_available_dates()

            if available_dates:
                _logger.info("FINRA connection successful - found %d available report dates", len(available_dates))
                return True
            else:
                _logger.warning("FINRA connection test failed - no available dates found")
                return False

        except Exception:
            _logger.exception("FINRA connection test failed:")
            return False

    # ============================================================================
    # TRF Data Methods (from FinraTRFDownloader)
    # ============================================================================

    @property
    def finra_url(self) -> str:
        """Construct the FINRA API URL."""
        return f"{self.FINRA_API_BASE}/group/{self.FINRA_GROUP}/name/{self.FINRA_DATASET}"

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
            if datetime.now(timezone.utc) < self._token_expires_at:
                _logger.debug("Using cached access token")
                return self._access_token

        _logger.info("Requesting new access token from FINRA")

        try:
            # Use Basic Auth with Client ID and Secret
            response = requests.post(
                self.FINRA_AUTH_URL,
                auth=(self._finra_api_client, self._finra_api_secret),
                timeout=30
            )
            response.raise_for_status()

            token_data = response.json()
            self._access_token = token_data["access_token"]

            # Cache token for 30 minutes (or use expires_in from response)
            expires_in = int(token_data.get("expires_in", 1800))  # Default 30 minutes
            self._token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)

            _logger.info("Successfully obtained access token (expires in %s seconds)", expires_in)
            return self._access_token

        except requests.RequestException as e:
            _logger.error("Failed to obtain access token: %s", str(e))
            raise

    def download_trf_data(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Download TRF data from FINRA API.

        Args:
            date: Date string in 'YYYY-MM-DD' format. If None, uses instance date or yesterday.

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
        # Use provided date or instance date
        trf_date = self._parse_date(date) if date else (self.date if self.date else datetime.now() - timedelta(days=1))
        date_str = trf_date.strftime("%Y-%m-%d")
        _logger.info("Downloading TRF data for %s", date_str)

        try:
            # Get OAuth access token
            access_token = self._get_access_token()

            # Prepare the request headers with Bearer token
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {access_token}'
            }

            # Filter for the specific trade date
            filters = {
                "compareFilters": [
                    {
                        "compareType": "EQUAL",
                        "fieldName": "tradeReportDate",
                        "fieldValue": date_str
                    }
                ],
                "limit": 10000  # Adjust based on expected data volume
            }

            # Make POST request to FINRA API
            response = requests.post(
                self.finra_url,
                headers=headers,
                json=filters,
                timeout=30
            )

            # Log response details for debugging
            _logger.debug("Response status: %s", response.status_code)
            if response.status_code >= 400:
                _logger.error("Response body: %s", response.text)

            # Handle 404 specifically
            if response.status_code == 404:
                if self._is_weekend_or_holiday(trf_date):
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
            df = df.rename(columns={
                "securitiesInformationProcessorSymbolIdentifier": "ticker",
                "tradeReportDate": "date",
                "shortParQuantity": "short_volume",
                "shortExemptParQuantity": "short_exempt_volume",
                "totalParQuantity": "total_volume",
                "marketCode": "market_code",
                "reportingFacilityCode": "facility_code"
            })

            # Convert date column
            df["date"] = pd.to_datetime(df["date"]).dt.date

            # Calculate short ratio
            df["short_ratio"] = df["short_volume"] / df["total_volume"]

            # Keep only relevant columns
            keep_columns = ["date", "ticker", "short_volume", "short_exempt_volume",
                          "total_volume", "short_ratio", "market_code", "facility_code"]
            df = df[[col for col in keep_columns if col in df.columns]]

            return df

        except requests.RequestException as e:
            _logger.error("Failed to download TRF data: %s", str(e))
            raise

    def get_volume_data(self, tickers: List[str], batch_size: int = 100, target_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch volume data from yfinance for the given tickers.

        Args:
            tickers: List of ticker symbols to fetch.
            batch_size: Number of tickers to process in each batch (default: 100).
            target_date: Date to fetch volume for. If None, uses instance date or yesterday.

        Returns:
            DataFrame with volume and OHLC data from yfinance.
        """
        if not tickers:
            return pd.DataFrame()

        # Use provided date or instance date
        trf_date = target_date if target_date else (self.date if self.date else datetime.now() - timedelta(days=1))

        _logger.info("Fetching volume data for %d tickers in batches of %d", len(tickers), batch_size)

        volume_data = []
        failed_tickers = []

        # Process tickers in batches to balance speed and error handling
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(tickers) + batch_size - 1) // batch_size

            _logger.debug("Processing batch %d/%d (%d tickers)", batch_num, total_batches, len(batch))

            try:
                # Download data for batch with error suppression
                # Use threads=False to avoid thread-related errors with delisted tickers
                data = yf.download(
                    batch,
                    start=trf_date - timedelta(days=1),
                    end=trf_date + timedelta(days=1),
                    group_by='ticker',
                    progress=False,
                    threads=False  # Disable threading to avoid error propagation
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
                        target_date_str = trf_date.strftime("%Y-%m-%d")
                        if target_date_str in ticker_data.index:
                            row = ticker_data.loc[target_date_str]

                            # Check if Volume is valid (not NaN and > 0)
                            volume = row.get("Volume") if isinstance(row, pd.Series) else None
                            if pd.notna(volume) and volume > 0:
                                volume_data.append({
                                    "ticker": ticker,
                                    "date": trf_date.date(),
                                    "volume": volume,
                                    "open": row.get("Open"),
                                    "high": row.get("High"),
                                    "low": row.get("Low"),
                                    "close": row.get("Close"),
                                    "adj_close": row.get("Adj Close", row.get("Close"))
                                })
                            else:
                                _logger.debug("No volume data for ticker %s on %s", ticker, target_date_str)
                                failed_tickers.append(ticker)
                        else:
                            _logger.debug("Date %s not found for ticker %s", target_date_str, ticker)
                            failed_tickers.append(ticker)

                    except Exception as e:
                        _logger.debug("Failed to process ticker %s: %s", ticker, str(e))
                        failed_tickers.append(ticker)

            except Exception as e:
                _logger.warning("Batch %d failed: %s (marking %d tickers as failed)",
                              batch_num, str(e), len(batch))
                failed_tickers.extend(batch)

        # Log summary
        if volume_data:
            _logger.info("Successfully fetched volume data for %d/%d tickers",
                        len(volume_data), len(tickers))
        else:
            _logger.warning("No volume data fetched for any tickers")

        if failed_tickers:
            _logger.info("Failed to fetch %d tickers (delisted or no data)", len(failed_tickers))
            if len(failed_tickers) <= 50:
                _logger.debug("Failed tickers: %s", ", ".join(failed_tickers))
            else:
                _logger.debug("Failed tickers (first 50): %s", ", ".join(failed_tickers[:50]))

        return pd.DataFrame(volume_data)

    def run(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Run the TRF and volume data download process.

        Args:
            date: Date string in 'YYYY-MM-DD' format. If None, uses instance date or yesterday.

        Returns:
            DataFrame with merged TRF and volume data.
        """
        # Use provided date or instance date
        trf_date = self._parse_date(date) if date else (self.date if self.date else datetime.now() - timedelta(days=1))
        date_str = trf_date.strftime("%Y-%m-%d")
        _logger.info("Starting TRF data download for %s", date_str)

        try:
            # Step 1: Download TRF data
            trf_df = self.download_trf_data(date)

            # If no data and it's a weekend/holiday, just return
            if trf_df.empty:
                _logger.info("No data available (market closed or no data)")
                return trf_df

            # Get list of unique tickers
            tickers = trf_df["ticker"].unique().tolist()
            _logger.info("Found %d unique tickers in TRF data", len(tickers))

            # Step 2: Get volume data for these tickers (if requested)
            if self.fetch_yfinance_data and tickers:
                volume_df = self.get_volume_data(tickers, target_date=trf_date)

                # Step 3: Merge TRF and volume data
                if not volume_df.empty:
                    # Merge on ticker and date
                    result_df = pd.merge(
                        trf_df,
                        volume_df,
                        on=["ticker", "date"],
                        how="left",
                        suffixes=("_finra", "_yf")
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

            # Save results if output file is configured
            if self.output_file:
                result_df.to_csv(self.output_file, index=False)
                _logger.info("Saved TRF data to %s", self.output_file)

            return result_df

        except Exception as e:
            _logger.error("Error in TRF download process: %s", str(e), exc_info=True)
            raise


# Factory function for backward compatibility
def create_finra_downloader(rate_limit_delay: float = 1.0) -> FinraDataDownloader:
    """
    Factory function to create FINRA downloader.

    Args:
        rate_limit_delay: Delay between requests in seconds

    Returns:
        Configured FINRA downloader instance
    """
    return FinraDataDownloader(rate_limit_delay=rate_limit_delay)


# Example usage
if __name__ == "__main__":
    # Create FINRA downloader
    finra = create_finra_downloader()

    # Test connection
    if finra.test_connection():
        print("✅ FINRA connection successful")

        # Get available dates
        dates = finra.get_available_dates()
        if dates:
            print(f"✅ Found {len(dates)} available report dates")
            print(f"Most recent: {dates[-1].strftime('%Y-%m-%d')}")

            # Test getting data for a specific symbol
            symbol_data = finra.get_short_interest_for_symbol('AAPL')
            if symbol_data:
                print(f"✅ Found FINRA data for AAPL: {symbol_data.get('ShortVolume', 'N/A')} short volume")
            else:
                print("❌ No FINRA data found for AAPL")

            # Test bulk download
            bulk_data = finra.get_bulk_short_interest(['AAPL', 'TSLA', 'GME'])
            print(f"✅ Bulk download: {len(bulk_data)} symbols found")

        else:
            print("❌ No available dates found")
    else:
        print("❌ FINRA connection failed")
