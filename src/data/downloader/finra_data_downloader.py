#!/usr/bin/env python3
"""
FINRA Data Downloader

This module provides data downloading capabilities from FINRA (Financial Industry Regulatory Authority).
FINRA provides official short interest data that is updated bi-weekly.

FINRA Short Interest Data: https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data
"""

import requests
import pandas as pd
import io
import time
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class FINRADataDownloader:
    """FINRA data downloader for official short interest data."""

    def __init__(self, rate_limit_delay: float = 1.0):
        """
        Initialize FINRA data downloader.

        Args:
            rate_limit_delay: Delay between requests in seconds (be respectful to FINRA)
        """

        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://cdn.finra.org/equity/regsho/daily"

        # FINRA publishes short interest data twice per month
        # Settlement dates are typically around the 15th and last day of each month

        _logger.info("FINRA Data Downloader initialized")

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

    def get_short_interest_data(self, date: Union[datetime, datetime.date] = None) -> Optional[pd.DataFrame]:
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

    def get_short_interest_for_symbol(self, symbol: str, date: Union[datetime, datetime.date] = None) -> Optional[Dict[str, Any]]:
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


# Factory function
def create_finra_downloader(rate_limit_delay: float = 1.0) -> FINRADataDownloader:
    """
    Factory function to create FINRA downloader.

    Args:
        rate_limit_delay: Delay between requests in seconds

    Returns:
        Configured FINRA downloader instance
    """
    return FINRADataDownloader(rate_limit_delay)


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