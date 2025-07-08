import logging
import os
from typing import Dict, List
from datetime import datetime

import pandas as pd
import yfinance as yf
from src.notification.logger import setup_logger
from src.model.telegram_bot import Fundamentals
_logger = setup_logger(__name__)

from .base_data_downloader import BaseDataDownloader

"""
Data downloader implementation for Yahoo Finance, fetching historical market data for analysis and backtesting.

This module provides the YahooDataDownloader class for downloading historical OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance. It supports fetching, saving, loading, and updating data for single or multiple symbols, and is suitable for both research and production trading workflows.

Main Features:
- Download historical data for any stock or ticker from Yahoo Finance
- Save and load data as CSV files
- Update existing data files with new data
- Download data for multiple symbols in batch
- Inherits common logic from BaseDataDownloader for file management

Valid values:
- interval: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
- period: Any string like '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y', etc. (used to calculate start_date/end_date)

Classes:
- YahooDataDownloader: Main class for interacting with Yahoo Finance and managing data downloads
"""


class YahooDataDownloader(BaseDataDownloader):
    """
    A class to download historical data from Yahoo Finance.

    This class provides methods to:
    1. Download historical OHLCV data for a given symbol
    2. Save data to CSV files
    3. Load data from CSV files
    4. Update existing data files with new data
    5. Get comprehensive fundamental data for stocks

    **Fundamental Data Capabilities:**
    - ✅ PE Ratio (trailing and forward)
    - ✅ Financial Ratios (P/B, ROE, ROA, debt/equity, current ratio, quick ratio)
    - ✅ Growth Metrics (revenue growth, net income growth)
    - ✅ Company Information (name, sector, industry, country, exchange)
    - ✅ Market Data (market cap, current price, shares outstanding)
    - ✅ Profitability Metrics (operating margin, profit margin, free cash flow)
    - ✅ Valuation Metrics (beta, PEG ratio, price-to-sales, enterprise value)

    **Data Quality:** High - Yahoo Finance provides comprehensive fundamental data
    **Rate Limits:** None for basic usage
    **Coverage:** Global stocks and ETFs

    Parameters:
    -----------
    data_dir : str
        Directory to store downloaded data files

    Example:
    --------
    >>> from datetime import datetime
    >>> downloader = YahooDataDownloader()
    >>> df = downloader.get_ohlcv("AAPL", "1d", datetime(2023, 1, 1), datetime(2023, 12, 31))
    >>> # Get fundamental data
    >>> fundamentals = downloader.get_fundamentals("AAPL")
    >>> print(f"PE Ratio: {fundamentals.pe_ratio}")
    >>> print(f"Market Cap: ${fundamentals.market_cap:,.0f}")
    """

    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir=data_dir)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def get_ohlcv(
        self, symbol: str, interval: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Download historical data for a given symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime

        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            df = ticker.history(start=start_str, end=end_str, interval=interval)

            # Rename columns to match standard format
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            # Add timestamp column
            df["timestamp"] = df.index

            # Reset index to make timestamp a regular column
            df = df.reset_index(drop=True)

            # Ensure all required columns are present
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            return df

        except Exception as e:
            _logger.error("Error downloading data for %s: %s", symbol, e, exc_info=True)
            raise

    def save_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> str:
        """
        Save downloaded data to a CSV file.

        Args:
            df: DataFrame containing historical data
            symbol: Stock symbol
            interval: Data interval
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime

        Returns:
            str: Path to the saved file
        """
        try:
            return super().save_data(df, symbol, interval, start_date, end_date)

        except Exception as e:
            _logger.error("Error saving data for %s: %s", symbol, e, exc_info=True)
            raise

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            filepath: Path to the CSV file

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            return super().load_data(filepath)

        except Exception as e:
            _logger.error("Error loading data from %s: %s", filepath, e, exc_info=True)
            raise

    def update_data(self, symbol: str, interval: str) -> str:
        """
        Update existing data file with new data.

        Args:
            symbol: Stock symbol
            interval: Data interval

        Returns:
            str: Path to the updated file
        """
        try:
            # Find existing data file
            existing_files = [
                f
                for f in os.listdir(self.data_dir)
                if f.startswith(f"{symbol}_{interval}_")
            ]
            if not existing_files:
                # If no existing file, download new data
                df = self.get_ohlcv(
                    symbol,
                    interval,
                    datetime.fromtimestamp(0),
                    datetime.now(),
                )
                return self.save_data(df, symbol, interval)

            # Load existing data
            latest_file = max(existing_files)
            filepath = os.path.join(self.data_dir, latest_file)
            existing_df = self.load_data(filepath)

            # Get last date in existing data
            last_date = existing_df["timestamp"].max()

            # Download new data from last date
            new_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            new_end = pd.Timestamp.today().strftime("%Y-%m-%d")
            new_df = self.get_ohlcv(symbol, interval, datetime.fromtimestamp(int(new_start)), datetime.fromtimestamp(int(new_end)))

            if new_df.empty:
                _logger.info("No new data available for %s", symbol)
                return filepath

            # Combine existing and new data
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df.drop_duplicates(subset=["timestamp"])
            combined_df = combined_df.sort_values("timestamp")

            # Save updated data
            return self.save_data(combined_df, symbol, interval)

        except Exception as e:
            _logger.error("Error updating data for %s: %s", symbol, e, exc_info=True)
            raise

    def download_multiple_symbols(
        self, symbols: List[str], interval: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, str]:
        """
        Download data for multiple symbols.

        Args:
            symbols: List of stock symbols
            interval: Data interval
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime

        Returns:
            Dict[str, str]: Dictionary mapping symbols to file paths
        """
        def download_func(symbol, interval, start_date, end_date):
            return self.get_ohlcv(symbol, interval, start_date, end_date)
        return super().download_multiple_symbols(
            symbols, download_func, interval, start_date, end_date
        )

    def get_periods(self) -> list:
        return ['1d', '7d', '1mo', '3mo', '6mo', '1y', '2y']

    def get_intervals(self) -> list:
        return ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

    def is_valid_period_interval(self, period, interval) -> bool:
        return interval in self.get_intervals() and period in self.get_periods()

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        """
        Get comprehensive fundamental data for a given stock using Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Fundamentals: Comprehensive fundamental data for the stock
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                _logger.error("No data returned from yfinance for ticker %s", symbol)
                return Fundamentals(
                    ticker=symbol.upper(),
                    company_name="Unknown",
                    current_price=0.0,
                    market_cap=0.0,
                    pe_ratio=0.0,
                    forward_pe=0.0,
                    dividend_yield=0.0,
                    earnings_per_share=0.0,
                    data_source="Yahoo Finance",
                    last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )

            _logger.debug("Retrieved fundamentals for %s: %s", symbol, info.get('shortName', 'Unknown'))

            return Fundamentals(
                ticker=symbol.upper(),
                company_name=info.get("longName", "Unknown"),
                current_price=info.get("regularMarketPrice", 0.0),
                market_cap=info.get("marketCap", 0.0),
                pe_ratio=info.get("trailingPE", 0.0),
                forward_pe=info.get("forwardPE", 0.0),
                dividend_yield=info.get("dividendYield", 0.0),
                earnings_per_share=info.get("trailingEps", 0.0),
                # Additional fields
                price_to_book=info.get("priceToBook", None),
                return_on_equity=info.get("returnOnEquity", None),
                return_on_assets=info.get("returnOnAssets", None),
                debt_to_equity=info.get("debtToEquity", None),
                current_ratio=info.get("currentRatio", None),
                quick_ratio=info.get("quickRatio", None),
                revenue=info.get("totalRevenue", None),
                revenue_growth=info.get("revenueGrowth", None),
                net_income=info.get("netIncomeToCommon", None),
                net_income_growth=info.get("netIncomeGrowth", None),
                free_cash_flow=info.get("freeCashflow", None),
                operating_margin=info.get("operatingMargins", None),
                profit_margin=info.get("profitMargins", None),
                beta=info.get("beta", None),
                sector=info.get("sector", None),
                industry=info.get("industry", None),
                country=info.get("country", None),
                exchange=info.get("exchange", None),
                currency=info.get("currency", None),
                shares_outstanding=info.get("sharesOutstanding", None),
                float_shares=info.get("floatShares", None),
                short_ratio=info.get("shortRatio", None),
                payout_ratio=info.get("payoutRatio", None),
                peg_ratio=info.get("pegRatio", None),
                price_to_sales=info.get("priceToSalesTrailing12Months", None),
                enterprise_value=info.get("enterpriseValue", None),
                enterprise_value_to_ebitda=info.get("enterpriseToEbitda", None),
                data_source="Yahoo Finance",
                last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

        except Exception as e:
            _logger.error("Failed to get fundamentals for %s: %s", symbol, e, exc_info=True)
            return Fundamentals(
                ticker=symbol.upper(),
                company_name="Unknown",
                current_price=0.0,
                market_cap=0.0,
                pe_ratio=0.0,
                forward_pe=0.0,
                dividend_yield=0.0,
                earnings_per_share=0.0,
                data_source="Yahoo Finance",
                last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
