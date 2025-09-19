"""
CoinGecko Data Downloader Module
-------------------------------

This module provides the CoinGeckoDataDownloader class for downloading historical OHLCV (Open, High, Low, Close, Volume) data from the CoinGecko API. It supports fetching data for a single symbol and saving the results as CSV files for use in backtesting and analysis workflows.

Main Features:
- Download historical candlestick data for any CoinGecko trading pair and interval
- Save data to CSV files in a structured format
- Inherits common logic from BaseDataDownloader for file management

Valid values:
- interval: '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo' (resampling is done from CoinGecko's available data)
- period: Any string like '1d', '7d', '1w', '1mo', '3mo', '6mo', '1y', '2y', etc. (used to calculate start_date/end_date)

Classes:
- CoinGeckoDataDownloader: Main class for interacting with the CoinGecko API and managing data downloads
"""

from datetime import datetime
from typing import Optional, List, Dict
import pandas as pd
import requests
from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.notification.logger import setup_logger
from src.model.schemas import OptionalFundamentals

_logger = setup_logger(__name__)

class CoinGeckoDataDownloader(BaseDataDownloader):
    """
    A class to download historical data from CoinGecko.

    This class provides methods to:
    1. Download historical OHLCV data for cryptocurrencies
    2. Save data to CSV files
    3. Load data from CSV files
    4. Update existing data files with new data
    5. Get fundamental data (NotImplementedError - CoinGecko doesn't provide stock fundamentals)

    **Fundamental Data Capabilities:**
    - ❌ PE Ratio (cryptocurrency exchange)
    - ❌ Financial Ratios (cryptocurrency exchange)
    - ❌ Growth Metrics (cryptocurrency exchange)
    - ❌ Company Information (cryptocurrency exchange)
    - ❌ Market Data (only cryptocurrency data)
    - ❌ Profitability Metrics (cryptocurrency exchange)
    - ❌ Valuation Metrics (cryptocurrency exchange)

    **Data Quality:** N/A - CoinGecko is for cryptocurrencies, not stocks
    **Rate Limits:** 50 calls per minute (free tier)
    **Coverage:** Cryptocurrencies only

    Parameters:
    -----------
    data_dir : str
        Directory to store downloaded data files

    Example:
    --------
    >>> from datetime import datetime
    >>> downloader = CoinGeckoDataDownloader()
    >>> df = downloader.get_ohlcv("bitcoin", "1d", datetime(2023, 1, 1), datetime(2023, 12, 31))
    >>> # Get fundamental data (will raise NotImplementedError)
    >>> try:
    >>>     fundamentals = downloader.get_fundamentals("AAPL")
    >>> except NotImplementedError as e:
    >>>     print(f"Not supported: {e}")
    """

    def __init__(self):
        super().__init__()
        self.base_url = "https://api.coingecko.com/api/v3"

    def download_historical_data(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        save_to_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Download historical OHLCV data from CoinGecko.

        Args:
            symbol: Trading pair symbol (e.g., 'bitcoin', 'ethereum')
            interval: Kline interval (e.g., '1h', '4h', '1d')
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime
            save_to_csv: Whether to save the data to a CSV file

        Returns:
            DataFrame containing the historical data
        """
        # CoinGecko expects coin id, not symbol. You may need to map symbol to id externally.
        # For simplicity, we assume symbol is the CoinGecko id (e.g., 'bitcoin', 'ethereum').
        vs_currency = "usd"
        # Convert dates to UNIX timestamps (seconds)
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        url = f"{self.base_url}/coins/{symbol}/market_chart/range"
        params = {
            "vs_currency": vs_currency,
            "from": start_timestamp,
            "to": end_timestamp,
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            _logger.error("CoinGecko API error: %s %s", response.status_code, response.text)
            raise RuntimeError(f"CoinGecko API error: {response.status_code} {response.text}")
        data = response.json()

        # CoinGecko returns 'prices', 'market_caps', 'total_volumes'. We'll use 'prices' and 'total_volumes'.
        # 'prices' is a list of [timestamp(ms), price].
        # There is no direct OHLCV, so we approximate OHLCV from price/volume data.
        try:
            prices = data.get("prices", [])
            volumes = {v[0]: v[1] for v in data.get("total_volumes", [])}

            # Group by day/hour depending on interval
            df = pd.DataFrame(prices, columns=["timestamp", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["volume"] = df["timestamp"].map(lambda ts: volumes.get(int(ts.timestamp() * 1000), 0))

            # Resample to requested interval
            if interval.endswith("d"):
                rule = f'{interval[:-1]}D' if interval != "1d" else "D"
            elif interval.endswith("h"):
                rule = f'{interval[:-1]}H' if interval != "1h" else "H"
            elif interval.endswith("m"):
                rule = f'{interval[:-1]}T' if interval != "1m" else "T"
            else:
                rule = "D"

            ohlcv = df.resample(rule, on="timestamp").agg({
                "close": ["first", "max", "min", "last"],
                "volume": "sum"
            })
            ohlcv.columns = ["open", "high", "low", "close", "volume"]
            ohlcv = ohlcv.reset_index()
            ohlcv = ohlcv.rename(columns={"timestamp": "timestamp"})
            ohlcv = ohlcv[["timestamp", "open", "high", "low", "close", "volume"]]

            # Save to CSV if requested
            if save_to_csv:
                self.save_data(ohlcv, symbol, start_date, end_date)

            return ohlcv
        except Exception as e:
            _logger.exception("Error processing CoinGecko data for %s: %s", symbol, str(e))
            raise

    def get_periods(self) -> list:
        return ['1d', '7d', '1w', '1mo', '3mo', '6mo', '1y', '2y']

    def get_intervals(self) -> list:
        return ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo']

    def is_valid_period_interval(self, period, interval) -> bool:
        return interval in self.get_intervals() and period in self.get_periods()

    def get_ohlcv(self, symbol, interval, start_date: datetime, end_date: datetime, **kwargs):
        save_to_csv = kwargs.get('save_to_csv', False)
        return self.download_historical_data(symbol, interval, start_date, end_date, save_to_csv=save_to_csv)

    def get_fundamentals(self, symbol: str) -> OptionalFundamentals:
        """
        Get fundamental data for a given symbol.

        Note: CoinGecko is for cryptocurrencies, not stocks, so this method raises NotImplementedError.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Fundamentals: Fundamental data for the stock

        Raises:
            NotImplementedError: CoinGecko is for cryptocurrencies, not stocks
        """
        raise NotImplementedError("CoinGecko is for cryptocurrencies, not stocks")

    def download_multiple_symbols(
        self, symbols: List[str], interval: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, str]:
        def download_func(symbol, interval, start_date, end_date):
            return self.download_historical_data(symbol, interval, start_date, end_date, save_to_csv=False)
        return super().download_multiple_symbols(
            symbols, download_func, interval, start_date, end_date
        )

    def get_supported_intervals(self) -> List[str]:
        """Return list of supported intervals for CoinGecko."""
        return ['1d']  # CoinGecko only supports daily data