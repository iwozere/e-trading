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
from typing import Optional
import pandas as pd
import requests
from .base_data_downloader import BaseDataDownloader

class CoinGeckoDataDownloader(BaseDataDownloader):
    """Implementation of a data downloader for CoinGecko, fetching historical market data using the CoinGecko API."""

    def __init__(self, data_dir: Optional[str] = None, interval: Optional[str] = None):
        super().__init__(data_dir=data_dir, interval=interval)
        self.base_url = "https://api.coingecko.com/api/v3"

    def download_historical_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        save_to_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Download historical OHLCV data from CoinGecko.

        Args:
            symbol: Trading pair symbol (e.g., 'bitcoin', 'ethereum')
            interval: Kline interval (e.g., '1h', '4h', '1d')
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            save_to_csv: Whether to save the data to a CSV file

        Returns:
            DataFrame containing the historical data
        """
        # CoinGecko expects coin id, not symbol. You may need to map symbol to id externally.
        # For simplicity, we assume symbol is the CoinGecko id (e.g., 'bitcoin', 'ethereum').
        vs_currency = "usd"
        # Convert dates to UNIX timestamps (seconds)
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

        url = f"{self.base_url}/coins/{symbol}/market_chart/range"
        params = {
            "vs_currency": vs_currency,
            "from": start_timestamp,
            "to": end_timestamp,
        }
        response = requests.get(url, params=params)
        data = response.json()

        # CoinGecko returns 'prices', 'market_caps', 'total_volumes'. We'll use 'prices' and 'total_volumes'.
        # 'prices' is a list of [timestamp(ms), price].
        # There is no direct OHLCV, so we approximate OHLCV from price/volume data.
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

    def get_periods(self) -> list:
        return ['1d', '7d', '1w', '1mo', '3mo', '6mo', '1y', '2y']

    def get_intervals(self) -> list:
        return ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo']

    def is_valid_period_interval(self, period, interval) -> bool:
        return interval in self.get_intervals() and period in self.get_periods()