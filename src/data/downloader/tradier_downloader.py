"""
Tradier Options Data Downloader

This module provides functionality to fetch and process options data from the Tradier API.
Now refactored to inherit from BaseDataDownloader for consistency with other downloaders.
"""

import os
import time
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

import pandas as pd

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TradierDataDownloader(BaseDataDownloader):
    """
    Tradier Options Data Downloader.

    Inherits from BaseDataDownloader for consistency with other downloaders.
    Tradier specializes in options data, not traditional OHLCV data.
    """

    BASE_URL = "https://api.tradier.com/v1"

    def __init__(self, api_key: Optional[str] = None, rate_limit_sleep: float = 0.3):
        """
        Initialize Tradier data downloader.

        Args:
            api_key: Tradier API key. If None, uses TRADIER_API from environment.
            rate_limit_sleep: Delay between requests in seconds (default: 0.3)
        """
        super().__init__()
        self.api_key = api_key or os.getenv("TRADIER_API")
        if not self.api_key:
            _logger.warning("Tradier API key not provided. Some operations may fail.")
        self.rate_limit_sleep = rate_limit_sleep

        self.session = requests.Session()
        if self.api_key:
            self.session.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            }

    def get_supported_intervals(self) -> List[str]:
        """
        Return the list of supported intervals for this data downloader.

        Note: Tradier specializes in options data, not interval-based OHLCV data.
        """
        return []  # Tradier doesn't provide interval-based OHLCV data

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

        Note: Tradier doesn't provide traditional OHLCV data. This method
        returns an empty DataFrame as Tradier focuses on options data.

        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            interval: Data interval (not used for Tradier)
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            **kwargs: Additional provider-specific parameters

        Returns:
            Empty DataFrame (Tradier doesn't provide OHLCV data)
        """
        _logger.warning(
            "Tradier doesn't provide OHLCV data. Use get_expirations() or "
            "get_chain() for Tradier-specific options data."
        )
        return pd.DataFrame()

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        """
        Perform safe GET requests with rate limiting.

        Args:
            endpoint: API endpoint (e.g., "/markets/options/expirations")
            params: Optional query parameters

        Returns:
            JSON response as dict, or None if request fails
        """
        url = f"{self.BASE_URL}{endpoint}"
        try:
            r = self.session.get(url, params=params, timeout=10)
            if r.status_code == 429:
                _logger.warning("Rate limit hit. Sleeping...")
                time.sleep(2)
                return self._get(endpoint, params)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            _logger.error(f"Error fetching {url}: {e}")
            return None

    def get_expirations(self, ticker: str) -> List[str]:
        """
        Get available option expirations for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            List of expiration date strings
        """
        data = self._get(f"/markets/options/expirations", {"symbol": ticker})
        if not data or "expirations" not in data or "date" not in data["expirations"]:
            return []
        return data["expirations"]["date"]

    def get_chain(self, ticker: str, expiration: str) -> List[dict]:
        """
        Download entire option chain for a specific expiration.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            expiration: Expiration date string (e.g., '2024-01-19')

        Returns:
            List of option contract dictionaries
        """
        data = self._get(f"/markets/options/chains", {"symbol": ticker, "expiration": expiration})
        if not data or "options" not in data or "option" not in data["options"]:
            return []
        return data["options"]["option"]

    def _save(self, path: str, data: dict) -> None:
        """
        Save JSON data to disk.

        Args:
            path: File path to save to
            data: Data dictionary to save
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def download_ticker(self, ticker: str, out_dir: str) -> None:
        """
        Main download routine for a ticker.

        Downloads all available option chains for a ticker and saves them to disk.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            out_dir: Output directory path
        """
        _logger.info(f"Processing {ticker}...")

        expirations = self.get_expirations(ticker)
        if not expirations:
            _logger.warning(f"No expirations for {ticker}")
            return

        for exp in expirations:
            chain = self.get_chain(ticker, exp)
            if not chain:
                continue

            path = os.path.join(out_dir, ticker.upper(), f"options_{exp}.json")
            self._save(path, chain)
            _logger.info(f"Saved {ticker} {exp} ({len(chain)} contracts)")

            time.sleep(self.rate_limit_sleep)

    def download_universe(self, tickers: List[str], out_dir: str) -> None:
        """
        Batch download for multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            out_dir: Output directory path
        """
        for t in tickers:
            try:
                self.download_ticker(t, out_dir)
            except Exception as e:
                _logger.error(f"Failed {t}: {e}")
            time.sleep(self.rate_limit_sleep)


# Example usage
if __name__ == "__main__":
    API_KEY = os.environ.get("TRADIER_API", "YOUR_KEY_HERE")

    downloader = TradierDataDownloader(API_KEY)

    # Example small universe (you will pass your filtered list)
    universe = ["AAPL", "AMD", "TSLA"]

    downloader.download_universe(universe, out_dir="./data/tradier/")
