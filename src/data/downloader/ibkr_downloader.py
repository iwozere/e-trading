#!/usr/bin/env python3
"""
IBKR Downloader with Integrated Caching
---------------------------------------
Handles fetching OHLCV data from Interactive Brokers with internal
persistence to DATA_CACHE_DIR. Designed for high efficiency and
delayed market data support.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from ib_insync import IB, Stock, Contract, Forex

from src.data.downloader.base_data_downloader import BaseDataDownloader
from config.donotshare.donotshare import DATA_CACHE_DIR, IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class IBKRDownloader(BaseDataDownloader):
    """
    Downloader for IBKR that implements the 'Data Healer' pattern:
    Checks local CSV cache -> Identifies missing gaps -> Fetches from IBKR -> Merges -> Returns.
    """

    def __init__(self, ib_instance: Optional[IB] = None):
        """
        Initialize the downloader.

        Args:
            ib_instance: Optional existing IB connection. If None, will create its own.
        """
        self.ib = ib_instance
        self.cache_dir = DATA_CACHE_DIR
        self.market_data_type = 3  # Delayed data

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            _logger.info("Created cache directory: %s", self.cache_dir)

    def _ensure_connected(self):
        """Ensure connection to IBKR TWS/Gateway."""
        if self.ib is None or not self.ib.isConnected():
            try:
                self.ib = IB()
                # Use a specific clientId offset to avoid conflicts with main brokers
                self.ib.connect(IBKR_HOST, IBKR_PORT, clientId=int(IBKR_CLIENT_ID) + 50)
                self.ib.reqMarketDataType(self.market_data_type)
                _logger.info("Connected to IBKR for downloading (Delayed Data Mode)")
            except Exception as e:
                _logger.error("Failed to connect to IBKR: %s", e)
                raise

    def get_supported_intervals(self) -> List[str]:
        return ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

    def _get_cache_path(self, symbol: str, interval: str) -> str:
        """Construct the file path for the cached CSV."""
        return os.path.join(self.cache_dir, f"{symbol.upper()}_{interval}.csv")

    def _map_interval(self, interval: str) -> str:
        """Map project interval to IBKR bar size."""
        mapping = {
            '1m': '1 min',
            '5m': '5 mins',
            '15m': '15 mins',
            '30m': '30 mins',
            '1h': '1 hour',
            '4h': '4 hours',
            '1d': '1 day'
        }
        return mapping.get(interval, '1 min')

    def _get_contract(self, symbol: str) -> Contract:
        """Simple contract creation. Defaults to Stock then Forex."""
        # For simplicity in screener, we assume STK/SMART/USD.
        # In a real scenario, this would be more robust.
        if len(symbol) == 6 and any(x in symbol for x in ['USD', 'EUR', 'GBP', 'JPY']):
            return Forex(symbol)
        return Stock(symbol.upper(), 'SMART', 'USD')

    def get_ohlcv(self, symbol: str, interval: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """
        Orchestrates the intelligent sync flow.
        """
        cache_path = self._get_cache_path(symbol, interval)
        df_cached = pd.DataFrame()

        # 1. Load cached data
        if os.path.exists(cache_path):
            try:
                df_cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                # Ensure index is datetime and sorted
                df_cached.index = pd.to_datetime(df_cached.index)
                df_cached.sort_index(inplace=True)
            except Exception as e:
                _logger.warning("Failed to load cache for %s: %s. Starting fresh.", symbol, e)

        # 2. Check for missing data
        now = datetime.now()
        sync_required = False

        if df_cached.empty:
            sync_required = True
            sync_start = start_date
        else:
            last_ts = df_cached.index[-1]
            if last_ts < now - timedelta(minutes=self._get_interval_minutes(interval) * 2):
                sync_required = True
                sync_start = last_ts
            else:
                _logger.debug("Cache for %s is up to date.", symbol)

        # 3. Fetch from IBKR if needed
        if sync_required:
            self._ensure_connected()
            df_new = self._fetch_from_ibkr(symbol, interval, sync_start, now)

            if not df_new.empty:
                # Merge and update cache
                if df_cached.empty:
                    df_combined = df_new
                else:
                    df_combined = pd.concat([df_cached, df_new])
                    # Remove duplicates (e.g., overlapping bars)
                    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]

                df_combined.sort_index(inplace=True)
                df_combined.to_csv(cache_path)
                df_cached = df_combined
                _logger.info("Updated cache for %s (%d new bars)", symbol, len(df_new))

        # 4. Return the requested period
        mask = (df_cached.index >= start_date) & (df_cached.index <= end_date)
        return df_cached.loc[mask]

    def _get_interval_minutes(self, interval: str) -> int:
        mult = {'m': 1, 'h': 60, 'd': 1440}
        return int(interval[:-1]) * mult[interval[-1]]

    def _fetch_from_ibkr(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Internal fetch from IBKR using ib_insync."""
        contract = self._get_contract(symbol)
        ib_interval = self._map_interval(interval)

        # Calculate duration
        delta = end - start
        if delta.days > 365:
            duration = f"{delta.days // 365 + 1} Y"
        elif delta.days > 30:
            duration = f"{delta.days // 30 + 1} M"
        else:
            duration = f"{delta.days + 1} D"

        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='', # most recent
                durationStr=duration,
                barSizeSetting=ib_interval,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1,
                keepUpToDate=False
            )

            if not bars:
                _logger.warning("No data returned for %s from IBKR", symbol)
                return pd.DataFrame()

            df = pd.DataFrame([{
                'timestamp': b.date,
                'open': b.open,
                'high': b.high,
                'low': b.low,
                'close': b.close,
                'volume': b.volume
            } for b in bars])

            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index)
            return df

        except Exception as e:
            _logger.error("Error fetching data for %s: %s", symbol, e)
            return pd.DataFrame()

    # Stub implementations for other abstract methods
    def get_fundamentals(self, symbol: str) -> Dict: return {}
    def get_periods(self) -> List[str]: return ['1 D', '1 W', '1 M', '1 Y']
    def get_intervals(self) -> List[str]: return self.get_supported_intervals()
