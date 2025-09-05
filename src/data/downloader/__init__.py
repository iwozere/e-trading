"""
Data Downloaders Module
-----------------------

This module contains all data downloader implementations for various financial data providers.
All downloaders inherit from BaseDataDownloader and provide a unified interface for
fetching historical OHLCV data from different sources.

Available Downloaders:
- BinanceDataDownloader: For cryptocurrency data
- YahooDataDownloader: For stock data via Yahoo Finance
- AlphaVantageDataDownloader: For comprehensive financial data
- FMPDataDownloader: For Financial Modeling Prep data
- TiingoDataDownloader: For Tiingo financial data
- PolygonDataDownloader: For Polygon.io data
- TwelveDataDownloader: For Twelve Data API
- FinnhubDataDownloader: For Finnhub data
- CoinGeckoDataDownloader: For cryptocurrency data via CoinGecko

Base Class:
- BaseDataDownloader: Abstract base class defining the downloader interface
"""

from .base_data_downloader import BaseDataDownloader
from .binance_data_downloader import BinanceDataDownloader
from .yahoo_data_downloader import YahooDataDownloader
from .alpha_vantage_data_downloader import AlphaVantageDataDownloader
from .fmp_data_downloader import FMPDataDownloader
from .tiingo_data_downloader import TiingoDataDownloader
from .polygon_data_downloader import PolygonDataDownloader
from .twelvedata_data_downloader import TwelveDataDownloader
from .finnhub_data_downloader import FinnhubDataDownloader
from .coingecko_data_downloader import CoinGeckoDataDownloader

__all__ = [
    'BaseDataDownloader',
    'BinanceDataDownloader',
    'YahooDataDownloader',
    'AlphaVantageDataDownloader',
    'FMPDataDownloader',
    'TiingoDataDownloader',
    'PolygonDataDownloader',
    'TwelveDataDownloader',
    'FinnhubDataDownloader',
    'CoinGeckoDataDownloader',
]
