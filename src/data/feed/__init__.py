"""
Live Data Feeds Module
---------------------

This module contains all live data feed implementations for real-time market data.
All feeds inherit from BaseLiveDataFeed and provide real-time data streaming
capabilities for various financial data providers.

Available Live Feeds:
- BinanceLiveDataFeed: For real-time cryptocurrency data
- YahooLiveDataFeed: For real-time stock data via Yahoo Finance
- IBKRLiveDataFeed: For Interactive Brokers data
- CoinGeckoLiveDataFeed: For cryptocurrency data via CoinGecko

Base Class:
- BaseLiveDataFeed: Abstract base class defining the live feed interface
"""

from .base_live_data_feed import BaseLiveDataFeed
from .binance_live_feed import BinanceLiveDataFeed
from .yahoo_live_feed import YahooLiveDataFeed
from .ibkr_live_feed import IBKRLiveDataFeed
from .coingecko_live_feed import CoinGeckoLiveDataFeed

__all__ = [
    'BaseLiveDataFeed',
    'BinanceLiveDataFeed',
    'YahooLiveDataFeed',
    'IBKRLiveDataFeed',
    'CoinGeckoLiveDataFeed',
]
