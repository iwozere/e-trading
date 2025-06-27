"""
Data Feed Factory Module
-----------------------

This module provides a factory for creating live data feeds based on configuration.
It helps determine which data feed implementation to use based on the data source.

Classes:
- DataFeedFactory: Factory for creating live data feeds
"""

from typing import Dict, Any, Optional
from src.data.base_live_data_feed import BaseLiveDataFeed
from src.data.binance_live_feed import BinanceLiveDataFeed
from src.data.yahoo_live_feed import YahooLiveDataFeed
from src.data.ibkr_live_feed import IBKRLiveDataFeed
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class DataFeedFactory:
    """
    Factory for creating live data feeds based on configuration.
    
    This factory helps determine which data feed implementation to use
    based on the data source and configuration parameters.
    """
    
    @staticmethod
    def create_data_feed(config: Dict[str, Any]) -> Optional[BaseLiveDataFeed]:
        """
        Create a live data feed based on configuration.
        
        Args:
            config: Configuration dictionary with data feed parameters
            
        Returns:
            Live data feed instance, or None if creation fails
            
        Configuration format:
        {
            "data_source": "binance|yahoo|ibkr",
            "symbol": "BTCUSDT",
            "interval": "1m",
            "lookback_bars": 1000,
            "retry_interval": 60,
            "on_new_bar": callback_function,
            
            # Binance specific
            "api_key": "your_api_key",
            "api_secret": "your_api_secret", 
            "testnet": false,
            
            # Yahoo specific
            "polling_interval": 60,
            
            # IBKR specific
            "host": "127.0.0.1",
            "port": 7497,
            "client_id": 1
        }
        """
        try:
            data_source = config.get("data_source", "").lower()
            
            if data_source == "binance":
                return DataFeedFactory._create_binance_feed(config)
            elif data_source == "yahoo":
                return DataFeedFactory._create_yahoo_feed(config)
            elif data_source == "ibkr":
                return DataFeedFactory._create_ibkr_feed(config)
            else:
                _logger.error(f"Unknown data source: {data_source}")
                return None
                
        except Exception as e:
            _logger.error(f"Error creating data feed: {str(e)}")
            return None
    
    @staticmethod
    def _create_binance_feed(config: Dict[str, Any]) -> BinanceLiveDataFeed:
        """Create a Binance live data feed."""
        return BinanceLiveDataFeed(
            symbol=config["symbol"],
            interval=config["interval"],
            lookback_bars=config.get("lookback_bars", 1000),
            retry_interval=config.get("retry_interval", 60),
            on_new_bar=config.get("on_new_bar"),
            api_key=config.get("api_key"),
            api_secret=config.get("api_secret"),
            testnet=config.get("testnet", False)
        )
    
    @staticmethod
    def _create_yahoo_feed(config: Dict[str, Any]) -> YahooLiveDataFeed:
        """Create a Yahoo Finance live data feed."""
        return YahooLiveDataFeed(
            symbol=config["symbol"],
            interval=config["interval"],
            lookback_bars=config.get("lookback_bars", 1000),
            retry_interval=config.get("retry_interval", 60),
            on_new_bar=config.get("on_new_bar"),
            polling_interval=config.get("polling_interval", 60)
        )
    
    @staticmethod
    def _create_ibkr_feed(config: Dict[str, Any]) -> IBKRLiveDataFeed:
        """Create an IBKR live data feed."""
        return IBKRLiveDataFeed(
            symbol=config["symbol"],
            interval=config["interval"],
            lookback_bars=config.get("lookback_bars", 1000),
            retry_interval=config.get("retry_interval", 60),
            on_new_bar=config.get("on_new_bar"),
            host=config.get("host", "127.0.0.1"),
            port=config.get("port", 7497),
            client_id=config.get("client_id", 1)
        )
    
    @staticmethod
    def get_supported_sources() -> list:
        """
        Get list of supported data sources.
        
        Returns:
            List of supported data source names
        """
        return ["binance", "yahoo", "ibkr"]
    
    @staticmethod
    def get_source_info() -> Dict[str, Dict[str, Any]]:
        """
        Get information about supported data sources.
        
        Returns:
            Dictionary with information about each data source
        """
        return {
            "binance": {
                "name": "Binance",
                "description": "Cryptocurrency exchange with WebSocket support",
                "symbols": "Crypto pairs (e.g., BTCUSDT, ETHUSDT)",
                "intervals": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                "real_time": "WebSocket",
                "requires_auth": False,  # For public data
                "rate_limits": "High frequency",
                "cost": "Free for public data"
            },
            "yahoo": {
                "name": "Yahoo Finance",
                "description": "Stock and ETF data via polling",
                "symbols": "Stocks, ETFs, indices (e.g., AAPL, SPY, ^GSPC)",
                "intervals": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                "real_time": "Polling",
                "requires_auth": False,
                "rate_limits": "Moderate",
                "cost": "Free"
            },
            "ibkr": {
                "name": "Interactive Brokers",
                "description": "Professional trading platform with real-time data",
                "symbols": "Stocks, options, futures, forex",
                "intervals": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                "real_time": "Native API",
                "requires_auth": True,
                "rate_limits": "High frequency",
                "cost": "Market data subscriptions required"
            }
        } 