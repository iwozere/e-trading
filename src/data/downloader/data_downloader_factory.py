"""
Data Downloader Factory Module
------------------------------

This module provides a factory for creating data downloaders based on provider codes.
It helps determine which data downloader implementation to use based on short provider names.

Classes:
- DataDownloaderFactory: Factory for creating data downloaders
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Type
from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.data.downloader.alpha_vantage_data_downloader import AlphaVantageDataDownloader
from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader
from src.data.downloader.polygon_data_downloader import PolygonDataDownloader
from src.data.downloader.twelvedata_data_downloader import TwelveDataDataDownloader
from src.data.downloader.binance_data_downloader import BinanceDataDownloader
from src.data.downloader.coingecko_data_downloader import CoinGeckoDataDownloader
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.data.downloader.tiingo_data_downloader import TiingoDataDownloader
from src.data.downloader.alpaca_data_downloader import AlpacaDataDownloader
from src.data.downloader.finra_data_downloader import FinraDataDownloader
from src.data.downloader.eodhd_downloader import EODHDDataDownloader
from src.data.downloader.tradier_downloader import TradierDataDownloader
from src.data.downloader.vix_downloader import VIXDataDownloader
from src.data.downloader.santiment_data_downloader import SantimentDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class DataDownloaderFactory:
    """
    Factory for creating data downloaders based on provider codes.

    This factory helps determine which data downloader implementation to use
    based on short provider codes and configuration parameters.

    Supported Provider Codes:
    - "yf" or "yahoo" -> Yahoo Finance
    - "av" or "alphavantage" -> Alpha Vantage
    - "fh" or "finnhub" -> Finnhub
    - "pg" or "polygon" -> Polygon.io
    - "td" or "twelvedata" -> Twelve Data
    - "bnc" or "binance" -> Binance
    - "cg" or "coingecko" -> CoinGecko
    - "fmp" or "financial_modeling_prep" -> Financial Modeling Prep
    - "tiingo" -> Tiingo
    - "alp" or "alpaca" -> Alpaca
    - "finra" or "finra_trf" -> FINRA (short interest and TRF data)
    - "eodhd" or "eod" -> EODHD (options data)
    - "trdr" or "tradier" -> Tradier (options data)
    - "vix" -> VIX (volatility index data)
    """

    # Provider code mapping
    PROVIDER_MAP = {
        # Yahoo Finance
        "yf": "yahoo",
        "yahoo": "yahoo",
        "yf_finance": "yahoo",
        "yfinance": "yahoo",

        # Alpha Vantage
        "av": "alphavantage",
        "alphavantage": "alphavantage",
        "alpha_vantage": "alphavantage",

        # Finnhub
        "fh": "finnhub",
        "finnhub": "finnhub",

        # Polygon.io
        "pg": "polygon",
        "polygon": "polygon",
        "polygon_io": "polygon",

        # Twelve Data
        "td": "twelvedata",
        "twelvedata": "twelvedata",
        "twelve_data": "twelvedata",

        # Binance
        "bnc": "binance",
        "binance": "binance",

        # CoinGecko
        "cg": "coingecko",
        "coingecko": "coingecko",
        "coin_gecko": "coingecko",

        # Financial Modeling Prep
        "fmp": "fmp",
        "financial_modeling_prep": "fmp",
        "financialmodelingprep": "fmp",

        # Tiingo
        "tiingo": "tiingo",

        # Alpaca
        "alpaca": "alpaca",
        "alp": "alpaca",

        # FINRA
        "finra": "finra",
        "finra_trf": "finra",

        # EODHD
        "eodhd": "eodhd",
        "eod": "eodhd",

        # Tradier
        "trdr": "tradier",
        "tradier": "tradier",

        # VIX
        "vix": "vix",

        # Santiment
        "san": "santiment",
        "santiment": "santiment",
        "santiment_net": "santiment"
    }

    @staticmethod
    def create_downloader(provider_code: str, **kwargs) -> Optional[BaseDataDownloader]:
        """
        Create a data downloader based on provider code.

        Args:
            provider_code: Short provider code (e.g., "yf", "av", "bnc")
            **kwargs: Additional arguments for the downloader

        Returns:
            Data downloader instance, or None if creation fails

        Examples:
            >>> # Yahoo Finance (no API key needed)
            >>> downloader = DataDownloaderFactory.create_downloader("yf")

            >>> # Alpha Vantage (API key required)
            >>> downloader = DataDownloaderFactory.create_downloader("av", api_key="your_key")

            >>> # Binance (API key and secret required)
            >>> downloader = DataDownloaderFactory.create_downloader("bnc",
            ...     api_key="your_key", secret_key="your_secret")
        """
        try:
            # Normalize provider code
            normalized_provider = DataDownloaderFactory._normalize_provider(provider_code)
            if not normalized_provider:
                _logger.error("Unknown provider code: %s", provider_code)
                return None

            # Get the downloader class
            downloader_class = DataDownloaderFactory._get_downloader_class(normalized_provider)
            if not downloader_class:
                _logger.error("No downloader class found for provider: %s", normalized_provider)
                return None

            # Create the downloader instance
            return DataDownloaderFactory._create_downloader_instance(downloader_class, normalized_provider, **kwargs)

        except Exception as e:
            _logger.exception("Error creating downloader for provider %s: %s", provider_code, str(e))
            return None

    @staticmethod
    def _normalize_provider(provider_code: str) -> Optional[str]:
        """
        Normalize provider code to standard name.

        Args:
            provider_code: Input provider code

        Returns:
            Normalized provider name or None if not found
        """
        return DataDownloaderFactory.PROVIDER_MAP.get(provider_code.lower())

    @staticmethod
    def _get_downloader_class(provider: str) -> Optional[Type[BaseDataDownloader]]:
        """
        Get the downloader class for a provider.

        Args:
            provider: Normalized provider name

        Returns:
            Downloader class or None if not found
        """
        downloader_classes = {
            "yahoo": YahooDataDownloader,
            "alphavantage": AlphaVantageDataDownloader,
            "finnhub": FinnhubDataDownloader,
            "polygon": PolygonDataDownloader,
            "twelvedata": TwelveDataDataDownloader,
            "binance": BinanceDataDownloader,
            "coingecko": CoinGeckoDataDownloader,
            "fmp": FMPDataDownloader,
            "tiingo": TiingoDataDownloader,
            "alpaca": AlpacaDataDownloader,
            "finra": FinraDataDownloader,
            "eodhd": EODHDDataDownloader,
            "tradier": TradierDataDownloader,
            "vix": VIXDataDownloader,
            "santiment": SantimentDataDownloader,
        }
        return downloader_classes.get(provider)

    @staticmethod
    def _create_downloader_instance(downloader_class: Type[BaseDataDownloader],
                                   provider: str, **kwargs) -> BaseDataDownloader:
        """
        Create a downloader instance with appropriate parameters.

        Args:
            downloader_class: The downloader class to instantiate
            provider: Provider name for parameter extraction
            **kwargs: Additional arguments

        Returns:
            Configured downloader instance
        """
        # Provider-specific parameter extraction
        if provider == "alphavantage":
            api_key = kwargs.get("api_key") or os.getenv("ALPHA_VANTAGE_API_KEY")
            if not api_key:
                raise ValueError("Alpha Vantage API key is required")
            return downloader_class(api_key=api_key)

        elif provider == "finnhub":
            api_key = kwargs.get("api_key") or os.getenv("FINNHUB_API_KEY")
            if not api_key:
                raise ValueError("Finnhub API key is required")
            return downloader_class(api_key=api_key)

        elif provider == "polygon":
            api_key = kwargs.get("api_key") or os.getenv("POLYGON_API_KEY")
            if not api_key:
                raise ValueError("Polygon.io API key is required")
            return downloader_class(api_key=api_key)

        elif provider == "twelvedata":
            api_key = kwargs.get("api_key") or os.getenv("TWELVE_DATA_API_KEY")
            if not api_key:
                raise ValueError("Twelve Data API key is required")
            return downloader_class(api_key=api_key)

        elif provider == "binance":
            api_key = kwargs.get("api_key")
            api_secret = kwargs.get("api_secret")
            return downloader_class(api_key=api_key, api_secret=api_secret)

        elif provider == "fmp":
            api_key = kwargs.get("api_key") or os.getenv("FMP_API_KEY")
            if not api_key:
                raise ValueError("FMP API key is required")
            return downloader_class(api_key=api_key)

        elif provider == "tiingo":
            api_key = kwargs.get("api_key") or os.getenv("TIINGO_API_KEY")
            if not api_key:
                raise ValueError("Tiingo API key is required")
            return downloader_class(api_key=api_key)

        elif provider == "alpaca":
            api_key = kwargs.get("api_key") or os.getenv("ALPACA_API_KEY")
            secret_key = kwargs.get("secret_key") or os.getenv("ALPACA_SECRET_KEY")
            base_url = kwargs.get("base_url") or os.getenv("ALPACA_BASE_URL")
            if not api_key or not secret_key:
                raise ValueError("Alpaca API key and secret key are required")
            return downloader_class(api_key=api_key, secret_key=secret_key, base_url=base_url)

        elif provider == "finra":
            # FINRA can work with or without explicit parameters
            # It uses OAuth for TRF data and direct API for short interest
            rate_limit_delay = kwargs.get("rate_limit_delay", 1.0)
            date = kwargs.get("date")
            output_dir = kwargs.get("output_dir")
            output_filename = kwargs.get("output_filename", "finra_trf.csv")
            fetch_yfinance_data = kwargs.get("fetch_yfinance_data", True)
            return downloader_class(
                rate_limit_delay=rate_limit_delay,
                date=date,
                output_dir=output_dir,
                output_filename=output_filename,
                fetch_yfinance_data=fetch_yfinance_data
            )

        elif provider == "eodhd":
            api_key = kwargs.get("api_key") or os.getenv("EODHD_API_KEY")
            return downloader_class(api_key=api_key)

        elif provider == "tradier":
            api_key = kwargs.get("api_key") or os.getenv("TRADIER_API")
            rate_limit_sleep = kwargs.get("rate_limit_sleep", 0.3)
            return downloader_class(api_key=api_key, rate_limit_sleep=rate_limit_sleep)

        elif provider == "vix":
            # VIX doesn't require API key
            return downloader_class()

        elif provider == "santiment":
            # Santiment uses config internally or falls back to limited access
            return downloader_class()

        elif provider in ["yahoo", "coingecko"]:
            # These don't require API keys
            return downloader_class()

        else:
            # Fallback for unknown providers
            return downloader_class()

    @staticmethod
    def get_supported_providers() -> list:
        """
        Get list of supported provider codes.

        Returns:
            List of supported provider codes
        """
        return list(DataDownloaderFactory.PROVIDER_MAP.keys())

    @staticmethod
    def get_provider_info() -> Dict[str, Dict[str, Any]]:
        """
        Get information about supported providers.

        Returns:
            Dictionary with information about each provider
        """
        return {
            "yahoo": {
                "codes": ["yf", "yahoo", "yf_finance"],
                "name": "Yahoo Finance",
                "description": "Comprehensive fundamental data and global stock coverage",
                "requires_api_key": False,
                "rate_limits": "None for basic usage",
                "cost": "Free",
                "fundamental_data": "Comprehensive",
                "coverage": "Global stocks and ETFs"
            },
            "alphavantage": {
                "codes": ["av", "alphavantage", "alpha_vantage"],
                "name": "Alpha Vantage",
                "description": "High-quality fundamental data with API key",
                "requires_api_key": True,
                "rate_limits": "5 calls/minute, 500/day (free tier)",
                "cost": "Free tier available",
                "fundamental_data": "Comprehensive",
                "coverage": "Global stocks and ETFs"
            },
            "finnhub": {
                "codes": ["fh", "finnhub"],
                "name": "Finnhub",
                "description": "Real-time data and comprehensive fundamentals",
                "requires_api_key": True,
                "rate_limits": "60 calls/minute (free tier)",
                "cost": "Free tier available",
                "fundamental_data": "Comprehensive",
                "coverage": "Global stocks and ETFs"
            },
            "polygon": {
                "codes": ["pg", "polygon", "polygon_io"],
                "name": "Polygon.io",
                "description": "US market data with basic fundamentals (free tier)",
                "requires_api_key": True,
                "rate_limits": "5 calls/minute (free tier)",
                "cost": "Free tier available",
                "fundamental_data": "Basic (free tier)",
                "coverage": "US stocks and ETFs (free tier)"
            },
            "twelvedata": {
                "codes": ["td", "twelvedata", "twelve_data"],
                "name": "Twelve Data",
                "description": "Global coverage with basic fundamental data",
                "requires_api_key": True,
                "rate_limits": "8 calls/minute, 800/day (free tier)",
                "cost": "Free tier available",
                "fundamental_data": "Basic",
                "coverage": "Global stocks and ETFs"
            },
            "binance": {
                "codes": ["bnc", "binance"],
                "name": "Binance",
                "description": "Cryptocurrency data only",
                "requires_api_key": True,
                "rate_limits": "1200 requests/minute (free tier)",
                "cost": "Free for public data",
                "fundamental_data": "Not applicable (crypto)",
                "coverage": "Cryptocurrencies only"
            },
            "coingecko": {
                "codes": ["cg", "coingecko", "coin_gecko"],
                "name": "CoinGecko",
                "description": "Cryptocurrency data with no API key required",
                "requires_api_key": False,
                "rate_limits": "50 calls/minute (free tier)",
                "cost": "Free",
                "fundamental_data": "Not applicable (crypto)",
                "coverage": "Cryptocurrencies only"
            },
            "fmp": {
                "codes": ["fmp", "financial_modeling_prep"],
                "name": "Financial Modeling Prep",
                "description": "Comprehensive financial data and fundamentals",
                "requires_api_key": True,
                "rate_limits": "3000 calls/minute (free tier)",
                "cost": "Free tier available",
                "fundamental_data": "Comprehensive",
                "coverage": "Global stocks and ETFs"
            },
            "tiingo": {
                "codes": ["tiingo"],
                "name": "Tiingo",
                "description": "Historical data back to 1962 for active tickers",
                "requires_api_key": True,
                "rate_limits": "1000 calls/day (free tier)",
                "cost": "Free tier available",
                "fundamental_data": "Basic",
                "coverage": "US stocks and ETFs"
            },
            "alpaca": {
                "codes": ["alpaca", "alp"],
                "name": "Alpaca Markets",
                "description": "Professional-grade US market data with comprehensive fundamentals",
                "requires_api_key": True,
                "rate_limits": "200 requests/minute (free tier)",
                "cost": "Free tier available",
                "fundamental_data": "Comprehensive",
                "coverage": "US stocks and ETFs"
            },
            "finra": {
                "codes": ["finra", "finra_trf"],
                "name": "FINRA",
                "description": "Official short interest data (bi-weekly) and TRF daily short sale volume data",
                "requires_api_key": True,
                "rate_limits": "Respectful rate limiting recommended (1+ second delays)",
                "cost": "Free",
                "fundamental_data": "Short interest and TRF data only",
                "coverage": "US stocks (short interest and TRF data)"
            },
            "eodhd": {
                "codes": ["eodhd", "eod"],
                "name": "EODHD",
                "description": "Options data and historical market data",
                "requires_api_key": True,
                "rate_limits": "Varies by plan",
                "cost": "Free tier available",
                "fundamental_data": "Options data only",
                "coverage": "Global stocks (options data)"
            },
            "tradier": {
                "codes": ["trdr", "tradier"],
                "name": "Tradier",
                "description": "Options data and market data",
                "requires_api_key": True,
                "rate_limits": "Varies by plan",
                "cost": "Free tier available",
                "fundamental_data": "Options data only",
                "coverage": "US stocks (options data)"
            },
            "vix": {
                "codes": ["vix"],
                "name": "VIX",
                "description": "Volatility Index data from Yahoo Finance",
                "requires_api_key": False,
                "rate_limits": "None for basic usage",
                "cost": "Free",
                "fundamental_data": "Volatility index data only",
                "coverage": "VIX volatility index"
            },
            "santiment": {
                "codes": ["san", "santiment", "santiment_net"],
                "name": "Santiment",
                "description": "Crypto and stock sentiment/social metrics",
                "requires_api_key": False,
                "rate_limits": "100 calls/month (free), varies by plan",
                "cost": "Free tier available",
                "fundamental_data": "Sentiment and social metrics",
                "coverage": "Crypto and Stocks"
            }
        }

    @staticmethod
    def get_provider_by_code(provider_code: str) -> Optional[str]:
        """
        Get the normalized provider name by code.

        Args:
            provider_code: Provider code (e.g., "yf", "av")

        Returns:
            Normalized provider name or None if not found
        """
        return DataDownloaderFactory._normalize_provider(provider_code)

    @staticmethod
    def list_providers() -> None:
        """
        Print a formatted list of all supported providers and their codes.
        """
        print("Supported Data Providers:")
        print("=" * 80)

        provider_info = DataDownloaderFactory.get_provider_info()

        for provider, info in provider_info.items():
            codes = ", ".join(info["codes"])
            print(f"\n{info['name']} ({provider})")
            print(f"  Codes: {codes}")
            print(f"  Description: {info['description']}")
            print(f"  API Key Required: {'Yes' if info['requires_api_key'] else 'No'}")
            print(f"  Rate Limits: {info['rate_limits']}")
            print(f"  Cost: {info['cost']}")
            print(f"  Fundamental Data: {info['fundamental_data']}")
            print(f"  Coverage: {info['coverage']}")
            print("-" * 80)
