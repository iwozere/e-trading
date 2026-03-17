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

    # Provider code mapping (Unified Canonical Names as values)
    PROVIDER_MAP = {
        # Yahoo Finance
        "yf": "yahoo",
        "yahoo": "yahoo",
        "yf_finance": "yahoo",
        "yfinance": "yahoo",

        # Alpha Vantage
        "av": "alpha_vantage",
        "alphavantage": "alpha_vantage",
        "alpha_vantage": "alpha_vantage",

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
        "santiment_net": "santiment",

        # IBKR
        "ibkr": "ibkr",

        # NewsAPI
        "newsapi": "newsapi"
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
            "alpha_vantage": AlphaVantageDataDownloader,
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
        if provider == "alpha_vantage":
            api_key = kwargs.get("api_key") or os.getenv("ALPHA_VANTAGE_API_KEY")
            if not api_key:
                api_key = BaseDataDownloader._get_config_value("ALPHA_VANTAGE_API_KEY")
            if not api_key:
                raise ValueError("Alpha Vantage API key is required")
            return downloader_class(api_key=api_key)

        elif provider == "finnhub":
            api_key = kwargs.get("api_key") or os.getenv("FINNHUB_API_KEY")
            if not api_key:
                api_key = BaseDataDownloader._get_config_value("FINNHUB_API_KEY")
            if not api_key:
                raise ValueError("Finnhub API key is required")
            return downloader_class(api_key=api_key)

        elif provider == "polygon":
            api_key = kwargs.get("api_key") or os.getenv("POLYGON_API_KEY")
            if not api_key:
                api_key = BaseDataDownloader._get_config_value("POLYGON_API_KEY")
            if not api_key:
                raise ValueError("Polygon.io API key is required")
            return downloader_class(api_key=api_key)

        elif provider == "twelvedata":
            api_key = kwargs.get("api_key") or os.getenv("TWELVE_DATA_API_KEY")
            if not api_key:
                api_key = BaseDataDownloader._get_config_value("TWELVE_DATA_API_KEY")
            if not api_key:
                raise ValueError("Twelve Data API key is required")
            return downloader_class(api_key=api_key)

        elif provider == "fmp":
            api_key = kwargs.get("api_key") or os.getenv("FMP_API_KEY")
            if not api_key:
                api_key = BaseDataDownloader._get_config_value("FMP_API_KEY")
            if not api_key:
                raise ValueError("FMP API key is required")
            return downloader_class(api_key=api_key)

        elif provider == "tiingo":
            api_key = kwargs.get("api_key") or os.getenv("TIINGO_API_KEY")
            if not api_key:
                api_key = BaseDataDownloader._get_config_value("TIINGO_API_KEY")
            if not api_key:
                raise ValueError("Tiingo API key is required")
            return downloader_class(api_key=api_key)

        elif provider == "alpaca":
            api_key = kwargs.get("api_key") or os.getenv("ALPACA_API_KEY")
            secret_key = kwargs.get("secret_key") or os.getenv("ALPACA_SECRET_KEY")
            if not api_key or not secret_key:
                api_key = BaseDataDownloader._get_config_value("ALPACA_API_KEY")
                secret_key = BaseDataDownloader._get_config_value("ALPACA_SECRET_KEY")
            if not api_key or not secret_key:
                raise ValueError("Alpaca API key and secret key are required")
            base_url = kwargs.get("base_url") or os.getenv("ALPACA_BASE_URL")
            return downloader_class(api_key=api_key, secret_key=secret_key, base_url=base_url)

        elif provider == "finra":
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
            if not api_key:
                api_key = BaseDataDownloader._get_config_value("EODHD_API_KEY")
            return downloader_class(api_key=api_key)

        elif provider == "tradier":
            api_key = kwargs.get("api_key") or os.getenv("TRADIER_API")
            if not api_key:
                api_key = BaseDataDownloader._get_config_value("TRADIER_API")
            rate_limit_sleep = kwargs.get("rate_limit_sleep", 0.3)
            return downloader_class(api_key=api_key, rate_limit_sleep=rate_limit_sleep)

        # Providers not requiring API keys or using default initialization
        return downloader_class()

    @staticmethod
    def get_supported_providers() -> list:
        """Get list of supported provider codes."""
        return list(DataDownloaderFactory.PROVIDER_MAP.keys())

    @staticmethod
    def get_provider_info() -> Dict[str, Dict[str, Any]]:
        """Get information about supported providers."""
        return {
            "yahoo": {
                "codes": ["yf", "yahoo"],
                "name": "Yahoo Finance",
                "requires_api_key": False,
            },
            "alpha_vantage": {
                "codes": ["av", "alpha_vantage"],
                "name": "Alpha Vantage",
                "requires_api_key": True,
            },
            "finnhub": {
                "codes": ["fh", "finnhub"],
                "name": "Finnhub",
                "requires_api_key": True,
            },
            "polygon": {
                "codes": ["pg", "polygon"],
                "name": "Polygon.io",
                "requires_api_key": True,
            },
            "twelvedata": {
                "codes": ["td", "twelvedata"],
                "name": "Twelve Data",
                "requires_api_key": True,
            },
            "fmp": {
                "codes": ["fmp"],
                "name": "Financial Modeling Prep",
                "requires_api_key": True,
            },
            "binance": {
                "codes": ["bnc", "binance"],
                "name": "Binance",
                "requires_api_key": True,
            },
            "coingecko": {
                "codes": ["cg", "coingecko"],
                "name": "CoinGecko",
                "requires_api_key": False,
            },
            "alpaca": {
                "codes": ["alp", "alpaca"],
                "name": "Alpaca Markets",
                "requires_api_key": True,
            },
            "tiingo": {
                "codes": ["tiingo"],
                "name": "Tiingo",
                "requires_api_key": True,
            },
            "finra": {
                "codes": ["finra", "finra_trf"],
                "name": "FINRA",
                "requires_api_key": True,
            },
            "eodhd": {
                "codes": ["eodhd", "eod"],
                "name": "EODHD",
                "requires_api_key": True,
            },
            "tradier": {
                "codes": ["trdr", "tradier"],
                "name": "Tradier",
                "requires_api_key": True,
            },
            "vix": {
                "codes": ["vix"],
                "name": "VIX",
                "requires_api_key": False,
            },
            "santiment": {
                "codes": ["san", "santiment"],
                "name": "Santiment",
                "requires_api_key": False,
            }
        }

    @staticmethod
    def get_provider_by_code(provider_code: str) -> Optional[str]:
        """Get the normalized provider name by code."""
        return DataDownloaderFactory._normalize_provider(provider_code)

    @staticmethod
    def list_providers() -> None:
        """Print a formatted list of all supported providers."""
        print("Supported Data Providers:")
        print("=" * 40)
        provider_info = DataDownloaderFactory.get_provider_info()
        for provider, info in provider_info.items():
            print(f"{info['name']} ({provider}) - Codes: {', '.join(info['codes'])}")
