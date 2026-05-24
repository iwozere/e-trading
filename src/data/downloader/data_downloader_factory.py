"""
Data Downloader Factory Module
------------------------------

This module provides a factory for creating data downloaders based on provider codes.

The registry is defined once via ``ProviderSpec`` dataclasses — a single place to add,
remove, or rename a provider. ``PROVIDER_MAP`` and the old class-lookup dict are both
derived automatically from this registry, so they can never go out of sync.

Classes:
- ProviderSpec: Metadata for a single data provider
- DataDownloaderFactory: Factory for creating data downloaders
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Type

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


@dataclass(frozen=True)
class ProviderSpec:
    """
    Single source of truth for one data provider.

    Attributes:
        canonical:       Normalised provider name used internally (e.g. ``"alpha_vantage"``).
        display_name:    Human-readable name for UI / logging.
        cls:             Downloader class that handles this provider.
        aliases:         All accepted alias codes that map to this provider (lower-case).
                         The canonical name itself does NOT need to be repeated here.
        requires_api_key: Whether an API key is mandatory for this provider.
    """
    canonical: str
    display_name: str
    cls: Type[BaseDataDownloader]
    aliases: List[str] = field(default_factory=list)
    requires_api_key: bool = False


# ---------------------------------------------------------------------------
# THE REGISTRY — one entry per provider.  Add / remove providers HERE only.
# ---------------------------------------------------------------------------
def _build_registry() -> List[ProviderSpec]:
    # IBKR and NewsAPI are imported lazily to avoid hard dependency at module load.
    # NOTE: We intentionally do NOT manipulate the global asyncio event loop here.
    # IBKR's own __init__ is responsible for any event-loop setup it needs; doing
    # asyncio.set_event_loop() at module-import time conflicts with async
    # frameworks (FastAPI, aiogram) that manage their own loop.
    try:
        from src.data.downloader.ibkr_downloader import IBKRDownloader as _IBKR
    except Exception:
        _IBKR = None  # type: ignore

    try:
        from src.data.downloader.newsapi_data_downloader import NewsAPIDataDownloader as _NewsAPI
    except ImportError:
        _NewsAPI = None  # type: ignore

    specs = [
        ProviderSpec("yahoo",        "Yahoo Finance",          YahooDataDownloader,
                     aliases=["yf", "yf_finance", "yfinance"],         requires_api_key=False),
        ProviderSpec("alpha_vantage","Alpha Vantage",           AlphaVantageDataDownloader,
                     aliases=["av", "alphavantage"],                    requires_api_key=True),
        ProviderSpec("finnhub",      "Finnhub",                 FinnhubDataDownloader,
                     aliases=["fh"],                                    requires_api_key=True),
        ProviderSpec("polygon",      "Polygon.io",              PolygonDataDownloader,
                     aliases=["pg", "polygon_io"],                      requires_api_key=True),
        ProviderSpec("twelvedata",   "Twelve Data",             TwelveDataDataDownloader,
                     aliases=["td", "twelve_data"],                     requires_api_key=True),
        ProviderSpec("binance",      "Binance",                 BinanceDataDownloader,
                     aliases=["bnc"],                                   requires_api_key=True),
        ProviderSpec("coingecko",    "CoinGecko",               CoinGeckoDataDownloader,
                     aliases=["cg", "coin_gecko"],                      requires_api_key=False),
        ProviderSpec("fmp",          "Financial Modeling Prep", FMPDataDownloader,
                     aliases=["financial_modeling_prep", "financialmodelingprep"],
                                                                        requires_api_key=True),
        ProviderSpec("tiingo",       "Tiingo",                  TiingoDataDownloader,
                     aliases=[],                                        requires_api_key=True),
        ProviderSpec("alpaca",       "Alpaca Markets",          AlpacaDataDownloader,
                     aliases=["alp"],                                   requires_api_key=True),
        ProviderSpec("finra",        "FINRA",                   FinraDataDownloader,
                     aliases=["finra_trf"],                             requires_api_key=True),
        ProviderSpec("eodhd",        "EODHD",                   EODHDDataDownloader,
                     aliases=["eod"],                                   requires_api_key=True),
        ProviderSpec("tradier",      "Tradier",                 TradierDataDownloader,
                     aliases=["trdr"],                                  requires_api_key=True),
        ProviderSpec("vix",          "VIX",                     VIXDataDownloader,
                     aliases=[],                                        requires_api_key=False),
        ProviderSpec("santiment",    "Santiment",               SantimentDataDownloader,
                     aliases=["san", "santiment_net"],                  requires_api_key=False),
    ]

    if _IBKR:
        specs.append(ProviderSpec("ibkr", "Interactive Brokers", _IBKR,
                                  aliases=[], requires_api_key=False))
    if _NewsAPI:
        specs.append(ProviderSpec("newsapi", "NewsAPI", _NewsAPI,
                                  aliases=[], requires_api_key=True))

    return specs


_REGISTRY: List[ProviderSpec] = _build_registry()

# Derived lookups — built once from the registry.
# alias → ProviderSpec  (canonical name is also an alias to itself)
_BY_ALIAS: Dict[str, ProviderSpec] = {}
for _spec in _REGISTRY:
    _BY_ALIAS[_spec.canonical] = _spec
    for _alias in _spec.aliases:
        _BY_ALIAS[_alias] = _spec


class DataDownloaderFactory:
    """
    Factory for creating data downloaders based on provider codes.

    The provider registry lives in ``_REGISTRY`` (module level). To add a new
    provider, append a ``ProviderSpec`` there — nothing else needs to change.

    All supported provider codes and aliases are derived automatically.
    """

    # Backward-compatible class attribute: alias → canonical name
    PROVIDER_MAP: Dict[str, str] = {alias: spec.canonical for alias, spec in _BY_ALIAS.items()}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_downloader(provider_code: str, **kwargs) -> Optional[BaseDataDownloader]:
        """
        Create a data downloader based on provider code.

        Args:
            provider_code: Any registered alias or canonical name (case-insensitive).
            **kwargs: Optional parameters forwarded to the downloader constructor.
                      All downloaders are self-configuring; kwargs allow overrides.

        Returns:
            Downloader instance, or None if the code is unknown or creation fails.
        """
        try:
            spec = _BY_ALIAS.get(provider_code.lower())
            if not spec:
                _logger.error("Unknown provider code: %s", provider_code)
                return None
            return spec.cls(**kwargs)
        except Exception:
            _logger.exception("Error creating downloader for provider %s", provider_code)
            return None

    @staticmethod
    def get_supported_providers() -> List[str]:
        """Return all accepted provider code strings (aliases + canonical names)."""
        return list(_BY_ALIAS.keys())

    @staticmethod
    def get_provider_info() -> Dict[str, Dict[str, Any]]:
        """Return a dict of canonical_name → info for all registered providers."""
        return {
            spec.canonical: {
                "codes": [spec.canonical] + list(spec.aliases),
                "name": spec.display_name,
                "requires_api_key": spec.requires_api_key,
            }
            for spec in _REGISTRY
        }

    @staticmethod
    def get_provider_by_code(provider_code: str) -> Optional[str]:
        """Return the canonical provider name for a given alias/code, or None."""
        spec = _BY_ALIAS.get(provider_code.lower())
        return spec.canonical if spec else None

    @staticmethod
    def list_providers() -> None:
        """Log a formatted list of all registered providers."""
        _logger.info("Supported Data Providers:")
        _logger.info("=" * 40)
        for spec in _REGISTRY:
            codes = ", ".join([spec.canonical] + list(spec.aliases))
            _logger.info("%s (%s) - Codes: %s", spec.display_name, spec.canonical, codes)

    # ------------------------------------------------------------------ #
    #  Legacy private helpers kept for backward compatibility              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_provider(provider_code: str) -> Optional[str]:
        """Map an alias/code to its canonical provider name."""
        return DataDownloaderFactory.get_provider_by_code(provider_code)

    @staticmethod
    def _get_downloader_class(provider: str) -> Optional[Type[BaseDataDownloader]]:
        """Return the downloader class for a canonical provider name."""
        spec = _BY_ALIAS.get(provider)
        return spec.cls if spec else None

    @staticmethod
    def _create_downloader_instance(downloader_class: Type[BaseDataDownloader],
                                    provider: str, **kwargs) -> BaseDataDownloader:
        """Instantiate a downloader class (all are self-configuring)."""
        return downloader_class(**kwargs)
