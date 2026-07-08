"""
Provider selection logic for the data layer.

Extracted from data_manager.py so that module stays focused on OHLCV retrieval,
caching, and the DataManager facade.  Re-exported from data_manager.py for
backward compatibility.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.data.downloader.data_downloader_factory import DataDownloaderFactory
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class ProviderSelector:
    """
    Provider selection logic based on symbol type, timeframe, and data quality.

    This class encapsulates the provider selection rules defined in PROVIDER_COMPARISON.md
    and allows for configuration-driven provider selection.
    """

    def __init__(self, config_path: str | None = None, cache_dir: str | None = None):
        """
        Initialize provider selector.

        Args:
            config_path: Path to YAML configuration file for provider rules
            cache_dir: Cache directory for downloaders that need it
        """
        self.config_path = config_path or "config/data/provider_rules.yaml"
        self.cache_dir = cache_dir
        self.rules = self._load_provider_rules()
        self.downloaders: dict[Any, Any] = {}  # Lazy initialization

    def _load_provider_rules(self) -> Dict[str, Any]:
        """Load provider selection rules from configuration file."""
        try:
            # Try multiple possible paths for the config file
            possible_paths = [
                self.config_path,
                os.path.join(os.getcwd(), self.config_path),
                os.path.join(Path(__file__).parent.parent.parent, self.config_path),
                os.path.join(Path(__file__).parent.parent.parent.parent, self.config_path),
            ]

            config_file_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    config_file_path = path
                    break

            if config_file_path:
                with open(config_file_path) as f:
                    rules = yaml.safe_load(f)
                    _logger.debug("Loaded provider rules from %s", config_file_path)
                    return rules
            else:
                _logger.warning("Provider rules config file not found at any of: %s", possible_paths)
                # Default rules if config file doesn't exist
                return self._get_default_rules()
        except Exception as e:
            _logger.warning("Failed to load provider rules from %s: %s", self.config_path, e)
            return self._get_default_rules()

    def _get_default_rules(self) -> Dict[str, Any]:
        """Get default provider selection rules."""
        return {
            "crypto": {
                "primary": "binance",
                "backup": ["coingecko", "alpha_vantage"],
                "timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"],
            },
            "stock_intraday": {
                "primary": "fmp",
                "backup": ["alpaca", "alpha_vantage", "polygon"],
                "timeframes": ["1m", "5m", "15m", "30m", "1h", "4h"],
            },
            "stock_daily": {
                "primary": "yahoo",
                "backup": ["alpaca", "tiingo", "fmp"],
                "timeframes": ["1d", "1w", "1M"],
            },
            "stock_weekly_monthly": {"primary": "tiingo", "backup": ["yahoo", "fmp"], "timeframes": ["1w", "1M"]},
        }

    def _initialize_downloader(self, name):
        """Lazy initialize a specific downloader."""
        # Use DataDownloaderFactory for unified name resolution
        canonical_name = DataDownloaderFactory.get_provider_by_code(name)
        if not canonical_name:
            _logger.error(f"Unknown provider name/alias: {name}")
            return None

        # Check if already initialized under canonical name
        if canonical_name in self.downloaders:
            # If requested by alias, also store under alias for quick lookups
            if name != canonical_name:
                self.downloaders[name] = self.downloaders[canonical_name]
            return self.downloaders[canonical_name]

        _logger.debug(f"Initializing downloader for {canonical_name} (requested as {name})")

        # Instantiate using factory
        try:
            downloader = DataDownloaderFactory.create_downloader(canonical_name)
            if downloader:
                self.downloaders[canonical_name] = downloader
                # Also store under the name it was requested by if different
                if name != canonical_name:
                    self.downloaders[name] = downloader
                return downloader
            else:
                _logger.warning(f"Failed to create downloader for {canonical_name}")
                return None
        except Exception as e:
            _logger.warning(f"Error initializing downloader for {canonical_name}: {e}")
            return None

    def _classify_symbol(self, symbol: str) -> str:
        """
        Classify symbol type (crypto, stock, etc.) using configuration-driven rules.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'AAPL')

        Returns:
            Symbol classification
        """
        symbol_upper = symbol.upper()

        # Get classification rules from config
        classification_rules = self.rules.get("symbol_classification", {})

        # Check crypto patterns (only for tickers with 6+ characters)
        if len(symbol_upper) >= 6:
            crypto_rules = classification_rules.get("crypto", {})
            crypto_patterns = crypto_rules.get("patterns", [])
            crypto_suffixes = crypto_rules.get("suffixes", [])
            crypto_assets = crypto_rules.get("known_assets", [])

            # Check against crypto patterns
            for pattern in crypto_patterns:
                if re.match(pattern, symbol_upper):
                    return "crypto"

            # Check against crypto suffixes
            if any(symbol_upper.endswith(suffix) for suffix in crypto_suffixes):
                return "crypto"

            # Check if starts with known crypto asset
            for asset in crypto_assets:
                if symbol_upper.startswith(asset) and len(symbol_upper) > len(asset):
                    return "crypto"

        # Check stock patterns
        stock_rules = classification_rules.get("stock", {})
        stock_patterns = stock_rules.get("patterns", [])

        for pattern in stock_patterns:
            if re.match(pattern, symbol_upper):
                return "stock"

        # Check for exchange suffixes
        exchange_suffixes = stock_rules.get("exchange_suffixes", {})
        for suffix in exchange_suffixes.keys():
            if symbol_upper.endswith(suffix):
                return "stock"

        # Default to unknown for unrecognized symbols
        return "unknown"

    def classify_symbol(self, symbol: str) -> str:
        """
        Public method to classify symbol type (crypto, stock, etc.).

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'AAPL')

        Returns:
            Symbol classification ('crypto', 'stock', 'unknown')
        """
        return self._classify_symbol(symbol)

    def classify_symbol_for_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Enhanced symbol classification specifically for fundamentals data retrieval.

        This method provides detailed symbol information needed for optimal
        fundamentals provider selection, including market detection, exchange
        identification, and international symbol handling.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'GOOGL', 'ASML.AS')

        Returns:
            Dictionary containing detailed symbol classification:
            {
                'symbol': str,           # Original symbol
                'normalized': str,       # Normalized symbol
                'symbol_type': str,      # 'stock', 'etf', 'reit', 'crypto', 'unknown'
                'market': str,           # 'US', 'UK', 'EU', 'ASIA', 'unknown'
                'exchange': str,         # 'NASDAQ', 'NYSE', 'LSE', 'AMS', etc.
                'country': str,          # 'US', 'GB', 'NL', etc.
                'international': bool,   # True if non-US symbol
                'currency': str,         # 'USD', 'EUR', 'GBP', etc.
                'fundamentals_support': str  # 'full', 'limited', 'none'
            }
        """
        try:
            # Handle None or invalid input
            if not symbol or not isinstance(symbol, str):
                return {
                    "symbol": symbol,
                    "normalized": "",
                    "symbol_type": "unknown",
                    "market": "unknown",
                    "exchange": "unknown",
                    "country": "unknown",
                    "international": True,
                    "currency": "USD",
                    "fundamentals_support": "none",
                }

            symbol_upper = symbol.upper().strip()

            # Handle empty string after stripping
            if not symbol_upper:
                return {
                    "symbol": symbol,
                    "normalized": "",
                    "symbol_type": "unknown",
                    "market": "unknown",
                    "exchange": "unknown",
                    "country": "unknown",
                    "international": True,
                    "currency": "USD",
                    "fundamentals_support": "none",
                }

            # Initialize classification result
            classification = {
                "symbol": symbol,
                "normalized": symbol_upper,
                "symbol_type": "unknown",
                "market": "unknown",
                "exchange": "unknown",
                "country": "unknown",
                "international": False,
                "currency": "USD",  # Default to USD
                "fundamentals_support": "none",
            }

            # 1. Detect exchange from symbol suffix
            exchange_info = self._detect_exchange_from_symbol(symbol_upper)
            if exchange_info:
                classification.update(exchange_info)

            # 2. Classify symbol type
            symbol_type = self._classify_symbol_type_detailed(symbol_upper, classification)
            classification["symbol_type"] = symbol_type

            # 3. Determine market and country
            market_info = self._determine_market_and_country(symbol_upper, classification)
            classification.update(market_info)

            # 4. Assess fundamentals support
            fundamentals_support = self._assess_fundamentals_support(classification)
            classification["fundamentals_support"] = fundamentals_support

            # 5. Set international flag
            classification["international"] = classification["country"] != "US"

            return classification

        except Exception:
            _logger.exception("Error classifying symbol %s for fundamentals:", symbol)
            return {
                "symbol": symbol,
                "normalized": symbol.upper(),
                "symbol_type": "unknown",
                "market": "unknown",
                "exchange": "unknown",
                "country": "unknown",
                "international": True,
                "currency": "USD",
                "fundamentals_support": "none",
            }

    def _detect_exchange_from_symbol(self, symbol: str) -> Dict[str, str] | None:
        """
        Detect exchange information from symbol suffix.

        Args:
            symbol: Uppercase symbol

        Returns:
            Dictionary with exchange information or None
        """
        # Exchange suffix mappings
        exchange_mappings = {
            ".L": {"exchange": "LSE", "country": "GB", "market": "UK", "currency": "GBP"},
            ".TO": {"exchange": "TSX", "country": "CA", "market": "CANADA", "currency": "CAD"},
            ".SW": {"exchange": "SWX", "country": "CH", "market": "SWISS", "currency": "CHF"},
            ".DE": {"exchange": "XETRA", "country": "DE", "market": "EU", "currency": "EUR"},
            ".PA": {"exchange": "EPA", "country": "FR", "market": "EU", "currency": "EUR"},
            ".AS": {"exchange": "AMS", "country": "NL", "market": "EU", "currency": "EUR"},
            ".MI": {"exchange": "BIT", "country": "IT", "market": "EU", "currency": "EUR"},
            ".MC": {"exchange": "BME", "country": "ES", "market": "EU", "currency": "EUR"},
            ".BR": {"exchange": "EURONEXT", "country": "BE", "market": "EU", "currency": "EUR"},
            ".VI": {"exchange": "WBAG", "country": "AT", "market": "EU", "currency": "EUR"},
            ".HK": {"exchange": "HKEX", "country": "HK", "market": "ASIA", "currency": "HKD"},
            ".T": {"exchange": "TSE", "country": "JP", "market": "ASIA", "currency": "JPY"},
            ".SS": {"exchange": "SSE", "country": "CN", "market": "ASIA", "currency": "CNY"},
            ".SZ": {"exchange": "SZSE", "country": "CN", "market": "ASIA", "currency": "CNY"},
            ".AX": {"exchange": "ASX", "country": "AU", "market": "OCEANIA", "currency": "AUD"},
            ".NZ": {"exchange": "NZX", "country": "NZ", "market": "OCEANIA", "currency": "NZD"},
        }

        for suffix, info in exchange_mappings.items():
            if symbol.endswith(suffix):
                # Remove suffix from normalized symbol
                normalized = symbol[: -len(suffix)]
                result = info.copy()
                result["normalized"] = normalized
                return result

        return None

    def _classify_symbol_type_detailed(self, symbol: str, classification: Dict[str, Any]) -> str:
        """
        Classify symbol type with detailed analysis.

        Args:
            symbol: Uppercase symbol
            classification: Current classification info

        Returns:
            Symbol type ('stock', 'etf', 'reit', 'crypto', 'unknown')
        """
        # First check if it's crypto using the original symbol (before suffix removal)
        original_symbol = classification.get("symbol", symbol).upper()
        if self._classify_symbol(original_symbol) == "crypto":
            return "crypto"

        # ETF patterns (common ETF suffixes and patterns)
        etf_patterns = [
            r".*ETF$",  # Ends with ETF
            r"^SPY$",  # SPDR S&P 500
            r"^QQQ$",  # Invesco QQQ
            r"^IWM$",  # iShares Russell 2000
            r"^VTI$",  # Vanguard Total Stock Market
            r"^VOO$",  # Vanguard S&P 500
            r"^VEA$",  # Vanguard FTSE Developed Markets
            r"^VWO$",  # Vanguard FTSE Emerging Markets
            r"^BND$",  # Vanguard Total Bond Market
            r"^GLD$",  # SPDR Gold Shares
            r"^SLV$",  # iShares Silver Trust
        ]

        for pattern in etf_patterns:
            if re.match(pattern, symbol):
                return "etf"

        # REIT patterns
        reit_patterns = [
            r".*REIT$",  # Ends with REIT
            r"^REI[T]?$",  # REI or REIT
        ]

        for pattern in reit_patterns:
            if re.match(pattern, symbol):
                return "reit"

        # Check for common stock patterns
        # Most symbols without special suffixes are stocks
        if re.match(r"^[A-Z]{1,5}$", symbol):  # 1-5 letter symbols
            return "stock"

        # Symbols with numbers might be preferred shares or special classes
        if re.match(r"^[A-Z]{1,4}[0-9]$", symbol):
            return "stock"

        # Class shares (e.g., BRK-A, BRK-B, GOOGL, GOOG)
        if re.match(r"^[A-Z]{1,4}[-.]?[A-Z]$", symbol):
            return "stock"

        return "stock"  # Default to stock for most cases

    def _determine_market_and_country(self, symbol: str, classification: Dict[str, Any]) -> Dict[str, str]:
        """
        Determine market and country information.

        Args:
            symbol: Uppercase symbol
            classification: Current classification info

        Returns:
            Dictionary with market and country info
        """
        # If exchange info already determined market/country, use it
        if classification.get("country") != "unknown":
            return {}

        # For symbols without exchange suffix, assume US market
        market_info = {
            "market": "US",
            "country": "US",
            "exchange": self._determine_us_exchange(symbol),
            "currency": "USD",
        }

        return market_info

    def _determine_us_exchange(self, symbol: str) -> str:
        """
        Determine likely US exchange for a symbol.

        Args:
            symbol: Uppercase symbol

        Returns:
            Exchange name ('NASDAQ', 'NYSE', 'AMEX', 'OTC')
        """
        # Common NASDAQ patterns (tech companies, biotech, etc.)
        nasdaq_patterns = [
            r"^[A-Z]{4,5}$",  # 4-5 letter symbols often NASDAQ
            r"^Q[A-Z]{3}$",  # Q-prefixed symbols
        ]

        # Known NASDAQ symbols (partial list)
        nasdaq_symbols = {
            "AAPL",
            "MSFT",
            "GOOGL",
            "GOOG",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "NFLX",
            "ADBE",
            "CRM",
            "ORCL",
            "CSCO",
            "INTC",
            "AMD",
            "QCOM",
        }

        # Known NYSE symbols (partial list)
        nyse_symbols = {
            "JPM",
            "BAC",
            "WFC",
            "C",
            "GS",
            "MS",
            "V",
            "MA",
            "JNJ",
            "PG",
            "KO",
            "PEP",
            "WMT",
            "HD",
            "UNH",
            "CVX",
            "XOM",
            "T",
            "VZ",
        }

        if symbol in nasdaq_symbols:
            return "NASDAQ"
        elif symbol in nyse_symbols:
            return "NYSE"
        else:
            # Use patterns as heuristic
            for pattern in nasdaq_patterns:
                if re.match(pattern, symbol):
                    return "NASDAQ"

            # Default to NYSE for shorter symbols
            if len(symbol) <= 3:
                return "NYSE"
            else:
                return "NASDAQ"

    def _assess_fundamentals_support(self, classification: Dict[str, Any]) -> str:
        """
        Assess level of fundamentals support for this symbol.

        Args:
            classification: Symbol classification info

        Returns:
            Support level ('full', 'limited', 'none')
        """
        symbol_type = classification.get("symbol_type", "unknown")
        country = classification.get("country", "unknown")
        market = classification.get("market", "unknown")

        # Crypto symbols have no fundamentals
        if symbol_type == "crypto":
            return "none"

        # US stocks and ETFs have full support
        if country == "US" and symbol_type in ["stock", "etf", "reit"]:
            return "full"

        # Major international markets have good support
        major_markets = ["UK", "EU", "CANADA", "ASIA"]
        if market in major_markets and symbol_type in ["stock", "etf"]:
            return "limited"

        # Other international markets have limited support
        if country != "unknown" and symbol_type in ["stock", "etf"]:
            return "limited"

        return "none"

    def _get_rule_name(self, symbol_type: str, timeframe: str) -> str | None:
        """
        Map symbol type and timeframe to the appropriate rule name.

        Args:
            symbol_type: Type of symbol (crypto, stock, etc.)
            timeframe: Data timeframe (e.g., '1m', '1h', '1d')

        Returns:
            Rule name or None if no mapping found
        """
        if symbol_type == "crypto":
            return "crypto"
        elif symbol_type == "stock":
            # Map stock timeframes to appropriate rules
            if timeframe in ["1m", "5m", "15m", "30m", "1h", "4h"]:
                return "stock_intraday"
            elif timeframe in ["1d"]:
                return "stock_daily"
            elif timeframe in ["1w", "1M"]:
                return "stock_weekly_monthly"

        return None

    def get_best_provider(self, symbol: str, timeframe: str) -> str | None:
        """
        Get the best provider for a given symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe (e.g., '1m', '1h', '1d')

        Returns:
            Best provider name, or None if no suitable provider found
        """
        symbol_type = self._classify_symbol(symbol)

        # Map symbol type + timeframe to rule name
        rule_name = self._get_rule_name(symbol_type, timeframe)
        if not rule_name:
            _logger.warning("No rule found for %s %s", symbol_type, timeframe)
            return None

        # Get rules for this rule name
        if rule_name not in self.rules:
            _logger.warning("No rules found for rule: %s", rule_name)
            return None

        rules = self.rules[rule_name]

        # Check if timeframe is supported
        if timeframe not in rules.get("timeframes", []):
            _logger.warning("Timeframe %s not supported for %s", timeframe, rule_name)
            return None

        # Return primary provider if available
        primary = rules.get("primary")
        if primary:
            downloader = self._initialize_downloader(primary)
            if downloader:
                return primary

        # Try backup providers
        backup = rules.get("backup", [])
        for provider in backup:
            downloader = self._initialize_downloader(provider)
            if downloader:
                return provider

        _logger.error("No suitable provider found for %s (%s)", symbol, timeframe)
        return None

    def get_best_downloader(self, symbol: str, timeframe: str) -> BaseDataDownloader | None:
        """
        Get the best downloader for a given symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe (e.g., '1m', '1h', '1d')

        Returns:
            Best downloader instance, or None if no suitable provider found
        """
        provider_name = self.get_best_provider(symbol, timeframe)
        if provider_name:
            return self._initialize_downloader(provider_name)
        return None

    def get_provider_with_failover(self, symbol: str, timeframe: str) -> List[str]:
        """
        Get ordered list of providers with failover support.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            Ordered list of provider names (primary first, then backups)
        """
        symbol_type = self._classify_symbol(symbol)

        # Map symbol type + timeframe to rule name (same logic as get_best_provider)
        rule_name = self._get_rule_name(symbol_type, timeframe)
        if not rule_name:
            _logger.warning("No rule found for %s %s", symbol_type, timeframe)
            return []

        # Get rules for this rule name
        if rule_name not in self.rules:
            _logger.warning("No rules found for rule: %s", rule_name)
            return []

        rules = self.rules[rule_name]
        providers = []

        # Check if timeframe is supported
        if timeframe not in rules.get("timeframes", []):
            _logger.warning("Timeframe %s not supported for %s", timeframe, rule_name)
            return []

        # Add primary provider
        primary = rules.get("primary")
        if primary:
            downloader = self._initialize_downloader(primary)
            if downloader:
                providers.append(primary)

        # Add backup providers
        backup = rules.get("backup", [])
        for provider in backup:
            if provider not in providers:
                downloader = self._initialize_downloader(provider)
                if downloader:
                    providers.append(provider)

        return providers

    def get_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive ticker information for symbol classification and provider selection.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with ticker information
        """
        symbol_upper = symbol.upper()
        symbol_type = self._classify_symbol(symbol)

        info = {
            "original_ticker": symbol,
            "symbol_type": symbol_type,
            "formatted_ticker": symbol_upper,
            "exchange": None,
            "base_asset": None,
            "quote_asset": None,
        }

        # Get classification rules
        classification_rules = self.rules.get("symbol_classification", {})

        if symbol_type == "crypto":
            # Parse crypto pair
            crypto_rules = classification_rules.get("crypto", {})
            crypto_suffixes = crypto_rules.get("suffixes", [])

            # Try to parse base/quote assets
            for suffix in crypto_suffixes:
                if symbol_upper.endswith(suffix):
                    base = symbol_upper[: -len(suffix)]
                    if base:
                        info["base_asset"] = base
                        info["quote_asset"] = suffix
                        break
        elif symbol_type == "stock":
            # Check for exchange suffix
            stock_rules = classification_rules.get("stock", {})
            exchange_suffixes = stock_rules.get("exchange_suffixes", {})

            for suffix, exchange_name in exchange_suffixes.items():
                if symbol_upper.endswith(suffix):
                    info["exchange"] = exchange_name
                    break

            if not info["exchange"]:
                info["exchange"] = "US Markets (NASDAQ/NYSE)"

        return info

    def get_data_provider_config(self, symbol: str, interval: str | None = None) -> Dict[str, Any]:
        """
        Get configuration for data retrieval based on ticker and interval.

        Args:
            symbol: The ticker symbol
            interval: Time interval (1d, 1h, 5m, 15m, etc.)

        Returns:
            Dictionary with provider-specific configuration
        """
        ticker_info = self.get_ticker_info(symbol)
        best_provider = self.get_best_provider(symbol, interval or "1d")

        config = {
            "ticker": ticker_info["original_ticker"],
            "provider": best_provider,
            "formatted_ticker": ticker_info["formatted_ticker"],
            "best_provider": best_provider,
            "symbol_type": ticker_info["symbol_type"],
            "exchange": ticker_info["exchange"],
            "base_asset": ticker_info["base_asset"],
            "quote_asset": ticker_info["quote_asset"],
        }

        if best_provider == "binance":
            config.update(
                {
                    "interval": interval or "1d",
                    "limit": 1000,
                    "reason": "Crypto symbol - Binance provides best coverage",
                }
            )
        elif best_provider == "yahoo":
            config.update(
                {
                    "period": "1y",
                    "interval": interval or "1d",
                    "reason": f"Stock symbol with {interval} interval - yfinance for daily data",
                }
            )
        elif best_provider == "fmp":
            config.update(
                {
                    "interval": interval or "1d",
                    "reason": f"Stock symbol with {interval} interval - FMP for intraday data",
                }
            )

        return config

    def validate_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Comprehensive ticker validation.

        Args:
            symbol: The ticker to validate

        Returns:
            Dictionary with validation results
        """
        if not symbol:
            return {"valid": False, "error": "Empty ticker", "suggestions": []}

        symbol_upper = symbol.upper().strip()

        # Basic format validation
        if len(symbol_upper) < 1 or len(symbol_upper) > 15:
            return {
                "valid": False,
                "error": "Invalid ticker length",
                "suggestions": ["Ticker should be 1-15 characters long"],
            }

        if not re.match(r"^[A-Z0-9.-]+$", symbol_upper):
            return {
                "valid": False,
                "error": "Invalid ticker format",
                "suggestions": ["Use only alphanumeric characters, dots, and hyphens"],
            }

        # Classify the ticker
        ticker_info = self.get_ticker_info(symbol)

        if ticker_info["symbol_type"] == "unknown":
            return {
                "valid": False,
                "error": "Unknown ticker format",
                "suggestions": [
                    "Check if ticker exists on supported exchanges",
                    "Verify ticker symbol is correct",
                    "Consider adding exchange suffix (e.g., .L for London)",
                ],
            }

        return {
            "valid": True,
            "symbol_type": ticker_info["symbol_type"],
            "exchange": ticker_info["exchange"],
            "base_asset": ticker_info["base_asset"],
            "quote_asset": ticker_info["quote_asset"],
            "suggestions": [],
        }
