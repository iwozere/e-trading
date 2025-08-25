import re
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

class DataProvider(Enum):
    BINANCE = "binance"
    YFINANCE = "yfinance"
    UNKNOWN = "unknown"

@dataclass
class TickerInfo:
    original_ticker: str
    provider: DataProvider
    formatted_ticker: str
    exchange: Optional[str] = None
    base_asset: Optional[str] = None
    quote_asset: Optional[str] = None

class TickerClassifier:
    def __init__(self):
        # Common crypto pairs patterns
        self.crypto_patterns = [
            r'^[A-Z]{2,10}USD$',      # BTCUSD, ETHUSD, etc.
            r'^[A-Z]{2,10}USDT$',     # BTCUSDT, ETHUSDT, etc.
            r'^[A-Z]{2,10}BTC$',      # ETHBTC, ADABTC, etc.
            r'^[A-Z]{2,10}EUR$',      # BTCEUR, ETHEUR, etc.
            r'^[A-Z]{2,10}GBP$',      # BTCGBP, etc.
            r'^[A-Z]{2,10}BUSD$',     # BTCBUSD, etc.
            r'^[A-Z]{2,10}BNB$',      # ETHBNB, etc.
        ]

        # Stock exchange suffixes for yfinance
        self.stock_exchange_suffixes = {
            '.L': 'London Stock Exchange',
            '.LON': 'London Stock Exchange',
            '.LS': 'London Stock Exchange',
            '.SW': 'Swiss Exchange (SIX)',
            '.DE': 'XETRA (Germany)',
            '.F': 'Frankfurt Stock Exchange',
            '.PA': 'Euronext Paris',
            '.AS': 'Euronext Amsterdam',
            '.BR': 'Euronext Brussels',
            '.MI': 'Borsa Italiana',
            '.MC': 'Madrid Stock Exchange',
            '.TO': 'Toronto Stock Exchange',
            '.V': 'TSX Venture Exchange',
            '.AX': 'Australian Securities Exchange',
            '.NZ': 'New Zealand Exchange',
            '.HK': 'Hong Kong Stock Exchange',
            '.T': 'Tokyo Stock Exchange',
            '.KS': 'Korea Stock Exchange',
            '.SS': 'Shanghai Stock Exchange',
            '.SZ': 'Shenzhen Stock Exchange',
            '.BO': 'Bombay Stock Exchange',
            '.NS': 'National Stock Exchange of India',
            '.SA': 'Brazil Stock Exchange (B3)',
            '.MX': 'Mexican Stock Exchange',
            '.JK': 'Indonesia Stock Exchange',
            '.KL': 'Malaysia Stock Exchange',
            '.BK': 'Thailand Stock Exchange',
            '.SI': 'Singapore Stock Exchange',
            '.TWO': 'Taiwan OTC Exchange',
            '.TW': 'Taiwan Stock Exchange',
        }

        # Common US stock patterns (no suffix needed for yfinance)
        self.us_stock_patterns = [
            r'^[A-Z]{1,5}$',          # GOOGL, AAPL, MSFT, etc. (1-5 letters)
            r'^[A-Z]{1,4}\.[A-Z]$',   # BRK.A, BRK.B, etc.
        ]

        # Known crypto base assets (for better detection)
        self.crypto_assets = {
            'BTC', 'ETH', 'BNB', 'ADA', 'DOT', 'LINK', 'LTC', 'XRP',
            'SOL', 'MATIC', 'AVAX', 'LUNA', 'UNI', 'ATOM', 'ALGO',
            'VET', 'ICP', 'THETA', 'FIL', 'TRX', 'ETC', 'XLM', 'AAVE',
            'CAKE', 'SUSHI', 'CRV', 'COMP', 'YFI', 'SNX', 'MKR', 'ZEC',
            'DASH', 'NEO', 'QTUM', 'ONT', 'ZIL', 'ICX', 'OMG', 'BAT'
        }

    def classify_ticker(self, ticker: str) -> TickerInfo:
        """
        Classify a ticker and determine the appropriate data provider

        Args:
            ticker: The ticker symbol to classify

        Returns:
            TickerInfo object with provider and formatting information
        """
        ticker = ticker.upper().strip()

        # Check for stock exchange suffixes first
        for suffix, exchange_name in self.stock_exchange_suffixes.items():
            if ticker.endswith(suffix):
                return TickerInfo(
                    original_ticker=ticker,
                    provider=DataProvider.YFINANCE,
                    formatted_ticker=ticker,
                    exchange=exchange_name
                )

        # Check crypto patterns
        if self._is_crypto_ticker(ticker):
            base_asset, quote_asset = self._parse_crypto_pair(ticker)
            return TickerInfo(
                original_ticker=ticker,
                provider=DataProvider.BINANCE,
                formatted_ticker=ticker,
                base_asset=base_asset,
                quote_asset=quote_asset
            )

        # Check US stock patterns
        if self._is_us_stock(ticker):
            return TickerInfo(
                original_ticker=ticker,
                provider=DataProvider.YFINANCE,
                formatted_ticker=ticker,
                exchange="US Markets (NASDAQ/NYSE)"
            )

        # Default to unknown if no pattern matches
        return TickerInfo(
            original_ticker=ticker,
            provider=DataProvider.UNKNOWN,
            formatted_ticker=ticker
        )

    def _is_crypto_ticker(self, ticker: str) -> bool:
        """Check if ticker matches crypto patterns"""
        # Check against known patterns
        for pattern in self.crypto_patterns:
            if re.match(pattern, ticker):
                return True

        # Additional check: if it starts with known crypto asset
        for asset in self.crypto_assets:
            if ticker.startswith(asset) and len(ticker) > len(asset):
                return True

        return False

    def _parse_crypto_pair(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse crypto pair into base and quote assets"""
        # Try common quote currencies first (longest first to avoid conflicts)
        quote_currencies = ['USDT', 'BUSD', 'USD', 'EUR', 'GBP', 'BTC', 'ETH', 'BNB']

        for quote in quote_currencies:
            if ticker.endswith(quote):
                base = ticker[:-len(quote)]
                if base:  # Make sure there's a base asset
                    return base, quote

        return None, None

    def _is_us_stock(self, ticker: str) -> bool:
        """Check if ticker matches US stock patterns"""
        for pattern in self.us_stock_patterns:
            if re.match(pattern, ticker):
                return True
        return False

    def get_data_provider_config(self, ticker: str) -> Dict:
        """
        Get configuration for data retrieval based on ticker

        Returns:
            Dictionary with provider-specific configuration
        """
        ticker_info = self.classify_ticker(ticker)

        config = {
            'ticker': ticker_info.original_ticker,
            'provider': ticker_info.provider.value,
            'formatted_ticker': ticker_info.formatted_ticker,
        }

        if ticker_info.provider == DataProvider.BINANCE:
            config.update({
                'base_asset': ticker_info.base_asset,
                'quote_asset': ticker_info.quote_asset,
                'interval': '1d',  # default interval
                'limit': 100       # default limit
            })
        elif ticker_info.provider == DataProvider.YFINANCE:
            config.update({
                'exchange': ticker_info.exchange,
                'period': '1y',    # default period
                'interval': '1d'   # default interval
            })

        return config


# Example usage and testing
if __name__ == "__main__":
    classifier = TickerClassifier()

    # Test cases
    test_tickers = [
        "BTCUSD",      # Crypto - Binance
        "ETHUSD",      # Crypto - Binance
        "BTCUSDT",     # Crypto - Binance
        "VUSD.L",      # UK Stock - yfinance
        "GOOGL",       # US Stock - yfinance
        "HELN.SW",     # Swiss Stock - yfinance
        "AAPL",        # US Stock - yfinance
        "BRK.A",       # US Stock with dot - yfinance
        "TSLA",        # US Stock - yfinance
        "ADAUSDT",     # Crypto - Binance
        "SAP.DE",      # German Stock - yfinance
        "UNKNOWN123",  # Unknown format
    ]

    print("Ticker Classification Results:")
    print("=" * 80)

    for ticker in test_tickers:
        info = classifier.classify_ticker(ticker)
        config = classifier.get_data_provider_config(ticker)

        print(f"\nTicker: {ticker}")
        print(f"Provider: {info.provider.value}")
        print(f"Formatted: {info.formatted_ticker}")
        if info.exchange:
            print(f"Exchange: {info.exchange}")
        if info.base_asset and info.quote_asset:
            print(f"Pair: {info.base_asset}/{info.quote_asset}")
        print(f"Config: {config}")