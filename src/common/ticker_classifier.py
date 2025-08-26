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
        # Compile regex patterns for better performance
        self._compile_patterns()

        # Initialize exchange mappings
        self._init_exchange_mappings()

        # Initialize crypto assets
        self._init_crypto_assets()

    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        # Common crypto pairs patterns
        self.crypto_patterns = [
            re.compile(r'^[A-Z]{2,10}USD$'),      # BTCUSD, ETHUSD, etc.
            re.compile(r'^[A-Z]{2,10}USDT$'),     # BTCUSDT, ETHUSDT, etc.
            re.compile(r'^[A-Z]{2,10}BTC$'),      # ETHBTC, ADABTC, etc.
            re.compile(r'^[A-Z]{2,10}EUR$'),      # BTCEUR, ETHEUR, etc.
            re.compile(r'^[A-Z]{2,10}GBP$'),      # BTCGBP, etc.
            re.compile(r'^[A-Z]{2,10}BUSD$'),     # BTCBUSD, etc.
            re.compile(r'^[A-Z]{2,10}BNB$'),      # ETHBNB, etc.
            re.compile(r'^[A-Z]{2,10}ETH$'),      # BTCETH, etc.
        ]

        # Common US stock patterns (no suffix needed for yfinance)
        self.us_stock_patterns = [
            re.compile(r'^[A-Z]{2,5}$'),          # GOOGL, AAPL, MSFT, etc. (2-5 letters)
            re.compile(r'^[A-Z]{1,4}\.[A-Z]$'),   # BRK.A, BRK.B, etc.
        ]

    def _init_exchange_mappings(self):
        """Initialize comprehensive exchange suffix mappings."""
        # Stock exchange suffixes for yfinance
        self.stock_exchange_suffixes = {
            # UK and Ireland
            '.L': 'London Stock Exchange',
            '.LON': 'London Stock Exchange',
            '.LS': 'London Stock Exchange',
            '.IL': 'Irish Stock Exchange',

            # Swiss Exchange
            '.SW': 'Swiss Exchange (SIX)',
            '.SIX': 'Swiss Exchange (SIX)',

            # German Exchanges
            '.DE': 'XETRA (Germany)',
            '.F': 'Frankfurt Stock Exchange',
            '.BER': 'Berlin Stock Exchange',
            '.DUS': 'Düsseldorf Stock Exchange',
            '.HAM': 'Hamburg Stock Exchange',
            '.MUN': 'Munich Stock Exchange',
            '.STU': 'Stuttgart Stock Exchange',

            # French Exchanges
            '.PA': 'Euronext Paris',
            '.PAR': 'Euronext Paris',

            # Dutch Exchanges
            '.AS': 'Euronext Amsterdam',
            '.AMS': 'Euronext Amsterdam',

            # Belgian Exchanges
            '.BR': 'Euronext Brussels',
            '.BRU': 'Euronext Brussels',

            # Italian Exchanges
            '.MI': 'Borsa Italiana',
            '.MIL': 'Borsa Italiana',

            # Spanish Exchanges
            '.MC': 'Madrid Stock Exchange',
            '.MAD': 'Madrid Stock Exchange',
            '.BCN': 'Barcelona Stock Exchange',
            '.BIL': 'Bilbao Stock Exchange',
            '.VAL': 'Valencia Stock Exchange',

            # Portuguese Exchanges
            '.LS': 'Euronext Lisbon',
            '.LIS': 'Euronext Lisbon',

            # Nordic Exchanges
            '.CO': 'Copenhagen Stock Exchange (Denmark)',
            '.CPH': 'Copenhagen Stock Exchange (Denmark)',
            '.ST': 'Stockholm Stock Exchange (Sweden)',
            '.STO': 'Stockholm Stock Exchange (Sweden)',
            '.HE': 'Helsinki Stock Exchange (Finland)',
            '.HEL': 'Helsinki Stock Exchange (Finland)',
            '.OS': 'Oslo Stock Exchange (Norway)',
            '.OSL': 'Oslo Stock Exchange (Norway)',
            '.IC': 'Iceland Stock Exchange',
            '.REY': 'Iceland Stock Exchange',

            # Canadian Exchanges
            '.TO': 'Toronto Stock Exchange',
            '.TOR': 'Toronto Stock Exchange',
            '.V': 'TSX Venture Exchange',
            '.VAN': 'TSX Venture Exchange',
            '.NEO': 'NEO Exchange',

            # Australian Exchanges
            '.AX': 'Australian Securities Exchange',
            '.ASX': 'Australian Securities Exchange',

            # New Zealand Exchange
            '.NZ': 'New Zealand Exchange',
            '.NZX': 'New Zealand Exchange',

            # Asian Exchanges
            '.HK': 'Hong Kong Stock Exchange',
            '.HKG': 'Hong Kong Stock Exchange',
            '.T': 'Tokyo Stock Exchange',
            '.TYO': 'Tokyo Stock Exchange',
            '.KS': 'Korea Stock Exchange',
            '.KSE': 'Korea Stock Exchange',
            '.SS': 'Shanghai Stock Exchange',
            '.SHA': 'Shanghai Stock Exchange',
            '.SZ': 'Shenzhen Stock Exchange',
            '.SHE': 'Shenzhen Stock Exchange',
            '.TW': 'Taiwan Stock Exchange',
            '.TWO': 'Taiwan OTC Exchange',
            '.SG': 'Singapore Stock Exchange',
            '.SIN': 'Singapore Stock Exchange',
            '.KL': 'Malaysia Stock Exchange',
            '.KUL': 'Malaysia Stock Exchange',
            '.BK': 'Thailand Stock Exchange',
            '.BKK': 'Thailand Stock Exchange',
            '.JK': 'Indonesia Stock Exchange',
            '.JKT': 'Indonesia Stock Exchange',
            '.PH': 'Philippine Stock Exchange',
            '.PSE': 'Philippine Stock Exchange',

            # Indian Exchanges
            '.BO': 'Bombay Stock Exchange',
            '.BSE': 'Bombay Stock Exchange',
            '.NS': 'National Stock Exchange of India',
            '.NSE': 'National Stock Exchange of India',

            # Brazilian Exchanges
            '.SA': 'Brazil Stock Exchange (B3)',
            '.B3': 'Brazil Stock Exchange (B3)',

            # Mexican Exchanges
            '.MX': 'Mexican Stock Exchange',
            '.MEX': 'Mexican Stock Exchange',

            # Chilean Exchanges
            '.SN': 'Santiago Stock Exchange',
            '.SGO': 'Santiago Stock Exchange',

            # Colombian Exchanges (moved after Danish to avoid conflict)
            '.BOG': 'Colombia Stock Exchange',

            # Peruvian Exchanges
            '.LM': 'Lima Stock Exchange',
            '.LIM': 'Lima Stock Exchange',

            # Argentine Exchanges
            '.BA': 'Buenos Aires Stock Exchange',
            '.BUE': 'Buenos Aires Stock Exchange',

            # South African Exchanges
            '.JO': 'Johannesburg Stock Exchange',
            '.JSE': 'Johannesburg Stock Exchange',

            # Egyptian Exchanges
            '.CA': 'Cairo Stock Exchange',
            '.CAI': 'Cairo Stock Exchange',

            # Nigerian Exchanges
            '.NG': 'Nigerian Stock Exchange',
            '.LAG': 'Nigerian Stock Exchange',

            # Kenyan Exchanges
            '.NA': 'Nairobi Stock Exchange',
            '.NBO': 'Nairobi Stock Exchange',

            # Israeli Exchanges
            '.TA': 'Tel Aviv Stock Exchange',
            '.TLV': 'Tel Aviv Stock Exchange',

            # Turkish Exchanges
            '.IS': 'Istanbul Stock Exchange',
            '.IST': 'Istanbul Stock Exchange',

            # Russian Exchanges
            '.ME': 'Moscow Exchange',
            '.MOS': 'Moscow Exchange',

            # Polish Exchanges
            '.WA': 'Warsaw Stock Exchange',
            '.WAR': 'Warsaw Stock Exchange',

            # Czech Exchanges
            '.PR': 'Prague Stock Exchange',
            '.PRA': 'Prague Stock Exchange',

            # Hungarian Exchanges
            '.BD': 'Budapest Stock Exchange',
            '.BUD': 'Budapest Stock Exchange',

            # Romanian Exchanges
            '.BU': 'Bucharest Stock Exchange',
            '.BUC': 'Bucharest Stock Exchange',

            # Bulgarian Exchanges
            '.SO': 'Sofia Stock Exchange',
            '.SOF': 'Sofia Stock Exchange',

            # Croatian Exchanges
            '.ZG': 'Zagreb Stock Exchange',
            '.ZAG': 'Zagreb Stock Exchange',

            # Slovenian Exchanges
            '.LJ': 'Ljubljana Stock Exchange',
            '.LJU': 'Ljubljana Stock Exchange',

            # Slovak Exchanges
            '.BA': 'Bratislava Stock Exchange',
            '.BRA': 'Bratislava Stock Exchange',

            # Lithuanian Exchanges
            '.VS': 'Vilnius Stock Exchange',
            '.VIL': 'Vilnius Stock Exchange',

            # Latvian Exchanges
            '.RG': 'Riga Stock Exchange',
            '.RIG': 'Riga Stock Exchange',

            # Estonian Exchanges
            '.TL': 'Tallinn Stock Exchange',
            '.TAL': 'Tallinn Stock Exchange',
        }

    def _init_crypto_assets(self):
        """Initialize comprehensive crypto asset list."""
        # Known crypto base assets (for better detection)
        self.crypto_assets = {
            # Major cryptocurrencies
            'BTC', 'ETH', 'BNB', 'ADA', 'DOT', 'LINK', 'LTC', 'XRP',
            'SOL', 'MATIC', 'AVAX', 'LUNA', 'UNI', 'ATOM', 'ALGO',
            'VET', 'ICP', 'THETA', 'FIL', 'TRX', 'ETC', 'XLM', 'AAVE',
            'CAKE', 'SUSHI', 'CRV', 'COMP', 'YFI', 'SNX', 'MKR', 'ZEC',
            'DASH', 'NEO', 'QTUM', 'ONT', 'ZIL', 'ICX', 'OMG', 'BAT',

            # Additional major cryptocurrencies
            'BCH', 'EOS', 'IOTA', 'NEO', 'XMR', 'VET', 'THETA', 'FTT',
            'CRO', 'LEO', 'HT', 'OKB', 'KCS', 'BTT', 'DOGE', 'SHIB',
            'LINK', 'CHZ', 'HOT', 'ENJ', 'MANA', 'SAND', 'AXS', 'GALA',
            'ROSE', 'ONE', 'HARMONY', 'IOTX', 'ANKR', 'OCEAN', 'ALPHA',
            'AUDIO', 'RLC', 'STORJ', 'BAND', 'NMR', 'REN', 'KNC', 'ZRX',

            # DeFi tokens
            'AAVE', 'COMP', 'SNX', 'MKR', 'YFI', 'CRV', 'SUSHI', 'UNI',
            'BAL', 'SUSHI', '1INCH', 'DYDX', 'PERP', 'RAD', 'RARE',
            'ENS', 'IMX', 'OP', 'ARB', 'MATIC', 'AVAX', 'FTM', 'NEAR',
            'ATOM', 'OSMO', 'JUNO', 'SCRT', 'AKT', 'FET', 'OCEAN',

            # Layer 1 and Layer 2 tokens
            'SOL', 'AVAX', 'FTM', 'NEAR', 'ATOM', 'DOT', 'ADA', 'ALGO',
            'MATIC', 'OP', 'ARB', 'IMX', 'ZKSYNC', 'STARK', 'POLYGON',

            # Gaming and Metaverse tokens
            'AXS', 'SAND', 'MANA', 'GALA', 'ENJ', 'CHZ', 'ALICE', 'TLM',
            'HERO', 'HERO', 'HERO', 'HERO', 'HERO', 'HERO', 'HERO',

            # AI and Data tokens
            'OCEAN', 'FET', 'AGIX', 'NMR', 'BAND', 'LINK', 'CHAINLINK',

            # Privacy tokens
            'XMR', 'ZEC', 'DASH', 'PIVX', 'BEAM', 'GRIN', 'SCRT',

            # Exchange tokens
            'BNB', 'FTT', 'CRO', 'LEO', 'HT', 'OKB', 'KCS', 'BTT',
        }

    def classify_ticker(self, ticker: str) -> TickerInfo:
        """
        Classify a ticker and determine the appropriate data provider

        Args:
            ticker: The ticker symbol to classify

        Returns:
            TickerInfo object with provider and formatting information
        """
        if not ticker or not isinstance(ticker, str):
            return TickerInfo(
                original_ticker=ticker or "",
                provider=DataProvider.UNKNOWN,
                formatted_ticker=ticker or ""
            )

        ticker = ticker.upper().strip()

        # Validate ticker format
        if not self._is_valid_ticker_format(ticker):
            return TickerInfo(
                original_ticker=ticker,
                provider=DataProvider.UNKNOWN,
                formatted_ticker=ticker
            )

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

    def _is_valid_ticker_format(self, ticker: str) -> bool:
        """
        Validate ticker format for basic sanity checks.

        Args:
            ticker: The ticker to validate

        Returns:
            True if ticker format is valid, False otherwise
        """
        if not ticker:
            return False

        # Check length (1-15 characters for most tickers)
        if len(ticker) < 1 or len(ticker) > 15:
            return False

        # Check for valid characters (alphanumeric, dots, hyphens)
        if not re.match(r'^[A-Z0-9.-]+$', ticker):
            return False

        # Check for common invalid patterns
        if ticker.startswith('.') or ticker.endswith('.'):
            return False

        # Check for consecutive dots
        if '..' in ticker:
            return False

        return True

    def _is_crypto_ticker(self, ticker: str) -> bool:
        """Check if ticker matches crypto patterns"""
        # Check against compiled patterns
        for pattern in self.crypto_patterns:
            if pattern.match(ticker):
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
            if pattern.match(ticker):
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

    def get_supported_exchanges(self) -> Dict[str, str]:
        """
        Get list of all supported exchanges.

        Returns:
            Dictionary mapping exchange suffixes to exchange names
        """
        return self.stock_exchange_suffixes.copy()

    def get_supported_crypto_assets(self) -> set:
        """
        Get list of all supported crypto assets.

        Returns:
            Set of supported crypto asset symbols
        """
        return self.crypto_assets.copy()

    def validate_ticker(self, ticker: str) -> Dict[str, any]:
        """
        Comprehensive ticker validation.

        Args:
            ticker: The ticker to validate

        Returns:
            Dictionary with validation results
        """
        if not ticker:
            return {
                'valid': False,
                'error': 'Empty ticker',
                'suggestions': []
            }

        ticker = ticker.upper().strip()

        # Basic format validation
        if not self._is_valid_ticker_format(ticker):
            return {
                'valid': False,
                'error': 'Invalid ticker format',
                'suggestions': ['Use only alphanumeric characters, dots, and hyphens', 'Length should be 1-15 characters']
            }

        # Classify the ticker
        ticker_info = self.classify_ticker(ticker)

        if ticker_info.provider == DataProvider.UNKNOWN:
            return {
                'valid': False,
                'error': 'Unknown ticker format',
                'suggestions': [
                    'Check if ticker exists on supported exchanges',
                    'Verify ticker symbol is correct',
                    'Consider adding exchange suffix (e.g., .L for London)'
                ]
            }

        return {
            'valid': True,
            'provider': ticker_info.provider.value,
            'exchange': ticker_info.exchange,
            'base_asset': ticker_info.base_asset,
            'quote_asset': ticker_info.quote_asset,
            'suggestions': []
        }


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
        "NOVO.CO",     # Danish Stock - yfinance
        "UNKNOWN123",  # Unknown format
    ]

    print("Ticker Classification Results:")
    print("=" * 80)

    for ticker in test_tickers:
        info = classifier.classify_ticker(ticker)
        config = classifier.get_data_provider_config(ticker)
        validation = classifier.validate_ticker(ticker)

        print(f"\nTicker: {ticker}")
        print(f"Provider: {info.provider.value}")
        print(f"Formatted: {info.formatted_ticker}")
        print(f"Valid: {validation['valid']}")
        if info.exchange:
            print(f"Exchange: {info.exchange}")
        if info.base_asset and info.quote_asset:
            print(f"Pair: {info.base_asset}/{info.quote_asset}")
        if not validation['valid']:
            print(f"Error: {validation['error']}")
            if validation['suggestions']:
                print(f"Suggestions: {', '.join(validation['suggestions'])}")
        print(f"Config: {config}")