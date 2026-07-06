"""
Common business logic for data provider management and technicals/fundamentals access.

Usage:
    from src.common import get_fundamentals, get_ohlcv, analyze_period_interval
    fundamentals = get_fundamentals('AAPL', provider='yf')
    df = get_ohlcv('AAPL', '1d', '2y', provider='yahoo')
"""

from datetime import UTC
from typing import Optional

from src.data.data_manager import ProviderSelector
from src.data.downloader.data_downloader_factory import DataDownloaderFactory

PROVIDER_CODES = list(set(DataDownloaderFactory.PROVIDER_MAP.values()))

# Initialize the provider selector
_provider_selector = ProviderSelector()


def determine_provider(ticker: str) -> str:
    """
    Intelligently determine the appropriate data provider based on ticker characteristics.
    Uses the ProviderSelector for comprehensive pattern matching.

    Args:
        ticker: Stock or crypto ticker (e.g., 'AAPL', 'BTCUSDT', 'VUSD.L')

    Returns:
        str: Canonical provider name (e.g., 'yahoo', 'binance')
    """
    # Get the best provider canonical name for this ticker
    best_provider_name = _provider_selector.get_best_provider(ticker, "1d")

    if best_provider_name:
        return best_provider_name

    return "yahoo"  # Default fallback


def get_ticker_info(ticker: str):
    """
    Get comprehensive ticker information using the ProviderSelector.

    Args:
        ticker: Stock or crypto ticker (e.g., 'AAPL', 'BTCUSDT', 'VUSD.L')

    Returns:
        Dictionary with ticker information including provider, exchange, and asset information
    """
    return _provider_selector.get_ticker_info(ticker)


def get_data_provider_config(ticker: str):
    """
    Get configuration for data retrieval based on ticker classification.

    Args:
        ticker: Stock or crypto ticker (e.g., 'AAPL', 'BTCUSDT', 'VUSD.L')

    Returns:
        Dictionary with provider-specific configuration
    """
    return _provider_selector.get_data_provider_config(ticker)


def analyze_period_interval(period: str = "2y", interval: str = "1d"):
    """
    Given a period string (e.g. '2y', '6mo', '1w', '30d') return
    (start_date, end_date) as ``datetime`` objects.

    Supported suffixes (checked longest-first to avoid ambiguity):
        ``mo`` – calendar months  (30-day approximation)
        ``y``  – years            (365-day approximation)
        ``w``  – weeks
        ``d``  – days
        ``m``  – months           (alias for ``mo``)
    """
    from datetime import datetime, timedelta

    end_date = datetime.now(UTC)

    period = period.strip().lower()

    # NOTE: check 'mo' before 'm' – otherwise '6mo' matches 'm' first.
    if period.endswith("mo"):
        start_date = end_date - timedelta(days=30 * int(period[:-2]))
    elif period.endswith("y"):
        start_date = end_date - timedelta(days=365 * int(period[:-1]))
    elif period.endswith("w"):
        start_date = end_date - timedelta(weeks=int(period[:-1]))
    elif period.endswith("d"):
        start_date = end_date - timedelta(days=int(period[:-1]))
    elif period.endswith("m"):
        # 'm' is an alias for 'mo' (months), NOT minutes
        start_date = end_date - timedelta(days=30 * int(period[:-1]))
    else:
        # Unrecognised format – default to 2 years
        start_date = end_date - timedelta(days=730)

    return start_date, end_date


_data_manager = None


import threading as _threading

_data_manager_lock = _threading.Lock()


def _get_data_manager():
    """Lazy, thread-safe singleton for DataManager. Expensive to construct; create once."""
    global _data_manager
    if _data_manager is None:
        with _data_manager_lock:
            if _data_manager is None:
                import tempfile

                from src.data.data_manager import DataManager

                try:
                    from config.donotshare.donotshare import DATA_CACHE_DIR
                except ImportError:
                    DATA_CACHE_DIR = str((__import__("pathlib").Path(tempfile.gettempdir())) / "e-trading-cache")
                _data_manager = DataManager(cache_dir=DATA_CACHE_DIR)
    return _data_manager


def get_ohlcv(ticker: str, interval: str, period: str, provider: Optional[str] = None, **kwargs):
    """
    Retrieve OHLCV data for a ticker using the DataManager with caching support.
    If provider is None, the DataManager will automatically select the best provider.

    Args:
        ticker: Stock or crypto ticker (e.g., 'AAPL', 'BTCUSDT', 'VUSD.L')
        interval: Data interval (e.g., '1m', '5m', '15m', '1h', '1d')
        period: Period string (e.g., '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y')
        provider: Data provider code (e.g., 'yf', 'bnc', 'av', etc.) - optional, auto-selected if None
        **kwargs: Additional arguments (force_refresh, etc.)

    Returns:
        pd.DataFrame: OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    Raises:
        ValueError: If period/interval combination is invalid
        RuntimeError: If data retrieval fails
    """
    start_date, end_date = analyze_period_interval(period, interval)
    return _get_data_manager().get_ohlcv(ticker, interval, start_date, end_date, **kwargs)


def get_fundamentals(ticker: str, data_type: str = "general", **kwargs):
    """
    Retrieve fundamentals data for a ticker using the DataManager with caching support.

    Args:
        ticker: Stock or crypto ticker (e.g., 'AAPL')
        data_type: Type of fundamental data ('general', 'ratios', 'statements')
        **kwargs: Additional arguments forwarded to DataManager

    Returns:
        dict: Fundamentals data, or None if unavailable
    """
    return _get_data_manager().get_fundamentals(ticker, data_type=data_type, **kwargs)
