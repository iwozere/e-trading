"""
Common business logic for data provider management and technicals/fundamentals access.

Usage:
    from src.common import get_fundamentals, get_ohlcv, analyze_period_interval
    fundamentals = get_fundamentals('AAPL', provider='yf')
    df = get_ohlcv('AAPL', '1d', '2y', provider='yf')

Supported providers:
    'yf' (Yahoo Finance, default), 'av', 'fh', 'td', 'pg', 'bnc', 'cg'
"""
import datetime
from src.data.data_downloader_factory import DataDownloaderFactory

PROVIDER_CODES = ['yf', 'av', 'fh', 'td', 'pg', 'bnc', 'cg']


def analyze_period_interval(period: str = "2y", interval: str = "1d"):
    """
    Given a period (e.g., '2y', '6mo', '1w') and interval, return (start_date, end_date) as datetime.datetime objects.
    """
    from datetime import datetime, timedelta
    start_date = None
    end_date = datetime.now()

    if period.endswith("y"):
        years = int(period[:-1])
        start_date = datetime.now() - timedelta(days=365*years)
    elif period.endswith("m"):
        months = int(period[:-1])
        start_date = datetime.now() - timedelta(days=30*months)
    elif period.endswith("mo"):
        months = int(period[:-2])
        start_date = datetime.now() - timedelta(days=30*months)
    elif period.endswith("w"):
        weeks = int(period[:-1])
        start_date = datetime.now() - timedelta(days=7*weeks)
    else:
        # Default to 2 years if period format is not recognized
        start_date = datetime.now() - timedelta(days=730)

    return start_date, end_date


def get_ohlcv(ticker: str, interval: str, period: str, provider: str = None, **kwargs):
    """
    Retrieve OHLCV data for a ticker using the specified data provider.
    If provider is None, try to infer from ticker length (yf for stocks, bnc for crypto).

    Args:
        ticker: Stock or crypto ticker (e.g., 'AAPL', 'BTCUSDT')
        interval: Data interval (e.g., '1m', '5m', '15m', '1h', '1d')
        period: Period string (e.g., '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y')
        provider: Data provider code (e.g., 'yf', 'bnc', 'av', etc.)
        **kwargs: Additional arguments to pass to the downloader

    Returns:
        pd.DataFrame: OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    Raises:
        ValueError: If provider is unknown or period/interval combination is invalid
        RuntimeError: If data retrieval fails
    """
    # Infer provider if not specified
    if provider is None:
        provider = "yf" if len(ticker) < 5 else "bnc"

    # Get downloader
    downloader = DataDownloaderFactory.create_downloader(provider, **kwargs)
    if downloader is None:
        raise ValueError(f"Unknown or unsupported provider: {provider}")

    # Validate period/interval combination
    if not downloader.is_valid_period_interval(period, interval):
        raise ValueError(f"Invalid period/interval combination for provider {provider}: {period}/{interval}")

    # Calculate date range
    start_date, end_date = analyze_period_interval(period, interval)

    # Get OHLCV data
    return downloader.get_ohlcv(ticker, interval, start_date, end_date)