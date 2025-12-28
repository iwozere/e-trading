import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pytest
import pandas as pd
from datetime import datetime, timedelta

# Import all downloaders
from src.data.downloader.binance_data_downloader import BinanceDataDownloader
from src.data.downloader.coingecko_data_downloader import CoinGeckoDataDownloader
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.data.downloader.alpha_vantage_data_downloader import AlphaVantageDataDownloader
from src.data.downloader.polygon_data_downloader import PolygonDataDownloader
from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader
from src.data.downloader.twelvedata_data_downloader import TwelveDataDataDownloader

# Import API keys from donotshare configuration
from config.donotshare.donotshare import (
    ALPHA_VANTAGE_API_KEY,
    POLYGON_API_KEY,
    FINNHUB_API_KEY,
    TWELVE_DATA_API_KEY
)

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

@pytest.mark.parametrize("downloader_class", [
    BinanceDataDownloader,
    CoinGeckoDataDownloader,
    YahooDataDownloader,
    AlphaVantageDataDownloader,
    PolygonDataDownloader,
    FinnhubDataDownloader,
    TwelveDataDataDownloader,
])
def test_get_periods_and_intervals(downloader_class):
    # For API-key-based downloaders, pass dummy key
    if downloader_class in [AlphaVantageDataDownloader, PolygonDataDownloader, FinnhubDataDownloader, TwelveDataDataDownloader]:
        instance = downloader_class(api_key="DUMMY")
    else:
        instance = downloader_class()
    periods = instance.get_periods()
    intervals = instance.get_intervals()
    assert isinstance(periods, list) and len(periods) > 0
    assert isinstance(intervals, list) and len(intervals) > 0

@pytest.mark.parametrize("downloader_class,period,interval", [
    (BinanceDataDownloader, '1d', '1h'),
    (CoinGeckoDataDownloader, '1d', '1d'),
    (YahooDataDownloader, '1d', '1d'),
    (AlphaVantageDataDownloader, '1d', '1d'),
    (PolygonDataDownloader, '1d', '1d'),
    (FinnhubDataDownloader, '1d', '1d'),
    (TwelveDataDataDownloader, '1d', '1d'),
])
def test_is_valid_period_interval(downloader_class, period, interval):
    if downloader_class in [AlphaVantageDataDownloader, PolygonDataDownloader, FinnhubDataDownloader, TwelveDataDataDownloader]:
        instance = downloader_class(api_key="DUMMY")
    else:
        instance = downloader_class()
    assert instance.is_valid_period_interval(period, interval)
    # Test invalid
    assert not instance.is_valid_period_interval('invalid', 'invalid')

@pytest.mark.parametrize("downloader_class,symbol,interval,api_env", [
    (BinanceDataDownloader, 'BTCUSDT', '1h', None),
    (CoinGeckoDataDownloader, 'bitcoin', '1d', None),
    (YahooDataDownloader, 'AAPL', '1d', None),
    (AlphaVantageDataDownloader, 'AAPL', '1d', 'ALPHA_VANTAGE_API_KEY'),
    (PolygonDataDownloader, 'AAPL', '1d', 'POLYGON_API_KEY'),
    (FinnhubDataDownloader, 'AAPL', '1d', 'FINNHUB_API_KEY'),
    (TwelveDataDataDownloader, 'AAPL', '1d', 'TWELVE_DATA_API_KEY'),
])
@pytest.mark.network
def test_get_ohlcv_smoke(downloader_class, symbol, interval, api_env):
    # Use a bigger date range in the past for YahooDataDownloader
    if downloader_class is YahooDataDownloader:
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)
    else:
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now() - timedelta(days=29)
    # API key logic
    if api_env:
        # Map environment variable names to imported API keys
        api_key_map = {
            'ALPHA_VANTAGE_API_KEY': ALPHA_VANTAGE_API_KEY,
            'POLYGON_API_KEY': POLYGON_API_KEY,
            'FINNHUB_API_KEY': FINNHUB_API_KEY,
            'TWELVE_DATA_API_KEY': TWELVE_DATA_API_KEY
        }
        api_key = api_key_map.get(api_env)
        if not api_key:
            pytest.skip(f"API key for {downloader_class.__name__} not found in donotshare.py ({api_env})")
        instance = downloader_class(api_key=api_key)
    else:
        instance = downloader_class()
    # CoinGecko uses download_historical_data
    if downloader_class is CoinGeckoDataDownloader:
        df = instance.download_historical_data(symbol, interval, start_date, end_date, save_to_csv=False)
    else:
        df = instance.get_ohlcv(symbol, interval, start_date, end_date)
    assert isinstance(df, pd.DataFrame)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column {col} in {downloader_class.__name__}"
    assert not df.empty, f"No data returned for {downloader_class.__name__}"

@pytest.mark.parametrize("downloader_class", [
    BinanceDataDownloader,
    CoinGeckoDataDownloader,
    YahooDataDownloader,
    AlphaVantageDataDownloader,
    PolygonDataDownloader,
    FinnhubDataDownloader,
    TwelveDataDataDownloader,
])
def test_importable(downloader_class):
    assert downloader_class is not None

@pytest.mark.parametrize("downloader_class", [
    AlphaVantageDataDownloader,
    PolygonDataDownloader,
    FinnhubDataDownloader,
    TwelveDataDataDownloader,
])
def test_missing_api_key_raises(downloader_class):
    with pytest.raises(TypeError):
        downloader_class()

# Placeholder for summary table test
def test_docs_summary_table_up_to_date():
    # This is a placeholder. Implement a parser to check docs/DATA_DOWNLOADERS.md if needed.
    pass

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
