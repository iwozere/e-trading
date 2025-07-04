from src.data.yahoo_data_downloader import YahooDataDownloader
from src.data.binance_data_downloader import BinanceDataDownloader
from src.data.alpha_vantage_data_downloader import AlphaVantageDataDownloader
from src.data.polygon_data_downloader import PolygonDataDownloader
from src.data.finnhub_data_downloader import FinnhubDataDownloader
from src.data.twelvedata_data_downloader import TwelveDataDataDownloader
from src.data.coingecko_data_downloader import CoingeckoDataDownloader

PROVIDER_MAP = {
    "yf": YahooDataDownloader,
    "yahoo": YahooDataDownloader,
    "bnc": BinanceDataDownloader,
    "binance": BinanceDataDownloader,
    "av": AlphaVantageDataDownloader,
    "alpha": AlphaVantageDataDownloader,
    "polygon": PolygonDataDownloader,
    "finnhub": FinnhubDataDownloader,
    "twelvedata": TwelveDataDataDownloader,
    "coingecko": CoingeckoDataDownloader,
}

def get_downloader(provider: str):
    """
    Return the correct data downloader instance for the given provider string.
    Supported providers: yf, bnc, av, polygon, finnhub, twelvedata, coingecko.
    Aliases are supported (see PROVIDER_MAP).
    Raises ValueError for unknown providers.
    """
    key = provider.lower()
    if key not in PROVIDER_MAP:
        raise ValueError(f"Unknown data provider: {provider}")
    return PROVIDER_MAP[key]()