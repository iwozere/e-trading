import tempfile
import datetime

from src.screener.telegram.chart import generate_chart
from src.screener.telegram.fundamentals import get_fundamentals
from src.model.model import TickerAnalysis
from src.screener.telegram.technicals import calculate_technicals_from_df
from src.data.yahoo_data_downloader import YahooDataDownloader
from src.data.binance_data_downloader import BinanceDataDownloader
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


def analyze_period_interval(period: str = "2y", interval: str = "1d"):
    start_date = None
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if period.endswith("y"):
        years = int(period[:-1])
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365*years)).strftime("%Y-%m-%d")
    elif period.endswith("m"):
        months = int(period[:-1])
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30*months)).strftime("%Y-%m-%d")
    elif period.endswith("mo"):
        months = int(period[:-2])
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30*months)).strftime("%Y-%m-%d")
    elif period.endswith("w"):
        weeks = int(period[:-1])
        start_date = (datetime.datetime.now() - datetime.timedelta(days=7*weeks)).strftime("%Y-%m-%d")
    else:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    return start_date, end_date

def analyze_ticker(ticker: str, period: str = "2y", interval: str = "1d", provider: str = None) -> TickerAnalysis:
    """Analyze ticker with enhanced technical analysis and recommendations, supporting both Yahoo Finance and Binance providers."""
    logger.info("Analyzing ticker: %s, period: %s, interval: %s, provider: %s", ticker, period, interval, provider)
    try:
        df = None
        fundamentals = None
        start_date, end_date = analyze_period_interval(period, interval)

        if not provider:
            provider = "yf" if len(ticker) < 5 else "bnc"

        logger.debug("start_date: %s, end_date: %s, provider: %s", start_date, end_date, provider)

        if provider.lower() == "yf":
            # Yahoo Finance
            downloader = YahooDataDownloader()
            df = downloader.get_ohlcv(ticker, interval, start_date, end_date)
            fundamentals = get_fundamentals(ticker)
        elif provider.lower() == "bnc":
            downloader = BinanceDataDownloader()
            df = downloader.get_ohlcv(ticker, interval, start_date, end_date, save_to_csv=False)
            fundamentals = None
        else:
            raise ValueError(f"Unknown provider: {provider}")

        logger.debug(f"Downloaded data for {ticker} from {start_date} to {end_date}")

        # Calculate technicals and update df with indicator columns
        df, technicals = calculate_technicals_from_df(df)

        # Create a temporary TickerAnalysis for charting, since we need to pass the object
        temp_analysis = TickerAnalysis(
            ticker=ticker.upper(),
            fundamentals=fundamentals,
            technicals=technicals,
            chart_image=None,
            df=df
        )
        chart_image = generate_chart(temp_analysis)
        return TickerAnalysis(
            ticker=ticker.upper(),
            fundamentals=fundamentals,
            technicals=technicals,
            chart_image=chart_image,
            df=df
        )
    except Exception as e:
        logger.error("Error in analyze_ticker: %s", e, exc_info=True)
        raise
