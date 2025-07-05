from src.common.ticker_chart import generate_chart
from src.common.fundamentals import get_fundamentals
from src.common import get_ohlcv, analyze_period_interval
from src.model.telegram_bot import TickerAnalysis
from src.common.technicals import calculate_technicals_from_df
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


def analyze_ticker(ticker: str, period: str = "2y", interval: str = "1d", provider: str = None) -> TickerAnalysis:
    """Analyze ticker with enhanced technical analysis and recommendations, supporting multiple providers."""
    logger.info("Analyzing ticker: %s, period: %s, interval: %s, provider: %s", ticker, period, interval, provider)
    try:
        # Infer provider if not specified
        if not provider:
            provider = "yf" if len(ticker) < 5 else "bnc"

        logger.debug("provider: %s", provider)

        # Get OHLCV data using common function
        df = get_ohlcv(ticker, interval, period, provider)

        # Get fundamentals using common function (only for stock providers)
        fundamentals = None
        if provider.lower() in ["yf", "av", "fh", "td", "pg"]:
            fundamentals = get_fundamentals(ticker, provider)

        logger.debug(f"Downloaded data for {ticker}")

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
