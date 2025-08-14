from src.common.ticker_chart import generate_chart
from src.common.fundamentals import get_fundamentals, format_fundamental_analysis
from src.common import get_ohlcv
from src.model.telegram_bot import TickerAnalysis
from src.common.technicals import calculate_technicals_from_df, format_technical_analysis
from src.notification.logger import setup_logger
#from src.backtester.plotter.base_plotter import


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

        logger.debug("Downloaded data for %s", ticker)

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
        logger.exception("Error in analyze_ticker: ")
        raise

def format_ticker_report(analysis: TickerAnalysis) -> dict:
    """
    Formats a TickerAnalysis into a message and generates a chart image (as bytes).
    Returns a dict with 'message' and 'chart_bytes' (bytes or None).
    The caller should use 'chart_bytes' for sending to Telegram/email.
    """
    # Format fundamentals
    fundamentals_msg = format_fundamental_analysis(analysis.fundamentals)
    # Format technicals
    technicals_msg = ""
    if analysis.technicals is not None:
        technicals_msg = format_technical_analysis(analysis.ticker, analysis.technicals)
    # Generate chart as bytes
    chart_bytes = None
    if analysis.ohlcv is not None:
        try:
            chart_bytes = generate_chart(analysis)
            analysis.chart_image = chart_bytes
        except Exception as e:
            logger.exception("generate_chart: ")
            chart_bytes = None
            analysis.chart_image = None

    # Compose full message
    full_msg = f"<b>{analysis.ticker}</b>\n\n{fundamentals_msg}\n{technicals_msg}"
    return {
        "message": full_msg.strip(),
        "chart_bytes": chart_bytes
    }
