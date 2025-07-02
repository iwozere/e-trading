import datetime

from src.screener.telegram.chart import generate_chart
from src.screener.telegram.fundamentals import get_fundamentals
from src.screener.telegram.models import TickerAnalysis
from src.screener.telegram.technicals import calculate_technicals_from_df
from src.data.yahoo_data_downloader import YahooDataDownloader
from src.data.binance_data_downloader import BinanceDataDownloader
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


def generate_recommendation(recommendations: dict) -> str:
    """Generate trading recommendation based on technical indicators."""
    if not recommendations or 'overall' not in recommendations:
        return "Unable to generate recommendation - insufficient data"
    overall_rec = recommendations.get("overall", {})
    signal = overall_rec.get("signal", "HOLD")
    reason = overall_rec.get("reason", "No reason available")
    return f"{signal}: {reason}"

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


def analyze_ticker(ticker: str, period: str = "2y", interval: str = "1d", provider: str = "yf") -> TickerAnalysis:
    """Analyze ticker with enhanced technical analysis and recommendations, supporting both Yahoo Finance and Binance providers."""
    logger.info("Analyzing ticker: %s, period: %s, interval: %s, provider: %s", ticker, period, interval, provider)
    try:
        df = None
        fundamentals = None
        start_date, end_date = analyze_period_interval(period, interval)

        logger.debug("start_date: %s, end_date: %s", start_date, end_date)

        if provider.lower() == "yf":
            # Yahoo Finance
            downloader = YahooDataDownloader()
            df = downloader.download_data(ticker, interval, start_date, end_date)
            fundamentals = get_fundamentals(ticker)
        elif provider.lower() == "bnc":
            downloader = BinanceDataDownloader()
            df = downloader.download_historical_data(ticker, interval, start_date, end_date, save_to_csv=False)
            fundamentals = None
        else:
            raise ValueError(f"Unknown provider: {provider}")

        logger.info(f"Downloaded data for {ticker} from {start_date} to {end_date}")

        # Calculate technicals and update df with indicator columns
        df, technicals = calculate_technicals_from_df(df)

        # Create a temporary TickerAnalysis for charting, since we need to pass the object
        temp_analysis = TickerAnalysis(
            ticker=ticker.upper(),
            fundamentals=fundamentals,
            technicals=technicals,
            chart_image=None,
            recommendation=None,
            df=df
        )
        chart_image = generate_chart(temp_analysis)
        recommendation = generate_recommendation(technicals.recommendations if technicals else None)
        return TickerAnalysis(
            ticker=ticker.upper(),
            fundamentals=fundamentals,
            technicals=technicals,
            chart_image=chart_image,
            recommendation=recommendation,
            df=df
        )
    except Exception as e:
        logger.error("Error in analyze_ticker: %s", e, exc_info=True)
        raise

def format_recommendation(recommendations, name: str) -> str:
    if recommendations:
        rec = recommendations.get(name, {})
        return f" - <b>{rec.get('signal', 'HOLD')}</b> - {rec.get('reason', '')}"
    return None

def analyze_and_respond(ticker_requests):
    """
    ticker_requests: list of dicts, each with keys: ticker, provider, period, interval
    email_flag: bool, whether to collect email_body for email sending
    format_comprehensive_analysis: function to format HTML for email (pass from combine.py)
    Returns: (actions, email_body)
        actions: list of dicts for Telegram (type: 'text' or 'photo')
        email_body: list of HTML strings (or None if email_flag is False)
    """
    import tempfile
    actions = []

    for req in ticker_requests:
        ticker = req['ticker']
        provider = req.get('provider', 'yf')
        period = req.get('period', '2y')
        interval = req.get('interval', '1d')
        try:
            result = analyze_ticker(ticker, period=period, interval=interval, provider=provider)
            # Format Telegram text
            technicals = result.technicals
            fundamentals = result.fundamentals
            recommendations = getattr(technicals, 'recommendations', {}) if technicals else {}

            # Compose per-indicator signals
            text = (
                f"📈 <b>{result.ticker}</b> - {getattr(fundamentals, 'company_name', 'Unknown')}\n\n"
                f"💵 Price: ${getattr(fundamentals, 'current_price', 0.0):.2f}\n"
                f"🏦 P/E: {getattr(fundamentals, 'pe_ratio', 0.0):.2f}, Forward P/E: {getattr(fundamentals, 'forward_pe', 0.0):.2f}\n"
                f"💸 Market Cap: ${(getattr(fundamentals, 'market_cap', 0.0)/1e9):.2f}B\n"
                f"📊 EPS: ${getattr(fundamentals, 'earnings_per_share', 0.0):.2f}, Div Yield: {(getattr(fundamentals, 'dividend_yield', 0.0)*100):.2f}%\n\n"
                f"📉 Technical Analysis:\n"
                f"RSI: {getattr(technicals, 'rsi', 0.0):.2f}{format_recommendation(recommendations, 'rsi')}\n"
                f"Stochastic %K: {getattr(technicals, 'stoch_k', 0.0):.2f}, %D: {getattr(technicals, 'stoch_d', 0.0):.2f}{format_recommendation(recommendations, 'stochastic')}\n"
                f"ADX: {getattr(technicals, 'adx', 0.0):.2f}, +DI: {getattr(technicals, 'plus_di', 0.0):.2f}, -DI: {getattr(technicals, 'minus_di', 0.0):.2f}{format_recommendation(recommendations, 'adx')}\n"
                f"OBV: {getattr(technicals, 'obv', 0.0):.0f}{format_recommendation(recommendations, 'obv')}\n"
                f"ADR: {getattr(technicals, 'adr', 0.0):.2f}, Avg ADR: {getattr(technicals, 'avg_adr', 0.0):.2f}{format_recommendation(recommendations, 'adr')}\n"
                f"MA(50): ${getattr(technicals, 'sma_50', 0.0):.2f}\n"
                f"MA(200): ${getattr(technicals, 'sma_200', 0.0):.2f}\n"
                f"MACD: {getattr(technicals, 'macd', 0.0):.4f}, Signal: {getattr(technicals, 'macd_signal', 0.0):.4f}, Hist: {getattr(technicals, 'macd_histogram', 0.0):.4f}{format_recommendation(recommendations, 'macd')}\n"
                f"Trend: {getattr(technicals, 'trend', '-')}\n\n"
                f"📊 Bollinger Bands:{format_recommendation(recommendations, 'bollinger')}\n"
                f"Upper: ${getattr(technicals, 'bb_upper', 0.0):.2f}\n"
                f"Middle: ${getattr(technicals, 'bb_middle', 0.0):.2f}\n"
                f"Lower: ${getattr(technicals, 'bb_lower', 0.0):.2f}\n"
                f"Width: {getattr(technicals, 'bb_width', 0.0):.4f}\n\n"
                f"🎯 Overall: {format_recommendation(recommendations, 'overall')}"
            )
            # Save chart to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_file.write(result.chart_image)
                temp_file.flush()
                actions.append({
                    "type": "photo",
                    "file": temp_file.name,
                    "caption": text
                })
        except Exception as e:
            actions.append({
                "type": "text",
                "content": f"⚠️ Error analyzing {ticker}:\nPlease check if the ticker symbol is correct and try again.\nReason: {e}"
            })
    return actions
