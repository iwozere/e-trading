from src.screener.telegram.chart import generate_enhanced_chart
from src.screener.telegram.fundamentals import get_fundamentals
from src.screener.telegram.models import (Fundamentals, Technicals,
                                          TickerAnalysis)
from src.screener.telegram.technicals import calculate_technicals_from_df, format_technical_analysis
from src.data.yahoo_data_downloader import YahooDataDownloader
from src.data.binance_data_downloader import BinanceDataDownloader
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


def generate_recommendation(technicals_data: dict) -> str:
    """Generate trading recommendation based on technical indicators."""
    if not technicals_data or not technicals_data.get("recommendations"):
        return "Unable to generate recommendation - insufficient data"

    overall_rec = technicals_data.get("recommendations", {}).get("overall", {})
    signal = overall_rec.get("signal", "HOLD")
    reason = overall_rec.get("reason", "No reason available")

    return f"{signal}: {reason}"


def analyze_ticker(ticker: str, period: str = "2y", interval: str = "1d", provider: str = "yf") -> TickerAnalysis:
    """Analyze ticker with enhanced technical analysis and recommendations, supporting both Yahoo Finance and Binance providers."""
    import pandas as pd
    try:
        df = None
        fundamentals = None
        if provider.lower() == "yf":
            # Yahoo Finance
            downloader = YahooDataDownloader()
            # Convert period/interval to start/end dates
            import datetime
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            if period.endswith("y"):
                years = int(period[:-1])
                start_date = (datetime.datetime.now() - datetime.timedelta(days=365*years)).strftime("%Y-%m-%d")
            elif period.endswith("m"):
                months = int(period[:-1])
                start_date = (datetime.datetime.now() - datetime.timedelta(days=30*months)).strftime("%Y-%m-%d")
            else:
                start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
            df = downloader.download_data(ticker, interval, start_date, end_date)
            fundamentals = get_fundamentals(ticker)
        elif provider.lower() == "bnc":
            downloader = BinanceDataDownloader()
            import datetime
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            if period.endswith("y"):
                years = int(period[:-1])
                start_date = (datetime.datetime.now() - datetime.timedelta(days=365*years)).strftime("%Y-%m-%d")
            elif period.endswith("m"):
                months = int(period[:-1])
                start_date = (datetime.datetime.now() - datetime.timedelta(days=30*months)).strftime("%Y-%m-%d")
            else:
                start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
            df = downloader.download_historical_data(ticker, interval, start_date, end_date, save_to_csv=False)
            fundamentals = None
        else:
            raise ValueError(f"Unknown provider: {provider}")
        technicals = calculate_technicals_from_df(df)
        chart_image = generate_enhanced_chart(ticker, technicals)
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


def format_comprehensive_analysis(ticker: str, technicals: Technicals, fundamentals: Fundamentals) -> str:
    """Format comprehensive analysis for email with all recommendations"""
    if not technicals:
        return f"❌ Unable to analyze {ticker}"

    # Technical analysis text (HTML)
    technical_text = format_technical_analysis(ticker, technicals)
    technical_text_html = technical_text.replace('\n', '<br>').replace('*', '').replace('  ', ' ')
    # Fundamental info (HTML)
    fundamental_text = f"""
<b>📊 Fundamental Analysis: {ticker}</b><br>
<br>
💰 Current Price: ${fundamentals.current_price:.2f}<br>
🏢 Company: {fundamentals.company_name}<br>
🏦 Market Cap: ${fundamentals.market_cap:,.0f}<br>
📈 P/E: {fundamentals.pe_ratio:.2f}, Forward P/E: {fundamentals.forward_pe:.2f}<br>
💸 EPS: ${fundamentals.earnings_per_share:.2f}, Div Yield: {fundamentals.dividend_yield * 100:.2f}%<br>
"""
    return fundamental_text + "<br>" + technical_text_html
