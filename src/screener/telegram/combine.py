from src.screener.telegram.chart import generate_enhanced_chart
from src.screener.telegram.fundamentals import get_fundamentals
from src.screener.telegram.models import (Fundamentals, Technicals,
                                          TickerAnalysis)
from src.screener.telegram.technicals import calculate_technicals, format_technical_analysis


def generate_recommendation(technicals_data: dict) -> str:
    """Generate trading recommendation based on technical indicators."""
    if not technicals_data or not technicals_data.get("recommendations"):
        return "Unable to generate recommendation - insufficient data"

    overall_rec = technicals_data.get("recommendations", {}).get("overall", {})
    signal = overall_rec.get("signal", "HOLD")
    reason = overall_rec.get("reason", "No reason available")

    return f"{signal}: {reason}"


def analyze_ticker(ticker: str, period: str = "2y", interval: str = "1d") -> TickerAnalysis:
    """Analyze ticker with enhanced technical analysis and recommendations"""
    try:
        # Get fundamental data (now a dataclass)
        fundamentals = get_fundamentals(ticker)
        # Get enhanced technical analysis with recommendations (now a dataclass)
        technicals = calculate_technicals(ticker, period=period, interval=interval)
        # Generate enhanced chart with all indicators
        chart_image = generate_enhanced_chart(ticker, technicals)
        # Generate recommendation using enhanced technical analysis
        recommendation = generate_recommendation(technicals.recommendations if technicals else None)
        return TickerAnalysis(
            ticker=ticker.upper(),
            fundamentals=fundamentals,
            technicals=technicals,
            chart_image=chart_image,
            recommendation=recommendation
        )
    except Exception as e:
        print(f"Error in analyze_ticker: {e}")
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
