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


def analyze_ticker(ticker: str) -> TickerAnalysis:
    """Analyze ticker with enhanced technical analysis and recommendations"""
    try:
        # Get fundamental data
        fundamentals_data = get_fundamentals(ticker)
        
        # Get enhanced technical analysis with recommendations
        technicals_data = calculate_technicals(ticker)
        
        # Generate enhanced chart with all indicators
        chart_image = generate_enhanced_chart(ticker, technicals_data)
        
        # Create fundamentals object
        fundamentals = Fundamentals(
            ticker=ticker.upper(),
            company_name=fundamentals_data.get("company_name", "Unknown"),
            current_price=fundamentals_data.get("current_price", 0.0),
            market_cap=fundamentals_data.get("market_cap", 0.0),
            pe_ratio=fundamentals_data.get("pe_ratio", 0.0),
            forward_pe=fundamentals_data.get("forward_pe", 0.0),
            dividend_yield=fundamentals_data.get("dividend_yield", 0.0),
            earnings_per_share=fundamentals_data.get("earnings_per_share", 0.0),
        )

        # Create technicals object with enhanced data
        technicals = Technicals(
            rsi=technicals_data.get("rsi", 0.0),
            sma_50=technicals_data.get("sma_50", 0.0),
            sma_200=technicals_data.get("sma_200", 0.0),
            macd_signal=technicals_data.get("macd_signal", 0.0),
            trend=technicals_data.get("trend", "Unknown"),
            bb_upper=technicals_data.get("bb_upper", 0.0),
            bb_middle=technicals_data.get("bb_middle", 0.0),
            bb_lower=technicals_data.get("bb_lower", 0.0),
            bb_width=technicals_data.get("bb_width", 0.0),
        )

        # Generate recommendation using enhanced technical analysis
        recommendation = generate_recommendation(technicals_data)

        return TickerAnalysis(
            ticker=ticker.upper(),
            fundamentals=fundamentals,
            technicals=technicals,
            chart_image=chart_image,
            recommendation=recommendation,
        )
        
    except Exception as e:
        # Return a basic analysis with error information
        fundamentals = Fundamentals(
            ticker=ticker.upper(),
            company_name="Error",
            current_price=0.0,
            market_cap=0.0,
            pe_ratio=0.0,
            forward_pe=0.0,
            dividend_yield=0.0,
            earnings_per_share=0.0,
        )
        
        technicals = Technicals(
            rsi=0.0,
            sma_50=0.0,
            sma_200=0.0,
            macd_signal=0.0,
            trend="Error",
            bb_upper=0.0,
            bb_middle=0.0,
            bb_lower=0.0,
            bb_width=0.0,
        )
        
        # Generate error chart
        chart_image = generate_enhanced_chart(ticker)
        
        return TickerAnalysis(
            ticker=ticker.upper(),
            fundamentals=fundamentals,
            technicals=technicals,
            chart_image=chart_image,
            recommendation=f"Error analyzing {ticker}: {str(e)}",
        )


def format_comprehensive_analysis(ticker: str, technicals_data: dict, fundamentals_data: dict) -> str:
    """Format comprehensive analysis for email with all recommendations"""
    if not technicals_data:
        return f"❌ Unable to analyze {ticker}"
    
    # Get technical analysis formatted text and convert to HTML
    technical_text = format_technical_analysis(ticker, technicals_data)
    technical_text_html = technical_text.replace('\n', '<br>').replace('*', '').replace('  ', ' ')
    
    # Add fundamental information with <br> for line breaks
    fundamental_text = f"""
<b>📊 Fundamental Analysis: {ticker}</b><br>
<br>
💰 Current Price: ${fundamentals_data.get('current_price', 0.0):.2f}<br>
🏢 Company: {fundamentals_data.get('company_name', 'Unknown')}<br>
💸 Market Cap: ${(fundamentals_data.get('market_cap', 0.0)/1e9):.2f}B<br>
📈 P/E Ratio: {fundamentals_data.get('pe_ratio', 0.0):.2f}<br>
📊 Forward P/E: {fundamentals_data.get('forward_pe', 0.0):.2f}<br>
💵 EPS: ${fundamentals_data.get('earnings_per_share', 0.0):.2f}<br>
🎯 Dividend Yield: {(fundamentals_data.get('dividend_yield', 0.0)*100):.2f}%<br>
"""
    
    return fundamental_text + "<br>" + technical_text_html
