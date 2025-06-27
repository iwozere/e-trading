from src.screener.telegram.chart import generate_price_chart
from src.screener.telegram.fundamentals import get_fundamentals
from src.screener.telegram.models import (Fundamentals, Technicals,
                                          TickerAnalysis)
from src.screener.telegram.technicals import calculate_technicals


def generate_recommendation(technicals: Technicals) -> str:
    """Generate trading recommendation based on technical indicators."""
    if technicals.rsi > 70:
        return "Overbought - Consider selling"
    elif technicals.rsi < 30:
        return "Oversold - Consider buying"
    elif technicals.trend == "Uptrend":
        return "Bullish - Consider buying"
    elif technicals.trend == "Downtrend":
        return "Bearish - Consider selling"
    else:
        return "Neutral - Hold or wait for better entry"


def analyze_ticker(ticker: str) -> TickerAnalysis:
    fundamentals_data = get_fundamentals(ticker)
    technicals_data = calculate_technicals(ticker)
    chart_image = generate_price_chart(ticker)

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

    recommendation = generate_recommendation(technicals)

    return TickerAnalysis(
        ticker=ticker.upper(),
        fundamentals=fundamentals,
        technicals=technicals,
        chart_image=chart_image,
        recommendation=recommendation,
    )
