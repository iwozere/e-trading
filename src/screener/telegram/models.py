# ticker_bot/analyzer/models.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class Fundamentals:
    ticker: str
    company_name: str
    current_price: float
    market_cap: float
    pe_ratio: float
    forward_pe: float
    dividend_yield: float
    earnings_per_share: float


@dataclass
class Technicals:
    rsi: float
    sma_50: float
    sma_200: float
    macd_signal: float
    trend: str
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float  # (upper - lower) / middle


@dataclass
class TickerAnalysis:
    ticker: str
    fundamentals: Fundamentals
    technicals: Technicals
    chart_image: bytes
    recommendation: str = "Neutral"  # Default value

    # Resume
    recommendation: Optional[str]
