# ticker_bot/analyzer/models.py

from dataclasses import dataclass
from typing import Optional, Dict


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
    macd: float
    macd_signal: float
    macd_histogram: float
    stoch_k: float
    stoch_d: float
    adx: float
    plus_di: float
    minus_di: float
    obv: float
    adr: float
    avg_adr: float
    trend: str
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    recommendations: Optional[Dict] = None


@dataclass
class TickerAnalysis:
    """Encapsulates the result of a ticker analysis, including fundamentals, technicals, chart image, recommendation, and raw pricing DataFrame."""
    ticker: str
    fundamentals: Fundamentals
    technicals: Technicals
    chart_image: bytes
    recommendation: Optional[str] = "Neutral"  # Default value
    df: Optional[object] = None  # DataFrame with pricing info
