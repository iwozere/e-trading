"""
Data models for Telegram bot integration.

Includes:
- Fundamentals and technicals data structures
- Ticker analysis result encapsulation
- Command specification and parsing models
"""
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field

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
    provider: str
    period: str
    interval: str
    fundamentals: Fundamentals
    technicals: Technicals
    chart_image: bytes
    ohlcv: Optional[object] = None  # DataFrame with pricing info
    error: Optional[str] = None


@dataclass
class CommandSpec:
    parameters: Dict[str, Type]  # param_name: type
    defaults: Dict[str, Any] = field(default_factory=dict)
    positional: List[str] = field(default_factory=list)  # names for positional args (e.g., tickers)

@dataclass
class ParsedCommand:
    command: str
    args: Dict[str, Any] = field(default_factory=dict)
    positionals: List[Any] = field(default_factory=list)
    raw_args: List[str] = field(default_factory=list)
    extra_flags: Dict[str, Any] = field(default_factory=dict)
