"""
Data models for Telegram bot integration.

Includes:
- Fundamentals and technicals data structures
- Ticker analysis result encapsulation
- Command specification and parsing models
- Fundamental screener data models
"""
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field

@dataclass
class Fundamentals:
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    dividend_yield: Optional[float] = None
    earnings_per_share: Optional[float] = None
    # Additional fields for comprehensive fundamental analysis
    price_to_book: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    revenue: Optional[float] = None
    revenue_growth: Optional[float] = None
    net_income: Optional[float] = None
    net_income_growth: Optional[float] = None
    free_cash_flow: Optional[float] = None
    operating_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    beta: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    shares_outstanding: Optional[float] = None
    float_shares: Optional[float] = None
    short_ratio: Optional[float] = None
    payout_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_sales: Optional[float] = None
    enterprise_value: Optional[float] = None
    enterprise_value_to_ebitda: Optional[float] = None
    # Data source information
    data_source: Optional[str] = None
    last_updated: Optional[str] = None
    # Track which provider supplied each value
    sources: Optional[Dict[str, str]] = field(default_factory=dict)

@dataclass
class Technicals:
    rsi: float
    sma_fast: float
    sma_slow: float
    ema_fast: float
    ema_slow: float
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
    cci: float
    roc: float
    mfi: float
    williams_r: float
    atr: float
    recommendations: Dict[str, Any]


@dataclass
class TickerAnalysis:
    """Encapsulates the result of a ticker analysis, including fundamentals, technicals, chart image, recommendation, and raw pricing DataFrame."""
    ticker: str
    provider: str
    period: str
    interval: str
    fundamentals: Optional[Fundamentals] = None
    technicals: Optional[Technicals] = None
    chart_image: Optional[bytes] = None
    ohlcv: Optional[object] = None  # DataFrame with pricing info
    error: Optional[str] = None
    current_price: Optional[float] = None
    change_percentage: Optional[float] = None


@dataclass
class DCFResult:
    """Discounted Cash Flow valuation result."""
    ticker: str
    fair_value: Optional[float] = None
    growth_rate: Optional[float] = None
    discount_rate: Optional[float] = None
    terminal_value: Optional[float] = None
    assumptions: Optional[Dict[str, float]] = None
    confidence_level: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ScreenerResult:
    """Result of fundamental screening for a single ticker."""
    ticker: str
    fundamentals: Optional[Fundamentals] = None
    technicals: Optional[Technicals] = None
    dcf_valuation: Optional[DCFResult] = None
    composite_score: Optional[float] = None
    screening_status: Optional[Dict[str, bool]] = None
    recommendation: Optional[str] = None  # "BUY", "HOLD", "SELL"
    reasoning: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ScreenerReport:
    """Complete fundamental screener report."""
    list_type: str
    total_tickers_processed: int
    total_tickers_with_data: int
    top_results: List[ScreenerResult]
    summary_stats: Optional[Dict[str, Any]] = None
    generated_at: Optional[str] = None
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
