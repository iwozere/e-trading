"""
Data models for Telegram bot integration.

Includes:
- Fundamentals and technicals data structures
- Ticker analysis result encapsulation
- Command specification and parsing models
- Fundamental screener data models
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Type


@dataclass
class Fundamentals:
    ticker: str | None = None
    company_name: str | None = None
    current_price: float | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    dividend_yield: float | None = None
    earnings_per_share: float | None = None
    # Additional fields for comprehensive fundamental analysis
    price_to_book: float | None = None
    return_on_equity: float | None = None
    return_on_assets: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    revenue: float | None = None
    revenue_growth: float | None = None
    net_income: float | None = None
    net_income_growth: float | None = None
    free_cash_flow: float | None = None
    operating_margin: float | None = None
    profit_margin: float | None = None
    beta: float | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    exchange: str | None = None
    currency: str | None = None
    shares_outstanding: float | None = None
    float_shares: float | None = None
    short_ratio: float | None = None
    payout_ratio: float | None = None
    peg_ratio: float | None = None
    price_to_sales: float | None = None
    enterprise_value: float | None = None
    enterprise_value_to_ebitda: float | None = None
    # Data source information
    data_source: str | None = None
    last_updated: str | None = None
    # Track which provider supplied each value
    sources: Dict[str, str] | None = field(default_factory=dict)


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
    fundamentals: Fundamentals | None = None
    technicals: Technicals | None = None
    chart_image: bytes | None = None
    ohlcv: object | None = None  # DataFrame with pricing info
    error: str | None = None
    current_price: float | None = None
    change_percentage: float | None = None


@dataclass
class DCFResult:
    """Discounted Cash Flow valuation result."""

    ticker: str
    fair_value: float | None = None
    growth_rate: float | None = None
    discount_rate: float | None = None
    terminal_value: float | None = None
    assumptions: Dict[str, float] | None = None
    confidence_level: str | None = None
    error: str | None = None


@dataclass
class ScreenerResult:
    """Result of fundamental screening for a single ticker."""

    ticker: str
    fundamentals: Fundamentals | None = None
    technicals: Technicals | None = None
    dcf_valuation: DCFResult | None = None
    composite_score: float | None = None
    screening_status: Dict[str, bool] | None = None
    recommendation: str | None = None  # "BUY", "HOLD", "SELL"
    reasoning: str | None = None
    error: str | None = None


@dataclass
class ScreenerReport:
    """Complete fundamental screener report."""

    list_type: str
    total_tickers_processed: int
    total_tickers_with_data: int
    top_results: List[ScreenerResult]
    summary_stats: Dict[str, Any] | None = None
    generated_at: str | None = None
    error: str | None = None


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
