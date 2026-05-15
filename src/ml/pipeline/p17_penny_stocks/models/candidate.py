"""
P17 Candidate Model
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Candidate:
    """
    Penny stock candidate with all enriched data and computed scores.

    Fields are populated progressively as each agent runs.  Missing data
    is represented as None and is handled gracefully by the scoring agent.
    """

    # ── Identity ───────────────────────────────────────────────────────────
    ticker: str
    company_name: str = ""
    exchange: str = ""
    sector: str = ""
    industry: str = ""

    # ── Market snapshot ────────────────────────────────────────────────────
    price: float = 0.0
    market_cap: float = 0.0
    volume: float = 0.0
    avg_volume_30d: float = 0.0
    float_shares: float = 0.0
    shares_outstanding: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0
    institutional_pct: Optional[float] = None

    # ── Technical features ─────────────────────────────────────────────────
    relative_volume: float = 0.0
    price_5d_return: float = 0.0
    price_20d_return: float = 0.0
    price_60d_return: float = 0.0
    sma20: float = 0.0
    sma50: float = 0.0
    above_sma50: bool = False
    breakout_20d: bool = False
    breakout_50d: bool = False
    bb_squeeze: bool = False
    atr_pct: float = 0.0          # ATR as % of price; proxy for recent volatility
    obv_slope: float = 0.0        # OBV linear regression slope (positive = accumulation)
    accumulation_days: int = 0    # high-volume green days in last 20 sessions

    # ── Short squeeze ──────────────────────────────────────────────────────
    short_interest_pct_float: Optional[float] = None
    days_to_cover: Optional[float] = None
    finra_short_vol_ratio: Optional[float] = None   # daily short sale vol / total vol

    # ── Fundamentals ───────────────────────────────────────────────────────
    revenue_growth_yoy: Optional[float] = None
    gross_margin: Optional[float] = None
    total_cash: Optional[float] = None
    total_debt: Optional[float] = None
    cash_runway_months: Optional[float] = None
    operating_cashflow: Optional[float] = None
    fundamentals_stale: bool = False   # True when data is older than one quarter

    # ── Dilution ───────────────────────────────────────────────────────────
    dilution_penalty: float = 0.0
    dilution_signals: List[str] = field(default_factory=list)

    # ── Sub-scores (each normalized 0–100) ────────────────────────────────
    momentum_score: float = 0.0
    volume_score: float = 0.0
    technical_score: float = 0.0
    fundamentals_score: float = 0.0
    catalyst_score: float = 0.0       # Phase 1: defaults to 0 (no catalyst agent yet)
    short_squeeze_score: float = 0.0
    accumulation_score: float = 0.0

    # ── Final output ───────────────────────────────────────────────────────
    weighted_score: float = 0.0       # weighted sum before dilution deduction
    final_score: float = 0.0          # weighted_score minus dilution_penalty (floored at 0)
    tier: str = "W"                   # A | B | C | W
    explosive_candidate: bool = False
    signals: List[str] = field(default_factory=list)

    # ── Metadata ───────────────────────────────────────────────────────────
    data_as_of: Optional[datetime] = None
    run_date: str = ""

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Flat dictionary suitable for CSV/JSON export."""
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "exchange": self.exchange,
            "sector": self.sector,
            "industry": self.industry,
            "price": round(self.price, 2),
            "market_cap": int(self.market_cap) if self.market_cap else 0,
            "volume": int(self.volume) if self.volume else 0,
            "avg_volume_30d": int(self.avg_volume_30d) if self.avg_volume_30d else 0,
            "float_shares": int(self.float_shares) if self.float_shares else 0,
            "high_52w": round(self.high_52w, 2),
            "low_52w": round(self.low_52w, 2),
            "relative_volume": round(self.relative_volume, 2),
            "price_5d_return": round(self.price_5d_return, 4),
            "price_20d_return": round(self.price_20d_return, 4),
            "price_60d_return": round(self.price_60d_return, 4),
            "above_sma50": self.above_sma50,
            "breakout_20d": self.breakout_20d,
            "breakout_50d": self.breakout_50d,
            "bb_squeeze": self.bb_squeeze,
            "atr_pct": round(self.atr_pct, 4),
            "obv_slope": round(self.obv_slope, 4),
            "accumulation_days": self.accumulation_days,
            "short_interest_pct_float": self.short_interest_pct_float,
            "days_to_cover": self.days_to_cover,
            "finra_short_vol_ratio": self.finra_short_vol_ratio,
            "revenue_growth_yoy": self.revenue_growth_yoy,
            "gross_margin": self.gross_margin,
            "total_cash": self.total_cash,
            "total_debt": self.total_debt,
            "cash_runway_months": self.cash_runway_months,
            "dilution_penalty": self.dilution_penalty,
            "dilution_signals": "|".join(self.dilution_signals),
            "momentum_score": round(self.momentum_score, 1),
            "volume_score": round(self.volume_score, 1),
            "technical_score": round(self.technical_score, 1),
            "fundamentals_score": round(self.fundamentals_score, 1),
            "catalyst_score": round(self.catalyst_score, 1),
            "short_squeeze_score": round(self.short_squeeze_score, 1),
            "accumulation_score": round(self.accumulation_score, 1),
            "weighted_score": round(self.weighted_score, 1),
            "final_score": round(self.final_score, 1),
            "tier": self.tier,
            "explosive_candidate": self.explosive_candidate,
            "signals": "|".join(self.signals),
            "run_date": self.run_date,
        }

    def score_breakdown(self) -> Dict:
        """Score breakdown dict for the JSON report."""
        return {
            "momentum": round(self.momentum_score, 1),
            "volume": round(self.volume_score, 1),
            "technical": round(self.technical_score, 1),
            "fundamentals": round(self.fundamentals_score, 1),
            "catalyst": round(self.catalyst_score, 1),
            "short_squeeze": round(self.short_squeeze_score, 1),
            "accumulation": round(self.accumulation_score, 1),
            "weighted_sum": round(self.weighted_score, 1),
            "dilution_penalty": round(self.dilution_penalty, 1),
            "final": round(self.final_score, 1),
        }
