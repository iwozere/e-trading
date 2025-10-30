"""
Unified recommendation engine for indicators (tech + fundamentals).

This module consolidates recommendation logic from multiple sources and provides
a comprehensive recommendation system for all technical and fundamental indicators.

Features:
- Complete coverage of all 23 technical indicators from legacy system
- Complete coverage of all 21 fundamental indicators from legacy system
- Context-aware recommendations for multi-output indicators
- Composite recommendations with confidence scoring
- Backward compatibility with legacy interfaces

Public API:
- get_recommendation(indicator: str, value: float, context: Dict | None) -> Recommendation
- get_composite_recommendation(indicator_set: IndicatorSet) -> CompositeRecommendation
- get_legacy_recommendation(...) -> Tuple[str, str] (kept for convenience)
"""

from __future__ import annotations

from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass

from src.indicators.models import (
    Recommendation, RecommendationType, IndicatorCategory,
    IndicatorSet, CompositeRecommendation
)
from src.indicators.registry import get_canonical_name, get_indicator_meta
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# ------------------------------ Helpers & thresholds ------------------------------

def _is_none(*vals) -> bool:
    return any(v is None for v in vals)

def _pct(a: float, b: float) -> float:
    return (a - b) / b * 100 if (a is not None and b not in (None, 0)) else 0.0

@dataclass(frozen=True)
class TechThresholds:
    rsi_buy1: float = 40.0
    rsi_buy2: float = 30.0
    rsi_sell1: float = 60.0
    rsi_sell2: float = 70.0
    adx_trend: float = 25.0
    di_strong: float = 25.0
    di_weak: float = 15.0
    sma_strong: float = 5.0
    sma_mod: float = 2.0
    stoch_os: float = 20.0
    stoch_ob: float = 80.0
    roc_strong: float = 2.0
    atr_high: float = 5.0
    atr_low: float = 1.0

@dataclass(frozen=True)
class FundThresholds:
    pe1: float = 15.0
    pe2: float = 25.0
    pe3: float = 35.0
    pb1: float = 1.5
    pb2: float = 3.0
    pb3: float = 5.0
    ps1: float = 1.0
    ps2: float = 3.0
    ps3: float = 5.0
    peg1: float = 1.0
    peg2: float = 1.5
    peg3: float = 2.0
    roe1: float = 0.15
    roe2: float = 0.10
    roe3: float = 0.05
    roa1: float = 0.05
    roa2: float = 0.03
    roa3: float = 0.01
    dte1: float = 0.5
    dte2: float = 1.0
    dte3: float = 2.0
    cr1: float = 2.0
    cr2: float = 1.5
    cr3: float = 1.0
    qr1: float = 1.0
    qr2: float = 0.8
    qr3: float = 0.5
    margin1: float = 0.15
    margin2: float = 0.10
    margin3: float = 0.05
    growth1: float = 0.10
    growth2: float = 0.05
    growth3: float = 0.02
    dy1: float = 4.0
    dy2: float = 2.0
    dy3: float = 1.0
    payout1: float = 0.50
    payout2: float = 0.75

T = TechThresholds()
F = FundThresholds()

# ------------------------------ Technical rules ------------------------------

def rule_rsi(value: float) -> Tuple[RecommendationType, float, str]:
    if value is None:
        return RecommendationType.HOLD, 0.3, "No RSI"
    if value < T.rsi_buy2:  return RecommendationType.STRONG_BUY, 0.9, "RSI < 30 (oversold)"
    if value < T.rsi_buy1:  return RecommendationType.BUY, 0.7, "RSI < 40"
    if value > T.rsi_sell2: return RecommendationType.STRONG_SELL, 0.9, "RSI > 70 (overbought)"
    if value > T.rsi_sell1: return RecommendationType.SELL, 0.7, "RSI > 60"
    return RecommendationType.HOLD, 0.5, "RSI neutral"

def rule_bbands(close: float, upper: float, middle: float, lower: float) -> Tuple[RecommendationType, float, str]:
    if _is_none(close, upper, middle, lower) or upper <= lower:
        return RecommendationType.HOLD, 0.5, "Invalid Bollinger context"
    band_pos = (close - lower) / (upper - lower)
    if close <= lower: return RecommendationType.STRONG_BUY, 0.9, "Price at/below lower band"
    if close >= upper: return RecommendationType.STRONG_SELL, 0.9, "Price at/above upper band"
    if band_pos < 0.2:  return RecommendationType.BUY, 0.7, f"Near lower band ({band_pos:.0%})"
    if band_pos > 0.8:  return RecommendationType.SELL, 0.7, f"Near upper band ({band_pos:.0%})"
    if band_pos < 0.4:  return RecommendationType.BUY, 0.6, "Lower range"
    if band_pos > 0.6:  return RecommendationType.SELL, 0.6, "Upper range"
    return RecommendationType.HOLD, 0.5, "Middle range"

def rule_macd(macd: float, signal: float, hist: float) -> Tuple[RecommendationType, float, str]:
    if _is_none(macd, signal, hist):
        return RecommendationType.HOLD, 0.5, "Insufficient MACD"
    if macd > signal and hist > 0:
        return (RecommendationType.STRONG_BUY if hist > 0.5 else RecommendationType.BUY,
                0.7, "Bullish crossover")
    if macd < signal and hist < 0:
        return (RecommendationType.STRONG_SELL if hist < -0.5 else RecommendationType.SELL,
                0.7, "Bearish crossover")
    return RecommendationType.HOLD, 0.5, "MACD neutral"

def rule_stoch(k: float, d: float) -> Tuple[RecommendationType, float, str]:
    if _is_none(k, d): return RecommendationType.HOLD, 0.5, "Insufficient Stochastic"
    if k < T.stoch_os and d < T.stoch_os: return RecommendationType.STRONG_BUY, 0.9, "K & D oversold"
    if k > T.stoch_ob and d > T.stoch_ob: return RecommendationType.STRONG_SELL, 0.9, "K & D overbought"
    if k < 30 and k > d: return RecommendationType.BUY, 0.7, "K> D in oversold"
    if k > 70 and k < d: return RecommendationType.SELL, 0.7, "K< D in overbought"
    return RecommendationType.HOLD, 0.5, "Stochastic neutral"

def rule_adx(adx: float, plus_di: float, minus_di: float) -> Tuple[RecommendationType, float, str]:
    if _is_none(adx, plus_di, minus_di): return RecommendationType.HOLD, 0.5, "Insufficient ADX/DI"
    if adx > T.adx_trend:
        if plus_di > minus_di: return RecommendationType.BUY, 0.8, f"Uptrend (ADX {adx:.1f})"
        return RecommendationType.SELL, 0.8, f"Downtrend (ADX {adx:.1f})"
    return RecommendationType.HOLD, 0.5, "Weak trend"

def rule_ma(price: float, ma: float, *, trend: str = "unknown",
            fast: float | None = None, slow: float | None = None) -> Tuple[RecommendationType, float, str]:
    if _is_none(price, ma): return RecommendationType.HOLD, 0.5, "No price/MA"
    dist = _pct(price, ma)
    # strong by distance + trend
    if dist >= T.sma_strong and trend == "up":   return RecommendationType.STRONG_BUY, 0.8, f"{dist:.1f}% above rising MA"
    if dist <= -T.sma_strong and trend == "down":return RecommendationType.STRONG_SELL, 0.8, f"{dist:.1f}% below falling MA"
    # moderate
    if dist > T.sma_mod and trend == "up":       return RecommendationType.BUY, 0.7, f"{dist:.1f}% above rising MA"
    if dist < -T.sma_mod and trend == "down":    return RecommendationType.SELL, 0.7, f"{dist:.1f}% below falling MA"
    # crossovers (optional)
    if fast is not None and slow is not None:
        if fast > slow and price > ma and dist <= T.sma_mod:
            return RecommendationType.BUY, 0.7, "Bullish MA crossover"
        if fast < slow and price < ma and dist >= -T.sma_mod:
            return RecommendationType.SELL, 0.7, "Bearish MA crossover"
    # weak bias by side
    return (RecommendationType.BUY if price > ma else RecommendationType.SELL,
            0.6, f"Price {'above' if price>ma else 'below'} MA ({dist:.1f}%)")

def rule_cci(cci: float) -> Tuple[RecommendationType, float, str]:
    if cci is None: return RecommendationType.HOLD, 0.5, "No CCI"
    if cci <= -100: return RecommendationType.BUY, 0.7, "CCI oversold"
    if cci >= 100:  return RecommendationType.SELL, 0.7, "CCI overbought"
    return RecommendationType.HOLD, 0.5, "CCI neutral"

def rule_mfi(mfi: float) -> Tuple[RecommendationType, float, str]:
    if mfi is None: return RecommendationType.HOLD, 0.5, "No MFI"
    if mfi <= 20: return RecommendationType.BUY, 0.7, "MFI oversold"
    if mfi >= 80: return RecommendationType.SELL, 0.7, "MFI overbought"
    return RecommendationType.HOLD, 0.5, "MFI neutral"

def rule_obv(obv: float, obv_prev: float | None = None) -> Tuple[RecommendationType, float, str]:
    if obv is None: return RecommendationType.HOLD, 0.5, "No OBV"
    if obv_prev is None: return (RecommendationType.BUY if obv > 0 else RecommendationType.HOLD, 0.5, "OBV magnitude")
    chg = obv - obv_prev
    if chg > 0: return RecommendationType.BUY, 0.6, "OBV rising (accumulation)"
    if chg < 0: return RecommendationType.SELL, 0.6, "OBV falling (distribution)"
    return RecommendationType.HOLD, 0.5, "OBV flat"

def rule_adr(adr: float, price: float | None = None) -> Tuple[RecommendationType, float, str]:
    if adr is None: return RecommendationType.HOLD, 0.5, "No ADR"
    if price:
        p = (adr / price) * 100
        if p > T.atr_high: return RecommendationType.HOLD, 0.7, f"High volatility ({p:.1f}%)"
        if p > 3:          return RecommendationType.HOLD, 0.6, f"Moderate volatility ({p:.1f}%)"
        return RecommendationType.HOLD, 0.5, f"Low volatility ({p:.1f}%)"
    return RecommendationType.HOLD, 0.5, f"ADR {adr:.2f}"

def rule_roc(roc: float) -> Tuple[RecommendationType, float, str]:
    if roc is None: return RecommendationType.HOLD, 0.5, "No ROC"
    if roc > T.roc_strong:  return RecommendationType.BUY, 0.7, f"Strong momentum (+{roc:.2f}%)"
    if roc < -T.roc_strong: return RecommendationType.SELL, 0.7, f"Strong momentum ({roc:.2f}%)"
    return RecommendationType.HOLD, 0.5, f"Neutral momentum ({roc:.2f}%)"

def rule_atr(atr: float) -> Tuple[RecommendationType, float, str]:
    if atr is None: return RecommendationType.HOLD, 0.5, "No ATR"
    if atr > T.atr_high: return RecommendationType.HOLD, 0.6, f"High volatility (ATR {atr:.2f})"
    if atr < T.atr_low:  return RecommendationType.HOLD, 0.6, f"Low volatility (ATR {atr:.2f})"
    return RecommendationType.HOLD, 0.5, f"Normal volatility (ATR {atr:.2f})"

def rule_williams_r(wr: float) -> Tuple[RecommendationType, float, str]:
    if wr is None: return RecommendationType.HOLD, 0.5, "No Williams %R"
    if wr > -20:  return RecommendationType.SELL, 0.8, "Overbought"
    if wr < -80:  return RecommendationType.BUY, 0.8, "Oversold"
    return RecommendationType.HOLD, 0.5, "Neutral"

# ------------------------------ Fundamental rules ------------------------------

def f_pe(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No PE"
    if v <= F.pe1: return RecommendationType.STRONG_BUY, 0.8, "Low P/E"
    if v <= F.pe2: return RecommendationType.BUY, 0.6, "Reasonable P/E"
    if v <= F.pe3: return RecommendationType.HOLD, 0.5, "Moderate P/E"
    return RecommendationType.SELL, 0.7, "High P/E"

def f_pb(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No PB"
    if v <= F.pb1: return RecommendationType.STRONG_BUY, 0.8, "Low P/B"
    if v <= F.pb2: return RecommendationType.BUY, 0.6, "Reasonable P/B"
    if v <= F.pb3: return RecommendationType.HOLD, 0.5, "Moderate P/B"
    return RecommendationType.SELL, 0.7, "High P/B"

def f_ps(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No PS"
    if v <= F.ps1: return RecommendationType.STRONG_BUY, 0.8, "Low P/S"
    if v <= F.ps2: return RecommendationType.BUY, 0.6, "Reasonable P/S"
    if v <= F.ps3: return RecommendationType.HOLD, 0.5, "Moderate P/S"
    return RecommendationType.SELL, 0.7, "High P/S"

def f_peg(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No PEG"
    if v <= F.peg1: return RecommendationType.STRONG_BUY, 0.8, "Low PEG"
    if v <= F.peg2: return RecommendationType.BUY, 0.6, "Reasonable PEG"
    if v <= F.peg3: return RecommendationType.HOLD, 0.5, "Moderate PEG"
    return RecommendationType.SELL, 0.7, "High PEG"

def f_roe(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No ROE"
    if v >= F.roe1: return RecommendationType.STRONG_BUY, 0.8, "High ROE"
    if v >= F.roe2: return RecommendationType.BUY, 0.6, "Good ROE"
    if v >= F.roe3: return RecommendationType.HOLD, 0.5, "Moderate ROE"
    return RecommendationType.SELL, 0.7, "Low ROE"

def f_roa(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No ROA"
    if v >= F.roa1: return RecommendationType.STRONG_BUY, 0.8, "High ROA"
    if v >= F.roa2: return RecommendationType.BUY, 0.6, "Good ROA"
    if v >= F.roa3: return RecommendationType.HOLD, 0.5, "Moderate ROA"
    return RecommendationType.SELL, 0.7, "Low ROA"

def f_de_ratio(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No D/E"
    if v <= F.dte1: return RecommendationType.STRONG_BUY, 0.8, "Low leverage"
    if v <= F.dte2: return RecommendationType.BUY, 0.6, "Moderate leverage"
    if v <= F.dte3: return RecommendationType.HOLD, 0.5, "High leverage"
    return RecommendationType.SELL, 0.7, "Very high leverage"

def f_current_ratio(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No Current Ratio"
    if v >= F.cr1: return RecommendationType.STRONG_BUY, 0.8, "High liquidity"
    if v >= F.cr2: return RecommendationType.BUY, 0.6, "Good liquidity"
    if v >= F.cr3: return RecommendationType.HOLD, 0.5, "Adequate liquidity"
    return RecommendationType.SELL, 0.7, "Low liquidity"

def f_quick_ratio(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No Quick Ratio"
    if v >= F.qr1: return RecommendationType.STRONG_BUY, 0.8, "Strong quick liquidity"
    if v >= F.qr2: return RecommendationType.BUY, 0.6, "Good quick liquidity"
    if v >= F.qr3: return RecommendationType.HOLD, 0.5, "Adequate quick liquidity"
    return RecommendationType.SELL, 0.7, "Low quick liquidity"

def f_margin(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No margin"
    if v >= F.margin1: return RecommendationType.STRONG_BUY, 0.8, "High margin"
    if v >= F.margin2: return RecommendationType.BUY, 0.6, "Good margin"
    if v >= F.margin3: return RecommendationType.HOLD, 0.5, "Moderate margin"
    return RecommendationType.SELL, 0.7, "Low margin"

def f_growth(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No growth"
    if v >= F.growth1: return RecommendationType.STRONG_BUY, 0.8, "High growth"
    if v >= F.growth2: return RecommendationType.BUY, 0.6, "Good growth"
    if v >= F.growth3: return RecommendationType.HOLD, 0.5, "Moderate growth"
    return RecommendationType.SELL, 0.7, "Low growth"

def f_fcf(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No FCF"
    if v > 0:    return RecommendationType.STRONG_BUY, 0.8, "Positive FCF"
    if v >= -1_000_000: return RecommendationType.HOLD, 0.5, "Slight negative FCF"
    return RecommendationType.SELL, 0.7, "Negative FCF"

def f_div_yield(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No dividend yield"
    if v >= F.dy1: return RecommendationType.STRONG_BUY, 0.8, "High dividend yield"
    if v >= F.dy2: return RecommendationType.BUY, 0.6, "Good dividend yield"
    if v >= F.dy3: return RecommendationType.HOLD, 0.5, "Moderate dividend yield"
    return RecommendationType.SELL, 0.7, "Low dividend yield"

def f_payout(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No payout ratio"
    if v <= F.payout1: return RecommendationType.STRONG_BUY, 0.8, "Low payout (sustainable)"
    if v <= F.payout2: return RecommendationType.BUY, 0.6, "Moderate payout"
    if v <= 1.0:       return RecommendationType.HOLD, 0.5, "High payout (watch)"
    return RecommendationType.SELL, 0.7, "Very high payout (risk)"

def f_beta(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No beta"
    if v < 0.5: return RecommendationType.HOLD, 0.6, "Low volatility (defensive)"
    if v <= 1.2: return RecommendationType.BUY, 0.7, "Moderate volatility"
    if v <= 2.0: return RecommendationType.HOLD, 0.5, "High volatility"
    return RecommendationType.SELL, 0.6, "Very high volatility (risky)"

def f_market_cap(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No market cap"
    if v >= 200_000_000_000: return RecommendationType.BUY, 0.7, "Large cap (stable)"
    if v >= 10_000_000_000: return RecommendationType.BUY, 0.6, "Mid cap (growth potential)"
    if v >= 2_000_000_000: return RecommendationType.HOLD, 0.5, "Small cap (higher risk)"
    return RecommendationType.SELL, 0.6, "Micro cap (very risky)"

def f_enterprise_value(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No enterprise value"
    # EV is informational, neutral recommendation
    return RecommendationType.HOLD, 0.5, f"Enterprise value: {v:,.0f}"

def f_ev_ebitda(v: float) -> Tuple[RecommendationType, float, str]:
    if v is None: return RecommendationType.HOLD, 0.3, "No EV/EBITDA"
    if v <= 8: return RecommendationType.STRONG_BUY, 0.8, "Low EV/EBITDA (undervalued)"
    if v <= 15: return RecommendationType.BUY, 0.6, "Reasonable EV/EBITDA"
    if v <= 25: return RecommendationType.HOLD, 0.5, "Moderate EV/EBITDA"
    return RecommendationType.SELL, 0.7, "High EV/EBITDA (overvalued)"

# ------------------------------ Engine ------------------------------

class RecommendationEngine:
    """
    Unified recommendation engine for all indicators.

    This engine provides recommendations for both technical and fundamental indicators,
    with support for context-aware recommendations and composite scoring.
    """

    def __init__(self) -> None:
        # Technical indicator mappings (canonical names and legacy names)
        self._tech_map = {
            # Canonical names
            "rsi": lambda v, ctx: rule_rsi(v),
            "macd": self._wrap_macd,
            "bbands": self._wrap_bbands,
            "stoch": self._wrap_stoch,
            "adx": self._wrap_adx,
            "plus_di": self._wrap_di_simple,
            "minus_di": self._wrap_di_simple,
            "sma": self._wrap_ma,
            "ema": self._wrap_ma,
            "cci": lambda v, ctx: rule_cci(v),
            "roc": lambda v, ctx: rule_roc(v),
            "mfi": lambda v, ctx: rule_mfi(v),
            "williams_r": lambda v, ctx: rule_williams_r(v),
            "atr": lambda v, ctx: rule_atr(v),
            "obv": self._wrap_obv,
            "adr": self._wrap_adr,
            "aroon": self._wrap_aroon,
            "sar": self._wrap_sar,
            "ichimoku": self._wrap_ichimoku,
            "super_trend": self._wrap_super_trend,

            # Legacy names for backward compatibility
            "RSI": lambda v, ctx: rule_rsi(v),
            "MACD": self._wrap_macd,
            "MACD_SIGNAL": self._wrap_macd,
            "MACD_HISTOGRAM": self._wrap_macd,
            "BB_UPPER": self._wrap_bbands,
            "BB_MIDDLE": self._wrap_bbands,
            "BB_LOWER": self._wrap_bbands,
            "SMA_FAST": self._wrap_ma,
            "SMA_SLOW": self._wrap_ma,
            "SMA_50": self._wrap_ma,
            "SMA_200": self._wrap_ma,
            "EMA_FAST": self._wrap_ma,
            "EMA_SLOW": self._wrap_ma,
            "EMA_12": self._wrap_ma,
            "EMA_26": self._wrap_ma,
            "ADX": self._wrap_adx,
            "PLUS_DI": self._wrap_di_simple,
            "MINUS_DI": self._wrap_di_simple,
            "STOCH_K": self._wrap_stoch,
            "STOCH_D": self._wrap_stoch,
            "OBV": self._wrap_obv,
            "ADR": self._wrap_adr,
            "CCI": lambda v, ctx: rule_cci(v),
            "MFI": lambda v, ctx: rule_mfi(v),
            "WILLIAMS_R": lambda v, ctx: rule_williams_r(v),
            "ROC": lambda v, ctx: rule_roc(v),
            "ATR": lambda v, ctx: rule_atr(v),
        }

        # Fundamental indicator mappings (canonical names and legacy names)
        self._fund_map = {
            # Canonical names
            "pe_ratio": f_pe,
            "forward_pe": f_pe,
            "pb_ratio": f_pb,
            "ps_ratio": f_ps,
            "peg_ratio": f_peg,
            "roe": f_roe,
            "roa": f_roa,
            "debt_to_equity": f_de_ratio,
            "current_ratio": f_current_ratio,
            "quick_ratio": f_quick_ratio,
            "operating_margin": f_margin,
            "profit_margin": f_margin,
            "revenue_growth": f_growth,
            "net_income_growth": f_growth,
            "free_cash_flow": f_fcf,
            "dividend_yield": f_div_yield,
            "payout_ratio": f_payout,
            "beta": f_beta,
            "market_cap": f_market_cap,
            "enterprise_value": f_enterprise_value,
            "ev_to_ebitda": f_ev_ebitda,

            # Legacy names for backward compatibility
            "PE_RATIO": f_pe,
            "FORWARD_PE": f_pe,
            "PB_RATIO": f_pb,
            "PS_RATIO": f_ps,
            "PEG_RATIO": f_peg,
            "ROE": f_roe,
            "ROA": f_roa,
            "DEBT_TO_EQUITY": f_de_ratio,
            "CURRENT_RATIO": f_current_ratio,
            "QUICK_RATIO": f_quick_ratio,
            "OPERATING_MARGIN": f_margin,
            "PROFIT_MARGIN": f_margin,
            "REVENUE_GROWTH": f_growth,
            "NET_INCOME_GROWTH": f_growth,
            "FREE_CASH_FLOW": f_fcf,
            "DIVIDEND_YIELD": f_div_yield,
            "PAYOUT_RATIO": f_payout,
            "BETA": f_beta,
            "MARKET_CAP": f_market_cap,
            "ENTERPRISE_VALUE": f_enterprise_value,
            "EV_TO_EBITDA": f_ev_ebitda,
        }

    # ---- Public API ---------------------------------------------------------

    def get_recommendation(self, indicator: str, value: float, context: Dict[str, Any] | None = None) -> Recommendation:
        """
        Get recommendation for any indicator.

        Args:
            indicator: Indicator name (canonical or legacy)
            value: Indicator value
            context: Additional context for context-dependent indicators

        Returns:
            Recommendation object with type, confidence, and reasoning
        """
        try:
            # Try direct lookup first (for performance)
            if indicator in self._tech_map:
                rec, conf, reason = self._tech_map[indicator](value, context or {})
                cat = IndicatorCategory.TECHNICAL
            elif indicator in self._fund_map:
                rec, conf, reason = self._fund_map[indicator](value)
                cat = IndicatorCategory.FUNDAMENTAL
            else:
                # Try canonical name lookup
                canonical_name = get_canonical_name(indicator)
                meta = get_indicator_meta(canonical_name)

                if canonical_name in self._tech_map:
                    rec, conf, reason = self._tech_map[canonical_name](value, context or {})
                    cat = IndicatorCategory.TECHNICAL
                elif canonical_name in self._fund_map:
                    rec, conf, reason = self._fund_map[canonical_name](value)
                    cat = IndicatorCategory.FUNDAMENTAL
                elif meta:
                    # Use metadata to determine category
                    cat = IndicatorCategory.TECHNICAL if meta.kind == "tech" else IndicatorCategory.FUNDAMENTAL
                    rec, conf, reason = RecommendationType.HOLD, 0.5, f"No specific rule for {indicator}"
                else:
                    rec, conf, reason = RecommendationType.HOLD, 0.5, f"Unknown indicator: {indicator}"
                    cat = IndicatorCategory.TECHNICAL

            return Recommendation(
                recommendation=rec,
                confidence=conf,
                reason=reason,
                threshold_used=value,
                context=context,
            )
        except Exception as e:
            _logger.exception("Recommendation error for %s: %s", indicator, e)
            return Recommendation(
                recommendation=RecommendationType.HOLD,
                confidence=0.0,
                reason=str(e),
                threshold_used=value,
                context=context,
            )

    def get_legacy_recommendation(self, indicator: str, value: float, context: Dict[str, Any] | None = None) -> Tuple[str, str]:
        r = self.get_recommendation(indicator, value, context)
        rec_str = (
            "BUY"  if r.recommendation in (RecommendationType.STRONG_BUY, RecommendationType.BUY)
            else "SELL" if r.recommendation in (RecommendationType.STRONG_SELL, RecommendationType.SELL)
            else "HOLD"
        )
        return rec_str, r.reason

    def get_composite_recommendation(self, indicator_set: IndicatorSet) -> CompositeRecommendation:
        """
        Generate composite recommendation from all indicators in the set.

        Args:
            indicator_set: Collection of indicators with their recommendations

        Returns:
            CompositeRecommendation with overall assessment
        """
        try:
            all_ind = indicator_set.get_all_indicators()
            if not all_ind:
                return CompositeRecommendation(
                    recommendation=RecommendationType.HOLD,
                    confidence=0.0,
                    reasoning="No indicators available",
                    contributing_indicators=[],
                )

            tech_scores, fund_scores, contrib = [], [], []
            strong_signals = 0

            for name, ind in all_ind.items():
                rec = ind.recommendation.recommendation
                conf = ind.recommendation.confidence

                # Calculate weighted score based on recommendation type
                if rec == RecommendationType.STRONG_BUY:
                    score = conf * 1.0
                    strong_signals += 1
                elif rec == RecommendationType.BUY:
                    score = conf * 0.7
                elif rec == RecommendationType.STRONG_SELL:
                    score = -conf * 1.0
                    strong_signals += 1
                elif rec == RecommendationType.SELL:
                    score = -conf * 0.7
                else:
                    score = 0.0

                if score != 0.0:
                    contrib.append(name)

                # Categorize by indicator type
                if ind.category == IndicatorCategory.TECHNICAL:
                    tech_scores.append(score)
                else:
                    fund_scores.append(score)

            # Calculate category averages
            tech_avg = sum(tech_scores) / len(tech_scores) if tech_scores else 0.0
            fund_avg = sum(fund_scores) / len(fund_scores) if fund_scores else 0.0

            # Weight technical vs fundamental (can be adjusted)
            tech_weight = 0.6 if len(tech_scores) > len(fund_scores) else 0.4
            fund_weight = 1.0 - tech_weight

            composite_score = tech_avg * tech_weight + fund_avg * fund_weight

            # Boost confidence if we have strong signals
            confidence_boost = min(strong_signals * 0.1, 0.3)
            final_confidence = min(abs(composite_score) + confidence_boost, 1.0)

            # Determine overall recommendation with enhanced logic
            if composite_score >= 0.4:
                rec, why = RecommendationType.STRONG_BUY, f"Strong consensus from {len(contrib)} indicators ({strong_signals} strong signals)"
            elif composite_score >= 0.15:
                rec, why = RecommendationType.BUY, f"Positive consensus from {len(contrib)} indicators"
            elif composite_score <= -0.4:
                rec, why = RecommendationType.STRONG_SELL, f"Strong negative consensus from {len(contrib)} indicators ({strong_signals} strong signals)"
            elif composite_score <= -0.15:
                rec, why = RecommendationType.SELL, f"Negative consensus from {len(contrib)} indicators"
            else:
                rec, why = RecommendationType.HOLD, f"Mixed or neutral signals from {len(contrib)} indicators"

            return CompositeRecommendation(
                recommendation=rec,
                confidence=final_confidence,
                reasoning=why,
                contributing_indicators=contrib,
                technical_score=tech_avg,
                fundamental_score=fund_avg,
                composite_score=composite_score,
            )
        except Exception as e:
            _logger.exception("Composite recommendation error: %s", e)
            return CompositeRecommendation(
                recommendation=RecommendationType.HOLD,
                confidence=0.0,
                reasoning=str(e),
                contributing_indicators=[],
            )

    def get_contextual_recommendation(self, indicator: str, value: float, all_indicators: Dict[str, Any]) -> Recommendation:
        """
        Get recommendation with enhanced context from related indicators.

        Args:
            indicator: Primary indicator name
            value: Primary indicator value
            all_indicators: Dictionary of all available indicator values

        Returns:
            Recommendation with enhanced context consideration
        """
        # Build enhanced context based on indicator relationships
        context = {}
        canonical_name = get_canonical_name(indicator)

        # Add current price if available
        if "current_price" in all_indicators:
            context["current_price"] = all_indicators["current_price"]

        # Context for MACD family
        if canonical_name == "macd" or indicator.startswith("MACD"):
            for key in ["macd", "signal", "hist", "MACD", "MACD_SIGNAL", "MACD_HISTOGRAM"]:
                if key in all_indicators:
                    context[key.lower().replace("_", "_")] = all_indicators[key]

        # Context for Bollinger Bands
        if canonical_name == "bbands" or indicator.startswith("BB_"):
            for key in ["upper", "middle", "lower", "BB_UPPER", "BB_MIDDLE", "BB_LOWER"]:
                if key in all_indicators:
                    context[f"bb_{key.lower().replace('bb_', '')}"] = all_indicators[key]

        # Context for Stochastic
        if canonical_name == "stoch" or indicator.startswith("STOCH_"):
            for key in ["k", "d", "STOCH_K", "STOCH_D"]:
                if key in all_indicators:
                    context[f"stoch_{key.lower().replace('stoch_', '')}"] = all_indicators[key]

        # Context for ADX system
        if canonical_name in ["adx", "plus_di", "minus_di"] or indicator in ["ADX", "PLUS_DI", "MINUS_DI"]:
            for key in ["adx", "plus_di", "minus_di", "ADX", "PLUS_DI", "MINUS_DI"]:
                if key in all_indicators:
                    context[key.lower()] = all_indicators[key]

        # Context for moving averages
        if canonical_name in ["sma", "ema"] or "MA" in indicator:
            # Look for fast/slow MA pairs
            for key in all_indicators:
                if "FAST" in key or "SLOW" in key or "50" in key or "200" in key:
                    context[key.lower()] = all_indicators[key]

        # Context for Aroon
        if canonical_name == "aroon":
            for key in ["aroon_up", "aroon_down", "AROON_UP", "AROON_DOWN"]:
                if key in all_indicators:
                    context[key.lower()] = all_indicators[key]

        # Context for Ichimoku
        if canonical_name == "ichimoku":
            for key in ["tenkan", "kijun", "senkou_a", "senkou_b", "chikou"]:
                if key in all_indicators:
                    context[key] = all_indicators[key]

        # Context for Super Trend
        if canonical_name == "super_trend":
            for key in ["trend", "value"]:
                if f"super_trend_{key}" in all_indicators:
                    context[key] = all_indicators[f"super_trend_{key}"]

        return self.get_recommendation(indicator, value, context)

    def get_indicator_relationships(self) -> Dict[str, List[str]]:
        """
        Get dictionary of indicator relationships for contextual recommendations.

        Returns:
            Dictionary mapping indicators to their related indicators
        """
        return {
            "macd": ["macd", "signal", "hist"],
            "bbands": ["upper", "middle", "lower"],
            "stoch": ["k", "d"],
            "adx": ["adx", "plus_di", "minus_di"],
            "sma": ["sma_fast", "sma_slow", "sma_50", "sma_200"],
            "ema": ["ema_fast", "ema_slow", "ema_12", "ema_26"],
            "aroon": ["aroon_up", "aroon_down"],
            "ichimoku": ["tenkan", "kijun", "senkou_a", "senkou_b", "chikou"],
            "super_trend": ["value", "trend"]
        }

    # ---- Wrappers for context-dependent tech rules --------------------------

    def _wrap_bbands(self, value: float, ctx: Dict[str, Any]) -> Tuple[RecommendationType, float, str]:
        c, u, m, l = ctx.get("current_price"), ctx.get("bb_upper"), ctx.get("bb_middle"), ctx.get("bb_lower")
        return rule_bbands(c, u, m, l)

    def _wrap_macd(self, value: float, ctx: Dict[str, Any]) -> Tuple[RecommendationType, float, str]:
        # Standardize on 'hist' (what adapters return)
        hist = ctx.get("hist", ctx.get("macd_hist", ctx.get("macd_histogram")))
        signal = ctx.get("signal", ctx.get("macd_signal"))
        return rule_macd(value, signal, hist)

    def _wrap_stoch(self, value: float, ctx: Dict[str, Any]) -> Tuple[RecommendationType, float, str]:
        k = value if "STOCH_K" in ctx.get("name", "STOCH_K") else ctx.get("stoch_k", value)
        d = ctx.get("stoch_d", value) if k is value else value
        return rule_stoch(k, d)

    def _wrap_adx(self, value: float, ctx: Dict[str, Any]) -> Tuple[RecommendationType, float, str]:
        return rule_adx(value, ctx.get("plus_di"), ctx.get("minus_di"))

    def _wrap_ma(self, value: float, ctx: Dict[str, Any]) -> Tuple[RecommendationType, float, str]:
        return rule_ma(
            price=ctx.get("current_price"),
            ma=value,
            trend=ctx.get("ma_trend", "unknown"),
            fast=ctx.get("fast"),
            slow=ctx.get("slow"),
        )

    def _wrap_obv(self, value: float, ctx: Dict[str, Any]) -> Tuple[RecommendationType, float, str]:
        return rule_obv(value, ctx.get("obv_prev"))

    def _wrap_adr(self, value: float, ctx: Dict[str, Any]) -> Tuple[RecommendationType, float, str]:
        return rule_adr(value, ctx.get("current_price"))

    def _wrap_di_simple(self, value: float, ctx: Dict[str, Any]) -> Tuple[RecommendationType, float, str]:
        # Standalone DI heuristic (same behavior as previous engine)
        if value is None: return RecommendationType.HOLD, 0.5, "No DI"
        if value > T.di_strong: return RecommendationType.BUY, 0.7, f"Strong directional movement ({value:.1f})"
        if value < T.di_weak:   return RecommendationType.SELL, 0.7, f"Weak directional movement ({value:.1f})"
        return RecommendationType.HOLD, 0.5, f"Moderate directional movement ({value:.1f})"

    def _wrap_aroon(self, value: float, ctx: Dict[str, Any]) -> Tuple[RecommendationType, float, str]:
        """Wrapper for Aroon indicator."""
        aroon_up = ctx.get("aroon_up", value)
        aroon_down = ctx.get("aroon_down", 0)

        if aroon_up is None: return RecommendationType.HOLD, 0.5, "No Aroon data"

        if aroon_up > 70 and aroon_down < 30:
            return RecommendationType.BUY, 0.7, "Strong uptrend (Aroon Up > 70)"
        elif aroon_down > 70 and aroon_up < 30:
            return RecommendationType.SELL, 0.7, "Strong downtrend (Aroon Down > 70)"
        else:
            return RecommendationType.HOLD, 0.5, "Aroon signals neutral"

    def _wrap_sar(self, value: float, ctx: Dict[str, Any]) -> Tuple[RecommendationType, float, str]:
        """Wrapper for Parabolic SAR."""
        current_price = ctx.get("current_price")
        if value is None or current_price is None:
            return RecommendationType.HOLD, 0.5, "Insufficient SAR data"

        if current_price > value:
            return RecommendationType.BUY, 0.7, "Price above SAR (uptrend)"
        else:
            return RecommendationType.SELL, 0.7, "Price below SAR (downtrend)"

    def _wrap_ichimoku(self, value: float, ctx: Dict[str, Any]) -> Tuple[RecommendationType, float, str]:
        """Wrapper for Ichimoku Cloud."""
        current_price = ctx.get("current_price")
        tenkan = ctx.get("tenkan")
        kijun = ctx.get("kijun")
        senkou_a = ctx.get("senkou_a")
        senkou_b = ctx.get("senkou_b")

        if not all([current_price, tenkan, kijun, senkou_a, senkou_b]):
            return RecommendationType.HOLD, 0.5, "Insufficient Ichimoku data"

        # Determine cloud position
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        if current_price > cloud_top and tenkan > kijun:
            return RecommendationType.STRONG_BUY, 0.8, "Price above cloud with bullish signals"
        elif current_price < cloud_bottom and tenkan < kijun:
            return RecommendationType.STRONG_SELL, 0.8, "Price below cloud with bearish signals"
        elif current_price > cloud_top:
            return RecommendationType.BUY, 0.6, "Price above cloud"
        elif current_price < cloud_bottom:
            return RecommendationType.SELL, 0.6, "Price below cloud"
        else:
            return RecommendationType.HOLD, 0.5, "Price in cloud (neutral)"

    def _wrap_super_trend(self, value: float, ctx: Dict[str, Any]) -> Tuple[RecommendationType, float, str]:
        """Wrapper for Super Trend."""
        current_price = ctx.get("current_price")
        trend = ctx.get("trend", 1)

        if value is None or current_price is None:
            return RecommendationType.HOLD, 0.5, "Insufficient Super Trend data"

        if trend > 0:  # Uptrend
            return RecommendationType.BUY, 0.7, "Super Trend uptrend"
        else:  # Downtrend
            return RecommendationType.SELL, 0.7, "Super Trend downtrend"
