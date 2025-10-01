"""
Unified recommendation engine for indicators (tech + fundamentals).

- Keeps the same indicator coverage and keys as your previous engine:
  TECH:  RSI, MACD, BB_UPPER/BB_MIDDLE/BB_LOWER, SMA_FAST/SMA_SLOW, EMA_FAST/EMA_SLOW,
         ADX, PLUS_DI, MINUS_DI, STOCH_K, STOCH_D, OBV, ADR, CCI, MFI, WILLIAMS_R, ROC, ATR
  FUND:  PE_RATIO, FORWARD_PE, PB_RATIO, PS_RATIO, PEG_RATIO, ROE, ROA, DEBT_TO_EQUITY,
         CURRENT_RATIO, QUICK_RATIO, OPERATING_MARGIN, PROFIT_MARGIN, REVENUE_GROWTH,
         NET_INCOME_GROWTH, FREE_CASH_FLOW, DIVIDEND_YIELD, PAYOUT_RATIO

- Cleaner structure:
  * Pure, small rule functions (no I/O)
  * Central thresholds
  * Consistent wrappers for context-dependent indicators
  * Stable Recommendation objects

- Public API:
  * get_recommendation(indicator: str, value: float, context: Dict | None) -> Recommendation
  * get_composite_recommendation(indicator_set: IndicatorSet) -> CompositeRecommendation
  * get_legacy_recommendation(...) -> Tuple[str, str] (kept for convenience)
"""

from __future__ import annotations

from typing import Dict, Tuple, Any
from dataclasses import dataclass

from src.model.indicators import (
    Recommendation, RecommendationType, IndicatorCategory,
    IndicatorSet, CompositeRecommendation
)
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

# ------------------------------ Engine ------------------------------

class RecommendationEngine:
    """
    Unified recommendation engine for all indicators.
    Public entrypoints:
      - get_recommendation()
      - get_legacy_recommendation()
      - get_composite_recommendation()
    """

    def __init__(self) -> None:
        # map tech keys -> wrappers
        self._tech_map = {
            "RSI":        lambda v, ctx: rule_rsi(v),
            "MACD":       self._wrap_macd,
            "BB_UPPER":   self._wrap_bbands,
            "BB_MIDDLE":  self._wrap_bbands,
            "BB_LOWER":   self._wrap_bbands,
            "SMA_FAST":   self._wrap_ma,
            "SMA_SLOW":   self._wrap_ma,
            "EMA_FAST":   self._wrap_ma,
            "EMA_SLOW":   self._wrap_ma,
            "ADX":        self._wrap_adx,
            "PLUS_DI":    self._wrap_di_simple,
            "MINUS_DI":   self._wrap_di_simple,
            "STOCH_K":    self._wrap_stoch,
            "STOCH_D":    self._wrap_stoch,
            "OBV":        self._wrap_obv,
            "ADR":        self._wrap_adr,
            "CCI":        lambda v, ctx: rule_cci(v),
            "MFI":        lambda v, ctx: rule_mfi(v),
            "WILLIAMS_R": lambda v, ctx: rule_williams_r(v),
            "ROC":        lambda v, ctx: rule_roc(v),
            "ATR":        lambda v, ctx: rule_atr(v),
        }

        # map fund keys -> functions
        self._fund_map = {
            "PE_RATIO":         f_pe,
            "FORWARD_PE":       f_pe,
            "PB_RATIO":         f_pb,
            "PS_RATIO":         f_ps,
            "PEG_RATIO":        f_peg,
            "ROE":              f_roe,
            "ROA":              f_roa,
            "DEBT_TO_EQUITY":   f_de_ratio,
            "CURRENT_RATIO":    f_current_ratio,
            "QUICK_RATIO":      f_quick_ratio,
            "OPERATING_MARGIN": f_margin,
            "PROFIT_MARGIN":    f_margin,
            "REVENUE_GROWTH":   f_growth,
            "NET_INCOME_GROWTH":f_growth,
            "FREE_CASH_FLOW":   f_fcf,
            "DIVIDEND_YIELD":   f_div_yield,
            "PAYOUT_RATIO":     f_payout,
        }

    # ---- Public API ---------------------------------------------------------

    def get_recommendation(self, indicator: str, value: float, context: Dict[str, Any] | None = None) -> Recommendation:
        try:
            if indicator in self._tech_map:
                rec, conf, reason = self._tech_map[indicator](value, context or {})
                cat = IndicatorCategory.TECHNICAL
            elif indicator in self._fund_map:
                rec, conf, reason = self._fund_map[indicator](value)
                cat = IndicatorCategory.FUNDAMENTAL
            else:
                rec, conf, reason = RecommendationType.HOLD, 0.5, f"No rule for {indicator}"
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
            for name, ind in all_ind.items():
                rec = ind.recommendation.recommendation
                score = ind.recommendation.confidence if rec in (RecommendationType.BUY, RecommendationType.STRONG_BUY) else \
                        -ind.recommendation.confidence if rec in (RecommendationType.SELL, RecommendationType.STRONG_SELL) else 0.0
                if score != 0.0: contrib.append(name)
                (tech_scores if ind.category == IndicatorCategory.TECHNICAL else fund_scores).append(score)

            tech = sum(tech_scores) / len(tech_scores) if tech_scores else 0.0
            fund = sum(fund_scores) / len(fund_scores) if fund_scores else 0.0
            comp = tech * 0.4 + fund * 0.6

            if comp >= 0.30: rec, why = RecommendationType.STRONG_BUY, f"Strong positives from {len(contrib)} indicators"
            elif comp >= 0.10: rec, why = RecommendationType.BUY, f"Positives from {len(contrib)} indicators"
            elif comp <= -0.30: rec, why = RecommendationType.STRONG_SELL, f"Strong negatives from {len(contrib)} indicators"
            elif comp <= -0.10: rec, why = RecommendationType.SELL, f"Negatives from {len(contrib)} indicators"
            else: rec, why = RecommendationType.HOLD, f"Mixed signals from {len(contrib)} indicators"

            return CompositeRecommendation(
                recommendation=rec,
                confidence=abs(comp),
                reasoning=why,
                contributing_indicators=contrib,
                technical_score=tech,
                fundamental_score=fund,
                composite_score=comp,
            )
        except Exception as e:
            _logger.exception("Composite recommendation error: %s", e)
            return CompositeRecommendation(
                recommendation=RecommendationType.HOLD,
                confidence=0.0,
                reasoning=str(e),
                contributing_indicators=[],
            )

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
