"""Fundamental signal scoring — §6.2 of the P05 spec."""

from typing import Any, Dict, Tuple

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def score_fundamentals(
    fundamentals: Any | None,
    sector_medians: Dict[str, Dict[str, float]],
    weights: Dict[str, int] | None = None,
) -> Tuple[float, Dict[str, object]]:
    """
    Score equity fundamentals per spec §6.2.

    Args:
        fundamentals: Fundamentals dataclass instance (or dict), or None.
        sector_medians: Per-sector median PE ratios from build_sector_medians().
        weights: Weight overrides. Defaults to spec values.

    Returns:
        Tuple of (total_score, signal_breakdown). Returns (0.0, {}) when
        fundamentals is None or unavailable.
    """
    from src.ml.pipeline.p05_ai_selector.config import FUNDAMENTAL_WEIGHTS

    if fundamentals is None:
        return (0.0, {})

    w = weights or FUNDAMENTAL_WEIGHTS

    def _get(attr: str) -> float | None:
        if isinstance(fundamentals, dict):
            return fundamentals.get(attr)
        return getattr(fundamentals, attr, None)

    pe_ratio: float | None = _get("pe_ratio")
    profit_margin: float | None = _get("profit_margin")
    debt_to_equity: float | None = _get("debt_to_equity")
    revenue_growth: float | None = _get("revenue_growth")
    dividend_yield: float | None = _get("dividend_yield")
    sector: str | None = _get("sector") or "Unknown"

    breakdown: Dict[str, object] = {
        "pe_ratio": pe_ratio,
        "profit_margin": profit_margin,
        "debt_to_equity": debt_to_equity,
        "revenue_growth": revenue_growth,
        "dividend_yield": dividend_yield,
        "value_signal": False,
        "quality_signal": False,
        "safety_signal": False,
        "growth_signal": False,
        "dividend_signal": False,
    }

    score = 0.0

    # Value: P/E < sector median
    if pe_ratio is not None and pe_ratio > 0:
        sector_pe = sector_medians.get(str(sector), {}).get("median_pe")
        if sector_pe is not None and pe_ratio < sector_pe:
            score += w.get("value", 10)
            breakdown["value_signal"] = True

    # Quality: net profit margin > 15 %
    if profit_margin is not None:
        # revenue_growth from DataManager may be a fraction (0.15) or percent (15)
        margin_val = profit_margin * 100 if abs(profit_margin) <= 1.0 else profit_margin
        if margin_val > 15.0:
            score += w.get("quality", 10)
            breakdown["quality_signal"] = True

    # Safety: debt-to-equity < 1.5
    if debt_to_equity is not None and 0 <= debt_to_equity < 1.5:
        score += w.get("safety", 5)
        breakdown["safety_signal"] = True

    # Growth: revenue YoY > 10 %
    if revenue_growth is not None:
        growth_val = revenue_growth * 100 if abs(revenue_growth) <= 1.0 else revenue_growth
        if growth_val > 10.0:
            score += w.get("growth", 10)
            breakdown["growth_signal"] = True

    # Dividend: yield > 0
    if dividend_yield is not None and dividend_yield > 0:
        score += w.get("dividend", 3)
        breakdown["dividend_signal"] = True

    return (score, breakdown)


def build_sector_medians(
    all_fundamentals: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-sector median P/E from the full candidate set.

    Args:
        all_fundamentals: Mapping of ticker → Fundamentals (or dict).

    Returns:
        Dict of {sector: {"median_pe": float}}.
    """
    sector_pes: Dict[str, list] = {}

    for _ticker, fund in all_fundamentals.items():
        if fund is None:
            continue

        def _get(attr: str) -> float | None:
            if isinstance(fund, dict):
                return fund.get(attr)
            return getattr(fund, attr, None)

        pe = _get("pe_ratio")
        sector = _get("sector") or "Unknown"
        if pe is not None and pe > 0:
            sector_pes.setdefault(str(sector), []).append(float(pe))

    result: Dict[str, Dict[str, float]] = {}
    for sector, pes in sector_pes.items():
        if pes:
            sorted_pes = sorted(pes)
            mid = len(sorted_pes) // 2
            median = (sorted_pes[mid - 1] + sorted_pes[mid]) / 2 if len(sorted_pes) % 2 == 0 else sorted_pes[mid]
            result[sector] = {"median_pe": round(median, 2)}

    return result
