"""
P20 Kestrel — Sleeve A screen (Turnaround / Fallen Angels).

Implements §4.1 hard filters and §4.2.1 interim scoring
(REVISIONS_FEED_AVAILABLE=False renormalization).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.config import (
    REVISIONS_FEED_AVAILABLE,
    SLEEVE_A_DOSSIER_THRESHOLD,
    SLEEVE_A_MIN_ADV_USD,
    SLEEVE_A_PUSH_THRESHOLD,
)
from src.data.db.services.kestrel_service import KestrelService as _KestrelService

_kestrel = _KestrelService()
get_active_tickers = _kestrel.get_active_tickers
get_signals = _kestrel.get_signals
get_universe_row = _kestrel.get_universe_row
upsert_signals = _kestrel.upsert_signals
upsert_watchlist = _kestrel.upsert_watchlist
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_SLEEVE = "A"
_DRAWDOWN_MIN = -0.75
_DRAWDOWN_MAX = -0.40
_MCAP_MIN_USD = 500_000_000
_NET_DEBT_EBITDA_MAX = 3.0
_INTEREST_COVERAGE_MIN = 4.0
_REVENUE_YOY_MIN = -0.15
_GROSS_MARGIN_MIN = 0.0


def _passes_hard_filters(
    universe_row: Dict[str, Any],
    signals: Dict[str, Any],
) -> Optional[str]:
    """
    Check all §4.1 hard filters.

    Args:
        universe_row: Row from k20_universe.
        signals: Dict of signal_type → value for the ticker.

    Returns:
        None if passes, or a string describing the failing filter.
    """
    mcap = universe_row.get("mcap")
    adv_20d = universe_row.get("adv_20d") or signals.get("adv_20d")
    drawdown = signals.get("drawdown_from_2y_high")
    revenue_growth = universe_row.get("revenue_yoy_growth")
    gross_margin = universe_row.get("gross_margin")
    net_debt_ebitda = universe_row.get("net_debt_ebitda")
    interest_coverage = universe_row.get("interest_coverage")

    if not mcap or mcap < _MCAP_MIN_USD:
        return f"mcap_below_500M (${mcap:,.0f})" if mcap else "mcap_missing"

    if not adv_20d or adv_20d < SLEEVE_A_MIN_ADV_USD:
        return f"adv_below_10M (${adv_20d:,.0f})" if adv_20d else "adv_missing"

    if drawdown is None:
        return "drawdown_missing"
    if not (_DRAWDOWN_MIN <= drawdown <= _DRAWDOWN_MAX):
        return f"drawdown_out_of_range ({drawdown:.1%})"

    if revenue_growth is not None and revenue_growth < _REVENUE_YOY_MIN:
        return f"revenue_growth_too_negative ({revenue_growth:.1%})"

    if gross_margin is not None and gross_margin <= _GROSS_MARGIN_MIN:
        return f"negative_gross_margin ({gross_margin:.1%})"

    has_net_cash = signals.get("net_cash", 0) or 0
    net_debt_ok = (
        bool(has_net_cash)
        or (net_debt_ebitda is not None and net_debt_ebitda < _NET_DEBT_EBITDA_MAX)
        or (interest_coverage is not None and interest_coverage > _INTEREST_COVERAGE_MIN)
    )
    if not net_debt_ok:
        return "balance_sheet_too_weak"

    # Technical confirmation: price above rising 50DMA OR ≥2 higher weekly lows
    price_vs_50dma = signals.get("price_vs_50dma")
    sma_50_rising = signals.get("sma_50_rising")
    if price_vs_50dma is not None and sma_50_rising is not None:
        if price_vs_50dma == 0.0 and sma_50_rising == 0.0:
            return "no_technical_confirmation"

    return None


def _score_interim(signals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute §4.2.1 interim score without revisions feed.

    Args:
        signals: Dict of signal_type → value for the ticker.

    Returns:
        Dict with score, score_partial, components.
    """
    score_partial = 0.0
    components: Dict[str, float] = {}

    # Insider net buying (20 pts)
    insider_val = signals.get("insider_buy_value_90d") or 0.0
    if insider_val > 0:
        insider_pts = min(20.0, 20.0 * (insider_val / 5_000_000.0))
        components["insider"] = round(insider_pts, 1)
        score_partial += insider_pts

    # Balance sheet / financial health (15 pts)
    net_debt_ebitda = signals.get("net_debt_ebitda_signal") or 0.0
    interest_cov = signals.get("interest_coverage_signal") or 0.0
    bs_pts = 0.0
    if net_debt_ebitda < 1.0 or interest_cov > 8.0:
        bs_pts = 15.0
    elif net_debt_ebitda < 2.0 or interest_cov > 6.0:
        bs_pts = 10.0
    elif net_debt_ebitda < 3.0 or interest_cov > 4.0:
        bs_pts = 5.0
    components["balance_sheet"] = bs_pts
    score_partial += bs_pts

    # Technical base confirmation (15 pts)
    price_vs_50dma = signals.get("price_vs_50dma") or 0.0
    sma_50_rising = signals.get("sma_50_rising") or 0.0
    tech_pts = 0.0
    if price_vs_50dma > 0.5 and sma_50_rising > 0.5:
        tech_pts = 15.0
    elif price_vs_50dma > 0.5 or sma_50_rising > 0.5:
        tech_pts = 7.5
    components["technical"] = tech_pts
    score_partial += tech_pts

    # Buyback (10 pts)
    buyback_signal = signals.get("buyback_authorization") or 0.0
    buyback_pts = 10.0 if buyback_signal > 0 else 0.0
    components["buyback"] = buyback_pts
    score_partial += buyback_pts

    # Short covering (5 pts)
    short_decline = signals.get("short_volume_declining") or 0.0
    short_pts = 5.0 if short_decline > 0 else 0.0
    components["short_covering"] = short_pts
    score_partial += short_pts

    # Attention vacuum bonus (5 pts)
    crowding = signals.get("crowding_score")
    attention_pts = 0.0
    if crowding is not None and crowding < -0.5:
        attention_pts = 5.0
    elif crowding is not None and crowding > 2.0:
        attention_pts = -5.0  # hype penalty
    components["attention_vacuum"] = attention_pts
    score_partial += attention_pts

    if REVISIONS_FEED_AVAILABLE:
        # Add revisions component when available
        revisions = signals.get("revisions_score") or 0.0
        components["revisions"] = revisions
        score_partial += revisions
        total_score = round(score_partial)
    else:
        # Renormalize: score_partial / 70 * 100
        total_score = round(score_partial * 100 / 70)

    return {
        "score": total_score,
        "score_partial": round(score_partial, 1),
        "components": components,
        "interim_mode": not REVISIONS_FEED_AVAILABLE,
    }


def run(as_of_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Run the Sleeve A weekly screen and upsert results to watchlist.

    Args:
        as_of_date: Date to run screen for (defaults to today).

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("Sleeve A screen for %s (interim_mode=%s)", target_date, not REVISIONS_FEED_AVAILABLE)

    tickers = get_active_tickers()
    passed_filters = 0
    candidates: List[Dict[str, Any]] = []

    for ticker in tickers:
        universe_row = get_universe_row(ticker)
        if not universe_row:
            continue

        signals_list = get_signals(ticker, target_date)
        signals: Dict[str, Any] = {s["signal_type"]: s["value"] for s in signals_list}

        fail_reason = _passes_hard_filters(universe_row, signals)
        if fail_reason:
            continue

        passed_filters += 1
        score_info = _score_interim(signals)

        upsert_watchlist({
            "ticker": ticker,
            "sleeve": _SLEEVE,
            "score": score_info["score"],
            "state": "screening",
        })

        # Store score as a signal for tracking
        upsert_signals([{
            "ticker": ticker,
            "date": target_date,
            "signal_type": "sleeve_a_score",
            "value": score_info["score"],
            "sleeve": _SLEEVE,
        }])

        if score_info["score"] >= SLEEVE_A_DOSSIER_THRESHOLD:
            candidates.append({"ticker": ticker, **score_info})
            state_label = "candidate" if score_info["score"] < SLEEVE_A_PUSH_THRESHOLD else "candidate"
            upsert_watchlist({
                "ticker": ticker,
                "sleeve": _SLEEVE,
                "score": score_info["score"],
                "state": state_label,
            })

    candidates_sorted = sorted(candidates, key=lambda r: r["score"], reverse=True)

    _logger.info(
        "Sleeve A: %d tickers, %d passed filters, %d candidates (score≥%d)",
        len(tickers), passed_filters, len(candidates_sorted), SLEEVE_A_DOSSIER_THRESHOLD,
    )
    return {
        "tickers_screened": len(tickers),
        "passed_filters": passed_filters,
        "candidates": len(candidates_sorted),
        "top_candidates": [{"ticker": c["ticker"], "score": c["score"]} for c in candidates_sorted[:10]],
    }
