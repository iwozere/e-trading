"""
P20 Kestrel — Sleeve C screen (Momentum in Live Themes).

RS = 0.5×(3m pct) + 0.5×(6m pct).
Eligible: top decile, price > 50DMA > 200DMA, ADV ≥ $20M,
positive revenue growth, breakout volume ≥ 1.5×.
Regime filter: SPY < 200DMA → no new entries.
Crowding overlay (§7.6): skip if crowding score > 2σ.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.kestrel_service import KestrelService as _KestrelService
from src.ml.pipeline.p20_kestrel.config import SLEEVE_C_MIN_ADV_USD

_kestrel = _KestrelService()
get_active_tickers = _kestrel.get_active_tickers
get_latest_signal = _kestrel.get_latest_signal
get_signals_for_date = _kestrel.get_signals_for_date
get_universe_row = _kestrel.get_universe_row
upsert_signals = _kestrel.upsert_signals
upsert_watchlist = _kestrel.upsert_watchlist
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_SLEEVE = "C"
_RS_TOP_DECILE_CUTOFF = 0.90  # top 10% of RS scores
_CROWDING_SKIP_THRESHOLD = 2.0


def _compute_rs_score(sig_map: Dict[str, float]) -> float | None:
    """Compute RS score from 3m and 6m return signals."""
    r3m = sig_map.get("return_3m")
    r6m = sig_map.get("return_6m")
    if r3m is None or r6m is None:
        return None
    return 0.5 * r3m + 0.5 * r6m


def _regime_allows_new_entry() -> bool:
    """Return False if SPY < 200DMA (regime filter)."""
    spy_sig = get_latest_signal("SPY", "price_vs_200dma")
    if spy_sig is None:
        return True  # fail-open if no data
    return spy_sig > 0.5


def run(as_of_date: date | None = None) -> Dict[str, Any]:
    """
    Run Sleeve C momentum screen.

    Args:
        as_of_date: Date to run (defaults to the last completed trading day —
            see the target_date comment below for why this isn't "today").

    Returns:
        Summary dict.
    """
    # Match eod_ingest.py's own default ("yesterday", not "today" — it runs at
    # 20:00 UTC, right at the close, before today's candle is finalized). This
    # screen reads eod-derived signals (adv_20d, price_vs_50dma, sma_50/200,
    # return_3m/6m) via get_signals_for_date(), an exact date match — defaulting
    # to "today" here meant every lookup targeted a date eod_ingest never wrote
    # a row for, on any run, ever. Confirmed via prod logs: eod_ingest logs
    # "Running EOD ingest for 2026-09-07" while a same-day screen run logged
    # target_date 2026-09-08 — a permanent one-day miss that surfaced as 100%
    # of tickers failing the very first eod-derived filter (adv_below_min).
    target_date = as_of_date or (date.today() - timedelta(days=1))
    _logger.info("Sleeve C screen for %s", target_date)

    if not _regime_allows_new_entry():
        _logger.info("Sleeve C: regime filter blocks new entries (SPY < 200DMA)")
        return {"new_entries_blocked": True, "candidates": 0}

    tickers = get_active_tickers()
    rs_scores: List[tuple[str, float]] = []
    rejection_reasons: Counter[str] = Counter()

    for ticker in tickers:
        universe_row = get_universe_row(ticker)
        if not universe_row:
            rejection_reasons["no_universe_row"] += 1
            continue

        sig_map = get_signals_for_date(ticker, target_date)

        # adv_20d is written daily to k20_signals by eod_ingest.py, not to the
        # k20_universe row itself (that column is only ever left null) — fall
        # back to the signal like sleeve_a.py already does, or every ticker
        # gets rejected here before RS is ever computed.
        adv_20d = universe_row.get("adv_20d") or sig_map.get("adv_20d")
        if not adv_20d or adv_20d < SLEEVE_C_MIN_ADV_USD:
            rejection_reasons["adv_below_min"] += 1
            continue

        revenue_growth = universe_row.get("revenue_yoy_growth")
        if revenue_growth is not None and revenue_growth <= 0:
            rejection_reasons["negative_revenue_growth"] += 1
            continue

        # Price regime: price > 50DMA > 200DMA
        price_vs_50 = sig_map.get("price_vs_50dma", 0)
        price_vs_200 = sig_map.get("price_vs_200dma", 0)
        sma_50 = sig_map.get("sma_50")
        sma_200 = sig_map.get("sma_200")
        if price_vs_50 < 0.5 or price_vs_200 < 0.5:
            rejection_reasons["price_below_dma"] += 1
            continue
        if sma_50 is not None and sma_200 is not None and sma_50 <= sma_200:
            rejection_reasons["sma50_below_sma200"] += 1
            continue

        rs = _compute_rs_score(sig_map)
        if rs is None:
            rejection_reasons["rs_score_missing_returns"] += 1
            continue

        rs_scores.append((ticker, rs))

    # Sort and take top decile
    rs_scores.sort(key=lambda x: x[1], reverse=True)
    cutoff_idx = max(1, int(len(rs_scores) * (1 - _RS_TOP_DECILE_CUTOFF)))
    top_decile = rs_scores[:cutoff_idx]

    candidates = 0
    for ticker, rs in top_decile:
        # Crowding overlay: skip if crowding > threshold
        sig_map = get_signals_for_date(ticker, target_date)
        crowding = sig_map.get("crowding_score")
        if crowding is not None and crowding > _CROWDING_SKIP_THRESHOLD:
            _logger.debug("Sleeve C crowding skip: %s (crowding=%.1f)", ticker, crowding)
            continue

        upsert_signals(
            [
                {
                    "ticker": ticker,
                    "date": target_date,
                    "signal_type": "rs_score",
                    "value": round(rs, 4),
                    "sleeve": _SLEEVE,
                }
            ]
        )
        upsert_watchlist(
            {
                "ticker": ticker,
                "sleeve": _SLEEVE,
                "score": round(rs * 100, 1),
                "state": "screening",
            }
        )
        candidates += 1

    top_rejections = rejection_reasons.most_common(10)

    _logger.info(
        "Sleeve C: %d tickers → %d top decile → %d candidates (post-crowding)",
        len(tickers),
        len(top_decile),
        candidates,
    )
    if not rs_scores and top_rejections:
        # An empty eligible pool makes "top decile" trivially 0 regardless of
        # market conditions — indistinguishable from a genuinely quiet market
        # unless the rejection reasons are visible. See sleeve_a.py's same
        # instrumentation for the reasoning.
        _logger.warning("Sleeve C: 0 eligible tickers — rejection breakdown (top 10): %s", top_rejections)

    return {
        "tickers_screened": len(tickers),
        "rs_computed": len(rs_scores),
        "top_decile": len(top_decile),
        "candidates": candidates,
        "rejection_breakdown": dict(top_rejections),
    }
