"""
P20 Kestrel — Sleeve B screen (Event Catalysts).

B1: FDA run-ups from catalyst calendar (PDUFA/AdCom/readout).
B2: Spin-offs — entry window day 20-60 post-spin.
B3: S&P/Nasdaq index changes + activist 13D watchlist adds.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services.kestrel_service import KestrelService as _KestrelService

_kestrel = _KestrelService()
get_active_tickers = _kestrel.get_active_tickers
get_catalysts_in_window = _kestrel.get_catalysts_in_window
get_latest_signal = _kestrel.get_latest_signal
get_past_spinoffs = _kestrel.get_past_spinoffs
get_universe_row = _kestrel.get_universe_row
upsert_watchlist = _kestrel.upsert_watchlist
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_SLEEVE = "B"
_B1_MCAP_MIN = 300_000_000
_B1_MCAP_MAX = 10_000_000_000
_B1_DAYS_MIN = 10
_B1_DAYS_MAX = 90
_CROWDING_SPIKE_THRESHOLD = 3.0  # mention_z20 > 3σ before T−10 → skip B1
_B2_DAYS_MIN = 20   # post-spin entry window start
_B2_DAYS_MAX = 60   # post-spin entry window end
_B2_MCAP_MIN = 150_000_000  # smaller floor — spin-offs often start with smaller float


def screen_b1(as_of_date: date) -> List[Dict[str, Any]]:
    """
    Screen B1: FDA/PDUFA run-up candidates from catalyst calendar.

    Args:
        as_of_date: Date to run screen.

    Returns:
        List of candidate dicts.
    """
    catalysts = get_catalysts_in_window(days_ahead=_B1_DAYS_MAX)
    fda_types = {"pdufa", "adcom", "fda_readout", "clinical_readout"}

    candidates: List[Dict[str, Any]] = []
    for c in catalysts:
        event_type = str(c.get("event_type", "")).lower()
        if event_type not in fda_types:
            continue

        ticker = str(c.get("ticker", "")).upper()
        event_date = c.get("event_date")
        if not event_date:
            continue

        if isinstance(event_date, str):
            event_date = date.fromisoformat(event_date)

        days_out = (event_date - as_of_date).days
        if not (_B1_DAYS_MIN <= days_out <= _B1_DAYS_MAX):
            continue

        universe_row = get_universe_row(ticker)
        if not universe_row:
            continue

        mcap = universe_row.get("mcap")
        if not mcap or not (_B1_MCAP_MIN <= mcap <= _B1_MCAP_MAX):
            continue

        # Crowding spike check: skip if mention_z20 > 3σ before T−10
        if days_out <= 10:
            crowding = get_latest_signal(ticker, "z_social")
            if crowding is not None and float(crowding) > _CROWDING_SPIKE_THRESHOLD:
                _logger.info("B1 crowding skip: %s (z=%.1f, T-%d)", ticker, crowding, days_out)
                continue

        candidates.append({
            "ticker": ticker,
            "sleeve": _SLEEVE,
            "sub_sleeve": "B1",
            "event_type": event_type,
            "event_date": event_date,
            "days_out": days_out,
            "mcap": mcap,
        })

    return candidates


def screen_b2(as_of_date: date) -> List[Dict[str, Any]]:
    """
    Screen B2: Spin-offs in the 20–60 day post-spin entry window.

    Institutional forced selling typically clears within the first 3 weeks.
    Day 20–60 is the entry window where price dislocation persists but
    ownership is stabilising.

    Args:
        as_of_date: Date to run screen.

    Returns:
        List of candidate dicts.
    """
    spinoffs = get_past_spinoffs(days_min=_B2_DAYS_MIN, days_max=_B2_DAYS_MAX)
    candidates: List[Dict[str, Any]] = []

    for spinoff in spinoffs:
        ticker = str(spinoff.get("ticker", "")).upper()
        if not ticker:
            continue

        event_date = spinoff.get("event_date")
        if event_date is None:
            continue
        if isinstance(event_date, str):
            event_date = date.fromisoformat(event_date)

        universe_row = get_universe_row(ticker)
        if not universe_row:
            continue

        mcap = universe_row.get("mcap")
        if not mcap or mcap < _B2_MCAP_MIN:
            continue

        days_since_spin = (as_of_date - event_date).days
        candidates.append({
            "ticker": ticker,
            "sleeve": _SLEEVE,
            "sub_sleeve": "B2",
            "event_type": "spinoff",
            "event_date": event_date,
            "days_since_spin": days_since_spin,
            "mcap": mcap,
        })

    return candidates


def screen_b3_activist(as_of_date: date) -> List[Dict[str, Any]]:
    """
    Screen B3: Tickers with recent activist 13D signals not yet on watchlist.

    Args:
        as_of_date: Date to screen.

    Returns:
        List of candidate dicts.
    """
    candidates: List[Dict[str, Any]] = []
    tickers = get_active_tickers()

    for ticker in tickers:
        activist_value = get_latest_signal(ticker, "activist_13d")
        has_activist = activist_value is not None and float(activist_value) > 0
        if has_activist:
            candidates.append({
                "ticker": ticker,
                "sleeve": _SLEEVE,
                "sub_sleeve": "B3",
                "trigger": "activist_13d",
            })

    return candidates


def run(as_of_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Run Sleeve B screens and upsert candidates to watchlist.

    Args:
        as_of_date: Date to run (defaults to today).

    Returns:
        Summary dict.
    """
    target_date = as_of_date or date.today()
    _logger.info("Sleeve B screen for %s", target_date)

    b1 = screen_b1(target_date)
    b2 = screen_b2(target_date)
    b3 = screen_b3_activist(target_date)
    all_candidates = b1 + b2 + b3

    for c in all_candidates:
        upsert_watchlist({
            "ticker": c["ticker"],
            "sleeve": _SLEEVE,
            "state": "screening",
        })

    _logger.info(
        "Sleeve B: B1=%d FDA run-ups, B2=%d spin-offs, B3=%d activists",
        len(b1), len(b2), len(b3),
    )
    return {
        "b1_fda_runups": len(b1),
        "b2_spinoffs": len(b2),
        "b3_activists": len(b3),
        "total_candidates": len(all_candidates),
    }
