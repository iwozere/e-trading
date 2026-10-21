"""
Signal Ledger — configuration.

One place for the paper-book sizing the user chose (2026-10-21): ~1% of capital
per position (USD 2,000), at most 40 open positions per signal stream, idle cash
parked in VT. Costs are a flat round-trip estimate per stream so that the
scorecard is judged after costs, not before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_DIR = RESULTS_ROOT / "signal_ledger"
STATE_DIR = RESULTS_DIR / "_state"
PRICE_CACHE_PATH = STATE_DIR / "prices.parquet"

# Forward-return horizons in trading sessions (entry = next session's open, exit = close h sessions later).
HORIZONS: Tuple[int, ...] = (1, 5, 10, 20)
# VT is the "rest of the money" alternative; SPY/IWM give large- and small-cap context.
BENCHMARKS: Tuple[str, ...] = ("VT", "SPY", "IWM")
# Ticker whose trading days define the session calendar.
CALENDAR_TICKER = "SPY"

# Paper book (portfolio simulation).
START_CAPITAL_USD = 200_000.0
POSITION_USD = 2_000.0
MAX_POSITIONS = 40
DEFAULT_HOLD_SESSIONS = 20

# Round-trip cost estimate in basis points, per source (spread + slippage + commission).
DEFAULT_COST_BPS = 10.0
COST_BPS: Dict[str, float] = {
    "p17_candidates": 100.0,
    "p19_momentum_alerts": 100.0,
    "p06_phase2_alerts": 30.0,
    "p06_final_universe": 30.0,
    "p10_final_passed": 30.0,
}

# Nightly harvest only rescans recent result folders; --full rescans everything (backfill).
RECENT_HARVEST_DAYS = 10
# A signal whose entry session still has no price after this many calendar days is marked no_data.
NO_DATA_AFTER_DAYS = 10
# A one-session move of this factor (up or down, close/close or open/previous close) inside a holding
# window flags the signal: in penny stocks it is usually a reverse split Yahoo never adjusted
# (IPDN 0.09 -> 3.18 overnight). Flagged signals are kept but excluded from scorecard and books.
JUMP_RATIO = 5.0
# yfinance batch size for the price cache refresh.
PRICE_BATCH_SIZE = 150


def cost_bps(source: str) -> float:
    """
    Return the round-trip cost estimate for a signal source.

    Args:
        source: Signal source name (e.g. ``"p05_top_picks"``).

    Returns:
        Cost in basis points.
    """
    return COST_BPS.get(source, DEFAULT_COST_BPS)
