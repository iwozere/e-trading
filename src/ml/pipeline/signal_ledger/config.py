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
# 60/120/250 serve the slow streams (P22 takeovers, P24 weekly); a signal stays pending until 250 is known.
HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 60, 120, 250)
# Window of the signal-level price-jump flag and of the MAE/MFE columns (a jump later only voids the longer horizons).
CORE_HORIZON = 20
# Every ticker with a signal this recent is re-fetched nightly, so book positions held past the
# signal's own horizons (P22 for ~12 months) keep fresh prices.
ACTIVE_PRICE_DAYS = 400
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
    # P24 trades crashed names: tiered costs average ~0.3% per side in the backtest.
    "p24_daily": 60.0,
    "p24_weekly": 60.0,
    # P22 names are small/mid-cap biotechs: wider spreads than large caps.
    "p22_tier_entry": 60.0,
}

# Exit policies (see exits.py). Native exits of streams that republish their picks: the position is sold at the
# next open once the ticker has not been re-signalled for this many sessions (P21 rebalances monthly).
DECAY_SESSIONS: Dict[str, int] = {
    "p05_top_picks": 5,
    "p06_phase2_alerts": 5,
    "p06_final_universe": 5,
    "p10_final_passed": 5,
    "p17_candidates": 5,
    "p21_targets": 25,
}
# Streams whose signals carry the pipeline's own exit (``meta["exit_signal_date"]``): P22 tier drop, P24 paper exit.
META_EXIT_SOURCES: Tuple[str, ...] = ("p22_tier_entry", "p24_daily", "p24_weekly")
# Longest any policy holds a position (sessions).
MAX_POLICY_HOLD = 250
BRACKET_STOP_PCT = 0.15
BRACKET_TARGET_PCT = 0.30
BRACKET_MAX_HOLD = 120
TRAIL_ATR_MULT = 3.0
ATR_PERIOD = 14
# Trailing-stop width when no ATR can be computed (fraction of the entry price).
TRAIL_FALLBACK_PCT = 0.20

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
