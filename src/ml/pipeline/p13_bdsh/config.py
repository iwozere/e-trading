import os
from pathlib import Path

"""
Configuration for Pipeline p13: VIX-Threshold Scaling Strategy (BDSH)
"""

# Z-Score Thresholds for Entry Tiers
ENTRY_TIERS = {
    "Tier 1": {"z_threshold": 1.5, "allocation": 0.33},
    "Tier 2": {"z_threshold": 2.5, "allocation": 0.33},
    "Tier 3": {"z_threshold": 3.5, "allocation": 0.34},
}

# Exit Threshold (Mean Reversion)
EXIT_Z_THRESHOLD = 0.0

# Lookback period for Rolling Mean/Std (Z-Score calculation)
VIX_LOOKBACK = 30

# Risk & Simulation Parameters
INITIAL_CAPITAL = 100000.0  # $100,000
SLIPPAGE_PCT = 0.001  # 0.1% for slippage and commissions
STOP_LOSS_PCT = 0.10  # 10% hard stop-loss (Legacy/Fallback)
ATR_PERIOD = 14       # Window for ATR calculation
ATR_MULTIPLIER = 2.0  # Multiplier for ATR stop-loss (2x ATR)

# Symbols
VIX_SYMBOL = "^VIX"
# Defaults
RESOLUTION = "1d"
DEFAULT_START_DATE = "2006-01-01"
DEFAULT_END_DATE = "2026-01-01"


# Paths
BASE_DIR = Path(__file__).resolve().parent
# Project root is 4 levels up from src/ml/pipeline/p13_bdsh/
PROJECT_ROOT = BASE_DIR.parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "p13_bdsh"
STATE_FILE = RESULTS_DIR / "state_p13.json"
