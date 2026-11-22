"""
Survivor-bias study — every frozen number (docs/Design.md, rules frozen 2026-11-22).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "results" / "survivor_bias"
PANEL_CACHE = "panel.parquet"

# Universes
FULL = "full"
VINTAGE_LIKE = "vintage_like"
SURVIVORS = "survivors"
UNIVERSES: Tuple[str, ...] = (FULL, VINTAGE_LIKE, SURVIVORS)

# Delisting rule
LIVE = "live"
ORDINARY = "ordinary"
DISTRESSED = "distressed"
DELISTED_GAP_DAYS = 10
DISTRESS_PRICE = 1.0
DISTRESS_LOOKBACK_BARS = 63
DISTRESS_DROP = -0.50
HAIRCUT = -0.30
LAST_CLOSE = "last_close"
WITH_HAIRCUT = "haircut"
DELISTING_VERSIONS: Tuple[str, ...] = (LAST_CLOSE, WITH_HAIRCUT)

# Monthly panel and eligibility
DATA_START = date(1999, 1, 1)
DOLLAR_VOLUME_BARS = 63
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 5_000_000.0
MIN_BARS = 63
MAX_STALE_DAYS = 7

# Momentum
SKIP_BARS = 21
LOOKBACK_BARS = 252
MIN_MOMENTUM_BARS = 260
WEEKS_IN_WINDOW = 46
MIN_WEEKLY_RETURNS = 40
MIN_VOL = 0.05
DECILE = 0.10
P21_POOL = 500
P21_ENTRY_RANK = 20
P21_HOLD_RANK = 60
P21_TARGET = 20
P21_COST_PER_SIDE = 0.0010

# Sensitivity
CLIP_MONTHLY_RETURN = 3.0

# Amendment 1 (docs/Design.md): a return counts only where our adjustment and EODHD's adjusted close agree
MAX_SOURCE_GAP = 0.10

# Amendment 2 (docs/Design.md): bad raw prints
SPIKE_MOVE = 3.0
SPIKE_BACK = 1.5
SPIKE_REVERT_BARS = 5
IMPOSSIBLE_MOVE = 20.0

# Periods by formation / signal date (inclusive)
PERIODS: Dict[str, Tuple[date, date]] = {
    "2000-2007": (date(2000, 1, 1), date(2007, 12, 31)),
    "2008-2011": (date(2008, 1, 1), date(2011, 12, 31)),
    "2012-2021": (date(2012, 1, 1), date(2021, 12, 31)),
    "2022-2026": (date(2022, 1, 1), date(2026, 12, 31)),
    "2000-2020": (date(2000, 1, 1), date(2020, 12, 31)),
}

# P24 run
P24_VARIANTS: Tuple[str, ...] = ("full", "drop_only")
P24_SIGNALS_FROM = date(2000, 1, 1)
