"""
P21 Momentum — Central configuration.

All pipeline-wide constants live here, mirroring
``docs/pipeline-specification.md`` §16 verbatim. Import this module from
every P21 file that needs a path or a strategy parameter — do not
re-declare a constant locally.

Once Phase 1 (docs/pipeline-specification.md §15) ends, ``config/frozen_params.json``
is written and every job's ``main()`` calls ``verify_frozen_params()``
(added in Step 20) to detect drift from this file.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results" / "p21_momentum"
STATE_DIR = RESULTS_DIR / "_state"
CACHE_DIR = STATE_DIR / "cache"
LEDGER_PATH = STATE_DIR / "ledger.jsonl"
CURRENT_POSITIONS_PATH = STATE_DIR / "current_positions.json"
NAV_DAILY_PATH = STATE_DIR / "nav_daily.csv"
REGIME_HISTORY_PATH = STATE_DIR / "regime_history.json"
FUNDAMENTALS_CACHE_PATH = CACHE_DIR / "fundamentals.json"
SECTORS_CACHE_PATH = CACHE_DIR / "sectors.json"

EXCLUSIONS_PATH = PROJECT_ROOT / "config" / "pipeline" / "p21_exclusions.json"
FROZEN_PARAMS_PATH = PROJECT_ROOT / "config" / "frozen_params.json"  # written at end of Phase 1, §15
CHANGELOG_PATH = PROJECT_ROOT / "config" / "pipeline" / "p21_changelog.md"  # written at start of Phase 2, §15

# ---------------------------------------------------------------------------
# Signal (§4)
# ---------------------------------------------------------------------------
LOOKBACK_START = 252
SKIP_RECENT = 21
MIN_HISTORY = 260
MIN_WEEKLY_BARS = 40  # vol computation needs at least this many W-FRI observations
MIN_VOL = 0.05  # guard against division by ~0

# ---------------------------------------------------------------------------
# Filters (§5 / §16)
# ---------------------------------------------------------------------------
MIN_ADV_USD = 50_000_000
ADV_WINDOW_DAYS = 60
GAP_FILTER_TOP3_SHARE = 0.40
EARNINGS_BLACKOUT_DAYS = 5
FUNDAMENTALS_CACHE_TTL_DAYS = 90
SECTORS_CACHE_TTL_DAYS = 30
F4_DATA_MISSING_WARN_PCT = 0.15  # §12.6 D5 / §17 item 2

# ---------------------------------------------------------------------------
# Selection (§6 / §16)
# ---------------------------------------------------------------------------
ENTRY_RANK = 20
HOLD_RANK = 60
MAX_PER_SECTOR = 4
TARGET_COUNT = 20
FALLBACK_POOL_RANK = 40
MIN_POSITIONS_ABORT = 8  # §13: < 8 -> ABORT
MIN_POSITIONS_WARN = 20  # §13: 8-19 -> WARN (WARN_UNDERFILLED)

# ---------------------------------------------------------------------------
# Sizing (§7 / §16)
# ---------------------------------------------------------------------------
NAV_TOTAL_USD = 250_000
SLEEVE_TARGET_PCT = 0.20
MAX_POSITION_PCT = 0.01  # of TOTAL NAV, not sleeve — spec §7 note
MIN_TRADE_USD = 150  # chatter threshold
SIZING_MAX_ITERATIONS = 10
SHARES_DECIMALS = 4  # IBKR fractional granularity

# ---------------------------------------------------------------------------
# Regime (§8 / §16)
# ---------------------------------------------------------------------------
BEAR_LOOKBACK_DAYS = 252
MA_DAYS = 200
VIX_SMOOTHING_DAYS = 20
VIX_THRESHOLD = 28.0
SCALAR_NORMAL = 1.00
SCALAR_BEAR_LOWVOL = 0.60
SCALAR_BEAR_HIGHVOL = 0.25
UPGRADE_CONFIRMATION_MONTHS = 2

# ---------------------------------------------------------------------------
# Execution (§10 / §16)
# ---------------------------------------------------------------------------
SLIPPAGE_BPS = 3.0
COMMISSION_MIN_USD = 0.35
COMMISSION_PER_SHARE = 0.0035
COMMISSION_MAX_PCT = 0.01

# ---------------------------------------------------------------------------
# Risk (§16)
# ---------------------------------------------------------------------------
CATASTROPHIC_STOP_PCT = -0.35
ANOMALY_DAILY_CHANGE_PCT = 0.35  # §11 step 5 / §13 "Daily price change"

# ---------------------------------------------------------------------------
# Benchmark / tracks (§9, §16)
# ---------------------------------------------------------------------------
PROXY_TICKER = "MTUM"
MARKET_REFERENCE = "SPY"
REGIME_INDEX_TICKER = "^GSPC"
REGIME_VOL_TICKER = "^VIX"
TER_ADJUSTMENT_ANNUAL = 0.0005  # MTUM 0.15% -> QDVA 0.20%, §2

# ---------------------------------------------------------------------------
# Data quality gates (§13)
# ---------------------------------------------------------------------------
MIN_CONSTITUENTS = 450
MIN_PRICE_COVERAGE_PCT = 0.95
TARGET_WEIGHT_SUM_TOLERANCE_USD = 1.0
MAX_POSITION_TOLERANCE_USD = 1.0
