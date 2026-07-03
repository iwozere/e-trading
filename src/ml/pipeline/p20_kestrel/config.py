"""
P20 Kestrel — Central configuration.

All pipeline-wide constants live here. Import this module from every P20 file
that needs a path, model name, budget cap, or feature flag.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

DATA_CACHE_PATH = Path(DATA_CACHE_DIR)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"

# Monthly spend cap in USD. Classification continues to 120 %; dossiers pause at 100 %.
LLM_MONTHLY_BUDGET_USD: float = 75.0

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
# Set True when a sell-side EPS-revision feed is wired (§4.2.1).
# Until then Sleeve A uses renormalized 70-point scoring.
REVISIONS_FEED_AVAILABLE: bool = False

# ---------------------------------------------------------------------------
# Universe source
# ---------------------------------------------------------------------------
# Nasdaq-listed tickers CSV produced by P06 / downloaded from Nasdaq.
# Columns must include at minimum: Symbol, Market Cap, Industry, Sector.
NASDAQ_TICKERS_CSV = DATA_CACHE_PATH / "universe" / "nasdaq_screener.csv"

# Curated activist investor list for 13D/G monitoring (Flow C in filings_ingest).
ACTIVISTS_JSON = PROJECT_ROOT / "config" / "pipeline" / "activists.json"

# ---------------------------------------------------------------------------
# Sentiment staleness thresholds (days) — used by sentiment_aggregator.py
# ---------------------------------------------------------------------------
STALENESS_DAYS: dict[str, int] = {
    "social": 3,    # stocktwits / reddit / apewisdom
    "gdelt": 2,
    "trends": 10,
    "av_news": 3,
}

# Google Trends anchor term (fixed; rescaling baseline uses this same term).
TRENDS_ANCHOR_TERM = "stock market"

# Alpha Vantage daily quota (free tier: 25; 5 reserved for retries).
AV_DAILY_QUOTA: int = 20

# ---------------------------------------------------------------------------
# Sleeve C momentum parameters
# ---------------------------------------------------------------------------
# Minimum 20-day average dollar volume for Sleeve C eligibility.
SLEEVE_C_MIN_ADV_USD: float = 20_000_000

# Minimum 20-day average dollar volume for Sleeve A eligibility.
SLEEVE_A_MIN_ADV_USD: float = 10_000_000

# Sleeve A dossier threshold (score ≥ this → queue LLM dossier).
SLEEVE_A_DOSSIER_THRESHOLD: int = 60

# Sleeve A push-alert threshold (score ≥ this → same-day push if advanced by LLM).
SLEEVE_A_PUSH_THRESHOLD: int = 75

# ---------------------------------------------------------------------------
# Logging / results directory
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results" / "p20_kestrel"
