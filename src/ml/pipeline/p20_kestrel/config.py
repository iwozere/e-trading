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

# 2026-08-27: wired via ingest/pdufa_calendar_ingest.py (pdufa.bio's
# search-index.json covers PDUFA/AdCom/clinical-readout in one fetch, no
# per-ticker requests). Was False from at least 2026-08-10 through 2026-08-26
# (17/17 days), during which screen_b1() in sleeve_b.py could never match a
# candidate — confirmed via production logs. Unlike REVISIONS_FEED_AVAILABLE
# this doesn't gate a scoring formula, just the Data Health warning, so there
# was no shadow-mode review gate to wait out before flipping it.
PDUFA_CALENDAR_AVAILABLE: bool = True

# 2026-08-27: wired via ingest/spinoff_ingest.py (EDGAR Form 10/10-12B quarterly
# index scan). Was False from at least 2026-08-10 through 2026-08-26 (17/17
# days), during which screen_b2() in sleeve_b.py could never match a candidate
# — confirmed via production logs. Note: event_date is the filing date, not a
# confirmed distribution date — see spinoff_ingest.py docstring for why, and
# docs/Tasks.md for the LLM-dossier-based follow-up that would close that gap.
SPINOFF_MONITOR_AVAILABLE: bool = True

# ---------------------------------------------------------------------------
# Universe source
# ---------------------------------------------------------------------------
# Nasdaq-listed tickers CSV produced by P06 / downloaded from Nasdaq.
# Columns must include at minimum: Symbol, Market Cap, Industry, Sector.
NASDAQ_TICKERS_CSV = DATA_CACHE_PATH / "universe" / "nasdaq_screener.csv"

# Minimum market cap (USD) to include in k20_universe.
# Applied as a pre-filter when the CSV carries a Market Cap column (primary API).
# Without Market Cap in the CSV (nasdaqtrader.com fallback), ticker-format
# filtering is the only guard; screening steps apply real mcap gates.
UNIVERSE_MIN_MCAP_USD: float = 10_000_000  # $10M floor — broad enough for small-cap focus

# Curated activist investor list for 13D/G monitoring (Flow C in filings_ingest).
ACTIVISTS_JSON = PROJECT_ROOT / "config" / "pipeline" / "activists.json"

# ---------------------------------------------------------------------------
# Sentiment staleness thresholds (days) — used by sentiment_aggregator.py
# ---------------------------------------------------------------------------
STALENESS_DAYS: dict[str, int] = {
    "social": 3,  # stocktwits / reddit / apewisdom
    "gdelt": 2,
    "trends": 10,
    "av_news": 3,
}

# Google Trends anchor term (fixed; rescaling baseline uses this same term).
TRENDS_ANCHOR_TERM = "stock market"

# Alpha Vantage daily quota (free tier: 25; 5 reserved for retries).
AV_DAILY_QUOTA: int = 20

# ---------------------------------------------------------------------------
# Revisions feed ingest (gap 10.1, §4.2.1) — tunable component weights
# ---------------------------------------------------------------------------
# revisions_ingest.py blends three analyst signals into `revisions_score`
# (stored in k20_signals, sleeve="A"), so the score is useful immediately
# rather than needing a cold start:
#   - Finnhub recommendation-trend momentum (immediate — history in one call)
#   - FMP rating-change net count, trailing 60d (immediate — dated history)
#   - FMP forward-EPS-consensus 60d delta (warms up over ~60-90d of our own
#     daily snapshots — the literal "forward-EPS revisions up 60d" signal)
# Point budgets sum to 30, matching the full-mode formula's `30 * revisions`
# weight in §4.2. All are first-cut heuristics — expected to be recalibrated
# after reviewing shadow-mode output (REVISIONS_FEED_AVAILABLE stays False
# while this job writes signals but sleeve_a.py ignores them).
REVISIONS_GRADES_WINDOW_DAYS: int = 60
REVISIONS_EPS_DELTA_WINDOW_DAYS: int = 60
REVISIONS_EPS_DELTA_TOLERANCE_DAYS: int = 7  # +/- around the 60d mark when matching a past snapshot

REVISIONS_FINNHUB_MAX_PTS: float = 12.0
REVISIONS_GRADES_MAX_PTS: float = 8.0
REVISIONS_EPS_DELTA_MAX_PTS: float = 10.0

# Momentum/count values that earn the full point budget for each component
# (linearly scaled below this, clipped at/above it; negative values clip to 0
# — v1 deliberately does not penalize negative revisions, matching §4.2.1's
# "renormalize, don't zero" philosophy of never making interim/new signals
# punitive before they're calibrated).
REVISIONS_FINNHUB_FULL_CREDIT_MOMENTUM: float = 1.0  # net-bullish-score swing over ~3mo
REVISIONS_GRADES_FULL_CREDIT_NET: float = 4.0  # net upgrades (upgrades - downgrades) in 60d
REVISIONS_EPS_DELTA_FULL_CREDIT_PCT: float = 0.05  # +5% consensus EPS bump over 60d

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
