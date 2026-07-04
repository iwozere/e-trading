"""P05 AI Selector — pipeline configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_CACHE = Path(DATA_CACHE_DIR)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_BASE = _PROJECT_ROOT / "results" / "p05_ai_selector"
STAGE1_CACHE_DIR = _DATA_CACHE / "p05" / "stage1"
STAGE2_CACHE_DIR = _DATA_CACHE / "p05" / "stage2"
EARNINGS_CACHE_DIR = _DATA_CACHE / "p05" / "earnings"

P18_RESULTS_BASE = _PROJECT_ROOT / "results" / "p18_institutional_flow"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_MODEL = "claude-sonnet-4-6"
# 5 fully-detailed picks (thesis + risk_factors + exit_strategy with 4 sub-arrays
# and profit_targets) plus market_context routinely exceed 4096 output tokens, which
# truncates the forced tool_use call mid-JSON and yields an empty tool_input. Give the
# model enough headroom to emit the complete structured response.
LLM_MAX_TOKENS = 16384

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
CRYPTO_TICKERS: List[str] = [
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "SOL-USD",
    "ADA-USD",
    "AVAX-USD",
    "DOT-USD",
    "LINK-USD",
    "MATIC-USD",
    "UNI-USD",
    "XRP-USD",
    "LTC-USD",
    "ATOM-USD",
    "NEAR-USD",
    "ICP-USD",
    "FIL-USD",
    "APT-USD",
    "ARB-USD",
    "OP-USD",
    "DOGE-USD",
]

# ---------------------------------------------------------------------------
# Stage 1 thresholds
# ---------------------------------------------------------------------------
MIN_PRICE: float = 2.0
MIN_AVG_DAILY_VOLUME_USD: float = 5_000_000.0
MIN_CRYPTO_DAILY_VOLUME: float = 100_000.0
STAGE1_LOOKBACK_DAYS: int = 60
STAGE1_TOP_N: int = 200

# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------
STAGE2_TOP_N: int = 25
P18_HIGH_SCORE_THRESHOLD: int = 60

# ---------------------------------------------------------------------------
# Signal weights — §6 of spec
# ---------------------------------------------------------------------------
TECHNICAL_WEIGHTS: Dict[str, int] = {
    "sma_crossover_bullish": 15,
    "sma_crossover_bearish": 10,
    "rsi_oversold": 12,
    "rsi_overbought": 8,
    "volume_surge": 15,
    "momentum_5d": 10,
    "atr_compression": 5,
    "near_52w_high": 8,
    "near_52w_low": 8,
}

FUNDAMENTAL_WEIGHTS: Dict[str, int] = {
    "value": 10,
    "quality": 10,
    "safety": 5,
    "growth": 10,
    "dividend": 3,
}

P18_WEIGHTS: Dict[str, int] = {
    "high_score": 40,
    "consensus_exit": 25,
    "form4_insider_buy": 15,
    "schedule_13dg": 15,
}


@dataclass
class P05Config:
    """Consolidated config dataclass for dependency injection in tests."""

    crypto_tickers: List[str] = field(default_factory=lambda: CRYPTO_TICKERS)
    min_price: float = MIN_PRICE
    min_avg_daily_volume_usd: float = MIN_AVG_DAILY_VOLUME_USD
    min_crypto_daily_volume: float = MIN_CRYPTO_DAILY_VOLUME
    stage1_lookback_days: int = STAGE1_LOOKBACK_DAYS
    stage1_top_n: int = STAGE1_TOP_N
    stage2_top_n: int = STAGE2_TOP_N
    p18_high_score_threshold: int = P18_HIGH_SCORE_THRESHOLD
    technical_weights: Dict[str, int] = field(default_factory=lambda: dict(TECHNICAL_WEIGHTS))
    fundamental_weights: Dict[str, int] = field(default_factory=lambda: dict(FUNDAMENTAL_WEIGHTS))
    p18_weights: Dict[str, int] = field(default_factory=lambda: dict(P18_WEIGHTS))
    llm_model: str = LLM_MODEL
    results_base: Path = field(default_factory=lambda: RESULTS_BASE)
    stage1_cache_dir: Path = field(default_factory=lambda: STAGE1_CACHE_DIR)
    stage2_cache_dir: Path = field(default_factory=lambda: STAGE2_CACHE_DIR)
    earnings_cache_dir: Path = field(default_factory=lambda: EARNINGS_CACHE_DIR)
    p18_results_base: Path = field(default_factory=lambda: P18_RESULTS_BASE)

    @classmethod
    def create_default(cls) -> "P05Config":
        return cls()
