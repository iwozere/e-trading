"""
EMPS3 Pipeline Configuration

Configuration dataclasses for the EMPS3 (Accumulation Phase Detection) pipeline.
"""

from dataclasses import dataclass
from typing import Optional

from src.ml.pipeline.shared.config import (
    FundamentalFilterConfig,
    SentimentFilterConfig,
    UniverseConfig
)


@dataclass
class EMPS3FilterConfig(FundamentalFilterConfig):
    """
    Configuration for Stage 2 & 3 filters for Precursor / Accumulation Phase.
    """
    # Fundamental filters (Stage 2)
    min_price: float = 1.0
    min_avg_volume: int = 400_000
    min_market_cap: int = 50_000_000      # $50M
    max_market_cap: int = 5_000_000_000   # $5B
    max_float: int = 60_000_000           # 60M shares

    # Accumulation Analyzer Config (Stage C)
    min_vol_zscore: float = 1.5
    max_price_impact: float = 0.03 # was - 0.025
    min_vol_rv_ratio: float = 1.5 # was - 2.0
    lookback_days: int = 10
    require_dark_pool_surge: bool = True
    max_distance_from_resistance: float = 0.05 # was - 3% from 52w High
    max_distance_from_sma20: float = 0.10      # 10% from SMA20

    # Data parameters
    interval: str = "1h"
    atr_period: int = 14

    # Chunking for OHLCV download (keeps peak memory bounded on Pi; also
    # the checkpoint granularity — diagnostics are persisted after each chunk).
    ohlcv_chunk_size: int = 50


@dataclass
class EMPS3RollingMemoryConfig:
    """
    Configuration for Rolling Memory and Phase 1.5 detection.
    """
    enabled: bool = True
    lookback_days: int = 5

    # Phase 1.5 detection (Early Warning)
    phase1_5_min_appearances: int = 3
    min_vol_slope: float = 0.05      # Minimum volume acceleration slope
    max_atr_slope: float = -0.0001   # Maximum ATR contraction slope (must be negative)

    # Alert settings
    send_alerts: bool = True
    alert_on_phase_1_5: bool = True
    alert_on_prebreakout: bool = True

    # Output settings
    save_rolling_candidates: bool = True


@dataclass
class EMPS3PipelineConfig:
    """
    Complete EMPS3 pipeline configuration.
    """
    filter_config: EMPS3FilterConfig
    universe_config: UniverseConfig
    rolling_memory_config: EMPS3RollingMemoryConfig
    # We can reuse sentiment config if we want to run sentiment stage
    sentiment_config: SentimentFilterConfig

    mode: str = "precursor"

    # Output settings
    save_intermediate_results: bool = True
    generate_summary: bool = True
    verbose_logging: bool = True
    user_id: Optional[str] = None

    # Cache parameters
    fundamental_cache_enabled: bool = True
    fundamental_cache_ttl_days: int = 14
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 100

    @classmethod
    def create_default(cls) -> "EMPS3PipelineConfig":
        universe_config = UniverseConfig()
        universe_config.cache_ttl_hours = 168  # 7 days
        universe_config.exclude_etfs = True  # match p06: equities only from NASDAQ Trader ETF flag

        return cls(
            filter_config=EMPS3FilterConfig(),
            universe_config=universe_config,
            rolling_memory_config=EMPS3RollingMemoryConfig(),
            sentiment_config=SentimentFilterConfig(enabled=True),
        )
