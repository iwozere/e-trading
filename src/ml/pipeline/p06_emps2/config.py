"""
EMPS2 Pipeline Configuration

Configuration dataclasses for the EMPS2 (Enhanced Explosive Move Pre-Screener) pipeline.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EMPS2FilterConfig:
    """
    EMPS2 filtering parameters.

    These thresholds define the multi-stage filtering criteria for identifying
    stocks with explosive move potential.
    """

    # Fundamental filters (Stage 2)
    min_price: float = 0.5
    min_avg_volume: int = 400_000
    min_market_cap: int = 50_000_000      # $50M
    max_market_cap: int = 5_000_000_000   # $5B
    max_float: int = 60_000_000           # 60M shares

    # Volatility filters (Stage 3) - Enhanced with P05 EMPS indicators
    min_volatility_threshold: float = 0.02  # ATR/Price > 2%
    min_price_range: float = 0.05           # 5% range over lookback period
    min_vol_zscore: float = 1.2             # Volume Z-Score > 1.2 (early spike detection)
    min_vol_rv_ratio: float = 0.3           # Volume/Volatility Ratio > 0.5 (accumulation detection).Too many candidates? Increase to 0.7 or 1.0. Too few candidates? Decrease to 0.3 or 0.4.

    # Data parameters
    lookback_days: int = 7
    interval: str = "1h"
    atr_period: int = 14

    # Processing parameters
    batch_size: int = 100  # For batching API calls
    max_retries: int = 3   # Retry failed API calls

    # Cache parameters
    fundamental_cache_enabled: bool = True
    fundamental_cache_ttl_days: int = 3  # Cache TTL for profile2 data

    # Checkpoint parameters
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 100  # Save checkpoint every N tickers


@dataclass
class EMPS2UniverseConfig:
    """
    Configuration for universe downloading.

    Controls how the initial NASDAQ universe is fetched and cached.
    """

    # NASDAQ Trader URLs (updated 2025 - uses SymbolDirectory path)
    nasdaq_url: str = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    other_url: str = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"

    # Filtering
    exclude_test_issues: bool = True
    alphabetic_only: bool = True  # Remove tickers with numbers/special chars

    # Caching (in results folder, not data/cache)
    cache_enabled: bool = True
    cache_ttl_hours: int = 24


@dataclass
class RollingMemoryConfig:
    """
    Configuration for 14-day rolling memory and phase detection.

    Tracks tickers across multiple days to detect accumulation patterns
    and identify phase transitions (Phase 1 → Phase 2).
    """

    # Rolling memory settings
    enabled: bool = True
    lookback_days: int = 14  # How many days back to scan

    # Phase 1 detection (Quiet Accumulation)
    phase1_min_appearances: int = 3  # Must appear 5+ times in lookback period
    phase1_max_sentiment: float = 0.5  # Low/neutral sentiment

    # Phase 2 detection (Early Public Signal)
    phase2_min_vol_zscore: float = 3.0  # Volume acceleration
    phase2_min_sentiment: float = 0.5  # Sentiment starts rising
    phase2_min_virality: float = 1.5  # Going viral

    # Alert settings
    send_alerts: bool = True
    alert_on_phase1: bool = False  # Only alert on Phase 2 by default
    alert_on_phase2: bool = True

    # Output settings
    save_rolling_candidates: bool = True
    save_phase1_watchlist: bool = True
    save_phase2_alerts: bool = True



@dataclass
class SentimentFilterConfig:
    """
    Sentiment filtering parameters.

    These thresholds define social momentum criteria for identifying
    stocks with crowd psychology support.
    """

    min_mentions_24h: int = 10
    min_sentiment_score: float = 0.5
    max_bot_pct: float = 0.3
    min_virality_index: float = 1.2
    min_unique_authors: int = 5
    enabled: bool = True


@dataclass
class EMPS2PipelineConfig:
    """
    Complete EMPS2 pipeline configuration.

    Combines filter config and universe config for end-to-end pipeline execution.
    """

    filter_config: EMPS2FilterConfig
    universe_config: EMPS2UniverseConfig
    rolling_memory_config: RollingMemoryConfig
    sentiment_config: SentimentFilterConfig

    # Output settings
    save_intermediate_results: bool = True
    generate_summary: bool = True
    verbose_logging: bool = True
    enable_uoa_analysis: bool = False # requires paid subscription

    @classmethod
    def create_default(cls) -> "EMPS2PipelineConfig":
        """Create pipeline config with default settings."""
        return cls(
            filter_config=EMPS2FilterConfig(),
            universe_config=EMPS2UniverseConfig(),
            rolling_memory_config=RollingMemoryConfig(),
            sentiment_config=SentimentFilterConfig(enabled=True),
            save_intermediate_results=True,
            generate_summary=True,
            verbose_logging=True,
            enable_uoa_analysis=False # requires paid subscription
        )

    @classmethod
    def create_aggressive(cls) -> "EMPS2PipelineConfig":
        """
        Create aggressive filtering config for highest volatility stocks.

        More restrictive filters to identify the most explosive candidates.
        """
        filter_config = EMPS2FilterConfig(
            min_price=2.0,
            min_avg_volume=500_000,
            min_market_cap=100_000_000,     # $100M
            max_market_cap=3_000_000_000,   # $3B
            max_float=40_000_000,           # 40M shares
            min_volatility_threshold=0.025,  # ATR/Price > 2.5%
            min_price_range=0.07,           # 7% range
            min_vol_zscore=3.0,             # Strong volume spike
            min_vol_rv_ratio=1.0,           # Strong accumulation signal
            lookback_days=7,
            interval="15m",
            atr_period=14
        )

        return cls(
            filter_config=filter_config,
            universe_config=EMPS2UniverseConfig(),
            rolling_memory_config=RollingMemoryConfig(),
            sentiment_config=SentimentFilterConfig(enabled=True),
            save_intermediate_results=True,
            generate_summary=True,
            verbose_logging=True,
            enable_uoa_analysis=False # requires paid subscription
        )

    @classmethod
    def create_conservative(cls) -> "EMPS2PipelineConfig":
        """
        Create conservative filtering config for broader universe.

        Less restrictive filters to capture more candidates.
        """
        filter_config = EMPS2FilterConfig(
            min_price=0.5,
            min_avg_volume=300_000,
            min_market_cap=25_000_000,      # $25M
            max_market_cap=10_000_000_000,  # $10B
            max_float=100_000_000,          # 100M shares
            min_volatility_threshold=0.015,  # ATR/Price > 1.5%
            min_price_range=0.03,           # 3% range
            min_vol_zscore=1.5,             # Moderate volume spike
            min_vol_rv_ratio=0.3,           # Moderate accumulation signal
            lookback_days=7,
            interval="15m",
            atr_period=14
        )

        return cls(
            filter_config=filter_config,
            universe_config=EMPS2UniverseConfig(),
            rolling_memory_config=RollingMemoryConfig(),
            sentiment_config=SentimentFilterConfig(enabled=True),
            save_intermediate_results=True,
            generate_summary=True,
            verbose_logging=True,
            enable_uoa_analysis=False # requires paid subscription
        )
