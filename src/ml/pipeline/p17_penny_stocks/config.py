"""
P17 Penny Stock Screener — Pipeline Configuration
"""

from dataclasses import dataclass
from typing import Optional

from src.ml.pipeline.shared.config import FundamentalFilterConfig, UniverseConfig


@dataclass
class P17FilterConfig(FundamentalFilterConfig):
    """Hard filter thresholds applied at the universe screening stage."""

    # Price range — defines the penny stock universe
    min_price: float = 0.50
    max_price: float = 10.00

    # Liquidity
    min_daily_volume: int = 500_000
    min_avg_dollar_volume: float = 1_000_000.0
    min_avg_volume: int = 500_000      # satisfies FundamentalFilterConfig.min_avg_volume

    # Market cap
    min_market_cap: int = 30_000_000   # $30M
    max_market_cap: int = 2_000_000_000  # $2B

    # Float (public shares)
    min_float: int = 5_000_000         # 5M shares
    max_float: int = 50_000_000        # 50M shares

    # Financial survival hard-stops
    min_cash_runway_months: float = 6.0
    max_debt_to_cash: float = 5.0

    # Risk controls
    max_intraday_spread_pct: float = 0.08   # 8% bid-ask spread
    max_offerings_last_12m: int = 3

    # Data download parameters
    ohlcv_lookback_days: int = 90
    ohlcv_chunk_size: int = 50


@dataclass
class P17TechnicalConfig:
    """Technical analysis indicator parameters."""

    # Relative volume
    rvol_lookback_days: int = 30
    rvol_strong_threshold: float = 3.0

    # Price momentum
    momentum_20d_max: float = 3.00      # >300% 20d return → late euphoric spike, penalise

    # Bollinger Band squeeze detection
    bb_period: int = 20
    bb_std: float = 2.0
    bb_squeeze_width_pct: float = 0.10  # BB width < 10% of mid → squeeze

    # ATR
    atr_period: int = 14

    # Breakout lookback windows
    breakout_lookback_20d: int = 20
    breakout_lookback_50d: int = 50

    # Accumulation: count of high-volume green days
    accumulation_lookback_days: int = 20


@dataclass
class P17ShortSqueezeConfig:
    """Short squeeze detection thresholds."""

    si_moderate_threshold: float = 0.10   # 10% of float
    si_high_threshold: float = 0.20       # 20% of float
    si_extreme_threshold: float = 0.30    # 30% of float — adds halt risk flag

    days_to_cover_threshold: float = 3.0
    min_rvol_for_trigger: float = 2.5     # squeeze only scores high when volume confirms

    # Extra liquidity requirement when SI is extreme
    high_si_min_volume: int = 1_000_000


@dataclass
class P17ScoringConfig:
    """Composite score weights and tier assignment thresholds."""

    # Weights — must sum to 1.0
    # catalyst is a Phase 1 placeholder (always 0); its 0.10 redistributed proportionally
    weight_momentum: float = 0.28
    weight_volume: float = 0.22
    weight_technical: float = 0.17
    weight_fundamentals: float = 0.17
    weight_catalyst: float = 0.00
    weight_short_squeeze: float = 0.11
    weight_accumulation: float = 0.05

    # Dilution hard penalties — deducted from weighted sum after scoring
    penalty_atm_offering: float = 20.0
    penalty_convertible_debt: float = 30.0
    penalty_reverse_split: float = 40.0
    penalty_shelf_offering: float = 15.0
    penalty_warrant_issuance: float = 10.0

    # Tier thresholds (applied to final_score after dilution deductions)
    tier_a_min_score: float = 75.0
    tier_b_min_score: float = 55.0
    tier_c_min_score: float = 35.0

    # Mandatory conditions to mark as EXPLOSIVE_CANDIDATE
    mandatory_rvol: float = 3.0
    mandatory_dilution_penalty_max: float = 30.0

    # Minimum score to trigger alert notification
    min_alert_score: float = 75.0


@dataclass
class P17AlertConfig:
    """Alert and notification settings."""

    enabled: bool = True
    min_alert_score: float = 75.0
    telegram_enabled: bool = True
    email_enabled: bool = False


@dataclass
class P17PipelineConfig:
    """Complete P17 pipeline configuration."""

    filter_config: P17FilterConfig
    universe_config: UniverseConfig
    technical_config: P17TechnicalConfig
    short_squeeze_config: P17ShortSqueezeConfig
    scoring_config: P17ScoringConfig
    alert_config: P17AlertConfig

    save_intermediate_results: bool = True
    verbose_logging: bool = True
    user_id: Optional[str] = None

    @classmethod
    def create_default(cls) -> "P17PipelineConfig":
        """Create a pipeline config with sensible production defaults."""
        universe_cfg = UniverseConfig()
        universe_cfg.cache_ttl_hours = 24
        universe_cfg.exclude_etfs = True
        universe_cfg.exclude_test_issues = True

        return cls(
            filter_config=P17FilterConfig(),
            universe_config=universe_cfg,
            technical_config=P17TechnicalConfig(),
            short_squeeze_config=P17ShortSqueezeConfig(),
            scoring_config=P17ScoringConfig(),
            alert_config=P17AlertConfig(),
        )
