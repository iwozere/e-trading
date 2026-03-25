from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BaseFilterConfig:
    """Base configuration for all filters."""
    enabled: bool = True

@dataclass
class FundamentalFilterConfig(BaseFilterConfig):
    """Configuration for fundamental filtering."""
    min_price: float = 1.0
    min_avg_volume: int = 100_000
    min_market_cap: int = 50_000_000
    max_market_cap: int = 2_000_000_000
    max_float: int = 100_000_000

@dataclass
class VolatilityFilterConfig(BaseFilterConfig):
    """Configuration for volatility filtering."""
    min_price: float = 1.0
    min_volatility_threshold: float = 0.02
    min_price_range: float = 0.05
    lookback_days: int = 15
    interval: str = "1h"
    atr_period: int = 14
    min_vol_zscore: float = 1.2
    min_vol_rv_ratio: float = 0.3
    min_z_vol_delta: float = 1.5
    min_z_intensity: float = 1.0
    min_z_gap: float = 0.5
    min_z_atr: float = 0.5
    min_close_pos: float = 0.75

@dataclass
class SentimentFilterConfig(BaseFilterConfig):
    """Configuration for sentiment filtering."""
    min_mentions_24h: int = 10
    min_sentiment_score: float = 0.2
    max_bot_pct: float = 0.3
    min_virality_index: float = 1.5
    min_unique_authors: int = 5

@dataclass
class UniverseConfig(BaseFilterConfig):
    """Configuration for universe downloading."""
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    exclude_test_issues: bool = True
    alphabetic_only: bool = True
