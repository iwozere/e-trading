from dataclasses import dataclass


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
class UniverseConfig(BaseFilterConfig):
    """Configuration for universe downloading."""

    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    exclude_test_issues: bool = True
    exclude_etfs: bool = False
    alphabetic_only: bool = True
