"""
Configuration data classes for the Short Squeeze Detection Pipeline.

This module defines type-safe configuration classes for all pipeline components.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class SchedulingConfig:
    """Configuration for pipeline scheduling."""
    screener_frequency: str = "weekly"
    screener_day: str = "monday"
    screener_time: str = "08:00"
    deep_scan_frequency: str = "daily"
    deep_scan_time: str = "10:00"
    timezone: str = "Europe/Zurich"


@dataclass
class UniverseConfig:
    """Configuration for universe loading and filtering."""
    min_market_cap: int = 100_000_000  # $100M
    max_market_cap: int = 10_000_000_000  # $10B
    min_avg_volume: int = 200_000
    exchanges: List[str] = field(default_factory=lambda: ['NYSE', 'NASDAQ'])
    # Optimization settings
    use_multi_strategy_loading: bool = True
    prioritize_known_candidates: bool = True
    max_universe_size: int = 1000  # Limit universe size for efficiency
    use_finra_filtering: bool = True  # Enable FINRA filtering using database data


@dataclass
class ScreenerFilters:
    """Configuration for screener filtering criteria."""
    si_percent_min: float = 0.15
    days_to_cover_min: float = 5.0
    float_max: int = 100_000_000
    top_k_candidates: int = 50


@dataclass
class ScreenerWeights:
    """Configuration for screener scoring weights."""
    short_interest_pct: float = 0.4
    days_to_cover: float = 0.3
    float_ratio: float = 0.2
    volume_consistency: float = 0.1


@dataclass
class ScreenerConfig:
    """Configuration for weekly screener module."""
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    filters: ScreenerFilters = field(default_factory=ScreenerFilters)
    scoring: ScreenerWeights = field(default_factory=ScreenerWeights)


@dataclass
class DeepScanMetrics:
    """Configuration for deep scan metrics calculation."""
    volume_lookback_days: int = 14
    sentiment_lookback_hours: int = 24
    options_min_volume: int = 100


@dataclass
class DeepScanWeights:
    """Configuration for deep scan scoring weights."""
    volume_spike: float = 0.35
    sentiment_24h: float = 0.25
    call_put_ratio: float = 0.20
    borrow_fee: float = 0.20


@dataclass
class DeepScanConfig:
    """Configuration for daily deep scan module."""
    batch_size: int = 10
    api_delay_seconds: float = 0.2
    metrics: DeepScanMetrics = field(default_factory=DeepScanMetrics)
    scoring: DeepScanWeights = field(default_factory=DeepScanWeights)


@dataclass
class AlertThreshold:
    """Configuration for a single alert threshold level."""
    squeeze_score: float
    min_si_percent: float
    min_volume_spike: float
    min_sentiment: float


@dataclass
class AlertThresholds:
    """Configuration for all alert threshold levels."""
    high: AlertThreshold = field(default_factory=lambda: AlertThreshold(
        squeeze_score=0.8, min_si_percent=0.25, min_volume_spike=4.0, min_sentiment=0.6
    ))
    medium: AlertThreshold = field(default_factory=lambda: AlertThreshold(
        squeeze_score=0.6, min_si_percent=0.20, min_volume_spike=3.0, min_sentiment=0.5
    ))
    low: AlertThreshold = field(default_factory=lambda: AlertThreshold(
        squeeze_score=0.4, min_si_percent=0.15, min_volume_spike=2.0, min_sentiment=0.4
    ))


@dataclass
class AlertCooldown:
    """Configuration for alert cooldown periods."""
    high_alert_days: int = 7
    medium_alert_days: int = 5
    low_alert_days: int = 3


@dataclass
class AlertChannels:
    """Configuration for alert notification channels."""
    telegram_enabled: bool = True
    telegram_chat_ids: List[str] = field(default_factory=lambda: ['@trading_alerts'])
    email_enabled: bool = True
    email_recipients: List[str] = field(default_factory=lambda: ['trader@example.com'])


@dataclass
class AlertConfig:
    """Configuration for alert engine."""
    thresholds: AlertThresholds = field(default_factory=AlertThresholds)
    cooldown: AlertCooldown = field(default_factory=AlertCooldown)
    channels: AlertChannels = field(default_factory=AlertChannels)


@dataclass
class AdHocConfig:
    """Configuration for ad-hoc candidate management."""
    default_ttl_days: int = 7
    max_active_candidates: int = 20
    auto_promote_threshold: float = 0.7


@dataclass
class WeeklyReportConfig:
    """Configuration for weekly reports."""
    top_candidates: int = 20
    include_charts: bool = True
    formats: List[str] = field(default_factory=lambda: ['html', 'csv'])


@dataclass
class DailyReportConfig:
    """Configuration for daily reports."""
    top_scores: int = 10
    include_trends: bool = True
    formats: List[str] = field(default_factory=lambda: ['html'])


@dataclass
class ReportConfig:
    """Configuration for reporting engine."""
    weekly_summary: WeeklyReportConfig = field(default_factory=WeeklyReportConfig)
    daily_report: DailyReportConfig = field(default_factory=DailyReportConfig)


@dataclass
class ApiRateLimits:
    """Configuration for API rate limiting."""
    fmp_calls_per_minute: int = 250  # Leave buffer from 300 limit
    finnhub_calls_per_minute: int = 50  # Leave buffer from 60 limit


@dataclass
class DatabaseConfig:
    """Configuration for database operations."""
    batch_size: int = 100
    connection_timeout: int = 30
    query_timeout: int = 60


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling."""
    max_retries: int = 3
    backoff_factor: float = 2.0
    circuit_breaker_threshold: float = 0.5


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    api_rate_limits: ApiRateLimits = field(default_factory=ApiRateLimits)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)


@dataclass
class ScoringConfig:
    """Configuration for scoring engine."""
    normalization_method: str = "minmax"
    score_bounds: tuple = (0.0, 1.0)
    weight_validation: bool = True


@dataclass
class SentimentProviders:
    """Configuration for sentiment data providers."""
    stocktwits: bool = True
    reddit_pushshift: bool = True
    news: bool = True
    google_trends: bool = False  # Optional, conservative rate limit
    twitter: bool = False  # Requires API access
    discord: bool = False  # Requires channel access
    hf_enabled: bool = False  # ML enhancement (CPU intensive)


@dataclass
class SentimentBatching:
    """Configuration for sentiment batch processing."""
    concurrency: int = 8  # Parallel requests
    rate_limit_delay_sec: float = 0.3
    batch_size: int = 50  # Tickers per batch


@dataclass
class SentimentWeights:
    """Configuration for sentiment source weighting."""
    stocktwits: float = 0.4  # High quality, trading-focused
    reddit: float = 0.3  # Good coverage, some noise
    news: float = 0.2  # Credible but lagging
    google_trends: float = 0.1  # Supplementary
    heuristic_vs_hf: float = 0.5  # 50/50 if HF enabled


@dataclass
class SentimentThresholds:
    """Configuration for sentiment quality thresholds."""
    min_mentions_for_hf: int = 20  # Only use ML if sufficient data
    bot_pct_warning: float = 0.5  # Warn if >50% bot activity
    min_data_quality_sources: int = 1  # Require at least 1 source


@dataclass
class SentimentCache:
    """Configuration for sentiment caching."""
    enabled: bool = True
    ttl_seconds: int = 1800  # 30 minutes
    redis_enabled: bool = False  # Use in-memory cache by default


@dataclass
class SentimentMonitoring:
    """Configuration for sentiment monitoring."""
    log_failures: bool = True
    alert_on_all_providers_down: bool = True
    performance_profiling: bool = False


@dataclass
class SentimentConfig:
    """Configuration for sentiment module integration."""
    providers: SentimentProviders = field(default_factory=SentimentProviders)
    batching: SentimentBatching = field(default_factory=SentimentBatching)
    weights: SentimentWeights = field(default_factory=SentimentWeights)
    thresholds: SentimentThresholds = field(default_factory=SentimentThresholds)
    cache: SentimentCache = field(default_factory=SentimentCache)
    monitoring: SentimentMonitoring = field(default_factory=SentimentMonitoring)


@dataclass
class PipelineConfig:
    """Main configuration class for the Short Squeeze Detection Pipeline."""
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    deep_scan: DeepScanConfig = field(default_factory=DeepScanConfig)
    alerting: AlertConfig = field(default_factory=AlertConfig)
    adhoc: AdHocConfig = field(default_factory=AdHocConfig)
    reporting: ReportConfig = field(default_factory=ReportConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)

    # Runtime configuration
    run_id: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize runtime fields."""
        if self.run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.created_at is None:
            self.created_at = datetime.now()