# src/common/sentiments/collect_sentiment_async.py
"""
Standalone async sentiment collection and aggregation module.

This module provides a standalone sentiment analysis system that can be used
independently of any specific pipeline or framework.

Exposes:
- async collect_sentiment_batch(tickers, lookback_hours=24, config=None, history_lookup=None)
- collect_sentiment_batch_sync(...) - sync wrapper using asyncio.run
- SentimentFeatures dataclass for structured output

Features:
- Configurable data providers (StockTwits, Reddit, HuggingFace)
- Flexible output formats (dataclass, dict, JSON)
- Comprehensive error handling and circuit breaker support
- Health monitoring and adapter management
"""

from __future__ import annotations

import asyncio
import json
import inspect
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Union, cast

from src.common.sentiments.processing.calibration import (
    CalibrationStats,
    calibrate_score,
    calibration_status,
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


# -------------------------
# Dataclass for output
# -------------------------
@dataclass
class SentimentFeatures:
    """
    Structured sentiment analysis results for a ticker.

    This dataclass provides a standardized format for sentiment analysis output
    that can be easily serialized to JSON or converted to dictionaries.
    """

    ticker: str
    mentions_24h: int
    unique_authors_24h: int
    mentions_growth_7d: float | None
    positive_ratio_24h: float | None
    sentiment_score_24h: float  # -1..+1, retail signal class only
    sentiment_normalized: float  # 0..1 mapped for scoring, retail signal class only
    virality_index: float
    bot_pct: float  # 0..1
    data_quality: Dict[str, str]  # provider -> 'ok'|'partial'|'missing'|'hf_disabled'|'hf_failed'
    raw_payload: Dict[str, Any]  # raw provider payloads for audit

    # tech_discourse signal class (Hacker News) -- reported separately, never blended into the
    # retail fields above (sentiment-spec-rev2.md §2.5.6). tech_coverage_available=False means
    # the ticker isn't in the entity map at all; the other tech_* fields are then None, not a
    # fabricated neutral reading -- distinct from tech_coverage_available=True with 0 mentions.
    tech_mentions_24h: int | None = None
    tech_sentiment_score_24h: float | None = None  # -1..+1
    tech_sentiment_normalized: float | None = None  # 0..1
    tech_discussion_depth: float | None = None  # mean comments per matched story
    tech_coverage_available: bool | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)


# -------------------------
# Configuration management
# -------------------------
def _load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config: Dict[str, Any] = {}

    # Provider settings
    config["providers"] = {
        "stocktwits": os.getenv("SENTIMENT_STOCKTWITS_ENABLED", "true").lower() == "true",
        "reddit": os.getenv("SENTIMENT_REDDIT_ENABLED", "true").lower() == "true",
        "discord": os.getenv("SENTIMENT_DISCORD_ENABLED", "true").lower() == "true",
        "twitter": os.getenv("SENTIMENT_TWITTER_ENABLED", "false").lower() == "true",
        "finnhub": os.getenv("SENTIMENT_FINNHUB_ENABLED", "true").lower() == "true",
        "hf_enabled": os.getenv("SENTIMENT_HF_ENABLED", "false").lower() == "true",
    }

    # Timing settings
    config["lookback_hours"] = int(os.getenv("SENTIMENT_LOOKBACK_HOURS", "24"))
    config["min_mentions_for_hf"] = int(os.getenv("SENTIMENT_MIN_MENTIONS_HF", "20"))
    config["min_mentions_for_confident_signal"] = int(os.getenv("SENTIMENT_MIN_MENTIONS_SIGNAL", "5"))

    # HuggingFace settings
    config["hf"] = {
        "model_name": os.getenv("SENTIMENT_HF_MODEL", "cardiffnlp/twitter-roberta-base-sentiment"),
        "device": int(os.getenv("SENTIMENT_HF_DEVICE", "-1")),
        "max_workers": int(os.getenv("SENTIMENT_HF_WORKERS", "1")),
    }

    # Batching settings
    config["batching"] = {
        "concurrency": int(os.getenv("SENTIMENT_CONCURRENCY", "8")),
        "rate_limit_delay_sec": float(os.getenv("SENTIMENT_RATE_DELAY", "0.3")),
    }

    # Provider weights
    config["weights"] = {
        "stocktwits": float(os.getenv("SENTIMENT_WEIGHT_STOCKTWITS", "0.2")),
        "reddit": float(os.getenv("SENTIMENT_WEIGHT_REDDIT", "0.2")),
        "news": float(os.getenv("SENTIMENT_WEIGHT_NEWS", "0.2")),
        "finnhub": float(os.getenv("SENTIMENT_WEIGHT_FINNHUB", "0.2")),
        "twitter": float(os.getenv("SENTIMENT_WEIGHT_TWITTER", "0.1")),
        "discord": float(os.getenv("SENTIMENT_WEIGHT_DISCORD", "0.1")),
        "heuristic_vs_hf": float(os.getenv("SENTIMENT_WEIGHT_HF", "0.5")),
    }

    # Heuristic settings
    positive_tokens = os.getenv("SENTIMENT_POSITIVE_TOKENS", "moon,🚀,diamond,buy,long,hold,to the moon,rocket")
    negative_tokens = os.getenv("SENTIMENT_NEGATIVE_TOKENS", "short,sell,dump,bankrupt,bagholder,paper hands,bag")

    config["heuristic"] = {
        "positive_tokens": [t.strip() for t in positive_tokens.split(",") if t.strip()],
        "negative_tokens": [t.strip() for t in negative_tokens.split(",") if t.strip()],
        "engagement_weight_formula": os.getenv("SENTIMENT_ENGAGEMENT_FORMULA", "sqrt"),
    }

    # Caching settings
    config["caching"] = {
        "redis_enabled": os.getenv("SENTIMENT_REDIS_ENABLED", "true").lower() == "true",
        "redis_host": os.getenv("SENTIMENT_REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("SENTIMENT_REDIS_PORT", "6379")),
        "redis_db": int(os.getenv("SENTIMENT_REDIS_DB", "0")),
        "redis_password": os.getenv("SENTIMENT_REDIS_PASSWORD"),
        "memory_max_size": int(os.getenv("SENTIMENT_CACHE_MEMORY_SIZE", "1000")),
        "memory_ttl": int(os.getenv("SENTIMENT_CACHE_MEMORY_TTL", "3600")),
        "redis_ttl": int(os.getenv("SENTIMENT_CACHE_REDIS_TTL", "7200")),
        "warming_enabled": os.getenv("SENTIMENT_CACHE_WARMING", "true").lower() == "true",
        "cleanup_interval": int(os.getenv("SENTIMENT_CACHE_CLEANUP_INTERVAL", "300")),
    }

    return config


DEFAULT_CONFIG = {
    "providers": {
        "stocktwits": True,
        "reddit": True,
        "news": True,
        "trends": True,
        "discord": True,
        "twitter": False,
        "finnhub": True,
        "apewisdom": True,
        "hackernews": True,  # tech_discourse signal class -- no auth required
        # Gated off until BLUESKY_HANDLE/BLUESKY_APP_PASSWORD are configured (spec §2.3).
        "bluesky": False,
        "hf_enabled": False,
    },
    "lookback_hours": 24,
    "min_mentions_for_hf": 20,
    "min_mentions_for_confident_signal": 5,
    "hf": {
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment",  # back-compat alias for models.retail
        "models": {
            "retail": "cardiffnlp/twitter-roberta-base-sentiment",
            "tech_discourse": "distilbert-base-uncased-finetuned-sst-2-english",
        },
        "long_text_strategy": "truncate",  # 'truncate' | 'chunk_mean' | 'chunk_max' -- spec §2.5.4
        "device": -1,
        "max_workers": 1,
    },
    "calibration": {
        "enabled": True,
        "window_days": 30,
        "min_observations": 200,
    },
    "batching": {"concurrency": 8, "rate_limit_delay_sec": 0.3},
    "weights": {
        "stocktwits": 0.2,
        "reddit": 0.2,
        "news": 0.2,
        "finnhub": 0.2,
        "trends": 0.0,  # Trends is strictly for interest volume, not sentiment polarity
        "discord": 0.1,
        "twitter": 0.1,
        "apewisdom": 0.2,
        # Inert while providers.bluesky is False. Renormalized alongside the other retail
        # weights once enabled -- set deliberately, not adopted from the spec's minimal example.
        "bluesky": 0.0,
        # hackernews is intentionally absent: it is signal_class="tech_discourse" and is never
        # blended into the retail score regardless of weight (spec §2.5.6) -- routed out in
        # collect_sentiment_batch by adapter.signal_class, not by this weights dict.
        "heuristic_vs_hf": 0.5,
    },
    "hackernews": {
        "corpus_lookback_hours": 48,
        "max_depth": 4,
        "max_items_per_thread": 300,
        "rate_limit_rps": 10.0,
        "hn_corpus_ttl_seconds": 1800,
        "entity_map_path": None,
    },
    "bluesky": {
        "lang_filter": "en",
        "max_posts_per_ticker": 200,
        "search_terms": ["cashtag", "company_name"],
        "entity_map_path": None,
    },
    "heuristic": {
        "positive_tokens": ["moon", "🚀", "diamond", "buy", "long", "hold", "to the moon", "rocket"],
        "negative_tokens": ["short", "sell", "dump", "bankrupt", "bagholder", "paper hands", "bag"],
        "engagement_weight_formula": "sqrt",  # 'sqrt' or 'linear'
    },
    "caching": {
        "redis_enabled": True,
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 0,
        "redis_password": None,
        "memory_max_size": 1000,
        "memory_ttl": 3600,  # 1 hour
        "redis_ttl": 7200,  # 2 hours
        "warming_enabled": True,
        "cleanup_interval": 300,  # 5 minutes
    },
}


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration, optionally overridden by environment variables.

    Returns:
        Configuration dictionary with default values and environment overrides
    """
    config = dict(DEFAULT_CONFIG)

    # Try to load environment overrides
    try:
        env_config = _load_config_from_env()
        # Deep merge environment config
        for key, value in env_config.items():
            if isinstance(value, dict) and key in config:
                existing = config[key]
                if isinstance(existing, dict):
                    existing.update(value)
                else:
                    config[key] = value
            else:
                config[key] = value
    except Exception as e:
        _logger.debug("Could not load environment config: %s", e)

    return config


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize configuration values.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Validated and normalized configuration

    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        # Static type is Dict[str, Any], but callers outside the type-checked
        # boundary (e.g. JSON-loaded config) can still pass the wrong runtime type.
        raise ValueError("Config must be a dictionary")  # pyright: ignore[reportUnreachable]

    # Validate required sections
    required_sections = ["providers", "batching", "weights", "heuristic"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate numeric values
    if config.get("lookback_hours", 0) <= 0:
        raise ValueError("lookback_hours must be positive")

    if config.get("batching", {}).get("concurrency", 0) <= 0:
        raise ValueError("batching.concurrency must be positive")

    # Normalize weights to sum to 1.0
    active_providers = [p for p, enabled in config.get("providers", {}).items() if enabled and p != "hf_enabled"]
    if not active_providers:
        return config

    weights = config.get("weights", {})
    provider_weight_sum = sum(weights.get(p, 0.0) for p in active_providers)

    if provider_weight_sum <= 0:
        # Default to equal weighting
        equal_weight = 1.0 / len(active_providers)
        for p in active_providers:
            weights[p] = equal_weight
    elif provider_weight_sum != 1.0:
        _logger.debug("Normalizing provider weights from sum %.3f to 1.0", provider_weight_sum)
        for p in active_providers:
            weights[p] = weights.get(p, 0.0) / provider_weight_sum

    return config


# -------------------------
# Helper functions
# -------------------------
def token_polarity(text: str, pos_tokens: List[str], neg_tokens: List[str]) -> int:
    """
    Calculate simple heuristic polarity based on token presence.

    Args:
        text: Text to analyze for sentiment tokens
        pos_tokens: List of positive sentiment tokens
        neg_tokens: List of negative sentiment tokens

    Returns:
        Polarity score: +1 if positive tokens dominate, -1 if negative tokens dominate, 0 otherwise
    """
    if not text:
        return 0
    t = text.lower()
    pos = sum(t.count(tok) for tok in pos_tokens)
    neg = sum(t.count(tok) for tok in neg_tokens)
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


def compute_engagement(m: Dict[str, Any]) -> float:
    """
    Compute raw engagement score from message metrics.

    Args:
        m: Message dictionary with engagement metrics (likes, replies, retweets)

    Returns:
        Engagement score calculated as: likes + 2*replies + 1.5*retweets
    """
    likes = int(m.get("likes") or 0)
    replies = int(m.get("replies") or 0)
    retweets = int(m.get("retweets") or m.get("retweets_count") or 0)
    # engagement formula: likes + 2*replies + 1.5*retweets
    return likes + 2 * replies + 1.5 * retweets


def message_weight(engagement: float, engagement_weight_formula: str = "sqrt") -> float:
    """
    Calculate message weight based on engagement score.

    Args:
        engagement: Raw engagement score
        engagement_weight_formula: Formula to use ('sqrt' or 'linear')

    Returns:
        Weighted engagement score
    """
    if engagement_weight_formula == "sqrt":
        return math.sqrt(engagement + 1.0)
    return max(1.0, engagement)


async def _call_lookup(fn: Callable[[str], Any] | None, arg: str) -> Any:
    """
    Call a caller-injected lookup callback (``history_lookup`` or ``calibration_lookup``),
    transparently supporting both sync and async callables without blocking the event loop.

    Args:
        fn: The lookup callable, or ``None`` (returns ``None`` immediately).
        arg: The single positional argument to pass (ticker or provider name).

    Returns:
        The callback's resolved return value, or ``None`` if ``fn`` is ``None``.
    """
    if fn is None:
        return None
    if inspect.iscoroutinefunction(fn):
        return await fn(arg)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, arg)


def combine_scores(heuristic: float, hf: float | None, hf_weight: float) -> float:
    """
    Combine heuristic and HuggingFace sentiment scores.

    Args:
        heuristic: Heuristic sentiment score (-1 to 1)
        hf: HuggingFace sentiment score (-1 to 1), optional
        hf_weight: Weight for HuggingFace score (0 to 1)

    Returns:
        Combined sentiment score (-1 to 1)
    """
    if hf is None:
        return heuristic
    return hf_weight * hf + (1.0 - hf_weight) * heuristic


# -------------------------
# Adapter initialization
# -------------------------
def _initialize_adapters(manager, config: Dict[str, Any]) -> None:
    """Initialize and register adapters with the manager."""
    try:
        from src.common.sentiments.adapters.adapter_manager import register_default_adapters

        register_default_adapters()

        # Register and add all enabled adapters
        for provider, enabled in config["providers"].items():
            if enabled and provider != "hf_enabled":
                # Get default config for this adapter type if available
                adapter_config = {
                    "concurrency": config["batching"]["concurrency"],
                    "rate_limit_delay": config["batching"]["rate_limit_delay_sec"],
                }

                # Special handling for Hacker News -- shared-corpus knobs, not part of the
                # generic batching config.
                if provider == "hackernews":
                    hn_config = config.get("hackernews", {})
                    adapter_config.update(
                        {
                            "corpus_lookback_hours": hn_config.get("corpus_lookback_hours", 48),
                            "max_depth": hn_config.get("max_depth", 4),
                            "max_items_per_thread": hn_config.get("max_items_per_thread", 300),
                            "rate_limit_rps": hn_config.get("rate_limit_rps", 10.0),
                            "hn_corpus_ttl_seconds": hn_config.get("hn_corpus_ttl_seconds", 1800),
                            "entity_map_path": hn_config.get("entity_map_path"),
                        }
                    )

                # Special handling for Bluesky -- auth/search knobs, not part of the generic
                # batching config. handle/app_password default to None here so the adapter falls
                # back to config.donotshare.donotshare.BLUESKY_HANDLE/BLUESKY_APP_PASSWORD itself.
                if provider == "bluesky":
                    bsky_config = config.get("bluesky", {})
                    adapter_config.update(
                        {
                            "lang_filter": bsky_config.get("lang_filter", "en"),
                            "max_posts_per_ticker": bsky_config.get("max_posts_per_ticker", 200),
                            "search_terms": tuple(bsky_config.get("search_terms", ["cashtag", "company_name"])),
                            "entity_map_path": bsky_config.get("entity_map_path"),
                        }
                    )

                manager.add_adapter(provider, adapter_config)

        # HuggingFace enhancement -- gated on hf_enabled, not part of the provider loop above:
        # it enhances messages already fetched from retail/tech_discourse providers rather than
        # being a message source itself. (This used to be special-cased *inside* the provider
        # loop keyed on a "huggingface" entry that never existed in config["providers"], so the
        # adapter was never actually added and the whole enhancement path was silently dead
        # whenever hf_enabled=True -- fixed here while wiring per-source model routing.)
        if config["providers"].get("hf_enabled", False):
            hf_config = config.get("hf", {})
            models = hf_config.get("models", {})
            long_text_strategy = hf_config.get("long_text_strategy", "truncate")
            device = hf_config.get("device", -1)
            max_workers = hf_config.get("max_workers", 1)

            # Per-source model routing (spec §2.5.4): twitter-roberta is tuned for short informal
            # posts and degrades on Hacker News' long-form technical prose.
            retail_model = models.get("retail") or hf_config.get(
                "model_name", "cardiffnlp/twitter-roberta-base-sentiment"
            )
            manager.add_adapter(
                "huggingface",
                {
                    "model_name": retail_model,
                    "device": device,
                    "max_workers": max_workers,
                    "long_text_strategy": long_text_strategy,
                },
            )

            # Only loaded when Hacker News is actually enabled -- avoids paying for a second
            # heavy model when tech_discourse isn't being collected at all.
            if config["providers"].get("hackernews", False):
                tech_model = models.get("tech_discourse", "distilbert-base-uncased-finetuned-sst-2-english")
                manager.add_adapter(
                    "huggingface_tech",
                    {
                        "model_name": tech_model,
                        "device": device,
                        "max_workers": max_workers,
                        "long_text_strategy": long_text_strategy,
                    },
                )

    except Exception as e:
        _logger.warning("Could not initialize some adapters: %s", e)


# -------------------------
# Cache integration
# -------------------------
def _initialize_cache(config: Dict[str, Any]):
    """Initialize cache manager with configuration."""
    try:
        from src.common.sentiments.caching.cache_manager import CacheConfig, CacheManager

        # Create cache config from sentiment config
        cache_config = CacheConfig(
            redis_enabled=config.get("caching", {}).get("redis_enabled", True),
            redis_host=config.get("caching", {}).get("redis_host", "localhost"),
            redis_port=config.get("caching", {}).get("redis_port", 6379),
            redis_db=config.get("caching", {}).get("redis_db", 0),
            redis_password=config.get("caching", {}).get("redis_password"),
            memory_max_size=config.get("caching", {}).get("memory_max_size", 1000),
            memory_default_ttl=config.get("caching", {}).get("memory_ttl", 3600),
            redis_default_ttl=config.get("caching", {}).get("redis_ttl", 7200),
        )

        return CacheManager(cache_config)
    except ImportError as e:
        _logger.warning("Could not initialize cache manager: %s", e)
        return None


# -------------------------
# Core async collector
# -------------------------
async def collect_sentiment_batch(
    tickers: List[str],
    lookback_hours: int | None = None,
    config: Dict[str, Any] | None = None,
    history_lookup: Callable[[str], float | None] | Callable[[str], Awaitable[float | None]] | None = None,
    calibration_lookup: Callable[[str], "CalibrationStats | None"]
    | Callable[[str], Awaitable["CalibrationStats | None"]]
    | None = None,
    output_format: str = "dataclass",
) -> Union[Dict[str, SentimentFeatures | None], Dict[str, Dict[str, Any] | None], str]:
    """
    Collect sentiment features for a list of tickers concurrently.

    This is the main entry point for sentiment analysis. It supports multiple data providers,
    configurable output formats, and comprehensive error handling.

    Args:
        tickers: List of ticker symbols (will be normalized to uppercase)
        lookback_hours: Hours to look back for data (default: from config)
        config: Configuration dictionary (default: get_default_config())
        history_lookup: Optional function to get historical mention averages for growth calculation
        calibration_lookup: Optional function, keyed by provider name, returning that provider's
            pooled trailing-window ``CalibrationStats`` (or ``None`` if there's no history yet).
            Mirrors ``history_lookup``'s injection pattern so this module stays DB-agnostic --
            callers (e.g. ``daily_deep_scan.py``) own persistence of the trailing distribution in
            ``ss_sentiment_calibration`` and inject a read of it here. When omitted, retail scores
            are blended raw (spec §2.5.6's calibration step is skipped entirely, not silently
            no-op'd -- ``data_quality["calibration"]`` is left out of the output in that case).
        output_format: Output format - "dataclass", "dict", or "json"

    Returns:
        Dictionary mapping tickers to sentiment features, format depends on output_format:
        - "dataclass": Dict[str, Optional[SentimentFeatures]]
        - "dict": Dict[str, Optional[Dict[str, Any]]]
        - "json": JSON string

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If no adapters are available
    """
    # Validate inputs
    if not tickers:
        return {} if output_format != "json" else "{}"

    if output_format not in ["dataclass", "dict", "json"]:
        raise ValueError("output_format must be 'dataclass', 'dict', or 'json'")

    # Get and validate configuration
    if config is None:
        config = get_default_config()
    else:
        # Merge with defaults
        default_config = get_default_config()
        for key, value in config.items():
            if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
        config = default_config

    config = validate_config(config)

    # Initialize adapter manager and cache
    from src.common.sentiments.adapters.adapter_manager import get_adapter_manager

    manager = get_adapter_manager()
    cache_manager = _initialize_cache(config)

    try:
        _initialize_adapters(manager, config)

        # Start background monitoring tasks (global rate-limit coordination).
        # Must be awaited here — start() is intentionally not called in __init__
        # so the manager can be safely instantiated from synchronous code.
        await manager.start()

        # Check if any adapters are available
        available_adapters = manager.get_available_adapters()
        if not available_adapters:
            raise RuntimeError("No sentiment adapters are available")

        _logger.info("Using adapters: %s", available_adapters)

        # Initialize cache key strategy
        from src.common.sentiments.caching.cache_manager import CacheKeyStrategy

        cache_keys = CacheKeyStrategy()
        config_hash = cache_keys.config_hash(config)

        # Initialize performance optimization (needed for decorator below)
        from src.common.sentiments.performance.performance_profiler import (
            PerformanceProfiler,
            ProfilerConfig,
        )

        # process_one_ticker fans out concurrent HTTP calls to multiple external sentiment
        # providers (each allowed up to 180s before it's treated as a timeout -- see
        # fetch_one_summary below), so the profiler's generic 5s bottleneck_threshold_seconds
        # default fires on ordinary network latency. Raise it so the WARNING only fires when a
        # ticker is genuinely stalling, not on routine multi-provider fetch variance.
        profiler = PerformanceProfiler(config=ProfilerConfig(bottleneck_threshold_seconds=60.0))

        # Extract configuration values
        lookback = lookback_hours or config.get("lookback_hours", 24)
        concurrency = config["batching"]["concurrency"]
        min_mentions_hf = config.get("min_mentions_for_hf", 20)
        weights = config["weights"]
        heuristic_config = config["heuristic"]

        # Batch-level observability accumulators (spec §2.10) -- populated synchronously inside
        # process_one_ticker (safe under asyncio's cooperative scheduling: no `await` splits the
        # read-modify-write), logged once after the whole batch completes rather than per ticker.
        provider_available_tickers: Dict[str, int] = {}
        provider_zero_mentions: Dict[str, int] = {}

        # Precompute calibration stats once per provider, not once per (ticker, provider) --
        # calibration history is provider-level, not ticker-level, so this also gives a clean
        # point to log sentiment.calibration.status{provider} once for the whole batch instead of
        # once per ticker (spec §2.10).
        calibration_config = config.get("calibration", {})
        calibration_enabled = calibration_config.get("enabled", True) and calibration_lookup is not None
        min_calibration_obs = calibration_config.get("min_observations", 200)
        calibration_stats_by_provider: Dict[str, CalibrationStats | None] = {}
        if calibration_enabled:
            retail_providers_for_calibration = [
                p
                for p, enabled in config["providers"].items()
                if enabled and p != "hf_enabled" and manager.get_signal_class(p) == "retail"
            ]
            for provider in retail_providers_for_calibration:
                try:
                    stats = await _call_lookup(calibration_lookup, provider)
                except Exception as e:
                    _logger.debug("Calibration lookup failed for %s: %s", provider, e)
                    stats = None
                calibration_stats_by_provider[provider] = stats
                _logger.info(
                    "sentiment.calibration.status provider=%s status=%s",
                    provider,
                    calibration_status(stats, min_calibration_obs),
                )

        # Concurrency semaphore
        sem = asyncio.Semaphore(concurrency)

        @profiler.time_function("process_one_ticker")
        async def process_one_ticker(ticker: str) -> SentimentFeatures | None:
            """Process sentiment analysis for a single ticker."""
            async with sem:
                tk = ticker.upper().strip()
                if not tk:
                    return None

                data_quality: Dict[str, str] = {}
                raw_payload: Dict[str, Any] = {}

                try:
                    since_ts = int((datetime.now(UTC) - timedelta(hours=lookback)).timestamp())

                    # Check cache for aggregated result first
                    aggregated_cache_key = cache_keys.aggregated_sentiment_key(tk, lookback, config_hash)
                    if cache_manager:
                        cached_result = cache_manager.get(aggregated_cache_key)
                        if cached_result:
                            _logger.debug("Cache hit for aggregated sentiment: %s", tk)
                            return cached_result

                    # Collect summaries from available adapters dynamically
                    summaries = {}
                    active_providers = [
                        p for p, enabled in config["providers"].items() if enabled and p != "hf_enabled"
                    ]

                    async def fetch_one_summary(provider):
                        try:
                            # Check cache first
                            cache_key = cache_keys.sentiment_summary_key(tk, since_ts, provider)
                            summary = None
                            if cache_manager:
                                summary = cache_manager.get(cache_key)

                            if not summary:
                                try:
                                    # Add timeout to prevent one slow adapter from hanging the whole process
                                    summary = await asyncio.wait_for(
                                        manager.fetch_summary_from_adapter(provider, tk, since_ts), timeout=180.0
                                    )
                                except TimeoutError:
                                    _logger.warning("%s summary timed out for %s", provider.capitalize(), tk)
                                    summary = {"error": "timeout", "mentions": 0, "sentiment_score": 0.0}

                                if summary and cache_manager:
                                    cache_manager.set(cache_key, summary, 1800)  # 30 min TTL

                            return provider, summary
                        except Exception as e:
                            _logger.debug("%s summary error for %s: %s", provider.capitalize(), tk, e)
                            return provider, {"error": str(e), "mentions": 0, "sentiment_score": 0.0}

                    # Fetch all summaries concurrently
                    fetch_tasks = [fetch_one_summary(p) for p in active_providers if p in available_adapters]
                    fetched_results = await asyncio.gather(*fetch_tasks)

                    for provider, summary in fetched_results:
                        if summary:
                            summaries[provider] = summary
                            if "error" in summary:
                                data_quality[provider] = f"error: {summary['error']}"
                            else:
                                data_quality[provider] = "ok"
                                # sentiment.coverage.tickers_with_zero_mentions{provider} (spec
                                # §2.10) -- "watch this one first". Accumulated per-batch, logged
                                # once after all tickers are processed.
                                provider_available_tickers[provider] = provider_available_tickers.get(provider, 0) + 1
                                if summary.get("mentions", 0) == 0:
                                    provider_zero_mentions[provider] = provider_zero_mentions.get(provider, 0) + 1
                        else:
                            data_quality[provider] = "missing"

                    raw_payload.update(summaries)

                    # Split providers by signal_class -- retail and tech_discourse are never
                    # blended into the same score (spec §2.5.6). Routing is by adapter class,
                    # never by adapter name (spec §2.2's invariant).
                    retail_summaries = {p: s for p, s in summaries.items() if manager.get_signal_class(p) == "retail"}
                    tech_summaries = {
                        p: s for p, s in summaries.items() if manager.get_signal_class(p) == "tech_discourse"
                    }

                    # Aggregate retail basic metrics
                    total_mentions = 0
                    unique_authors = 0
                    weighted_sentiment = 0.0
                    total_weight = 0.0
                    calibration_insufficient = False

                    for provider, summary in retail_summaries.items():
                        mentions = summary.get("mentions", 0)
                        total_mentions += mentions

                        if provider in ("reddit", "bluesky"):
                            unique_authors += summary.get("unique_authors", 0)

                        # Weight sentiment scores. Calibrate the raw per-provider score against
                        # its own trailing distribution before blending (spec §2.5.6) -- raw
                        # scores aren't comparable across platforms (Bluesky finance chatter
                        # skews promotional-positive), so blending them directly would import
                        # that platform bias straight into sentiment_24h. Stats are precomputed
                        # once per provider for the whole batch, not refetched per ticker.
                        raw_sentiment = summary.get("sentiment_score", 0.0)
                        sentiment = raw_sentiment
                        if calibration_enabled:
                            stats = calibration_stats_by_provider.get(provider)
                            if calibration_status(stats, min_calibration_obs) == "insufficient_history":
                                calibration_insufficient = True
                            sentiment = calibrate_score(raw_sentiment, stats, min_calibration_obs)

                        weight = weights.get(provider, 0.0)
                        weighted_sentiment += sentiment * weight
                        total_weight += weight

                    # Calculate combined sentiment. Weights are renormalized implicitly here --
                    # total_weight only ever sums the weights of providers that actually
                    # responded, so an outage doesn't silently halve the score (spec §2.5.6).
                    if total_weight > 0:
                        combined_sentiment = weighted_sentiment / total_weight
                    else:
                        combined_sentiment = 0.0

                    # Clamp to the schema's -1..+1 range. When calibration is active this also
                    # folds the (unbounded) z-score blend back into the documented range.
                    combined_sentiment = max(-1.0, min(1.0, combined_sentiment))

                    if calibration_enabled:
                        data_quality["calibration"] = "insufficient_history" if calibration_insufficient else "ok"

                    # Aggregate tech_discourse basic metrics. tech_coverage_available stays None
                    # when no tech_discourse provider returned a summary at all (distinct from
                    # "covered, zero mentions"); every other tech_* value stays None until it is
                    # confirmed True, never defaulted to a fabricated neutral reading (spec §2.4).
                    tech_coverage_available: bool | None = None
                    tech_total_mentions = 0
                    tech_weighted_sentiment = 0.0
                    tech_discussion_depths: List[float] = []

                    for summary in tech_summaries.values():
                        coverage = summary.get("tech_coverage_available")
                        if coverage is not None:
                            tech_coverage_available = coverage if tech_coverage_available is None else (
                                tech_coverage_available or coverage
                            )
                        mentions = summary.get("mentions", 0)
                        tech_total_mentions += mentions
                        tech_weighted_sentiment += summary.get("sentiment_score", 0.0) * mentions
                        if "discussion_depth" in summary:
                            tech_discussion_depths.append(summary["discussion_depth"])

                    if tech_coverage_available:
                        tech_combined_sentiment: float | None = (
                            max(-1.0, min(1.0, tech_weighted_sentiment / tech_total_mentions))
                            if tech_total_mentions > 0
                            else 0.0
                        )
                        tech_discussion_depth: float | None = (
                            sum(tech_discussion_depths) / len(tech_discussion_depths) if tech_discussion_depths else 0.0
                        )
                    else:
                        tech_combined_sentiment = None
                        tech_discussion_depth = None

                    # Enhanced analysis with HuggingFace if enabled and threshold met. Retail uses
                    # the "huggingface" adapter (twitter-roberta by default); tech_discourse uses
                    # its own "huggingface_tech" adapter/model (distilbert by default) and its own
                    # tech_discourse lexicon -- never the retail model/lexicon, which would
                    # silently reintroduce the exact cross-source bias this spec exists to remove
                    # (spec §2.5.3/§2.5.4).
                    enhanced_sentiment = combined_sentiment
                    positive_ratio = None
                    bot_pct: float | None = 0.0
                    virality_index = 0.0
                    hf_enabled = config["providers"].get("hf_enabled", False)

                    if total_mentions >= min_mentions_hf and "huggingface" in available_adapters and hf_enabled:
                        try:
                            # Fetch detailed messages for HF analysis from retail providers only
                            all_messages = []

                            fetch_msg_tasks = [
                                manager.fetch_messages_from_adapter(p, tk, since_ts, 150) for p in retail_summaries
                            ]

                            results = await asyncio.gather(*fetch_msg_tasks, return_exceptions=True)
                            for res in results:
                                if isinstance(res, list):
                                    all_messages.extend(res)

                            if all_messages:
                                # Check cache for HF predictions
                                texts = [msg.get("body", "") for msg in all_messages if msg.get("body")]
                                hf_cache_key = cache_keys.hf_predictions_key(texts[:100])  # Limit for key size

                                cached_hf_result = None
                                if cache_manager:
                                    cached_hf_result = cache_manager.get(hf_cache_key)

                                if cached_hf_result:
                                    enhanced_sentiment, positive_ratio, bot_pct, virality_index = cached_hf_result
                                    _logger.debug("Cache hit for HF predictions: %s", tk)
                                else:
                                    # Process with HuggingFace
                                    (
                                        enhanced_sentiment,
                                        positive_ratio,
                                        bot_pct,
                                        virality_index,
                                    ) = await _process_messages_with_hf(
                                        all_messages,
                                        manager,
                                        heuristic_config,
                                        weights.get("heuristic_vs_hf", 0.5),
                                        hf_adapter_name="huggingface",
                                    )
                                    # Cache HF results
                                    if cache_manager:
                                        hf_result = (enhanced_sentiment, positive_ratio, bot_pct, virality_index)
                                        cache_manager.set(hf_cache_key, hf_result, 3600)  # 1 hour TTL

                                data_quality["huggingface"] = "ok"
                            else:
                                data_quality["huggingface"] = "no_messages"

                        except Exception as e:
                            _logger.warning("HuggingFace processing failed for %s: %s", tk, e)
                            data_quality["huggingface"] = "failed"

                    # Tech_discourse HF enhancement -- symmetric to the retail block above but
                    # routed through "huggingface_tech" and skipping bot detection entirely (HN is
                    # heavily human-moderated with negligible automated posting; applying
                    # Bluesky-style thresholds there would misflag prolific legitimate commenters,
                    # spec §2.5.2). Only runs for tickers actually covered by the entity map.
                    if (
                        tech_coverage_available
                        and tech_total_mentions >= min_mentions_hf
                        and "huggingface_tech" in available_adapters
                        and hf_enabled
                    ):
                        try:
                            tech_fetch_tasks = [
                                manager.fetch_messages_from_adapter(p, tk, since_ts, 150) for p in tech_summaries
                            ]
                            tech_results = await asyncio.gather(*tech_fetch_tasks, return_exceptions=True)
                            tech_messages: List[Dict[str, Any]] = []
                            for res in tech_results:
                                if isinstance(res, list):
                                    tech_messages.extend(res)

                            if tech_messages:
                                (
                                    tech_hf_sentiment,
                                    _tech_positive_ratio,
                                    _tech_bot_pct,
                                    _tech_virality,
                                ) = await _process_messages_with_hf(
                                    tech_messages,
                                    manager,
                                    heuristic_config,
                                    weights.get("heuristic_vs_hf", 0.5),
                                    hf_adapter_name="huggingface_tech",
                                    skip_bot_detection=True,
                                )
                                tech_combined_sentiment = tech_hf_sentiment
                                data_quality["huggingface_tech"] = "ok"
                            else:
                                data_quality["huggingface_tech"] = "no_messages"

                        except Exception as e:
                            _logger.warning("HuggingFace (tech_discourse) processing failed for %s: %s", tk, e)
                            data_quality["huggingface_tech"] = "failed"

                    # Calculate mentions growth if history lookup provided. Reddit-era baselines
                    # no longer exist (spec §2.5.5) -- callers should have their history_lookup
                    # return None for tickers with no post-migration history, which naturally
                    # falls through here as "growth stays None" rather than a fabricated 1.0.
                    mentions_growth = None
                    if history_lookup and total_mentions > 0:
                        try:
                            prev_avg = cast("float | None", await _call_lookup(history_lookup, tk))
                            if prev_avg and prev_avg > 0:
                                mentions_growth = total_mentions / prev_avg
                        except Exception as e:
                            _logger.debug("History lookup failed for %s: %s", tk, e)

                    # Create final features. bot_pct is retail-only in the output schema (spec
                    # §1.2 -- Hacker News has no bot_pct field at all, skip_bot_detection=True on
                    # the retail call never applies), so the retail HF block above always returns
                    # a float here; this narrows the shared helper's float|None return type.
                    if bot_pct is None:
                        bot_pct = 0.0

                    sentiment_normalized = max(0.0, min(1.0, (enhanced_sentiment + 1.0) / 2.0))

                    tech_sentiment_normalized = (
                        max(0.0, min(1.0, (tech_combined_sentiment + 1.0) / 2.0))
                        if tech_combined_sentiment is not None
                        else None
                    )

                    features = SentimentFeatures(
                        ticker=tk,
                        mentions_24h=total_mentions,
                        unique_authors_24h=unique_authors,
                        mentions_growth_7d=mentions_growth,
                        positive_ratio_24h=positive_ratio,
                        sentiment_score_24h=float(enhanced_sentiment),
                        sentiment_normalized=float(sentiment_normalized),
                        virality_index=float(virality_index),
                        bot_pct=float(bot_pct),
                        data_quality=data_quality,
                        raw_payload=raw_payload,
                        tech_mentions_24h=tech_total_mentions if tech_coverage_available else None,
                        tech_sentiment_score_24h=tech_combined_sentiment,
                        tech_sentiment_normalized=tech_sentiment_normalized,
                        tech_discussion_depth=tech_discussion_depth,
                        tech_coverage_available=tech_coverage_available,
                    )

                    # Cache the final aggregated result
                    if cache_manager:
                        cache_manager.set(aggregated_cache_key, features, 1800)  # 30 min TTL

                    return features

                except Exception as e:
                    _logger.exception("Error processing ticker %s: %s", tk, e)
                    return None

        # Initialize batch optimizer
        from src.common.sentiments.performance.batch_optimizer import BatchOptimizer

        batch_optimizer = BatchOptimizer()

        # Process tickers in optimized batches
        _logger.info("Processing %d tickers with batch optimization", len(tickers))

        # Create optimized batches
        ticker_batches = batch_optimizer.create_batches(tickers, "sentiment_collection")

        # Process batches in parallel
        async def process_ticker_batch(ticker_batch: List[str]) -> List[SentimentFeatures | None]:
            """Process a batch of tickers."""
            batch_tasks = [asyncio.create_task(process_one_ticker(ticker)) for ticker in ticker_batch]
            raw_res = await asyncio.gather(*batch_tasks, return_exceptions=True)
            res: List[SentimentFeatures | None] = []
            for item in raw_res:
                if isinstance(item, BaseException):
                    _logger.error("Error processing ticker: %s", item)
                    res.append(None)
                else:
                    res.append(item)
            return res

        batch_results = await batch_optimizer.process_batches_parallel(
            ticker_batches, process_ticker_batch, "sentiment_collection"
        )

        # Flatten results
        results = []
        for batch_result in batch_results:
            if batch_result:
                results.extend(batch_result)

        # Compile results
        output: Dict[str, SentimentFeatures | None] = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                _logger.error("Exception processing ticker %s: %s", ticker, result)
                output[ticker.upper()] = None
            else:
                output[ticker.upper()] = result

        # Batch-level observability summary (spec §2.10), logged once per collect_sentiment_batch
        # call -- mirrors how daily_deep_scan.py already logs its own batch summaries. No
        # Prometheus in this repo; these are structured log lines, not metrics-server counters.
        for provider in sorted(set(provider_available_tickers) | set(provider_zero_mentions)):
            _logger.info(
                "sentiment.coverage.tickers_with_zero_mentions provider=%s zero=%d available=%d batch_size=%d",
                provider,
                provider_zero_mentions.get(provider, 0),
                provider_available_tickers.get(provider, 0),
                len(tickers),
            )
        _logger.info("sentiment.blend.providers_available providers=%s", sorted(provider_available_tickers.keys()))

        hn_adapter = manager._adapters.get("hackernews")
        get_hn_stats = getattr(hn_adapter, "get_observability_stats", None) if hn_adapter is not None else None
        if get_hn_stats is not None:
            hn_stats = get_hn_stats()
            if hn_stats:
                _logger.info(
                    "sentiment.hn.corpus_size=%d sentiment.hn.entity_match_rate=%.4f",
                    hn_stats["corpus_size"],
                    hn_stats["entity_match_rate"],
                )

        bluesky_adapter = manager._adapters.get("bluesky")
        if bluesky_adapter is not None:
            _logger.info(
                "sentiment.bluesky.auth_refresh_count=%d sentiment.bluesky.pagination_fallback_count=%d",
                getattr(bluesky_adapter, "auth_refresh_count", 0),
                getattr(bluesky_adapter, "pagination_fallback_count", 0),
            )

        # Format output based on requested format
        if output_format == "dataclass":
            return output
        elif output_format == "dict":
            return {k: v.to_dict() if v else None for k, v in output.items()}
        else:  # "json" — output_format was validated above
            dict_output = {k: v.to_dict() if v else None for k, v in output.items()}
            return json.dumps(dict_output, default=str, indent=2)

    finally:
        # Clean up adapter manager and cache
        await manager.close_all()
        if cache_manager:
            # Perform cleanup if needed
            if cache_manager.should_cleanup():
                cleanup_results = cache_manager.cleanup_expired()
                _logger.debug("Cache cleanup: %s", cleanup_results)

            # Report metrics if enabled
            if cache_manager._metrics:
                cache_manager._metrics.report_metrics()

        # Report performance metrics
        if "profiler" in locals():
            profiler.auto_report()

        # Report batch optimization stats
        if "batch_optimizer" in locals():
            perf_summary = batch_optimizer.get_performance_summary()
            if perf_summary:
                _logger.info("Batch processing summary: %s", perf_summary)


def _percentile_ranks(values: List[float]) -> List[float]:
    """
    Rank each value's percentile position (0..1) within ``values``, using average rank for ties.

    Backs ``normalized_engagement`` (spec §2.5.5): "a per-source percentile rank, not a raw
    count" -- an HN story score of 300 and a Bluesky like count of 300 are not comparable
    quantities, but "this message is in the top 10% of engagement for this batch" is.
    """
    n = len(values)
    if n <= 1:
        return [1.0] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # Average rank (0-indexed) across the tied run, then scale to 0..1.
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank / (n - 1)
        i = j + 1
    return ranks


async def _process_messages_with_hf(
    messages: List[Dict[str, Any]],
    manager,
    heuristic_config: Dict[str, Any],
    hf_weight: float,
    hf_adapter_name: str = "huggingface",
    skip_bot_detection: bool = False,
) -> tuple[float, float | None, float | None, float]:
    """
    Score messages with HuggingFace + heuristic sentiment and compute reach/bot metrics.

    Args:
        messages: Normalized messages from one or more adapters of the *same* signal class
            (retail or tech_discourse) -- callers must not mix classes in one call (spec §2.5.6).
        manager: The adapter manager, used to reach the HF adapter instance.
        heuristic_config: ``config["heuristic"]`` -- token lists and engagement formula name.
        hf_weight: Blend weight for the HF score vs. the heuristic score (0..1).
        hf_adapter_name: Which registered HF adapter instance to use -- ``"huggingface"``
            (retail-tuned model) or ``"huggingface_tech"`` (tech_discourse-tuned model), per the
            per-source model routing in spec §2.5.4.
        skip_bot_detection: True for Hacker News -- HN is heavily human-moderated with negligible
            automated posting, so Bluesky-style thresholds would misflag prolific legitimate
            commenters (spec §2.5.2). Returns ``bot_pct=None`` (not 0.0) in that case.

    Returns:
        ``(sentiment_score, positive_ratio, bot_pct, virality_index)``. ``sentiment_score`` and
        ``virality_index`` are computed as orthogonal quantities (spec §2.5.5) -- the former is
        signed direction only, the latter is unsigned reach only; they are no longer conflated
        into a single formula the way Rev 1 did.
    """
    if not messages:
        return 0.0, None, None if skip_bot_detection else 0.0, 0.0

    # Extract text for HF processing
    texts = []
    message_data = []

    for msg in messages:
        body = str(msg.get("body", "")).strip()
        if body:
            texts.append(body)
            message_data.append(msg)

    if not texts:
        return 0.0, None, None if skip_bot_detection else 0.0, 0.0

    # Get HF predictions from the per-source-routed model
    try:
        hf_predictions = await manager._adapters[hf_adapter_name].predict_batch(texts)
    except Exception as e:
        _logger.warning("HF prediction failed (%s): %s", hf_adapter_name, e)
        return 0.0, None, None if skip_bot_detection else 0.0, 0.0

    pos_tokens = heuristic_config.get("positive_tokens", [])
    neg_tokens = heuristic_config.get("negative_tokens", [])

    # normalized_engagement is a per-batch percentile rank, not a raw count (spec §2.5.5) -- an
    # HN story score and a Bluesky like count are not comparable quantities, but "top decile of
    # engagement in this batch" is.
    engagements = [compute_engagement(msg) for msg in message_data]
    engagement_percentiles = _percentile_ranks(engagements)

    positive_count = 0
    negative_count = 0
    bot_count = 0
    weighted_sentiment_sum = 0.0
    total_weight = 0.0
    reach_sum = 0.0
    unique_authors: set[str] = set()

    for msg, hf_pred, engagement, norm_engagement in zip(message_data, hf_predictions, engagements, engagement_percentiles):
        # Convert HF prediction to sentiment score
        label = hf_pred.get("label", "").upper()
        if "POS" in label or "POSITIVE" in label or "LABEL_2" in label:
            hf_sentiment = 1.0
        elif "NEG" in label or "NEGATIVE" in label or "LABEL_0" in label:
            hf_sentiment = -1.0
        else:
            hf_sentiment = 0.0

        # Calculate heuristic sentiment
        heuristic_sentiment = float(token_polarity(msg.get("body", ""), pos_tokens, neg_tokens))

        # Combine HF and heuristic
        combined_sentiment = hf_weight * hf_sentiment + (1.0 - hf_weight) * heuristic_sentiment

        # Provider-native bot signal, set at the adapter boundary before the raw author identity
        # is hashed away (meta.is_bot -- e.g. Bluesky's post-volume/account-age heuristic,
        # StockTwits/Reddit/Twitter/Discord's username-shape heuristic). HN skips bot detection
        # entirely (spec §2.5.2).
        is_bot = bool(msg.get("meta", {}).get("is_bot")) if not skip_bot_detection else False
        if is_bot:
            bot_count += 1

        # message weight = sqrt(normalized_engagement + 1) * author_trust (spec §2.5.5).
        # author_trust: suspected bots 0.2, HN uniformly 1.0 (bot detection skipped), otherwise
        # 1.0 -- no per-author quality signal survives the salted-hash boundary to derive a finer
        # 0.5..1.0 trust score from.
        author_trust = 1.0 if skip_bot_detection else (0.2 if is_bot else 1.0)
        weight = math.sqrt(norm_engagement + 1.0) * author_trust

        weighted_sentiment_sum += combined_sentiment * weight
        total_weight += weight

        if combined_sentiment > 0:
            positive_count += 1
        elif combined_sentiment < 0:
            negative_count += 1

        reach_sum += engagement
        author_id = msg.get("user", {}).get("id")
        if author_id:
            unique_authors.add(str(author_id))

    # sentiment_score = Σ(polarity * weight) / Σ(weight) -- direction only.
    final_sentiment = weighted_sentiment_sum / total_weight if total_weight > 0 else 0.0
    positive_ratio = (
        positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else None
    )
    bot_percentage = None if skip_bot_detection else (bot_count / len(message_data) if message_data else 0.0)
    # virality_index = Σ(engagement) / sqrt(unique_authors + 1) -- reach only, unsigned. No longer
    # multiplied by |sentiment| (Rev 1's conflation: a viral negative post and a quiet positive
    # one produced the same value) -- see spec §2.5.5.
    virality_index = reach_sum / math.sqrt(len(unique_authors) + 1)

    return final_sentiment, positive_ratio, bot_percentage, virality_index


# -------------------------
# Sync wrapper for convenience
# -------------------------
def collect_sentiment_batch_sync(
    *args, **kwargs
) -> Union[Dict[str, SentimentFeatures | None], Dict[str, Dict[str, Any] | None], str]:
    """
    Sync wrapper for callers that don't use asyncio.

    This function runs the async collect_sentiment_batch in a new event loop,
    making it accessible from synchronous code.

    Args:
        *args: Positional arguments passed to collect_sentiment_batch
        **kwargs: Keyword arguments passed to collect_sentiment_batch

    Returns:
        Same as collect_sentiment_batch, format depends on output_format parameter
    """
    return asyncio.run(collect_sentiment_batch(*args, **kwargs))


# -------------------------
# Small helper: convert features to dict for DB storage
# -------------------------
def features_to_record(f: SentimentFeatures) -> Dict[str, Any]:
    rec = asdict(f)
    # convert raw_payload to JSON string where necessary
    rec["raw_payload_json"] = json.dumps(rec.pop("raw_payload", {}), default=str)
    rec["data_quality_json"] = json.dumps(rec.pop("data_quality", {}), default=str)
    return rec


if __name__ == "__main__":
    import pprint

    # 1. Define tickers to test
    test_tickers = ["AAPL", "TSLA", "NVDA", "BTC"]

    # 2. Load configuration
    cfg = get_default_config()

    # 3. Enable some providers explicitly for testing if not already enabled
    # Actually, we'll just use the default which we've already updated

    print(f"--- Running Quick Sentiment Test for {test_tickers} ---")
    print(f"Lookback: {cfg.get('lookback_hours', 24)}h")

    # 4. Run collection
    results = collect_sentiment_batch_sync(
        tickers=test_tickers, lookback_hours=24, config=cfg, output_format="dataclass"
    )

    if isinstance(results, dict):
        # 5. Print summary
        print("\n--- Results Summary ---")
        for ticker, features in results.items():
            if isinstance(features, SentimentFeatures):
                print(f"\n[{ticker}]")
                print(
                    f"  Sentiment Score: {features.sentiment_score_24h:.4f} (Normalized: {features.sentiment_normalized:.4f})"
                )
                print(f"  Total Mentions: {features.mentions_24h}")
                print(f"  Virality Index: {features.virality_index:.2f}")
                print(f"  Data Quality: {features.data_quality}")

                # Diagnostic for missing providers
                missing = [p for p, q in features.data_quality.items() if q == "missing"]
                if missing:
                    print(f"  [!] Missing data from: {missing}")
                    # Check raw_payload for hints
                    for p in missing:
                        if p in features.raw_payload and "error" in features.raw_payload[p]:
                            print(f"      - {p} error: {features.raw_payload[p]['error']}")
            else:
                print(f"\n[{ticker}] Failed to collect sentiment.")

        print("\n--- Raw Payload Sample (AAPL) ---")
        aapl_feat = results.get("AAPL")
        if isinstance(aapl_feat, SentimentFeatures):
            pprint.pprint(aapl_feat.raw_payload)
