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
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable, Any, Union
from datetime import datetime, timedelta, timezone
import math
import json
from pathlib import Path
import sys
import os

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

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
    mentions_growth_7d: Optional[float]
    positive_ratio_24h: Optional[float]
    sentiment_score_24h: float       # -1..+1
    sentiment_normalized: float     # 0..1 mapped for scoring
    virality_index: float
    bot_pct: float                  # 0..1
    data_quality: Dict[str, str]    # provider -> 'ok'|'partial'|'missing'|'hf_disabled'|'hf_failed'
    raw_payload: Dict[str, Any]     # raw provider payloads for audit

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
    config = {}

    # Provider settings
    config["providers"] = {
        "stocktwits": os.getenv("SENTIMENT_STOCKTWITS_ENABLED", "true").lower() == "true",
        "reddit": os.getenv("SENTIMENT_REDDIT_ENABLED", "true").lower() == "true",
        "reddit_pushshift": os.getenv("SENTIMENT_PUSHSHIFT_ENABLED", "false").lower() == "true",
        "hf_enabled": os.getenv("SENTIMENT_HF_ENABLED", "false").lower() == "true"
    }

    # Timing settings
    config["lookback_hours"] = int(os.getenv("SENTIMENT_LOOKBACK_HOURS", "24"))
    config["min_mentions_for_hf"] = int(os.getenv("SENTIMENT_MIN_MENTIONS_HF", "20"))
    config["min_mentions_for_confident_signal"] = int(os.getenv("SENTIMENT_MIN_MENTIONS_SIGNAL", "5"))

    # HuggingFace settings
    config["hf"] = {
        "model_name": os.getenv("SENTIMENT_HF_MODEL", "cardiffnlp/twitter-roberta-base-sentiment"),
        "device": int(os.getenv("SENTIMENT_HF_DEVICE", "-1")),
        "max_workers": int(os.getenv("SENTIMENT_HF_WORKERS", "1"))
    }

    # Batching settings
    config["batching"] = {
        "concurrency": int(os.getenv("SENTIMENT_CONCURRENCY", "8")),
        "rate_limit_delay_sec": float(os.getenv("SENTIMENT_RATE_DELAY", "0.3"))
    }

    # Provider weights
    config["weights"] = {
        "stocktwits": float(os.getenv("SENTIMENT_WEIGHT_STOCKTWITS", "0.4")),
        "reddit": float(os.getenv("SENTIMENT_WEIGHT_REDDIT", "0.6")),
        "heuristic_vs_hf": float(os.getenv("SENTIMENT_WEIGHT_HF", "0.5"))
    }

    # Heuristic settings
    positive_tokens = os.getenv("SENTIMENT_POSITIVE_TOKENS", "moon,🚀,diamond,buy,long,hold,to the moon,rocket")
    negative_tokens = os.getenv("SENTIMENT_NEGATIVE_TOKENS", "short,sell,dump,bankrupt,bagholder,paper hands,bag")

    config["heuristic"] = {
        "positive_tokens": [t.strip() for t in positive_tokens.split(",") if t.strip()],
        "negative_tokens": [t.strip() for t in negative_tokens.split(",") if t.strip()],
        "engagement_weight_formula": os.getenv("SENTIMENT_ENGAGEMENT_FORMULA", "sqrt")
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
        "cleanup_interval": int(os.getenv("SENTIMENT_CACHE_CLEANUP_INTERVAL", "300"))
    }

    return config

DEFAULT_CONFIG = {
    "providers": {
        "stocktwits": True,
        "reddit": True,
        "news": True,
        "trends": True,
        "discord": True,
        "twitter": True,
        "reddit_pushshift": False,
        "hf_enabled": False
    },
    "lookback_hours": 24,
    "min_mentions_for_hf": 20,
    "min_mentions_for_confident_signal": 5,
    "hf": {
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment",
        "device": -1,
        "max_workers": 1
    },
    "batching": {
        "concurrency": 8,
        "rate_limit_delay_sec": 0.3
    },
    "weights": {
        "stocktwits": 0.2,
        "reddit": 0.3,
        "news": 0.2,
        "trends": 0.1,
        "discord": 0.1,
        "twitter": 0.1,
        "heuristic_vs_hf": 0.5
    },
    "heuristic": {
        "positive_tokens": ["moon","🚀","diamond","buy","long","hold","to the moon","rocket"],
        "negative_tokens": ["short","sell","dump","bankrupt","bagholder","paper hands","bag"],
        "engagement_weight_formula": "sqrt"  # 'sqrt' or 'linear'
    },
    "caching": {
        "redis_enabled": True,
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 0,
        "redis_password": None,
        "memory_max_size": 1000,
        "memory_ttl": 3600,  # 1 hour
        "redis_ttl": 7200,   # 2 hours
        "warming_enabled": True,
        "cleanup_interval": 300  # 5 minutes
    }
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
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
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
        raise ValueError("Config must be a dictionary")

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
    return float(likes + 2 * replies + 1.5 * retweets)

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

def combine_scores(heuristic: float, hf: Optional[float], hf_weight: float) -> float:
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
                    "rate_limit_delay": config["batching"]["rate_limit_delay_sec"]
                }

                # Special handling for HuggingFace
                if provider == "huggingface":
                    adapter_config.update({
                        "model_name": config["hf"]["model_name"],
                        "device": config["hf"]["device"],
                        "max_workers": config["hf"]["max_workers"]
                    })

                manager.add_adapter(provider, adapter_config)

    except Exception as e:
        _logger.warning("Could not initialize some adapters: %s", e)

# -------------------------
# Cache integration
# -------------------------
def _initialize_cache(config: Dict[str, Any]):
    """Initialize cache manager with configuration."""
    try:
        from src.common.sentiments.caching.cache_manager import CacheManager, CacheConfig

        # Create cache config from sentiment config
        cache_config = CacheConfig(
            redis_enabled=config.get("caching", {}).get("redis_enabled", True),
            redis_host=config.get("caching", {}).get("redis_host", "localhost"),
            redis_port=config.get("caching", {}).get("redis_port", 6379),
            redis_db=config.get("caching", {}).get("redis_db", 0),
            redis_password=config.get("caching", {}).get("redis_password"),
            memory_max_size=config.get("caching", {}).get("memory_max_size", 1000),
            memory_default_ttl=config.get("caching", {}).get("memory_ttl", 3600),
            redis_default_ttl=config.get("caching", {}).get("redis_ttl", 7200)
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
    lookback_hours: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    history_lookup: Optional[Callable[[str], Optional[float]]] = None,
    output_format: str = "dataclass"
) -> Union[Dict[str, Optional[SentimentFeatures]], Dict[str, Optional[Dict[str, Any]]], str]:
    """
    Collect sentiment features for a list of tickers concurrently.

    This is the main entry point for sentiment analysis. It supports multiple data providers,
    configurable output formats, and comprehensive error handling.

    Args:
        tickers: List of ticker symbols (will be normalized to uppercase)
        lookback_hours: Hours to look back for data (default: from config)
        config: Configuration dictionary (default: get_default_config())
        history_lookup: Optional function to get historical mention averages for growth calculation
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
        from src.common.sentiments.performance.performance_profiler import PerformanceProfiler
        profiler = PerformanceProfiler()

        # Extract configuration values
        lookback = lookback_hours or config.get("lookback_hours", 24)
        concurrency = config["batching"]["concurrency"]
        min_mentions_hf = config.get("min_mentions_for_hf", 20)
        weights = config["weights"]
        heuristic_config = config["heuristic"]

        # Concurrency semaphore
        sem = asyncio.Semaphore(concurrency)

        @profiler.time_function("process_one_ticker")
        async def process_one_ticker(ticker: str) -> Optional[SentimentFeatures]:
            """Process sentiment analysis for a single ticker."""
            async with sem:
                tk = ticker.upper().strip()
                if not tk:
                    return None

                data_quality: Dict[str, str] = {}
                raw_payload: Dict[str, Any] = {}

                try:
                    since_ts = int((datetime.now(timezone.utc) - timedelta(hours=lookback)).timestamp())

                    # Check cache for aggregated result first
                    aggregated_cache_key = cache_keys.aggregated_sentiment_key(tk, lookback, config_hash)
                    if cache_manager:
                        cached_result = cache_manager.get(aggregated_cache_key)
                        if cached_result:
                            _logger.debug("Cache hit for aggregated sentiment: %s", tk)
                            return cached_result

                    # Collect summaries from available adapters dynamically
                    summaries = {}
                    active_providers = [p for p, enabled in config["providers"].items() if enabled and p != "hf_enabled"]

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
                                        manager.fetch_summary_from_adapter(provider, tk, since_ts),
                                        timeout=180.0
                                    )
                                except asyncio.TimeoutError:
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
                        else:
                            data_quality[provider] = "missing"

                    raw_payload.update(summaries)

                    # Aggregate basic metrics
                    total_mentions = 0
                    unique_authors = 0
                    weighted_sentiment = 0.0
                    total_weight = 0.0

                    for provider, summary in summaries.items():
                        mentions = summary.get("mentions", 0)
                        total_mentions += mentions

                        if provider == "reddit":
                            unique_authors += summary.get("unique_authors", 0)

                        # Weight sentiment scores
                        sentiment = summary.get("sentiment_score", 0.0)
                        weight = weights.get(provider, 0.0)
                        weighted_sentiment += sentiment * weight
                        total_weight += weight

                    # Calculate combined sentiment
                    if total_weight > 0:
                        combined_sentiment = weighted_sentiment / total_weight
                    else:
                        combined_sentiment = 0.0

                    # Clamp sentiment to valid range
                    combined_sentiment = max(-1.0, min(1.0, combined_sentiment))

                    # Enhanced analysis with HuggingFace if enabled and threshold met
                    enhanced_sentiment = combined_sentiment
                    positive_ratio = None
                    bot_pct = 0.0
                    virality_index = 0.0

                    if (total_mentions >= min_mentions_hf and
                        "huggingface" in available_adapters and
                        config["providers"].get("hf_enabled", False)):

                        try:
                            # Fetch detailed messages for HF analysis
                            all_messages = []

                            # Fetch messages for HF analysis from all active providers
                            fetch_msg_tasks = []
                            for p in active_providers:
                                if p in available_adapters:
                                    fetch_msg_tasks.append(manager.fetch_messages_from_adapter(p, tk, since_ts, 150))

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
                                    enhanced_sentiment, positive_ratio, bot_pct, virality_index = await _process_messages_with_hf(
                                        all_messages, manager, heuristic_config, weights.get("heuristic_vs_hf", 0.5)
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

                    # Calculate mentions growth if history lookup provided
                    mentions_growth = None
                    if history_lookup and total_mentions > 0:
                        try:
                            if asyncio.iscoroutinefunction(history_lookup):
                                prev_avg = await history_lookup(tk)
                            else:
                                loop = asyncio.get_running_loop()
                                prev_avg = await loop.run_in_executor(None, history_lookup, tk)

                            if prev_avg and prev_avg > 0:
                                mentions_growth = total_mentions / prev_avg
                        except Exception as e:
                            _logger.debug("History lookup failed for %s: %s", tk, e)

                    # Create final features
                    sentiment_normalized = max(0.0, min(1.0, (enhanced_sentiment + 1.0) / 2.0))

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
                        raw_payload=raw_payload
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
        async def process_ticker_batch(ticker_batch: List[str]) -> List[Optional[SentimentFeatures]]:
            """Process a batch of tickers."""
            batch_tasks = [asyncio.create_task(process_one_ticker(ticker)) for ticker in ticker_batch]
            return await asyncio.gather(*batch_tasks, return_exceptions=True)

        batch_results = await batch_optimizer.process_batches_parallel(
            ticker_batches, process_ticker_batch, "sentiment_collection"
        )

        # Flatten results
        results = []
        for batch_result in batch_results:
            if batch_result:
                results.extend(batch_result)

        # Compile results
        output: Dict[str, Optional[SentimentFeatures]] = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                _logger.error("Exception processing ticker %s: %s", ticker, result)
                output[ticker.upper()] = None
            else:
                output[ticker.upper()] = result

        # Format output based on requested format
        if output_format == "dataclass":
            return output
        elif output_format == "dict":
            return {k: v.to_dict() if v else None for k, v in output.items()}
        elif output_format == "json":
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
        if 'profiler' in locals():
            profiler.auto_report()

        # Report batch optimization stats
        if 'batch_optimizer' in locals():
            perf_summary = batch_optimizer.get_performance_summary()
            if perf_summary:
                _logger.info("Batch processing summary: %s", perf_summary)


async def _process_messages_with_hf(messages: List[Dict[str, Any]], manager, heuristic_config: Dict[str, Any],
                                   hf_weight: float) -> tuple[float, Optional[float], float, float]:
    """Process messages with HuggingFace sentiment analysis."""
    if not messages:
        return 0.0, None, 0.0, 0.0

    # Extract text for HF processing
    texts = []
    message_data = []

    for msg in messages:
        body = str(msg.get("body", "")).strip()
        if body:
            texts.append(body)
            message_data.append(msg)

    if not texts:
        return 0.0, None, 0.0, 0.0

    # Get HF predictions
    try:
        hf_predictions = await manager._adapters["huggingface"].predict_batch(texts)
    except Exception as e:
        _logger.warning("HF prediction failed: %s", e)
        return 0.0, None, 0.0, 0.0

    # Process results
    positive_count = 0
    negative_count = 0
    bot_count = 0
    weighted_sentiment_sum = 0.0
    total_weight = 0.0
    virality_sum = 0.0

    pos_tokens = heuristic_config.get("positive_tokens", [])
    neg_tokens = heuristic_config.get("negative_tokens", [])

    for msg, hf_pred in zip(message_data, hf_predictions):
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

        # Calculate engagement weight
        engagement = compute_engagement(msg)
        weight = message_weight(engagement, heuristic_config.get("engagement_weight_formula", "sqrt"))

        # Accumulate metrics
        weighted_sentiment_sum += combined_sentiment * weight
        total_weight += weight

        if combined_sentiment > 0:
            positive_count += 1
        elif combined_sentiment < 0:
            negative_count += 1

        # Bot detection (simple heuristic)
        author = str(msg.get("user", {}).get("username", "")).lower()
        if "bot" in author or "auto" in author:
            bot_count += 1

        # Virality calculation
        virality_sum += engagement * abs(combined_sentiment)

    # Calculate final metrics
    final_sentiment = weighted_sentiment_sum / total_weight if total_weight > 0 else 0.0
    positive_ratio = positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else None
    bot_percentage = bot_count / len(message_data) if message_data else 0.0
    virality_index = virality_sum / max(1.0, math.sqrt(len(message_data)))

    return final_sentiment, positive_ratio, bot_percentage, virality_index


# -------------------------
# Sync wrapper for convenience
# -------------------------
def collect_sentiment_batch_sync(*args, **kwargs) -> Union[Dict[str, Optional[SentimentFeatures]], Dict[str, Optional[Dict[str, Any]]], str]:
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
        tickers=test_tickers,
        lookback_hours=24,
        config=cfg,
        output_format="dataclass"
    )

    # 5. Print summary
    print("\n--- Results Summary ---")
    for ticker, features in results.items():
        if features:
            print(f"\n[{ticker}]")
            print(f"  Sentiment Score: {features.sentiment_score_24h:.4f} (Normalized: {features.sentiment_normalized:.4f})")
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
    if results.get("AAPL"):
        pprint.pprint(results["AAPL"].raw_payload)
