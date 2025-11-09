"""
Rate Limiting System

Token bucket algorithm implementation for per-user, per-channel rate limiting.
Supports configurable limits, priority bypass, and violation tracking.
"""

import asyncio
import time
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
import threading

from src.data.db.services.database_service import get_database_service
from src.data.db.models.model_notification import MessagePriority
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class RateLimitResult(Enum):
    """Rate limit check result."""
    ALLOWED = "ALLOWED"
    RATE_LIMITED = "RATE_LIMITED"
    BYPASSED = "BYPASSED"


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a user-channel combination."""

    max_tokens: int = 60  # Maximum tokens in bucket
    refill_rate: int = 60  # Tokens added per minute
    window_minutes: int = 1  # Time window for rate limiting
    burst_allowance: float = 1.5  # Allow burst up to 150% of rate

    def __post_init__(self):
        """Validate configuration."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.refill_rate <= 0:
            raise ValueError("refill_rate must be positive")
        if self.window_minutes <= 0:
            raise ValueError("window_minutes must be positive")
        if self.burst_allowance < 1.0:
            raise ValueError("burst_allowance must be >= 1.0")


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    tokens: float
    max_tokens: int
    refill_rate: float  # tokens per second
    last_refill: float
    config: RateLimitConfig

    def __post_init__(self):
        """Initialize bucket state."""
        if self.tokens > self.max_tokens:
            self.tokens = self.max_tokens

    def refill(self, current_time: float) -> None:
        """Refill tokens based on elapsed time."""
        if self.last_refill == 0:
            self.last_refill = current_time
            return

        time_elapsed = current_time - self.last_refill
        if time_elapsed <= 0:
            return

        # Calculate tokens to add
        tokens_to_add = time_elapsed * self.refill_rate

        # Apply burst allowance
        max_burst_tokens = int(self.max_tokens * self.config.burst_allowance)

        self.tokens = min(max_burst_tokens, self.tokens + tokens_to_add)
        self.last_refill = current_time

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get time to wait until tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        if self.tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


@dataclass
class RateLimitViolation:
    """Rate limit violation record."""

    user_id: str
    channel: str
    timestamp: datetime
    requested_tokens: int
    available_tokens: float
    priority: str
    bypassed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "channel": self.channel,
            "timestamp": self.timestamp.isoformat(),
            "requested_tokens": self.requested_tokens,
            "available_tokens": self.available_tokens,
            "priority": self.priority,
            "bypassed": self.bypassed
        }


class RateLimiter:
    """
    Per-user, per-channel rate limiter with token bucket algorithm.

    Features:
    - Token bucket algorithm for smooth rate limiting
    - Configurable limits per user-channel combination
    - Priority bypass for high-priority messages
    - Violation tracking and statistics
    - Burst allowance for temporary spikes
    - Persistent state in database
    """

    def __init__(self):
        """Initialize rate limiter."""
        self._buckets: Dict[str, TokenBucket] = {}
        self._violations: List[RateLimitViolation] = []
        self._lock = threading.RLock()
        self._logger = setup_logger(f"{__name__}.RateLimiter")

        # Default configurations per channel
        self._default_configs = {
            "telegram_channel": RateLimitConfig(max_tokens=30, refill_rate=30),
            "email_channel": RateLimitConfig(max_tokens=10, refill_rate=10),
            "sms_channel": RateLimitConfig(max_tokens=5, refill_rate=5),
            "default": RateLimitConfig(max_tokens=60, refill_rate=60)
        }

        # Priority bypass settings
        self._bypass_priorities = {MessagePriority.CRITICAL, MessagePriority.HIGH}
        self._bypass_enabled = True

    def _get_bucket_key(self, user_id: str, channel: str) -> str:
        """Generate bucket key for user-channel combination."""
        return f"{user_id}:{channel}"

    def _get_config(self, channel: str) -> RateLimitConfig:
        """Get rate limit configuration for channel."""
        return self._default_configs.get(channel, self._default_configs["default"])

    async def _load_bucket_from_db(self, user_id: str, channel: str) -> Optional[TokenBucket]:
        """Load token bucket state from database."""
        try:
            db_service = get_database_service()
            with db_service.uow() as r:
                rate_limit = r.notifications.rate_limits.get_rate_limit(user_id, channel)

                if rate_limit:
                    config = self._get_config(channel)

                    # Update config from database if available
                    config.max_tokens = rate_limit.max_tokens
                    config.refill_rate = rate_limit.refill_rate

                    # Convert refill rate from per-minute to per-second
                    refill_rate_per_second = rate_limit.refill_rate / 60.0

                    bucket = TokenBucket(
                        tokens=float(rate_limit.tokens),
                        max_tokens=rate_limit.max_tokens,
                        refill_rate=refill_rate_per_second,
                        last_refill=rate_limit.last_refill.timestamp(),
                        config=config
                    )

                    return bucket

        except Exception as e:
            self._logger.error("Failed to load bucket from database for %s:%s: %s", user_id, channel, e)

        return None

    async def _save_bucket_to_db(self, user_id: str, channel: str, bucket: TokenBucket) -> None:
        """Save token bucket state to database."""
        try:
            db_service = get_database_service()
            with db_service.uow() as r:
                rate_limit_data = {
                    "user_id": user_id,
                    "channel": channel,
                    "tokens": int(bucket.tokens),
                    "max_tokens": bucket.max_tokens,
                    "refill_rate": int(bucket.refill_rate * 60),  # Convert to per-minute
                    "last_refill": datetime.fromtimestamp(bucket.last_refill)
                }

                r.notifications.rate_limits.create_or_update_rate_limit(rate_limit_data)

        except Exception as e:
            self._logger.error("Failed to save bucket to database for %s:%s: %s", user_id, channel, e)

    async def _get_or_create_bucket(self, user_id: str, channel: str) -> TokenBucket:
        """Get or create token bucket for user-channel combination."""
        bucket_key = self._get_bucket_key(user_id, channel)

        with self._lock:
            # Check in-memory cache first
            if bucket_key in self._buckets:
                return self._buckets[bucket_key]

            # Try to load from database
            bucket = await self._load_bucket_from_db(user_id, channel)

            if bucket is None:
                # Create new bucket
                config = self._get_config(channel)
                bucket = TokenBucket(
                    tokens=float(config.max_tokens),
                    max_tokens=config.max_tokens,
                    refill_rate=config.refill_rate / 60.0,  # Convert to per-second
                    last_refill=time.time(),
                    config=config
                )

                self._logger.info("Created new rate limit bucket for %s:%s", user_id, channel)

            # Cache in memory
            self._buckets[bucket_key] = bucket
            return bucket

    async def check_rate_limit(
        self,
        user_id: str,
        channel: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        tokens: int = 1
    ) -> Tuple[RateLimitResult, Optional[float]]:
        """
        Check if request is within rate limits.

        Args:
            user_id: User identifier
            channel: Channel name
            priority: Message priority
            tokens: Number of tokens to consume

        Returns:
            Tuple of (result, wait_time_seconds)
        """
        current_time = time.time()

        # Check for priority bypass
        if self._bypass_enabled and priority in self._bypass_priorities:
            self._logger.debug("Rate limit bypassed for %s priority message from %s:%s", priority.value, user_id, channel)

            # Still record as violation for tracking, but mark as bypassed
            violation = RateLimitViolation(
                user_id=user_id,
                channel=channel,
                timestamp=datetime.now(timezone.utc),
                requested_tokens=tokens,
                available_tokens=0,  # Would have been rate limited
                priority=priority.value,
                bypassed=True
            )

            with self._lock:
                self._violations.append(violation)

            return RateLimitResult.BYPASSED, None

        # Get or create bucket
        bucket = await self._get_or_create_bucket(user_id, channel)

        with self._lock:
            # Refill tokens
            bucket.refill(current_time)

            # Try to consume tokens
            if bucket.consume(tokens):
                # Save updated state to database (async)
                asyncio.create_task(self._save_bucket_to_db(user_id, channel, bucket))

                self._logger.debug("Rate limit check passed for %s:%s (tokens: %.2f)", user_id, channel, bucket.tokens)
                return RateLimitResult.ALLOWED, None

            # Rate limited - calculate wait time
            wait_time = bucket.get_wait_time(tokens)

            # Record violation
            violation = RateLimitViolation(
                user_id=user_id,
                channel=channel,
                timestamp=datetime.now(timezone.utc),
                requested_tokens=tokens,
                available_tokens=bucket.tokens,
                priority=priority.value,
                bypassed=False
            )

            self._violations.append(violation)

            self._logger.warning(
                "Rate limit exceeded for %s:%s (available: %.2f, requested: %d, wait: %.2fs)",
                user_id, channel, bucket.tokens, tokens, wait_time
            )

            return RateLimitResult.RATE_LIMITED, wait_time

    async def consume_tokens(
        self,
        user_id: str,
        channel: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        tokens: int = 1
    ) -> bool:
        """
        Consume tokens if available (convenience method).

        Args:
            user_id: User identifier
            channel: Channel name
            priority: Message priority
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if rate limited
        """
        result, _ = await self.check_rate_limit(user_id, channel, priority, tokens)
        return result in [RateLimitResult.ALLOWED, RateLimitResult.BYPASSED]

    async def get_bucket_status(self, user_id: str, channel: str) -> Dict[str, Any]:
        """
        Get current bucket status for user-channel.

        Args:
            user_id: User identifier
            channel: Channel name

        Returns:
            Dictionary with bucket status information
        """
        bucket = await self._get_or_create_bucket(user_id, channel)

        with self._lock:
            bucket.refill(time.time())

            return {
                "user_id": user_id,
                "channel": channel,
                "tokens": bucket.tokens,
                "max_tokens": bucket.max_tokens,
                "refill_rate_per_minute": bucket.refill_rate * 60,
                "last_refill": datetime.fromtimestamp(bucket.last_refill).isoformat(),
                "config": {
                    "max_tokens": bucket.config.max_tokens,
                    "refill_rate": bucket.config.refill_rate * 60,
                    "window_minutes": bucket.config.window_minutes,
                    "burst_allowance": bucket.config.burst_allowance
                }
            }

    def set_channel_config(self, channel: str, config: RateLimitConfig) -> None:
        """
        Set rate limit configuration for a channel.

        Args:
            channel: Channel name
            config: Rate limit configuration
        """
        self._default_configs[channel] = config
        self._logger.info("Updated rate limit config for channel %s: %s tokens/min", channel, config.refill_rate)

    def enable_priority_bypass(self, enabled: bool = True) -> None:
        """
        Enable or disable priority bypass for high-priority messages.

        Args:
            enabled: Whether to enable priority bypass
        """
        self._bypass_enabled = enabled
        self._logger.info("Priority bypass %s", "enabled" if enabled else "disabled")

    def set_bypass_priorities(self, priorities: List[MessagePriority]) -> None:
        """
        Set which priorities bypass rate limiting.

        Args:
            priorities: List of priorities that bypass rate limits
        """
        self._bypass_priorities = set(priorities)
        self._logger.info("Priority bypass set for: %s", [p.value for p in priorities])

    def get_violations(
        self,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[RateLimitViolation]:
        """
        Get rate limit violations with optional filtering.

        Args:
            user_id: Filter by user ID
            channel: Filter by channel
            since: Filter violations since this time
            limit: Maximum number of violations to return

        Returns:
            List of rate limit violations
        """
        with self._lock:
            violations = self._violations.copy()

        # Apply filters
        if user_id:
            violations = [v for v in violations if v.user_id == user_id]

        if channel:
            violations = [v for v in violations if v.channel == channel]

        if since:
            violations = [v for v in violations if v.timestamp >= since]

        # Sort by timestamp (newest first) and limit
        violations.sort(key=lambda v: v.timestamp, reverse=True)
        return violations[:limit]

    def get_statistics(
        self,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get rate limiting statistics.

        Args:
            user_id: Filter by user ID
            channel: Filter by channel
            hours: Time window in hours

        Returns:
            Dictionary with statistics
        """
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        violations = self.get_violations(user_id, channel, since)

        # Calculate statistics
        total_violations = len(violations)
        bypassed_violations = len([v for v in violations if v.bypassed])
        rate_limited_violations = total_violations - bypassed_violations

        # Group by channel
        channel_stats = {}
        for violation in violations:
            ch = violation.channel
            if ch not in channel_stats:
                channel_stats[ch] = {"total": 0, "bypassed": 0, "rate_limited": 0}

            channel_stats[ch]["total"] += 1
            if violation.bypassed:
                channel_stats[ch]["bypassed"] += 1
            else:
                channel_stats[ch]["rate_limited"] += 1

        # Group by priority
        priority_stats = {}
        for violation in violations:
            priority = violation.priority
            if priority not in priority_stats:
                priority_stats[priority] = {"total": 0, "bypassed": 0, "rate_limited": 0}

            priority_stats[priority]["total"] += 1
            if violation.bypassed:
                priority_stats[priority]["bypassed"] += 1
            else:
                priority_stats[priority]["rate_limited"] += 1

        return {
            "time_window_hours": hours,
            "total_violations": total_violations,
            "rate_limited_violations": rate_limited_violations,
            "bypassed_violations": bypassed_violations,
            "bypass_rate": bypassed_violations / total_violations if total_violations > 0 else 0,
            "channel_statistics": channel_stats,
            "priority_statistics": priority_stats,
            "filter": {
                "user_id": user_id,
                "channel": channel
            }
        }

    async def cleanup_old_violations(self, days_to_keep: int = 7) -> int:
        """
        Clean up old violation records.

        Args:
            days_to_keep: Number of days of violations to keep

        Returns:
            Number of violations removed
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

        with self._lock:
            original_count = len(self._violations)
            self._violations = [v for v in self._violations if v.timestamp >= cutoff_time]
            removed_count = original_count - len(self._violations)

        if removed_count > 0:
            self._logger.info("Cleaned up %d old rate limit violations", removed_count)

        return removed_count

    async def reset_user_limits(self, user_id: str, channel: Optional[str] = None) -> int:
        """
        Reset rate limits for a user (admin function).

        Args:
            user_id: User ID to reset
            channel: Specific channel to reset, or None for all channels

        Returns:
            Number of buckets reset
        """
        reset_count = 0

        with self._lock:
            keys_to_remove = []

            for bucket_key, bucket in self._buckets.items():
                key_user_id, key_channel = bucket_key.split(":", 1)

                if key_user_id == user_id and (channel is None or key_channel == channel):
                    # Reset bucket to full capacity
                    bucket.tokens = float(bucket.max_tokens)
                    bucket.last_refill = time.time()
                    keys_to_remove.append(bucket_key)
                    reset_count += 1

            # Remove from cache to force reload from database
            for key in keys_to_remove:
                del self._buckets[key]

        # Also reset in database
        try:
            db_service = get_database_service()
            with db_service.uow() as r:
                if channel:
                    rate_limit = r.notifications.rate_limits.get_rate_limit(user_id, channel)
                    if rate_limit:
                        r.notifications.rate_limits.create_or_update_rate_limit({
                            "user_id": user_id,
                            "channel": channel,
                            "tokens": rate_limit.max_tokens,
                            "max_tokens": rate_limit.max_tokens,
                            "refill_rate": rate_limit.refill_rate,
                            "last_refill": datetime.now(timezone.utc)
                        })
                else:
                    # Reset all channels for user - would need custom query
                    pass

        except Exception as e:
            self._logger.error("Failed to reset rate limits in database for %s: %s", user_id, e)

        self._logger.info("Reset rate limits for user %s (%s): %d buckets", user_id, channel or "all channels", reset_count)
        return reset_count


# Global rate limiter instance
rate_limiter = RateLimiter()