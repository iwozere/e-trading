"""
Rate Limiting System

Token bucket algorithm implementation for per-user, per-channel rate limiting.
Supports priority message bypass and violation tracking.
"""

import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import threading

from src.data.db.repos.repo_notification import NotificationRepository
from src.notification.service.dependencies import get_repository_context
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class RateLimitResult(Enum):
    """Rate limit check result."""
    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    BYPASSED = "BYPASSED"


@dataclass
class RateLimitInfo:
    """Rate limit information for a user-channel combination."""
    user_id: str
    channel: str
    tokens: int
    max_tokens: int
    refill_rate: int  # tokens per minute
    last_refill: datetime
    created_at: datetime

    @property
    def is_available(self) -> bool:
        """Check if tokens are available."""
        return self.tokens > 0

    @property
    def time_until_next_token(self) -> float:
        """Get seconds until next token is available."""
        if self.tokens > 0:
            return 0.0

        # Calculate when next token will be available
        seconds_per_token = 60.0 / self.refill_rate if self.refill_rate > 0 else float('inf')
        time_since_refill = (datetime.utcnow() - self.last_refill).total_seconds()

        return max(0.0, seconds_per_token - time_since_refill)


@dataclass
class RateLimitViolation:
    """Rate limit violation record."""
    user_id: str
    channel: str
    timestamp: datetime
    message_priority: str
    bypassed: bool
    tokens_requested: int
    tokens_available: int


class TokenBucket:
    """
    Token bucket implementation for rate limiting.

    Thread-safe token bucket that refills at a constant rate.
    """

    def __init__(self, max_tokens: int, refill_rate: int, initial_tokens: Optional[int] = None):
        """
        Initialize token bucket.

        Args:
            max_tokens: Maximum number of tokens in bucket
            refill_rate: Tokens added per minute
            initial_tokens: Initial token count (defaults to max_tokens)
        """
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per minute
        self.tokens = initial_tokens if initial_tokens is not None else max_tokens
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def peek(self) -> int:
        """
        Get current token count without consuming.

        Returns:
            Current number of tokens available
        """
        with self._lock:
            self._refill()
            return self.tokens

    def _refill(self):
        """Refill tokens based on elapsed time."""
        current_time = time.time()
        time_elapsed = current_time - self.last_refill

        if time_elapsed > 0:
            # Calculate tokens to add (refill_rate is per minute)
            tokens_to_add = (time_elapsed / 60.0) * self.refill_rate

            if tokens_to_add >= 1.0:
                # Only add whole tokens
                whole_tokens = int(tokens_to_add)
                self.tokens = min(self.max_tokens, self.tokens + whole_tokens)
                self.last_refill = current_time

    def get_info(self) -> Dict[str, Any]:
        """Get bucket information."""
        with self._lock:
            self._refill()
            return {
                "tokens": self.tokens,
                "max_tokens": self.max_tokens,
                "refill_rate": self.refill_rate,
                "last_refill": self.last_refill
            }


class RateLimiter:
    """
    Per-user, per-channel rate limiter with token bucket algorithm.

    Supports:
    - Configurable limits per user-channel combination
    - Priority message bypass
    - Violation tracking and statistics
    - Persistent storage in database
    """

    def __init__(self):
        """Initialize rate limiter."""
        self._buckets: Dict[str, TokenBucket] = {}
        self._violations: list[RateLimitViolation] = []
        self._lock = threading.Lock()
        self._logger = setup_logger(f"{__name__}.RateLimiter")

        # Default configuration
        self.default_limits = {
            "telegram_channel": 30,  # messages per minute
            "email_channel": 10,
            "sms_channel": 5
        }

        # Priority bypass configuration
        self.bypass_priorities = {"CRITICAL", "HIGH"}
        self.bypass_enabled = True

    def _get_bucket_key(self, user_id: str, channel: str) -> str:
        """Generate bucket key for user-channel combination."""
        return f"{user_id}:{channel}"

    async def check_rate_limit(
        self,
        user_id: str,
        channel: str,
        priority: str = "NORMAL",
        tokens_requested: int = 1
    ) -> Tuple[RateLimitResult, Optional[RateLimitInfo]]:
        """
        Check if user can send message within rate limits.

        Args:
            user_id: User identifier
            channel: Channel name
            priority: Message priority (CRITICAL, HIGH, NORMAL, LOW)
            tokens_requested: Number of tokens to consume

        Returns:
            Tuple of (result, rate_limit_info)
        """
        try:
            # Check for priority bypass
            if self.bypass_enabled and priority in self.bypass_priorities:
                self._logger.debug(
                    "Rate limit bypassed for %s priority message (user: %s, channel: %s)",
                    priority, user_id, channel
                )

                # Still track the bypass
                violation = RateLimitViolation(
                    user_id=user_id,
                    channel=channel,
                    timestamp=datetime.utcnow(),
                    message_priority=priority,
                    bypassed=True,
                    tokens_requested=tokens_requested,
                    tokens_available=0  # Will be updated below
                )

                # Get current rate limit info for tracking
                rate_info = await self._get_rate_limit_info(user_id, channel)
                if rate_info:
                    violation.tokens_available = rate_info.tokens

                self._record_violation(violation)

                return RateLimitResult.BYPASSED, rate_info

            # Get or create token bucket
            bucket = await self._get_or_create_bucket(user_id, channel)

            # Try to consume tokens
            if bucket.consume(tokens_requested):
                # Update database
                await self._update_rate_limit_in_db(user_id, channel, bucket)

                rate_info = await self._get_rate_limit_info(user_id, channel)

                self._logger.debug(
                    "Rate limit check passed (user: %s, channel: %s, tokens: %d/%d)",
                    user_id, channel, bucket.tokens, bucket.max_tokens
                )

                return RateLimitResult.ALLOWED, rate_info

            # Rate limit exceeded
            rate_info = await self._get_rate_limit_info(user_id, channel)

            # Record violation
            violation = RateLimitViolation(
                user_id=user_id,
                channel=channel,
                timestamp=datetime.utcnow(),
                message_priority=priority,
                bypassed=False,
                tokens_requested=tokens_requested,
                tokens_available=bucket.tokens
            )
            self._record_violation(violation)

            self._logger.warning(
                "Rate limit exceeded (user: %s, channel: %s, requested: %d, available: %d)",
                user_id, channel, tokens_requested, bucket.tokens
            )

            return RateLimitResult.DENIED, rate_info

        except Exception as e:
            self._logger.error("Rate limit check failed: %s", e)
            # On error, allow the message (fail open)
            return RateLimitResult.ALLOWED, None

    async def _get_or_create_bucket(self, user_id: str, channel: str) -> TokenBucket:
        """Get existing bucket or create new one."""
        bucket_key = self._get_bucket_key(user_id, channel)

        with self._lock:
            if bucket_key in self._buckets:
                return self._buckets[bucket_key]

        # Load from database or create new
        rate_limit_info = await self._get_rate_limit_info(user_id, channel)

        if rate_limit_info:
            # Create bucket from database info
            bucket = TokenBucket(
                max_tokens=rate_limit_info.max_tokens,
                refill_rate=rate_limit_info.refill_rate,
                initial_tokens=rate_limit_info.tokens
            )
            bucket.last_refill = rate_limit_info.last_refill.timestamp()
        else:
            # Create new bucket with default limits
            max_tokens = self.default_limits.get(channel, 60)
            bucket = TokenBucket(
                max_tokens=max_tokens,
                refill_rate=max_tokens  # Same as max for 1-minute window
            )

            # Save to database
            await self._create_rate_limit_in_db(user_id, channel, bucket)

        with self._lock:
            self._buckets[bucket_key] = bucket

        return bucket

    async def _get_rate_limit_info(self, user_id: str, channel: str) -> Optional[RateLimitInfo]:
        """Get rate limit info from database."""
        try:
            with get_repository_context() as repo:
                rate_limit = repo.rate_limits.get_rate_limit(user_id, channel)

                if rate_limit:
                    return RateLimitInfo(
                        user_id=rate_limit.user_id,
                        channel=rate_limit.channel,
                        tokens=rate_limit.tokens,
                        max_tokens=rate_limit.max_tokens,
                        refill_rate=rate_limit.refill_rate,
                        last_refill=rate_limit.last_refill,
                        created_at=rate_limit.created_at
                    )

                return None

        except Exception as e:
            self._logger.error("Failed to get rate limit info: %s", e)
            return None

    async def _create_rate_limit_in_db(self, user_id: str, channel: str, bucket: TokenBucket):
        """Create new rate limit record in database."""
        try:
            with get_repository_context() as repo:
                rate_limit_data = {
                    "user_id": user_id,
                    "channel": channel,
                    "tokens": bucket.tokens,
                    "max_tokens": bucket.max_tokens,
                    "refill_rate": bucket.refill_rate,
                    "last_refill": datetime.utcnow()
                }

                repo.rate_limits.create_or_update_rate_limit(rate_limit_data)

        except Exception as e:
            self._logger.error("Failed to create rate limit in database: %s", e)

    async def _update_rate_limit_in_db(self, user_id: str, channel: str, bucket: TokenBucket):
        """Update rate limit record in database."""
        try:
            with get_repository_context() as repo:
                rate_limit_data = {
                    "user_id": user_id,
                    "channel": channel,
                    "tokens": bucket.tokens,
                    "last_refill": datetime.fromtimestamp(bucket.last_refill)
                }

                repo.rate_limits.create_or_update_rate_limit(rate_limit_data)

        except Exception as e:
            self._logger.error("Failed to update rate limit in database: %s", e)

    def _record_violation(self, violation: RateLimitViolation):
        """Record rate limit violation."""
        with self._lock:
            self._violations.append(violation)

            # Keep only recent violations (last 1000)
            if len(self._violations) > 1000:
                self._violations = self._violations[-1000:]

    async def get_user_rate_limits(self, user_id: str) -> Dict[str, RateLimitInfo]:
        """
        Get all rate limits for a user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary mapping channel names to rate limit info
        """
        try:
            with get_repository_context() as repo:
                # This would need a method in the repository to get all rate limits for a user
                # For now, we'll return empty dict as placeholder
                return {}

        except Exception as e:
            self._logger.error("Failed to get user rate limits: %s", e)
            return {}

    async def set_user_rate_limit(
        self,
        user_id: str,
        channel: str,
        max_tokens: int,
        refill_rate: Optional[int] = None
    ):
        """
        Set custom rate limit for user-channel combination.

        Args:
            user_id: User identifier
            channel: Channel name
            max_tokens: Maximum tokens in bucket
            refill_rate: Tokens per minute (defaults to max_tokens)
        """
        if refill_rate is None:
            refill_rate = max_tokens

        try:
            # Update in-memory bucket
            bucket_key = self._get_bucket_key(user_id, channel)

            with self._lock:
                if bucket_key in self._buckets:
                    bucket = self._buckets[bucket_key]
                    bucket.max_tokens = max_tokens
                    bucket.refill_rate = refill_rate
                    # Don't change current tokens, just the limits
                else:
                    # Create new bucket
                    bucket = TokenBucket(max_tokens, refill_rate)
                    self._buckets[bucket_key] = bucket

            # Update database
            await self._create_rate_limit_in_db(user_id, channel, bucket)

            self._logger.info(
                "Updated rate limit for user %s, channel %s: %d tokens, %d/min refill",
                user_id, channel, max_tokens, refill_rate
            )

        except Exception as e:
            self._logger.error("Failed to set user rate limit: %s", e)

    def get_violations(
        self,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> list[RateLimitViolation]:
        """
        Get rate limit violations.

        Args:
            user_id: Filter by user ID
            channel: Filter by channel
            since: Filter by timestamp
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

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get rate limiting statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            violations = self._violations.copy()
            buckets = self._buckets.copy()

        # Calculate statistics
        total_violations = len(violations)
        bypassed_violations = sum(1 for v in violations if v.bypassed)

        # Violations by channel
        channel_violations = {}
        for violation in violations:
            channel = violation.channel
            if channel not in channel_violations:
                channel_violations[channel] = {"total": 0, "bypassed": 0}

            channel_violations[channel]["total"] += 1
            if violation.bypassed:
                channel_violations[channel]["bypassed"] += 1

        # Recent violations (last hour)
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_violations = [v for v in violations if v.timestamp >= one_hour_ago]

        return {
            "total_violations": total_violations,
            "bypassed_violations": bypassed_violations,
            "recent_violations_1h": len(recent_violations),
            "violations_by_channel": channel_violations,
            "active_buckets": len(buckets),
            "default_limits": self.default_limits,
            "bypass_enabled": self.bypass_enabled,
            "bypass_priorities": list(self.bypass_priorities)
        }

    def configure_bypass(self, enabled: bool, priorities: Optional[set[str]] = None):
        """
        Configure priority bypass settings.

        Args:
            enabled: Whether to enable priority bypass
            priorities: Set of priorities that bypass rate limits
        """
        self.bypass_enabled = enabled

        if priorities is not None:
            self.bypass_priorities = priorities

        self._logger.info(
            "Rate limit bypass configured: enabled=%s, priorities=%s",
            enabled, list(self.bypass_priorities)
        )

    def clear_violations(self):
        """Clear all recorded violations."""
        with self._lock:
            self._violations.clear()

        self._logger.info("Rate limit violations cleared")


# Global rate limiter instance
rate_limiter = RateLimiter()