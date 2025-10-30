"""
Channel Health Monitoring System
Comprehensive health monitoring for notification channels with automatic failure detection,
recovery mechanisms, and health metrics tracking.
"""
import asyncio
import time
import statistics
from typing import Dict, Any, List, Optional, Set, Callable, Union
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import threading
import json
from collections import defaultdict, deque
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class HealthStatus(Enum):
    """Health status levels for channels."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    CRITICAL = "CRITICAL"
    DISABLED = "DISABLED"
    UNKNOWN = "UNKNOWN"


class HealthCheckType(Enum):
    """Types of health checks."""
    CONNECTIVITY = "CONNECTIVITY"
    RESPONSE_TIME = "RESPONSE_TIME"
    SUCCESS_RATE = "SUCCESS_RATE"
    ERROR_RATE = "ERROR_RATE"
    THROUGHPUT = "THROUGHPUT"
    AUTHENTICATION = "AUTHENTICATION"
    CUSTOM = "CUSTOM"


@dataclass
class HealthMetric:
    """Individual health metric measurement."""
    metric_type: HealthCheckType
    value: float
    timestamp: datetime
    channel: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "channel": self.channel,
            "details": self.details
        }


@dataclass
class HealthThreshold:
    """Health threshold configuration for a metric."""
    metric_type: HealthCheckType
    healthy_min: Optional[float] = None
    healthy_max: Optional[float] = None
    degraded_min: Optional[float] = None
    degraded_max: Optional[float] = None
    unhealthy_min: Optional[float] = None
    unhealthy_max: Optional[float] = None
    critical_min: Optional[float] = None
    critical_max: Optional[float] = None

    def evaluate_status(self, value: float) -> HealthStatus:
        """Evaluate health status based on metric value."""
        # Critical thresholds (most severe)
        if self.critical_min is not None and value < self.critical_min:
            return HealthStatus.CRITICAL
        if self.critical_max is not None and value > self.critical_max:
            return HealthStatus.CRITICAL

        # Unhealthy thresholds
        if self.unhealthy_min is not None and value < self.unhealthy_min:
            return HealthStatus.UNHEALTHY
        if self.unhealthy_max is not None and value > self.unhealthy_max:
            return HealthStatus.UNHEALTHY

        # Degraded thresholds
        if self.degraded_min is not None and value < self.degraded_min:
            return HealthStatus.DEGRADED
        if self.degraded_max is not None and value > self.degraded_max:
            return HealthStatus.DEGRADED

        # Healthy thresholds
        if self.healthy_min is not None and value < self.healthy_min:
            return HealthStatus.DEGRADED  # Below healthy minimum
        if self.healthy_max is not None and value > self.healthy_max:
            return HealthStatus.DEGRADED  # Above healthy maximum

        return HealthStatus.HEALTHY


@dataclass
class ChannelHealthStatus:
    """Current health status for a channel."""
    channel: str
    overall_status: HealthStatus
    last_updated: datetime
    metrics: Dict[HealthCheckType, HealthMetric] = field(default_factory=dict)
    failure_count: int = 0
    consecutive_failures: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    uptime_percentage: float = 100.0
    average_response_time_ms: float = 0.0
    is_enabled: bool = True
    auto_disabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "channel": self.channel,
            "overall_status": self.overall_status.value,
            "last_updated": self.last_updated.isoformat(),
            "metrics": {k.value: v.to_dict() for k, v in self.metrics.items()},
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "uptime_percentage": self.uptime_percentage,
            "average_response_time_ms": self.average_response_time_ms,
            "is_enabled": self.is_enabled,
            "auto_disabled": self.auto_disabled
        }


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    channel: str
    check_interval_seconds: int = 60
    enabled_checks: Set[HealthCheckType] = field(default_factory=lambda: {
        HealthCheckType.CONNECTIVITY,
        HealthCheckType.RESPONSE_TIME,
        HealthCheckType.SUCCESS_RATE
    })
    thresholds: Dict[HealthCheckType, HealthThreshold] = field(default_factory=dict)
    auto_disable_threshold: int = 5  # Consecutive failures before auto-disable
    auto_enable_threshold: int = 3   # Consecutive successes before auto-enable
    metric_history_size: int = 100   # Number of metrics to keep in memory

    def __post_init__(self):
        """Initialize default thresholds if not provided."""
        if not self.thresholds:
            self.thresholds = self._get_default_thresholds()

    def _get_default_thresholds(self) -> Dict[HealthCheckType, HealthThreshold]:
        """Get default health thresholds."""
        return {
            HealthCheckType.SUCCESS_RATE: HealthThreshold(
                metric_type=HealthCheckType.SUCCESS_RATE,
                healthy_min=95.0,      # 95%+ success rate is healthy
                degraded_min=85.0,     # 85-95% is degraded
                unhealthy_min=70.0,    # 70-85% is unhealthy
                critical_min=50.0      # <50% is critical
            ),
            HealthCheckType.RESPONSE_TIME: HealthThreshold(
                metric_type=HealthCheckType.RESPONSE_TIME,
                healthy_max=1000.0,    # <1s is healthy
                degraded_max=3000.0,   # 1-3s is degraded
                unhealthy_max=10000.0, # 3-10s is unhealthy
                critical_max=30000.0   # >30s is critical
            ),
            HealthCheckType.ERROR_RATE: HealthThreshold(
                metric_type=HealthCheckType.ERROR_RATE,
                healthy_max=5.0,       # <5% error rate is healthy
                degraded_max=15.0,     # 5-15% is degraded
                unhealthy_max=30.0,    # 15-30% is unhealthy
                critical_max=50.0      # >50% is critical
            )
        }


class HealthMonitor:
    """
    Comprehensive channel health monitoring system.

    Features:
    - Periodic health checks for all channels
    - Configurable health thresholds and metrics
    - Automatic channel disable/enable based on health
    - Health history tracking and trend analysis
    - Failure pattern detection and alerting
    - Integration with delivery tracking system
    """

    def __init__(self):
        """Initialize health monitor."""
        self._configs: Dict[str, HealthCheckConfig] = {}
        self._status: Dict[str, ChannelHealthStatus] = {}
        self._metric_history: Dict[str, Dict[HealthCheckType, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=100))
        )
        self._lock = threading.RLock()
        self._logger = setup_logger(f"{__name__}.HealthMonitor")

        # Health check tasks
        self._running = False
        self._check_tasks: Dict[str, asyncio.Task] = {}

        # Event callbacks
        self._status_change_callbacks: List[Callable] = []
        self._health_alert_callbacks: List[Callable] = []

        # Statistics
        self._stats = {
            "total_checks": 0,
            "failed_checks": 0,
            "channels_monitored": 0,
            "auto_disabled_channels": 0,
            "auto_enabled_channels": 0
        }

    async def start(self):
        """Start the health monitoring system."""
        if self._running:
            return

        self._running = True
        self._logger.info("Health monitor started")

        # Start health checks for all configured channels
        with self._lock:
            for channel in self._configs.keys():
                await self._start_channel_monitoring(channel)

    async def stop(self):
        """Stop the health monitoring system."""
        if not self._running:
            return

        self._running = False

        # Stop all health check tasks
        tasks_to_cancel = []
        with self._lock:
            tasks_to_cancel = list(self._check_tasks.values())
            self._check_tasks.clear()

        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._logger.info("Health monitor stopped")

    def configure_channel(self, config: HealthCheckConfig):
        """Configure health monitoring for a channel."""
        with self._lock:
            self._configs[config.channel] = config

            # Initialize status if not exists
            if config.channel not in self._status:
                self._status[config.channel] = ChannelHealthStatus(
                    channel=config.channel,
                    overall_status=HealthStatus.UNKNOWN,
                    last_updated=datetime.now(timezone.utc)
                )

            # Update stats
            self._stats["channels_monitored"] = len(self._configs)

        # Start monitoring if system is running
        if self._running:
            asyncio.create_task(self._start_channel_monitoring(config.channel))

        self._logger.info(
            "Configured health monitoring for channel %s (interval: %ds)",
            config.channel, config.check_interval_seconds
        )

    def remove_channel(self, channel: str) -> bool:
        """Remove health monitoring for a channel."""
        with self._lock:
            if channel not in self._configs:
                return False

            # Stop monitoring task
            if channel in self._check_tasks:
                task = self._check_tasks.pop(channel)
                if not task.done():
                    task.cancel()

            # Remove configuration and status
            del self._configs[channel]
            if channel in self._status:
                del self._status[channel]
            if channel in self._metric_history:
                del self._metric_history[channel]

            # Update stats
            self._stats["channels_monitored"] = len(self._configs)

        self._logger.info("Removed health monitoring for channel %s", channel)
        return True

    async def _start_channel_monitoring(self, channel: str):
        """Start health monitoring task for a specific channel."""
        if channel in self._check_tasks:
            # Stop existing task
            task = self._check_tasks[channel]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Start new monitoring task
        self._check_tasks[channel] = asyncio.create_task(
            self._monitor_channel_health(channel)
        )

    async def _monitor_channel_health(self, channel: str):
        """Monitor health for a specific channel."""
        self._logger.debug("Started health monitoring for channel %s", channel)

        try:
            while self._running:
                config = self._configs.get(channel)
                if not config:
                    break

                # Perform health checks
                await self._perform_health_checks(channel, config)

                # Wait for next check interval
                await asyncio.sleep(config.check_interval_seconds)

        except asyncio.CancelledError:
            self._logger.debug("Health monitoring cancelled for channel %s", channel)
        except Exception as e:
            self._logger.error("Health monitoring error for channel %s: %s", channel, e)

    async def _perform_health_checks(self, channel: str, config: HealthCheckConfig):
        """Perform all configured health checks for a channel."""
        try:
            current_metrics = {}

            # Perform each enabled check
            for check_type in config.enabled_checks:
                try:
                    metric = await self._perform_single_check(channel, check_type)
                    if metric:
                        current_metrics[check_type] = metric

                        # Store in history
                        self._metric_history[channel][check_type].append(metric)

                except Exception as e:
                    self._logger.warning(
                        "Health check %s failed for channel %s: %s",
                        check_type.value, channel, e
                    )

            # Update channel status
            await self._update_channel_status(channel, config, current_metrics)

            # Update statistics
            self._stats["total_checks"] += len(config.enabled_checks)

        except Exception as e:
            self._logger.error("Health check error for channel %s: %s", channel, e)
            self._stats["failed_checks"] += 1

    async def _perform_single_check(self, channel: str, check_type: HealthCheckType) -> Optional[HealthMetric]:
        """Perform a single health check."""
        start_time = time.time()

        try:
            if check_type == HealthCheckType.CONNECTIVITY:
                return await self._check_connectivity(channel)
            elif check_type == HealthCheckType.RESPONSE_TIME:
                return await self._check_response_time(channel)
            elif check_type == HealthCheckType.SUCCESS_RATE:
                return await self._check_success_rate(channel)
            elif check_type == HealthCheckType.ERROR_RATE:
                return await self._check_error_rate(channel)
            elif check_type == HealthCheckType.THROUGHPUT:
                return await self._check_throughput(channel)
            else:
                self._logger.warning("Unknown health check type: %s", check_type)
                return None

        except Exception as e:
            # Create failure metric
            return HealthMetric(
                metric_type=check_type,
                value=0.0,  # Failure value
                timestamp=datetime.now(timezone.utc),
                channel=channel,
                details={"error": str(e), "check_duration_ms": (time.time() - start_time) * 1000}
            )

    async def _check_connectivity(self, channel: str) -> HealthMetric:
        """Check channel connectivity."""
        # This would integrate with actual channel implementations
        # For now, we'll simulate based on recent delivery attempts

        # Get recent delivery attempts from delivery tracker
        success_count = 0
        total_count = 0

        # In a real implementation, this would query the delivery tracker
        # For simulation, we'll use a simple heuristic
        try:
            # Simulate connectivity check
            await asyncio.sleep(0.1)  # Simulate network call
            connectivity_score = 100.0  # Assume healthy for simulation

            return HealthMetric(
                metric_type=HealthCheckType.CONNECTIVITY,
                value=connectivity_score,
                timestamp=datetime.now(timezone.utc),
                channel=channel,
                details={"method": "simulated", "success_count": success_count, "total_count": total_count}
            )

        except Exception as e:
            return HealthMetric(
                metric_type=HealthCheckType.CONNECTIVITY,
                value=0.0,
                timestamp=datetime.now(timezone.utc),
                channel=channel,
                details={"error": str(e)}
            )

    async def _check_response_time(self, channel: str) -> HealthMetric:
        """Check channel response time."""
        start_time = time.time()

        try:
            # Simulate response time check
            await asyncio.sleep(0.05)  # Simulate network call

            response_time_ms = (time.time() - start_time) * 1000

            return HealthMetric(
                metric_type=HealthCheckType.RESPONSE_TIME,
                value=response_time_ms,
                timestamp=datetime.now(timezone.utc),
                channel=channel,
                details={"method": "ping_test"}
            )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return HealthMetric(
                metric_type=HealthCheckType.RESPONSE_TIME,
                value=response_time_ms,
                timestamp=datetime.now(timezone.utc),
                channel=channel,
                details={"error": str(e)}
            )

    async def _check_success_rate(self, channel: str) -> HealthMetric:
        """Check channel success rate based on recent deliveries."""
        # This would integrate with delivery tracker to get recent statistics

        # For simulation, calculate based on metric history
        recent_metrics = list(self._metric_history[channel][HealthCheckType.CONNECTIVITY])

        if not recent_metrics:
            # No history, assume healthy
            success_rate = 95.0
        else:
            # Calculate success rate from recent connectivity checks
            recent_count = min(10, len(recent_metrics))  # Last 10 checks
            recent_successes = sum(1 for m in recent_metrics[-recent_count:] if m.value > 50.0)
            success_rate = (recent_successes / recent_count) * 100.0

        return HealthMetric(
            metric_type=HealthCheckType.SUCCESS_RATE,
            value=success_rate,
            timestamp=datetime.now(timezone.utc),
            channel=channel,
            details={"calculation_method": "recent_connectivity", "sample_size": len(recent_metrics)}
        )

    async def _check_error_rate(self, channel: str) -> HealthMetric:
        """Check channel error rate."""
        # Calculate error rate as inverse of success rate
        success_metric = await self._check_success_rate(channel)
        error_rate = 100.0 - success_metric.value

        return HealthMetric(
            metric_type=HealthCheckType.ERROR_RATE,
            value=error_rate,
            timestamp=datetime.now(timezone.utc),
            channel=channel,
            details={"derived_from": "success_rate"}
        )

    async def _check_throughput(self, channel: str) -> HealthMetric:
        """Check channel throughput."""
        # This would integrate with delivery tracker to get throughput metrics

        # For simulation, use a baseline throughput
        throughput_msgs_per_minute = 100.0  # Simulate 100 messages per minute capacity

        return HealthMetric(
            metric_type=HealthCheckType.THROUGHPUT,
            value=throughput_msgs_per_minute,
            timestamp=datetime.now(timezone.utc),
            channel=channel,
            details={"unit": "messages_per_minute", "method": "simulated"}
        )

    async def _update_channel_status(
        self,
        channel: str,
        config: HealthCheckConfig,
        current_metrics: Dict[HealthCheckType, HealthMetric]
    ):
        """Update channel health status based on current metrics."""
        with self._lock:
            status = self._status[channel]

            # Update metrics
            status.metrics.update(current_metrics)
            status.last_updated = datetime.now(timezone.utc)

            # Calculate overall health status
            metric_statuses = []
            for metric_type, metric in current_metrics.items():
                if metric_type in config.thresholds:
                    threshold = config.thresholds[metric_type]
                    metric_status = threshold.evaluate_status(metric.value)
                    metric_statuses.append(metric_status)

            # Determine overall status (worst case)
            if not metric_statuses:
                overall_status = HealthStatus.UNKNOWN
            else:
                # Priority order: CRITICAL > UNHEALTHY > DEGRADED > HEALTHY
                if HealthStatus.CRITICAL in metric_statuses:
                    overall_status = HealthStatus.CRITICAL
                elif HealthStatus.UNHEALTHY in metric_statuses:
                    overall_status = HealthStatus.UNHEALTHY
                elif HealthStatus.DEGRADED in metric_statuses:
                    overall_status = HealthStatus.DEGRADED
                else:
                    overall_status = HealthStatus.HEALTHY

            # Update failure tracking
            previous_status = status.overall_status
            status.overall_status = overall_status

            if overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                status.failure_count += 1
                status.consecutive_failures += 1
                status.last_failure = datetime.now(timezone.utc)
            else:
                if status.consecutive_failures > 0:
                    # Reset consecutive failures on success
                    status.consecutive_failures = 0
                status.last_success = datetime.now(timezone.utc)

            # Calculate uptime percentage
            await self._update_uptime_percentage(status)

            # Update average response time
            await self._update_average_response_time(status, channel)

            # Check for auto-disable/enable
            await self._check_auto_disable_enable(status, config)

            # Trigger callbacks if status changed
            if previous_status != overall_status:
                await self._trigger_status_change_callbacks(channel, previous_status, overall_status)

    async def _update_uptime_percentage(self, status: ChannelHealthStatus):
        """Update uptime percentage based on recent health checks."""
        # Calculate uptime from recent metrics
        channel = status.channel
        all_metrics = []

        # Collect all recent metrics
        for metric_type, metric_deque in self._metric_history[channel].items():
            all_metrics.extend(list(metric_deque))

        if not all_metrics:
            return

        # Sort by timestamp
        all_metrics.sort(key=lambda m: m.timestamp)

        # Calculate uptime from last 24 hours or available data
        now = datetime.now(timezone.utc)
        cutoff_time = now - timedelta(hours=24)
        recent_metrics = [m for m in all_metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return

        # Count successful checks by metric type to avoid double counting
        metric_type_counts = {}
        successful_by_type = {}

        for metric in recent_metrics:
            metric_type = metric.metric_type
            if metric_type not in metric_type_counts:
                metric_type_counts[metric_type] = 0
                successful_by_type[metric_type] = 0

            metric_type_counts[metric_type] += 1

            # Consider successful if value indicates good health
            if ((metric_type == HealthCheckType.CONNECTIVITY and metric.value > 50.0) or
                (metric_type == HealthCheckType.SUCCESS_RATE and metric.value > 50.0) or
                (metric_type == HealthCheckType.RESPONSE_TIME and metric.value < 10000.0) or
                (metric_type == HealthCheckType.ERROR_RATE and metric.value < 50.0)):
                successful_by_type[metric_type] += 1

        # Calculate overall success rate across all metric types
        total_checks = sum(metric_type_counts.values())
        successful_checks = sum(successful_by_type.values())

        if total_checks > 0:
            status.uptime_percentage = (successful_checks / total_checks) * 100.0

    async def _update_average_response_time(self, status: ChannelHealthStatus, channel: str):
        """Update average response time from recent metrics."""
        response_time_metrics = list(self._metric_history[channel][HealthCheckType.RESPONSE_TIME])

        if response_time_metrics:
            # Calculate average from last 10 measurements
            recent_count = min(10, len(response_time_metrics))
            recent_times = [m.value for m in response_time_metrics[-recent_count:]]
            status.average_response_time_ms = statistics.mean(recent_times)

    async def _check_auto_disable_enable(self, status: ChannelHealthStatus, config: HealthCheckConfig):
        """Check if channel should be auto-disabled or auto-enabled."""
        # Auto-disable logic
        if (status.is_enabled and not status.auto_disabled and
            status.consecutive_failures >= config.auto_disable_threshold):

            status.is_enabled = False
            status.auto_disabled = True
            self._stats["auto_disabled_channels"] += 1

            self._logger.warning(
                "Auto-disabled channel %s after %d consecutive failures",
                status.channel, status.consecutive_failures
            )

            # Trigger alert callback
            await self._trigger_health_alert_callbacks(
                status.channel,
                "auto_disabled",
                f"Channel auto-disabled after {status.consecutive_failures} consecutive failures"
            )

        # Auto-enable logic
        elif (not status.is_enabled and status.auto_disabled and
              status.consecutive_failures == 0 and
              status.overall_status == HealthStatus.HEALTHY):

            # Check if we have enough consecutive successes
            recent_metrics = []
            for metric_deque in self._metric_history[status.channel].values():
                recent_metrics.extend(list(metric_deque)[-config.auto_enable_threshold:])

            # Count recent successful checks
            successful_recent = sum(1 for m in recent_metrics if m.value > 50.0)

            if successful_recent >= config.auto_enable_threshold:
                status.is_enabled = True
                status.auto_disabled = False
                self._stats["auto_enabled_channels"] += 1

                self._logger.info(
                    "Auto-enabled channel %s after %d consecutive successes",
                    status.channel, successful_recent
                )

                # Trigger alert callback
                await self._trigger_health_alert_callbacks(
                    status.channel,
                    "auto_enabled",
                    f"Channel auto-enabled after {successful_recent} consecutive successes"
                )

    async def _trigger_status_change_callbacks(
        self,
        channel: str,
        old_status: HealthStatus,
        new_status: HealthStatus
    ):
        """Trigger status change callbacks."""
        for callback in self._status_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(channel, old_status, new_status)
                else:
                    callback(channel, old_status, new_status)
            except Exception as e:
                self._logger.exception("Status change callback error:")

    async def _trigger_health_alert_callbacks(self, channel: str, alert_type: str, message: str):
        """Trigger health alert callbacks."""
        for callback in self._health_alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(channel, alert_type, message)
                else:
                    callback(channel, alert_type, message)
            except Exception as e:
                self._logger.exception("Health alert callback error:")

    def add_status_change_callback(self, callback: Callable):
        """Add callback for status changes."""
        self._status_change_callbacks.append(callback)

    def add_health_alert_callback(self, callback: Callable):
        """Add callback for health alerts."""
        self._health_alert_callbacks.append(callback)

    def get_channel_status(self, channel: str) -> Optional[ChannelHealthStatus]:
        """Get current health status for a channel."""
        with self._lock:
            return self._status.get(channel)

    def get_all_statuses(self) -> Dict[str, ChannelHealthStatus]:
        """Get health status for all channels."""
        with self._lock:
            return self._status.copy()

    def get_channel_metrics(self, channel: str, metric_type: Optional[HealthCheckType] = None) -> List[HealthMetric]:
        """Get metric history for a channel."""
        with self._lock:
            if channel not in self._metric_history:
                return []

            if metric_type:
                return list(self._metric_history[channel][metric_type])
            else:
                # Return all metrics
                all_metrics = []
                for metric_deque in self._metric_history[channel].values():
                    all_metrics.extend(list(metric_deque))
                return sorted(all_metrics, key=lambda m: m.timestamp)

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        with self._lock:
            status_counts = defaultdict(int)
            total_channels = len(self._status)

            for status in self._status.values():
                status_counts[status.overall_status.value] += 1

            return {
                "total_channels": total_channels,
                "status_distribution": dict(status_counts),
                "healthy_percentage": (status_counts["HEALTHY"] / total_channels * 100) if total_channels > 0 else 0,
                "statistics": self._stats.copy(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def manually_disable_channel(self, channel: str, reason: str = "Manual disable") -> bool:
        """Manually disable a channel."""
        with self._lock:
            if channel not in self._status:
                return False

            status = self._status[channel]
            status.is_enabled = False
            status.auto_disabled = False  # This is manual, not auto

            self._logger.info("Manually disabled channel %s: %s", channel, reason)
            return True

    def manually_enable_channel(self, channel: str, reason: str = "Manual enable") -> bool:
        """Manually enable a channel."""
        with self._lock:
            if channel not in self._status:
                return False

            status = self._status[channel]
            status.is_enabled = True
            status.auto_disabled = False

            self._logger.info("Manually enabled channel %s: %s", channel, reason)
            return True

    def reset_channel_metrics(self, channel: str) -> bool:
        """Reset metrics and failure counts for a channel."""
        with self._lock:
            if channel not in self._status:
                return False

            # Reset status counters
            status = self._status[channel]
            status.failure_count = 0
            status.consecutive_failures = 0
            status.last_failure = None
            status.uptime_percentage = 100.0

            # Clear metric history
            if channel in self._metric_history:
                for metric_deque in self._metric_history[channel].values():
                    metric_deque.clear()

            self._logger.info("Reset metrics for channel %s", channel)
            return True


# Global health monitor instance
health_monitor = HealthMonitor()


# Convenience functions for common configurations
def create_default_health_config(channel: str, **kwargs) -> HealthCheckConfig:
    """Create a default health configuration for a channel."""
    return HealthCheckConfig(
        channel=channel,
        **kwargs
    )


def create_strict_health_config(channel: str, **kwargs) -> HealthCheckConfig:
    """Create a strict health configuration with tighter thresholds."""
    strict_thresholds = {
        HealthCheckType.SUCCESS_RATE: HealthThreshold(
            metric_type=HealthCheckType.SUCCESS_RATE,
            healthy_min=98.0,      # 98%+ success rate is healthy
            degraded_min=95.0,     # 95-98% is degraded
            unhealthy_min=90.0,    # 90-95% is unhealthy
            critical_min=80.0      # <80% is critical
        ),
        HealthCheckType.RESPONSE_TIME: HealthThreshold(
            metric_type=HealthCheckType.RESPONSE_TIME,
            healthy_max=500.0,     # <500ms is healthy
            degraded_max=1000.0,   # 500ms-1s is degraded
            unhealthy_max=3000.0,  # 1-3s is unhealthy
            critical_max=10000.0   # >10s is critical
        ),
        HealthCheckType.ERROR_RATE: HealthThreshold(
            metric_type=HealthCheckType.ERROR_RATE,
            healthy_max=2.0,       # <2% error rate is healthy
            degraded_max=5.0,      # 2-5% is degraded
            unhealthy_max=10.0,    # 5-10% is unhealthy
            critical_max=20.0      # >20% is critical
        )
    }

    return HealthCheckConfig(
        channel=channel,
        thresholds=strict_thresholds,
        auto_disable_threshold=3,  # Disable after 3 failures
        auto_enable_threshold=5,   # Enable after 5 successes
        **kwargs
    )


def create_lenient_health_config(channel: str, **kwargs) -> HealthCheckConfig:
    """Create a lenient health configuration with relaxed thresholds."""
    lenient_thresholds = {
        HealthCheckType.SUCCESS_RATE: HealthThreshold(
            metric_type=HealthCheckType.SUCCESS_RATE,
            healthy_min=85.0,      # 85%+ success rate is healthy
            degraded_min=70.0,     # 70-85% is degraded
            unhealthy_min=50.0,    # 50-70% is unhealthy
            critical_min=25.0      # <25% is critical
        ),
        HealthCheckType.RESPONSE_TIME: HealthThreshold(
            metric_type=HealthCheckType.RESPONSE_TIME,
            healthy_max=3000.0,    # <3s is healthy
            degraded_max=10000.0,  # 3-10s is degraded
            unhealthy_max=30000.0, # 10-30s is unhealthy
            critical_max=60000.0   # >60s is critical
        ),
        HealthCheckType.ERROR_RATE: HealthThreshold(
            metric_type=HealthCheckType.ERROR_RATE,
            healthy_max=15.0,      # <15% error rate is healthy
            degraded_max=30.0,     # 15-30% is degraded
            unhealthy_max=50.0,    # 30-50% is unhealthy
            critical_max=75.0      # >75% is critical
        )
    }

    return HealthCheckConfig(
        channel=channel,
        thresholds=lenient_thresholds,
        auto_disable_threshold=10,  # Disable after 10 failures
        auto_enable_threshold=2,    # Enable after 2 successes
        **kwargs
    )