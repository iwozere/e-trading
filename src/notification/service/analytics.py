"""
Notification Service Analytics and Statistics System

Comprehensive analytics system for tracking delivery performance, success rates,
response times, and trends across channels, users, and time periods.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import threading

from src.data.db.services.database_service import get_database_service
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TimeGranularity(Enum):
    """Time granularity for statistics aggregation."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class ChannelStats:
    """Statistics for a specific channel."""

    channel: str
    total_messages: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    partial_deliveries: int = 0

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0

    avg_response_time_ms: float = 0.0
    min_response_time_ms: Optional[int] = None
    max_response_time_ms: Optional[int] = None

    # Rate calculations
    success_rate: float = 0.0
    attempt_success_rate: float = 0.0

    # Time-based metrics
    messages_per_hour: float = 0.0
    peak_hour: Optional[int] = None
    peak_messages: int = 0

    def calculate_rates(self):
        """Calculate success rates."""
        self.success_rate = (
            self.successful_deliveries / self.total_messages
            if self.total_messages > 0 else 0.0
        )

        self.attempt_success_rate = (
            self.successful_attempts / self.total_attempts
            if self.total_attempts > 0 else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel": self.channel,
            "total_messages": self.total_messages,
            "successful_deliveries": self.successful_deliveries,
            "failed_deliveries": self.failed_deliveries,
            "partial_deliveries": self.partial_deliveries,
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "avg_response_time_ms": self.avg_response_time_ms,
            "min_response_time_ms": self.min_response_time_ms,
            "max_response_time_ms": self.max_response_time_ms,
            "success_rate": self.success_rate,
            "attempt_success_rate": self.attempt_success_rate,
            "messages_per_hour": self.messages_per_hour,
            "peak_hour": self.peak_hour,
            "peak_messages": self.peak_messages
        }


@dataclass
class UserStats:
    """Statistics for a specific user."""

    user_id: str
    total_messages: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    partial_deliveries: int = 0

    # Channel breakdown
    channel_stats: Dict[str, ChannelStats] = field(default_factory=dict)

    # Priority breakdown
    priority_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Rate limiting stats
    rate_limit_violations: int = 0
    bypassed_rate_limits: int = 0

    # Time-based metrics
    avg_response_time_ms: float = 0.0
    messages_per_day: float = 0.0
    most_active_hour: Optional[int] = None

    def calculate_rates(self):
        """Calculate success rates."""
        for channel_stat in self.channel_stats.values():
            channel_stat.calculate_rates()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "total_messages": self.total_messages,
            "successful_deliveries": self.successful_deliveries,
            "failed_deliveries": self.failed_deliveries,
            "partial_deliveries": self.partial_deliveries,
            "channel_stats": {
                channel: stats.to_dict()
                for channel, stats in self.channel_stats.items()
            },
            "priority_stats": self.priority_stats,
            "rate_limit_violations": self.rate_limit_violations,
            "bypassed_rate_limits": self.bypassed_rate_limits,
            "avg_response_time_ms": self.avg_response_time_ms,
            "messages_per_day": self.messages_per_day,
            "most_active_hour": self.most_active_hour
        }


@dataclass
class TimeSeriesPoint:
    """A single point in time series data."""

    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "metadata": self.metadata
        }


@dataclass
class TrendAnalysis:
    """Trend analysis results."""

    metric_name: str
    time_series: List[TimeSeriesPoint]
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0 to 1.0
    change_percentage: float

    # Statistical measures
    mean: float
    median: float
    std_deviation: float
    min_value: float
    max_value: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "time_series": [point.to_dict() for point in self.time_series],
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "change_percentage": self.change_percentage,
            "statistics": {
                "mean": self.mean,
                "median": self.median,
                "std_deviation": self.std_deviation,
                "min_value": self.min_value,
                "max_value": self.max_value
            }
        }


class NotificationAnalytics:
    """
    Comprehensive analytics and statistics system for notification service.

    Features:
    - Delivery rate calculations per channel and user
    - Average response times and success rates
    - Daily/weekly/monthly statistics aggregation
    - Performance trend analysis
    - Real-time metrics collection
    - Historical data analysis
    """

    def __init__(self):
        """Initialize analytics system."""
        self._lock = threading.RLock()
        self._logger = setup_logger(f"{__name__}.NotificationAnalytics")

        # In-memory cache for recent statistics
        self._channel_stats_cache: Dict[str, ChannelStats] = {}
        self._user_stats_cache: Dict[str, UserStats] = {}
        self._cache_expiry = timedelta(minutes=15)
        self._last_cache_update = datetime.now(timezone.utc)

        # Configuration
        self._max_trend_points = 100
        self._default_trend_days = 30

    async def get_delivery_rates(
        self,
        channel: Optional[str] = None,
        user_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get delivery rate calculations per channel and user.

        Args:
            channel: Filter by specific channel
            user_id: Filter by specific user
            days: Number of days to analyze

        Returns:
            Dictionary with delivery rate statistics
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            db_service = get_database_service()
            with db_service.uow() as r:
                # Get delivery statistics from database
                delivery_stats = r.notifications.delivery_status.get_delivery_statistics(
                    channel=channel, days=days
                )

                # Get message statistics
                message_stats = self._get_message_statistics(
                    repo, channel, user_id, cutoff_date
                )

                # Calculate rates
                total_messages = message_stats.get("total_messages", 0)
                successful_deliveries = delivery_stats.get("status_counts", {}).get("DELIVERED", 0)
                failed_deliveries = delivery_stats.get("status_counts", {}).get("FAILED", 0)

                overall_success_rate = (
                    successful_deliveries / total_messages
                    if total_messages > 0 else 0.0
                )

                # Get channel-specific rates
                channel_rates = self._calculate_channel_rates(
                    repo, channel, cutoff_date
                )

                # Get user-specific rates if requested
                user_rates = {}
                if user_id:
                    user_rates = self._calculate_user_rates(
                        repo, user_id, cutoff_date
                    )

                return {
                    "period_days": days,
                    "cutoff_date": cutoff_date.isoformat(),
                    "overall_statistics": {
                        "total_messages": total_messages,
                        "successful_deliveries": successful_deliveries,
                        "failed_deliveries": failed_deliveries,
                        "success_rate": overall_success_rate,
                        "average_response_time_ms": delivery_stats.get("average_response_time_ms")
                    },
                    "channel_rates": channel_rates,
                    "user_rates": user_rates,
                    "filters": {
                        "channel": channel,
                        "user_id": user_id
                    }
                }

        except Exception:
            self._logger.exception("Failed to get delivery rates:")
            raise

    async def get_response_time_analysis(
        self,
        channel: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get detailed response time analysis.

        Args:
            channel: Filter by specific channel
            days: Number of days to analyze

        Returns:
            Dictionary with response time statistics
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            db_service = get_database_service()
            with db_service.uow() as r:
                # Get response time data from database
                response_times = self._get_response_time_data(
                    repo, channel, cutoff_date
                )

                if not response_times:
                    return {
                        "period_days": days,
                        "channel": channel,
                        "statistics": {
                            "count": 0,
                            "average_ms": None,
                            "median_ms": None,
                            "min_ms": None,
                            "max_ms": None,
                            "std_deviation_ms": None
                        },
                        "percentiles": {},
                        "channel_breakdown": {}
                    }

                # Calculate statistics
                response_times.sort()
                count = len(response_times)
                average = sum(response_times) / count
                median = response_times[count // 2]
                minimum = min(response_times)
                maximum = max(response_times)

                # Calculate standard deviation
                variance = sum((x - average) ** 2 for x in response_times) / count
                std_deviation = variance ** 0.5

                # Calculate percentiles
                percentiles = {}
                for p in [50, 75, 90, 95, 99]:
                    index = int((p / 100) * count)
                    if index >= count:
                        index = count - 1
                    percentiles[f"p{p}"] = response_times[index]

                # Get channel breakdown
                channel_breakdown = self._get_channel_response_breakdown(
                    repo, cutoff_date
                )

                return {
                    "period_days": days,
                    "channel": channel,
                    "statistics": {
                        "count": count,
                        "average_ms": average,
                        "median_ms": median,
                        "min_ms": minimum,
                        "max_ms": maximum,
                        "std_deviation_ms": std_deviation
                    },
                    "percentiles": percentiles,
                    "channel_breakdown": channel_breakdown
                }

        except Exception:
            self._logger.exception("Failed to get response time analysis:")
            raise

    async def get_aggregated_statistics(
        self,
        granularity: TimeGranularity = TimeGranularity.DAILY,
        days: int = 30,
        channel: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated statistics by time period.

        Args:
            granularity: Time granularity for aggregation
            days: Number of days to analyze
            channel: Filter by specific channel

        Returns:
            Dictionary with time-aggregated statistics
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            db_service = get_database_service()
            with db_service.uow() as r:
                # Get time-series data
                time_series = self._get_time_series_data(
                    repo, granularity, cutoff_date, channel
                )

                # Calculate aggregated metrics
                aggregated_stats = self._calculate_aggregated_metrics(time_series)

                return {
                    "period_days": days,
                    "granularity": granularity.value,
                    "channel": channel,
                    "time_series": time_series,
                    "aggregated_metrics": aggregated_stats,
                    "summary": {
                        "total_periods": len(time_series),
                        "avg_messages_per_period": aggregated_stats.get("avg_messages", 0),
                        "peak_period": aggregated_stats.get("peak_period"),
                        "lowest_period": aggregated_stats.get("lowest_period")
                    }
                }

        except Exception:
            self._logger.exception("Failed to get aggregated statistics:")
            raise

    async def get_trend_analysis(
        self,
        metric: str = "success_rate",
        days: int = 30,
        channel: Optional[str] = None
    ) -> TrendAnalysis:
        """
        Perform trend analysis on a specific metric.

        Args:
            metric: Metric to analyze (success_rate, response_time, message_count)
            days: Number of days to analyze
            channel: Filter by specific channel

        Returns:
            TrendAnalysis object with trend information
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            db_service = get_database_service()
            with db_service.uow() as r:
                # Get time series data for the metric
                time_series_data = self._get_metric_time_series(
                    repo, metric, cutoff_date, channel
                )

                if len(time_series_data) < 2:
                    # Not enough data for trend analysis
                    return TrendAnalysis(
                        metric_name=metric,
                        time_series=time_series_data,
                        trend_direction="stable",
                        trend_strength=0.0,
                        change_percentage=0.0,
                        mean=0.0,
                        median=0.0,
                        std_deviation=0.0,
                        min_value=0.0,
                        max_value=0.0
                    )

                # Calculate trend
                values = [point.value for point in time_series_data]
                trend_analysis = self._calculate_trend(values)

                # Calculate statistics
                mean_value = sum(values) / len(values)
                sorted_values = sorted(values)
                median_value = sorted_values[len(sorted_values) // 2]
                min_value = min(values)
                max_value = max(values)

                # Calculate standard deviation
                variance = sum((x - mean_value) ** 2 for x in values) / len(values)
                std_deviation = variance ** 0.5

                # Calculate change percentage
                if len(values) >= 2:
                    change_percentage = ((values[-1] - values[0]) / values[0] * 100
                                       if values[0] != 0 else 0.0)
                else:
                    change_percentage = 0.0

                return TrendAnalysis(
                    metric_name=metric,
                    time_series=time_series_data,
                    trend_direction=trend_analysis["direction"],
                    trend_strength=trend_analysis["strength"],
                    change_percentage=change_percentage,
                    mean=mean_value,
                    median=median_value,
                    std_deviation=std_deviation,
                    min_value=min_value,
                    max_value=max_value
                )

        except Exception:
            self._logger.exception("Failed to perform trend analysis:")
            raise

    async def get_channel_performance_comparison(
        self, days: int = 30
    ) -> Dict[str, Any]:
        """
        Compare performance across all channels.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with channel performance comparison
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            # Get all channels first
            db_service = get_database_service()
            with db_service.uow() as r:
                channels = self._get_active_channels(r, cutoff_date)

            channel_comparisons = {}

            # Process each channel
            for channel in channels:
                # Get channel statistics
                with db_service.uow() as r:
                    channel_stats = self._get_channel_statistics(
                        r, channel, cutoff_date
                    )

                # Get trend analysis for this channel
                success_trend = await self.get_trend_analysis(
                    metric="success_rate",
                    days=days,
                    channel=channel
                )

                response_trend = await self.get_trend_analysis(
                    metric="response_time",
                    days=days,
                    channel=channel
                )

                channel_comparisons[channel] = {
                    "statistics": channel_stats,
                    "success_rate_trend": success_trend.to_dict(),
                    "response_time_trend": response_trend.to_dict(),
                    "performance_score": self._calculate_performance_score(
                        channel_stats, success_trend, response_trend
                    )
                }

                # Rank channels by performance
                ranked_channels = sorted(
                    channel_comparisons.items(),
                    key=lambda x: x[1]["performance_score"],
                    reverse=True
                )

                return {
                    "period_days": days,
                    "channel_comparisons": channel_comparisons,
                    "rankings": {
                        "by_performance": [
                            {"channel": channel, "score": data["performance_score"]}
                            for channel, data in ranked_channels
                        ],
                        "by_success_rate": sorted(
                            [(ch, data["statistics"].get("success_rate", 0))
                             for ch, data in channel_comparisons.items()],
                            key=lambda x: x[1], reverse=True
                        ),
                        "by_response_time": sorted(
                            [(ch, data["statistics"].get("avg_response_time_ms", float('inf')))
                             for ch, data in channel_comparisons.items()],
                            key=lambda x: x[1]
                        )
                    }
                }

        except Exception:
            self._logger.exception("Failed to get channel performance comparison:")
            raise

    # Helper methods

    def _get_message_statistics(
        self, repo, channel: Optional[str], user_id: Optional[str], cutoff_date: datetime
    ) -> Dict[str, Any]:
        """Get message statistics from database."""
        try:
            # Get basic delivery statistics
            delivery_stats = repo.delivery_status.get_delivery_statistics(
                channel=channel,
                days=(datetime.now(timezone.utc) - cutoff_date).days
            )

            return {
                "total_messages": delivery_stats.get("total_deliveries", 0),
                "by_status": delivery_stats.get("status_counts", {})
            }
        except Exception:
            self._logger.exception("Failed to get message statistics:")
            # Return default values on error
            return {
                "total_messages": 0,
                "by_status": {}
            }

    def _calculate_channel_rates(
        self, repo, channel: Optional[str], cutoff_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate delivery rates per channel."""
        try:
            channel_rates = repo.delivery_status.get_channel_delivery_rates(
                cutoff_date=cutoff_date,
                channel=channel
            )

            # Convert to ChannelStats format
            result = {}
            for ch, stats in channel_rates.items():
                channel_stats = ChannelStats(
                    channel=ch,
                    total_attempts=stats["total_attempts"],
                    successful_attempts=stats["successful_attempts"],
                    failed_attempts=stats["failed_attempts"],
                    avg_response_time_ms=stats["avg_response_time_ms"] or 0.0,
                    success_rate=stats["success_rate"]
                )
                result[ch] = channel_stats.to_dict()

            return result

        except Exception:
            self._logger.exception("Failed to calculate channel rates:")
            return {}

    def _calculate_user_rates(
        self, repo, user_id: str, cutoff_date: datetime
    ) -> Dict[str, Any]:
        """Calculate delivery rates for a specific user."""
        try:
            user_rates = repo.delivery_status.get_user_delivery_rates(
                user_id=user_id,
                cutoff_date=cutoff_date
            )
            return user_rates

        except Exception:
            self._logger.exception("Failed to calculate user rates:")
            return {
                "user_id": user_id,
                "total_messages": 0,
                "successful_deliveries": 0,
                "failed_deliveries": 0,
                "success_rate": 0.0,
                "avg_response_time_ms": None
            }

    def _get_response_time_data(
        self, repo, channel: Optional[str], cutoff_date: datetime
    ) -> List[int]:
        """Get response time data from database."""
        try:
            response_times = repo.delivery_status.get_response_time_data(
                cutoff_date=cutoff_date,
                channel=channel
            )
            return response_times

        except Exception:
            self._logger.exception("Failed to get response time data:")
            return []

    def _get_channel_response_breakdown(
        self, repo, cutoff_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """Get response time breakdown by channel."""
        # This would query and group response times by channel
        # For now, return mock data
        return {
            "telegram": {
                "count": 50,
                "average_ms": 1200.0,
                "median_ms": 1100.0,
                "min_ms": 800,
                "max_ms": 2000
            },
            "email": {
                "count": 30,
                "average_ms": 2500.0,
                "median_ms": 2400.0,
                "min_ms": 1800,
                "max_ms": 3200
            }
        }

    def _get_time_series_data(
        self, repo, granularity: TimeGranularity, cutoff_date: datetime, channel: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Get time series data for aggregation."""
        try:
            time_series = repo.delivery_status.get_time_series_data(
                cutoff_date=cutoff_date,
                granularity=granularity.value,
                channel=channel
            )

            # Convert to expected format
            result = []
            for point in time_series:
                result.append({
                    "timestamp": point["timestamp"],
                    "message_count": point["total_attempts"],
                    "success_count": point["successful_attempts"],
                    "avg_response_time_ms": point["avg_response_time_ms"]
                })

            return result

        except Exception:
            self._logger.exception("Failed to get time series data:")
            return []

    def _calculate_aggregated_metrics(self, time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregated metrics from time series data."""
        if not time_series:
            return {}

        total_messages = sum(point.get("message_count", 0) for point in time_series)
        total_success = sum(point.get("success_count", 0) for point in time_series)
        avg_messages = total_messages / len(time_series)

        # Find peak and lowest periods
        peak_period = max(time_series, key=lambda x: x.get("message_count", 0))
        lowest_period = min(time_series, key=lambda x: x.get("message_count", 0))

        return {
            "total_messages": total_messages,
            "total_success": total_success,
            "avg_messages": avg_messages,
            "overall_success_rate": total_success / total_messages if total_messages > 0 else 0,
            "peak_period": peak_period,
            "lowest_period": lowest_period
        }

    def _get_metric_time_series(
        self, repo, metric: str, cutoff_date: datetime, channel: Optional[str]
    ) -> List[TimeSeriesPoint]:
        """Get time series data for a specific metric."""
        # This would query metric data over time
        # For now, return mock data
        time_series = []
        current_date = cutoff_date
        end_date = datetime.now(timezone.utc)

        while current_date < end_date:
            if metric == "success_rate":
                value = 0.8 + (hash(current_date.isoformat()) % 20) / 100
            elif metric == "response_time":
                value = 1000 + (hash(current_date.isoformat()) % 1000)
            else:  # message_count
                value = 10 + (hash(current_date.isoformat()) % 20)

            time_series.append(TimeSeriesPoint(
                timestamp=current_date,
                value=value,
                metadata={"channel": channel} if channel else {}
            ))

            current_date += timedelta(days=1)

        return time_series

    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction and strength."""
        if len(values) < 2:
            return {"direction": "stable", "strength": 0.0}

        # Simple linear regression to determine trend
        n = len(values)
        x_values = list(range(n))

        # Calculate slope
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n

        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return {"direction": "stable", "strength": 0.0}

        slope = numerator / denominator

        # Determine direction and strength
        if abs(slope) < 0.01:
            direction = "stable"
            strength = 0.0
        elif slope > 0:
            direction = "increasing"
            strength = min(abs(slope) * 10, 1.0)  # Normalize to 0-1
        else:
            direction = "decreasing"
            strength = min(abs(slope) * 10, 1.0)  # Normalize to 0-1

        return {"direction": direction, "strength": strength}

    def _get_active_channels(self, repo, cutoff_date: datetime) -> List[str]:
        """Get list of active channels."""
        try:
            channels = repo.delivery_status.get_active_channels(cutoff_date=cutoff_date)
            return channels

        except Exception:
            self._logger.exception("Failed to get active channels:")
            return []

    def _get_channel_statistics(
        self, repo, channel: str, cutoff_date: datetime
    ) -> Dict[str, Any]:
        """Get statistics for a specific channel."""
        # This would query channel-specific statistics
        # For now, return mock data
        return {
            "total_messages": 50,
            "successful_deliveries": 45,
            "failed_deliveries": 5,
            "success_rate": 0.9,
            "avg_response_time_ms": 1200.0
        }

    def _calculate_performance_score(
        self, stats: Dict[str, Any], success_trend: TrendAnalysis, response_trend: TrendAnalysis
    ) -> float:
        """Calculate overall performance score for a channel."""
        # Weight different factors
        success_rate_weight = 0.4
        response_time_weight = 0.3
        trend_weight = 0.3

        # Success rate score (0-1)
        success_rate = stats.get("success_rate", 0)

        # Response time score (inverse, normalized)
        avg_response_time = stats.get("avg_response_time_ms", 5000)
        response_time_score = max(0, 1 - (avg_response_time / 5000))  # Normalize to 5s max

        # Trend score
        trend_score = 0.5  # Neutral
        if success_trend.trend_direction == "increasing":
            trend_score += success_trend.trend_strength * 0.3
        elif success_trend.trend_direction == "decreasing":
            trend_score -= success_trend.trend_strength * 0.3

        if response_trend.trend_direction == "decreasing":  # Lower response time is better
            trend_score += response_trend.trend_strength * 0.2
        elif response_trend.trend_direction == "increasing":
            trend_score -= response_trend.trend_strength * 0.2

        # Calculate weighted score
        performance_score = (
            success_rate * success_rate_weight +
            response_time_score * response_time_weight +
            trend_score * trend_weight
        )

        return max(0, min(1, performance_score))  # Clamp to 0-1


# Global analytics instance
notification_analytics = NotificationAnalytics()