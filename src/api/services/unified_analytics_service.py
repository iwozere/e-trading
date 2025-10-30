"""
Unified Analytics Service for Main API

Consolidates analytics functionality for both notifications and trading data.
Provides a unified interface for analytics across all system domains.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class UnifiedAnalyticsService:
    """
    Unified analytics service that consolidates analytics for notifications and trading.

    This service provides a single interface for all analytics operations,
    supporting both notification delivery analytics and trading performance analytics.
    """

    def __init__(self):
        """Initialize the unified analytics service."""
        self._logger = setup_logger(f"{__name__}.UnifiedAnalyticsService")

        # Initialize notification analytics
        self._notification_analytics = None
        self._trading_analytics = None  # Future implementation

    def _get_notification_analytics(self):
        """Get notification analytics instance (lazy loading)."""
        if self._notification_analytics is None:
            try:
                from src.notification.service.analytics import notification_analytics
                self._notification_analytics = notification_analytics
                self._logger.info("Notification analytics initialized")
            except Exception as e:
                self._logger.exception("Failed to initialize notification analytics:")
                raise

        return self._notification_analytics

    def _get_trading_analytics(self):
        """Get trading analytics instance (future implementation)."""
        if self._trading_analytics is None:
            # TODO: Initialize trading analytics when implemented
            self._logger.warning("Trading analytics not yet implemented")
            return None

        return self._trading_analytics

    # Notification Analytics Methods

    async def get_notification_delivery_rates(
        self,
        channel: Optional[str] = None,
        user_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get notification delivery rate analytics.

        Args:
            channel: Filter by specific channel
            user_id: Filter by specific user
            days: Number of days to analyze

        Returns:
            Delivery rate analytics
        """
        try:
            analytics = self._get_notification_analytics()
            return await analytics.get_delivery_rates(
                channel=channel,
                user_id=user_id,
                days=days
            )
        except Exception as e:
            self._logger.exception("Failed to get notification delivery rates:")
            raise

    async def get_notification_response_times(
        self,
        channel: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get notification response time analytics.

        Args:
            channel: Filter by specific channel
            days: Number of days to analyze

        Returns:
            Response time analytics
        """
        try:
            analytics = self._get_notification_analytics()
            return await analytics.get_response_time_analysis(
                channel=channel,
                days=days
            )
        except Exception as e:
            self._logger.exception("Failed to get notification response times:")
            raise

    async def get_notification_trends(
        self,
        metric: str = "success_rate",
        channel: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get notification trend analysis.

        Args:
            metric: Metric to analyze
            channel: Filter by specific channel
            days: Number of days to analyze

        Returns:
            Trend analysis
        """
        try:
            analytics = self._get_notification_analytics()
            trend_analysis = await analytics.get_trend_analysis(
                metric=metric,
                days=days,
                channel=channel
            )
            return trend_analysis.to_dict()
        except Exception as e:
            self._logger.exception("Failed to get notification trends:")
            raise

    async def get_notification_aggregated_stats(
        self,
        granularity: str = "daily",
        channel: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get aggregated notification statistics.

        Args:
            granularity: Time granularity
            channel: Filter by specific channel
            days: Number of days to analyze

        Returns:
            Aggregated statistics
        """
        try:
            from src.notification.service.analytics import TimeGranularity

            analytics = self._get_notification_analytics()
            granularity_enum = TimeGranularity(granularity)

            return await analytics.get_aggregated_statistics(
                granularity=granularity_enum,
                days=days,
                channel=channel
            )
        except Exception as e:
            self._logger.exception("Failed to get notification aggregated stats:")
            raise

    async def get_notification_channel_comparison(
        self, days: int = 30
    ) -> Dict[str, Any]:
        """
        Get notification channel performance comparison.

        Args:
            days: Number of days to analyze

        Returns:
            Channel performance comparison
        """
        try:
            analytics = self._get_notification_analytics()
            return await analytics.get_channel_performance_comparison(days=days)
        except Exception as e:
            self._logger.exception("Failed to get notification channel comparison:")
            raise

    # Trading Analytics Methods (Future Implementation)

    async def get_trading_performance_analytics(
        self,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get trading performance analytics (future implementation).

        Args:
            strategy_id: Filter by specific strategy
            symbol: Filter by specific symbol
            days: Number of days to analyze

        Returns:
            Trading performance analytics
        """
        # TODO: Implement trading analytics
        self._logger.warning("Trading analytics not yet implemented")
        return {
            "message": "Trading analytics not yet implemented",
            "available_methods": [
                "get_notification_delivery_rates",
                "get_notification_response_times",
                "get_notification_trends",
                "get_notification_aggregated_stats",
                "get_notification_channel_comparison"
            ]
        }

    async def get_strategy_performance_comparison(
        self, days: int = 30
    ) -> Dict[str, Any]:
        """
        Get strategy performance comparison (future implementation).

        Args:
            days: Number of days to analyze

        Returns:
            Strategy performance comparison
        """
        # TODO: Implement strategy analytics
        self._logger.warning("Strategy analytics not yet implemented")
        return {
            "message": "Strategy analytics not yet implemented",
            "note": "This will be implemented as part of trading analytics consolidation"
        }

    # Cross-Domain Analytics Methods

    async def get_unified_dashboard_data(
        self, days: int = 30
    ) -> Dict[str, Any]:
        """
        Get unified dashboard data combining notifications and trading analytics.

        Args:
            days: Number of days to analyze

        Returns:
            Unified dashboard data
        """
        try:
            # Get notification analytics
            notification_data = {}
            try:
                notification_data = {
                    "delivery_rates": await self.get_notification_delivery_rates(days=days),
                    "channel_comparison": await self.get_notification_channel_comparison(days=days),
                    "success_trend": await self.get_notification_trends(
                        metric="success_rate", days=days
                    )
                }
            except Exception as e:
                self._logger.exception("Failed to get notification data for dashboard:")
                notification_data = {"error": str(e)}

            # Get trading analytics (future)
            trading_data = {
                "message": "Trading analytics not yet implemented"
            }

            return {
                "period_days": days,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "notifications": notification_data,
                "trading": trading_data,
                "cross_domain_insights": {
                    "total_system_health": self._calculate_system_health_score(
                        notification_data, trading_data
                    )
                }
            }

        except Exception as e:
            self._logger.exception("Failed to get unified dashboard data:")
            raise

    async def get_correlation_analysis(
        self,
        notification_metric: str = "success_rate",
        trading_metric: str = "win_rate",
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze correlations between notification and trading metrics (future implementation).

        Args:
            notification_metric: Notification metric to analyze
            trading_metric: Trading metric to analyze
            days: Number of days to analyze

        Returns:
            Correlation analysis results
        """
        # TODO: Implement cross-domain correlation analysis
        self._logger.warning("Cross-domain correlation analysis not yet implemented")
        return {
            "message": "Cross-domain correlation analysis not yet implemented",
            "requested_analysis": {
                "notification_metric": notification_metric,
                "trading_metric": trading_metric,
                "period_days": days
            }
        }

    # Helper Methods

    def _calculate_system_health_score(
        self, notification_data: Dict[str, Any], trading_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate overall system health score based on all analytics.

        Args:
            notification_data: Notification analytics data
            trading_data: Trading analytics data

        Returns:
            System health score and breakdown
        """
        try:
            # Calculate notification health score
            notification_score = 0.5  # Default neutral score

            if "delivery_rates" in notification_data and "overall_statistics" in notification_data["delivery_rates"]:
                success_rate = notification_data["delivery_rates"]["overall_statistics"].get("success_rate", 0)
                notification_score = success_rate

            # Calculate trading health score (future)
            trading_score = 0.5  # Default neutral score when not implemented

            # Calculate overall score (weighted average)
            notification_weight = 0.4
            trading_weight = 0.6  # Trading is more critical for system health

            overall_score = (
                notification_score * notification_weight +
                trading_score * trading_weight
            )

            # Determine health status
            if overall_score >= 0.8:
                health_status = "excellent"
            elif overall_score >= 0.6:
                health_status = "good"
            elif overall_score >= 0.4:
                health_status = "fair"
            else:
                health_status = "poor"

            return {
                "overall_score": overall_score,
                "health_status": health_status,
                "breakdown": {
                    "notification_score": notification_score,
                    "trading_score": trading_score,
                    "weights": {
                        "notifications": notification_weight,
                        "trading": trading_weight
                    }
                },
                "recommendations": self._generate_health_recommendations(
                    overall_score, notification_score, trading_score
                )
            }

        except Exception as e:
            self._logger.exception("Failed to calculate system health score:")
            return {
                "overall_score": 0.0,
                "health_status": "unknown",
                "error": str(e)
            }

    def _generate_health_recommendations(
        self, overall_score: float, notification_score: float, trading_score: float
    ) -> List[str]:
        """
        Generate health improvement recommendations.

        Args:
            overall_score: Overall system health score
            notification_score: Notification system score
            trading_score: Trading system score

        Returns:
            List of recommendations
        """
        recommendations = []

        if notification_score < 0.6:
            recommendations.append("Investigate notification delivery issues")
            recommendations.append("Check channel health and configuration")

        if trading_score < 0.6:
            recommendations.append("Review trading strategy performance")
            recommendations.append("Check risk management settings")

        if overall_score < 0.5:
            recommendations.append("Consider system maintenance window")
            recommendations.append("Review overall system configuration")

        if not recommendations:
            recommendations.append("System is performing well")

        return recommendations

    def get_available_analytics(self) -> Dict[str, Any]:
        """
        Get information about available analytics methods.

        Returns:
            Dictionary describing available analytics
        """
        return {
            "notification_analytics": {
                "available": True,
                "methods": [
                    "get_notification_delivery_rates",
                    "get_notification_response_times",
                    "get_notification_trends",
                    "get_notification_aggregated_stats",
                    "get_notification_channel_comparison"
                ]
            },
            "trading_analytics": {
                "available": False,
                "methods": [
                    "get_trading_performance_analytics",
                    "get_strategy_performance_comparison"
                ],
                "note": "Trading analytics will be implemented in future updates"
            },
            "cross_domain_analytics": {
                "available": "partial",
                "methods": [
                    "get_unified_dashboard_data",
                    "get_correlation_analysis"
                ],
                "note": "Cross-domain analytics available for notifications only"
            }
        }


# Global unified analytics service instance
unified_analytics_service = UnifiedAnalyticsService()