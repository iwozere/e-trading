"""
Short Squeeze Detection Pipeline Alert Engine

This module implements the alert engine that evaluates scored candidates against
thresholds and manages alert generation with cooldown logic.
"""

from pathlib import Path
import sys
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p04_short_squeeze.core.models import (
    ScoredCandidate, Alert, AlertLevel
)
from src.ml.pipeline.p04_short_squeeze.config.data_classes import AlertConfig
from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.notification.service.client import NotificationServiceClient, MessageType, MessagePriority
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class AlertEngine:
    """
    Alert engine for short squeeze detection pipeline.

    Evaluates scored candidates against configurable thresholds and manages
    alert generation with cooldown logic and notification integration.
    """

    def __init__(self, config: AlertConfig, notification_client: Optional[NotificationServiceClient] = None):
        """
        Initialize the alert engine.

        Args:
            config: Alert configuration
            notification_client: Optional notification client (will create default if None)
        """
        self.config = config
        self.notification_client = notification_client or NotificationServiceClient()
        self._logger = setup_logger(f"{__name__}.AlertEngine")

        # Validate configuration
        self._validate_config()

        _logger.info("AlertEngine initialized with thresholds: high=%.2f, medium=%.2f, low=%.2f",
                    self.config.thresholds.high.squeeze_score,
                    self.config.thresholds.medium.squeeze_score,
                    self.config.thresholds.low.squeeze_score)

    def _validate_config(self) -> None:
        """Validate alert configuration."""
        # Check threshold ordering
        if not (self.config.thresholds.high.squeeze_score >
                self.config.thresholds.medium.squeeze_score >
                self.config.thresholds.low.squeeze_score):
            raise ValueError("Alert thresholds must be ordered: high > medium > low")

        # Check cooldown periods
        if not (self.config.cooldown.high_alert_days >=
                self.config.cooldown.medium_alert_days >=
                self.config.cooldown.low_alert_days):
            raise ValueError("Cooldown periods should be ordered: high >= medium >= low")

    def evaluate_alerts(self, scored_candidates: List[ScoredCandidate]) -> List[Alert]:
        """
        Evaluate scored candidates and generate alerts.

        Args:
            scored_candidates: List of candidates with squeeze scores

        Returns:
            List of alerts to be sent
        """
        alerts = []

        for candidate in scored_candidates:
            try:
                alert = self._evaluate_candidate_alert(candidate)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                self._logger.error("Error evaluating alert for %s: %s",
                                 candidate.candidate.ticker, e)

        self._logger.info("Generated %d alerts from %d candidates",
                         len(alerts), len(scored_candidates))

        return alerts

    def _evaluate_candidate_alert(self, scored_candidate: ScoredCandidate) -> Optional[Alert]:
        """
        Evaluate a single candidate for alert generation.

        Args:
            scored_candidate: Candidate with squeeze score

        Returns:
            Alert if conditions are met, None otherwise
        """
        ticker = scored_candidate.candidate.ticker
        squeeze_score = scored_candidate.squeeze_score
        structural = scored_candidate.candidate.structural_metrics
        transient = scored_candidate.transient_metrics

        # Determine alert level based on thresholds
        alert_level = self._determine_alert_level(squeeze_score, structural, transient)

        if not alert_level:
            self._logger.debug("No alert level met for %s (score: %.3f)", ticker, squeeze_score)
            return None

        # Check cooldown
        if self._is_in_cooldown(ticker, alert_level):
            self._logger.debug("Ticker %s is in cooldown for %s alerts", ticker, alert_level.value)
            return None

        # Generate alert reason
        reason = self._generate_alert_reason(scored_candidate, alert_level)

        # Calculate cooldown expiration
        cooldown_expires = self._calculate_cooldown_expiration(alert_level)

        alert = Alert(
            ticker=ticker,
            alert_level=alert_level,
            reason=reason,
            squeeze_score=squeeze_score,
            timestamp=datetime.now(),
            cooldown_expires=cooldown_expires
        )

        self._logger.info("Generated %s alert for %s: score=%.3f, reason=%s",
                         alert_level.value, ticker, squeeze_score, reason)

        return alert

    def _determine_alert_level(self,
                             squeeze_score: float,
                             structural: Any,
                             transient: Any) -> Optional[AlertLevel]:
        """
        Determine alert level based on score and additional criteria.

        Args:
            squeeze_score: Final squeeze score
            structural: Structural metrics
            transient: Transient metrics

        Returns:
            Alert level if thresholds are met, None otherwise
        """
        # Check high threshold
        high_threshold = self.config.thresholds.high
        if (squeeze_score >= high_threshold.squeeze_score and
            structural.short_interest_pct >= high_threshold.min_si_percent and
            transient.volume_spike >= high_threshold.min_volume_spike and
            transient.sentiment_24h >= high_threshold.min_sentiment):
            return AlertLevel.HIGH

        # Check medium threshold
        medium_threshold = self.config.thresholds.medium
        if (squeeze_score >= medium_threshold.squeeze_score and
            structural.short_interest_pct >= medium_threshold.min_si_percent and
            transient.volume_spike >= medium_threshold.min_volume_spike and
            transient.sentiment_24h >= medium_threshold.min_sentiment):
            return AlertLevel.MEDIUM

        # Check low threshold
        low_threshold = self.config.thresholds.low
        if (squeeze_score >= low_threshold.squeeze_score and
            structural.short_interest_pct >= low_threshold.min_si_percent and
            transient.volume_spike >= low_threshold.min_volume_spike and
            transient.sentiment_24h >= low_threshold.min_sentiment):
            return AlertLevel.LOW

        return None

    def _is_in_cooldown(self, ticker: str, alert_level: AlertLevel) -> bool:
        """
        Check if ticker is in cooldown for the given alert level.

        Args:
            ticker: Stock ticker
            alert_level: Alert level to check

        Returns:
            True if in cooldown, False otherwise
        """
        try:
            # Service manages sessions internally via UoW pattern
            service = ShortSqueezeService()
            return service.repo.alerts.check_cooldown(ticker, alert_level)
        except Exception as e:
            self._logger.error("Error checking cooldown for %s: %s", ticker, e)
            # Assume not in cooldown on error to avoid missing alerts
            return False

    def _generate_alert_reason(self, scored_candidate: ScoredCandidate, alert_level: AlertLevel) -> str:
        """
        Generate human-readable alert reason.

        Args:
            scored_candidate: Scored candidate
            alert_level: Alert level

        Returns:
            Alert reason string
        """
        ticker = scored_candidate.candidate.ticker
        squeeze_score = scored_candidate.squeeze_score
        structural = scored_candidate.candidate.structural_metrics
        transient = scored_candidate.transient_metrics

        # Build reason components
        reasons = []

        # Squeeze score
        reasons.append(f"Squeeze score: {squeeze_score:.1%}")

        # Key structural metrics
        reasons.append(f"Short interest: {structural.short_interest_pct:.1%}")
        reasons.append(f"Days to cover: {structural.days_to_cover:.1f}")

        # Key transient metrics
        reasons.append(f"Volume spike: {transient.volume_spike:.1f}x")
        reasons.append(f"Sentiment: {transient.sentiment_24h:.2f}")

        # Optional metrics
        if transient.call_put_ratio is not None:
            reasons.append(f"Call/Put ratio: {transient.call_put_ratio:.2f}")

        if transient.borrow_fee_pct is not None:
            reasons.append(f"Borrow fee: {transient.borrow_fee_pct:.1%}")

        return f"{alert_level.value} squeeze alert for {ticker}: " + ", ".join(reasons)

    def _calculate_cooldown_expiration(self, alert_level: AlertLevel) -> datetime:
        """
        Calculate cooldown expiration time for alert level.

        Args:
            alert_level: Alert level

        Returns:
            Cooldown expiration datetime
        """
        if alert_level == AlertLevel.HIGH:
            days = self.config.cooldown.high_alert_days
        elif alert_level == AlertLevel.MEDIUM:
            days = self.config.cooldown.medium_alert_days
        else:  # LOW
            days = self.config.cooldown.low_alert_days

        return datetime.now() + timedelta(days=days)

    async def send_alert(self, alert: Alert) -> bool:
        """
        Send alert through notification system.

        Args:
            alert: Alert to send

        Returns:
            True if alert was sent successfully
        """
        try:
            # Determine notification priority based on alert level
            if alert.alert_level == AlertLevel.HIGH:
                priority = MessagePriority.CRITICAL
            elif alert.alert_level == AlertLevel.MEDIUM:
                priority = MessagePriority.HIGH
            else:  # LOW
                priority = MessagePriority.NORMAL

            # Prepare notification content
            title = f"🚨 {alert.alert_level.value} Short Squeeze Alert: {alert.ticker}"
            message = alert.reason

            # Add additional context
            additional_info = [
                f"⏰ Alert time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                f"📊 Squeeze score: {alert.squeeze_score:.1%}",
                f"🔄 Next alert after: {alert.cooldown_expires.strftime('%Y-%m-%d %H:%M:%S') if alert.cooldown_expires else 'N/A'}"
            ]

            message += "\n\n" + "\n".join(additional_info)

            # Determine channels based on configuration
            channels = []
            if self.config.channels.telegram_enabled:
                channels.append("telegram")
            if self.config.channels.email_enabled:
                channels.append("email")

            # Send notification
            success = await self.notification_client.send_notification(
                notification_type=MessageType.ALERT,
                title=title,
                message=message,
                priority=priority,
                channels=channels,
                data={
                    "ticker": alert.ticker,
                    "alert_level": alert.alert_level.value,
                    "squeeze_score": alert.squeeze_score,
                    "alert_type": "short_squeeze"
                }
            )

            if success:
                self._logger.info("Successfully sent %s alert for %s",
                                alert.alert_level.value, alert.ticker)

                # Mark alert as sent in database
                await self._mark_alert_sent(alert, "notification_sent")
            else:
                self._logger.error("Failed to send %s alert for %s",
                                 alert.alert_level.value, alert.ticker)

            return success

        except Exception as e:
            self._logger.error("Error sending alert for %s: %s", alert.ticker, e)
            return False

    async def _mark_alert_sent(self, alert: Alert, notification_id: str) -> None:
        """
        Mark alert as sent in database.

        Args:
            alert: Alert that was sent
            notification_id: Notification system ID
        """
        try:
            # Service manages sessions internally via UoW pattern
            service = ShortSqueezeService()

            # First create the alert in database if it doesn't exist
            alert_id = service.create_alert(
                ticker=alert.ticker,
                alert_level=alert.alert_level,
                reason=alert.reason,
                squeeze_score=alert.squeeze_score,
                cooldown_days=self._get_cooldown_days(alert.alert_level)
            )

            if alert_id:
                # Mark as sent
                service.mark_alert_sent(alert_id, notification_id)
                alert.sent = True
                alert.notification_id = notification_id

        except Exception as e:
            self._logger.error("Error marking alert as sent for %s: %s", alert.ticker, e)

    def _get_cooldown_days(self, alert_level: AlertLevel) -> int:
        """Get cooldown days for alert level."""
        if alert_level == AlertLevel.HIGH:
            return self.config.cooldown.high_alert_days
        elif alert_level == AlertLevel.MEDIUM:
            return self.config.cooldown.medium_alert_days
        else:  # LOW
            return self.config.cooldown.low_alert_days

    async def process_alerts(self, scored_candidates: List[ScoredCandidate]) -> Dict[str, Any]:
        """
        Complete alert processing workflow.

        Args:
            scored_candidates: List of scored candidates

        Returns:
            Processing results summary
        """
        try:
            # Generate alerts
            alerts = self.evaluate_alerts(scored_candidates)

            if not alerts:
                self._logger.info("No alerts generated from %d candidates", len(scored_candidates))
                return {
                    "candidates_processed": len(scored_candidates),
                    "alerts_generated": 0,
                    "alerts_sent": 0,
                    "alerts_failed": 0
                }

            # Send alerts
            sent_count = 0
            failed_count = 0

            for alert in alerts:
                try:
                    success = await self.send_alert(alert)
                    if success:
                        sent_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    self._logger.error("Failed to process alert for %s: %s", alert.ticker, e)
                    failed_count += 1

            results = {
                "candidates_processed": len(scored_candidates),
                "alerts_generated": len(alerts),
                "alerts_sent": sent_count,
                "alerts_failed": failed_count,
                "alert_details": [
                    {
                        "ticker": alert.ticker,
                        "level": alert.alert_level.value,
                        "score": alert.squeeze_score,
                        "sent": alert.sent
                    }
                    for alert in alerts
                ]
            }

            self._logger.info("Alert processing complete: %s", results)
            return results

        except Exception as e:
            self._logger.exception("Error in alert processing:")
            return {
                "candidates_processed": len(scored_candidates),
                "alerts_generated": 0,
                "alerts_sent": 0,
                "alerts_failed": 0,
                "error": str(e)
            }

    def get_alert_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get alert statistics for the specified period.

        Args:
            days: Number of days to look back

        Returns:
            Alert statistics
        """
        try:
            # Service manages sessions internally via UoW pattern
            service = ShortSqueezeService()

            # Get recent alerts
            recent_alerts = service.repo.alerts.get_recent_alerts(days)

                # Count by level
                level_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
                sent_count = 0

                for alert in recent_alerts:
                    level_counts[alert.alert_level] += 1
                    if alert.sent:
                        sent_count += 1

                return {
                    "period_days": days,
                    "total_alerts": len(recent_alerts),
                    "alerts_sent": sent_count,
                    "alerts_pending": len(recent_alerts) - sent_count,
                    "by_level": level_counts,
                    "success_rate": sent_count / len(recent_alerts) if recent_alerts else 0.0
                }

        except Exception as e:
            self._logger.exception("Error getting alert statistics:")
            return {"error": str(e)}

    async def close(self):
        """Close notification client connection."""
        if self.notification_client:
            await self.notification_client.close()