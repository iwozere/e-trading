"""
Alert Monitor V2 - Updated to use Notification Service

Background service to monitor price alerts and send notifications when triggered.
Uses the new notification service client instead of direct AsyncNotificationManager.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from src.data.db.services import telegram_service as db
from src.common import get_ohlcv
from src.telegram.screener.alert_logic_evaluator import AlertLogicEvaluator, evaluate_alert
from src.telegram.screener.rearm_alert_system import ReArmAlertEvaluator

# Import new notification service client
from src.notification.client import NotificationServiceClient, NotificationServiceError
from src.model.notification import NotificationType, NotificationPriority

from src.notification.logger import setup_logger, set_logging_context
_logger = setup_logger(__name__)


class AlertMonitorV2:
    """
    Background service to monitor price alerts and send notifications when triggered.
    Updated to use the new notification service client.
    """

    def __init__(self, notification_service_url: str = "http://localhost:8080", telegram_service=None):
        """
        Initialize the alert monitor.

        Args:
            notification_service_url: URL of the notification service
            telegram_service: Telegram service instance for database operations
        """
        self.notification_client = NotificationServiceClient(
            base_url=notification_service_url,
            timeout=30,
            max_retries=3
        )
        self.running = False
        self.evaluator = AlertLogicEvaluator()
        self.telegram_service = telegram_service or db
        self.rearm_evaluator = ReArmAlertEvaluator(telegram_service=self.telegram_service)

    async def start(self):
        """Start the alert monitoring loop."""
        self.running = True

        # Set logging context
        set_logging_context("telegram_alert_monitor_v2")

        _logger.info("Alert monitor V2 started (using notification service)")

        # Test notification service connectivity
        try:
            health = await self.notification_client.health_check_async()
            _logger.info("Notification service health: %s", health.get('status', 'unknown'))
        except Exception as e:
            _logger.warning("Could not connect to notification service: %s", e)

        while self.running:
            try:
                await self.check_alerts()
                await asyncio.sleep(5 * 60)  # Check every 5 minutes
            except Exception as e:
                _logger.exception("Error in alert monitor: ")
                await asyncio.sleep(5 * 60)  # Shorter sleep on error (5 minutes)

    async def stop(self):
        """Stop the alert monitoring loop."""
        self.running = False
        await self.notification_client.close_async()
        _logger.info("Alert monitor V2 stopped")

    async def check_alerts(self):
        """Check all active alerts and trigger notifications if conditions are met."""
        try:
            # Get all active alerts using the service layer with error handling
            start_time = time.time()
            try:
                _logger.debug("Retrieving active alerts from telegram service")
                alerts = self.telegram_service.list_active_alerts()
                elapsed_ms = int((time.time() - start_time) * 1000)
                _logger.debug("Retrieved %d active alerts from service (took %dms)", len(alerts) if alerts else 0, elapsed_ms)
            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                _logger.error("Failed to retrieve active alerts from service (took %dms): %s", elapsed_ms, e)
                return  # Skip this check cycle if we can't get alerts

            if not alerts:
                _logger.debug("No active alerts to check")
                return

            _logger.info("Checking %d active alerts", len(alerts))

            for alert in alerts:
                await self.check_single_alert(alert)

        except Exception as e:
            _logger.exception("Error checking alerts: ")

    async def check_single_alert(self, alert: Dict[str, Any]):
        """Check a single alert and trigger notification if condition is met."""
        try:
            alert_id = alert["id"]
            user_id = alert["user_id"]
            ticker = alert["ticker"]
            alert_type = alert.get("alert_type", "price")

            # Get current price with improved error handling
            current_price = await self._get_current_price(ticker)
            if current_price is None:
                _logger.warning("Could not get current price for %s, skipping alert %d", ticker, alert_id)
                return

            # Check if this is a re-arm alert (has re_arm_config)
            if alert.get("re_arm_config"):
                # Use re-arm evaluator
                triggered, evaluation_details = self.rearm_evaluator.evaluate_alert(alert, current_price)

                # Update alert state using service layer (ReArmAlertEvaluator now handles this internally)
                success = self.rearm_evaluator.update_alert_state(alert_id, evaluation_details, current_price)
                if not success:
                    _logger.warning("Failed to update alert state for alert %d", alert_id)

                if triggered:
                    await self.trigger_rearm_alert(alert, evaluation_details)

            else:
                # Use legacy evaluator for indicator alerts or old price alerts
                if alert_type == "indicator":
                    triggered, evaluation_details = await self.evaluator.evaluate_alert(alert)
                else:
                    # Legacy price alert logic
                    triggered, evaluation_details = self._evaluate_legacy_price_alert(alert, current_price)

                if "error" in evaluation_details:
                    _logger.error("Error evaluating alert %d for %s: %s", alert_id, ticker, evaluation_details["error"])
                    if alert_type == "indicator":
                        await self._notify_admin_of_error(alert, evaluation_details["error"])
                    return

                if triggered:
                    await self.trigger_alert(alert, evaluation_details)

        except Exception as e:
            _logger.exception("Error checking alert %s: ", alert.get("id"))
            await self._notify_admin_of_error(alert, str(e))

    async def trigger_alert(self, alert: Dict[str, Any], evaluation_details: Dict[str, Any]):
        """Trigger an alert notification and deactivate the alert."""
        try:
            ticker = alert["ticker"]
            user_id = alert["user_id"]
            alert_id = alert["id"]
            alert_type = alert.get("alert_type", "price")
            alert_action = alert.get("alert_action", "notify")

            # Get user info for notifications
            _logger.debug("Retrieving user status for alert notification: user_id=%s, alert_id=%s", user_id, alert_id)
            user_status = self.telegram_service.get_user_status(str(user_id))
            user_email = user_status.get("email") if user_status and user_status.get("verified") else None
            _logger.debug("User status retrieved for alert: user_id=%s, verified=%s, has_email=%s",
                         user_id, user_status.get("verified") if user_status else False, bool(user_email))

            # Prepare notification message based on alert type
            if alert_type == "price":
                message = self._format_price_alert_message(alert, evaluation_details)
                title = f"Price Alert: {ticker}"
                notification_type = NotificationType.WARNING
            else:
                message = self._format_indicator_alert_message(alert, evaluation_details)
                title = f"Indicator Alert: {ticker} - {alert_action}"
                notification_type = NotificationType.INFO

            # Determine channels and recipient
            channels = []
            recipient_id = None

            # Always include Telegram if we have a user_id
            if user_id:
                channels.append("telegram")
                recipient_id = str(user_id)

            # Include email if user has verified email
            if user_email:
                channels.append("email")
                if not recipient_id:
                    recipient_id = user_email

            # Prepare metadata for Telegram
            metadata = {
                "telegram_chat_id": str(user_id),
                "alert_id": alert_id,
                "ticker": ticker,
                "alert_type": alert_type
            }

            # Send notification via notification service
            try:
                response = await self.notification_client.send_notification_async(
                    notification_type=notification_type,
                    title=title,
                    message=message,
                    priority=NotificationPriority.HIGH,
                    channels=channels,
                    recipient_id=recipient_id,
                    metadata=metadata
                )

                _logger.info("Alert #%d notification sent via service (ID: %d) for user %s: %s",
                           alert_id, response.message_id, user_id, ticker)

            except NotificationServiceError as e:
                _logger.error("Failed to send alert notification via service for user %s, alert #%d: %s",
                            user_id, alert_id, e)

                # Fallback: try to send via legacy method if service is down
                await self._send_fallback_notification(user_id, message, title)

            # Deactivate the alert (one-time trigger) using service layer with error handling
            try:
                self.telegram_service.update_alert(alert_id, active=False)
                _logger.debug("Alert %d deactivated after triggering", alert_id)
            except Exception as e:
                _logger.error("Failed to deactivate alert %d: %s", alert_id, e)

        except Exception as e:
            _logger.exception("Error triggering alert %s: ", alert.get("id"))

    async def trigger_rearm_alert(self, alert: Dict[str, Any], evaluation_details: Dict[str, Any]):
        """Trigger a re-arm alert notification."""
        try:
            from src.telegram.screener.rearm_alert_system import EnhancedAlertConfig

            ticker = alert["ticker"]
            user_id = alert["user_id"]
            alert_id = alert["id"]

            # Parse enhanced config
            config_json = alert.get("re_arm_config", "{}")
            try:
                import json
                config_dict = json.loads(config_json)
                config = EnhancedAlertConfig.from_dict(config_dict)
            except (json.JSONDecodeError, KeyError) as e:
                _logger.error("Invalid re_arm_config for alert %d: %s", alert_id, e)
                return

            # Format message using enhanced config
            message = self.rearm_evaluator.format_notification_message(config, evaluation_details)
            title = f"🚨 Price Alert: {ticker}"

            # Get user info for notifications
            _logger.debug("Retrieving user status for rearm alert notification: user_id=%s, alert_id=%s", user_id, alert_id)
            user_status = self.telegram_service.get_user_status(str(user_id))
            user_email = user_status.get("email") if user_status and user_status.get("verified") else None
            _logger.debug("User status retrieved for rearm alert: user_id=%s, verified=%s, has_email=%s",
                         user_id, user_status.get("verified") if user_status else False, bool(user_email))

            # Determine channels based on config
            channels = []
            recipient_id = None

            # Check config for enabled channels
            if "telegram" in config.notification_config.channels:
                channels.append("telegram")
                recipient_id = str(user_id)

            if "email" in config.notification_config.channels and user_email:
                channels.append("email")
                if not recipient_id:
                    recipient_id = user_email

            # Default to telegram if no channels specified
            if not channels and user_id:
                channels = ["telegram"]
                recipient_id = str(user_id)

            # Prepare metadata
            metadata = {
                "telegram_chat_id": str(user_id),
                "alert_id": alert_id,
                "ticker": ticker,
                "alert_type": "rearm_price",
                "rearm_config": config_json
            }

            # Send notification via notification service
            try:
                response = await self.notification_client.send_notification_async(
                    notification_type=NotificationType.WARNING,
                    title=title,
                    message=message,
                    priority=NotificationPriority.HIGH,
                    channels=channels,
                    recipient_id=recipient_id,
                    metadata=metadata
                )

                _logger.info("Re-arm alert #%d notification sent via service (ID: %d) for user %s: %s",
                           alert_id, response.message_id, user_id, ticker)

            except NotificationServiceError as e:
                _logger.error("Failed to send re-arm alert notification via service for user %s, alert #%d: %s",
                            user_id, alert_id, e)

                # Fallback: try to send via legacy method if service is down
                await self._send_fallback_notification(user_id, message, title)

        except Exception as e:
            _logger.exception("Error triggering re-arm alert %s: ", alert.get("id"))

    async def _notify_admin_of_error(self, alert: Dict[str, Any], error_message: str):
        """Send error notification to admin(s) via notification service."""
        try:
            # Get all admin users using service layer
            admin_users = []
            all_users = self.telegram_service.list_users()
            for user in all_users:
                if user.get("is_admin"):
                    admin_users.append(user["telegram_user_id"])

            if not admin_users:
                _logger.warning("No admin users found for error notification")
                return

            error_message_text = (
                f"⚠️ Alert Evaluation Error\n\n"
                f"Alert ID: #{alert.get('id')}\n"
                f"Ticker: {alert.get('ticker')}\n"
                f"User: {alert.get('user_id')}\n"
                f"Error: {error_message}\n\n"
                f"Please check the alert configuration and system logs."
            )

            # Send to all admins via notification service
            for admin_id in admin_users:
                try:
                    await self.notification_client.send_error_notification_async(
                        error_message=error_message_text,
                        source="alert_monitor",
                        channels=["telegram"],
                        metadata={"telegram_chat_id": str(admin_id)}
                    )
                except NotificationServiceError as e:
                    _logger.error("Failed to send error notification to admin %s via service: %s", admin_id, e)
                    # Fallback to legacy method
                    await self._send_fallback_notification(admin_id, error_message_text, "Alert System Error")

            _logger.info("Error notification sent to %d admin(s) via notification service", len(admin_users))

        except Exception as e:
            _logger.error("Failed to send admin error notification: %s", e)

    async def _send_fallback_notification(self, user_id: int, message: str, title: str):
        """Fallback notification method when notification service is unavailable."""
        try:
            from src.telegram.screener.http_api_client import send_notification_via_api

            success = await send_notification_via_api(
                user_id=user_id,
                message=message,
                title=title
            )

            if success:
                _logger.info("Fallback notification sent to user %s", user_id)
            else:
                _logger.error("Fallback notification failed for user %s", user_id)

        except Exception as e:
            _logger.error("Fallback notification error for user %s: %s", user_id, e)

    def _format_price_alert_message(self, alert: Dict[str, Any], details: Dict[str, Any]) -> str:
        """Format price alert message."""
        ticker = alert["ticker"]
        target_price = alert["price"]
        condition = alert["condition"]
        alert_id = alert["id"]
        current_price = details.get("current_price", 0)

        return (
            f"🚨 Price Alert Triggered!\n\n"
            f"Ticker: {ticker}\n"
            f"Current Price: ${current_price:.2f}\n"
            f"Alert: {condition} ${target_price:.2f}\n"
            f"Alert ID: #{alert_id}"
        )

    def _format_indicator_alert_message(self, alert: Dict[str, Any], details: Dict[str, Any]) -> str:
        """Format indicator alert message."""
        ticker = alert["ticker"]
        alert_id = alert["id"]
        alert_action = alert.get("alert_action", "notify")
        current_price = details.get("current_price", 0)
        timeframe = alert.get("timeframe", "15m")

        # Get alert summary
        summary = self.evaluator.get_alert_summary(alert)

        message = (
            f"🚨 Indicator Alert Triggered!\n\n"
            f"Ticker: {ticker}\n"
            f"Current Price: ${current_price:.2f}\n"
            f"Timeframe: {timeframe}\n"
            f"Action: {alert_action}\n"
            f"Alert ID: #{alert_id}\n\n"
        )

        if "indicators" in summary:
            message += f"Indicators: {', '.join(summary['indicators'])}\n"
            if "logic" in summary and summary["logic"] != "single":
                message += f"Logic: {summary['logic']}\n"

        # Add condition results if available
        condition_results = details.get("condition_results", [])
        if condition_results:
            message += "\nCondition Results:\n"
            for result in condition_results:
                indicator = result.get("indicator", "Unknown")
                result_status = "✅" if result.get("result") else "❌"
                value = result.get("value", "N/A")
                message += f"  {result_status} {indicator}: {value}\n"

        return message

    async def _get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for ticker with fallback providers."""
        providers_to_try = ["yf", "alpha_vantage", "polygon"]

        for provider in providers_to_try:
            try:
                # Try different intervals and periods for better success rate
                intervals_to_try = ["5m", "1m", "1d"]
                periods_to_try = ["1d", "5d"]

                for interval in intervals_to_try:
                    for period in periods_to_try:
                        try:
                            data = get_ohlcv(ticker, period=period, interval=interval, provider=provider)
                            if data is not None and not data.empty and 'Close' in data.columns:
                                current_price = float(data['Close'].iloc[-1])
                                _logger.debug("Got current price for %s from %s: $%.2f", ticker, provider, current_price)
                                return current_price
                        except Exception as e:
                            _logger.debug("Failed to get price for %s with %s/%s/%s: %s", ticker, provider, period, interval, e)
                            continue

            except Exception as e:
                _logger.debug("Provider %s failed for %s: %s", provider, ticker, e)
                continue

        _logger.warning("Could not get current price for %s from any provider", ticker)
        return None

    def _evaluate_legacy_price_alert(self, alert: Dict[str, Any], current_price: float) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate legacy price alert (simple threshold check)."""
        try:
            threshold = float(alert["price"])
            condition = alert["condition"]

            if condition == "above":
                triggered = current_price > threshold
            else:  # "below"
                triggered = current_price < threshold

            return triggered, {
                "current_price": current_price,
                "threshold": threshold,
                "condition": condition,
                "alert_type": "legacy_price"
            }

        except Exception as e:
            return False, {"error": f"Error evaluating legacy price alert: {str(e)}"}


async def main():
    """Main function to run the alert monitor as a standalone service."""
    try:
        # Create alert monitor with notification service
        monitor = AlertMonitorV2(
            notification_service_url="http://localhost:8080",
            telegram_service=db
        )
        await monitor.start()

    except KeyboardInterrupt:
        _logger.info("Alert monitor V2 stopped by user")
    except Exception as e:
        _logger.exception("Error in alert monitor V2 main: ")


if __name__ == "__main__":
    asyncio.run(main())