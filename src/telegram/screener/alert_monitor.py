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
from src.telegram.screener.http_api_client import BotHttpApiClient, send_notification_via_api
from src.telegram.screener.alert_logic_evaluator import AlertLogicEvaluator, evaluate_alert
from src.telegram.screener.rearm_alert_system import ReArmAlertEvaluator

from src.notification.logger import setup_logger, set_logging_context
_logger = setup_logger(__name__)


class AlertMonitor:
    """
    Background service to monitor price alerts and send notifications when triggered.
    Uses HTTP API to communicate with the bot microservice.
    """

    def __init__(self, api_client: BotHttpApiClient = None, telegram_service=None):
        self.api_client = api_client
        self.running = False
        self.evaluator = AlertLogicEvaluator()
        # Inject telegram_service into ReArmAlertEvaluator for service layer usage
        self.telegram_service = telegram_service or db
        self.rearm_evaluator = ReArmAlertEvaluator(telegram_service=self.telegram_service)

    async def start(self):
        """Start the alert monitoring loop."""
        self.running = True

        # Set logging context
        set_logging_context("telegram_alert_monitor")

        _logger.info("Alert monitor started")
        while self.running:
            try:
                await self.check_alerts()
                await asyncio.sleep(5 * 60)  # Check every 5 minutes
            except Exception as e:
                _logger.exception("Error in alert monitor: ")
                await asyncio.sleep(5 * 60)  # Shorter sleep on error (5 minutes)

    def stop(self):
        """Stop the alert monitoring loop."""
        self.running = False
        _logger.info("Alert monitor stopped")

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

                # If this alert has failed multiple times, consider notifying admin
                # For now, just skip and continue
                return

            # Check if this is a re-arm alert (has re_arm_config)
            if alert.get("re_arm_config"):
                # Use re-arm evaluator
                triggered, evaluation_details = self.rearm_evaluator.evaluate_alert(alert, current_price)

                # Update alert state using service layer (ReArmAlertEvaluator now handles this internally)
                success = self.rearm_evaluator.update_alert_state(alert_id, evaluation_details, current_price)
                if not success:
                    self._logger.warning("Failed to update alert state for alert %d", alert_id)

                if triggered:
                    await self.trigger_rearm_alert(alert, evaluation_details)

            else:
                # Use legacy evaluator for indicator alerts or old price alerts
                if alert_type == "indicator":
                    triggered, evaluation_details = self.evaluator.evaluate_alert(alert)
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

            # Get user info for email notification using service layer
            _logger.debug("Retrieving user status for alert notification: user_id=%s, alert_id=%s", user_id, alert_id)
            user_status = self.telegram_service.get_user_status(str(user_id))
            user_email = user_status.get("email") if user_status and user_status.get("verified") else None
            _logger.debug("User status retrieved for alert: user_id=%s, verified=%s, has_email=%s",
                         user_id, user_status.get("verified") if user_status else False, bool(user_email))

            # Prepare notification message based on alert type
            if alert_type == "price":
                message = self._format_price_alert_message(alert, evaluation_details)
                title = f"Price Alert: {ticker}"
            else:
                message = self._format_indicator_alert_message(alert, evaluation_details)
                title = f"Indicator Alert: {ticker} - {alert_action}"

            # Send Telegram notification via HTTP API
            success = await send_notification_via_api(
                user_id=user_id,
                message=message,
                title=title
            )

            if success:
                _logger.info("Alert #%d triggered for user %s: %s", alert_id, user_id, ticker)
            else:
                _logger.error("Failed to send alert notification for user %s, alert #%d", user_id, alert_id)

            # Send email notification if user has verified email
            if user_email:
                await self._send_email_alert(user_email, alert, evaluation_details, title)

            # Deactivate the alert (one-time trigger) using service layer with error handling
            try:
                self.telegram_service.update_alert(alert_id, active=False)
            except Exception as e:
                _logger.error("Failed to deactivate alert %d: %s", alert_id, e)

        except Exception as e:
            _logger.exception("Error triggering alert %s: ", alert.get("id"))

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

    async def _send_email_alert(self, user_email: str, alert: Dict[str, Any], details: Dict[str, Any], title: str):
        """Send email alert notification."""
        try:
            alert_type = alert.get("alert_type", "price")

            if alert_type == "price":
                email_message = self._format_price_email_message(alert, details)
            else:
                email_message = self._format_indicator_email_message(alert, details)

            # Use notification manager for email only
            from src.notification.async_notification_manager import initialize_notification_manager
            from config.donotshare.donotshare import SMTP_USER, SMTP_PASSWORD

            notification_manager = await initialize_notification_manager(
                telegram_token=None,  # Not needed for email-only
                telegram_chat_id=None,  # Not needed for email-only
                email_api_key=SMTP_PASSWORD,
                email_sender=SMTP_USER,
                email_receiver=SMTP_USER
            )

            await notification_manager.send_notification(
                notification_type="INFO",
                title=title,
                message=email_message,
                priority="HIGH",
                channels=["email"],
                email_receiver=user_email
            )

            _logger.info("Email alert sent to %s for alert #%d", user_email, alert["id"])
        except Exception as e:
            _logger.error("Failed to send email alert to %s: %s", user_email, e)

    def _format_price_email_message(self, alert: Dict[str, Any], details: Dict[str, Any]) -> str:
        """Format price alert email message."""
        ticker = alert["ticker"]
        target_price = alert["price"]
        condition = alert["condition"]
        current_price = details.get("current_price", 0)

        return (
            f"Hello,\n\n"
            f"Your price alert for {ticker} has been triggered:\n"
            f"Current price is ${current_price:.2f}, which is {condition} your threshold of ${target_price:.2f}.\n\n"
            f"Set via Alkotrader Telegram bot.\n\n"
            f"Best regards,\n"
            f"Alkotrader Team"
        )

    def _format_indicator_email_message(self, alert: Dict[str, Any], details: Dict[str, Any]) -> str:
        """Format indicator alert email message."""
        ticker = alert["ticker"]
        alert_action = alert.get("alert_action", "notify")
        current_price = details.get("current_price", 0)
        timeframe = alert.get("timeframe", "15m")
        summary = self.evaluator.get_alert_summary(alert)

        message = (
            f"Hello,\n\n"
            f"Your indicator alert for {ticker} has been triggered:\n"
            f"Current price: ${current_price:.2f}\n"
            f"Timeframe: {timeframe}\n"
            f"Action: {alert_action}\n\n"
        )

        if "indicators" in summary:
            message += f"Indicators: {', '.join(summary['indicators'])}\n"
            if "logic" in summary and summary["logic"] != "single":
                message += f"Logic: {summary['logic']}\n"

        message += "\nSet via Alkotrader Telegram bot.\n\nBest regards,\nAlkotrader Team"
        return message

    async def _notify_admin_of_error(self, alert: Dict[str, Any], error_message: str):
        """Send error notification to admin(s)."""
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

            # Send to all admins
            for admin_id in admin_users:
                try:
                    await send_notification_via_api(
                        user_id=admin_id,
                        message=error_message_text,
                        title="Alert System Error"
                    )
                except Exception as e:
                    _logger.error("Failed to send error notification to admin %s: %s", admin_id, e)

            _logger.info("Error notification sent to %d admin(s)", len(admin_users))

        except Exception as e:
            _logger.error("Failed to send admin error notification: %s", e)

    async def _get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for ticker with fallback providers."""
        providers_to_try = ["yf", "alpha_vantage", "polygon"]

        for provider in providers_to_try:
            try:
                # Try different intervals and periods for better success rate
                # Start with 5m interval as requested, fallback to others if needed
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

            # Get user info for email notification using service layer
            _logger.debug("Retrieving user status for rearm alert notification: user_id=%s, alert_id=%s", user_id, alert_id)
            user_status = self.telegram_service.get_user_status(str(user_id))
            user_email = user_status.get("email") if user_status and user_status.get("verified") else None
            _logger.debug("User status retrieved for rearm alert: user_id=%s, verified=%s, has_email=%s",
                         user_id, user_status.get("verified") if user_status else False, bool(user_email))

            # Send Telegram notification
            success = await send_notification_via_api(
                user_id=user_id,
                message=message,
                title=title
            )

            if success:
                _logger.info("Re-arm alert #%d triggered for user %s: %s", alert_id, user_id, ticker)
            else:
                _logger.error("Failed to send re-arm alert notification for user %s, alert #%d", user_id, alert_id)

            # Send email notification if configured and user has verified email
            if "email" in config.notification_config.channels and user_email:
                await self._send_email_alert(user_email, alert, evaluation_details, title)

        except Exception as e:
            _logger.exception("Error triggering re-arm alert %s: ", alert.get("id"))


async def main():
    """Main function to run the alert monitor as a standalone service."""
    try:
        # Create HTTP API client
        api_client = BotHttpApiClient()

        # Create and start alert monitor with service layer dependency
        monitor = AlertMonitor(api_client, telegram_service=db)
        await monitor.start()

    except KeyboardInterrupt:
        _logger.info("Alert monitor stopped by user")
    except Exception as e:
        _logger.exception("Error in alert monitor main: ")


if __name__ == "__main__":
    asyncio.run(main())
