import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import asyncio
import time
from typing import List, Dict, Any
from src.frontend.telegram import db
from src.common import get_ohlcv
from src.notification.async_notification_manager import initialize_notification_manager
from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SMTP_USER, SMTP_PASSWORD

from src.notification.logger import setup_logger, set_logging_context
logger = setup_logger("telegram_alert_monitor")


class AlertMonitor:
    """
    Background service to monitor price alerts and send notifications when triggered.
    """

    def __init__(self, notification_manager):
        self.notification_manager = notification_manager
        self.running = False

    async def start(self):
        """Start the alert monitoring loop."""
        self.running = True

        # Set logging context for notification manager
        set_logging_context("telegram_alert_monitor")

        logger.info("Alert monitor started")
        while self.running:
            try:
                await self.check_alerts()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.exception("Error in alert monitor: ")
                await asyncio.sleep(30)  # Shorter sleep on error

    def stop(self):
        """Stop the alert monitoring loop."""
        self.running = False
        logger.info("Alert monitor stopped")

    async def check_alerts(self):
        """Check all active alerts and trigger notifications if conditions are met."""
        try:
            db.init_db()

            # Get all active alerts
            conn = db.sqlite3.connect(db.DB_PATH)
            c = conn.cursor()
            c.execute("SELECT * FROM alerts WHERE active=1")
            alerts = c.fetchall()
            conn.close()

            if not alerts:
                return

            logger.debug("Checking %d active alerts", len(alerts))

            for alert_row in alerts:
                alert = dict(zip([d[0] for d in c.description], alert_row))
                await self.check_single_alert(alert)

        except Exception as e:
            logger.exception("Error checking alerts: ")

    async def check_single_alert(self, alert: Dict[str, Any]):
        """Check a single alert and trigger notification if condition is met."""
        try:
            ticker = alert["ticker"]
            target_price = alert["price"]
            condition = alert["condition"]  # 'above' or 'below'
            user_id = alert["user_id"]
            alert_id = alert["id"]

            # Get current price
            try:
                # Determine provider based on ticker length
                provider = "yf" if len(ticker) < 5 else "bnc"
                df = get_ohlcv(ticker, "1m", "1d", provider)

                if df is None or df.empty:
                    logger.warning("No data available for ticker %s", ticker)
                    return

                current_price = df['close'].iloc[-1]

            except Exception as e:
                logger.error("Error getting price for %s: %s", ticker, e)
                return

            # Check if alert condition is met
            triggered = False
            if condition == "above" and current_price > target_price:
                triggered = True
            elif condition == "below" and current_price < target_price:
                triggered = True

            if triggered:
                await self.trigger_alert(alert, current_price)

        except Exception as e:
            logger.exception("Error checking alert %s: ", alert.get("id"))

    async def trigger_alert(self, alert: Dict[str, Any], current_price: float):
        """Trigger an alert notification and deactivate the alert."""
        try:
            ticker = alert["ticker"]
            target_price = alert["price"]
            condition = alert["condition"]
            user_id = alert["user_id"]
            alert_id = alert["id"]

            # Get user info for email notification
            user_status = db.get_user_status(user_id)
            user_email = user_status.get("email") if user_status and user_status.get("verified") else None

            # Prepare notification message
            message = (
                f"🚨 Price Alert Triggered!\n\n"
                f"Ticker: {ticker}\n"
                f"Current Price: ${current_price:.2f}\n"
                f"Alert: {condition} ${target_price:.2f}\n"
                f"Alert ID: #{alert_id}"
            )

            # Send Telegram notification
            await self.notification_manager.send_notification(
                notification_type="INFO",
                title=f"Price Alert: {ticker} {condition} ${target_price:.2f}",
                message=message,
                priority="HIGH",
                channels=["telegram"],
                telegram_chat_id=int(user_id)
            )

            # Send email notification if user has verified email
            if user_email:
                email_message = (
                    f"Hello,\n\n"
                    f"Your price alert for {ticker} has been triggered:\n"
                    f"Current price is ${current_price:.2f}, which is {condition} your threshold of ${target_price:.2f}.\n\n"
                    f"Set via Alkotrader Telegram bot.\n\n"
                    f"Best regards,\n"
                    f"Alkotrader Team"
                )

                await self.notification_manager.send_notification(
                    notification_type="INFO",
                    title=f"Alkotrader Price Alert: {ticker} {condition} ${target_price:.2f}",
                    message=email_message,
                    priority="HIGH",
                    channels=["email"],
                    email_receiver=user_email
                )

            # Deactivate the alert (one-time trigger)
            db.update_alert(alert_id, active=False)

            logger.info("Alert #%d triggered for user %s: %s %s $%.2f (current: $%.2f)",
                       alert_id, user_id, ticker, condition, target_price, current_price)

        except Exception as e:
            logger.exception("Error triggering alert %s: ", alert.get("id"))


async def main():
    """Main function to run the alert monitor as a standalone service."""
    try:
        # Initialize notification manager
        notification_manager = await initialize_notification_manager(
            telegram_token=TELEGRAM_BOT_TOKEN,
            telegram_chat_id=TELEGRAM_CHAT_ID,
            email_api_key=SMTP_PASSWORD,
            email_sender=SMTP_USER,
            email_receiver=SMTP_USER  # Default receiver
        )

        # Create and start alert monitor
        monitor = AlertMonitor(notification_manager)
        await monitor.start()

    except KeyboardInterrupt:
        logger.info("Alert monitor stopped by user")
    except Exception as e:
        logger.exception("Error in alert monitor main: ")


if __name__ == "__main__":
    asyncio.run(main())
