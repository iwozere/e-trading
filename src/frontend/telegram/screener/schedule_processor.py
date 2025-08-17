import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import asyncio
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
from src.frontend.telegram import db
from src.frontend.telegram.screener.business_logic import analyze_ticker_business
from src.common.ticker_analyzer import format_ticker_report
from src.notification.async_notification_manager import initialize_notification_manager
from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SMTP_USER, SMTP_PASSWORD

from src.notification.logger import setup_logger, set_logging_context
logger = setup_logger("telegram_schedule_processor")


class ScheduleProcessor:
    """
    Background service to process scheduled reports and send them at the configured times.
    """

    def __init__(self, notification_manager):
        self.notification_manager = notification_manager
        self.running = False

    async def start(self):
        """Start the schedule processing loop."""
        self.running = True

        # Set logging context for notification manager
        set_logging_context("telegram_schedule_processor")

        logger.info("Schedule processor started")
        while self.running:
            try:
                await self.process_schedules()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.exception("Error in schedule processor: ")
                await asyncio.sleep(30)  # Shorter sleep on error

    def stop(self):
        """Stop the schedule processing loop."""
        self.running = False
        logger.info("Schedule processor stopped")

    async def process_schedules(self):
        """Check all active schedules and process those that are due."""
        try:
            db.init_db()

            # Get current time in UTC
            current_time = datetime.now(timezone.utc)
            current_time_str = current_time.strftime("%H:%M")
            current_date = current_time.date()

            # Get all active schedules
            conn = db.sqlite3.connect(db.DB_PATH)
            c = conn.cursor()
            c.execute("SELECT * FROM schedules WHERE active=1")
            schedules = c.fetchall()
            conn.close()

            if not schedules:
                return

            logger.debug("Checking %d active schedules at %s", len(schedules), current_time_str)

            for schedule_row in schedules:
                schedule = dict(zip([d[0] for d in c.description], schedule_row))
                await self.check_single_schedule(schedule, current_time_str, current_date)

        except Exception as e:
            logger.exception("Error processing schedules: ")

    async def check_single_schedule(self, schedule: Dict[str, Any], current_time_str: str, current_date):
        """Check if a schedule should be triggered and process it."""
        try:
            scheduled_time = schedule["scheduled_time"]
            period = schedule.get("period", "daily")
            schedule_id = schedule["id"]

            # Check if it's time to run this schedule
            if scheduled_time != current_time_str:
                return

            # Check if we've already processed this schedule today
            # (to avoid duplicate runs within the same minute)
            last_run_key = f"schedule_{schedule_id}_last_run"
            last_run_date = db.get_setting(last_run_key)

            if last_run_date == str(current_date):
                return  # Already processed today

            # Check period constraints
            should_run = False
            if period == "daily":
                should_run = True
            elif period == "weekly":
                # Run on Mondays (weekday 0)
                should_run = current_date.weekday() == 0
            elif period == "monthly":
                # Run on the 1st of the month
                should_run = current_date.day == 1
            else:
                # Default to daily
                should_run = True

            if should_run:
                await self.execute_schedule(schedule)
                # Mark as processed for today
                db.set_setting(last_run_key, str(current_date))

        except Exception as e:
            logger.exception("Error checking schedule %s: ", schedule.get("id"))

    async def execute_schedule(self, schedule: Dict[str, Any]):
        """Execute a scheduled report."""
        try:
            ticker = schedule["ticker"]
            user_id = schedule["user_id"]
            schedule_id = schedule["id"]
            send_email = schedule.get("email", 0)
            indicators = schedule.get("indicators")
            interval = schedule.get("interval", "1d")
            provider = schedule.get("provider")
            period = schedule.get("period", "daily")

            # Get user info
            user_status = db.get_user_status(user_id)
            if not user_status:
                logger.warning("User %s not found for schedule %d", user_id, schedule_id)
                return

            user_email = user_status.get("email") if user_status.get("verified") else None

            # Generate the report
            analysis = analyze_ticker_business(
                ticker=ticker,
                provider=provider,
                period="2y",  # Default period for scheduled reports
                interval=interval
            )

            report = format_ticker_report(analysis)

            if analysis.error:
                # Send error notification
                error_message = f"❌ Scheduled Report Error\n\nSchedule #{schedule_id}\nTicker: {ticker}\nError: {analysis.error}"

                await self.notification_manager.send_notification(
                    notification_type="ERROR",
                    title=f"Scheduled Report Error for {ticker}",
                    message=error_message,
                    priority="NORMAL",
                    channels=["telegram"],
                    telegram_chat_id=int(user_id)
                )
                return

            # Prepare success message
            scheduled_message = f"📊 Scheduled Report ({period})\n\n{report['message']}"

            # Send Telegram notification
            attachments = None
            if report.get("chart_bytes"):
                attachments = {f"{ticker}_chart.png": report["chart_bytes"]}

            await self.notification_manager.send_notification(
                notification_type="INFO",
                title=f"Scheduled Report for {ticker}",
                message=scheduled_message,
                attachments=attachments,
                priority="NORMAL",
                channels=["telegram"],
                telegram_chat_id=int(user_id)
            )

            # Send email notification if requested and user has verified email
            if send_email and user_email:
                await self.notification_manager.send_notification(
                    notification_type="INFO",
                    title=f"Alkotrader Scheduled Report for {ticker}",
                    message=report["message"],
                    attachments=attachments,
                    priority="NORMAL",
                    channels=["email"],
                    email_receiver=user_email
                )

            logger.info("Executed schedule #%d for user %s: %s (%s)",
                       schedule_id, user_id, ticker, period)

        except Exception as e:
            logger.exception("Error executing schedule %s: ", schedule.get("id"))


async def main():
    """Main function to run the schedule processor as a standalone service."""
    try:
        # Initialize notification manager
        notification_manager = await initialize_notification_manager(
            telegram_token=TELEGRAM_BOT_TOKEN,
            telegram_chat_id=TELEGRAM_CHAT_ID,
            email_api_key=SMTP_PASSWORD,
            email_sender=SMTP_USER,
            email_receiver=SMTP_USER  # Default receiver
        )

        # Create and start schedule processor
        processor = ScheduleProcessor(notification_manager)
        await processor.start()

    except KeyboardInterrupt:
        logger.info("Schedule processor stopped by user")
    except Exception as e:
        logger.exception("Error in schedule processor main: ")


if __name__ == "__main__":
    asyncio.run(main())
