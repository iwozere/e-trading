#!/usr/bin/env python3
"""
Background services runner for Telegram Screener Bot.
Runs alert monitoring and schedule processing in parallel.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import asyncio
from src.frontend.telegram.screener.alert_monitor import AlertMonitor
from src.frontend.telegram.screener.schedule_processor import ScheduleProcessor
from src.notification.async_notification_manager import initialize_notification_manager
from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SMTP_USER, SMTP_PASSWORD

from src.notification.logger import setup_logger, set_logging_context
logger = setup_logger("telegram_background_services")


async def main():
    """Main function to run all background services."""
    try:
        # Set logging context so that notification manager logs go to background services log file
        set_logging_context("telegram_background_services")

        # Initialize notification manager
        notification_manager = await initialize_notification_manager(
            telegram_token=TELEGRAM_BOT_TOKEN,
            telegram_chat_id=TELEGRAM_CHAT_ID,
            email_api_key=SMTP_PASSWORD,
            email_sender=SMTP_USER,
            email_receiver=SMTP_USER  # Default receiver
        )

        logger.info("Starting Telegram Screener Background Services...")

        # Create services
        alert_monitor = AlertMonitor(notification_manager)
        schedule_processor = ScheduleProcessor(notification_manager)

        # Run services in parallel
        await asyncio.gather(
            alert_monitor.start(),
            schedule_processor.start()
        )

    except KeyboardInterrupt:
        logger.info("Background services stopped by user")
    except Exception as e:
        logger.exception("Error in background services: ")


if __name__ == "__main__":
    asyncio.run(main())
