#!/usr/bin/env python3
"""
Background services runner for Telegram Screener Bot.
Runs alert monitoring and schedule processing in parallel.
Uses HTTP API to communicate with the bot microservice.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import asyncio
from src.frontend.telegram.screener.alert_monitor import AlertMonitor
from src.frontend.telegram.screener.schedule_processor import ScheduleProcessor
from src.frontend.telegram.screener.http_api_client import BotHttpApiClient

from src.notification.logger import setup_logger, set_logging_context
_logger = setup_logger(__name__)


async def main():
    """Main function to run all background services."""
    try:
        # Set logging context so that logs go to background services log file
        set_logging_context("telegram_background_services")

        # Create HTTP API client for communicating with bot microservice
        api_client = BotHttpApiClient()

        _logger.info("Starting Telegram Screener Background Services...")

        # Test connection to bot API
        _logger.info("Testing connection to bot API...")
        if await api_client.test_connection():
            _logger.info("✅ Bot API connection successful")
        else:
            _logger.warning("⚠️ Bot API connection failed - services will still run but notifications may fail")

        # Create services
        alert_monitor = AlertMonitor(api_client)
        schedule_processor = ScheduleProcessor(api_client)

        # Run services in parallel
        await asyncio.gather(
            alert_monitor.start(),
            schedule_processor.start()
        )

    except KeyboardInterrupt:
        _logger.info("Background services stopped by user")
    except Exception as e:
        _logger.exception("Error in background services: ")


if __name__ == "__main__":
    asyncio.run(main())
