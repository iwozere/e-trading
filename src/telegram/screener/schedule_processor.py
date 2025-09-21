import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import asyncio
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
from src.data.db import telegram_service as db
from src.telegram.screener.business_logic import analyze_ticker_business
from src.common.ticker_analyzer import format_ticker_report
from src.telegram.screener.http_api_client import BotHttpApiClient, send_notification_via_api

from src.notification.logger import setup_logger, set_logging_context
_logger = setup_logger(__name__)


class ScheduleProcessor:
    """
    Background service to process scheduled reports and send them at the configured times.
    Uses HTTP API to communicate with the bot microservice.
    """

    def __init__(self, api_client: BotHttpApiClient = None):
        self.api_client = api_client
        self.running = False

    async def start(self):
        """Start the schedule processing loop."""
        self.running = True

        # Set logging context
        set_logging_context("telegram_schedule_processor")

        _logger.info("Schedule processor started")
        while self.running:
            try:
                await self.process_schedules()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                _logger.exception("Error in schedule processor")
                await asyncio.sleep(30)  # Shorter sleep on error

    def stop(self):
        """Stop the schedule processing loop."""
        self.running = False
        _logger.info("Schedule processor stopped")

    async def process_schedules(self):
        """Check all active schedules and process those that are due."""
        try:
            db.init_db()

            # Get current time in UTC
            current_time = datetime.now(timezone.utc)
            current_time_str = current_time.strftime("%H:%M")
            current_date = current_time.date()

            # Get all active schedules using the new database service
            schedules = db.get_active_schedules()

            if not schedules:
                return

            _logger.debug("Checking %d active schedules at %s", len(schedules), current_time_str)

            for schedule in schedules:
                await self.check_single_schedule(schedule, current_time_str, current_date)

        except Exception as e:
            _logger.exception("Error processing schedules")

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
            _logger.exception("Error checking schedule %s", schedule.get("id"))

    async def execute_schedule(self, schedule: Dict[str, Any]):
        """Execute a scheduled report or screener."""
        try:
            schedule_type = schedule.get("schedule_type", "report")
            schedule_config = schedule.get("schedule_config", "simple")

            if schedule_config == "enhanced_screener":
                await self.execute_enhanced_screener_schedule(schedule)
            elif schedule_config == "report":
                await self.execute_json_report_schedule(schedule)
            elif schedule_type == "screener":
                await self.execute_screener_schedule(schedule)
            else:
                await self.execute_report_schedule(schedule)

        except Exception as e:
            _logger.exception("Error executing schedule %s", schedule.get("id"))
            return

    async def execute_report_schedule(self, schedule: Dict[str, Any]):
        """Execute a scheduled report."""
        try:
            schedule_id = schedule["id"]
            user_id = schedule["user_id"]
            ticker = schedule.get("ticker", "").upper()
            report_type = schedule.get("report_type", "simple")

            _logger.info("Executing report schedule %s for user %s, ticker %s", schedule_id, user_id, ticker)

            # Generate report
            if report_type == "simple":
                result = analyze_ticker_business(ticker, user_id)
            else:
                # For advanced reports, we might need to parse additional config
                result = analyze_ticker_business(ticker, user_id)

            if result["status"] == "ok":
                # Format report for Telegram
                message = format_ticker_report(result["data"])

                # Send via HTTP API
                if self.api_client:
                    await self.api_client.send_message_to_user(user_id, message)
                    _logger.info("Report sent successfully for schedule %s", schedule_id)
                else:
                    _logger.warning("No API client available for schedule %s", schedule_id)

            else:
                _logger.error("Report generation failed for schedule %s: %s", schedule_id, result.get("message", "Unknown error"))

        except Exception as e:
            _logger.exception("Error executing report schedule %s", schedule.get("id"))
            return

    async def execute_enhanced_screener_schedule(self, schedule: Dict[str, Any]):
        """Execute a scheduled enhanced screener with JSON configuration."""
        try:
            schedule_id = schedule["id"]
            user_id = schedule["user_id"]
            config_json = schedule.get("config_json")

            if not config_json:
                _logger.error("No config_json found for enhanced screener schedule %s", schedule_id)
                return

            _logger.info("Executing enhanced screener schedule %s for user %s", schedule_id, user_id)

            # Parse configuration
            from src.telegram.screener.screener_config_parser import parse_screener_config
            try:
                screener_config = parse_screener_config(config_json)
            except Exception as e:
                _logger.error("Failed to parse config for schedule %s: %s", schedule_id, e)
                return

            # Run enhanced screener
            from src.telegram.screener.enhanced_screener import EnhancedScreener
            screener = EnhancedScreener()
            report = await screener.run_enhanced_screener(screener_config)

            if report.error:
                _logger.error("Enhanced screener failed for schedule %s: %s", schedule_id, report.error)
                return

            # Check if email is requested
            email = schedule.get("email", False)

            if email:
                # Send via email
                try:
                    from src.data.db import telegram_service as db
                    user_status = db.get_user_status(user_id)
                    if user_status and user_status.get("email"):
                        from src.telegram.screener.notifications import send_screener_email

                        # Create a mock config for the email function
                        class MockConfig:
                            def __init__(self, list_type):
                                self.list_type = list_type

                        mock_config = MockConfig(screener_config.list_type)
                        send_screener_email(user_status["email"], report, mock_config)
                        _logger.info("Enhanced screener email sent successfully for schedule %s", schedule_id)
                    else:
                        _logger.warning("No email found for user %s in schedule %s", user_id, schedule_id)
                except Exception as e:
                    _logger.exception("Error sending enhanced screener email for schedule %s", schedule_id)

            # Format report for Telegram
            from src.telegram.screener.enhanced_screener import format_enhanced_telegram_message
            message = format_enhanced_telegram_message(report, screener_config)

            # Send via HTTP API
            if self.api_client:
                await self.api_client.send_message_to_user(user_id, message)
                _logger.info("Enhanced screener report sent successfully for schedule %s", schedule_id)
            else:
                _logger.warning("No API client available for schedule %s", schedule_id)

        except Exception as e:
            _logger.exception("Error executing enhanced screener schedule %s", schedule.get("id"))
            return


async def execute_json_report_schedule(self, schedule: Dict[str, Any]):
    """Execute a scheduled report with JSON configuration."""
    try:
        schedule_id = schedule["id"]
        user_id = schedule["user_id"]
        config_json = schedule.get("config_json")

        if not config_json:
            _logger.error("No config_json found for schedule %s", schedule_id)
            return

        _logger.info("Executing JSON report schedule %s for user %s", schedule_id, user_id)

        # Parse JSON configuration
        import json
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as e:
            _logger.error("Invalid JSON config for schedule %s: %s", schedule_id, e)
            return

        # Extract configuration - support both single ticker and multiple tickers
        ticker = config.get("ticker")  # Single ticker
        tickers = config.get("tickers", [])  # Multiple tickers

        # Determine which tickers to process
        if ticker and tickers:
            _logger.error("Both 'ticker' and 'tickers' specified in config for schedule %s", schedule_id)
            return
        elif ticker:
            tickers = [ticker]  # Convert single ticker to list
        elif not tickers:
            _logger.error("No tickers specified in config for schedule %s", schedule_id)
            return

        period = config.get("period", "2y")
        interval = config.get("interval", "1d")
        indicators = config.get("indicators", "")
        provider = config.get("provider", "yf")
        email = config.get("email", False)

        # Generate reports for all tickers
        all_reports = []
        for ticker in tickers:
            try:
                # Create a ParsedCommand-like structure for the report
                from src.telegram.command_parser import ParsedCommand
                parsed = ParsedCommand(
                    command="report",
                    args={
                        "telegram_user_id": user_id,
                        "indicators": indicators,
                        "period": period,
                        "interval": interval,
                        "provider": provider,
                        "email": email
                    },
                    positionals=[ticker]
                )

                # Generate report
                from src.telegram.screener.business_logic import handle_report
                result = handle_report(parsed)

                if result["status"] == "ok" and "data" in result:
                    all_reports.append({
                        "ticker": ticker,
                        "data": result["data"],
                        "message": result.get("message", "")
                    })
                else:
                    _logger.warning("Failed to generate report for %s in schedule %s: %s",
                                  ticker, schedule_id, result.get("message", "Unknown error"))

            except Exception as e:
                _logger.exception("Error generating report for %s in schedule %s", ticker, schedule_id)
                continue

        if not all_reports:
            _logger.error("No reports generated for schedule %s", schedule_id)
            return

        # Format combined report message
        message = f"📊 **Scheduled Report** - {len(all_reports)} tickers\n\n"

        for report in all_reports:
            ticker = report["ticker"]
            data = report["data"]

            # Format individual ticker report
            ticker_message = f"**{ticker}**\n"

            # Add price information
            if "current_price" in data:
                ticker_message += f"💰 Price: ${data['current_price']:.2f}\n"

            # Add technical indicators if available
            if "technicals" in data and data["technicals"]:
                tech = data["technicals"]
                if "rsi" in tech:
                    ticker_message += f"📈 RSI: {tech['rsi']:.2f}\n"
                if "macd" in tech:
                    ticker_message += f"📊 MACD: {tech['macd']:.2f}\n"

            # Add fundamental data if available
            if "fundamentals" in data and data["fundamentals"]:
                fund = data["fundamentals"]
                if "pe_ratio" in fund:
                    ticker_message += f"📊 P/E: {fund['pe_ratio']:.2f}\n"
                if "market_cap" in fund:
                    ticker_message += f"💼 Market Cap: ${fund['market_cap']/1e9:.2f}B\n"

            message += ticker_message + "\n"

        # Add configuration info
        message += f"⏰ **Configuration:**\n"
        message += f"• Period: {period}\n"
        message += f"• Interval: {interval}\n"
        if indicators:
            message += f"• Indicators: {indicators}\n"
        if email:
            message += f"• Email: ✅\n"

        # Send via HTTP API
        if self.api_client:
            await self.api_client.send_message_to_user(user_id, message)
            _logger.info("JSON report sent successfully for schedule %s", schedule_id)
        else:
            _logger.warning("No API client available for schedule %s", schedule_id)

    except Exception as e:
        _logger.exception("Error executing JSON report schedule %s", schedule.get("id"))
        return


async def execute_screener_schedule(self, schedule: Dict[str, Any]):
        """Execute a scheduled screener."""
        try:
            schedule_id = schedule["id"]
            user_id = schedule["user_id"]
            list_type = schedule.get("list_type", "us_medium_cap")
            screener_type = schedule.get("screener_type", "fundamental")

            _logger.info("Executing screener schedule %s for user %s, list %s", schedule_id, user_id, list_type)

            # Import screener based on type
            if screener_type == "fundamental":
                from src.telegram.screener.fundamental_screener import FundamentalScreener
                screener = FundamentalScreener()
                report = screener.run_screener(list_type)
            else:
                _logger.error("Unsupported screener type %s for schedule %s", screener_type, schedule_id)
                return

            if report.error:
                _logger.error("Screener failed for schedule %s: %s", schedule_id, report.error)
                return

            # Check if email is requested
            email = schedule.get("email", False)

            if email:
                # Send via email
                try:
                    from src.data.db import telegram_service as db
                    user_status = db.get_user_status(user_id)
                    if user_status and user_status.get("email"):
                        from src.telegram.screener.notifications import send_screener_email

                        # Create a mock config for the email function
                        class MockConfig:
                            def __init__(self, list_type):
                                self.list_type = list_type

                        mock_config = MockConfig(list_type)
                        send_screener_email(user_status["email"], report, mock_config)
                        _logger.info("Screener email sent successfully for schedule %s", schedule_id)
                    else:
                        _logger.warning("No email found for user %s in schedule %s", user_id, schedule_id)
                except Exception as e:
                    _logger.exception("Error sending screener email for schedule %s", schedule_id)

            # Format screener report for Telegram
            from src.telegram.screener.enhanced_screener import format_enhanced_telegram_message
            message = format_enhanced_telegram_message(report)

            # Send via HTTP API
            if self.api_client:
                await self.api_client.send_message_to_user(user_id, message)
                _logger.info("Screener report sent successfully for schedule %s", schedule_id)
            else:
                _logger.warning("No API client available for schedule %s", schedule_id)

        except Exception as e:
            _logger.exception("Error executing screener schedule %s", schedule.get("id"))
            return


async def main():
    """Main function to run the schedule processor."""
    try:
        # Initialize HTTP API client
        api_client = BotHttpApiClient()

        # Create and start schedule processor
        processor = ScheduleProcessor(api_client)

        _logger.info("Starting schedule processor...")
        await processor.start()

    except Exception as e:
        _logger.exception("Error in schedule processor main")
        return


if __name__ == "__main__":
    asyncio.run(main())
