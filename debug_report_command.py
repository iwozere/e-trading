#!/usr/bin/env python3
"""
Debug script to test the report command with multiple tickers.
This will help identify why only AAPL data is being returned.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from src.frontend.telegram.command_parser import parse_command
from src.frontend.telegram.screener.business_logic import handle_command
from src.notification.logger import setup_logger

logger = setup_logger("debug_report")

def test_report_command():
    """Test the report command with multiple tickers."""

    # Test command
    test_command = "/report AAPL MSFT VT"
    logger.info("Testing command: %s", test_command)

    # Parse the command
    parsed = parse_command(test_command)
    logger.info("Parsed command: %s", parsed)
    logger.info("Parsed args: %s", parsed.args)
    logger.info("Parsed positionals: %s", parsed.positionals)

    # Add telegram_user_id (required)
    parsed.args["telegram_user_id"] = "test_user_123"

    # Handle the command
    logger.info("Handling command...")
    result = handle_command(parsed)

    logger.info("Result status: %s", result.get("status"))
    logger.info("Result title: %s", result.get("title"))
    logger.info("Result message: %s", result.get("message"))

    if "reports" in result:
        logger.info("Number of reports: %d", len(result["reports"]))
        for i, report in enumerate(result["reports"]):
            logger.info("Report %d:", i + 1)
            logger.info("  Ticker: %s", report.get("ticker"))
            logger.info("  Error: %s", report.get("error"))
            logger.info("  Message length: %d", len(report.get("message", "")))
            logger.info("  Has chart: %s", bool(report.get("chart_bytes")))
    else:
        logger.info("No reports in result")

if __name__ == "__main__":
    test_report_command()
