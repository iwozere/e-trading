#!/usr/bin/env python3
"""
Telegram Service Layer Integration Example

This example demonstrates how to properly use the service layer
in the telegram bot context, showing the correct patterns for
database operations and indicator calculations.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone

# Service layer imports
from src.data.db.services import telegram_service
from src.indicators.service import IndicatorService
from src.indicators.models import TickerIndicatorsRequest, IndicatorResultSet

# Business logic imports
from src.telegram.command_parser import ParsedCommand
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TelegramServiceExample:
    """
    Example class showing proper service layer integration patterns.
    """

    def __init__(self, telegram_service_instance=None, indicator_service_instance=None):
        """
        Initialize with service dependencies using dependency injection.

        Args:
            telegram_service_instance: Telegram service for database operations
            indicator_service_instance: Indicator service for calculations
        """
        self.telegram_service = telegram_service_instance or telegram_service
        self.indicator_service = indicator_service_instance or IndicatorService()

    def safe_telegram_service_call(self, method, *args, **kwargs):
        """
        Safely call telegram service methods with error handling.

        Args:
            method: Service method to call
            *args: Method arguments
            **kwargs: Method keyword arguments

        Returns:
            Method result or None if error occurred
        """
        try:
            return method(*args, **kwargs)
        except Exception:
            _logger.exception("Telegram service error:")
            return None

    async def safe_indicator_calculation(self, request: TickerIndicatorsRequest) -> Optional[IndicatorResultSet]:
        """
        Safely calculate indicators with fallback behavior.

        Args:
            request: Indicator calculation request

        Returns:
            Indicator results or empty result if error occurred
        """
        try:
            return await self.indicator_service.compute_for_ticker(request)
        except Exception as e:
            _logger.warning("Indicator calculation failed for %s: %s", request.ticker, e)
            # Return empty result as fallback
            return IndicatorResultSet(
                ticker=request.ticker,
                technical={},
                fundamental={}
            )

    def example_user_management(self, telegram_user_id: str) -> Dict[str, Any]:
        """
        Example of proper user management through service layer.

        Args:
            telegram_user_id: Telegram user ID

        Returns:
            User management result
        """
        try:
            # Get user status through service layer
            user_status = self.safe_telegram_service_call(
                self.telegram_service.get_user_status,
                telegram_user_id
            )

            if not user_status:
                return {"status": "error", "message": "User not found"}

            # Check if user is approved
            if not user_status.get("approved"):
                return {"status": "error", "message": "User not approved"}

            # Update user limits through service layer
            self.safe_telegram_service_call(
                self.telegram_service.set_user_limit,
                telegram_user_id,
                "max_alerts",
                10
            )

            # Log user activity
            self.safe_telegram_service_call(
                self.telegram_service.log_command_audit,
                telegram_user_id,
                "user_management_example",
                status="success"
            )

            return {
                "status": "success",
                "user_status": user_status,
                "message": "User management completed"
            }

        except Exception:
            _logger.exception("User management failed:")
            return {"status": "error", "message": "User management failed"}

    def example_alert_management(self, telegram_user_id: str, ticker: str, price: float, condition: str) -> Dict[str, Any]:
        """
        Example of proper alert management through service layer.

        Args:
            telegram_user_id: Telegram user ID
            ticker: Ticker symbol
            price: Alert price threshold
            condition: Alert condition (above/below)

        Returns:
            Alert management result
        """
        try:
            # Create alert through service layer
            alert_id = self.safe_telegram_service_call(
                self.telegram_service.add_alert,
                telegram_user_id=telegram_user_id,
                ticker=ticker,
                price=price,
                condition=condition,
                active=True
            )

            if not alert_id:
                return {"status": "error", "message": "Failed to create alert"}

            # List user alerts
            alerts = self.safe_telegram_service_call(
                self.telegram_service.list_alerts,
                telegram_user_id
            )

            # Log alert creation
            self.safe_telegram_service_call(
                self.telegram_service.log_command_audit,
                telegram_user_id,
                "alert_create",
                ticker=ticker,
                price=price,
                condition=condition,
                alert_id=alert_id,
                status="success"
            )

            return {
                "status": "success",
                "alert_id": alert_id,
                "total_alerts": len(alerts) if alerts else 0,
                "message": f"Alert created for {ticker} {condition} {price}"
            }

        except Exception:
            _logger.exception("Alert management failed:")
            return {"status": "error", "message": "Alert management failed"}

    async def example_indicator_calculation(self, ticker: str, indicators: list) -> Dict[str, Any]:
        """
        Example of proper indicator calculation through service layer.

        Args:
            ticker: Ticker symbol
            indicators: List of indicators to calculate

        Returns:
            Indicator calculation result
        """
        try:
            # Create indicator request
            request = TickerIndicatorsRequest(
                ticker=ticker,
                indicators=indicators,
                timeframe="1d",
                period="1y"
            )

            # Calculate indicators through service layer
            result = await self.safe_indicator_calculation(request)

            if not result:
                return {"status": "error", "message": "Indicator calculation failed"}

            # Process results
            technical_data = {}
            fundamental_data = {}

            for indicator in indicators:
                if indicator in result.technical:
                    technical_data[indicator] = result.technical[indicator]
                if indicator in result.fundamental:
                    fundamental_data[indicator] = result.fundamental[indicator]

            return {
                "status": "success",
                "ticker": ticker,
                "technical": technical_data,
                "fundamental": fundamental_data,
                "message": f"Calculated {len(technical_data + fundamental_data)} indicators for {ticker}"
            }

        except Exception:
            _logger.exception("Indicator calculation failed:")
            return {"status": "error", "message": "Indicator calculation failed"}

    async def example_report_generation(self, parsed_command: ParsedCommand) -> Dict[str, Any]:
        """
        Example of complete report generation using service layer integration.

        Args:
            parsed_command: Parsed command from user

        Returns:
            Report generation result
        """
        try:
            telegram_user_id = parsed_command.telegram_user_id
            ticker = parsed_command.args.get("tickers", "AAPL")
            indicators = parsed_command.args.get("indicators", ["RSI", "MACD"])

            # Check user authorization through service layer
            user_status = self.safe_telegram_service_call(
                self.telegram_service.get_user_status,
                telegram_user_id
            )

            if not user_status or not user_status.get("approved"):
                return {"status": "error", "message": "Access denied"}

            # Calculate indicators through service layer
            indicator_result = await self.example_indicator_calculation(ticker, indicators)

            if indicator_result["status"] != "success":
                return indicator_result

            # Log command execution
            self.safe_telegram_service_call(
                self.telegram_service.log_command_audit,
                telegram_user_id,
                "report",
                ticker=ticker,
                indicators=indicators,
                status="success"
            )

            # Generate report (simplified)
            report_data = {
                "ticker": ticker,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "technical_indicators": indicator_result["technical"],
                "fundamental_indicators": indicator_result["fundamental"],
                "user_id": telegram_user_id
            }

            return {
                "status": "success",
                "report": report_data,
                "message": f"Report generated for {ticker}"
            }

        except Exception:
            _logger.exception("Report generation failed:")
            return {"status": "error", "message": "Report generation failed"}

    def example_schedule_management(self, telegram_user_id: str, ticker: str, scheduled_time: str) -> Dict[str, Any]:
        """
        Example of proper schedule management through service layer.

        Args:
            telegram_user_id: Telegram user ID
            ticker: Ticker symbol
            scheduled_time: Schedule time (HH:MM format)

        Returns:
            Schedule management result
        """
        try:
            # Create schedule configuration
            config = {
                "type": "report",
                "ticker": ticker,
                "indicators": ["RSI", "MACD"],
                "period": "1y",
                "interval": "1d",
                "email": True
            }

            # Create schedule through service layer
            schedule_id = self.safe_telegram_service_call(
                self.telegram_service.add_schedule,
                telegram_user_id=telegram_user_id,
                ticker=ticker,
                scheduled_time=scheduled_time,
                period="daily",
                config=config,
                active=True
            )

            if not schedule_id:
                return {"status": "error", "message": "Failed to create schedule"}

            # List user schedules
            schedules = self.safe_telegram_service_call(
                self.telegram_service.list_schedules,
                telegram_user_id
            )

            # Log schedule creation
            self.safe_telegram_service_call(
                self.telegram_service.log_command_audit,
                telegram_user_id,
                "schedule_create",
                ticker=ticker,
                scheduled_time=scheduled_time,
                schedule_id=schedule_id,
                status="success"
            )

            return {
                "status": "success",
                "schedule_id": schedule_id,
                "total_schedules": len(schedules) if schedules else 0,
                "message": f"Schedule created for {ticker} at {scheduled_time}"
            }

        except Exception:
            _logger.exception("Schedule management failed:")
            return {"status": "error", "message": "Schedule management failed"}


async def main():
    """
    Main function demonstrating service layer integration examples.
    """
    print("🚀 Telegram Service Layer Integration Examples")
    print("=" * 50)

    # Initialize example class with service dependencies
    example = TelegramServiceExample()

    # Example 1: User Management
    print("\n📋 Example 1: User Management")
    user_result = example.example_user_management("123456789")
    print(f"Result: {user_result}")

    # Example 2: Alert Management
    print("\n🚨 Example 2: Alert Management")
    alert_result = example.example_alert_management("123456789", "AAPL", 150.0, "above")
    print(f"Result: {alert_result}")

    # Example 3: Indicator Calculation
    print("\n📊 Example 3: Indicator Calculation")
    indicator_result = await example.example_indicator_calculation("AAPL", ["RSI", "MACD", "PE"])
    print(f"Result: {indicator_result}")

    # Example 4: Report Generation
    print("\n📈 Example 4: Report Generation")
    # Create mock parsed command
    parsed_command = ParsedCommand(
        telegram_user_id="123456789",
        command="report",
        args={"tickers": "AAPL", "indicators": ["RSI", "MACD"]},
        positionals=[],
        extra_flags={}
    )
    report_result = await example.example_report_generation(parsed_command)
    print(f"Result: {report_result}")

    # Example 5: Schedule Management
    print("\n⏰ Example 5: Schedule Management")
    schedule_result = example.example_schedule_management("123456789", "AAPL", "09:00")
    print(f"Result: {schedule_result}")

    print("\n✅ All examples completed!")
    print("\nKey Takeaways:")
    print("- Always use dependency injection for service instances")
    print("- Wrap service calls in error handling")
    print("- Use async/await for indicator service calls")
    print("- Log all operations for audit trail")
    print("- Provide user-friendly error messages")


if __name__ == "__main__":
    asyncio.run(main())