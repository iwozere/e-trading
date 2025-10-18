#!/usr/bin/env python3
"""
Test Service Layer Mocks
Comprehensive tests demonstrating the use of service layer mocks for telegram business logic testing.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pytest
import asyncio
from tests.fixtures.service_fixtures import (
    business_logic_with_mocks, telegram_service_mock, indicator_service_mock,
    sample_user_data, admin_user_data, unverified_user_data,
    setup_user_in_mock, setup_indicator_data_in_mock, create_parsed_command,
    assert_service_called, assert_service_call_count, get_last_service_call,
    simulate_database_error, simulate_rate_limit_error, simulate_indicator_api_error,
    ServiceTestContext
)
from src.telegram.command_parser import ParsedCommand
from src.indicators.models import IndicatorValue


class TestServiceLayerMocks:
    """Test service layer mocks functionality and integration."""

    def test_telegram_service_mock_basic_operations(self, telegram_service_mock):
        """Test basic telegram service mock operations."""
        # Test user creation and retrieval
        user_data = {"approved": True, "verified": True, "email": "test@example.com"}
        telegram_service_mock.add_test_user("test_user", **user_data)

        status = telegram_service_mock.get_user_status("test_user")
        assert status["approved"] is True
        assert status["verified"] is True
        assert status["email"] == "test@example.com"

        # Test call tracking
        assert telegram_service_mock.get_call_count("get_user_status") == 1
        assert telegram_service_mock.was_called_with("get_user_status", "test_user")

    def test_telegram_service_mock_error_simulation(self, telegram_service_mock):
        """Test error simulation in telegram service mock."""
        # Configure error
        telegram_service_mock.configure_error("get_user_status", ConnectionError("Database down"))

        # Test error is raised
        with pytest.raises(ConnectionError, match="Database down"):
            telegram_service_mock.get_user_status("test_user")

        # Clear error and test normal operation
        telegram_service_mock.clear_errors()
        result = telegram_service_mock.get_user_status("test_user")
        assert result is None  # No user configured

    @pytest.mark.asyncio
    async def test_indicator_service_mock_basic_operations(self, indicator_service_mock):
        """Test basic indicator service mock operations."""
        # Setup indicator data
        setup_indicator_data_in_mock(
            indicator_service_mock,
            "AAPL",
            technical_indicators={"RSI": 65.5, "MACD": 1.23},
            fundamental_indicators={"PE": 25.4}
        )

        # Create request
        from src.indicators.models import TickerIndicatorsRequest
        request = TickerIndicatorsRequest(
            ticker="AAPL",
            indicators=["RSI", "MACD", "PE"]
        )

        # Test computation
        result = await indicator_service_mock.compute_for_ticker(request)

        assert result.ticker == "AAPL"
        assert "RSI" in result.technical
        assert "PE" in result.fundamental
        assert result.technical["RSI"].value == 65.5

        # Test call tracking
        assert indicator_service_mock.get_call_count("compute_for_ticker") == 1

    @pytest.mark.asyncio
    async def test_indicator_service_mock_error_simulation(self, indicator_service_mock):
        """Test error simulation in indicator service mock."""
        # Configure error
        indicator_service_mock.configure_error("compute_for_ticker", ValueError("Invalid ticker"))

        from src.indicators.models import TickerIndicatorsRequest
        request = TickerIndicatorsRequest(ticker="INVALID", indicators=["RSI"])

        # Test error is raised
        with pytest.raises(ValueError, match="Invalid ticker"):
            await indicator_service_mock.compute_for_ticker(request)

    @pytest.mark.asyncio
    async def test_business_logic_with_mocked_services(self, business_logic_with_mocks):
        """Test business logic with both services mocked."""
        business_logic, telegram_service_mock, indicator_service_mock = business_logic_with_mocks

        # Setup test data using direct mock methods
        telegram_service_mock.add_test_user("test_user", approved=True, verified=True)
        setup_indicator_data_in_mock(
            indicator_service_mock,
            "AAPL",
            technical_indicators={"RSI": 65.5}
        )

        # Test indicator calculation through business logic
        result = await business_logic.calculate_indicators_for_ticker(
            ticker="AAPL",
            indicators=["RSI"]
        )

        assert result["status"] == "ok"
        assert result["ticker"] == "AAPL"
        assert "RSI" in result["technical"]

        # Verify service calls were made through business logic
        assert indicator_service_mock.get_call_count("compute_for_ticker") == 1

        # Test business logic service health check
        health = business_logic.get_service_health_status()
        assert health["telegram_service"]["available"] is True
        assert health["indicator_service"]["available"] is True

    def test_business_logic_user_status_checks(self, business_logic_with_mocks, sample_user_data):
        """Test user status checks with mocked telegram service."""
        business_logic, telegram_service_mock, _ = business_logic_with_mocks

        # Setup admin user
        admin_data = sample_user_data.copy()
        admin_data["is_admin"] = True
        telegram_service_mock.add_test_user("admin_user", **admin_data)

        # Setup regular user
        telegram_service_mock.add_test_user("regular_user", **sample_user_data)

        # Test business logic admin check methods
        assert business_logic.is_admin_user("admin_user") is True
        assert business_logic.is_admin_user("regular_user") is False

        # Test business logic approval check methods
        assert business_logic.is_approved_user("admin_user") is True
        assert business_logic.is_approved_user("regular_user") is True

        # Test business logic access control methods
        admin_access = business_logic.check_admin_access("admin_user")
        assert admin_access["status"] == "ok"

        regular_admin_access = business_logic.check_admin_access("regular_user")
        assert regular_admin_access["status"] == "error"
        assert "Admin privileges required" in regular_admin_access["message"]

        # Verify service calls were made through business logic
        assert telegram_service_mock.get_call_count("get_user_status") >= 4

    @pytest.mark.asyncio
    async def test_business_logic_error_handling(self, business_logic_with_mocks):
        """Test business logic error handling with service errors."""
        business_logic, telegram_service_mock, indicator_service_mock = business_logic_with_mocks

        # Simulate database error
        simulate_database_error(telegram_service_mock, "get_user_status")

        # Test that business logic handles the error gracefully through safe service calls
        is_admin = business_logic.is_admin_user("test_user")
        assert is_admin is False or is_admin is None  # Should return False or None on error

        # Test business logic error handling wrapper
        result = business_logic.handle_telegram_service_error(
            ConnectionError("Database connection failed"),
            "get_user_status",
            "test_context"
        )
        assert result["status"] == "error"
        assert result["fallback_available"] is True

        # Simulate indicator service error
        simulate_indicator_api_error(indicator_service_mock, "compute_for_ticker")

        # Test business logic indicator error handling
        calc_result = await business_logic.calculate_indicators_for_ticker(
            ticker="AAPL",
            indicators=["RSI"]
        )

        assert calc_result["status"] == "error"
        assert "API" in calc_result.get("message", "") or "connection" in calc_result.get("message", "").lower()

        # Test business logic indicator error categorization
        indicator_error_result = business_logic.handle_indicator_service_error(
            Exception("API key missing"),
            "test_context"
        )
        assert indicator_error_result["status"] == "error"
        assert "API configuration" in indicator_error_result["message"]

    def test_service_test_context_manager(self):
        """Test the ServiceTestContext context manager."""
        with ServiceTestContext() as ctx:
            # Setup test data
            ctx.setup_user("test_user", approved=True, verified=True)
            ctx.setup_indicators("AAPL", technical_indicators={"RSI": 65.5})

            # Test user setup
            status = ctx.telegram_service_mock.get_user_status("test_user")
            assert status["approved"] is True

            # Test indicator setup
            assert "AAPL" in ctx.indicator_service_mock.default_indicators

            # Create and test command
            command = ctx.create_command("info", telegram_user_id="test_user")
            assert command.command == "info"
            assert command.args["telegram_user_id"] == "test_user"

    @pytest.mark.asyncio
    async def test_alert_management_with_mocks(self, business_logic_with_mocks):
        """Test alert management operations with mocked services."""
        business_logic, telegram_service_mock, _ = business_logic_with_mocks

        # Setup user
        telegram_service_mock.add_test_user("test_user", approved=True, verified=True)

        # Test alert creation through service mock
        alert_id = telegram_service_mock.add_alert("test_user", "AAPL", 150.0, "above", email=True)
        assert alert_id == 1

        # Test alert retrieval
        alert = telegram_service_mock.get_alert(alert_id)
        assert alert["ticker"] == "AAPL"
        assert alert["price"] == 150.0
        assert alert["email"] is True

        # Test alert listing
        alerts = telegram_service_mock.list_alerts("test_user")
        assert len(alerts) == 1
        assert alerts[0]["id"] == alert_id

        # Verify service calls
        assert telegram_service_mock.get_call_count("add_alert") == 1
        assert telegram_service_mock.get_call_count("get_alert") == 1
        assert telegram_service_mock.get_call_count("list_alerts") == 1

    @pytest.mark.asyncio
    async def test_schedule_management_with_mocks(self, business_logic_with_mocks):
        """Test schedule management operations with mocked services."""
        business_logic, telegram_service_mock, _ = business_logic_with_mocks

        # Setup user
        telegram_service_mock.add_test_user("test_user", approved=True, verified=True)

        # Test schedule creation
        schedule_id = telegram_service_mock.add_schedule("test_user", "AAPL", "09:00")
        assert schedule_id == 1

        # Test schedule retrieval
        schedule = telegram_service_mock.get_schedule(schedule_id)
        assert schedule["ticker"] == "AAPL"
        assert schedule["scheduled_time"] == "09:00"

        # Test schedule update
        success = telegram_service_mock.update_schedule(schedule_id, enabled=False)
        assert success is True

        updated_schedule = telegram_service_mock.get_schedule(schedule_id)
        assert updated_schedule["enabled"] is False

        # Verify service calls
        assert telegram_service_mock.get_call_count("add_schedule") == 1
        assert telegram_service_mock.get_call_count("get_schedule") == 2
        assert telegram_service_mock.get_call_count("update_schedule") == 1

    def test_feedback_and_audit_with_mocks(self, business_logic_with_mocks):
        """Test feedback and audit operations with mocked services."""
        business_logic, telegram_service_mock, _ = business_logic_with_mocks

        # Test feedback creation
        feedback_id = telegram_service_mock.add_feedback("test_user", "bug", "Found an issue")
        assert feedback_id == 1

        # Test feedback listing
        feedback_list = telegram_service_mock.list_feedback("bug")
        assert len(feedback_list) == 1
        assert feedback_list[0]["message"] == "Found an issue"

        # Test audit logging
        audit_id = telegram_service_mock.log_command_audit("test_user", "report", ticker="AAPL")
        assert audit_id == 1

        # Test command history
        history = telegram_service_mock.get_user_command_history("test_user")
        assert len(history) == 1
        assert history[0]["command"] == "report"

        # Verify service calls
        assert telegram_service_mock.get_call_count("add_feedback") == 1
        assert telegram_service_mock.get_call_count("list_feedback") == 1
        assert telegram_service_mock.get_call_count("log_command_audit") == 1
        assert telegram_service_mock.get_call_count("get_user_command_history") == 1

    @pytest.mark.asyncio
    async def test_comprehensive_business_logic_flow(self, business_logic_with_mocks):
        """Test a comprehensive business logic flow using service mocks."""
        business_logic, telegram_service_mock, indicator_service_mock = business_logic_with_mocks

        # Setup test scenario
        user_data = {
            "approved": True,
            "verified": True,
            "email": "test@example.com",
            "language": "en",
            "is_admin": False,
            "max_alerts": 10,
            "max_schedules": 5
        }
        telegram_service_mock.add_test_user("test_user", **user_data)

        setup_indicator_data_in_mock(
            indicator_service_mock,
            "AAPL",
            technical_indicators={"RSI": 65.5, "MACD": 1.23, "SMA": 150.25},
            fundamental_indicators={"PE": 25.4, "MarketCap": 2500000000000}
        )

        # Test user info command
        info_command = create_parsed_command("info", "test_user")
        info_result = business_logic.handle_info(info_command)

        assert info_result["status"] == "ok"
        assert "test@example.com" in info_result["message"]
        assert "Yes" in info_result["message"]  # Verified and approved

        # Test indicator calculation
        calc_result = await business_logic.calculate_indicators_for_ticker(
            ticker="AAPL",
            indicators=["RSI", "MACD", "SMA", "PE"]
        )

        assert calc_result["status"] == "ok"
        assert calc_result["ticker"] == "AAPL"
        assert len(calc_result["technical"]) == 3  # RSI, MACD, SMA
        assert len(calc_result["fundamental"]) == 1  # PE

        # Test indicator value extraction
        from src.indicators.models import IndicatorResultSet
        result_set = IndicatorResultSet(
            ticker="AAPL",
            technical=calc_result["technical"],
            fundamental=calc_result["fundamental"]
        )

        formatted = business_logic.extract_indicator_values(result_set)
        assert "RSI" in formatted
        assert formatted["RSI"]["type"] == "technical"
        assert "65.50" in formatted["RSI"]["formatted"]

        # Verify all service interactions
        assert telegram_service_mock.get_call_count("get_user_status") >= 1
        assert indicator_service_mock.get_call_count("compute_for_ticker") == 1

        # Check that no errors occurred
        assert all(call["method"] != "configure_error" for call in telegram_service_mock.call_log)
        assert all(call["method"] != "configure_error" for call in indicator_service_mock.call_log)


if __name__ == "__main__":
    pytest.main([__file__])