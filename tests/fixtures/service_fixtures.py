"""
Test fixtures for service layer testing.

Provides pytest fixtures and helper functions for setting up
service mocks and business logic instances for testing.
"""

import pytest
from typing import Tuple, Dict, Any, List
from unittest.mock import Mock

from tests.mocks.telegram_service_mock import TelegramServiceMock
from tests.mocks.indicator_service_mock import IndicatorServiceMock
from src.telegram.screener.business_logic import TelegramBusinessLogic
from src.telegram.command_parser import ParsedCommand
from src.indicators.models import TickerIndicatorsRequest, IndicatorResultSet, IndicatorValue


@pytest.fixture
def telegram_service_mock() -> TelegramServiceMock:
    """Provide a fresh TelegramServiceMock for each test."""
    return TelegramServiceMock()


@pytest.fixture
def indicator_service_mock() -> IndicatorServiceMock:
    """Provide a fresh IndicatorServiceMock for each test."""
    return IndicatorServiceMock()


@pytest.fixture
def business_logic_with_mocks(telegram_service_mock, indicator_service_mock) -> Tuple[TelegramBusinessLogic, TelegramServiceMock, IndicatorServiceMock]:
    """Provide TelegramBusinessLogic with mocked services."""
    business_logic = TelegramBusinessLogic(
        telegram_service=telegram_service_mock,
        indicator_service=indicator_service_mock
    )
    return business_logic, telegram_service_mock, indicator_service_mock


@pytest.fixture
def sample_user_data() -> Dict[str, Any]:
    """Provide sample user data for testing."""
    return {
        "approved": True,
        "verified": True,
        "email": "test@example.com",
        "language": "en",
        "is_admin": False,
        "max_alerts": 10,
        "max_schedules": 5,
        "verification_code": None,
        "code_sent_time": None
    }


@pytest.fixture
def admin_user_data() -> Dict[str, Any]:
    """Provide admin user data for testing."""
    return {
        "approved": True,
        "verified": True,
        "email": "admin@example.com",
        "language": "en",
        "is_admin": True,
        "max_alerts": 50,
        "max_schedules": 20,
        "verification_code": None,
        "code_sent_time": None
    }


@pytest.fixture
def unverified_user_data() -> Dict[str, Any]:
    """Provide unverified user data for testing."""
    return {
        "approved": False,
        "verified": False,
        "email": "unverified@example.com",
        "language": "en",
        "is_admin": False,
        "max_alerts": 5,
        "max_schedules": 2,
        "verification_code": "123456",
        "code_sent_time": 1640995200  # Mock timestamp
    }


@pytest.fixture
def sample_parsed_command() -> ParsedCommand:
    """Provide a sample ParsedCommand for testing."""
    return ParsedCommand(
        command="report",
        args={
            "telegram_user_id": "test_user_123",
            "ticker": "AAPL",
            "indicators": ["RSI", "MACD"],
            "interval": "1d",
            "period": "1y"
        }
    )


@pytest.fixture
def sample_indicator_request() -> TickerIndicatorsRequest:
    """Provide a sample TickerIndicatorsRequest for testing."""
    return TickerIndicatorsRequest(
        ticker="AAPL",
        indicators=["RSI", "MACD", "SMA"],
        timeframe="1d",
        period="1y",
        provider="yahoo"
    )


@pytest.fixture
def sample_indicator_result() -> IndicatorResultSet:
    """Provide a sample IndicatorResultSet for testing."""
    return IndicatorResultSet(
        ticker="AAPL",
        technical={
            "RSI": IndicatorValue(name="RSI", value=65.5),
            "MACD": IndicatorValue(name="MACD", value=1.23),
            "SMA": IndicatorValue(name="SMA", value=150.25)
        },
        fundamental={
            "PE": IndicatorValue(name="PE", value=25.4),
            "MarketCap": IndicatorValue(name="MarketCap", value=2500000000000)
        }
    )


# Helper functions for test setup

def setup_user_in_mock(telegram_service_mock: TelegramServiceMock,
                      telegram_user_id: str,
                      user_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Set up a user in the telegram service mock.

    Args:
        telegram_service_mock: The mock service instance
        telegram_user_id: User ID to set up
        user_data: Optional user data, uses default if not provided

    Returns:
        The user data that was set up
    """
    if user_data is None:
        user_data = {
            "approved": True,
            "verified": True,
            "email": "test@example.com",
            "language": "en",
            "is_admin": False,
            "max_alerts": 10,
            "max_schedules": 5,
            "verification_code": None,
            "code_sent_time": None
        }

    telegram_service_mock.add_test_user(telegram_user_id, **user_data)
    return user_data


def setup_indicator_data_in_mock(indicator_service_mock: IndicatorServiceMock,
                                ticker: str,
                                technical_indicators: Dict[str, float] = None,
                                fundamental_indicators: Dict[str, float] = None):
    """
    Set up indicator data in the indicator service mock.

    Args:
        indicator_service_mock: The mock service instance
        ticker: Ticker symbol to set up data for
        technical_indicators: Dict of technical indicator names to values
        fundamental_indicators: Dict of fundamental indicator names to values
    """
    technical = {}
    if technical_indicators:
        for name, value in technical_indicators.items():
            technical[name] = IndicatorValue(name=name, value=value)

    fundamental = {}
    if fundamental_indicators:
        for name, value in fundamental_indicators.items():
            fundamental[name] = IndicatorValue(name=name, value=value)

    indicator_service_mock.add_ticker_data(ticker, technical, fundamental)


def create_parsed_command(command: str,
                         telegram_user_id: str = "test_user_123",
                         **args) -> ParsedCommand:
    """
    Create a ParsedCommand for testing.

    Args:
        command: Command name
        telegram_user_id: User ID for the command
        **args: Additional command arguments

    Returns:
        ParsedCommand instance
    """
    command_args = {"telegram_user_id": telegram_user_id}
    command_args.update(args)

    return ParsedCommand(command=command, args=command_args)


def assert_service_called(service_mock, method_name: str, *args, **kwargs):
    """
    Assert that a service method was called with specific arguments.

    Args:
        service_mock: The mock service instance
        method_name: Name of the method that should have been called
        *args, **kwargs: Expected arguments
    """
    assert service_mock.was_called_with(method_name, *args, **kwargs), \
        f"Expected {method_name} to be called with args={args}, kwargs={kwargs}"


def assert_service_call_count(service_mock, method_name: str, expected_count: int):
    """
    Assert that a service method was called a specific number of times.

    Args:
        service_mock: The mock service instance
        method_name: Name of the method to check
        expected_count: Expected number of calls
    """
    actual_count = service_mock.get_call_count(method_name)
    assert actual_count == expected_count, \
        f"Expected {method_name} to be called {expected_count} times, but was called {actual_count} times"


def get_last_service_call(service_mock, method_name: str) -> Dict[str, Any]:
    """
    Get the last call to a service method.

    Args:
        service_mock: The mock service instance
        method_name: Name of the method to check

    Returns:
        Dict with call information or None if not called
    """
    return service_mock.get_last_call(method_name)


# Error simulation helpers

def simulate_database_error(telegram_service_mock: TelegramServiceMock,
                           method_name: str = "get_user_status"):
    """Simulate a database connection error."""
    telegram_service_mock.configure_error(
        method_name,
        ConnectionError("Database connection failed")
    )


def simulate_rate_limit_error(telegram_service_mock: TelegramServiceMock,
                             method_name: str = "set_verification_code"):
    """Simulate a rate limit error."""
    telegram_service_mock.configure_error(
        method_name,
        Exception("Rate limit exceeded - too many requests")
    )


def simulate_indicator_api_error(indicator_service_mock: IndicatorServiceMock,
                                method_name: str = "compute_for_ticker"):
    """Simulate an indicator API error."""
    indicator_service_mock.configure_error(
        method_name,
        ConnectionError("Unable to fetch market data - API error")
    )


def simulate_invalid_ticker_error(indicator_service_mock: IndicatorServiceMock,
                                 method_name: str = "compute_for_ticker"):
    """Simulate an invalid ticker error."""
    indicator_service_mock.configure_error(
        method_name,
        ValueError("Ticker symbol not found")
    )


# Test data generators

def generate_test_alerts(count: int = 3) -> List[Dict[str, Any]]:
    """Generate test alert data."""
    alerts = []
    for i in range(count):
        alerts.append({
            "id": i + 1,
            "user_id": "test_user_123",
            "ticker": f"TEST{i}",
            "price": 100.0 + i * 10,
            "condition": "above",
            "email": i % 2 == 0,
            "status": "ARMED",
            "created_at": 1640995200 + i * 3600,
            "enabled": True
        })
    return alerts


def generate_test_schedules(count: int = 2) -> List[Dict[str, Any]]:
    """Generate test schedule data."""
    schedules = []
    for i in range(count):
        schedules.append({
            "id": i + 1,
            "user_id": "test_user_123",
            "ticker": f"SCHED{i}",
            "scheduled_time": f"0{9 + i} 00 * * *",
            "created_at": 1640995200 + i * 3600,
            "enabled": True
        })
    return schedules


# Integration test helpers

class ServiceTestContext:
    """
    Context manager for service layer testing that provides
    clean setup and teardown of mock services.
    """

    def __init__(self):
        self.telegram_service_mock = None
        self.indicator_service_mock = None
        self.business_logic = None

    def __enter__(self):
        self.telegram_service_mock = TelegramServiceMock()
        self.indicator_service_mock = IndicatorServiceMock()
        self.business_logic = TelegramBusinessLogic(
            telegram_service=self.telegram_service_mock,
            indicator_service=self.indicator_service_mock
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.telegram_service_mock:
            self.telegram_service_mock.reset_mock()
        if self.indicator_service_mock:
            self.indicator_service_mock.reset_mock()

    def setup_user(self, telegram_user_id: str, **user_data):
        """Set up a user in the context."""
        return setup_user_in_mock(self.telegram_service_mock, telegram_user_id, user_data)

    def setup_indicators(self, ticker: str, **indicators):
        """Set up indicator data in the context."""
        return setup_indicator_data_in_mock(self.indicator_service_mock, ticker, **indicators)

    def create_command(self, command: str, telegram_user_id: str = "test_user", **args):
        """Create a command in the context."""
        return create_parsed_command(command, telegram_user_id, **args)


# Usage example for tests:
#
# def test_business_logic_with_context():
#     with ServiceTestContext() as ctx:
#         ctx.setup_user("test_user", approved=True, verified=True)
#         ctx.setup_indicators("AAPL", technical_indicators={"RSI": 65.5})
#
#         command = ctx.create_command("report", ticker="AAPL")
#         result = await ctx.business_logic.handle_command(command)
#
#         assert result["status"] == "ok"