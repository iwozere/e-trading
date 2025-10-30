"""
Mock implementation of IndicatorService for testing.

This mock provides async indicator calculation methods with configurable
responses and error simulation for comprehensive testing.
"""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import Mock
import time

from src.indicators.models import (
    IndicatorBatchConfig, IndicatorResultSet,
    IndicatorSpec, IndicatorValue, TickerIndicatorsRequest
)
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class IndicatorServiceMock:
    """
    Mock implementation of IndicatorService for testing.

    Provides configurable async responses and tracks method calls for verification.
    """

    def __init__(self):
        """Initialize mock with default data and call tracking."""
        self.call_log = []
        self.adapters = {
            "ta-lib": Mock(),
            "pandas-ta": Mock(),
            "fundamentals": Mock()
        }

        # Configuration for mock behavior
        self.should_raise_errors = {}
        self.custom_responses = {}
        self.response_delays = {}

        # Default indicator values for different tickers
        self.default_indicators = {
            "AAPL": {
                "technical": {
                    "RSI": IndicatorValue(name="RSI", value=65.5),
                    "MACD": IndicatorValue(name="MACD", value=1.23),
                    "SMA": IndicatorValue(name="SMA", value=150.25),
                    "BollingerBands": IndicatorValue(name="BollingerBands", value=155.0)
                },
                "fundamental": {
                    "PE": IndicatorValue(name="PE", value=25.4),
                    "MarketCap": IndicatorValue(name="MarketCap", value=2500000000000)
                }
            },
            "TSLA": {
                "technical": {
                    "RSI": IndicatorValue(name="RSI", value=45.2),
                    "MACD": IndicatorValue(name="MACD", value=-0.87),
                    "SMA": IndicatorValue(name="SMA", value=245.75),
                    "BollingerBands": IndicatorValue(name="BollingerBands", value=250.0)
                },
                "fundamental": {
                    "PE": IndicatorValue(name="PE", value=65.8),
                    "MarketCap": IndicatorValue(name="MarketCap", value=800000000000)
                }
            },
            "BTCUSDT": {
                "technical": {
                    "RSI": IndicatorValue(name="RSI", value=55.8),
                    "MACD": IndicatorValue(name="MACD", value=2.45),
                    "SMA": IndicatorValue(name="SMA", value=45000.0),
                    "BollingerBands": IndicatorValue(name="BollingerBands", value=46000.0)
                },
                "fundamental": {}  # Crypto typically has limited fundamental data
            }
        }

    def _log_call(self, method_name: str, *args, **kwargs):
        """Log method calls for verification in tests."""
        self.call_log.append({
            "method": method_name,
            "args": args,
            "kwargs": kwargs,
            "timestamp": time.time()
        })
        _logger.debug("Mock call: %s(*%s, **%s)", method_name, args, kwargs)

    async def _check_error_config(self, method_name: str):
        """Check if method should raise an error based on configuration."""
        if method_name in self.should_raise_errors:
            error = self.should_raise_errors[method_name]

            # Add delay if configured
            if method_name in self.response_delays:
                await asyncio.sleep(self.response_delays[method_name])

            if callable(error):
                raise error()
            else:
                raise error

    def _get_custom_response(self, method_name: str, default_response):
        """Get custom response if configured, otherwise return default."""
        return self.custom_responses.get(method_name, default_response)

    async def compute_for_ticker(self, req: TickerIndicatorsRequest) -> IndicatorResultSet:
        """
        Compute indicators for a ticker with mock data.

        Args:
            req: TickerIndicatorsRequest with ticker and indicator specifications

        Returns:
            IndicatorResultSet with mock indicator values
        """
        self._log_call("compute_for_ticker", req)
        await self._check_error_config("compute_for_ticker")

        # Add response delay if configured
        if "compute_for_ticker" in self.response_delays:
            await asyncio.sleep(self.response_delays["compute_for_ticker"])

        # Get default indicators for the ticker
        ticker_upper = req.ticker.upper()
        if ticker_upper in self.default_indicators:
            ticker_data = self.default_indicators[ticker_upper]
        else:
            # Generate default indicators for unknown tickers
            ticker_data = {
                "technical": {
                    "RSI": IndicatorValue(name="RSI", value=50.0),
                    "MACD": IndicatorValue(name="MACD", value=0.0),
                    "SMA": IndicatorValue(name="SMA", value=100.0),
                    "BollingerBands": IndicatorValue(name="BollingerBands", value=105.0)
                },
                "fundamental": {
                    "PE": IndicatorValue(name="PE", value=20.0)
                }
            }

        # Filter indicators based on request
        technical_indicators = {}
        fundamental_indicators = {}

        for indicator_name in req.indicators:
            # Check technical indicators
            if indicator_name in ticker_data["technical"]:
                technical_indicators[indicator_name] = ticker_data["technical"][indicator_name]
            # Check fundamental indicators
            elif indicator_name in ticker_data["fundamental"]:
                fundamental_indicators[indicator_name] = ticker_data["fundamental"][indicator_name]
            else:
                # Create a default indicator value for unknown indicators
                default_value = IndicatorValue(name=indicator_name, value=None)
                if indicator_name.upper() in ["RSI", "MACD", "SMA", "EMA", "BOLLINGERBANDS"]:
                    technical_indicators[indicator_name] = default_value
                else:
                    fundamental_indicators[indicator_name] = default_value

        result = IndicatorResultSet(
            ticker=req.ticker,
            technical=technical_indicators,
            fundamental=fundamental_indicators
        )

        return self._get_custom_response("compute_for_ticker", result)

    async def compute(self, df, config: IndicatorBatchConfig, fund_params: Dict[str, Any] = None) -> Any:
        """
        Compute indicators in batch mode with mock data.

        Args:
            df: DataFrame with OHLCV data
            config: IndicatorBatchConfig with indicator specifications
            fund_params: Additional parameters for fundamental indicators

        Returns:
            Mock DataFrame with computed indicators
        """
        self._log_call("compute", df, config, fund_params)
        await self._check_error_config("compute")

        # Add response delay if configured
        if "compute" in self.response_delays:
            await asyncio.sleep(self.response_delays["compute"])

        # For batch compute, we'll return a mock DataFrame
        # In real tests, this would be more sophisticated
        import pandas as pd

        # Create a mock result DataFrame based on input
        result_df = df.copy() if hasattr(df, 'copy') else pd.DataFrame()

        # Add mock indicator columns based on config
        for spec in config.indicators:
            if hasattr(spec, 'output'):
                if isinstance(spec.output, dict):
                    for output_key, column_name in spec.output.items():
                        result_df[column_name] = 50.0  # Mock value
                else:
                    result_df[spec.output] = 50.0  # Mock value

        return self._get_custom_response("compute", result_df)

    def supports(self, indicator_name: str) -> bool:
        """Check if an indicator is supported."""
        self._log_call("supports", indicator_name)

        # Mock support for common indicators
        supported_indicators = [
            "RSI", "MACD", "SMA", "EMA", "BollingerBands", "PE", "MarketCap",
            "STOCH", "ADX", "CCI", "ROC", "MFI", "WILLR", "ATR"
        ]

        return indicator_name.upper() in [ind.upper() for ind in supported_indicators]

    # Test Configuration Methods

    def configure_error(self, method_name: str, error: Exception):
        """Configure a method to raise an error."""
        self.should_raise_errors[method_name] = error

    def configure_response(self, method_name: str, response: Any):
        """Configure a custom response for a method."""
        self.custom_responses[method_name] = response

    def configure_delay(self, method_name: str, delay_seconds: float):
        """Configure a response delay for a method."""
        self.response_delays[method_name] = delay_seconds

    def clear_errors(self):
        """Clear all error configurations."""
        self.should_raise_errors.clear()

    def clear_responses(self):
        """Clear all custom response configurations."""
        self.custom_responses.clear()

    def clear_delays(self):
        """Clear all response delay configurations."""
        self.response_delays.clear()

    def reset_mock(self):
        """Reset all mock data and configurations."""
        self.call_log.clear()
        self.should_raise_errors.clear()
        self.custom_responses.clear()
        self.response_delays.clear()

    def get_call_count(self, method_name: str) -> int:
        """Get the number of times a method was called."""
        return len([call for call in self.call_log if call["method"] == method_name])

    def get_last_call(self, method_name: str) -> Optional[Dict[str, Any]]:
        """Get the last call to a specific method."""
        calls = [call for call in self.call_log if call["method"] == method_name]
        return calls[-1] if calls else None

    def was_called_with(self, method_name: str, *args, **kwargs) -> bool:
        """Check if a method was called with specific arguments."""
        for call in self.call_log:
            if (call["method"] == method_name and
                call["args"] == args and
                call["kwargs"] == kwargs):
                return True
        return False

    def add_ticker_data(self, ticker: str, technical: Dict[str, IndicatorValue] = None,
                       fundamental: Dict[str, IndicatorValue] = None):
        """Add mock data for a specific ticker."""
        self.default_indicators[ticker.upper()] = {
            "technical": technical or {},
            "fundamental": fundamental or {}
        }

    def simulate_api_error(self, method_name: str = "compute_for_ticker"):
        """Simulate common API errors."""
        self.configure_error(method_name, ConnectionError("API connection failed"))

    def simulate_rate_limit(self, method_name: str = "compute_for_ticker"):
        """Simulate rate limit errors."""
        self.configure_error(method_name, Exception("Rate limit exceeded"))

    def simulate_invalid_ticker(self, method_name: str = "compute_for_ticker"):
        """Simulate invalid ticker errors."""
        self.configure_error(method_name, ValueError("Ticker not found"))

    def simulate_timeout(self, method_name: str = "compute_for_ticker"):
        """Simulate timeout errors."""
        self.configure_error(method_name, TimeoutError("Request timeout"))

    def simulate_slow_response(self, method_name: str = "compute_for_ticker", delay: float = 2.0):
        """Simulate slow API responses."""
        self.configure_delay(method_name, delay)


class AsyncMockIndicatorService:
    """
    Alternative async mock that can be used with unittest.mock.AsyncMock.

    This provides a simpler interface for basic mocking scenarios.
    """

    def __init__(self):
        self.compute_for_ticker = Mock()
        self.compute = Mock()
        self.adapters = {"ta-lib": Mock(), "pandas-ta": Mock(), "fundamentals": Mock()}

    async def mock_compute_for_ticker(self, req: TickerIndicatorsRequest) -> IndicatorResultSet:
        """Simple mock implementation for compute_for_ticker."""
        return IndicatorResultSet(
            ticker=req.ticker,
            technical={
                "RSI": IndicatorValue(name="RSI", value=65.5),
                "MACD": IndicatorValue(name="MACD", value=1.23)
            },
            fundamental={
                "PE": IndicatorValue(name="PE", value=25.4)
            }
        )

    async def mock_compute(self, df, config: IndicatorBatchConfig, fund_params: Dict[str, Any] = None):
        """Simple mock implementation for compute."""
        import pandas as pd
        return pd.DataFrame({"mock_indicator": [50.0, 51.0, 52.0]})


# Factory functions for easy test setup

def create_telegram_service_mock():
    """Create a new TelegramServiceMock instance."""
    from tests.mocks.telegram_service_mock import TelegramServiceMock
    return TelegramServiceMock()


def create_indicator_service_mock() -> IndicatorServiceMock:
    """Create a new IndicatorServiceMock instance."""
    return IndicatorServiceMock()


def create_business_logic_with_mocks():
    """Create TelegramBusinessLogic with mocked services for testing."""
    from src.telegram.screener.business_logic import TelegramBusinessLogic

    telegram_service_mock = create_telegram_service_mock()
    indicator_service_mock = create_indicator_service_mock()

    business_logic = TelegramBusinessLogic(
        telegram_service=telegram_service_mock,
        indicator_service=indicator_service_mock
    )

    return business_logic, telegram_service_mock, indicator_service_mock