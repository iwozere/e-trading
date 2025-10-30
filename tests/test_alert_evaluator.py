"""
Unit Tests for AlertEvaluator

Tests the AlertEvaluator service functionality including:
- Rule evaluation with various logical operators (and/or/not)
- Rearm logic with different configurations (hysteresis, cooldown, persistence)
- State persistence and recovery scenarios
- Market data and indicator integration
"""

import pytest
import json
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Add src to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

UTC = timezone.utc


@pytest.fixture(scope="session", autouse=True)
def setup_imports():
    """Setup imports for the test session."""
    global AlertEvaluator, AlertConfig, AlertEvaluationResult, RearmResult
    global AlertSchemaValidator, ValidationResult, DataManager, IndicatorService, JobsService
    global _logger

    from src.common.alerts.alert_evaluator import (
        AlertEvaluator, AlertConfig, AlertEvaluationResult, RearmResult
    )
    from src.common.alerts.schema_validator import AlertSchemaValidator, ValidationResult
    from src.data.data_manager import DataManager
    from src.indicators.service import IndicatorService
    from src.data.db.services.jobs_service import JobsService
    from src.notification.logger import setup_logger

    _logger = setup_logger(__name__)


class TestAlertEvaluator:
    """Test cases for AlertEvaluator functionality."""

    @pytest.fixture
    def mock_data_manager(self):
        """Create mock DataManager."""
        mock = AsyncMock(spec=DataManager)
        return mock

    @pytest.fixture
    def mock_indicator_service(self):
        """Create mock IndicatorService."""
        mock = Mock(spec=IndicatorService)
        return mock

    @pytest.fixture
    def mock_jobs_service(self):
        """Create mock JobsService."""
        mock = Mock(spec=JobsService)
        return mock

    @pytest.fixture
    def mock_schema_validator(self):
        """Create mock AlertSchemaValidator."""
        mock = Mock(spec=AlertSchemaValidator)
        mock.validate_alert_config = Mock(return_value=ValidationResult(
            is_valid=True, errors=[], warnings=[]
        ))
        return mock

    @pytest.fixture
    def alert_evaluator(self, mock_data_manager, mock_indicator_service,
                       mock_jobs_service, mock_schema_validator):
        """Create AlertEvaluator instance with mocked dependencies."""
        return AlertEvaluator(
            data_manager=mock_data_manager,
            indicator_service=mock_indicator_service,
            jobs_service=mock_jobs_service,
            schema_validator=mock_schema_validator
        )

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data DataFrame."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        np.random.seed(42)  # For reproducible tests

        # Generate realistic OHLCV data
        base_price = 100.0
        prices = []
        for i in range(100):
            price = base_price + np.random.normal(0, 2) + 0.1 * i  # Slight upward trend
            prices.append(price)

        df = pd.DataFrame({
            'open': prices,
            'high': [p + abs(np.random.normal(0, 1)) for p in prices],
            'low': [p - abs(np.random.normal(0, 1)) for p in prices],
            'close': [p + np.random.normal(0, 0.5) for p in prices],
            'volume': [1000 + abs(np.random.normal(0, 200)) for _ in prices]
        }, index=dates)

        # Ensure high >= low and realistic OHLC relationships
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)

        return df

    @pytest.fixture
    def sample_indicators(self):
        """Create sample indicators dictionary."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        return {
            'SMA_20': pd.Series([100 + i * 0.1 for i in range(100)], index=dates),
            'RSI': pd.Series([50 + np.random.normal(0, 10) for _ in range(100)], index=dates),
            'EMA_10': pd.Series([99 + i * 0.12 for i in range(100)], index=dates)
        }

    @pytest.fixture
    def mock_job_run(self):
        """Create mock job run object."""
        mock_schedule = Mock()
        mock_schedule.id = 123
        mock_schedule.task_params = {
            "ticker": "BTCUSDT",
            "timeframe": "1h",
            "rule": {
                "gt": {
                    "lhs": {"field": "close"},
                    "rhs": {"value": 100}
                }
            }
        }
        mock_schedule.state_json = None

        mock_run = Mock()
        mock_run.schedule = mock_schedule
        mock_run.id = 456

        return mock_run

    # Test Rule Evaluation with Logical Operators

    def test_evaluate_simple_gt_rule(self, alert_evaluator, sample_market_data, sample_indicators):
        """Test simple greater than rule evaluation."""
        rule = {
            "gt": {
                "lhs": {"field": "close"},
                "rhs": {"value": 100}
            }
        }

        result, sides, snapshot = alert_evaluator._evaluate_rule_tree(
            rule, sample_market_data, sample_indicators, {}
        )

        last_close = sample_market_data['close'].iloc[-1]
        expected_result = last_close > 100

        assert result == expected_result
        assert 'close' in snapshot
        assert 'value' in snapshot
        assert snapshot['close'] == last_close
        assert snapshot['value'] == 100

    def test_evaluate_and_rule(self, alert_evaluator, sample_market_data, sample_indicators):
        """Test AND logical operator."""
        rule = {
            "and": [
                {
                    "gt": {
                        "lhs": {"field": "close"},
                        "rhs": {"value": 100}
                    }
                },
                {
                    "lt": {
                        "lhs": {"field": "close"},
                        "rhs": {"value": 200}
                    }
                }
            ]
        }

        result, sides, snapshot = alert_evaluator._evaluate_rule_tree(
            rule, sample_market_data, sample_indicators, {}
        )

        last_close = sample_market_data['close'].iloc[-1]
        expected_result = 100 < last_close < 200

        assert result == expected_result
        assert 'close' in snapshot
        assert 'value' in snapshot

    def test_evaluate_or_rule(self, alert_evaluator, sample_market_data, sample_indicators):
        """Test OR logical operator."""
        rule = {
            "or": [
                {
                    "lt": {
                        "lhs": {"field": "close"},
                        "rhs": {"value": 50}  # Very low threshold
                    }
                },
                {
                    "gt": {
                        "lhs": {"field": "close"},
                        "rhs": {"value": 90}  # More likely threshold
                    }
                }
            ]
        }

        result, sides, snapshot = alert_evaluator._evaluate_rule_tree(
            rule, sample_market_data, sample_indicators, {}
        )

        last_close = sample_market_data['close'].iloc[-1]
        expected_result = last_close < 50 or last_close > 90

        assert result == expected_result

    def test_evaluate_not_rule(self, alert_evaluator, sample_market_data, sample_indicators):
        """Test NOT logical operator."""
        rule = {
            "not": {
                "gt": {
                    "lhs": {"field": "close"},
                    "rhs": {"value": 1000}  # Very high threshold
                }
            }
        }

        result, sides, snapshot = alert_evaluator._evaluate_rule_tree(
            rule, sample_market_data, sample_indicators, {}
        )

        last_close = sample_market_data['close'].iloc[-1]
        expected_result = not (last_close > 1000)

        assert result == expected_result

    def test_evaluate_between_rule(self, alert_evaluator, sample_market_data, sample_indicators):
        """Test BETWEEN range operator."""
        rule = {
            "between": {
                "value": {"field": "close"},
                "lower": {"value": 90},
                "upper": {"value": 120}
            }
        }

        result, sides, snapshot = alert_evaluator._evaluate_rule_tree(
            rule, sample_market_data, sample_indicators, {}
        )

        last_close = sample_market_data['close'].iloc[-1]
        expected_result = 90 <= last_close <= 120

        assert result == expected_result
        assert snapshot['close'] == last_close

    def test_evaluate_crosses_above_rule(self, alert_evaluator, sample_market_data, sample_indicators):
        """Test CROSSES_ABOVE operator with side tracking."""
        rule = {
            "crosses_above": {
                "lhs": {"field": "close"},
                "rhs": {"value": 105}
            }
        }

        # First evaluation - establish initial side
        result1, sides1, snapshot1 = alert_evaluator._evaluate_rule_tree(
            rule, sample_market_data, sample_indicators, {}
        )

        # Should not trigger on first evaluation (no previous side)
        assert result1 is False
        assert len(sides1) == 1  # One crossing key

        # Get the crossing key
        cross_key = list(sides1.keys())[0]

        # Simulate previous state where price was below threshold
        previous_sides = {cross_key: "below"}

        # Create modified data where close is above threshold
        modified_data = sample_market_data.copy()
        modified_data.loc[modified_data.index[-1], 'close'] = 110  # Above threshold

        result2, sides2, snapshot2 = alert_evaluator._evaluate_rule_tree(
            rule, modified_data, sample_indicators, previous_sides
        )

        # Should trigger crossing above
        assert result2 is True
        assert sides2[cross_key] == "above"

    def test_evaluate_indicator_rule(self, alert_evaluator, sample_market_data, sample_indicators):
        """Test rule evaluation with indicators."""
        rule = {
            "gt": {
                "lhs": {"indicator": {"type": "RSI", "output": "RSI"}},
                "rhs": {"value": 70}
            }
        }

        result, sides, snapshot = alert_evaluator._evaluate_rule_tree(
            rule, sample_market_data, sample_indicators, {}
        )

        last_rsi = sample_indicators['RSI'].iloc[-1]
        expected_result = last_rsi > 70

        assert result == expected_result
        assert 'RSI' in snapshot
        assert snapshot['RSI'] == last_rsi

    # Test Rearm Logic

    def test_simple_rearm_logic_no_config(self, alert_evaluator):
        """Test rearm logic when no rearm configuration is provided."""
        alert_config = AlertConfig(
            ticker="BTCUSDT",
            timeframe="1h",
            rule={"gt": {"lhs": {"field": "close"}, "rhs": {"value": 100}}},
            rearm=None,
            options={},
            notify={}
        )

        current_state = {"status": "ARMED"}

        # Test triggering without rearm config
        result = alert_evaluator._apply_rearm_logic(alert_config, current_state, True, False)

        assert result.should_rearm is False
        assert result.new_status == "TRIGGERED"
        assert "status" in result.state_updates
        assert "last_triggered" in result.state_updates

    def test_rearm_logic_with_cooldown(self, alert_evaluator):
        """Test rearm logic with cooldown period."""
        alert_config = AlertConfig(
            ticker="BTCUSDT",
            timeframe="1h",
            rule={"gt": {"lhs": {"field": "close"}, "rhs": {"value": 100}}},
            rearm={"enabled": True, "cooldown_minutes": 60},
            options={},
            notify={}
        )

        # Test within cooldown period
        recent_trigger = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
        current_state = {
            "status": "TRIGGERED",
            "last_triggered": recent_trigger
        }

        result = alert_evaluator._apply_rearm_logic(alert_config, current_state, False, True)

        assert result.new_status == "COOLDOWN"
        assert "status" in result.state_updates

    def test_rearm_logic_cooldown_expired(self, alert_evaluator):
        """Test rearm logic when cooldown period has expired."""
        alert_config = AlertConfig(
            ticker="BTCUSDT",
            timeframe="1h",
            rule={"gt": {"lhs": {"field": "close"}, "rhs": {"value": 100}}},
            rearm={"enabled": True, "cooldown_minutes": 60},
            options={},
            notify={}
        )

        # Test after cooldown period
        old_trigger = (datetime.now(UTC) - timedelta(minutes=120)).isoformat()
        current_state = {
            "status": "TRIGGERED",
            "last_triggered": old_trigger
        }

        result = alert_evaluator._apply_rearm_logic(alert_config, current_state, False, True)

        assert result.should_rearm is True
        assert result.new_status == "ARMED"
        assert "last_rearmed" in result.state_updates

    def test_rearm_logic_with_persistence_bars(self, alert_evaluator):
        """Test rearm logic with persistence bar requirements."""
        alert_config = AlertConfig(
            ticker="BTCUSDT",
            timeframe="1h",
            rule={"gt": {"lhs": {"field": "close"}, "rhs": {"value": 100}}},
            rearm={"enabled": True, "persistence_bars": 3},
            options={},
            notify={}
        )

        # Test insufficient persistence bars - should not trigger yet
        current_state = {
            "status": "ARMED",
            "consecutive_trigger_bars": 2
        }

        # The logic checks if consecutive_trigger_bars >= persistence_bars after increment
        # So with 2 bars + 1 (current trigger) = 3, which meets the requirement
        # Let's test with 1 bar to ensure it doesn't trigger
        current_state["consecutive_trigger_bars"] = 1

        result = alert_evaluator._apply_rearm_logic(alert_config, current_state, True, False)

        assert result.new_status == "ARMED"  # Should remain armed

        # Test sufficient persistence bars
        current_state["consecutive_trigger_bars"] = 2  # Will become 3 after increment

        result = alert_evaluator._apply_rearm_logic(alert_config, current_state, True, False)

        assert result.new_status == "TRIGGERED"

    def test_persistence_counter_updates(self, alert_evaluator):
        """Test persistence bar counter updates."""
        current_state = {"consecutive_trigger_bars": 2, "consecutive_rearm_bars": 0}

        # Test trigger increment
        updates = alert_evaluator._update_persistence_counters(current_state, True, False)
        assert updates["consecutive_trigger_bars"] == 3
        assert updates["consecutive_rearm_bars"] == 0

        # Test rearm increment
        updates = alert_evaluator._update_persistence_counters(current_state, False, True)
        assert updates["consecutive_trigger_bars"] == 0
        assert updates["consecutive_rearm_bars"] == 1

        # Test reset both
        updates = alert_evaluator._update_persistence_counters(current_state, False, False)
        assert updates["consecutive_trigger_bars"] == 0
        assert updates["consecutive_rearm_bars"] == 0

    # Test State Persistence and Recovery

    def test_load_alert_state_valid_json(self, alert_evaluator):
        """Test loading valid alert state from JSON."""
        state_json = json.dumps({
            "status": "TRIGGERED",
            "sides": {"cross_123": "above"},
            "last_triggered": "2023-01-01T12:00:00+00:00",
            "trigger_count": 5
        })

        state = alert_evaluator._load_alert_state(state_json)

        assert state["status"] == "TRIGGERED"
        assert state["sides"]["cross_123"] == "above"
        assert state["last_triggered"] == "2023-01-01T12:00:00+00:00"
        assert state["trigger_count"] == 5

    def test_load_alert_state_invalid_json(self, alert_evaluator):
        """Test loading invalid JSON returns default state."""
        invalid_json = '{"status": "TRIGGERED", invalid}'

        state = alert_evaluator._load_alert_state(invalid_json)

        # Should return default state
        assert state["status"] == "ARMED"
        assert state["sides"] == {}
        assert state["trigger_count"] == 0

    def test_load_alert_state_empty(self, alert_evaluator):
        """Test loading empty state returns default."""
        state = alert_evaluator._load_alert_state(None)

        assert state["status"] == "ARMED"
        assert state["sides"] == {}
        assert state["trigger_count"] == 0

    def test_sanitize_state_valid(self, alert_evaluator):
        """Test state sanitization with valid data."""
        raw_state = {
            "status": "TRIGGERED",
            "sides": {"cross_123": "above"},
            "last_triggered": "2023-01-01T12:00:00+00:00",
            "trigger_count": 5,
            "consecutive_trigger_bars": 2
        }

        sanitized = alert_evaluator._sanitize_state(raw_state)

        assert sanitized["status"] == "TRIGGERED"
        assert sanitized["sides"]["cross_123"] == "above"
        assert sanitized["trigger_count"] == 5
        assert "last_updated" in sanitized
        assert "version" in sanitized

    def test_sanitize_state_invalid_status(self, alert_evaluator):
        """Test state sanitization with invalid status."""
        raw_state = {"status": "INVALID_STATUS"}

        sanitized = alert_evaluator._sanitize_state(raw_state)

        assert sanitized["status"] == "ARMED"  # Should default to ARMED

    def test_sanitize_state_invalid_sides(self, alert_evaluator):
        """Test state sanitization with invalid sides data."""
        raw_state = {
            "sides": {
                "valid_key": "above",
                "invalid_key": "invalid_side",
                123: "above"  # Invalid key type
            }
        }

        sanitized = alert_evaluator._sanitize_state(raw_state)

        assert "valid_key" in sanitized["sides"]
        assert sanitized["sides"]["valid_key"] == "above"
        assert "invalid_key" not in sanitized["sides"]
        assert 123 not in sanitized["sides"]

    # Test Market Data and Indicator Integration

    @pytest.mark.asyncio
    async def test_fetch_market_data_success(self, alert_evaluator, mock_data_manager, sample_market_data):
        """Test successful market data fetching."""
        mock_data_manager.get_ohlcv.return_value = sample_market_data

        alert_config = AlertConfig(
            ticker="BTCUSDT",
            timeframe="1h",
            rule={},
            rearm=None,
            options={},
            notify={}
        )

        result = await alert_evaluator._fetch_market_data(alert_config)

        assert result is not None
        assert len(result) > 0
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    @pytest.mark.asyncio
    async def test_fetch_market_data_failure(self, alert_evaluator, mock_data_manager):
        """Test market data fetching failure."""
        mock_data_manager.get_ohlcv.side_effect = Exception("Data provider error")

        alert_config = AlertConfig(
            ticker="BTCUSDT",
            timeframe="1h",
            rule={},
            rearm=None,
            options={},
            notify={}
        )

        result = await alert_evaluator._fetch_market_data(alert_config)

        assert result is None

    def test_validate_market_data_valid(self, alert_evaluator, sample_market_data):
        """Test market data validation with valid data."""
        result = alert_evaluator._validate_market_data(sample_market_data, Mock(), 50)
        assert result is True

    def test_validate_market_data_empty(self, alert_evaluator):
        """Test market data validation with empty data."""
        empty_df = pd.DataFrame()
        result = alert_evaluator._validate_market_data(empty_df, Mock(), 50)
        assert result is False

    def test_validate_market_data_missing_columns(self, alert_evaluator):
        """Test market data validation with missing columns."""
        invalid_df = pd.DataFrame({'price': [100, 101, 102]})
        result = alert_evaluator._validate_market_data(invalid_df, Mock(), 50)
        assert result is False

    def test_validate_market_data_insufficient_data(self, alert_evaluator, sample_market_data):
        """Test market data validation with insufficient data."""
        small_df = sample_market_data.head(5)  # Only 5 rows
        result = alert_evaluator._validate_market_data(small_df, Mock(), 100)  # Expecting 100
        assert result is False

    def test_calculate_basic_indicators_sma(self, alert_evaluator, sample_market_data):
        """Test basic SMA indicator calculation."""
        specs = [{"type": "SMA", "params": {"period": 20, "source": "close"}, "output": "SMA_20"}]

        indicators = alert_evaluator._calculate_basic_indicators(sample_market_data, specs)

        assert "SMA_20" in indicators
        assert len(indicators["SMA_20"]) == len(sample_market_data)

        # Verify SMA calculation (last 20 values average)
        expected_sma = sample_market_data['close'].rolling(window=20).mean().iloc[-1]
        assert abs(indicators["SMA_20"].iloc[-1] - expected_sma) < 0.001

    def test_calculate_basic_indicators_rsi(self, alert_evaluator, sample_market_data):
        """Test basic RSI indicator calculation."""
        specs = [{"type": "RSI", "params": {"period": 14, "source": "close"}, "output": "RSI"}]

        indicators = alert_evaluator._calculate_basic_indicators(sample_market_data, specs)

        assert "RSI" in indicators
        assert len(indicators["RSI"]) == len(sample_market_data)

        # RSI should be between 0 and 100
        rsi_values = indicators["RSI"].dropna()
        assert all(0 <= val <= 100 for val in rsi_values)

    def test_calculate_required_lookback(self, alert_evaluator):
        """Test calculation of required lookback period."""
        alert_config = AlertConfig(
            ticker="BTCUSDT",
            timeframe="1h",
            rule={
                "gt": {
                    "lhs": {"indicator": {"type": "SMA", "params": {"period": 50}}},
                    "rhs": {"value": 100}
                }
            },
            rearm={
                "gt": {
                    "lhs": {"indicator": {"type": "EMA", "params": {"period": 20}}},
                    "rhs": {"value": 95}
                }
            },
            options={"lookback": 100},
            notify={}
        )

        lookback = alert_evaluator._calculate_required_lookback(alert_config)

        # Should be at least the configured lookback (100)
        assert lookback >= 100

    # Test Full Alert Evaluation

    @pytest.mark.asyncio
    async def test_evaluate_alert_success(self, alert_evaluator, mock_job_run,
                                        sample_market_data, mock_data_manager):
        """Test successful alert evaluation."""
        mock_data_manager.get_ohlcv.return_value = sample_market_data

        # Set up market data so rule will trigger (close > 100)
        modified_data = sample_market_data.copy()
        modified_data.loc[modified_data.index[-1], 'close'] = 150
        mock_data_manager.get_ohlcv.return_value = modified_data

        result = await alert_evaluator.evaluate_alert(mock_job_run)

        assert isinstance(result, AlertEvaluationResult)
        assert result.error is None
        assert isinstance(result.state_updates, dict)

    @pytest.mark.asyncio
    async def test_evaluate_alert_no_market_data(self, alert_evaluator, mock_job_run, mock_data_manager):
        """Test alert evaluation when no market data is available."""
        mock_data_manager.get_ohlcv.return_value = None

        result = await alert_evaluator.evaluate_alert(mock_job_run)

        assert isinstance(result, AlertEvaluationResult)
        assert result.triggered is False
        assert result.error == "No market data available"

    @pytest.mark.asyncio
    async def test_evaluate_alert_invalid_config(self, alert_evaluator, mock_job_run,
                                               mock_schema_validator):
        """Test alert evaluation with invalid configuration."""
        mock_schema_validator.validate_alert_config.return_value = ValidationResult(
            is_valid=False, errors=["Invalid ticker"], warnings=[]
        )

        result = await alert_evaluator.evaluate_alert(mock_job_run)

        assert isinstance(result, AlertEvaluationResult)
        assert result.triggered is False
        assert "Failed to parse alert configuration" in result.error

    def test_parse_alert_config_valid(self, alert_evaluator):
        """Test parsing valid alert configuration."""
        task_params = {
            "ticker": "BTCUSDT",
            "timeframe": "1h",
            "rule": {"gt": {"lhs": {"field": "close"}, "rhs": {"value": 100}}},
            "rearm": {"enabled": True},
            "options": {"lookback": 200},
            "notify": {"channels": ["telegram"]}
        }

        config = alert_evaluator._parse_alert_config(task_params)

        assert config is not None
        assert config.ticker == "BTCUSDT"
        assert config.timeframe == "1h"
        assert config.rule is not None
        assert config.rearm is not None
        assert config.options["lookback"] == 200

    def test_parse_alert_config_missing_required(self, alert_evaluator):
        """Test parsing alert configuration with missing required fields."""
        task_params = {
            "ticker": "BTCUSDT"
            # Missing timeframe and rule
        }

        config = alert_evaluator._parse_alert_config(task_params)

        assert config is None

    def test_prepare_notification_data(self, alert_evaluator, sample_market_data, sample_indicators):
        """Test preparation of notification data."""
        alert_config = AlertConfig(
            ticker="BTCUSDT",
            timeframe="1h",
            rule={},
            rearm=None,
            options={},
            notify={"channels": ["telegram"]}
        )

        rule_snapshot = {"close": 105.5, "value": 100}

        notification_data = alert_evaluator._prepare_notification_data(
            alert_config, sample_market_data, sample_indicators, rule_snapshot
        )

        assert notification_data["ticker"] == "BTCUSDT"
        assert notification_data["timeframe"] == "1h"
        assert "timestamp" in notification_data
        assert "price" in notification_data
        assert "volume" in notification_data
        assert notification_data["rule_snapshot"] == rule_snapshot
        assert "indicators" in notification_data

    def test_timeframe_to_minutes(self, alert_evaluator):
        """Test timeframe conversion to minutes."""
        assert alert_evaluator._timeframe_to_minutes("1m") == 1
        assert alert_evaluator._timeframe_to_minutes("5m") == 5
        assert alert_evaluator._timeframe_to_minutes("15m") == 15
        assert alert_evaluator._timeframe_to_minutes("30m") == 30
        assert alert_evaluator._timeframe_to_minutes("1h") == 60
        assert alert_evaluator._timeframe_to_minutes("4h") == 240
        assert alert_evaluator._timeframe_to_minutes("1d") == 1440
        assert alert_evaluator._timeframe_to_minutes("unknown") == 60  # Default

    def test_generate_cross_key(self, alert_evaluator):
        """Test generation of unique crossing keys."""
        node1 = {"lhs": {"field": "close"}, "rhs": {"value": 100}}
        node2 = {"lhs": {"field": "close"}, "rhs": {"value": 100}}
        node3 = {"lhs": {"field": "close"}, "rhs": {"value": 101}}

        key1 = alert_evaluator._generate_cross_key(node1)
        key2 = alert_evaluator._generate_cross_key(node2)
        key3 = alert_evaluator._generate_cross_key(node3)

        # Same nodes should generate same key
        assert key1 == key2
        # Different nodes should generate different keys
        assert key1 != key3
        # Keys should be strings
        assert isinstance(key1, str)

    def test_calculate_atr(self, alert_evaluator, sample_market_data):
        """Test ATR calculation."""
        atr = alert_evaluator._calculate_atr(sample_market_data, period=14)

        assert atr is not None
        assert isinstance(atr, float)
        assert atr > 0

    def test_calculate_atr_insufficient_data(self, alert_evaluator):
        """Test ATR calculation with insufficient data."""
        small_df = pd.DataFrame({
            'high': [100, 101],
            'low': [99, 100],
            'close': [100.5, 100.8]
        })

        atr = alert_evaluator._calculate_atr(small_df, period=14)

        assert atr is None