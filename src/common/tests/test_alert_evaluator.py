"""
Unit tests for AlertEvaluator service.

Tests cover:
- Rule evaluation with various logical combinations
- Rearm logic with different configurations
- State persistence and recovery scenarios
- Market data integration
- Error handling
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd

from src.common.alerts.alert_evaluator import (
    AlertEvaluator, AlertConfig, AlertEvaluationResult, RearmResult
)
from src.common.alerts.schema_validator import AlertSchemaValidator, ValidationResult


class TestAlertEvaluator:
    """Test cases for AlertEvaluator class."""

    @pytest.fixture
    def mock_data_manager(self):
        """Mock DataManager for testing."""
        mock = Mock()
        mock.get_ohlcv = Mock()
        return mock

    @pytest.fixture
    def mock_indicator_service(self):
        """Mock IndicatorService for testing."""
        mock = Mock()
        return mock

    @pytest.fixture
    def mock_jobs_service(self):
        """Mock JobsService for testing."""
        mock = Mock()
        return mock

    @pytest.fixture
    def mock_schema_validator(self):
        """Mock AlertSchemaValidator for testing."""
        mock = Mock()
        mock.validate_alert_config = Mock(return_value=ValidationResult(
            is_valid=True, errors=[], warnings=[]
        ))
        return mock

    @pytest.fixture
    def alert_evaluator(self, mock_data_manager, mock_indicator_service,
                       mock_jobs_service, mock_schema_validator):
        """Create AlertEvaluator instance for testing."""
        return AlertEvaluator(
            data_manager=mock_data_manager,
            indicator_service=mock_indicator_service,
            jobs_service=mock_jobs_service,
            schema_validator=mock_schema_validator
        )

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H', tz='UTC')
        data = {
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [101 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def simple_alert_config(self):
        """Create simple alert configuration for testing."""
        return AlertConfig(
            ticker="BTCUSDT",
            timeframe="1h",
            rule={"gt": {"lhs": {"field": "close"}, "rhs": {"value": 105}}},
            rearm=None,
            options={},
            notify={}
        )

    @pytest.fixture
    def rearm_alert_config(self):
        """Create alert configuration with rearm logic."""
        return AlertConfig(
            ticker="BTCUSDT",
            timeframe="1h",
            rule={"gt": {"lhs": {"field": "close"}, "rhs": {"value": 105}}},
            rearm={
                "enabled": True,
                "threshold": 105,
                "direction": "above",
                "hysteresis": 1.0,
                "hysteresis_type": "percentage",
                "cooldown_minutes": 15,
                "persistence_bars": 1
            },
            options={},
            notify={}
        )

    def test_parse_alert_config_valid(self, alert_evaluator):
        """Test parsing valid alert configuration."""
        task_params = {
            "ticker": "BTCUSDT",
            "timeframe": "1h",
            "rule": {"gt": {"lhs": {"field": "close"}, "rhs": {"value": 100}}},
            "options": {"lookback": 200},
            "notify": {"channels": ["telegram"]}
        }

        config = alert_evaluator._parse_alert_config(task_params)

        assert config is not None
        assert config.ticker == "BTCUSDT"
        assert config.timeframe == "1h"
        assert config.rule == {"gt": {"lhs": {"field": "close"}, "rhs": {"value": 100}}}
        assert config.options == {"lookback": 200}
        assert config.notify == {"channels": ["telegram"]}

    def test_parse_alert_config_invalid(self, alert_evaluator, mock_schema_validator):
        """Test parsing invalid alert configuration."""
        mock_schema_validator.validate_alert_config.return_value = ValidationResult(
            is_valid=False, errors=["Missing required field"], warnings=[]
        )

        task_params = {"invalid": "config"}
        config = alert_evaluator._parse_alert_config(task_params)

        assert config is None

    def test_load_alert_state_valid(self, alert_evaluator):
        """Test loading valid alert state."""
        state_json = json.dumps({
            "status": "ARMED",
            "sides": {"test_key": "above"},
            "trigger_count": 5
        })

        state = alert_evaluator._load_alert_state(state_json)

        assert state["status"] == "ARMED"
        assert state["sides"] == {"test_key": "above"}
        assert state["trigger_count"] == 5

    def test_load_alert_state_invalid_json(self, alert_evaluator):
        """Test loading invalid JSON state."""
        state_json = "invalid json"
        state = alert_evaluator._load_alert_state(state_json)

        # Should return default state
        assert state["status"] == "ARMED"
        assert state["sides"] == {}

    def test_load_alert_state_empty(self, alert_evaluator):
        """Test loading empty state."""
        state = alert_evaluator._load_alert_state(None)

        # Should return default state
        assert state["status"] == "ARMED"
        assert state["sides"] == {}

    def test_sanitize_state_valid(self, alert_evaluator):
        """Test state sanitization with valid data."""
        state = {
            "status": "TRIGGERED",
            "sides": {"key1": "above", "key2": "below"},
            "trigger_count": 3,
            "last_triggered": "2023-01-01T12:00:00+00:00"
        }

        sanitized = alert_evaluator._sanitize_state(state)

        assert sanitized["status"] == "TRIGGERED"
        assert sanitized["sides"] == {"key1": "above", "key2": "below"}
        assert sanitized["trigger_count"] == 3
        assert sanitized["last_triggered"] == "2023-01-01T12:00:00+00:00"
        assert "version" in sanitized

    def test_sanitize_state_invalid_status(self, alert_evaluator):
        """Test state sanitization with invalid status."""
        state = {"status": "INVALID_STATUS"}
        sanitized = alert_evaluator._sanitize_state(state)

        assert sanitized["status"] == "ARMED"  # Should default to ARMED

    def test_sanitize_state_invalid_sides(self, alert_evaluator):
        """Test state sanitization with invalid sides."""
        state = {
            "sides": {
                "valid_key": "above",
                "invalid_key": "invalid_side",
                123: "above"  # Invalid key type
            }
        }

        sanitized = alert_evaluator._sanitize_state(state)

        # Should only keep valid entries
        assert sanitized["sides"] == {"valid_key": "above"}

    def test_evaluate_comparison_gt(self, alert_evaluator, sample_market_data):
        """Test greater than comparison evaluation."""
        node = {"lhs": {"field": "close"}, "rhs": {"value": 105}}
        result, sides, snapshot = alert_evaluator._evaluate_comparison(
            "gt", node, sample_market_data, {}, {}
        )

        # Last close price should be around 109.5, so > 105 should be True
        assert result is True
        assert "close" in snapshot
        assert "value" in snapshot

    def test_evaluate_comparison_crosses_above(self, alert_evaluator):
        """Test crosses above comparison evaluation."""
        # Create data where price crosses above threshold
        dates = pd.date_range(start='2023-01-01', periods=3, freq='1H', tz='UTC')
        data = pd.DataFrame({
            'close': [99, 100, 101],  # Crosses above 100
            'open': [98, 99, 100],
            'high': [99.5, 100.5, 101.5],
            'low': [98, 99, 100],
            'volume': [1000, 1000, 1000]
        }, index=dates)

        node = {"lhs": {"field": "close"}, "rhs": {"value": 100}}

        # First evaluation - no previous side
        result1, sides1, snapshot1 = alert_evaluator._evaluate_comparison(
            "crosses_above", node, data, {}, {}
        )
        assert result1 is False  # No trigger on first observation

        # Second evaluation - should trigger crossing
        cross_key = list(sides1.keys())[0]
        sides_state = {cross_key: "below"}  # Previous side was below

        result2, sides2, snapshot2 = alert_evaluator._evaluate_comparison(
            "crosses_above", node, data, {}, sides_state
        )
        assert result2 is True  # Should trigger crossing above

    def test_evaluate_rule_tree_and_logic(self, alert_evaluator, sample_market_data):
        """Test rule tree evaluation with AND logic."""
        rule = {
            "and": [
                {"gt": {"lhs": {"field": "close"}, "rhs": {"value": 100}}},
                {"lt": {"lhs": {"field": "close"}, "rhs": {"value": 120}}}
            ]
        }

        result, sides, snapshot = alert_evaluator._evaluate_rule_tree(
            rule, sample_market_data, {}, {}
        )

        # Last close should be around 109.5, so both conditions should be true
        assert result is True

    def test_evaluate_rule_tree_or_logic(self, alert_evaluator, sample_market_data):
        """Test rule tree evaluation with OR logic."""
        rule = {
            "or": [
                {"gt": {"lhs": {"field": "close"}, "rhs": {"value": 200}}},  # False
                {"gt": {"lhs": {"field": "close"}, "rhs": {"value": 100}}}   # True
            ]
        }

        result, sides, snapshot = alert_evaluator._evaluate_rule_tree(
            rule, sample_market_data, {}, {}
        )

        assert result is True  # Should be true because second condition is true

    def test_evaluate_rule_tree_not_logic(self, alert_evaluator, sample_market_data):
        """Test rule tree evaluation with NOT logic."""
        rule = {
            "not": {"gt": {"lhs": {"field": "close"}, "rhs": {"value": 200}}}
        }

        result, sides, snapshot = alert_evaluator._evaluate_rule_tree(
            rule, sample_market_data, {}, {}
        )

        assert result is True  # NOT (close > 200) should be true

    def test_apply_rearm_logic_no_rearm(self, alert_evaluator):
        """Test rearm logic when no rearm configuration is provided."""
        alert_config = AlertConfig(
            ticker="BTCUSDT", timeframe="1h", rule={}, rearm=None, options={}, notify={}
        )
        current_state = {"status": "ARMED"}

        result = alert_evaluator._apply_rearm_logic(alert_config, current_state, True, False)

        assert result.new_status == "TRIGGERED"
        assert result.should_rearm is False

    def test_apply_rearm_logic_armed_to_triggered(self, alert_evaluator):
        """Test rearm logic transition from ARMED to TRIGGERED."""
        alert_config = AlertConfig(
            ticker="BTCUSDT", timeframe="1h", rule={},
            rearm={"enabled": True}, options={}, notify={}
        )
        current_state = {"status": "ARMED"}

        result = alert_evaluator._apply_rearm_logic(alert_config, current_state, True, False)

        assert result.new_status == "TRIGGERED"
        assert result.should_rearm is False
        assert "last_triggered" in result.state_updates

    def test_apply_rearm_logic_triggered_to_armed(self, alert_evaluator):
        """Test rearm logic transition from TRIGGERED to ARMED."""
        alert_config = AlertConfig(
            ticker="BTCUSDT", timeframe="1h", rule={},
            rearm={"enabled": True}, options={}, notify={}
        )
        current_state = {"status": "TRIGGERED"}

        result = alert_evaluator._apply_rearm_logic(alert_config, current_state, False, True)

        assert result.new_status == "ARMED"
        assert result.should_rearm is True
        assert "last_rearmed" in result.state_updates

    def test_check_cooldown_no_cooldown(self, alert_evaluator):
        """Test cooldown check when no cooldown is configured."""
        current_state = {}
        rearm_config = {"cooldown_minutes": 0}

        result = alert_evaluator._check_cooldown(current_state, rearm_config)
        assert result is True

    def test_check_cooldown_active(self, alert_evaluator):
        """Test cooldown check when cooldown is still active."""
        now = datetime.now(timezone.utc)
        last_triggered = (now - timedelta(minutes=5)).isoformat()

        current_state = {"last_triggered": last_triggered}
        rearm_config = {"cooldown_minutes": 15}

        result = alert_evaluator._check_cooldown(current_state, rearm_config)
        assert result is False  # Should still be in cooldown

    def test_check_cooldown_expired(self, alert_evaluator):
        """Test cooldown check when cooldown has expired."""
        now = datetime.now(timezone.utc)
        last_triggered = (now - timedelta(minutes=20)).isoformat()

        current_state = {"last_triggered": last_triggered}
        rearm_config = {"cooldown_minutes": 15}

        result = alert_evaluator._check_cooldown(current_state, rearm_config)
        assert result is True  # Cooldown should have expired

    def test_check_persistence_bars_no_requirement(self, alert_evaluator):
        """Test persistence bars check when no persistence is required."""
        current_state = {}
        rearm_config = {"persistence_bars": 1}

        result = alert_evaluator._check_persistence_bars(current_state, rearm_config, True, False)
        assert result is True  # No persistence requirement

    def test_check_persistence_bars_insufficient(self, alert_evaluator):
        """Test persistence bars check when insufficient bars."""
        current_state = {"consecutive_trigger_bars": 1}
        rearm_config = {"persistence_bars": 3}

        result = alert_evaluator._check_persistence_bars(current_state, rearm_config, True, False)
        assert result is False  # Insufficient persistence

    def test_check_persistence_bars_sufficient(self, alert_evaluator):
        """Test persistence bars check when sufficient bars."""
        current_state = {"consecutive_trigger_bars": 2}
        rearm_config = {"persistence_bars": 3}

        # This would be the 3rd consecutive bar
        result = alert_evaluator._check_persistence_bars(current_state, rearm_config, True, False)
        assert result is True  # Sufficient persistence

    def test_update_persistence_counters_trigger(self, alert_evaluator):
        """Test updating persistence counters on trigger."""
        current_state = {"consecutive_trigger_bars": 1, "consecutive_rearm_bars": 2}

        updates = alert_evaluator._update_persistence_counters(current_state, True, False)

        assert updates["consecutive_trigger_bars"] == 2
        assert updates["consecutive_rearm_bars"] == 0

    def test_update_persistence_counters_rearm(self, alert_evaluator):
        """Test updating persistence counters on rearm."""
        current_state = {"consecutive_trigger_bars": 2, "consecutive_rearm_bars": 1}

        updates = alert_evaluator._update_persistence_counters(current_state, False, True)

        assert updates["consecutive_trigger_bars"] == 0
        assert updates["consecutive_rearm_bars"] == 2

    def test_update_persistence_counters_neither(self, alert_evaluator):
        """Test updating persistence counters when neither triggered nor rearmed."""
        current_state = {"consecutive_trigger_bars": 2, "consecutive_rearm_bars": 1}

        updates = alert_evaluator._update_persistence_counters(current_state, False, False)

        assert updates["consecutive_trigger_bars"] == 0
        assert updates["consecutive_rearm_bars"] == 0

    def test_calculate_atr(self, alert_evaluator):
        """Test ATR calculation."""
        # Create test data with known ATR
        dates = pd.date_range(start='2023-01-01', periods=20, freq='1D', tz='UTC')
        data = pd.DataFrame({
            'high': [102, 103, 104, 105, 106] * 4,
            'low': [98, 99, 100, 101, 102] * 4,
            'close': [100, 101, 102, 103, 104] * 4,
            'open': [99, 100, 101, 102, 103] * 4,
            'volume': [1000] * 20
        }, index=dates)

        atr = alert_evaluator._calculate_atr(data, period=14)

        assert atr is not None
        assert isinstance(atr, float)
        assert atr > 0

    def test_calculate_atr_insufficient_data(self, alert_evaluator):
        """Test ATR calculation with insufficient data."""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='1D', tz='UTC')
        data = pd.DataFrame({
            'high': [102, 103, 104, 105, 106],
            'low': [98, 99, 100, 101, 102],
            'close': [100, 101, 102, 103, 104],
            'open': [99, 100, 101, 102, 103],
            'volume': [1000] * 5
        }, index=dates)

        atr = alert_evaluator._calculate_atr(data, period=14)
        assert atr is None

    def test_validate_market_data_valid(self, alert_evaluator, sample_market_data):
        """Test market data validation with valid data."""
        alert_config = AlertConfig(
            ticker="BTCUSDT", timeframe="1h", rule={}, rearm=None, options={}, notify={}
        )

        result = alert_evaluator._validate_market_data(sample_market_data, alert_config, 50)
        assert result is True

    def test_validate_market_data_missing_columns(self, alert_evaluator):
        """Test market data validation with missing columns."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H', tz='UTC')
        data = pd.DataFrame({
            'close': [100 + i * 0.1 for i in range(100)],
            # Missing other required columns
        }, index=dates)

        alert_config = AlertConfig(
            ticker="BTCUSDT", timeframe="1h", rule={}, rearm=None, options={}, notify={}
        )

        result = alert_evaluator._validate_market_data(data, alert_config, 50)
        assert result is False

    def test_validate_market_data_insufficient_data(self, alert_evaluator):
        """Test market data validation with insufficient data."""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='1H', tz='UTC')
        data = pd.DataFrame({
            'open': [100] * 5,
            'high': [101] * 5,
            'low': [99] * 5,
            'close': [100.5] * 5,
            'volume': [1000] * 5
        }, index=dates)

        alert_config = AlertConfig(
            ticker="BTCUSDT", timeframe="1h", rule={}, rearm=None, options={}, notify={}
        )

        result = alert_evaluator._validate_market_data(data, alert_config, 100)
        assert result is False

    def test_validate_market_data_zero_prices(self, alert_evaluator):
        """Test market data validation with zero prices."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H', tz='UTC')
        data = pd.DataFrame({
            'open': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'close': [0] * 100,  # Zero prices
            'volume': [1000] * 100
        }, index=dates)

        alert_config = AlertConfig(
            ticker="BTCUSDT", timeframe="1h", rule={}, rearm=None, options={}, notify={}
        )

        result = alert_evaluator._validate_market_data(data, alert_config, 50)
        assert result is False

    def test_migrate_state_legacy(self, alert_evaluator):
        """Test state migration from legacy format."""
        legacy_state = {
            "is_armed": True,
            "some_old_field": "value"
        }

        migrated = alert_evaluator._migrate_state(legacy_state, "0.0")

        assert migrated["status"] == "ARMED"
        assert "is_armed" not in migrated
        assert migrated["version"] == "1.0"
        assert migrated["migrated_from"] == "0.0"

    @pytest.mark.asyncio
    async def test_fetch_market_data_success(self, alert_evaluator, mock_data_manager, sample_market_data):
        """Test successful market data fetching."""
        mock_data_manager.get_ohlcv.return_value = sample_market_data

        alert_config = AlertConfig(
            ticker="BTCUSDT", timeframe="1h", rule={}, rearm=None, options={}, notify={}
        )

        result = await alert_evaluator._fetch_market_data(alert_config)

        assert result is not None
        assert len(result) > 0
        mock_data_manager.get_ohlcv.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_market_data_failure(self, alert_evaluator, mock_data_manager):
        """Test market data fetching failure with retries."""
        mock_data_manager.get_ohlcv.side_effect = Exception("Network error")

        alert_config = AlertConfig(
            ticker="BTCUSDT", timeframe="1h", rule={}, rearm=None, options={}, notify={}
        )

        result = await alert_evaluator._fetch_market_data(alert_config)

        assert result is None
        assert mock_data_manager.get_ohlcv.call_count == 3  # Should retry 3 times


if __name__ == "__main__":
    pytest.main([__file__])