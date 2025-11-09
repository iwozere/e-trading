"""
Comprehensive configuration validation tests.

Tests parameter validation, preset management, and configuration loading.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.indicators.config_manager import UnifiedConfigManager


class TestConfigurationValidation:
    """Test configuration validation and parameter management."""

    def test_parameter_validation_rsi(self):
        """Test RSI parameter validation."""
        config_manager = UnifiedConfigManager()

        # Valid parameters should return empty list (no errors)
        errors = config_manager.validate_parameters("rsi", {"timeperiod": 14})
        assert len(errors) == 0

        errors = config_manager.validate_parameters("rsi", {"timeperiod": 21})
        assert len(errors) == 0

        # Invalid parameters should return error list
        errors = config_manager.validate_parameters("rsi", {"timeperiod": 0})
        assert len(errors) > 0

        errors = config_manager.validate_parameters("rsi", {"timeperiod": -1})
        assert len(errors) > 0

    def test_parameter_validation_macd(self):
        """Test MACD parameter validation."""
        config_manager = UnifiedConfigManager()

        # Valid parameters should return empty list (no errors)
        valid_params = {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}
        errors = config_manager.validate_parameters("macd", valid_params)
        assert len(errors) == 0

        # Invalid parameters - unknown parameter should return errors
        invalid_params = {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9, "unknown_param": 5}
        errors = config_manager.validate_parameters("macd", invalid_params)
        assert len(errors) > 0

    def test_preset_loading(self):
        """Test preset configuration loading."""
        mock_config = {
            "presets": {
                "conservative": {
                    "description": "Conservative settings",
                    "parameters": {
                        "rsi": {"timeperiod": 21},
                        "macd": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}
                    }
                }
            }
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
            with patch("pathlib.Path.exists", return_value=True):
                config_manager = UnifiedConfigManager()

                presets = config_manager.get_available_presets()
                assert "conservative" in presets

                config_manager.set_preset("conservative")
                rsi_params = config_manager.get_parameters("rsi")
                assert rsi_params.get("timeperiod") == 21

    def test_runtime_override_precedence(self):
        """Test runtime overrides take precedence over presets."""
        config_manager = UnifiedConfigManager()

        # Set base parameters
        base_params = config_manager.get_parameters("rsi")
        base_period = base_params.get("timeperiod", 14)

        # Set runtime override
        config_manager.set_parameter_override("rsi", "timeperiod", 30)

        # Override should take precedence
        override_params = config_manager.get_parameters("rsi")
        assert override_params.get("timeperiod") == 30

        # Clear overrides
        config_manager.clear_parameter_overrides()
        cleared_params = config_manager.get_parameters("rsi")
        assert cleared_params.get("timeperiod") == base_period


if __name__ == '__main__':
    pytest.main([__file__, '-v'])