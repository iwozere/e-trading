"""
Indicator configuration manager for handling customizable indicator parameters.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class IndicatorConfig:
    """Configuration manager for indicator parameters."""

    def __init__(self, config_path: str = "config/indicators.json"):
        """
        Initialize the indicator configuration.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._current_preset = "default"
        self._custom_parameters = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                _logger.info(f"Loaded indicator configuration from {self.config_path}")
                return config
            else:
                _logger.warning(f"Configuration file {self.config_path} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            _logger.error(f"Error loading indicator configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file is not found."""
        return {
            "default_parameters": {
                "RSI": {"timeperiod": 14},
                "MACD": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
                "Bollinger_Bands": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2},
                "Stochastic": {"fastk_period": 14, "slowk_period": 3, "slowd_period": 3},
                "ADX": {"timeperiod": 14},
                "PLUS_DI": {"timeperiod": 14},
                "MINUS_DI": {"timeperiod": 14},
                "SMA_FAST": {"timeperiod": 50},
                "SMA_SLOW": {"timeperiod": 200},
                "EMA_FAST": {"timeperiod": 12},
                "EMA_SLOW": {"timeperiod": 26},
                "CCI": {"timeperiod": 14},
                "ROC": {"timeperiod": 10},
                "MFI": {"timeperiod": 14},
                "Williams_R": {"timeperiod": 14},
                "ATR": {"timeperiod": 14},
                "ADR": {"timeperiod": 14}
            },
            "indicator_mapping": {
                "SMA_50": "SMA_FAST",
                "SMA_200": "SMA_SLOW",
                "EMA_12": "EMA_FAST",
                "EMA_26": "EMA_SLOW",
                "WILLIAMS_R": "Williams_R",
                "BB_UPPER": "Bollinger_Bands",
                "BB_MIDDLE": "Bollinger_Bands",
                "BB_LOWER": "Bollinger_Bands",
                "STOCH_K": "Stochastic",
                "STOCH_D": "Stochastic"
            },
            "custom_presets": {}
        }

    def get_parameters(self, indicator: str) -> Dict[str, Any]:
        """
        Get parameters for a specific indicator.

        Args:
            indicator: Indicator name (e.g., 'RSI', 'SMA_FAST')

        Returns:
            Dictionary of parameters for the indicator
        """
        # Map old indicator names to new ones
        mapped_indicator = self.config.get("indicator_mapping", {}).get(indicator, indicator)

        # Get default parameters
        default_params = self.config.get("default_parameters", {}).get(mapped_indicator, {})

        # Apply current preset if available
        preset_params = {}
        if self._current_preset != "default" and self._current_preset in self.config.get("custom_presets", {}):
            preset_params = self.config["custom_presets"][self._current_preset].get(mapped_indicator, {})

        # Apply custom parameters if set
        custom_params = self._custom_parameters.get(mapped_indicator, {})

        # Merge parameters (custom > preset > default)
        final_params = default_params.copy()
        final_params.update(preset_params)
        final_params.update(custom_params)

        return final_params

    def set_preset(self, preset_name: str) -> bool:
        """
        Set the current preset.

        Args:
            preset_name: Name of the preset ('default', 'conservative', 'aggressive', 'day_trading')

        Returns:
            True if preset was set successfully, False otherwise
        """
        if preset_name == "default":
            self._current_preset = "default"
            _logger.info("Set indicator preset to default")
            return True

        available_presets = list(self.config.get("custom_presets", {}).keys())
        if preset_name in available_presets:
            self._current_preset = preset_name
            _logger.info(f"Set indicator preset to {preset_name}")
            return True
        else:
            _logger.warning(f"Preset '{preset_name}' not found. Available presets: {available_presets}")
            return False

    def get_current_preset(self) -> str:
        """Get the current preset name."""
        return self._current_preset

    def set_custom_parameter(self, indicator: str, parameter: str, value: Any) -> None:
        """
        Set a custom parameter for an indicator.

        Args:
            indicator: Indicator name
            parameter: Parameter name (e.g., 'timeperiod')
            value: Parameter value
        """
        mapped_indicator = self.config.get("indicator_mapping", {}).get(indicator, indicator)

        if mapped_indicator not in self._custom_parameters:
            self._custom_parameters[mapped_indicator] = {}

        self._custom_parameters[mapped_indicator][parameter] = value
        _logger.info(f"Set custom parameter {parameter}={value} for {indicator}")

    def clear_custom_parameters(self, indicator: Optional[str] = None) -> None:
        """
        Clear custom parameters.

        Args:
            indicator: Specific indicator to clear, or None to clear all
        """
        if indicator is None:
            self._custom_parameters.clear()
            _logger.info("Cleared all custom parameters")
        else:
            mapped_indicator = self.config.get("indicator_mapping", {}).get(indicator, indicator)
            if mapped_indicator in self._custom_parameters:
                del self._custom_parameters[mapped_indicator]
                _logger.info(f"Cleared custom parameters for {indicator}")

    def get_available_presets(self) -> list:
        """Get list of available presets."""
        return ["default"] + list(self.config.get("custom_presets", {}).keys())

    def get_preset_parameters(self, preset_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all parameters for a specific preset.

        Args:
            preset_name: Name of the preset

        Returns:
            Dictionary of indicator parameters for the preset
        """
        if preset_name == "default":
            return self.config.get("default_parameters", {})
        else:
            return self.config.get("custom_presets", {}).get(preset_name, {})

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
        _logger.info("Reloaded indicator configuration")


# Global instance
indicator_config = IndicatorConfig()
