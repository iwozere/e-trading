"""
Unified configuration management system for indicators.

This module consolidates configuration functionality from the removed src/common/indicator_config.py
and provides a unified interface for managing indicator parameters, presets, and mappings.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field

from src.indicators.registry import INDICATOR_META, get_canonical_name, get_indicator_meta
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class PresetConfig:
    """Configuration for an indicator preset."""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]  # indicator_name -> parameters
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedConfigManager:
    """Unified configuration manager for all indicator parameters and presets."""

    def __init__(self, config_path: str = "config/indicators.json"):
        """
        Initialize the unified configuration manager.

        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._current_preset = "default"
        self._runtime_overrides = {}
        self._presets = self._load_presets()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                _logger.info("Loaded indicator configuration from %s", self.config_path)
                return config
            else:
                _logger.warning("Configuration file %s not found, using defaults", self.config_path)
                return self._get_default_config()
        except Exception as e:
            _logger.exception("Error loading indicator configuration:")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration based on registry metadata."""
        default_config = {
            "version": "2.0",
            "default_parameters": {},
            "presets": {
                "default": {
                    "description": "Default parameters from registry",
                    "parameters": {}
                },
                "conservative": {
                    "description": "Conservative trading parameters",
                    "parameters": {
                        "rsi": {"timeperiod": 21},
                        "macd": {"fastperiod": 15, "slowperiod": 30, "signalperiod": 12},
                        "bbands": {"timeperiod": 25, "nbdevup": 2.5, "nbdevdn": 2.5},
                        "stoch": {"fastk_period": 21, "slowk_period": 5, "slowd_period": 5}
                    }
                },
                "aggressive": {
                    "description": "Aggressive trading parameters",
                    "parameters": {
                        "rsi": {"timeperiod": 7},
                        "macd": {"fastperiod": 8, "slowperiod": 17, "signalperiod": 5},
                        "bbands": {"timeperiod": 15, "nbdevup": 1.5, "nbdevdn": 1.5},
                        "stoch": {"fastk_period": 7, "slowk_period": 2, "slowd_period": 2}
                    }
                },
                "day_trading": {
                    "description": "Day trading optimized parameters",
                    "parameters": {
                        "rsi": {"timeperiod": 5},
                        "macd": {"fastperiod": 5, "slowperiod": 13, "signalperiod": 3},
                        "bbands": {"timeperiod": 10, "nbdevup": 1.8, "nbdevdn": 1.8},
                        "stoch": {"fastk_period": 5, "slowk_period": 1, "slowd_period": 1}
                    }
                }
            },
            "legacy_mappings": {
                # Legacy indicator name mappings
                "RSI": "rsi",
                "MACD": "macd",
                "MACD_SIGNAL": "macd",
                "MACD_HISTOGRAM": "macd",
                "BB_UPPER": "bbands",
                "BB_MIDDLE": "bbands",
                "BB_LOWER": "bbands",
                "SMA_FAST": "sma",
                "SMA_SLOW": "sma",
                "SMA_50": "sma",
                "SMA_200": "sma",
                "EMA_FAST": "ema",
                "EMA_SLOW": "ema",
                "EMA_12": "ema",
                "EMA_26": "ema",
                "STOCH_K": "stoch",
                "STOCH_D": "stoch",
                "WILLIAMS_R": "williams_r",
                "ADX": "adx",
                "PLUS_DI": "plus_di",
                "MINUS_DI": "minus_di",
                "ATR": "atr",
                "CCI": "cci",
                "ROC": "roc",
                "MFI": "mfi",
                "OBV": "obv",
                "ADR": "adr"
            }
        }

        # Add default parameters from registry
        for indicator_name, meta in INDICATOR_META.items():
            if meta.defaults:
                default_config["default_parameters"][indicator_name] = meta.defaults

        return default_config

    def _load_presets(self) -> Dict[str, PresetConfig]:
        """Load presets from configuration."""
        presets = {}
        preset_configs = self.config.get("presets", {})

        for preset_name, preset_data in preset_configs.items():
            presets[preset_name] = PresetConfig(
                name=preset_name,
                description=preset_data.get("description", ""),
                parameters=preset_data.get("parameters", {}),
                metadata=preset_data.get("metadata", {})
            )

        return presets

    def get_parameters(self, indicator: str, preset: Optional[str] = None) -> Dict[str, Any]:
        """
        Get parameters for a specific indicator.

        Args:
            indicator: Indicator name (canonical or legacy)
            preset: Preset name to use (defaults to current preset)

        Returns:
            Dictionary of parameters for the indicator
        """
        # Get canonical name
        canonical_name = get_canonical_name(indicator)

        # Get metadata for defaults
        meta = get_indicator_meta(canonical_name)
        if not meta:
            _logger.warning("Unknown indicator: %s", indicator)
            return {}

        # Start with registry defaults
        final_params = meta.defaults.copy() if meta.defaults else {}

        # Apply global defaults from config
        global_defaults = self.config.get("default_parameters", {}).get(canonical_name, {})
        final_params.update(global_defaults)

        # Apply preset parameters
        preset_name = preset or self._current_preset
        if preset_name in self._presets:
            preset_params = self._presets[preset_name].parameters.get(canonical_name, {})
            final_params.update(preset_params)

        # Apply runtime overrides
        runtime_params = self._runtime_overrides.get(canonical_name, {})
        final_params.update(runtime_params)

        return final_params

    def set_preset(self, preset_name: str) -> bool:
        """
        Set the current preset.

        Args:
            preset_name: Name of the preset to activate

        Returns:
            True if preset was set successfully, False otherwise
        """
        if preset_name in self._presets:
            self._current_preset = preset_name
            _logger.info("Set current preset to: %s", preset_name)
            return True
        else:
            available_presets = list(self._presets.keys())
            _logger.warning("Preset '%s' not found. Available presets: %s", preset_name, available_presets)
            return False

    def get_current_preset(self) -> str:
        """Get the current preset name."""
        return self._current_preset

    def get_available_presets(self) -> List[str]:
        """Get list of available presets."""
        return list(self._presets.keys())

    def get_preset_info(self, preset_name: str) -> Optional[PresetConfig]:
        """Get information about a specific preset."""
        return self._presets.get(preset_name)

    def set_parameter_override(self, indicator: str, parameter: str, value: Any) -> None:
        """
        Set a runtime parameter override for an indicator.

        Args:
            indicator: Indicator name (canonical or legacy)
            parameter: Parameter name
            value: Parameter value
        """
        canonical_name = get_canonical_name(indicator)

        if canonical_name not in self._runtime_overrides:
            self._runtime_overrides[canonical_name] = {}

        self._runtime_overrides[canonical_name][parameter] = value
        _logger.debug("Set parameter override: %s.%s = %s", canonical_name, parameter, value)

    def clear_parameter_overrides(self, indicator: Optional[str] = None) -> None:
        """
        Clear runtime parameter overrides.

        Args:
            indicator: Specific indicator to clear, or None to clear all
        """
        if indicator is None:
            self._runtime_overrides.clear()
            _logger.debug("Cleared all parameter overrides")
        else:
            canonical_name = get_canonical_name(indicator)
            if canonical_name in self._runtime_overrides:
                del self._runtime_overrides[canonical_name]
                _logger.debug("Cleared parameter overrides for: %s", canonical_name)

    def create_preset(self, name: str, description: str, parameters: Dict[str, Dict[str, Any]]) -> bool:
        """
        Create a new preset.

        Args:
            name: Preset name
            description: Preset description
            parameters: Dictionary of indicator parameters

        Returns:
            True if preset was created successfully
        """
        try:
            # Convert legacy names to canonical names
            canonical_parameters = {}
            for indicator, params in parameters.items():
                canonical_name = get_canonical_name(indicator)
                canonical_parameters[canonical_name] = params

            preset = PresetConfig(
                name=name,
                description=description,
                parameters=canonical_parameters
            )

            self._presets[name] = preset
            _logger.info("Created new preset: %s", name)
            return True
        except Exception as e:
            _logger.error("Error creating preset %s: %s", name, e)
            return False

    def save_config(self, path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.

        Args:
            path: Optional path to save to (defaults to original config path)

        Returns:
            True if saved successfully
        """
        try:
            save_path = path or self.config_path

            # Build config to save
            config_to_save = {
                "version": "2.0",
                "default_parameters": self.config.get("default_parameters", {}),
                "presets": {},
                "legacy_mappings": self.config.get("legacy_mappings", {})
            }

            # Add presets
            for preset_name, preset in self._presets.items():
                config_to_save["presets"][preset_name] = {
                    "description": preset.description,
                    "parameters": preset.parameters,
                    "metadata": preset.metadata
                }

            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # Save to file
            with open(save_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)

            _logger.info("Saved configuration to: %s", save_path)
            return True
        except Exception as e:
            _logger.exception("Error saving configuration:")
            return False

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
        self._presets = self._load_presets()
        _logger.info("Reloaded configuration from: %s", self.config_path)

    def validate_parameters(self, indicator: str, parameters: Dict[str, Any]) -> List[str]:
        """
        Validate parameters for an indicator.

        Args:
            indicator: Indicator name
            parameters: Parameters to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        canonical_name = get_canonical_name(indicator)
        meta = get_indicator_meta(canonical_name)

        if not meta:
            errors.append(f"Unknown indicator: {indicator}")
            return errors

        # Get expected parameters from defaults
        expected_params = set(meta.defaults.keys()) if meta.defaults else set()
        provided_params = set(parameters.keys())

        # Check for unexpected parameters
        unexpected = provided_params - expected_params
        if unexpected:
            errors.append(f"Unexpected parameters for {indicator}: {list(unexpected)}")

        # Validate parameter types and ranges (basic validation)
        for param, value in parameters.items():
            if param in expected_params and meta.defaults:
                expected_type = type(meta.defaults[param])
                if not isinstance(value, expected_type):
                    errors.append(f"Parameter {param} for {indicator} should be {expected_type.__name__}, got {type(value).__name__}")

                # Basic range validation for common parameters
                if param == "timeperiod" and isinstance(value, int) and value <= 0:
                    errors.append(f"Parameter {param} for {indicator} must be positive")

        return errors

    def get_legacy_mapping(self, legacy_name: str) -> Optional[str]:
        """Get canonical name for a legacy indicator name."""
        return self.config.get("legacy_mappings", {}).get(legacy_name)

    def get_config_info(self) -> Dict[str, Any]:
        """Get information about the current configuration."""
        return {
            "config_path": self.config_path,
            "current_preset": self._current_preset,
            "available_presets": list(self._presets.keys()),
            "runtime_overrides": len(self._runtime_overrides),
            "total_indicators": len(INDICATOR_META),
            "version": self.config.get("version", "1.0")
        }


# Global instance for easy access
_config_manager = None

def get_config_manager() -> UnifiedConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = UnifiedConfigManager()
    return _config_manager