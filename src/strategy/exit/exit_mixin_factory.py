# exit_mixin_factory.py - Improved version with a new approach

from typing import Any, Dict, Optional

from src.exit.atr_exit_mixin import ATRExitMixin
from src.exit.fixed_ratio_exit_mixin import FixedRatioExitMixin
from src.exit.ma_crossover_exit_mixin import MACrossoverExitMixin
from src.exit.rsi_bb_exit_mixin import RSIBBExitMixin
from src.exit.time_based_exit_mixin import TimeBasedExitMixin
from src.exit.trailing_stop_exit_mixin import TrailingStopExitMixin

# Import other mixins...

# Registry of all available entry mixins
EXIT_MIXIN_REGISTRY = {
    "ATRExitMixin": ATRExitMixin,  # 1
    "FixedRatioExitMixin": FixedRatioExitMixin,
    "MACrossoverExitMixin": MACrossoverExitMixin,
    "RSIBBExitMixin": RSIBBExitMixin,
    "TimeBasedExitMixin": TimeBasedExitMixin,
    "TrailingStopExitMixin": TrailingStopExitMixin,
    # Add other mixins...
}


def get_exit_mixin(mixin_name: str, params: Optional[Dict[str, Any]] = None):
    """
    Factory for creating instances of entry mixins with a new approach

    Args:
        mixin_name: Name of the mixin
        params: Dictionary of parameters for the mixin

    Returns:
        Instance of the mixin class

    Example:
        # Creating with default parameters
        mixin = get_entry_mixin("RSIBBEntryMixin")

        # Creating with custom parameters
        params = {'rsi_period': 21, 'bb_period': 15}
        mixin = get_entry_mixin("RSIBBEntryMixin", params)
    """
    if mixin_name not in EXIT_MIXIN_REGISTRY:
        available_mixins = list(EXIT_MIXIN_REGISTRY.keys())
        raise ValueError(
            f"Unknown entry mixin: {mixin_name}. Available: {available_mixins}"
        )

    mixin_class = EXIT_MIXIN_REGISTRY[mixin_name]
    return mixin_class(params=params)


def get_exit_mixin_from_config(config: Dict[str, Any]):
    """
    Creating a mixin from a full configuration

    Args:
        config: Full configuration of the mixin
                Must contain 'name' and optionally 'params'

    Returns:
        Instance of the mixin class

    Example:
        config = {
            'name': 'RSIBBEntryMixin',
            'params': {
                'rsi_period': 21,
                'bb_period': 15
            }
        }
        mixin = get_entry_mixin_from_config(config)
    """
    if "name" not in config:
        raise ValueError("Config must contain 'name' field")

    mixin_name = config["name"]
    params = config.get("params", {})

    return get_exit_mixin(mixin_name, params)


def list_available_entry_mixins():
    """Returns a list of available entry mixins"""
    return list(EXIT_MIXIN_REGISTRY.keys())


def get_entry_mixin_default_params(mixin_name: str) -> Dict[str, Any]:
    """
    Get default parameters for the mixin

    Args:
        mixin_name: Name of the mixin

    Returns:
        Dictionary of default parameters
    """
    if mixin_name not in EXIT_MIXIN_REGISTRY:
        raise ValueError(f"Unknown entry mixin: {mixin_name}")

    mixin_class = EXIT_MIXIN_REGISTRY[mixin_name]

    # Create a temporary instance to get default parameters
    temp_instance = mixin_class()
    return temp_instance.get_default_params()


def validate_entry_mixin_params(mixin_name: str, params: Dict[str, Any]) -> bool:
    """
    Validation of parameters for the mixin

    Args:
        mixin_name: Name of the mixin
        params: Parameters for validation

    Returns:
        True if parameters are valid

    Raises:
        ValueError: If parameters are invalid
    """
    if mixin_name not in EXIT_MIXIN_REGISTRY:
        raise ValueError(f"Unknown entry mixin: {mixin_name}")

    mixin_class = EXIT_MIXIN_REGISTRY[mixin_name]

    # Create a temporary instance for validation
    try:
        mixin_class(params=params)
        return True
    except ValueError as e:
        raise ValueError(f"Invalid parameters for {mixin_name}: {e}")


# Convenient functions for working with configurations
def create_exit_mixin_config(mixin_name: str, **params) -> Dict[str, Any]:
    """
    Creating a configuration for the mixin

    Args:
        mixin_name: Name

    Returns:
        Dictionary of configuration
    """
    return {"name": mixin_name, "params": params}
