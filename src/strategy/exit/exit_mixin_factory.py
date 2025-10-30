# exit_mixin_factory.py - Improved version with a new approach

from typing import Any, Dict, Optional

from src.strategy.exit.atr_exit_mixin import ATRExitMixin
from src.strategy.exit.advanced_atr_exit_mixin import AdvancedATRExitMixin
from src.strategy.exit.simple_atr_exit_mixin import SimpleATRExitMixin
from src.strategy.exit.fixed_ratio_exit_mixin import FixedRatioExitMixin
from src.strategy.exit.ma_crossover_exit_mixin import MACrossoverExitMixin
from src.strategy.exit.rsi_bb_exit_mixin import RSIBBExitMixin
from src.strategy.exit.rsi_or_bb_exit_mixin import RSIOrBBExitMixin
from src.strategy.exit.time_based_exit_mixin import TimeBasedExitMixin
from src.strategy.exit.trailing_stop_exit_mixin import TrailingStopExitMixin

# Import other mixins...

# Registry of all available exit mixins
EXIT_MIXIN_REGISTRY = {
    "ATRExitMixin": ATRExitMixin,
    "AdvancedATRExitMixin": AdvancedATRExitMixin,
    "SimpleATRExitMixin": SimpleATRExitMixin,
    "FixedRatioExitMixin": FixedRatioExitMixin,
    "MACrossoverExitMixin": MACrossoverExitMixin,
    "RSIBBExitMixin": RSIBBExitMixin,
    "RSIOrBBExitMixin": RSIOrBBExitMixin,
    "TimeBasedExitMixin": TimeBasedExitMixin,
    "TrailingStopExitMixin": TrailingStopExitMixin,
    # Add other mixins...
}


def get_exit_mixin(mixin_name: str, params: Optional[Dict[str, Any]] = None):
    """
    Factory for creating instances of exit mixins with a new approach

    Args:
        mixin_name: Name of the mixin
        params: Dictionary of parameters for the mixin

    Returns:
        Instance of the mixin class

    Example:
        # Creating with default parameters
        mixin = get_exit_mixin("AdvancedATRExitMixin")

        # Creating with custom parameters
        params = {'k_init': 2.5, 'k_run': 2.0}
        mixin = get_exit_mixin("AdvancedATRExitMixin", params)
    """
    if mixin_name not in EXIT_MIXIN_REGISTRY:
        available_mixins = list(EXIT_MIXIN_REGISTRY.keys())
        raise ValueError(
            f"Unknown exit mixin: {mixin_name}. Available: {available_mixins}"
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
            'name': 'AdvancedATRExitMixin',
            'params': {
                'k_init': 2.5,
                'k_run': 2.0
            }
        }
        mixin = get_exit_mixin_from_config(config)
    """
    if "name" not in config:
        raise ValueError("Config must contain 'name' field")

    mixin_name = config["name"]
    params = config.get("params", {})

    return get_exit_mixin(mixin_name, params)


def list_available_exit_mixins():
    """Returns a list of available exit mixins"""
    return list(EXIT_MIXIN_REGISTRY.keys())


def get_exit_mixin_default_params(mixin_name: str) -> Dict[str, Any]:
    """
    Get default parameters for the mixin

    Args:
        mixin_name: Name of the mixin

    Returns:
        Dictionary of default parameters
    """
    if mixin_name not in EXIT_MIXIN_REGISTRY:
        raise ValueError(f"Unknown exit mixin: {mixin_name}")

    mixin_class = EXIT_MIXIN_REGISTRY[mixin_name]

    # Create a temporary instance to get default parameters
    temp_instance = mixin_class()
    return temp_instance.get_default_params()


def validate_exit_mixin_params(mixin_name: str, params: Dict[str, Any]) -> bool:
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
        raise ValueError(f"Unknown exit mixin: {mixin_name}")

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
