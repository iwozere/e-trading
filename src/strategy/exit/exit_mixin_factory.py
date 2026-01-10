# exit_mixin_factory.py

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
from src.strategy.exit.eom_breakdown_exit_mixin import EOMBreakdownExitMixin
from src.strategy.exit.eom_macd_breakdown_exit_mixin import EOMMAcdBreakdownExitMixin
from src.strategy.exit.eom_rejection_exit_mixin import EOMRejectionExitMixin
from src.strategy.exit.multi_level_atr_exit_mixin import MultiLevelAtrExitMixin

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
    "EOMBreakdownExitMixin": EOMBreakdownExitMixin,
    "EOMMAcdBreakdownExitMixin": EOMMAcdBreakdownExitMixin,
    "EOMRejectionExitMixin": EOMRejectionExitMixin,
    "MultiLevelAtrExitMixin": MultiLevelAtrExitMixin,
}


def get_exit_mixin(mixin_name: str, params: Optional[Dict[str, Any]] = None):
    """Factory for creating instances of exit mixins."""
    if mixin_name not in EXIT_MIXIN_REGISTRY:
        available_mixins = list(EXIT_MIXIN_REGISTRY.keys())
        raise ValueError(
            f"Unknown exit mixin: {mixin_name}. Available: {available_mixins}"
        )

    mixin_class = EXIT_MIXIN_REGISTRY[mixin_name]
    return mixin_class(params=params)


def get_exit_mixin_from_config(config: Dict[str, Any]):
    """Creating a mixin from a full configuration."""
    if "name" not in config:
        raise ValueError("Config must contain 'name' field")

    mixin_name = config["name"]
    params = config.get("params", {})

    return get_exit_mixin(mixin_name, params)


def list_available_exit_mixins():
    """Returns a list of available exit mixins"""
    return list(EXIT_MIXIN_REGISTRY.keys())


def get_exit_mixin_default_params(mixin_name: str) -> Dict[str, Any]:
    """Get default parameters for the mixin."""
    if mixin_name not in EXIT_MIXIN_REGISTRY:
        raise ValueError(f"Unknown exit mixin: {mixin_name}")

    mixin_class = EXIT_MIXIN_REGISTRY[mixin_name]
    temp_instance = mixin_class()
    return temp_instance.get_default_params()


def validate_exit_mixin_params(mixin_name: str, params: Dict[str, Any]) -> bool:
    """Validation of parameters for the mixin."""
    if mixin_name not in EXIT_MIXIN_REGISTRY:
        raise ValueError(f"Unknown exit mixin: {mixin_name}")

    mixin_class = EXIT_MIXIN_REGISTRY[mixin_name]
    try:
        mixin_class(params=params)
        return True
    except Exception as e:
        raise ValueError(f"Invalid parameters for {mixin_name}: {e}")
