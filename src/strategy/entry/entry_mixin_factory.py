# entry_mixin_factory.py

from typing import Any, Dict, Optional

from src.strategy.entry.rsi_bb_entry_mixin import RSIBBEntryMixin
from src.strategy.entry.rsi_or_bb_entry_mixin import RSIOrBBEntryMixin
from src.strategy.entry.rsi_bb_volume_entry_mixin import RSIBBVolumeEntryMixin
from src.strategy.entry.rsi_ichimoku_entry_mixin import RSIIchimokuEntryMixin
from src.strategy.entry.bb_volume_supertrend_entry_mixin import BBVolumeSupertrendEntryMixin
from src.strategy.entry.rsi_volume_supertrend_entry_mixin import RSIVolumeSupertrendEntryMixin
from src.strategy.entry.eom_breakout_entry_mixin import EOMBreakoutEntryMixin
from src.strategy.entry.eom_macd_breakout_entry_mixin import EOMMAcdBreakoutEntryMixin
from src.strategy.entry.eom_pullback_entry_mixin import EOMPullbackEntryMixin
from src.strategy.entry.hmm_lstm_entry_mixin import HMMLSTMEntryMixin

# Registry of all available entry mixins
ENTRY_MIXIN_REGISTRY = {
    "RSIBBEntryMixin": RSIBBEntryMixin,
    "RSIIchimokuEntryMixin": RSIIchimokuEntryMixin,
    "RSIOrBBEntryMixin": RSIOrBBEntryMixin,
    "RSIBBVolumeEntryMixin": RSIBBVolumeEntryMixin,
    "RSIVolumeSupertrendEntryMixin": RSIVolumeSupertrendEntryMixin,
    "BBVolumeSuperTrendEntryMixin": BBVolumeSupertrendEntryMixin,
    "EOMBreakoutEntryMixin": EOMBreakoutEntryMixin,
    "EOMMAcdBreakoutEntryMixin": EOMMAcdBreakoutEntryMixin,
    "EOMPullbackEntryMixin": EOMPullbackEntryMixin,
    # "HMMLSTMEntryMixin": HMMLSTMEntryMixin,  # Excluded due to model architectural mismatches/loading errors
}


def get_entry_mixin(mixin_name: str, params: Optional[Dict[str, Any]] = None):
    """Factory for creating instances of entry mixins."""
    if mixin_name not in ENTRY_MIXIN_REGISTRY:
        available_mixins = list(ENTRY_MIXIN_REGISTRY.keys())
        raise ValueError(
            f"Unknown entry mixin: {mixin_name}. Available: {available_mixins}"
        )

    mixin_class = ENTRY_MIXIN_REGISTRY[mixin_name]
    return mixin_class(params=params)


def get_entry_mixin_from_config(config: Dict[str, Any]):
    """Creating a mixin from a full configuration."""
    if "name" not in config:
        raise ValueError("Config must contain 'name' field")

    mixin_name = config["name"]
    params = config.get("params", {})

    return get_entry_mixin(mixin_name, params)


def list_available_entry_mixins():
    """Returns a list of available entry mixins"""
    return list(ENTRY_MIXIN_REGISTRY.keys())


def get_entry_mixin_default_params(mixin_name: str) -> Dict[str, Any]:
    """Get default parameters for the mixin."""
    if mixin_name not in ENTRY_MIXIN_REGISTRY:
        raise ValueError(f"Unknown entry mixin: {mixin_name}")

    mixin_class = ENTRY_MIXIN_REGISTRY[mixin_name]
    temp_instance = mixin_class()
    return temp_instance.get_default_params()


def validate_entry_mixin_params(mixin_name: str, params: Dict[str, Any]) -> bool:
    """Validation of parameters for the mixin."""
    if mixin_name not in ENTRY_MIXIN_REGISTRY:
        raise ValueError(f"Unknown entry mixin: {mixin_name}")

    mixin_class = ENTRY_MIXIN_REGISTRY[mixin_name]
    try:
        mixin_class(params=params)
        return True
    except Exception as e:
        raise ValueError(f"Invalid parameters for {mixin_name}: {e}")
