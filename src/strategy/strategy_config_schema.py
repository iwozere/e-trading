"""
Strategy Configuration Schema

Pydantic v2 models for validating strategy config dicts before they reach
CustomStrategy or the scheduler.  Validates:
  - entry_logic.name and exit_logic.name exist in their respective registries
  - logic_params type-check against each mixin's default param types
  - position_size is in (0, 1]
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator

from src.strategy.entry.entry_mixin_factory import ENTRY_MIXIN_REGISTRY
from src.strategy.exit.exit_mixin_factory import EXIT_MIXIN_REGISTRY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_param_types(mixin_name: str, params: Dict[str, Any], registry: dict) -> None:
    """
    Type-check provided params against the mixin's declared defaults.

    Only keys that exist in the defaults are checked — extra/unknown keys are
    silently allowed (they may be legacy prefixed names like `e_rsi_period`).
    """
    mixin_class = registry[mixin_name]
    defaults: Dict[str, Any] = mixin_class.get_default_params()

    for key, value in params.items():
        if key not in defaults or defaults[key] is None:
            continue

        expected = type(defaults[key])

        if expected is bool:
            if not isinstance(value, bool):
                raise ValueError(f"Param '{key}' for {mixin_name} must be bool, got {type(value).__name__}")
        elif expected in (int, float):
            # Accept both int and float for numeric params, but not bool
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Param '{key}' for {mixin_name} must be numeric (int or float), got {type(value).__name__}"
                )
        elif expected is str:
            if not isinstance(value, str):
                raise ValueError(f"Param '{key}' for {mixin_name} must be str, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class MixinLogicConfig(BaseModel):
    """Config block for a single entry or exit mixin."""

    name: str
    logic_params: Dict[str, Any] | None = None
    params: Dict[str, Any] | None = None  # legacy alias for logic_params
    indicators: List[Dict[str, Any]] | None = None  # explicit indicator override

    def effective_params(self) -> Dict[str, Any]:
        """Return logic_params, falling back to params, then empty dict."""
        return self.logic_params or self.params or {}


# ---------------------------------------------------------------------------
# Top-level schema
# ---------------------------------------------------------------------------


class StrategyConfig(BaseModel):
    """
    Validated representation of a CustomStrategy config dict.

    Usage::

        from src.strategy.strategy_config_schema import validate_strategy_config

        cfg = validate_strategy_config({
            "entry_logic": {"name": "RSIBBEntryMixin", "logic_params": {"rsi_period": 14}},
            "exit_logic":  {"name": "FixedRatioExitMixin", "logic_params": {"take_profit": 0.05}},
            "position_size": 0.1,
        })
    """

    entry_logic: MixinLogicConfig
    exit_logic: MixinLogicConfig
    position_size: float = Field(default=0.1, gt=0.0, le=1.0)
    symbol: str = ""
    asset_type: str = "crypto"
    enable_database_logging: bool = False

    @field_validator("entry_logic")
    @classmethod
    def entry_mixin_must_exist(cls, v: MixinLogicConfig) -> MixinLogicConfig:
        if v.name not in ENTRY_MIXIN_REGISTRY:
            raise ValueError(f"Unknown entry mixin '{v.name}'. Available: {sorted(ENTRY_MIXIN_REGISTRY.keys())}")
        _check_param_types(v.name, v.effective_params(), ENTRY_MIXIN_REGISTRY)
        return v

    @field_validator("exit_logic")
    @classmethod
    def exit_mixin_must_exist(cls, v: MixinLogicConfig) -> MixinLogicConfig:
        if v.name not in EXIT_MIXIN_REGISTRY:
            raise ValueError(f"Unknown exit mixin '{v.name}'. Available: {sorted(EXIT_MIXIN_REGISTRY.keys())}")
        _check_param_types(v.name, v.effective_params(), EXIT_MIXIN_REGISTRY)
        return v

    def to_strategy_params(self) -> Dict[str, Any]:
        """Return the dict that can be passed as strategy_config to CustomStrategy."""
        return self.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_strategy_config(config: Dict[str, Any]) -> StrategyConfig:
    """
    Parse and validate a raw strategy config dict.

    Args:
        config: Raw dict (e.g. from a JSON payload or YAML file).

    Returns:
        Validated StrategyConfig instance.

    Raises:
        pydantic.ValidationError: If any field is invalid.
    """
    return StrategyConfig.model_validate(config)
