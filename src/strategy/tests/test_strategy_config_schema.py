"""
Unit tests for StrategyConfig Pydantic schema.

Covers:
- Valid configs parse without error
- Unknown mixin names raise ValidationError
- Invalid position_size raises ValidationError
- Wrong param types (str instead of int) raise ValidationError
- Unknown/extra params are silently allowed
- to_strategy_params() returns a plain dict
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy.strategy_config_schema import StrategyConfig, validate_strategy_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config(**overrides) -> dict:
    base = {
        "entry_logic": {
            "name": "RSIBBEntryMixin",
            "logic_params": {"rsi_period": 14, "rsi_oversold": 30},
        },
        "exit_logic": {
            "name": "FixedRatioExitMixin",
            "logic_params": {"take_profit": 0.05, "stop_loss": 0.03},
        },
        "position_size": 0.1,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------


class TestValidConfig:
    def test_minimal_config_parses(self):
        cfg = validate_strategy_config(_minimal_config())
        assert isinstance(cfg, StrategyConfig)
        assert cfg.entry_logic.name == "RSIBBEntryMixin"
        assert cfg.exit_logic.name == "FixedRatioExitMixin"
        assert cfg.position_size == pytest.approx(0.1)

    def test_position_size_exactly_one_is_valid(self):
        cfg = validate_strategy_config(_minimal_config(position_size=1.0))
        assert cfg.position_size == pytest.approx(1.0)

    def test_defaults_are_applied(self):
        cfg = validate_strategy_config(_minimal_config())
        assert cfg.symbol == ""
        assert cfg.asset_type == "crypto"
        assert cfg.enable_database_logging is False

    def test_to_strategy_params_returns_dict(self):
        cfg = validate_strategy_config(_minimal_config())
        params = cfg.to_strategy_params()
        assert isinstance(params, dict)
        assert "entry_logic" in params
        assert "exit_logic" in params

    def test_all_registered_entry_mixins_accepted(self):
        from src.strategy.entry.entry_mixin_factory import ENTRY_MIXIN_REGISTRY

        for name in ENTRY_MIXIN_REGISTRY:
            cfg = validate_strategy_config(_minimal_config(entry_logic={"name": name}))
            assert cfg.entry_logic.name == name

    def test_all_registered_exit_mixins_accepted(self):
        from src.strategy.exit.exit_mixin_factory import EXIT_MIXIN_REGISTRY

        for name in EXIT_MIXIN_REGISTRY:
            cfg = validate_strategy_config(_minimal_config(exit_logic={"name": name}))
            assert cfg.exit_logic.name == name

    def test_legacy_params_key_accepted(self):
        cfg = validate_strategy_config(
            _minimal_config(
                entry_logic={"name": "RSIBBEntryMixin", "params": {"rsi_period": 14}},
            )
        )
        assert cfg.entry_logic.effective_params()["rsi_period"] == 14

    def test_extra_unknown_params_are_allowed(self):
        """Unknown param keys are not blocked — they may be legacy e_-prefixed names."""
        cfg = validate_strategy_config(
            _minimal_config(
                entry_logic={
                    "name": "RSIBBEntryMixin",
                    "logic_params": {"e_rsi_period": 14, "rsi_oversold": 30},
                }
            )
        )
        assert cfg.entry_logic.name == "RSIBBEntryMixin"


# ---------------------------------------------------------------------------
# Entry logic errors
# ---------------------------------------------------------------------------


class TestEntryLogicErrors:
    def test_unknown_entry_mixin_raises(self):
        with pytest.raises(ValidationError, match="Unknown entry mixin"):
            validate_strategy_config(_minimal_config(entry_logic={"name": "DoesNotExistMixin"}))

    def test_entry_param_wrong_type_str_for_int(self):
        with pytest.raises(ValidationError, match="numeric"):
            validate_strategy_config(
                _minimal_config(
                    entry_logic={
                        "name": "RSIBBEntryMixin",
                        "logic_params": {"rsi_period": "fourteen"},  # must be int/float
                    }
                )
            )

    def test_entry_param_wrong_type_int_for_bool(self):
        with pytest.raises(ValidationError, match="bool"):
            validate_strategy_config(
                _minimal_config(
                    entry_logic={
                        "name": "RSIBBEntryMixin",
                        "logic_params": {"use_bb_touch": 1},  # must be bool, not int
                    }
                )
            )

    def test_entry_param_wrong_type_str_for_float(self):
        with pytest.raises(ValidationError, match="numeric"):
            validate_strategy_config(
                _minimal_config(
                    entry_logic={
                        "name": "RSIBBEntryMixin",
                        "logic_params": {"bb_dev": "two"},  # must be numeric
                    }
                )
            )


# ---------------------------------------------------------------------------
# Exit logic errors
# ---------------------------------------------------------------------------


class TestExitLogicErrors:
    def test_unknown_exit_mixin_raises(self):
        with pytest.raises(ValidationError, match="Unknown exit mixin"):
            validate_strategy_config(_minimal_config(exit_logic={"name": "GhostExitMixin"}))

    def test_exit_param_wrong_type_raises(self):
        with pytest.raises(ValidationError, match="numeric"):
            validate_strategy_config(
                _minimal_config(
                    exit_logic={
                        "name": "FixedRatioExitMixin",
                        "logic_params": {"take_profit": "five_percent"},  # must be float
                    }
                )
            )


# ---------------------------------------------------------------------------
# position_size errors
# ---------------------------------------------------------------------------


class TestPositionSizeErrors:
    def test_position_size_zero_raises(self):
        with pytest.raises(ValidationError):
            validate_strategy_config(_minimal_config(position_size=0.0))

    def test_position_size_negative_raises(self):
        with pytest.raises(ValidationError):
            validate_strategy_config(_minimal_config(position_size=-0.1))

    def test_position_size_over_one_raises(self):
        with pytest.raises(ValidationError):
            validate_strategy_config(_minimal_config(position_size=1.1))
