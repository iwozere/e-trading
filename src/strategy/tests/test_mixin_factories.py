"""
Unit tests for entry/exit mixin factories and CustomStrategy config builder.

Covers:
- EntryMixinFactory: list, get by name, get from config, default params, validation
- ExitMixinFactory: same
- BaseEntryMixin contract: missing required methods raise clear errors
- StrategyConfigBuilder: valid build, missing entry, position size guards
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy.custom_strategy import StrategyConfigBuilder
from src.strategy.entry.base_entry_mixin import BaseEntryMixin
from src.strategy.entry.entry_mixin_factory import (
    ENTRY_MIXIN_REGISTRY,
    get_entry_mixin,
    get_entry_mixin_default_params,
    get_entry_mixin_from_config,
    list_available_entry_mixins,
    validate_entry_mixin_params,
)
from src.strategy.exit.exit_mixin_factory import (
    EXIT_MIXIN_REGISTRY,
    get_exit_mixin,
    get_exit_mixin_default_params,
    get_exit_mixin_from_config,
    list_available_exit_mixins,
    validate_exit_mixin_params,
)

# ---------------------------------------------------------------------------
# Entry mixin factory
# ---------------------------------------------------------------------------


class TestEntryMixinFactory:
    def test_list_returns_non_empty(self):
        names = list_available_entry_mixins()
        assert len(names) > 0, "Entry mixin registry must not be empty"

    def test_list_matches_registry_keys(self):
        assert set(list_available_entry_mixins()) == set(ENTRY_MIXIN_REGISTRY.keys())

    def test_get_known_mixin_returns_instance(self):
        name = list_available_entry_mixins()[0]
        mixin = get_entry_mixin(name)
        assert mixin is not None
        assert isinstance(mixin, BaseEntryMixin)

    def test_get_unknown_mixin_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown entry mixin"):
            get_entry_mixin("DoesNotExist")

    def test_get_from_config_creates_instance(self):
        name = list_available_entry_mixins()[0]
        mixin = get_entry_mixin_from_config({"name": name})
        assert isinstance(mixin, BaseEntryMixin)

    def test_get_from_config_missing_name_raises(self):
        with pytest.raises(ValueError, match="'name'"):
            get_entry_mixin_from_config({})

    def test_default_params_returns_dict(self):
        name = list_available_entry_mixins()[0]
        params = get_entry_mixin_default_params(name)
        assert isinstance(params, dict)

    def test_default_params_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown entry mixin"):
            get_entry_mixin_default_params("GhostMixin")

    def test_validate_valid_params_returns_true(self):
        name = list_available_entry_mixins()[0]
        assert validate_entry_mixin_params(name, {}) is True

    def test_validate_unknown_mixin_raises(self):
        with pytest.raises(ValueError, match="Unknown entry mixin"):
            validate_entry_mixin_params("Ghost", {})

    def test_all_registered_mixins_are_instantiable(self):
        """Every mixin in the registry must instantiate without required params."""
        failures = []
        for name, cls in ENTRY_MIXIN_REGISTRY.items():
            try:
                cls()
            except Exception as exc:
                failures.append(f"{name}: {exc}")
        assert not failures, "Mixins failed to instantiate:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Exit mixin factory
# ---------------------------------------------------------------------------


class TestExitMixinFactory:
    def test_list_returns_non_empty(self):
        assert len(list_available_exit_mixins()) > 0

    def test_list_matches_registry_keys(self):
        assert set(list_available_exit_mixins()) == set(EXIT_MIXIN_REGISTRY.keys())

    def test_get_known_mixin_returns_instance(self):
        name = list_available_exit_mixins()[0]
        mixin = get_exit_mixin(name)
        assert mixin is not None

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown exit mixin"):
            get_exit_mixin("DoesNotExist")

    def test_get_from_config_creates_instance(self):
        name = list_available_exit_mixins()[0]
        mixin = get_exit_mixin_from_config({"name": name})
        assert mixin is not None

    def test_get_from_config_missing_name_raises(self):
        with pytest.raises(ValueError, match="'name'"):
            get_exit_mixin_from_config({})

    def test_default_params_returns_dict(self):
        name = list_available_exit_mixins()[0]
        params = get_exit_mixin_default_params(name)
        assert isinstance(params, dict)

    def test_validate_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown exit mixin"):
            validate_exit_mixin_params("Ghost", {})

    def test_all_registered_mixins_are_instantiable(self):
        failures = []
        for name, cls in EXIT_MIXIN_REGISTRY.items():
            try:
                cls()
            except Exception as exc:
                failures.append(f"{name}: {exc}")
        assert not failures, "Exit mixins failed to instantiate:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# BaseEntryMixin contract enforcement
# ---------------------------------------------------------------------------


class TestBaseEntryMixinContract:
    def test_missing_should_enter_raises(self):
        """A concrete subclass that omits should_enter must not instantiate."""

        class BrokenMixin(BaseEntryMixin):
            @classmethod
            def get_default_params(cls):
                return {}

            def get_required_params(self):
                return []

            def _init_indicators(self):
                pass

            # Intentionally NOT implementing should_enter

        with pytest.raises((TypeError, NotImplementedError)):
            BrokenMixin()

    def test_missing_get_default_params_raises(self):
        """A concrete subclass that omits get_default_params must not instantiate."""

        class BrokenMixin(BaseEntryMixin):
            def get_required_params(self):
                return []

            def _init_indicators(self):
                pass

            def should_enter(self):
                return False

            # Intentionally NOT implementing get_default_params

        with pytest.raises((TypeError, NotImplementedError)):
            BrokenMixin()

    def test_valid_subclass_instantiates(self):
        """A fully compliant subclass must instantiate without error."""

        class GoodMixin(BaseEntryMixin):
            @classmethod
            def get_default_params(cls):
                return {"threshold": 30}

            def get_required_params(self):
                return []

            def _init_indicators(self):
                pass

            def should_enter(self):
                return False

        mixin = GoodMixin()
        assert mixin is not None
        assert mixin.params["threshold"] == 30

    def test_required_param_missing_raises_value_error(self):
        """Missing a required parameter must raise ValueError on instantiation."""

        class StrictMixin(BaseEntryMixin):
            @classmethod
            def get_default_params(cls):
                return {}

            def get_required_params(self):
                return ["rsi_period"]

            def _init_indicators(self):
                pass

            def should_enter(self):
                return False

        with pytest.raises(ValueError, match="rsi_period"):
            StrictMixin()

    def test_required_param_provided_succeeds(self):
        class StrictMixin(BaseEntryMixin):
            @classmethod
            def get_default_params(cls):
                return {}

            def get_required_params(self):
                return ["rsi_period"]

            def _init_indicators(self):
                pass

            def should_enter(self):
                return False

        mixin = StrictMixin(params={"rsi_period": 14})
        assert mixin.params["rsi_period"] == 14


# ---------------------------------------------------------------------------
# StrategyConfigBuilder
# ---------------------------------------------------------------------------


class TestStrategyConfigBuilder:
    def test_build_valid_config(self):
        config = (
            StrategyConfigBuilder()
            .set_entry_mixin("RSIBBEntryMixin", {"rsi_period": 14})
            .set_exit_mixin("ATRExitMixin", {"atr_period": 14})
            .set_position_size(0.05)
            .build()
        )
        assert config["entry_logic"]["name"] == "RSIBBEntryMixin"
        assert config["exit_logic"]["name"] == "ATRExitMixin"
        assert config["position_size"] == 0.05

    def test_build_without_entry_raises(self):
        with pytest.raises(ValueError, match="[Ee]ntry"):
            (StrategyConfigBuilder().set_exit_mixin("ATRExitMixin").build())

    def test_build_without_exit_raises(self):
        with pytest.raises(ValueError, match="[Ee]xit"):
            (StrategyConfigBuilder().set_entry_mixin("RSIBBEntryMixin").build())

    def test_position_size_zero_raises(self):
        with pytest.raises(ValueError):
            StrategyConfigBuilder().set_position_size(0.0)

    def test_position_size_over_one_raises(self):
        with pytest.raises(ValueError):
            StrategyConfigBuilder().set_position_size(1.1)

    def test_position_size_exactly_one_is_valid(self):
        builder = StrategyConfigBuilder().set_position_size(1.0)
        assert builder.config["position_size"] == 1.0

    def test_chained_setters_return_self(self):
        builder = StrategyConfigBuilder()
        result = builder.set_entry_mixin("RSIBBEntryMixin")
        assert result is builder

    def test_default_position_size(self):
        builder = StrategyConfigBuilder()
        assert builder.config["position_size"] == 0.1
