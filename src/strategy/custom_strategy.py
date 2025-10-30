"""
Refactored Custom Strategy Module

This module demonstrates how CustomStrategy can be simplified by inheriting
from BaseBacktraderStrategy, which provides common functionality like trade
tracking, position management, and performance monitoring.
"""

from typing import Any, Dict

from src.notification.logger import setup_logger
from src.strategy.entry.entry_mixin_factory import ENTRY_MIXIN_REGISTRY
from src.strategy.exit.exit_mixin_factory import EXIT_MIXIN_REGISTRY
from src.strategy.base_strategy import BaseStrategy

_logger = setup_logger(__name__)


class CustomStrategy(BaseStrategy):
    """
    Refactored custom strategy with support for modular entry/exit mixins.

    This version inherits from BaseStrategy, which provides:
    - Trade tracking and management
    - Position sizing
    - Performance monitoring
    - Configuration management
    - Error handling

    The strategy only needs to implement:
    - Entry/exit mixin initialization
    - Strategy-specific logic in _execute_strategy_logic
    """

    def _initialize_strategy(self):
        """Initialize entry and exit mixins."""
        try:
            _logger.debug("Initializing CustomStrategy mixins...")

            # Set configuration from params
            if self.p.strategy_config:
                self.use_talib = self.p.strategy_config.get("use_talib", False)
                self.entry_logic = self.p.strategy_config.get("entry_logic")
                self.exit_logic = self.p.strategy_config.get("exit_logic")
                _logger.debug(
                    f"Strategy config loaded - Entry: {self.entry_logic['name']}, Exit: {self.exit_logic['name']}"
                )

            # Create entry mixin
            if self.entry_logic:
                entry_mixin_class = ENTRY_MIXIN_REGISTRY[self.entry_logic["name"]]
                if entry_mixin_class:
                    _logger.debug("Creating entry mixin: %s", self.entry_logic['name'])
                    self.entry_mixin = entry_mixin_class(
                        params=self.entry_logic["params"]
                    )
                    self.entry_mixin.init_entry(self)
                    _logger.debug(
                        f"Entry mixin created with params: {self.entry_logic['params']}"
                    )

            # Create exit mixin
            if self.exit_logic:
                exit_mixin_class = EXIT_MIXIN_REGISTRY[self.exit_logic["name"]]
                if exit_mixin_class:
                    _logger.debug("Creating exit mixin: %s", self.exit_logic['name'])
                    self.exit_mixin = exit_mixin_class(params=self.exit_logic["params"])
                    self.exit_mixin.init_exit(self)
                    _logger.debug(
                        f"Exit mixin created with params: {self.exit_logic['params']}"
                    )

        except Exception as e:
            _logger.exception("Error in _initialize_strategy")
            raise

    def _execute_strategy_logic(self):
        """Execute strategy-specific logic."""
        try:
            # Call mixins' next method to check for indicator reinitialization
            if self.entry_mixin:
                self.entry_mixin.next()
            if self.exit_mixin:
                self.exit_mixin.next()

            # Check for entry signals
            if (
                self.position.size == 0
                and self.entry_mixin
                and self.entry_mixin.should_enter()
            ):
                # Use position_size from config if available, else default to 0.10
                position_size = 0.10
                if self.p.strategy_config and "position_size" in self.p.strategy_config:
                    position_size = self.p.strategy_config["position_size"]

                # Use base class method for position entry
                self._enter_position(
                    direction='long',
                    confidence=1.0,
                    reason="Entry mixin signal"
                )

            # Check for exit signals
            if (
                self.position.size != 0
                and self.exit_mixin
                and self.exit_mixin.should_exit()
            ):
                # Get specific exit reason from mixin
                exit_reason = "unknown"
                if hasattr(self.exit_mixin, 'get_exit_reason'):
                    try:
                        exit_reason = self.exit_mixin.get_exit_reason()
                    except Exception as e:
                        _logger.warning("Error getting exit reason from mixin: %s", e)
                        exit_reason = "mixin_error"

                # Use base class method for position exit
                self._exit_position(reason=exit_reason)

        except Exception as e:
            _logger.exception("Error in _execute_strategy_logic")

    def notify_trade(self, trade):
        """Override to add mixin-specific trade handling."""
        # Call base class implementation first
        super().notify_trade(trade)

        # Add mixin-specific handling
        try:
            if self.entry_mixin:
                self.entry_mixin.notify_trade(trade)
            if self.exit_mixin:
                self.exit_mixin.notify_trade(trade)
        except Exception as e:
            _logger.exception("Error in mixin trade notification")


# Example of creating a strategy with the refactored approach
def create_refactored_strategy_example():
    """Example of creating a strategy with the refactored approach."""

    # Configuration 1: Simple RSI + BB strategy
    strategy_config_1 = {
        "entry_logic": {
            "name": "RSIBBMixin",
            "params": {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "bb_period": 20,
                "use_bb_touch": True,
            },
        },
        "exit_logic": {
            "name": "FixedRatioExitMixin",
            "params": {"profit_ratio": 1.5, "stop_loss_ratio": 0.5},
        },
        "position_size": 0.1,
        "use_talib": False,
    }

    # Configuration 2: More complex strategy with RSI + Ichimoku
    strategy_config_2 = {
        "entry_logic": {
            "name": "RSIIchimokuMixin",
            "params": {
                "rsi_period": 21,
                "rsi_oversold": 25,
                "tenkan_period": 9,
                "kijun_period": 26,
                "require_above_cloud": True,
            },
        },
        "exit_logic": {
            "name": "TrailingStopExitMixin",
            "params": {"trail_percent": 5.0, "min_profit_percent": 2.0},
        },
        "position_size": 0.1,
        "use_talib": False,
    }

    return strategy_config_1, strategy_config_2


class StrategyConfigBuilder:
    """Helper for creating strategy configurations."""

    def __init__(self):
        self.config = {
            "entry_logic": None,
            "exit_logic": None,
            "position_size": 0.1,
            "use_talib": False,
        }

    def set_entry_mixin(self, name: str, params: Dict[str, Any] = None):
        """Set entry mixin configuration."""
        self.config["entry_logic"] = {"name": name, "params": params or {}}
        return self

    def set_exit_mixin(self, name: str, params: Dict[str, Any] = None):
        """Set exit mixin configuration."""
        self.config["exit_logic"] = {"name": name, "params": params or {}}
        return self

    def set_position_size(self, size: float):
        """Set position size as fraction of capital."""
        if not 0 < size <= 1:
            raise ValueError("Position size must be between 0 and 1")
        self.config["position_size"] = size
        return self

    def set_use_talib(self, use_talib: bool):
        """Set whether to use TA-Lib."""
        self.config["use_talib"] = use_talib
        return self

    def build(self) -> Dict[str, Any]:
        """Build the final strategy configuration."""
        if not self.config["entry_logic"]:
            raise ValueError("Entry mixin configuration is required")
        if not self.config["exit_logic"]:
            raise ValueError("Exit mixin configuration is required")
        return self.config
