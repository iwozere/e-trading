"""
Custom Strategy Module

This module implements a custom trading strategy with support for modular entry/exit mixins.
It provides:
1. Flexible entry and exit logic through mixins
2. Position and trade tracking
3. Equity curve tracking
4. Performance metrics collection
"""

from typing import Any, Dict

import backtrader as bt
import pandas as pd
from src.entry.entry_mixin_factory import (ENTRY_MIXIN_REGISTRY,
                                           get_entry_mixin,
                                           get_entry_mixin_from_config)
from src.exit.exit_mixin_factory import (EXIT_MIXIN_REGISTRY, get_exit_mixin,
                                         get_exit_mixin_from_config)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class CustomStrategy(bt.Strategy):
    """
    Main strategy with support for modular entry/exit mixins.

    Parameters:
    -----------
    strategy_config : dict
        Configuration dictionary containing:
        - entry_logic: Entry mixin configuration
        - exit_logic: Exit mixin configuration
        - position_size: Position size as fraction of capital (default: 0.1)
        - use_talib: Whether to use TA-Lib for indicator calculations
    """

    params = (
        ("strategy_config", None),  # Strategy configuration
        ("position_size", 1.0),  # Default position size
    )

    def __init__(self):
        """Initialize strategy with configuration"""
        _logger.debug("CustomStrategy.__init__ called")
        super().__init__()  # Call parent's __init__ first

        # Initialize basic attributes
        self.use_talib = False  # Default value
        self.entry_logic = None
        self.exit_logic = None

        # Initialize trade tracking
        self.trades = []
        self.current_exit_reason = None  # Track current exit reason
        self.current_trade = None

        # Initialize equity curve tracking
        self.equity_curve = []
        self.equity_dates = []

        # Initialize entry and exit mixins
        self.entry_mixin = None
        self.exit_mixin = None

        self.trade = None

        _logger.debug("CustomStrategy.__init__ completed")

    def start(self):
        """Called once at the start of the strategy"""
        _logger.debug("CustomStrategy.start called")
        try:
            _logger.debug("Starting strategy initialization...")
            # Set configuration from params
            if self.p.strategy_config:
                self.use_talib = self.p.strategy_config.get("use_talib", False)
                self.entry_logic = self.p.strategy_config.get("entry_logic")
                self.exit_logic = self.p.strategy_config.get("exit_logic")
                _logger.debug(
                    f"Strategy config loaded - Entry: {self.entry_logic['name']}, Exit: {self.exit_logic['name']}"
                )

            # Create mixins but don't initialize indicators yet
            if self.entry_logic:
                entry_mixin_class = ENTRY_MIXIN_REGISTRY[self.entry_logic["name"]]
                if entry_mixin_class:
                    _logger.debug(f"Creating entry mixin: {self.entry_logic['name']}")
                    self.entry_mixin = entry_mixin_class(
                        params=self.entry_logic["params"]
                    )
                    self.entry_mixin.init_entry(self)
                    _logger.debug(
                        f"Entry mixin created with params: {self.entry_logic['params']}"
                    )

            if self.exit_logic:
                exit_mixin_class = EXIT_MIXIN_REGISTRY[self.exit_logic["name"]]
                if exit_mixin_class:
                    _logger.debug(f"Creating exit mixin: {self.exit_logic['name']}")
                    self.exit_mixin = exit_mixin_class(params=self.exit_logic["params"])
                    self.exit_mixin.init_exit(self)
                    _logger.debug(
                        f"Exit mixin created with params: {self.exit_logic['params']}"
                    )
        except Exception as e:
            _logger.error(f"Error in start: {e}", exc_info=e)
            raise

    def prenext(self):
        """Skip bars until we have enough data"""
        pass

    def next(self):
        """Called for each bar"""
        # Call mixins' next method to check for indicator reinitialization
        if self.entry_mixin:
            self.entry_mixin.next()
        if self.exit_mixin:
            self.exit_mixin.next()

        # Check for entry signals
        if (
            self.current_trade is None
            and self.entry_mixin
            and self.entry_mixin.should_enter()
        ):
            # Use position_size from config if available, else default to 0.10
            position_size = 0.10
            if self.p.strategy_config and "position_size" in self.p.strategy_config:
                position_size = self.p.strategy_config["position_size"]
            cash = self.broker.get_cash()
            size = (cash * position_size) / self.data.close[0]
            if size > 0:
                self.buy(size=size)

        # Check for exit signals
        if (
            self.current_trade is not None
            and self.exit_mixin
            and self.exit_mixin.should_exit()
        ):
            self.sell(size=self.current_trade["size"])

    def notify_trade(self, trade):
        """Record trade information"""
        self.trade = trade
        try:
            _logger.info(
                f"Trade notification received - Status: {'CLOSED' if trade.isclosed else 'OPEN'}, "
                f"Size: {trade.size}, PnL: {trade.pnl}, "
                f"Price: {trade.price}"
            )

            # Update position state based on trade status
            if trade.isclosed:
                # Calculate trade duration in minutes
                duration_days = trade.dtclose - trade.dtopen
                duration_minutes = duration_days * 24 * 60  # Convert days to minutes

                # Calculate PnL
                # trade.price is an average price of multiple BUY orders for the same position.
                entry_value = self.current_trade["entry_price"] * self.current_trade["size"]

                # Instead of trade.price we should use the close price from the time when trade got closed
                # TODO: in the future, if we need to support multiple SELL activities on the order, we should keep track of all of them.
                exit_value = self.data.close[0] * self.current_trade["size"]
                gross_pnl = exit_value - entry_value
                net_pnl = gross_pnl - trade.commission

                # Convert Backtrader datetime to pandas datetime
                exit_time = self.data.num2date(trade.dtclose)

                # Update trade record with exit information
                self.current_trade.update(
                    {
                        "exit_time": exit_time,
                        "exit_price": self.data.close[0],  # Price per asset, not total value
                        "exit_value": exit_value,
                        "exit_reason": self.current_exit_reason or "unknown",
                        "commission": trade.commission,
                        "duration_minutes": duration_minutes,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "pnl_percentage": ((net_pnl / entry_value) * 100 if entry_value != 0 else 0),
                        "trade_type": ("long" if self.current_trade["size"] > 0 else "short"),
                        "status": "closed",
                    }
                )

                self.trades.append(self.current_trade)
                _logger.info(
                    f"Position closed - Entry: {self.current_trade['entry_price']}, "
                    f"Exit: {trade.price}, PnL: {net_pnl:.2f} ({self.current_trade['pnl_percentage']:.2f}%), "
                    f"Duration: {duration_minutes:.1f} minutes"
                )

                self.current_trade = None
                self.current_exit_reason = None  # Reset exit reason
                self.trade = None
            else:
                # Convert Backtrader datetime to pandas datetime
                entry_time = self.data.num2date(trade.dtopen)

                # Trade is opened
                self.current_trade = {
                    "entry_time": entry_time,
                    "entry_price": trade.price,
                    "entry_value": trade.price * trade.size,
                    "size": trade.size,
                    "symbol": self.data._name,
                    "commission": trade.commission,
                    "exit_time": None,
                    "exit_price": None,
                    "exit_value": None,
                    "exit_reason": None,
                    "status": "open",
                    "trade_type": "long" if trade.size > 0 else "short",
                }
                _logger.info(
                    f"Position opened - Price: {trade.price}, Size: {trade.size}"
                )

            if self.entry_mixin:
                self.entry_mixin.notify_trade(trade)
            if self.exit_mixin:
                self.exit_mixin.notify_trade(trade)

        except Exception as e:
            _logger.error(f"Error in notify_trade: {e}")
            raise


# Example of creating a strategy with a new approach
def create_strategy_example():
    """Example of creating a strategy with different configurations"""

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
    """Helper for creating strategy configurations"""

    def __init__(self):
        self.config = {
            "entry_logic": None,
            "exit_logic": None,
            "position_size": 0.1,
            "use_talib": False,
        }

    def set_entry_mixin(self, name: str, params: Dict[str, Any] = None):
        """Set entry mixin configuration"""
        self.config["entry_logic"] = {"name": name, "params": params or {}}
        return self

    def set_exit_mixin(self, name: str, params: Dict[str, Any] = None):
        """Set exit mixin configuration"""
        self.config["exit_logic"] = {"name": name, "params": params or {}}
        return self

    def set_position_size(self, size: float):
        """Set position size as fraction of capital"""
        if not 0 < size <= 1:
            raise ValueError("Position size must be between 0 and 1")
        self.config["position_size"] = size
        return self

    def set_use_talib(self, use_talib: bool):
        """Set whether to use TA-Lib"""
        self.config["use_talib"] = use_talib
        return self

    def build(self) -> Dict[str, Any]:
        """Build the final strategy configuration"""
        if not self.config["entry_logic"]:
            raise ValueError("Entry mixin configuration is required")
        if not self.config["exit_logic"]:
            raise ValueError("Exit mixin configuration is required")
        return self.config
