"""
Time-based Exit Mixin

This module implements an exit strategy based on time duration. The strategy exits a position
after a specified number of bars or calendar days have elapsed since entry.
"""

from typing import Any, Dict, Optional

from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class TimeBasedExitMixin(BaseExitMixin):
    """Exit mixin based on time."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.entry_bar = None
        self.entry_time = None
        self.use_new_architecture = False

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "x_max_bars": 20,
            "x_use_time": False,
            "x_max_minutes": 60,
        }

    def init_exit(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        """Standardize architecture detection."""
        if hasattr(strategy, 'indicators') and strategy.indicators:
            self.use_new_architecture = True
        else:
            self.use_new_architecture = False
        super().init_exit(strategy, additional_params)

    def _init_indicators(self):
        """No indicators required for this mixin."""
        pass

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required (0 for this mixin)."""
        return 0

    def should_exit(self) -> bool:
        """Check if we should exit a position"""
        if not self.strategy.position:
            return False

        try:
            # Standardized parameter retrieval
            use_time = self.get_param("use_time") or self.get_param("x_use_time", False)
            max_minutes = self.get_param("max_minutes") or self.get_param("x_max_minutes", 60)
            max_bars = self.get_param("max_bars") or self.get_param("x_max_bars", 20)

            if use_time:
                # Use calendar time
                if self.strategy.current_trade is None:
                    return False

                current_time = self.strategy.data.datetime.datetime(0)
                entry_time = self.strategy.current_trade.get("entry_time")

                if entry_time is None:
                    return False

                time_diff = (current_time - entry_time).total_seconds() / 60
                return_value = time_diff >= max_minutes

                if return_value:
                    logger.debug(f"TIME EXIT - Time held: {time_diff:.2f}m, Max: {max_minutes}m")
                    self.strategy.current_exit_reason = "time_limit_minutes"
            else:
                # Use bar count
                if self.entry_bar is None:
                    return False

                bars_held = len(self.strategy.data) - self.entry_bar
                return_value = bars_held >= max_bars

                if return_value:
                    logger.debug(f"TIME EXIT - Bars held: {bars_held}, Max: {max_bars}")
                    self.strategy.current_exit_reason = "time_limit_bars"

            return return_value

        except Exception:
            logger.exception("Error in TimeBasedExitMixin.should_exit")
            return False

    def on_entry(self, entry_price: float, entry_time, position_size: float, direction: str):
        """Called when a position is entered"""
        self.entry_bar = len(self.strategy.data)
        self.entry_time = self.strategy.data.datetime.datetime(0)

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return getattr(self.strategy, 'current_exit_reason', 'time_limit')