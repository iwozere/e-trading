"""
Fixed Ratio Exit Mixin

This module implements an exit strategy based on fixed profit and loss ratios.
The strategy exits a position when:
1. Price reaches the take profit level (entry price * (1 + profit_ratio))
2. Price reaches the stop loss level (entry price * (1 - stop_loss_ratio))

Parameters:
    profit_ratio (float): Ratio for take profit level (default: 0.02)
    stop_loss_ratio (float): Ratio for stop loss level (default: 0.01)
    use_trailing_stop (bool): Whether to use trailing stop (default: False)
    trail_percent (float): Percentage to trail the stop (default: 0.5)

This strategy is particularly effective for:
1. Setting clear profit targets
2. Managing risk with fixed stop losses
3. Protecting profits with trailing stops
"""

from typing import Any, Dict, Optional

from src.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class FixedRatioExitMixin(BaseExitMixin):
    """Exit mixin based on a fixed ratio of profit or loss"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)
        self.highest_price = 0
        self.lowest_price = float("inf")

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "x_take_profit": 0.1,
            "x_stop_loss": 0.05,
            "x_use_trailing_stop": False,
            "x_trail_percent": 0.5,
        }

    def _init_indicators(self):
        """Initialize any required indicators"""
        if not hasattr(self, "strategy"):
            return

    def should_exit(self) -> bool:
        """Check if we should exit a position"""
        if not self.strategy.position:
            return False
        entry_price = self.strategy.position.price
        current_price = self.strategy.data.close[0]
        profit_ratio = (current_price - entry_price) / entry_price

        return_value = False
        if profit_ratio >= self.get_param("x_take_profit"):
            self.strategy.current_exit_reason = "take_profit"
            return_value = True
        elif profit_ratio <= -self.get_param("x_stop_loss"):
            self.strategy.current_exit_reason = "stop_loss"
            return_value = True

        if return_value:
            logger.debug(
                f"EXIT: Price: {current_price}, Entry: {entry_price}, "
                f"Profit %: {profit_ratio*100:.2f}%, "
                f"Take Profit: {self.get_param('x_take_profit')*100:.2f}%, "
                f"Stop Loss: {self.get_param('x_stop_loss')*100:.2f}%"
            )
        return return_value
