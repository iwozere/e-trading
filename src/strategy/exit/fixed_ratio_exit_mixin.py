"""
Fixed Ratio Exit Mixin

This module implements an exit strategy based on fixed profit and loss ratios.
The strategy exits a position when:
1. Price reaches the take profit level (entry price * (1 + profit_ratio))
2. Price reaches the stop loss level (entry price * (1 - stop_loss_ratio))
"""

from typing import Any, Dict, Optional, List

from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class FixedRatioExitMixin(BaseExitMixin):
    """Exit mixin based on a fixed ratio of profit or loss.
    New Architecture only.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the mixin with parameters"""
        super().__init__(params)

    def get_required_params(self) -> list:
        """There are no required parameters - all have default values"""
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Default parameters"""
        return {
            "take_profit": 0.1,
            "stop_loss": 0.05,
        }

    @classmethod
    def get_indicator_config(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """No indicators required for this mixin."""
        return []

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

        entry_price = self.strategy.position.price
        current_price = self.strategy.data.close[0]
        profit_ratio = (current_price - entry_price) / entry_price

        # Standardized parameter retrieval
        tp_ratio = self.get_param("take_profit") or self.get_param("x_take_profit", 0.1)
        sl_ratio = self.get_param("stop_loss") or self.get_param("x_stop_loss", 0.05)

        return_value = False
        if profit_ratio >= tp_ratio:
            self.strategy.current_exit_reason = "take_profit"
            return_value = True
        elif profit_ratio <= -sl_ratio:
            self.strategy.current_exit_reason = "stop_loss"
            return_value = True

        if return_value:
            logger.debug(
                f"FIXED RATIO EXIT - Price: {current_price:.2f}, Entry: {entry_price:.2f}, "
                f"Profit %: {profit_ratio*100:.2f}%, Reason: {self.strategy.current_exit_reason}"
            )
        return return_value

    def get_exit_reason(self) -> str:
        """Get the reason for exit"""
        return getattr(self.strategy, 'current_exit_reason', 'fixed_ratio')
