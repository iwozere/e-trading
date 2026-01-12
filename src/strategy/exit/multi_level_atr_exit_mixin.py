"""
Multi-Level ATR Exit Mixin
--------------------------

Implements a hierarchical stop-loss system using multiple timeframes as described in TODO.md:
- S0 (HTF ATR): Strategic stop, anchored to entry price. Protects against wide swings.
- S1 (LTF ATR): Tactical trailing stop, anchored to the highest price since entry.
- S2 (Micro ATR/BE): Break-even protection with a small buffer, activated after a profit threshold.

The final stop is: STOP = max(S0, S1, S2)

It also supports dynamic volatility adaptation (ATR-of-ATR) where multipliers adjust
based on current volatility relative to its average.
"""

from typing import Any, Dict, Optional, List
import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

class MultiLevelAtrExitMixin(BaseExitMixin):
    """
    Hierarchical Multi-TF ATR trailing stop exit strategy.
    New Architecture only.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.stop_price = -float('inf')
        self.highest_price = -float('inf')
        self.entry_price = None
        self.be_activated = False

        # Ensure data feeds are always available (needed for should_exit logic)
        # These will be set in init_exit
        self.data_ltf = None
        self.data_htf = None
        self.data_micro = None

    def init_exit(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        """Override to ensure data feeds are initialized."""
        super().init_exit(strategy, additional_params)

        # Ensure data feeds are always available
        self.data_ltf = self.strategy.data0
        self.data_htf = self.strategy.data1 if len(self.strategy.datas) > 1 else self.strategy.data0
        self.data_micro = self.strategy.data0

    def get_required_params(self) -> list:
        return []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            "htf_atr_period": 14,
            "htf_sl_multiplier": 2.5,
            "ltf_atr_period": 14,
            "ltf_sl_multiplier": 2.0,
            "micro_atr_period": 14,
            "be_activation_atr": 1.0,  # Activate BE when profit > 1 * ATR_LTF
            "be_buffer_multiplier": 0.3,
            "use_dynamic_k": False,
            "vol_sma_period": 50,
            "htf_compression": 16,  # Multiplier relative to LTF (e.g. 15 * 16 = 240m = 4H)
        }

    @classmethod
    def get_indicator_config(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define indicators required by this mixin."""
        htf_atr_period = params.get("htf_atr_period") or params.get("x_htf_atr_period", 14)
        ltf_atr_period = params.get("ltf_atr_period") or params.get("x_ltf_atr_period", 14)
        micro_atr_period = params.get("micro_atr_period") or params.get("x_micro_atr_period", 14)
        use_dynamic_k = params.get("use_dynamic_k", params.get("x_use_dynamic_k", False))
        vol_sma_period = params.get("vol_sma_period") or params.get("x_vol_sma_period", 50)

        configs = [
            {
                "type": "ATR",
                "params": {"timeperiod": ltf_atr_period},
                "fields_mapping": {"atr": "exit_atr_ltf"}
            },
            {
                "type": "ATR",
                "params": {"timeperiod": htf_atr_period},
                "fields_mapping": {"atr": "exit_atr_htf"}
            },
            {
                "type": "ATR",
                "params": {"timeperiod": micro_atr_period},
                "fields_mapping": {"atr": "exit_atr_micro"}
            }
        ]

        if use_dynamic_k:
            configs.append({
                "type": "SMA",
                "params": {"timeperiod": vol_sma_period},
                "fields_mapping": {"sma": "exit_atr_ltf_sma"}
            })

        return configs

    def _init_indicators(self):
        """No-op for new architecture."""
        pass

    def on_entry(self, entry_price: float, entry_time, position_size: float, direction: str):
        """Called when a position is entered."""
        self.entry_price = entry_price
        self.highest_price = entry_price
        self.stop_price = -float('inf')
        self.be_activated = False
        logger.debug(f"MultiLevelAtrExitMixin entered at {entry_price}")

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        periods = [
            self.get_param("htf_atr_period") or self.get_param("x_htf_atr_period", 14),
            self.get_param("ltf_atr_period") or self.get_param("x_ltf_atr_period", 14),
            self.get_param("micro_atr_period") or self.get_param("x_micro_atr_period", 14)
        ]
        if self.get_param("use_dynamic_k") or self.get_param("x_use_dynamic_k", False):
            periods.append(self.get_param("vol_sma_period") or self.get_param("x_vol_sma_period", 50))

        return max(periods)

    def are_indicators_ready(self) -> bool:
        """Check if required indicators exist in the strategy registry."""
        required = ["exit_atr_ltf", "exit_atr_htf", "exit_atr_micro"]
        if self.get_param("use_dynamic_k") or self.get_param("x_use_dynamic_k", False):
            required.append("exit_atr_ltf_sma")

        return all(name in getattr(self.strategy, 'indicators', {}) for name in required)

    def should_exit(self) -> bool:
        if not self.are_indicators_ready() or self.entry_price is None:
            return False

        try:
            current_close = self.data_ltf.close[0]
            current_high = self.data_ltf.high[0]
            self.highest_price = max(self.highest_price, current_high)

            # Standardized Indicator Access
            atr_ltf = self.get_indicator("exit_atr_ltf")
            atr_htf = self.get_indicator("exit_atr_htf")
            atr_micro = self.get_indicator("exit_atr_micro")

            # Dynamic K calculation (ATR-of-ATR)
            vol_ratio = 1.0
            if self.get_param("use_dynamic_k") or self.get_param("x_use_dynamic_k", False):
                atr_ltf_sma = self.get_indicator("exit_atr_ltf_sma")
                if atr_ltf_sma > 0:
                    vol_ratio = atr_ltf / atr_ltf_sma

            htf_sl_multiplier = self.get_param("htf_sl_multiplier") or self.get_param("x_htf_sl_multiplier", 2.5)
            ltf_sl_multiplier = self.get_param("ltf_sl_multiplier") or self.get_param("x_ltf_sl_multiplier", 2.0)

            k_htf = htf_sl_multiplier * vol_ratio
            k_ltf = ltf_sl_multiplier * vol_ratio

            # Strategic Stop (S0) - Fixed relative to entry or HTF
            s0 = self.entry_price - atr_htf * k_htf

            # Tactical Trailing (S1) - Relative to highest since entry
            s1 = self.highest_price - atr_ltf * k_ltf

            # Break-even / Protection (S2)
            s2 = -float('inf')
            profit_in_atr = (current_close - self.entry_price) / atr_ltf if atr_ltf > 0 else 0

            be_activation_atr = self.get_param("be_activation_atr") or self.get_param("x_be_activation_atr", 1.0)
            if profit_in_atr >= be_activation_atr:
                if not self.be_activated:
                    logger.info(f"Break-even activated at profit {profit_in_atr:.2f} ATR")
                    self.be_activated = True

                # S2 = entry + small buffer based on Micro ATR
                be_buffer_multiplier = self.get_param("be_buffer_multiplier") or self.get_param("x_be_buffer_multiplier", 0.3)
                s2 = self.entry_price + atr_micro * be_buffer_multiplier

            # Final STOP = max(S0, S1, S2) - only moves UP
            new_stop = max(s0, s1, s2)
            self.stop_price = max(self.stop_price, new_stop)

            if current_close < self.stop_price:
                logger.debug(f"EXIT SIGNAL - Price: {current_close:.2f} < Stop: {self.stop_price:.2f}")
                self.strategy.current_exit_reason = "multi_tf_atr_stop"
                return True

            return False

        except Exception:
            logger.exception("Error in should_exit: ")
            return False

    def get_exit_reason(self) -> str:
        return getattr(self.strategy, 'current_exit_reason', 'multi_tf_atr_exit')

    def next(self):
        super().next()
        if not self.strategy.position:
            self.entry_price = None
            self.highest_price = -float('inf')
            self.stop_price = -float('inf')
            self.be_activated = False

    def get_exit_reason(self) -> str:
        return getattr(self.strategy, 'current_exit_reason', 'multi_tf_atr_exit')

    def next(self):
        super().next()
        if not self.strategy.position:
            self.entry_price = None
            self.highest_price = -float('inf')
            self.stop_price = -float('inf')
            self.be_activated = False
