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

from typing import Any, Dict, Optional
import backtrader as bt
from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

class MultiLevelAtrExitMixin(BaseExitMixin):
    """
    Hierarchical Multi-TF ATR trailing stop exit strategy.

    Parameters:
    - x_htf_atr_period: Period for HTF ATR (default: 14)
    - x_htf_sl_multiplier: Multiplier for HTF stop (default: 2.5)
    - x_ltf_atr_period: Period for LTF ATR (default: 14)
    - x_ltf_sl_multiplier: Multiplier for LTF stop (default: 2.0)
    - x_micro_atr_period: Period for Micro ATR (default: 14)
    - x_be_activation_atr: Profit in LTF ATR units to activate BE (default: 1.0)
    - x_be_buffer_multiplier: Multiplier for Micro ATR buffer in BE (default: 0.3)
    - x_use_dynamic_k: Enable ATR-of-ATR dynamic multipliers (default: False)
    - x_vol_sma_period: Period for ATR SMA for dynamic K (default: 50)
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.stop_price = -float('inf')
        self.highest_price = -float('inf')
        self.entry_price = None

        # Indicators
        self.atr_htf = None
        self.atr_ltf = None
        self.atr_micro = None
        self.atr_ltf_sma = None

        self.be_activated = False

        # Detect architecture mode
        self.use_new_architecture = False  # Will be set in init_exit()

    def init_exit(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        """Override to detect architecture mode before calling parent."""
        # Detect architecture: new if strategy has indicators dict with entries
        if hasattr(strategy, 'indicators') and strategy.indicators:
            self.use_new_architecture = True
            logger.debug("Using new TALib-based architecture")
        else:
            self.use_new_architecture = False
            logger.debug("Using legacy architecture")

        # Call parent init_exit which will call _init_indicators
        super().init_exit(strategy, additional_params)

        # Ensure data feeds are always available (needed for should_exit logic)
        self.data_ltf = self.strategy.data0
        self.data_htf = self.strategy.data1 if len(self.strategy.datas) > 1 else self.strategy.data0
        self.data_micro = self.strategy.data0

    def get_required_params(self) -> list:
        return []

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "x_htf_atr_period": 14,
            "x_htf_sl_multiplier": 2.5,
            "x_ltf_atr_period": 14,
            "x_ltf_sl_multiplier": 2.0,
            "x_micro_atr_period": 14,
            "x_be_activation_atr": 1.0,  # Activate BE when profit > 1 * ATR_LTF
            "x_be_buffer_multiplier": 0.3,
            "x_use_dynamic_k": False,
            "x_vol_sma_period": 50,
            "x_htf_compression": 16,  # Multiplier relative to LTF (e.g. 16 * 15m = 240m = 4H)
        }

    def _init_indicators(self):
        """Initialize indicators for all timeframes."""
        if self.use_new_architecture:
            # indicators are managed by the strategy config
            return

        if not hasattr(self, "strategy"):
            return

        # data0 = Base TF (LTF)
        # data1 = Resampled HTF (if exists, otherwise fallback to data0)
        # data_micro = Optional micro TF (if exists, otherwise fallback to data0)

        # LTF ATR (Tactical)
        self.atr_ltf = bt.indicators.ATR(self.data_ltf, period=self.params["x_ltf_atr_period"])
        self.register_indicator("exit_atr_ltf", self.atr_ltf)

        # HTF ATR (Strategic)
        # 1. Determine base interval from data (default to 15m if undetectable)
        base_interval = getattr(self.data_ltf, '_compression', 15)
        # Handle cases where _compression might be None or 0
        if not base_interval: base_interval = 15

        # 2. Calculate total HTF minutes
        comp_multiplier = self.params.get("x_htf_compression", 16)
        comp_minutes = int(comp_multiplier * base_interval)
        period = self.params.get("x_htf_atr_period", 14)
        precalc_col = f"atr_{comp_minutes}_{period}"

        if hasattr(self.data_ltf, precalc_col):
            logger.info("Using pre-calculated HTF ATR: %s", precalc_col)
            self.atr_htf = getattr(self.data_ltf, precalc_col)
        else:
            logger.debug("Calculating HTF ATR on the fly (data1 or fallback): %s", precalc_col)
            # data_htf is expected to be a resampled feed if not using pre-calculated
            self.atr_htf = bt.indicators.ATR(self.data_htf, period=period)

        self.register_indicator("exit_atr_htf", self.atr_htf)

        # Micro ATR (Protection)
        self.atr_micro = bt.indicators.ATR(self.data_micro, period=self.params["x_micro_atr_period"])
        self.register_indicator("exit_atr_micro", self.atr_micro)

        # ATR-of-ATR indicators
        if self.params["x_use_dynamic_k"]:
            self.atr_ltf_sma = bt.indicators.SMA(self.atr_ltf, period=self.params["x_vol_sma_period"])
            self.register_indicator("exit_atr_ltf_sma", self.atr_ltf_sma)

    def on_entry(self, entry_price: float, entry_time, position_size: float, direction: str):
        """Called when a position is entered."""
        self.entry_price = entry_price
        self.highest_price = entry_price
        self.stop_price = -float('inf')
        self.be_activated = False
        logger.debug(f"MultiLevelAtrExitMixin entered at {entry_price}")

    def get_minimum_lookback(self) -> int:
        """
        Returns the minimum number of bars required.
        """
        periods = [
            self.params.get("x_htf_atr_period", 14),
            self.params.get("x_ltf_atr_period", 14),
            self.params.get("x_micro_atr_period", 14)
        ]
        if self.params.get("x_use_dynamic_k", False):
            periods.append(self.params.get("x_vol_sma_period", 50))

        return max(periods)

    def are_indicators_ready(self) -> bool:
        """
        Check if indicators are initialized.
        History/Lookback is now handled by BaseStrategy.
        """
        required = ["exit_atr_ltf", "exit_atr_htf", "exit_atr_micro"]
        if self.params.get("x_use_dynamic_k", False):
            required.append("exit_atr_ltf_sma")

        if self.use_new_architecture:
            # Check strategy's central indicators dict
            return all(name in getattr(self.strategy, 'indicators', {}) for name in required)
        else:
            # Legacy: check mixin's own indicators dict
            return all(name in self.indicators for name in required)

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
            if self.params["x_use_dynamic_k"]:
                atr_ltf_sma = self.get_indicator("exit_atr_ltf_sma")
                if atr_ltf_sma > 0:
                    vol_ratio = atr_ltf / atr_ltf_sma

            k_htf = self.params["x_htf_sl_multiplier"] * vol_ratio
            k_ltf = self.params["x_ltf_sl_multiplier"] * vol_ratio

            # Strategic Stop (S0) - Fixed relative to entry or HTF
            s0 = self.entry_price - atr_htf * k_htf

            # Tactical Trailing (S1) - Relative to highest since entry
            s1 = self.highest_price - atr_ltf * k_ltf

            # Break-even / Protection (S2)
            s2 = -float('inf')
            profit_in_atr = (current_close - self.entry_price) / atr_ltf if atr_ltf > 0 else 0

            if profit_in_atr >= self.params["x_be_activation_atr"]:
                if not self.be_activated:
                    logger.info(f"Break-even activated at profit {profit_in_atr:.2f} ATR")
                    self.be_activated = True

                # S2 = entry + small buffer based on Micro ATR
                s2 = self.entry_price + atr_micro * self.params["x_be_buffer_multiplier"]

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
