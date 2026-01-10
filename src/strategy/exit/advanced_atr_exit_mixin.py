"""
Advanced ATR-Based Exit Mixin

This module implements a sophisticated, volatility-adaptive trailing stop exit strategy.

Configuration Example (New architecture):
    {
        "exit_logic": {
            "name": "AdvancedATRExitMixin",
            "indicators": [
                {
                    "type": "ATR",
                    "params": {"timeperiod": 7},
                    "fields_mapping": {"atr": "exit_atr_fast"}
                },
                {
                    "type": "ATR",
                    "params": {"timeperiod": 21},
                    "fields_mapping": {"atr": "exit_atr_slow"}
                },
                {
                    "type": "ATR",
                    "params": {"timeperiod": 21},
                    "fields_mapping": {"atr": "exit_atr_htf"}
                }
            ],
            "logic_params": {
                "k_init": 2.5,
                "k_run": 2.0,
                "k_phase2": 1.5,
                "use_htf_atr": true
            }
        }
    }
"""

import backtrader as bt
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from collections import deque
import math

from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class ExitState(Enum):
    """Exit strategy state machine states."""
    INIT = "INIT"
    ARMED = "ARMED"
    PHASE1 = "PHASE1"
    PHASE2 = "PHASE2"
    LOCKED = "LOCKED"
    EXIT = "EXIT"


class AdvancedATRExitMixin(BaseExitMixin):
    """
    Advanced ATR-based trailing stop exit strategy.
    Supports both new architecture and legacy configurations.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

        # Core parameters (standardized names or legacy)
        self.anchor = self.get_param("anchor") or self.get_param("x_anchor", "high")
        self.update_on = self.get_param("update_on") or self.get_param("x_update_on", "bar_close")
        self.k_init = self.get_param("k_init") or self.get_param("x_k_init", 2.5)
        self.k_run = self.get_param("k_run") or self.get_param("x_k_run", 2.0)
        self.k_phase2 = self.get_param("k_phase2") or self.get_param("x_k_phase2", 1.5)

        # ATR parameters
        self.p_fast = self.get_param("p_fast") or self.get_param("x_p_fast", 7)
        self.p_slow = self.get_param("p_slow") or self.get_param("x_p_slow", 21)
        self.use_htf_atr = self.get_param("use_htf_atr") or self.get_param("x_use_htf_atr", True)
        self.htf = self.get_param("htf") or self.get_param("x_htf", "4h")
        self.alpha_fast = self.get_param("alpha_fast") or self.get_param("x_alpha_fast", 1.0)
        self.alpha_slow = self.get_param("alpha_slow") or self.get_param("x_alpha_slow", 1.0)
        self.alpha_htf = self.get_param("alpha_htf") or self.get_param("x_alpha_htf", 1.0)
        self.atr_floor = self.get_param("atr_floor") or self.get_param("x_atr_floor", 0.0)

        # Break-even and phases
        self.arm_at_R = self.get_param("arm_at_R") or self.get_param("x_arm_at_R", 1.0)
        self.breakeven_offset_atr = self.get_param("breakeven_offset_atr") or self.get_param("x_breakeven_offset_atr", 0.0)
        self.phase2_at_R = self.get_param("phase2_at_R") or self.get_param("x_phase2_at_R", 2.0)

        # Structural ratchet
        self.use_swing_ratchet = self.get_param("use_swing_ratchet") or self.get_param("x_use_swing_ratchet", True)
        self.swing_lookback = self.get_param("swing_lookback") or self.get_param("x_swing_lookback", 10)
        self.struct_buffer_atr = self.get_param("struct_buffer_atr") or self.get_param("x_struct_buffer_atr", 0.25)

        # Time-based tightening
        self.tighten_if_stagnant_bars = self.get_param("tighten_if_stagnant_bars") or self.get_param("x_tighten_if_stagnant_bars", 20)
        self.tighten_k_factor = self.get_param("tighten_k_factor") or self.get_param("x_tighten_k_factor", 0.8)
        self.min_bars_between_tighten = self.get_param("min_bars_between_tighten") or self.get_param("x_min_bars_between_tighten", 5)

        # Noise and step filters
        self.min_stop_step = self.get_param("min_stop_step") or self.get_param("x_min_stop_step", 0.0)
        self.noise_filter_atr = self.get_param("noise_filter_atr") or self.get_param("x_noise_filter_atr", 0.0)
        self.max_trail_freq = self.get_param("max_trail_freq") or self.get_param("x_max_trail_freq", 1)

        # State variables
        self.state = ExitState.INIT
        self.current_stop = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.initial_risk = 0.0
        self.k_current = self.k_init
        self.highest_high_since_entry = 0.0
        self.lowest_low_since_entry = 0.0
        self.last_trail_bar = 0
        self.last_tighten_bar = 0
        self.pt_levels_hit = set()
        self.direction = "long"

        # Legacy indicators
        self.atr_fast = None
        self.atr_slow = None
        self.atr_htf = None

        self.price_history = deque(maxlen=self.swing_lookback * 2)
        self.use_new_architecture = False

    def get_required_params(self) -> list:
        return []

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "x_k_init": 2.5, "x_k_run": 2.0, "x_k_phase2": 1.5,
            "x_p_fast": 7, "x_p_slow": 21, "x_use_htf_atr": True,
            "x_swing_lookback": 10, "x_arm_at_R": 1.0, "x_phase2_at_R": 2.0
        }

    def init_exit(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        if hasattr(strategy, 'indicators') and strategy.indicators:
            self.use_new_architecture = True
        else:
            self.use_new_architecture = False
        super().init_exit(strategy, additional_params)

    def _init_indicators(self):
        """Initialize indicators (legacy architecture only)."""
        if self.use_new_architecture:
            return

        if not hasattr(self, "strategy"):
            return

        try:
            self.atr_fast = bt.indicators.ATR(self.strategy.data, period=self.p_fast)
            self.register_indicator("exit_atr_fast", self.atr_fast)

            self.atr_slow = bt.indicators.ATR(self.strategy.data, period=self.p_slow)
            self.register_indicator("exit_atr_slow", self.atr_slow)

            if self.use_htf_atr:
                self.atr_htf = bt.indicators.ATR(self.strategy.data, period=max(self.p_fast, self.p_slow))
                self.register_indicator("exit_atr_htf", self.atr_htf)

        except Exception:
            logger.exception("Error initializing ATR indicators:")
            raise

    def get_minimum_lookback(self) -> int:
        """Returns the minimum number of bars required."""
        return max(self.p_fast, self.p_slow, self.swing_lookback * 2)

    def are_indicators_ready(self) -> bool:
        """Check indicator readiness."""
        required = ['exit_atr_fast', 'exit_atr_slow']
        if self.use_htf_atr:
            required.append('exit_atr_htf')

        if self.use_new_architecture:
            return all(alias in getattr(self.strategy, 'indicators', {}) for alias in required)
        else:
            return all(alias in self.indicators for alias in required)

    def on_entry(self, entry_price: float, entry_time, position_size: float, direction: str):
        """Called when a position is entered."""
        if math.isnan(entry_price) or math.isinf(entry_price):
            return

        self.entry_price = entry_price
        self.entry_time = entry_time
        self.direction = direction.lower()
        self.state = ExitState.INIT
        self.k_current = self.k_init

        atr_eff = self._get_effective_atr()
        self.initial_risk = self.k_init * atr_eff

        if self.direction == "long":
            self.current_stop = entry_price - self.initial_risk
        else:
            self.current_stop = entry_price + self.initial_risk

        self.highest_high_since_entry = entry_price
        self.lowest_low_since_entry = entry_price
        self.pt_levels_hit = set()
        logger.debug(f"Advanced ATR Entry: SL={self.current_stop:.2f}")

    def _get_effective_atr(self) -> float:
        """Calculate effective ATR using multi-timeframe aggregation."""
        try:
            if self.use_new_architecture:
                atr_fast_val = self.get_indicator('exit_atr_fast')
                atr_slow_val = self.get_indicator('exit_atr_slow')
                atr_htf_val = self.get_indicator('exit_atr_htf') if self.use_htf_atr else 0.0
            else:
                atr_fast_val = self.indicators['exit_atr_fast'][0]
                atr_slow_val = self.indicators['exit_atr_slow'][0]
                atr_htf_val = self.indicators['exit_atr_htf'][0] if self.use_htf_atr else 0.0

            atr_eff = max(
                self.alpha_fast * (atr_fast_val or 0),
                self.alpha_slow * (atr_slow_val or 0),
                self.alpha_htf * (atr_htf_val or 0),
                self.atr_floor
            )
            return atr_eff if not math.isnan(atr_eff) else self.atr_floor
        except Exception:
            return self.atr_floor

    def should_exit(self) -> bool:
        """Check if position should be exited."""
        if not self.strategy.position:
            return False

        if not self.are_indicators_ready():
            return False

        try:
            current_low = self.strategy.data.low[0]
            current_high = self.strategy.data.high[0]
            current_bar = len(self.strategy.data)

            self.highest_high_since_entry = max(self.highest_high_since_entry, current_high)
            self.lowest_low_since_entry = min(self.lowest_low_since_entry, current_low)
            self.price_history.append((current_high, current_low))

            # Stop hit check
            if self.direction == "long":
                if current_low <= self.current_stop:
                    self.strategy.current_exit_reason = "stop_hit"
                    return True
            else:
                if current_high >= self.current_stop:
                    self.strategy.current_exit_reason = "stop_hit"
                    return True

            # Trailing stop update
            if current_bar - self.last_trail_bar >= self.max_trail_freq:
                self._update_stop_logic(current_bar)

            return False
        except Exception:
            logger.exception("Error in AdvancedATRExitMixin.should_exit")
            return False

    def _update_stop_logic(self, current_bar: int):
        """Internal trailing stop update logic."""
        atr_eff = self._get_effective_atr()
        current_profit = (self.strategy.data.close[0] - self.entry_price) if self.direction == "long" else (self.entry_price - self.strategy.data.close[0])
        profit_in_R = current_profit / self.initial_risk if self.initial_risk > 0 else 0

        # State transitions
        if self.state == ExitState.INIT and profit_in_R >= self.arm_at_R:
            self.state = ExitState.PHASE1
            self.k_current = self.k_run
        elif self.state == ExitState.PHASE1 and profit_in_R >= self.phase2_at_R:
            self.state = ExitState.PHASE2
            self.k_current = self.k_phase2

        # Calculate candidate
        distance = self.k_current * atr_eff
        if self.anchor == "high":
            ref_price = self.highest_high_since_entry if self.direction == "long" else self.lowest_low_since_entry
        else:
            ref_price = self.strategy.data.close[0]

        if self.direction == "long":
            candidate = ref_price - distance
            self.current_stop = max(self.current_stop, candidate)
        else:
            candidate = ref_price + distance
            self.current_stop = min(self.current_stop, candidate)

        self.last_trail_bar = current_bar

    def get_exit_reason(self) -> str:
        return getattr(self.strategy, 'current_exit_reason', 'stop_hit')
