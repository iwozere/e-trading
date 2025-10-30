"""
Advanced ATR-Based Exit Mixin

This module implements a sophisticated, volatility-adaptive trailing stop exit strategy
based on the technical specification in advanced_exit.md. It provides:

- Multi-timeframe ATR calculation with effective ATR aggregation
- State machine with break-even, phase switching, and structural ratcheting
- Time-based tightening and noise filtering
- Partial take-profit capabilities
- Comprehensive logging and metrics

The strategy is designed for both backtesting and live trading with consistent behavior.
"""

import backtrader as bt
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from collections import deque
import numpy as np

from src.strategy.exit.base_exit_mixin import BaseExitMixin
from src.indicators.adapters.backtrader_wrappers import UnifiedATRIndicator
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

    Implements a sophisticated volatility-adaptive trailing stop with:
    - Multi-timeframe ATR calculation
    - State machine with break-even and phase switching
    - Structural ratcheting based on swing levels
    - Time-based tightening for stagnant markets
    - Partial take-profit capabilities
    - Comprehensive noise filtering
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

        # Core parameters
        self.anchor = self.get_param("anchor", "high")  # high, close, mid
        self.update_on = self.get_param("update_on", "bar_close")  # bar_close, intrabar, high_low
        self.k_init = self.get_param("k_init", 2.5)
        self.k_run = self.get_param("k_run", 2.0)
        self.k_phase2 = self.get_param("k_phase2", 1.5)

        # ATR parameters
        self.p_fast = self.get_param("p_fast", 7)
        self.p_slow = self.get_param("p_slow", 21)
        self.use_htf_atr = self.get_param("use_htf_atr", True)
        self.htf = self.get_param("htf", "4h")
        self.alpha_fast = self.get_param("alpha_fast", 1.0)
        self.alpha_slow = self.get_param("alpha_slow", 1.0)
        self.alpha_htf = self.get_param("alpha_htf", 1.0)
        self.atr_floor = self.get_param("atr_floor", 0.0)

        # Break-even and phases
        self.arm_at_R = self.get_param("arm_at_R", 1.0)
        self.breakeven_offset_atr = self.get_param("breakeven_offset_atr", 0.0)
        self.phase2_at_R = self.get_param("phase2_at_R", 2.0)

        # Structural ratchet
        self.use_swing_ratchet = self.get_param("use_swing_ratchet", True)
        self.swing_lookback = self.get_param("swing_lookback", 10)
        self.struct_buffer_atr = self.get_param("struct_buffer_atr", 0.25)

        # Time-based tightening
        self.tighten_if_stagnant_bars = self.get_param("tighten_if_stagnant_bars", 20)
        self.tighten_k_factor = self.get_param("tighten_k_factor", 0.8)
        self.min_bars_between_tighten = self.get_param("min_bars_between_tighten", 5)

        # Noise and step filters
        self.min_stop_step = self.get_param("min_stop_step", 0.0)
        self.noise_filter_atr = self.get_param("noise_filter_atr", 0.0)
        self.max_trail_freq = self.get_param("max_trail_freq", 1)

        # Partial take-profit
        self.pt_levels_R = self.get_param("pt_levels_R", [1.0, 2.0])
        self.pt_sizes = self.get_param("pt_sizes", [0.33, 0.33])
        self.retune_after_pt = self.get_param("retune_after_pt", True)

        # Execution
        self.fill_policy = self.get_param("fill_policy", "stop_mkt")
        self.tick_size = self.get_param("tick_size", 0.01)

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

        # ATR indicators
        self.atr_fast = None
        self.atr_slow = None
        self.atr_htf = None

        # Price history for swing detection
        self.price_history = deque(maxlen=self.swing_lookback * 2)

        # Logging
        self.exit_log = []

    def get_required_params(self) -> list:
        """Get list of required parameters for this exit mixin."""
        return []  # All parameters have defaults, so none are strictly required

    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this exit mixin."""
        return {
            # Core parameters
            "anchor": "high",
            "update_on": "bar_close",
            "k_init": 2.5,
            "k_run": 2.0,
            "k_phase2": 1.5,

            # ATR parameters
            "p_fast": 7,
            "p_slow": 21,
            "use_htf_atr": True,
            "htf": "4h",
            "alpha_fast": 1.0,
            "alpha_slow": 1.0,
            "alpha_htf": 1.0,
            "atr_floor": 0.0,

            # Break-even and phases
            "arm_at_R": 1.0,
            "breakeven_offset_atr": 0.0,
            "phase2_at_R": 2.0,

            # Structural ratchet
            "use_swing_ratchet": True,
            "swing_lookback": 10,
            "struct_buffer_atr": 0.25,

            # Time-based tightening
            "tighten_if_stagnant_bars": 20,
            "tighten_k_factor": 0.8,
            "min_bars_between_tighten": 5,

            # Noise and step filters
            "min_stop_step": 0.0,
            "noise_filter_atr": 0.0,
            "max_trail_freq": 1,

            # Partial take-profit
            "pt_levels_R": [1.0, 2.0],
            "pt_sizes": [0.33, 0.33],
            "retune_after_pt": True,

            # Execution
            "fill_policy": "stop_mkt",
            "tick_size": 0.01,
        }

    def init_exit(self, strategy):
        """Initialize the exit mixin with strategy reference."""
        super().init_exit(strategy)

        # Initialize ATR indicators
        self._init_indicators()

        logger.debug("AdvancedATRExitMixin initialized with state: %s", self.state)

    def _init_indicators(self):
        """Initialize ATR indicators."""
        try:
            # Fast ATR
            self.atr_fast = bt.indicators.ATR(
                self.strategy.data,
                period=self.p_fast
            )
            self.register_indicator("atr_fast", self.atr_fast)

            # Slow ATR
            self.atr_slow = bt.indicators.ATR(
                self.strategy.data,
                period=self.p_slow
            )
            self.register_indicator("atr_slow", self.atr_slow)

            # Higher timeframe ATR (if enabled)
            if self.use_htf_atr:
                # For now, use same timeframe - in live trading this would be different
                self.atr_htf = bt.indicators.ATR(
                    self.strategy.data,
                    period=max(self.p_fast, self.p_slow)
                )
                self.register_indicator("atr_htf", self.atr_htf)

            logger.debug("ATR indicators initialized successfully")

        except Exception as e:
            logger.exception("Error initializing ATR indicators:")
            raise

    def are_indicators_ready(self) -> bool:
        """Check if all indicators are ready."""
        if not super().are_indicators_ready():
            return False

        try:
            # Check ATR indicators
            if self.atr_fast is None or self.atr_slow is None:
                return False

            # Check if we have enough data
            if len(self.strategy.data) < max(self.p_fast, self.p_slow):
                return False

            # Check ATR values
            if (self.atr_fast[0] is None or self.atr_fast[0] <= 0 or
                self.atr_slow[0] is None or self.atr_slow[0] <= 0):
                return False

            if self.use_htf_atr and hasattr(self, 'atr_htf'):
                if self.atr_htf[0] is None or self.atr_htf[0] <= 0:
                    return False

            return True

        except Exception as e:
            logger.exception("Error checking indicator readiness:")
            return False

    def on_entry(self, entry_price: float, entry_time, position_size: float, direction: str):
        """Called when a position is entered."""
        import math

        # Validate entry_price
        if math.isnan(entry_price) or math.isinf(entry_price):
            logger.error("Invalid entry_price: %s, cannot proceed with entry", entry_price)
            return

        self.entry_price = entry_price
        self.entry_time = entry_time
        self.direction = direction.lower()  # Store direction for proper stop hit detection
        self.state = ExitState.INIT

        # Calculate initial risk
        if direction.lower() == "long":
            self.initial_risk = entry_price - (entry_price - self.k_init * self._get_effective_atr())
        else:
            self.initial_risk = (entry_price + self.k_init * self._get_effective_atr()) - entry_price

        # Set initial stop
        self._set_initial_stop(direction)

        # Initialize tracking variables
        self.highest_high_since_entry = entry_price
        self.lowest_low_since_entry = entry_price
        self.last_trail_bar = 0
        self.last_tighten_bar = 0
        self.pt_levels_hit = set()
        self.k_current = self.k_init

        logger.debug("Position entered: %s at %s, initial stop: %s", direction, entry_price, self.current_stop)

    def _get_effective_atr(self) -> float:
        """Calculate effective ATR using multi-timeframe aggregation."""
        try:
            import math

            atr_fast_val = self.atr_fast[0] if self.atr_fast[0] is not None else 0.0
            atr_slow_val = self.atr_slow[0] if self.atr_slow[0] is not None else 0.0

            # Fix: Properly check if atr_htf exists and has valid data
            atr_htf_val = 0.0
            if self.use_htf_atr and hasattr(self, 'atr_htf') and self.atr_htf[0] is not None:
                atr_htf_val = self.atr_htf[0]

            # Check for NaN or infinite values
            if math.isnan(atr_fast_val) or math.isinf(atr_fast_val):
                atr_fast_val = 0.0
            if math.isnan(atr_slow_val) or math.isinf(atr_slow_val):
                atr_slow_val = 0.0
            if math.isnan(atr_htf_val) or math.isinf(atr_htf_val):
                atr_htf_val = 0.0

            # Calculate effective ATR
            atr_eff = max(
                self.alpha_fast * atr_fast_val,
                self.alpha_slow * atr_slow_val,
                self.alpha_htf * atr_htf_val,
                self.atr_floor
            )

            # Final check for NaN result
            if math.isnan(atr_eff) or math.isinf(atr_eff):
                logger.warning("Effective ATR calculation resulted in invalid value: %s, using floor value", atr_eff)
                return self.atr_floor

            return atr_eff

        except Exception as e:
            logger.exception("Error calculating effective ATR:")
            return self.atr_floor

    def _set_initial_stop(self, direction: str):
        """Set initial stop loss."""
        atr_eff = self._get_effective_atr()

        if direction.lower() == "long":
            self.current_stop = self.entry_price - self.k_init * atr_eff
        else:
            self.current_stop = self.entry_price + self.k_init * atr_eff

        # Round to tick size
        self.current_stop = self._round_to_tick(self.current_stop)

    def should_exit(self) -> bool:
        """Check if position should be exited."""
        if not self.are_indicators_ready():
            return False

        try:
            current_price = self.strategy.data.close[0]
            current_low = self.strategy.data.low[0]
            current_high = self.strategy.data.high[0]
            current_bar = len(self.strategy.data)

            # Update price tracking
            self._update_price_tracking(current_high, current_low)

            # Check for stop hit
            if self._is_stop_hit(current_low, current_high):
                return True

            # Update trailing stop
            self._update_trailing_stop(current_bar)

            # Check for partial take-profit
            pt_exit = self._check_partial_take_profit(current_price)
            if pt_exit[0]:
                return True

            return False

        except Exception as e:
            logger.exception("Error in should_exit:")
            return False

    def get_exit_reason(self) -> str:
        """Get the reason for exit (called after should_exit returns True)."""
        try:
            current_price = self.strategy.data.close[0]
            current_low = self.strategy.data.low[0]
            current_high = self.strategy.data.high[0]

            # Check for stop hit
            if self._is_stop_hit(current_low, current_high):
                return "stop_hit"

            # Check for partial take-profit
            pt_exit = self._check_partial_take_profit(current_price)
            if pt_exit[0]:
                return pt_exit[1]

            return "unknown"

        except Exception as e:
            logger.exception("Error in get_exit_reason:")
            return "error"

    def _update_price_tracking(self, high: float, low: float):
        """Update highest high and lowest low since entry."""
        self.highest_high_since_entry = max(self.highest_high_since_entry, high)
        self.lowest_low_since_entry = min(self.lowest_low_since_entry, low)

        # Add to price history for swing detection
        self.price_history.append((high, low))

    def _is_stop_hit(self, low: float, high: float) -> bool:
        """Check if stop loss has been hit."""
        if not hasattr(self, 'direction') or not hasattr(self, 'current_stop'):
            return False

        if self.direction == "long":
            # For long positions, check if low went below stop
            return low <= self.current_stop
        else:
            # For short positions, check if high went above stop
            return high >= self.current_stop

    def _update_trailing_stop(self, current_bar: int):
        """Update trailing stop based on current state and conditions."""
        try:
            # Check if we should skip update due to frequency limit
            if current_bar - self.last_trail_bar < self.max_trail_freq:
                return

            # Get effective ATR
            atr_eff = self._get_effective_atr()

            # Check noise filter
            if self._should_skip_due_to_noise(atr_eff):
                return

            # Calculate candidate stop based on anchor
            candidate_stop = self._calculate_candidate_stop(atr_eff)

            # Apply structural ratchet if enabled
            if self.use_swing_ratchet:
                candidate_stop = self._apply_structural_ratchet(candidate_stop, atr_eff)

            # Apply state-specific logic
            candidate_stop = self._apply_state_logic(candidate_stop, atr_eff)

            # Apply time-based tightening
            candidate_stop = self._apply_time_tightening(candidate_stop, current_bar)

            # Ratchet the stop (never loosen)
            if self.direction == "long":
                # For long positions, stop can only move up (never down)
                new_stop = max(self.current_stop, candidate_stop)
            else:
                # For short positions, stop can only move down (never up)
                new_stop = min(self.current_stop, candidate_stop)

            # Check minimum step
            if abs(new_stop - self.current_stop) >= self.min_stop_step:
                self.current_stop = self._round_to_tick(new_stop)
                self.last_trail_bar = current_bar

                logger.debug("Stop updated to %s (candidate: %s)", self.current_stop, candidate_stop)

        except Exception as e:
            logger.exception("Error updating trailing stop:")

    def _should_skip_due_to_noise(self, atr_eff: float) -> bool:
        """Check if update should be skipped due to noise filter."""
        if self.noise_filter_atr <= 0:
            return False

        current_range = self.strategy.data.high[0] - self.strategy.data.low[0]
        return current_range < self.noise_filter_atr * atr_eff

    def _calculate_candidate_stop(self, atr_eff: float) -> float:
        """Calculate candidate stop based on anchor type."""
        distance = self.k_current * atr_eff

        if self.anchor == "high":
            anchor_price = self.highest_high_since_entry
        elif self.anchor == "close":
            anchor_price = self.strategy.data.close[0]
        elif self.anchor == "mid":
            anchor_price = (self.strategy.data.high[0] + self.strategy.data.low[0]) / 2
        else:
            anchor_price = self.highest_high_since_entry

        # Apply distance based on position direction
        if self.direction == "long":
            return anchor_price - distance
        else:
            return anchor_price + distance

    def _apply_structural_ratchet(self, candidate_stop: float, atr_eff: float) -> float:
        """Apply structural ratchet based on swing levels."""
        if len(self.price_history) < self.swing_lookback:
            return candidate_stop

        if self.direction == "long":
            # For long positions, find lowest low in lookback period
            recent_lows = [low for _, low in list(self.price_history)[-self.swing_lookback:]]
            last_swing_low = min(recent_lows)
            # Calculate structural stop below swing low
            struct_stop = last_swing_low - self.struct_buffer_atr * atr_eff
            # Return the more conservative (higher) stop
            return max(candidate_stop, struct_stop)
        else:
            # For short positions, find highest high in lookback period
            recent_highs = [high for high, _ in list(self.price_history)[-self.swing_lookback:]]
            last_swing_high = max(recent_highs)
            # Calculate structural stop above swing high
            struct_stop = last_swing_high + self.struct_buffer_atr * atr_eff
            # Return the more conservative (lower) stop
            return min(candidate_stop, struct_stop)

    def _apply_state_logic(self, candidate_stop: float, atr_eff: float) -> float:
        """Apply state-specific logic (break-even, phase switching)."""
        current_profit = self._calculate_current_profit()
        profit_in_R = current_profit / self.initial_risk if self.initial_risk > 0 else 0

        # Break-even logic
        if self.state == ExitState.INIT and profit_in_R >= self.arm_at_R:
            be_stop = self.entry_price + self.breakeven_offset_atr * atr_eff
            candidate_stop = max(candidate_stop, be_stop)
            self.state = ExitState.PHASE1
            self.k_current = self.k_run
            logger.debug("Break-even armed at R=%.2f", profit_in_R)

        # Phase 2 logic
        elif self.state == ExitState.PHASE1 and profit_in_R >= self.phase2_at_R:
            self.k_current = self.k_phase2
            self.state = ExitState.PHASE2
            logger.debug("Switched to Phase 2 at R=%.2f", profit_in_R)

        return candidate_stop

    def _apply_time_tightening(self, candidate_stop: float, current_bar: int) -> float:
        """Apply time-based tightening for stagnant markets."""
        if (current_bar - self.last_tighten_bar < self.min_bars_between_tighten or
            self.state not in [ExitState.PHASE1, ExitState.PHASE2]):
            return candidate_stop

        # Check for stagnation (no new high in specified bars)
        bars_since_new_high = current_bar - self._get_last_new_high_bar()

        if bars_since_new_high >= self.tighten_if_stagnant_bars:
            # Tighten the multiplier
            old_k = self.k_current
            self.k_current = max(0.8, self.k_current * self.tighten_k_factor)
            self.last_tighten_bar = current_bar

            # Recalculate candidate with new multiplier
            atr_eff = self._get_effective_atr()
            candidate_stop = self._calculate_candidate_stop(atr_eff)

            logger.debug("Time-based tightening: k %.2f -> %.2f", old_k, self.k_current)

        return candidate_stop

    def _get_last_new_high_bar(self) -> int:
        """Get the bar number when last new high was made."""
        # Simplified implementation - in reality would track this more precisely
        return len(self.strategy.data) - 5  # Placeholder

    def _calculate_current_profit(self) -> float:
        """Calculate current profit in price units."""
        current_price = self.strategy.data.close[0]
        if self.direction == "long":
            return current_price - self.entry_price
        else:
            return self.entry_price - current_price

    def _check_partial_take_profit(self, current_price: float) -> Tuple[bool, str]:
        """Check for partial take-profit levels."""
        if not self.pt_levels_R or not self.pt_sizes:
            return False, "no_pt"

        current_profit = self._calculate_current_profit()
        profit_in_R = current_profit / self.initial_risk if self.initial_risk > 0 else 0

        for i, level_R in enumerate(self.pt_levels_R):
            if level_R not in self.pt_levels_hit and profit_in_R >= level_R:
                self.pt_levels_hit.add(level_R)

                # Apply retuning after PT if enabled
                if self.retune_after_pt:
                    self.k_current = self.k_phase2
                    self.state = ExitState.PHASE2

                logger.debug("Partial take-profit hit at R=%.2f", level_R)
                return True, f"partial_tp_{i+1}"

        return False, "no_pt"

    def _round_to_tick(self, price: float) -> float:
        """Round price to tick size."""
        if self.tick_size <= 0:
            return price

        # Handle NaN values
        import math
        if math.isnan(price) or math.isinf(price):
            logger.warning("Invalid price value for rounding: %s, returning original value", price)
            return price

        return round(price / self.tick_size) * self.tick_size

    def get_current_stop(self) -> float:
        """Get current stop loss level."""
        return self.current_stop

    def get_state(self) -> str:
        """Get current state."""
        return self.state.value

    def get_exit_log(self) -> List[Dict[str, Any]]:
        """Get exit strategy log."""
        return self.exit_log.copy()

    def log_exit_event(self, event_type: str, details: Dict[str, Any]):
        """Log an exit strategy event."""
        log_entry = {
            "timestamp": self.strategy.data.datetime.datetime(),
            "symbol": getattr(self.strategy, 'symbol', 'Unknown'),
            "event_type": event_type,
            "state": self.state.value,
            "current_stop": self.current_stop,
            "entry_price": self.entry_price,
            "current_price": self.strategy.data.close[0],
            "k_current": self.k_current,
            "atr_eff": self._get_effective_atr(),
            **details
        }

        self.exit_log.append(log_entry)
        logger.debug("Exit event logged: %s", event_type)
