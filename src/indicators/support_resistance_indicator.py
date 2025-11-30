"""
Support and Resistance Indicator

This module implements a dynamic Support and Resistance indicator based on
swing high/low detection.

The indicator detects swing points using a lookback approach:
- Swing High: A high that is higher than N bars before and after
- Swing Low: A low that is lower than N bars before and after

From these swing points, it calculates:
- Nearest Resistance: The closest swing high above the current price
- Nearest Support: The closest swing low below the current price
"""

import backtrader as bt
from collections import deque
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class SupportResistanceIndicator(bt.Indicator):
    """
    Support and Resistance Indicator

    Detects swing highs/lows and calculates nearest support/resistance levels.

    Parameters:
        lookback_bars (int): Number of bars to look back/forward for swing detection (default: 2)
        max_swings (int): Maximum number of swing points to keep in memory (default: 50)

    Lines:
        resistance: Nearest resistance level above current price (or NaN if none found)
        support: Nearest support level below current price (or NaN if none found)
    """

    lines = ('resistance', 'support')
    params = (
        ('lookback_bars', 2),
        ('max_swings', 50),
    )

    def __init__(self):
        """Initialize the Support/Resistance indicator"""
        # Validate parameters
        if self.p.lookback_bars < 1:
            _logger.warning(
                "Invalid lookback_bars %s for S/R, using default 2",
                self.p.lookback_bars
            )
            self.p.lookback_bars = 2

        if self.p.max_swings < 10:
            _logger.warning(
                "max_swings too small %s for S/R, using minimum 10",
                self.p.max_swings
            )
            self.p.max_swings = 10

        # Storage for historical prices
        self.highs = deque(maxlen=self.p.lookback_bars * 2 + 10)
        self.lows = deque(maxlen=self.p.lookback_bars * 2 + 10)

        # Storage for detected swing points
        self.swing_highs = deque(maxlen=self.p.max_swings)
        self.swing_lows = deque(maxlen=self.p.max_swings)

        _logger.debug(
            "S/R indicator initialized with lookback_bars=%s, max_swings=%s",
            self.p.lookback_bars,
            self.p.max_swings
        )

    def next(self):
        """Process each new bar"""
        # Append current bar's high and low
        self.highs.append(self.data.high[0])
        self.lows.append(self.data.low[0])

        # Detect new swing points if we have enough data
        if len(self.highs) >= (self.p.lookback_bars * 2 + 1):
            self._detect_swings()

        # Calculate nearest support/resistance
        current_price = self.data.close[0]
        resistance = self._nearest_resistance(current_price)
        support = self._nearest_support(current_price)

        # Set line values (use NaN if no level found)
        self.lines.resistance[0] = resistance if resistance is not None else float('nan')
        self.lines.support[0] = support if support is not None else float('nan')

    def _detect_swings(self):
        """
        Detect swing highs and lows using N-bar lookback/lookahead.

        A swing high at index i is detected when:
        - highs[i] > all highs in [i-N, i-1]
        - highs[i] > all highs in [i+1, i+N]

        Similar logic for swing lows.
        """
        n = self.p.lookback_bars

        # Check the bar at index that's N bars from the end
        # (we need N bars after it for confirmation)
        check_index = len(self.highs) - n - 1

        if check_index < n:
            return  # Not enough data yet

        # Get values
        h = list(self.highs)
        l = list(self.lows)

        center_high = h[check_index]
        center_low = l[check_index]

        # Check for swing high
        is_swing_high = True
        for offset in range(1, n + 1):
            # Check bars before and after
            if check_index - offset >= 0:
                if h[check_index - offset] >= center_high:
                    is_swing_high = False
                    break
            if check_index + offset < len(h):
                if h[check_index + offset] >= center_high:
                    is_swing_high = False
                    break

        if is_swing_high:
            # Avoid duplicates (only add if significantly different from recent swings)
            if not self.swing_highs or abs(center_high - self.swing_highs[-1]) > center_high * 0.0001:
                self.swing_highs.append(center_high)
                _logger.debug("Detected swing high at %s", center_high)

        # Check for swing low
        is_swing_low = True
        for offset in range(1, n + 1):
            # Check bars before and after
            if check_index - offset >= 0:
                if l[check_index - offset] <= center_low:
                    is_swing_low = False
                    break
            if check_index + offset < len(l):
                if l[check_index + offset] <= center_low:
                    is_swing_low = False
                    break

        if is_swing_low:
            # Avoid duplicates
            if not self.swing_lows or abs(center_low - self.swing_lows[-1]) > center_low * 0.0001:
                self.swing_lows.append(center_low)
                _logger.debug("Detected swing low at %s", center_low)

    def _nearest_resistance(self, price):
        """
        Find the nearest resistance level above the current price.

        Args:
            price: Current price

        Returns:
            float: Nearest resistance level, or None if no resistance found
        """
        if not self.swing_highs:
            return None

        # Find swing highs above current price
        resistances = [high for high in self.swing_highs if high > price]

        if not resistances:
            return None

        # Return the minimum (closest to current price)
        return min(resistances)

    def _nearest_support(self, price):
        """
        Find the nearest support level below the current price.

        Args:
            price: Current price

        Returns:
            float: Nearest support level, or None if no support found
        """
        if not self.swing_lows:
            return None

        # Find swing lows below current price
        supports = [low for low in self.swing_lows if low < price]

        if not supports:
            return None

        # Return the maximum (closest to current price)
        return max(supports)


class SupportResistance(SupportResistanceIndicator):
    """Alias for SupportResistanceIndicator for convenience"""
    pass
