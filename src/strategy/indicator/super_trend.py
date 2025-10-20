"""
SuperTrend Indicator - Temporary Implementation

This is a temporary implementation of SuperTrend indicator that maintains
compatibility with existing strategy code until SuperTrend is added to
the unified indicator service.
"""

import backtrader as bt
import numpy as np
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class SuperTrend(bt.Indicator):
    """
    SuperTrend indicator implementation for Backtrader.

    This is a temporary implementation that maintains compatibility
    with existing strategy code.
    """

    lines = ('super_trend', 'direction', 'upper_band', 'lower_band')
    params = (
        ('period', 10),
        ('multiplier', 3.0),
    )

    def __init__(self):
        # Calculate ATR
        self.atr = bt.indicators.ATR(self.data, period=self.params.period)

        # Calculate high-low midpoint
        self.hl2 = (self.data.high + self.data.low) / 2.0

        # Initialize lines
        self.lines.super_trend = bt.LineBuffer()
        self.lines.direction = bt.LineBuffer()
        self.lines.upper_band = bt.LineBuffer()
        self.lines.lower_band = bt.LineBuffer()

        logger.debug("SuperTrend indicator initialized with period=%d, multiplier=%.1f",
                    self.params.period, self.params.multiplier)

    def next(self):
        # Calculate basic upper and lower bands
        basic_upper = self.hl2[0] + (self.params.multiplier * self.atr[0])
        basic_lower = self.hl2[0] - (self.params.multiplier * self.atr[0])

        # Get previous values (or initialize)
        if len(self.lines.super_trend) == 0:
            # First calculation
            final_upper = basic_upper
            final_lower = basic_lower
            super_trend = basic_lower
            direction = 1
        else:
            prev_final_upper = self.lines.upper_band[-1]
            prev_final_lower = self.lines.lower_band[-1]
            prev_super_trend = self.lines.super_trend[-1]
            prev_direction = self.lines.direction[-1]

            # Calculate final upper and lower bands
            final_upper = basic_upper if basic_upper < prev_final_upper or self.data.close[-1] > prev_final_upper else prev_final_upper
            final_lower = basic_lower if basic_lower > prev_final_lower or self.data.close[-1] < prev_final_lower else prev_final_lower

            # Determine direction and SuperTrend value
            if prev_super_trend == prev_final_upper:
                direction = -1 if self.data.close[0] <= final_upper else 1
            else:
                direction = 1 if self.data.close[0] >= final_lower else -1

            super_trend = final_upper if direction == -1 else final_lower

        # Set line values
        self.lines.super_trend[0] = super_trend
        self.lines.direction[0] = direction
        self.lines.upper_band[0] = final_upper
        self.lines.lower_band[0] = final_lower