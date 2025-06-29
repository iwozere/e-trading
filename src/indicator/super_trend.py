"""
Super Trend Indicator Module
---------------------------

This module implements the Super Trend technical indicator for use in trading strategies. The Super Trend indicator is used to identify the prevailing market trend and generate buy/sell signals based on price and volatility.

Main Features:
- Calculate Super Trend values for a given price series
- Generate trend direction and signal outputs
- Suitable for integration with trading and backtesting frameworks

Functions/Classes:
- super_trend: Main function to compute the Super Trend indicator
"""


import backtrader as bt
import numpy as np


# Custom SuperTrend Indicator
class SuperTrend(bt.Indicator):
    """SuperTrend indicator implementation"""

    lines = ("super_trend", "direction", "upper_band", "lower_band")
    params = (
        ("period", 10),
        ("multiplier", 3.0),
        ("use_talib", False),
    )

    def __init__(self):
        """Initialize the SuperTrend indicator"""
        super(SuperTrend, self).__init__()

        # Initialize ATR indicator
        if self.p.use_talib:
            try:
                import talib

                # Convert data to numpy arrays
                high_data = np.array(self.data.high.get(size=len(self.data)))
                low_data = np.array(self.data.low.get(size=len(self.data)))
                close_data = np.array(self.data.close.get(size=len(self.data)))

                # Calculate ATR using TA-Lib
                atr_values = talib.ATR(
                    high_data, low_data, close_data, timeperiod=self.p.period
                )

                # Create ATR indicator
                self.atr = bt.indicators.ATR(
                    self.data, period=self.p.period, plot=False
                )

                # Update ATR values one by one
                for i, value in enumerate(atr_values):
                    if i < len(self.atr.lines[0]):
                        self.atr.lines[0][i] = value
            except ImportError:
                self.log("TA-Lib not available, falling back to Backtrader ATR")
                self.atr = bt.indicators.ATR(self.data, period=self.p.period)
        else:
            self.atr = bt.indicators.ATR(self.data, period=self.p.period)

    def next(self):
        """Calculate next value of SuperTrend"""
        if len(self) == 1:
            # First bar - initialize values
            self.lines.upper_band[0] = (
                self.data.high[0] + self.data.low[0]
            ) / 2 + self.p.multiplier * self.atr[0]
            self.lines.lower_band[0] = (
                self.data.high[0] + self.data.low[0]
            ) / 2 - self.p.multiplier * self.atr[0]
            self.lines.super_trend[0] = self.lines.upper_band[0]
            self.lines.direction[0] = 1
            return

        # Calculate basic upper and lower bands
        basic_ub = (
            self.data.high[0] + self.data.low[0]
        ) / 2 + self.p.multiplier * self.atr[0]
        basic_lb = (
            self.data.high[0] + self.data.low[0]
        ) / 2 - self.p.multiplier * self.atr[0]

        # Calculate final upper and lower bands
        if (
            basic_ub < self.lines.upper_band[-1]
            or self.data.close[-1] > self.lines.upper_band[-1]
        ):
            self.lines.upper_band[0] = basic_ub
        else:
            self.lines.upper_band[0] = self.lines.upper_band[-1]

        if (
            basic_lb > self.lines.lower_band[-1]
            or self.data.close[-1] < self.lines.lower_band[-1]
        ):
            self.lines.lower_band[0] = basic_lb
        else:
            self.lines.lower_band[0] = self.lines.lower_band[-1]

        # Calculate SuperTrend and direction
        if self.lines.super_trend[-1] == self.lines.upper_band[-1]:
            if self.data.close[0] > self.lines.upper_band[0]:
                self.lines.super_trend[0] = self.lines.upper_band[0]
                self.lines.direction[0] = 1
            else:
                self.lines.super_trend[0] = self.lines.lower_band[0]
                self.lines.direction[0] = -1
        else:
            if self.data.close[0] < self.lines.lower_band[0]:
                self.lines.super_trend[0] = self.lines.lower_band[0]
                self.lines.direction[0] = -1
            else:
                self.lines.super_trend[0] = self.lines.upper_band[0]
                self.lines.direction[0] = 1 