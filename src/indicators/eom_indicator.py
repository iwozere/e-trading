"""
EOM (Ease of Movement) Indicator

This module implements the Ease of Movement (EOM) indicator, which relates
price movement to volume, identifying the ease with which price moves.

The EOM indicator is calculated as:
1. Distance Moved = (High + Low)/2 - (Previous High + Previous Low)/2
2. Box Ratio = (Volume / scale) / (High - Low)
3. EOM = Distance Moved / Box Ratio
4. EOM_SMA = SMA(EOM, period)

A positive EOM indicates that price is moving upward with relative ease,
while a negative EOM indicates downward price movement with ease.
"""

import backtrader as bt
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EOMIndicator(bt.Indicator):
    """
    Ease of Movement (EOM) Indicator

    Relates price movement to volume to identify the ease with which price moves.

    Parameters:
        timeperiod (int): Period for SMA smoothing (default: 14)
        scale (float): Scaling factor for volume normalization (default: 100000000)

    Lines:
        eom: Smoothed EOM value
    """

    lines = ('eom',)
    params = (
        ('timeperiod', 14),
        ('scale', 100000000.0),
    )

    def __init__(self):
        """Initialize the EOM indicator"""
        # Validate parameters
        if self.p.timeperiod <= 0:
            _logger.warning(
                "Invalid timeperiod %s for EOM, using default 14",
                self.p.timeperiod
            )
            self.p.timeperiod = 14

        if self.p.scale <= 0:
            _logger.warning(
                "Invalid scale %s for EOM, using default 100000000",
                self.p.scale
            )
            self.p.scale =100000000.0

        # Calculate mid-point (High + Low) / 2
        mid_point = (self.data.high + self.data.low) / 2.0

        # Calculate Distance Moved
        distance_moved = mid_point - mid_point(-1)

        # Calculate Box Ratio
        # Box Ratio = (Volume / scale) / (High - Low)
        high_low_range = self.data.high - self.data.low

        # Avoid division by zero
        # If high == low or volume == 0, set box_ratio to a very large number
        # This will make EOM approach zero (distance_moved / large_number ≈ 0)
        box_ratio = bt.If(
            bt.And(high_low_range > 0.0000001, self.data.volume > 0),
            (self.data.volume / self.p.scale) / high_low_range,
            1e10  # Very large number to make raw_eom ≈ 0
        )

        # Calculate raw EOM
        # EOM = Distance Moved / Box Ratio
        # If box_ratio is very large, EOM will be very small (≈ 0)
        raw_eom = distance_moved / box_ratio

        # Smooth with SMA
        self.lines.eom = bt.indicators.SMA(raw_eom, period=self.p.timeperiod)

        _logger.debug(
            "EOM indicator initialized with timeperiod=%s, scale=%s",
            self.p.timeperiod,
            self.p.scale
        )


class EOM(EOMIndicator):
    """Alias for EOMIndicator for convenience"""
    pass
