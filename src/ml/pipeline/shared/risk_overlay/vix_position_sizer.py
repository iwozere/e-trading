"""
VIX-based position sizer — shared risk overlay.

Computes target portfolio exposure (0.0–1.0) from a VIX time-series using
rolling Z-score tiers.  Extracted from P13 (p13_bdsh) so any pipeline can
apply the same VIX risk overlay without depending on P13 internals.

Typical usage::

    sizer = VixPositionSizer(z_lookback=63, entry_tiers=[
        {"z_threshold": 1.0, "allocation": 0.25},
        {"z_threshold": 1.5, "allocation": 0.25},
        {"z_threshold": 2.0, "allocation": 0.50},
    ], exit_z_threshold=0.0)
    exposures = sizer.compute_exposures(vix_series)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class VixTier:
    """A single VIX Z-score entry tier."""
    z_threshold: float
    allocation: float


@dataclass
class VixPositionSizerConfig:
    """Configuration for VixPositionSizer."""
    z_lookback: int = 63
    exit_z_threshold: float = 0.0
    entry_tiers: List[Dict[str, float]] = field(default_factory=lambda: [
        {"z_threshold": 1.0, "allocation": 0.25},
        {"z_threshold": 1.5, "allocation": 0.25},
        {"z_threshold": 2.0, "allocation": 0.50},
    ])


class VixPositionSizer:
    """
    VIX-based risk overlay that maps VIX Z-scores to portfolio exposure tiers.

    The overlay is designed to be pipeline-agnostic: it takes a raw VIX series
    and returns a same-length series of target exposures in [0.0, 1.0].

    Parameters
    ----------
    config : VixPositionSizerConfig
        Lookback window, exit threshold, and tier definitions.
    """

    def __init__(self, config: VixPositionSizerConfig) -> None:
        self.config = config
        self._sorted_tiers: List[VixTier] = sorted(
            [VixTier(**t) for t in config.entry_tiers],
            key=lambda t: t.z_threshold,
        )

    def compute_z_scores(self, vix_series: pd.Series) -> pd.Series:
        """
        Compute rolling Z-scores from a VIX series.

        Args:
            vix_series: Raw VIX close prices indexed by date.

        Returns:
            Same-length Series of Z-scores (NaN for the first z_lookback bars).
        """
        roll = vix_series.rolling(window=self.config.z_lookback)
        return (vix_series - roll.mean()) / roll.std()

    def _exposure_from_z(self, z: float) -> float:
        """Map a single Z-score to target exposure via the tier ladder."""
        if pd.isna(z) or z < self.config.exit_z_threshold:
            return 0.0
        total = sum(t.allocation for t in self._sorted_tiers if z > t.z_threshold)
        return min(1.0, total)

    def compute_exposures(self, vix_series: pd.Series) -> pd.Series:
        """
        Compute target exposures for each bar in *vix_series*.

        The exposure uses the **previous** bar's Z-score (shift(1)) to avoid
        look-ahead: a position entered on bar t is sized by the Z-score known
        at bar t-1 close.

        Args:
            vix_series: Raw VIX close prices indexed by date.

        Returns:
            Series of target exposures aligned to vix_series.index.
        """
        z_scores = self.compute_z_scores(vix_series)
        lagged_z = z_scores.shift(1)
        exposures = lagged_z.map(self._exposure_from_z)
        _logger.debug(
            "VixPositionSizer: computed exposures over %d bars "
            "(non-zero bars: %d, max_exposure: %.2f)",
            len(exposures),
            int((exposures > 0).sum()),
            float(exposures.max()) if len(exposures) > 0 else 0.0,
        )
        return exposures

    def scale_position_size(
        self,
        base_size: float,
        vix_series: pd.Series,
        as_of_date: Any,
    ) -> float:
        """
        Scale a single base position size by the VIX exposure at *as_of_date*.

        Convenience method for pipelines that compute base sizes independently
        and want to apply VIX scaling on-the-fly.

        Args:
            base_size: Unscaled position size (e.g. notional or number of shares).
            vix_series: VIX series covering at least z_lookback bars before as_of_date.
            as_of_date: The date for which to compute the scaling factor.

        Returns:
            Scaled position size (base_size * exposure).
        """
        exposures = self.compute_exposures(vix_series)
        if as_of_date not in exposures.index:
            _logger.warning(
                "VixPositionSizer.scale_position_size: as_of_date %s not in series index "
                "— returning base_size unchanged",
                as_of_date,
            )
            return base_size
        exposure = float(exposures.loc[as_of_date])
        return base_size * exposure
