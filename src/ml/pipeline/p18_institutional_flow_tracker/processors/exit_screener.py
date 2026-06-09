"""
Exit Screener

Filters position delta rows to those that represent meaningful institutional
exits, removing noise from tiny positions and small reductions.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class ExitScreener:
    """
    Screens a position delta DataFrame for institutional exits that clear
    both a percentage-reduction threshold and a minimum position-size filter.
    """

    def __init__(
        self,
        exit_threshold_pct: float = 0.30,
        min_position_pct_of_portfolio: float = 0.005,
        min_position_value_usd: int = 25_000_000,
    ):
        """
        Args:
            exit_threshold_pct: Minimum fraction by which a position must be
                reduced to qualify as an exit (0.30 = 30 %).
            min_position_pct_of_portfolio: Minimum prior-quarter portfolio weight
                (0.005 = 0.5 %) required for the exit to be counted.
            min_position_value_usd: Minimum prior-quarter position value as an
                alternative to the portfolio-weight filter.
        """
        self._exit_threshold = exit_threshold_pct
        self._min_pct = min_position_pct_of_portfolio
        self._min_value = min_position_value_usd

    def screen(self, delta_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return rows that represent significant institutional exits.

        A row qualifies when:
          - exit_type is 'full_exit' OR delta_pct ≤ -exit_threshold_pct
          - AND (pct_of_portfolio_prev ≥ min_pct OR value_usd_prev ≥ min_value)

        Args:
            delta_df: Output of PositionDeltaCalculator.calculate().

        Returns:
            Filtered DataFrame with the same schema as the input.
        """
        if delta_df.empty:
            return pd.DataFrame()

        is_exit = (
            (delta_df["exit_type"] == "full_exit") |
            (delta_df["delta_pct"] <= -self._exit_threshold)
        )
        is_significant = (
            (delta_df["pct_of_portfolio_prev"] >= self._min_pct) |
            (delta_df["value_usd_prev"] >= self._min_value)
        )

        filtered = delta_df[is_exit & is_significant]
        result = filtered.copy()
        assert isinstance(result, pd.DataFrame)
        _logger.info(
            "Exit screener: %d exits from %d delta rows (threshold=%.0f%%, min_portfolio=%.1f%%)",
            len(result), len(delta_df),
            self._exit_threshold * 100, self._min_pct * 100,
        )
        return result
