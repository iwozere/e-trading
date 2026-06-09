"""
Consensus Detector

Identifies stocks that have been exited (or significantly reduced) by multiple
institutions in the same quarter — the most actionable signal in the pipeline.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class ConsensusDetector:
    """
    Aggregates exit screener output by ticker to detect stocks being sold
    by multiple institutions simultaneously.
    """

    def __init__(self, min_institutions: int = 3):
        """
        Args:
            min_institutions: Minimum number of distinct institutions that must
                exit the same ticker to raise a consensus signal.
        """
        self._min_institutions = min_institutions

    def detect(self, exits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Group exits by ticker and return tickers with multi-institution consensus.

        Args:
            exits_df: Output of ExitScreener.screen(), must have columns:
                cik, ticker, value_usd_prev, delta_pct.

        Returns:
            DataFrame sorted by institution_count descending, with columns:
            ticker, institution_count, total_value_sold_usd, avg_exit_pct.
            Empty DataFrame if no ticker reaches the threshold.
        """
        if exits_df.empty:
            return pd.DataFrame()

        agg = (
            exits_df.groupby("ticker")
            .agg(
                institution_count=("cik", "nunique"),
                total_value_sold_usd=("value_usd_prev", "sum"),
                avg_exit_pct=("delta_pct", "mean"),
            )
            .reset_index()
        )

        above = agg[agg["institution_count"] >= self._min_institutions]
        result = above.copy()
        assert isinstance(result, pd.DataFrame)
        result = result.sort_values("institution_count", ascending=False).reset_index(drop=True)

        _logger.info(
            "Consensus detector: %d tickers with %d+ institution exits (from %d distinct tickers)",
            len(result), self._min_institutions, len(agg),
        )
        return result
