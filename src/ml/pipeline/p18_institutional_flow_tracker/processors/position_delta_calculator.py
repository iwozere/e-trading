"""
Position Delta Calculator

Computes quarter-over-quarter changes in institutional equity holdings by
comparing two consecutive 13F snapshots loaded from the cache.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_EXIT_TYPES = {
    "full_exit": "full_exit",
    "partial_exit": "partial_exit",
    "new_position": "new_position",
    "unchanged": "unchanged",
}


class PositionDeltaCalculator:
    """
    Computes Q-over-Q position delta for all institutions in the 13F cache.

    Inputs are the per-institution CSV.gz holdings files already cached by
    EdgarDownloader.  Outputs are saved to the pipeline results directory.
    """

    def __init__(self, results_dir: Path):
        """
        Args:
            results_dir: Directory where delta CSV files are written.
        """
        self._results_dir = results_dir

    def calculate(
        self,
        current_quarter_df: pd.DataFrame,
        prior_quarter_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute position deltas between two quarterly snapshots.

        Args:
            current_quarter_df: Holdings for quarter N, with at minimum columns:
                cik, ticker, shares, value_usd, pct_of_portfolio, institution_name.
            prior_quarter_df: Holdings for quarter N-1, same schema.

        Returns:
            DataFrame with columns: cik, institution_name, ticker, shares_prev,
            shares_curr, delta_pct, exit_type, value_usd_prev, pct_of_portfolio_prev.
            Rows where both snapshots have zero shares are excluded.
        """
        if current_quarter_df.empty and prior_quarter_df.empty:
            return pd.DataFrame()

        curr = current_quarter_df.copy()
        prev = prior_quarter_df.copy()

        # Normalise column names
        for df in (curr, prev):
            df["cik"] = df["cik"].astype(str)
            df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

        merged = pd.merge(
            prev[["cik", "institution_name", "ticker", "shares", "value_usd", "pct_of_portfolio"]],
            curr[["cik", "ticker", "shares", "value_usd", "pct_of_portfolio"]],
            on=["cik", "ticker"],
            how="outer",
            suffixes=("_prev", "_curr"),
        )

        merged["shares_prev"] = pd.to_numeric(merged["shares_prev"], errors="coerce").fillna(0.0)
        merged["shares_curr"] = pd.to_numeric(merged["shares_curr"], errors="coerce").fillna(0.0)
        merged["value_usd_prev"] = pd.to_numeric(merged["value_usd_prev"], errors="coerce").fillna(0.0)
        merged["pct_of_portfolio_prev"] = pd.to_numeric(merged["pct_of_portfolio_prev"], errors="coerce").fillna(0.0)

        # Fill institution_name from prev side (or curr if new position)
        if "institution_name_curr" in merged.columns:
            merged["institution_name"] = merged["institution_name"].fillna(merged["institution_name_curr"])
            merged.drop(columns=["institution_name_curr"], inplace=True, errors="ignore")

        merged["exit_type"] = merged.apply(_classify_exit, axis=1)
        merged["delta_pct"] = merged.apply(_compute_delta_pct, axis=1)

        # Drop rows where both sides are empty (shouldn't happen but guard)
        mask = ~((merged["shares_prev"] == 0) & (merged["shares_curr"] == 0))
        result = merged[mask].copy()
        assert isinstance(result, pd.DataFrame)

        _logger.info(
            "Delta calculation: %d positions total — %d full exits, %d partial exits, %d new",
            len(result),
            (result["exit_type"] == "full_exit").sum(),
            (result["exit_type"] == "partial_exit").sum(),
            (result["exit_type"] == "new_position").sum(),
        )
        return result


def _classify_exit(row: pd.Series) -> str:
    prev, curr = row["shares_prev"], row["shares_curr"]
    if prev > 0 and curr == 0:
        return _EXIT_TYPES["full_exit"]
    if prev == 0 and curr > 0:
        return _EXIT_TYPES["new_position"]
    if prev > 0 and curr < prev:
        return _EXIT_TYPES["partial_exit"]
    return _EXIT_TYPES["unchanged"]


def _compute_delta_pct(row: pd.Series) -> float:
    prev = float(row["shares_prev"])
    curr = float(row["shares_curr"])
    if prev == 0:
        return float("inf") if curr > 0 else 0.0
    return (curr - prev) / prev
