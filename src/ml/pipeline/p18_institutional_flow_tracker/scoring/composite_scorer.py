"""
Composite Scorer

Combines signals from all layers (13F consensus, volume anomaly, Form 4,
13D/G, seasonal calendar) into a single score per ticker.

Tickers exceeding the alert threshold are returned as actionable signals.
"""

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Seasonal redemption windows that increase distribution probability
_REDEMPTION_MONTHS = {
    10, 11, 12,  # Tax-loss harvesting / hedge-fund redemption gates (Dec 31)
    3,            # Fiscal year-end for many funds
    9,            # Q3 close / window dressing
}

# Graded large-exit award by total dollars sold (descending). A flat bonus made
# every $500M+ exit clear the alert bar, saturating the result (thousands of
# look-alike 65s). Tiering it lets a multi-billion-dollar liquidation outrank a
# borderline one and keeps modest exits below threshold.
_DEFAULT_LARGE_EXIT_TIERS_USD: List[Tuple[float, int]] = [
    (2_000_000_000, 25),
    (1_000_000_000, 18),
    (500_000_000, 10),
]

# Consensus breadth: extra credit per institution exiting beyond the detector's
# 3-institution minimum, so a stock 20 funds are dumping outranks one 3 are.
_DEFAULT_BREADTH_MIN_INSTITUTIONS = 3
_DEFAULT_BREADTH_POINTS_PER_INSTITUTION = 3
_DEFAULT_BREADTH_CAP = 15


class CompositeScorer:
    """
    Weights and sums active signals per ticker to produce a 0–115 composite score.

    Signal weights are configurable; defaults reflect the relative informativeness
    of each signal as described in the pipeline design doc.
    """

    def __init__(
        self,
        signal_weights: Optional[Dict[str, int]] = None,
        alert_threshold: int = 60,
        large_exit_tiers_usd: Optional[List[Tuple[float, int]]] = None,
        breadth_min_institutions: int = _DEFAULT_BREADTH_MIN_INSTITUTIONS,
        breadth_points_per_institution: int = _DEFAULT_BREADTH_POINTS_PER_INSTITUTION,
        breadth_cap: int = _DEFAULT_BREADTH_CAP,
    ):
        """
        Args:
            signal_weights: Override the default weight table.
            alert_threshold: Minimum composite score to flag a ticker as an alert.
            large_exit_tiers_usd: (threshold_usd, points) pairs in descending
                threshold order; the first tier a ticker's total dollars-sold meets
                is awarded. Defaults to _DEFAULT_LARGE_EXIT_TIERS_USD.
            breadth_min_institutions: Institution count at/below which no breadth
                bonus is given.
            breadth_points_per_institution: Points per institution above the minimum.
            breadth_cap: Maximum breadth bonus.
        """
        # Binary signal weights. The large-exit award is no longer here — it is
        # graded by dollar tier (see _large_exit_tiers); consensus breadth is a
        # separate graded bonus. Any "large_single_exit_500m" key in an override
        # is ignored.
        self._weights = signal_weights or {
            "consensus_exit_3plus": 40,
            "volume_spike_confirmed": 20,
            "form4_insider_sell": 10,
            "schedule_13dg_drop": 10,
            "seasonal_redemption_window": 5,
            "price_below_52w_high_15pct": 5,
        }
        self._threshold = alert_threshold
        self._large_exit_tiers = sorted(
            large_exit_tiers_usd or _DEFAULT_LARGE_EXIT_TIERS_USD,
            key=lambda t: t[0],
            reverse=True,
        )
        self._breadth_min = breadth_min_institutions
        self._breadth_per_institution = breadth_points_per_institution
        self._breadth_cap = breadth_cap

    def _large_exit_points(self, total_sold_usd: float) -> int:
        """Return the graded award for the first dollar tier the exit meets."""
        for threshold_usd, points in self._large_exit_tiers:
            if total_sold_usd >= threshold_usd:
                return points
        return 0

    def _breadth_points(self, institution_count: int) -> int:
        """Return the capped breadth bonus for institutions above the minimum."""
        extra = max(0, institution_count - self._breadth_min)
        return min(extra * self._breadth_per_institution, self._breadth_cap)

    def score(
        self,
        consensus_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        form4_df: pd.DataFrame,
        dg_df: pd.DataFrame,
        as_of_date: Optional[date] = None,
        price_proximity_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute composite scores for all tickers present in any input DataFrame.

        Args:
            consensus_df: Output of ConsensusDetector.detect() — columns: ticker,
                institution_count, total_value_sold_usd.
            volume_df: Output of VolumeAnomalyDetector.detect() — columns: ticker,
                volume_spike_ratio.
            form4_df: Output of Form4Monitor.get_significant_sells() — columns: ticker.
            dg_df: Output of Form4Monitor.get_13dg_drops() — columns: entity_name
                (used as a fuzzy match against ticker when no direct join available).
            as_of_date: Used to check seasonal windows. Defaults to today.
            price_proximity_df: Optional DataFrame with columns ticker and
                below_52w_high_15pct (bool). Tickers present and True receive +5.

        Returns:
            DataFrame of alerts (total_score >= alert_threshold), sorted by
            total_score then distribution magnitude (total_value_sold_usd,
            institution_count) descending. Columns: ticker, total_score,
            signals_active (int), institution_count (int), total_value_sold_usd
            (float), signal_detail (JSON string).
        """
        today = as_of_date or date.today()
        in_seasonal_window = today.month in _REDEMPTION_MONTHS

        # Build per-ticker score rows
        all_tickers: List[str] = list(set(
            list(consensus_df["ticker"].dropna().tolist() if not consensus_df.empty else []) +
            list(volume_df["ticker"].dropna().tolist() if not volume_df.empty else []) +
            list(form4_df["ticker"].dropna().tolist() if not form4_df.empty else [])
        ))

        if not all_tickers:
            return pd.DataFrame()

        rows = []
        for ticker in all_tickers:
            detail: Dict[str, bool] = {}
            score = 0
            institution_count = 0
            total_sold = 0.0

            # --- 13F consensus exit signal ---
            c_row = consensus_df[consensus_df["ticker"] == ticker] if not consensus_df.empty else pd.DataFrame()
            if not c_row.empty:
                detail["consensus_exit_3plus"] = True
                score += self._weights["consensus_exit_3plus"]

                institution_count = int(c_row.iloc[0].get("institution_count", 0) or 0)
                total_sold = float(c_row.iloc[0].get("total_value_sold_usd", 0) or 0)

                # Graded large-exit award (replaces the old flat $500M bonus).
                large_exit_points = self._large_exit_points(total_sold)
                if large_exit_points:
                    detail["large_single_exit"] = True
                    score += large_exit_points

                # Breadth: more institutions exiting → stronger consensus.
                breadth_points = self._breadth_points(institution_count)
                if breadth_points:
                    detail["consensus_breadth"] = True
                    score += breadth_points

            # --- Volume anomaly ---
            v_row = volume_df[volume_df["ticker"] == ticker] if not volume_df.empty else pd.DataFrame()
            if not v_row.empty:
                detail["volume_spike_confirmed"] = True
                score += self._weights["volume_spike_confirmed"]

            # --- Form 4 insider sell ---
            f_row = form4_df[form4_df["ticker"] == ticker] if not form4_df.empty else pd.DataFrame()
            if not f_row.empty:
                detail["form4_insider_sell"] = True
                score += self._weights["form4_insider_sell"]

            # --- Schedule 13D/G drop ---
            # dg_df currently has entity_name, not ticker (fuzzy match deferred).
            # Only apply when a ticker column is present to avoid scoring all tickers.
            dg_row = (
                dg_df[dg_df["ticker"] == ticker]
                if (not dg_df.empty and "ticker" in dg_df.columns)
                else pd.DataFrame()
            )
            if not dg_row.empty:
                detail["schedule_13dg_drop"] = True
                score += self._weights["schedule_13dg_drop"]

            # --- Seasonal redemption window ---
            if in_seasonal_window:
                detail["seasonal_redemption_window"] = True
                score += self._weights["seasonal_redemption_window"]

            # --- Price within 15% of 52-week high ---
            if price_proximity_df is not None and not price_proximity_df.empty:
                pp_row = price_proximity_df[price_proximity_df["ticker"] == ticker]
                if not pp_row.empty and bool(pp_row.iloc[0].get("below_52w_high_15pct", False)):
                    detail["price_below_52w_high_15pct"] = True
                    score += self._weights["price_below_52w_high_15pct"]

            rows.append({
                "ticker": ticker,
                "total_score": score,
                "signals_active": sum(1 for v in detail.values() if v),
                "institution_count": institution_count,
                "total_value_sold_usd": total_sold,
                "signal_detail": json.dumps(detail),
            })

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)
        alerts = result[result["total_score"] >= self._threshold].copy()
        assert isinstance(alerts, pd.DataFrame)
        # Break score ties by distribution magnitude so the top of the list is the
        # largest, broadest exits rather than an arbitrary order among look-alikes.
        alerts = alerts.sort_values(
            ["total_score", "total_value_sold_usd", "institution_count"],
            ascending=False,
        ).reset_index(drop=True)

        _logger.info(
            "Composite scorer: %d tickers scored, %d above threshold %d",
            len(result), len(alerts), self._threshold,
        )
        return alerts
