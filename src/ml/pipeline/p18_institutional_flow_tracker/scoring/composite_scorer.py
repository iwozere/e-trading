"""
Composite Scorer

Combines signals from all layers (13F consensus, volume anomaly, Form 4,
13D/G, seasonal calendar) into a single score per ticker.

Tickers exceeding the alert threshold are returned as actionable signals.
"""

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional
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
    ):
        """
        Args:
            signal_weights: Override the default weight table.
            alert_threshold: Minimum composite score to flag a ticker as an alert.
        """
        self._weights = signal_weights or {
            "consensus_exit_3plus": 40,
            "large_single_exit_500m": 25,
            "volume_spike_confirmed": 20,
            "form4_insider_sell": 10,
            "schedule_13dg_drop": 10,
            "seasonal_redemption_window": 5,
            "price_below_52w_high_15pct": 5,
        }
        self._threshold = alert_threshold

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
            DataFrame sorted by total_score descending, columns: ticker, total_score,
            signals_active (int), signal_detail (JSON string).
            Only rows with total_score >= alert_threshold are returned.
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

            # --- 13F consensus exit signal ---
            c_row = consensus_df[consensus_df["ticker"] == ticker] if not consensus_df.empty else pd.DataFrame()
            if not c_row.empty:
                detail["consensus_exit_3plus"] = True
                score += self._weights["consensus_exit_3plus"]

                # Large single-institution exit bonus
                total_sold = float(c_row.iloc[0].get("total_value_sold_usd", 0))
                if total_sold >= 500_000_000:
                    detail["large_single_exit_500m"] = True
                    score += self._weights["large_single_exit_500m"]

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
                "signal_detail": json.dumps(detail),
            })

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)
        alerts = result[result["total_score"] >= self._threshold].copy()
        assert isinstance(alerts, pd.DataFrame)
        alerts = alerts.sort_values("total_score", ascending=False).reset_index(drop=True)

        _logger.info(
            "Composite scorer: %d tickers scored, %d above threshold %d",
            len(result), len(alerts), self._threshold,
        )
        return alerts
