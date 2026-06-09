"""Unit tests for CompositeScorer."""

from datetime import date
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.ml.pipeline.p18_institutional_flow_tracker.scoring.composite_scorer import CompositeScorer


@pytest.fixture()
def scorer() -> CompositeScorer:
    return CompositeScorer(alert_threshold=60)


def test_consensus_only_reaches_threshold(scorer: CompositeScorer) -> None:
    # consensus (40) + large_single_exit_500m (25) = 65 >= threshold 60
    consensus = pd.DataFrame([{"ticker": "AAPL", "institution_count": 5, "total_value_sold_usd": 600_000_000, "avg_exit_pct": -0.8}])
    result = scorer.score(consensus, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), date(2024, 5, 1))
    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "AAPL"
    assert result.iloc[0]["total_score"] == 65


def test_consensus_plus_volume_scores_higher(scorer: CompositeScorer) -> None:
    consensus = pd.DataFrame([{"ticker": "TSLA", "institution_count": 4, "total_value_sold_usd": 100_000_000, "avg_exit_pct": -0.6}])
    volume = pd.DataFrame([{"ticker": "TSLA", "volume_spike_ratio": 4.2, "price_change_5d_pct": -5.0, "above_spike_days": 3}])
    result = scorer.score(consensus, volume, pd.DataFrame(), pd.DataFrame(), date(2024, 5, 1))
    assert result.iloc[0]["total_score"] >= 60


def test_below_threshold_not_returned(scorer: CompositeScorer) -> None:
    # Volume only = 20 points, below threshold of 60
    volume = pd.DataFrame([{"ticker": "XYZ", "volume_spike_ratio": 5.0, "price_change_5d_pct": -2.0, "above_spike_days": 2}])
    result = scorer.score(pd.DataFrame(), volume, pd.DataFrame(), pd.DataFrame(), date(2024, 5, 1))
    assert result.empty


def test_seasonal_window_adds_points() -> None:
    # Use a low threshold so consensus alone clears the bar in both months
    low_thresh_scorer = CompositeScorer(alert_threshold=35)
    consensus = pd.DataFrame([{"ticker": "NFLX", "institution_count": 3, "total_value_sold_usd": 50_000_000, "avg_exit_pct": -0.4}])
    result_dec = low_thresh_scorer.score(consensus, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), date(2024, 12, 1))
    result_may = low_thresh_scorer.score(consensus, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), date(2024, 5, 1))
    # December = 40 + 5 = 45; May = 40 + 0 = 40 → difference is +5
    assert result_dec.iloc[0]["total_score"] == result_may.iloc[0]["total_score"] + 5


def test_all_inputs_empty_returns_empty(scorer: CompositeScorer) -> None:
    result = scorer.score(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert result.empty
