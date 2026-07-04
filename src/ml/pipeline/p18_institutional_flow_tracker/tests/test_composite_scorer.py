"""Unit tests for CompositeScorer."""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.ml.pipeline.p18_institutional_flow_tracker.scoring.composite_scorer import CompositeScorer


@pytest.fixture()
def scorer() -> CompositeScorer:
    return CompositeScorer(alert_threshold=60)


def test_large_billion_dollar_exit_reaches_threshold(scorer: CompositeScorer) -> None:
    # consensus (40) + $2B+ large-exit tier (25) = 65 >= threshold 60
    consensus = pd.DataFrame(
        [{"ticker": "AAPL", "institution_count": 3, "total_value_sold_usd": 2_500_000_000, "avg_exit_pct": -0.8}]
    )
    result = scorer.score(consensus, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), date(2024, 5, 1))
    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "AAPL"
    assert result.iloc[0]["total_score"] == 65


def test_modest_exit_no_longer_saturates(scorer: CompositeScorer) -> None:
    # consensus (40) + $600M tier (10) + breadth for 5 inst (2*3=6) = 56 < 60.
    # A borderline distribution must NOT auto-clear the bar (anti-saturation).
    consensus = pd.DataFrame(
        [{"ticker": "AAPL", "institution_count": 5, "total_value_sold_usd": 600_000_000, "avg_exit_pct": -0.8}]
    )
    result = scorer.score(consensus, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), date(2024, 5, 1))
    assert result.empty


def test_large_exit_tiers_are_graded() -> None:
    # Same consensus base, different dollar magnitude → different score.
    scorer = CompositeScorer(alert_threshold=0)
    small = pd.DataFrame([{"ticker": "S", "institution_count": 3, "total_value_sold_usd": 600_000_000}])
    big = pd.DataFrame([{"ticker": "B", "institution_count": 3, "total_value_sold_usd": 3_000_000_000}])
    s_small = scorer.score(small, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), date(2024, 5, 1)).iloc[0][
        "total_score"
    ]
    s_big = scorer.score(big, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), date(2024, 5, 1)).iloc[0]["total_score"]
    assert s_big > s_small  # $3B (40+25) outranks $600M (40+10)


def test_breadth_bonus_rewards_more_institutions() -> None:
    scorer = CompositeScorer(alert_threshold=0)
    narrow = pd.DataFrame([{"ticker": "N", "institution_count": 3, "total_value_sold_usd": 100_000_000}])
    broad = pd.DataFrame([{"ticker": "W", "institution_count": 20, "total_value_sold_usd": 100_000_000}])
    s_narrow = scorer.score(narrow, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), date(2024, 5, 1)).iloc[0][
        "total_score"
    ]
    s_broad = scorer.score(broad, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), date(2024, 5, 1)).iloc[0][
        "total_score"
    ]
    # narrow: 40 + 0 breadth; broad: 40 + capped breadth (15)
    assert s_narrow == 40
    assert s_broad == 55


def test_ties_break_by_magnitude() -> None:
    # Two tickers with identical scores; the larger liquidation must rank first.
    scorer = CompositeScorer(alert_threshold=0)
    consensus = pd.DataFrame(
        [
            {"ticker": "SMALL", "institution_count": 3, "total_value_sold_usd": 2_100_000_000},
            {"ticker": "BIG", "institution_count": 3, "total_value_sold_usd": 9_000_000_000},
        ]
    )
    result = scorer.score(consensus, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), date(2024, 5, 1))
    # Both score 40+25=65; BIG sold more, so it is ranked first.
    assert result.iloc[0]["ticker"] == "BIG"
    assert result.iloc[0]["total_score"] == result.iloc[1]["total_score"]


def test_consensus_plus_volume_scores_higher(scorer: CompositeScorer) -> None:
    consensus = pd.DataFrame(
        [{"ticker": "TSLA", "institution_count": 4, "total_value_sold_usd": 1_200_000_000, "avg_exit_pct": -0.6}]
    )
    volume = pd.DataFrame(
        [{"ticker": "TSLA", "volume_spike_ratio": 4.2, "price_change_5d_pct": -5.0, "above_spike_days": 3}]
    )
    result = scorer.score(consensus, volume, pd.DataFrame(), pd.DataFrame(), date(2024, 5, 1))
    # 40 + $1B tier (18) + breadth (1*3=3) + volume (20) = 81
    assert result.iloc[0]["total_score"] >= 60


def test_below_threshold_not_returned(scorer: CompositeScorer) -> None:
    # Volume only = 20 points, below threshold of 60
    volume = pd.DataFrame(
        [{"ticker": "XYZ", "volume_spike_ratio": 5.0, "price_change_5d_pct": -2.0, "above_spike_days": 2}]
    )
    result = scorer.score(pd.DataFrame(), volume, pd.DataFrame(), pd.DataFrame(), date(2024, 5, 1))
    assert result.empty


def test_seasonal_window_adds_points() -> None:
    # Use a low threshold so consensus alone clears the bar in both months
    low_thresh_scorer = CompositeScorer(alert_threshold=35)
    consensus = pd.DataFrame(
        [{"ticker": "NFLX", "institution_count": 3, "total_value_sold_usd": 50_000_000, "avg_exit_pct": -0.4}]
    )
    result_dec = low_thresh_scorer.score(consensus, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), date(2024, 12, 1))
    result_may = low_thresh_scorer.score(consensus, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), date(2024, 5, 1))
    # December = 40 + 5 = 45; May = 40 + 0 = 40 → difference is +5
    assert result_dec.iloc[0]["total_score"] == result_may.iloc[0]["total_score"] + 5


def test_all_inputs_empty_returns_empty(scorer: CompositeScorer) -> None:
    result = scorer.score(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert result.empty
