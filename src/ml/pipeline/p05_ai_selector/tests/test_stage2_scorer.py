"""Tests for Stage2Scorer."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.ml.pipeline.p05_ai_selector.stages.stage2_scorer import Stage2Scorer
from src.ml.pipeline.p05_ai_selector.config import STAGE2_TOP_N


def _make_stage1_row(ticker: str, score: float = 30.0, vol_ratio: float = 1.2) -> dict:
    return {
        "ticker": ticker,
        "asset_type": "equity",
        "last_price": 100.0,
        "avg_vol_usd": 10_000_000.0,
        "momentum_score": score,
        "volume_surge_ratio": vol_ratio,
        "signal_breakdown": "{}",
    }


def _make_stage1_df(n: int = 5, base_score: float = 30.0) -> pd.DataFrame:
    rows = [_make_stage1_row(f"TICK{i}", score=base_score - i) for i in range(n)]
    return pd.DataFrame(rows)


class TestStage2Scorer:
    def test_p18_boost_applied(self, tmp_path):
        """Tickers in p18_data get score boost."""
        scorer = Stage2Scorer(cache_dir=tmp_path)
        stage1 = _make_stage1_df(3)
        stage1.loc[0, "ticker"] = "BOOSTED"

        p18_data = {
            "high_score_count": 1,
            "tickers": {"BOOSTED": 75},
            "consensus_tickers": set(),
            "form4_buy_tickers": set(),
            "13dg_tickers": set(),
        }

        with patch.object(scorer, "_fetch_all_fundamentals", return_value={"BOOSTED": None, "TICK1": None, "TICK2": None}):
            result = scorer.run(stage1, p18_data, {}, date(2026, 6, 14))

        boosted_row = result[result["ticker"] == "BOOSTED"]
        unboosted_row = result[result["ticker"] == "TICK1"]
        assert not boosted_row.empty
        assert float(boosted_row["p18_score"].iloc[0]) > 0
        assert float(boosted_row["total_score"].iloc[0]) > float(unboosted_row["total_score"].iloc[0])

    def test_fundamentals_missing_stays_in_funnel(self, tmp_path):
        """Tickers without fundamentals still appear in the output (score 0)."""
        scorer = Stage2Scorer(cache_dir=tmp_path)
        stage1 = _make_stage1_df(3)
        p18_data = {"high_score_count": 0, "tickers": {}, "consensus_tickers": set(),
                    "form4_buy_tickers": set(), "13dg_tickers": set()}

        with patch.object(scorer, "_fetch_all_fundamentals", return_value={
            "TICK0": None, "TICK1": None, "TICK2": None
        }):
            result = scorer.run(stage1, p18_data, {}, date(2026, 6, 14))

        assert len(result) == 3
        assert all(result["fundamental_score"] == 0.0)

    def test_output_capped_at_25(self, tmp_path):
        """Output is capped at STAGE2_TOP_N rows."""
        scorer = Stage2Scorer(cache_dir=tmp_path)
        stage1 = _make_stage1_df(40)
        p18_data = {"high_score_count": 0, "tickers": {}, "consensus_tickers": set(),
                    "form4_buy_tickers": set(), "13dg_tickers": set()}

        fund_map = {f"TICK{i}": None for i in range(40)}
        with patch.object(scorer, "_fetch_all_fundamentals", return_value=fund_map):
            result = scorer.run(stage1, p18_data, {}, date(2026, 6, 14))

        assert len(result) <= STAGE2_TOP_N

    def test_earnings_flag_set(self, tmp_path):
        """earnings_flag=True when ticker appears in earnings_flags dict."""
        scorer = Stage2Scorer(cache_dir=tmp_path)
        stage1 = _make_stage1_df(2)
        stage1.loc[0, "ticker"] = "EARNER"
        stage1.loc[1, "ticker"] = "NOEARNINGS"

        earnings_flags = {"EARNER": date(2026, 6, 18)}
        p18_data = {"high_score_count": 0, "tickers": {}, "consensus_tickers": set(),
                    "form4_buy_tickers": set(), "13dg_tickers": set()}

        with patch.object(scorer, "_fetch_all_fundamentals", return_value={"EARNER": None, "NOEARNINGS": None}):
            result = scorer.run(stage1, earnings_flags=earnings_flags, p18_data=p18_data, as_of_date=date(2026, 6, 14))

        earner_row = result[result["ticker"] == "EARNER"]
        noearner_row = result[result["ticker"] == "NOEARNINGS"]
        assert bool(earner_row["earnings_flag"].iloc[0]) is True
        assert bool(noearner_row["earnings_flag"].iloc[0]) is False
