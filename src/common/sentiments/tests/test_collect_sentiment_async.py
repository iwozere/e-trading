# src/common/sentiments/tests/test_collect_sentiment_async.py
"""
Unit tests for src.common.sentiments.collect_sentiment_async.

Covers:
- _percentile_ranks: per-batch percentile ranking used for normalized_engagement (spec §2.5.5)
- _process_messages_with_hf: reach-only virality_index, direction-only sentiment_score,
  provider-native meta.is_bot bot detection, and HN's skip_bot_detection=True -> bot_pct=None
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.sentiments.collect_sentiment_async import _percentile_ranks, _process_messages_with_hf

HEURISTIC_CONFIG = {
    "positive_tokens": ["moon", "rocket"],
    "negative_tokens": ["dump", "crash"],
    "engagement_weight_formula": "sqrt",
}


def _make_manager(predictions: list[dict], adapter_name: str = "huggingface") -> SimpleNamespace:
    adapter = SimpleNamespace(predict_batch=AsyncMock(return_value=predictions))
    return SimpleNamespace(_adapters={adapter_name: adapter})


def _msg(body: str, likes: int = 0, replies: int = 0, is_bot: bool = False, author_id: str = "a1") -> dict:
    return {
        "body": body,
        "likes": likes,
        "replies": replies,
        "retweets": 0,
        "user": {"id": author_id},
        "meta": {"is_bot": is_bot},
    }


class TestPercentileRanks:
    def test_empty(self):
        assert _percentile_ranks([]) == []

    def test_single_value_gets_top_rank(self):
        assert _percentile_ranks([42.0]) == [1.0]

    def test_ascending_values_rank_0_to_1(self):
        ranks = _percentile_ranks([1.0, 2.0, 3.0])
        assert ranks[0] == 0.0
        assert ranks[1] == 0.5
        assert ranks[2] == 1.0

    def test_order_independent_of_input_order(self):
        ranks = _percentile_ranks([3.0, 1.0, 2.0])
        assert ranks == [1.0, 0.0, 0.5]

    def test_ties_get_average_rank(self):
        # [1, 1, 2] -> tied pair average rank (0+1)/2=0.5, scaled by (n-1)=2 -> 0.25
        ranks = _percentile_ranks([1.0, 1.0, 2.0])
        assert ranks[0] == pytest.approx(0.25)
        assert ranks[1] == pytest.approx(0.25)
        assert ranks[2] == 1.0

    def test_all_identical_values_all_rank_at_midpoint(self):
        ranks = _percentile_ranks([5.0, 5.0, 5.0, 5.0])
        assert all(r == pytest.approx(0.5) for r in ranks)


class TestProcessMessagesWithHf:
    @pytest.mark.asyncio
    async def test_empty_messages_returns_neutral_zero(self):
        manager = _make_manager([])
        sentiment, pos_ratio, bot_pct, virality = await _process_messages_with_hf(
            [], manager, HEURISTIC_CONFIG, hf_weight=0.5
        )
        assert sentiment == 0.0
        assert pos_ratio is None
        assert bot_pct == 0.0
        assert virality == 0.0

    @pytest.mark.asyncio
    async def test_skip_bot_detection_returns_none_bot_pct(self):
        # Hacker News never runs bot detection (spec §2.5.2) -- bot_pct must be None, not 0.0,
        # so callers can't mistake "not measured" for "measured zero".
        manager = _make_manager([{"label": "POSITIVE", "score": 0.9}], adapter_name="huggingface_tech")
        messages = [_msg("Impressive, solid release.", is_bot=True)]
        _, _, bot_pct, _ = await _process_messages_with_hf(
            messages, manager, HEURISTIC_CONFIG, hf_weight=0.5, hf_adapter_name="huggingface_tech", skip_bot_detection=True
        )
        assert bot_pct is None

    @pytest.mark.asyncio
    async def test_bot_detection_reads_meta_is_bot(self):
        manager = _make_manager([{"label": "NEUTRAL", "score": 0.5}, {"label": "NEUTRAL", "score": 0.5}])
        messages = [_msg("first", is_bot=True, author_id="a1"), _msg("second", is_bot=False, author_id="a2")]
        _, _, bot_pct, _ = await _process_messages_with_hf(messages, manager, HEURISTIC_CONFIG, hf_weight=0.5)
        assert bot_pct == 0.5

    @pytest.mark.asyncio
    async def test_virality_index_is_unsigned_reach_not_sentiment_weighted(self):
        # Rev 1 conflated reach with |sentiment| (a viral negative post and a quiet positive one
        # could produce the same value). Two messages with identical engagement but opposite
        # sentiment must now produce the SAME virality_index (spec §2.5.5).
        manager_pos = _make_manager([{"label": "POSITIVE", "score": 0.99}])
        manager_neg = _make_manager([{"label": "NEGATIVE", "score": 0.99}])
        msg = [_msg("text", likes=100, replies=0, author_id="a1")]

        _, _, _, virality_pos = await _process_messages_with_hf(msg, manager_pos, HEURISTIC_CONFIG, hf_weight=1.0)
        _, _, _, virality_neg = await _process_messages_with_hf(msg, manager_neg, HEURISTIC_CONFIG, hf_weight=1.0)

        assert virality_pos == pytest.approx(virality_neg)
        assert virality_pos > 0

    @pytest.mark.asyncio
    async def test_virality_index_uses_unique_author_count(self):
        # virality_index = Σ(engagement) / sqrt(unique_authors + 1)
        manager = _make_manager([{"label": "NEUTRAL", "score": 0.5}, {"label": "NEUTRAL", "score": 0.5}])
        messages = [_msg("a", likes=10, author_id="same"), _msg("b", likes=10, author_id="same")]
        _, _, _, virality = await _process_messages_with_hf(messages, manager, HEURISTIC_CONFIG, hf_weight=0.0)
        # engagement = likes = 10 each -> sum = 20; unique_authors = 1 -> sqrt(2)
        assert virality == pytest.approx(20 / (2**0.5))

    @pytest.mark.asyncio
    async def test_hf_prediction_failure_returns_neutral_fallback(self):
        adapter = SimpleNamespace(predict_batch=AsyncMock(side_effect=RuntimeError("boom")))
        manager = SimpleNamespace(_adapters={"huggingface": adapter})
        sentiment, pos_ratio, bot_pct, virality = await _process_messages_with_hf(
            [_msg("hello")], manager, HEURISTIC_CONFIG, hf_weight=0.5
        )
        assert sentiment == 0.0
        assert pos_ratio is None
        assert bot_pct == 0.0
        assert virality == 0.0

    @pytest.mark.asyncio
    async def test_routes_to_the_named_hf_adapter(self):
        # hf_adapter_name selects between the retail and tech_discourse model instances
        # (spec §2.5.4) -- must not silently fall back to "huggingface".
        retail_adapter = SimpleNamespace(predict_batch=AsyncMock(return_value=[{"label": "POSITIVE", "score": 1.0}]))
        tech_adapter = SimpleNamespace(predict_batch=AsyncMock(return_value=[{"label": "NEGATIVE", "score": 1.0}]))
        manager = SimpleNamespace(_adapters={"huggingface": retail_adapter, "huggingface_tech": tech_adapter})

        sentiment, *_ = await _process_messages_with_hf(
            [_msg("text")], manager, HEURISTIC_CONFIG, hf_weight=1.0, hf_adapter_name="huggingface_tech"
        )
        assert sentiment < 0
        tech_adapter.predict_batch.assert_awaited_once()
        retail_adapter.predict_batch.assert_not_awaited()
