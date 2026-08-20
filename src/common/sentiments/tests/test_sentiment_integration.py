# src/common/sentiments/tests/test_sentiment_integration.py
"""
End-to-end integration and performance tests for collect_sentiment_batch (spec §2.9).

Integration: ["NVDA", "GME"] with mocked adapters -- NVDA exercises the Hacker News
(tech_discourse) path, GME exercises retail-only with tech_coverage_available=False. Asserts
tech_* fields are None (not 0.5) for the uncovered ticker.

Performance: Hacker News's shared-corpus fetch strategy must cost the same regardless of how
many tickers are in the batch -- if adapter fan-out scales with ticker count, the shared-corpus
strategy is broken (spec §2.4/§2.9).
"""

import sys
from pathlib import Path
from typing import Any, Dict, cast
from unittest.mock import AsyncMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.sentiments.collect_sentiment_async import SentimentFeatures, collect_sentiment_batch

CONFIG = {
    "providers": {"stocktwits": True, "hackernews": True, "hf_enabled": False},
    "weights": {"stocktwits": 1.0},
}


def _signal_class(provider: str) -> str:
    return "tech_discourse" if provider == "hackernews" else "retail"


def _make_manager(fetch_summary_side_effect) -> AsyncMock:
    """Build an AsyncMock-based adapter manager -- AsyncMock (not MagicMock) so every method,
    including ones this test doesn't explicitly configure (start/close_all), is awaitable."""
    manager = AsyncMock()
    manager.get_available_adapters = lambda: ["stocktwits", "hackernews"]
    manager.get_signal_class = lambda p: _signal_class(p)
    manager.fetch_summary_from_adapter = AsyncMock(side_effect=fetch_summary_side_effect)
    manager.fetch_messages_from_adapter = AsyncMock(return_value=[])
    return manager


class TestNvdaGmeIntegration:
    """NVDA (HN-covered) vs. GME (HN-uncovered) end to end, per spec §2.9's integration test."""

    @staticmethod
    async def _fetch_summary(provider: str, ticker: str, since_ts: int | None = None) -> Dict[str, Any]:
        del since_ts
        if provider == "stocktwits":
            return {"provider": "stocktwits", "mentions": 12, "sentiment_score": 0.4}
        # hackernews
        if ticker == "NVDA":
            return {
                "provider": "hackernews",
                "mentions": 8,
                "sentiment_score": 0.2,
                "discussion_depth": 15.0,
                "tech_coverage_available": True,
            }
        # GME is not in the tech_discourse entity map at all
        return {
            "provider": "hackernews",
            "mentions": 0,
            "sentiment_score": 0.0,
            "discussion_depth": 0.0,
            "tech_coverage_available": False,
        }

    @pytest.mark.asyncio
    async def test_nvda_gme_end_to_end(self):
        manager = _make_manager(self._fetch_summary)

        with patch("src.common.sentiments.adapters.adapter_manager.get_adapter_manager", return_value=manager):
            raw_results = await collect_sentiment_batch(["NVDA", "GME"], config=dict(CONFIG))

        results = cast(Dict[str, "SentimentFeatures | None"], raw_results)
        nvda = results["NVDA"]
        gme = results["GME"]
        assert isinstance(nvda, SentimentFeatures)
        assert isinstance(gme, SentimentFeatures)

        # NVDA: covered by the HN entity map -- real tech_* values, not None.
        assert nvda.tech_coverage_available is True
        assert nvda.tech_mentions_24h == 8
        assert nvda.tech_sentiment_score_24h is not None
        assert nvda.tech_sentiment_normalized is not None
        assert nvda.tech_discussion_depth == 15.0

        # GME: not in the HN entity map -- every tech_* field is None, never a fabricated 0.5/0.0
        # neutral reading (spec §2.4/§2.13's "no data vs. neutral" distinction).
        assert gme.tech_coverage_available is False
        assert gme.tech_mentions_24h is None
        assert gme.tech_sentiment_score_24h is None
        assert gme.tech_sentiment_normalized is None
        assert gme.tech_discussion_depth is None

        # Retail sentiment is unaffected by tech_discourse coverage either way.
        assert nvda.mentions_24h == 12
        assert gme.mentions_24h == 12


class TestHackerNewsFanOutFlatAcrossBatchSize:
    """
    Performance invariant (spec §2.9): HN corpus-fetch cost must be flat across batch size --
    the shared-corpus strategy fetches the corpus once per adapter instance regardless of how
    many tickers ask for it, not once per ticker.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("batch_size", [10, 25, 50])
    async def test_hn_summary_calls_do_not_scale_corpus_fetch(self, batch_size):
        from src.common.sentiments.adapters.async_hackernews import AsyncHackerNewsAdapter
        from src.common.sentiments.entity.resolver import EntityDef, EntityResolver

        resolver = EntityResolver({"NVDA": EntityDef(ticker="NVDA", names=["nvidia"])})
        adapter = AsyncHackerNewsAdapter(rate_limit_rps=1000.0, db_cache_enabled=False)
        adapter._resolver = resolver

        fetch_json_calls = {"count": 0}

        async def fake_fetch_json(path, timeout=10):
            del timeout
            fetch_json_calls["count"] += 1
            if path in ("newstories.json", "topstories.json"):
                return []
            return None

        tickers = [f"T{i}" for i in range(batch_size)]
        with patch.object(adapter, "_fetch_json", side_effect=fake_fetch_json):
            for ticker in tickers:
                await adapter.fetch_summary(ticker)

        # Exactly one corpus fetch (newstories + topstories = 2 calls) no matter the batch size --
        # if this scaled with batch_size, the shared-corpus strategy would be broken.
        assert fetch_json_calls["count"] == 2

        await adapter.close()
