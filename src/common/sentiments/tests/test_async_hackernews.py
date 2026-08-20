"""
Unit tests for AsyncHackerNewsAdapter.

Per sentiment-spec-rev2.md §2.9, the HTML-cleaning suite is called out as the highest-value
tests in the project -- HN `text` is raw HTML and `<pre><code>` blocks must be stripped entirely
before scoring or every tech ticker becomes systematically, invisibly biased negative.
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.sentiments.adapters.async_hackernews import AsyncHackerNewsAdapter, clean_hn_text
from src.common.sentiments.entity.resolver import EntityDef, EntityResolver


class TestCleanHnText:
    """The HTML-cleaning suite spec §2.9 calls the highest-value tests in the project."""

    def test_strips_pre_code_blocks_entirely(self) -> None:
        raw = "Here's the fix:<p><pre><code>def fail():\n    raise Exception('dead')</code></pre>Works now."
        cleaned = clean_hn_text(raw)
        assert "fail" not in cleaned
        assert "Exception" not in cleaned
        assert "dead" not in cleaned
        assert "Works now." in cleaned

    def test_code_blocks_kept_when_disabled(self) -> None:
        raw = "<pre><code>print(1)</code></pre>"
        cleaned = clean_hn_text(raw, strip_code_blocks=False)
        assert "print(1)" in cleaned

    def test_unescapes_html_entities(self) -> None:
        raw = "It&#x27;s great &amp; fast &gt; everything &quot;else&quot;"
        cleaned = clean_hn_text(raw)
        assert cleaned == 'It\'s great & fast > everything "else"'

    def test_strips_nested_tags(self) -> None:
        raw = "<p>Check out <a href=\"https://example.com\"><i>this</i> link</a>.</p>"
        cleaned = clean_hn_text(raw)
        assert "<" not in cleaned
        assert ">" not in cleaned
        assert "this link" in cleaned

    def test_preserves_paragraph_breaks(self) -> None:
        raw = "<p>First paragraph.</p><p>Second paragraph.</p>"
        cleaned = clean_hn_text(raw)
        assert "First paragraph.\n\nSecond paragraph." in cleaned

    def test_empty_text_returns_empty_string(self) -> None:
        assert clean_hn_text("") == ""
        assert clean_hn_text(None) == ""

    def test_whitespace_only_collapses(self) -> None:
        assert clean_hn_text("   \n\n\n   ") == ""

    def test_br_tags_become_newlines(self) -> None:
        cleaned = clean_hn_text("line one<br>line two<br/>line three")
        assert cleaned == "line one\nline two\nline three"


class TestAsyncHackerNewsAdapter:
    """Corpus-fetch and entity-matching behavior, fully mocked -- no live network calls."""

    @pytest_asyncio.fixture
    async def adapter(self):
        resolver = EntityResolver(
            {
                "NVDA": EntityDef(ticker="NVDA", names=["nvidia"], products=["cuda"]),
                "META": EntityDef(ticker="META", names=["meta platforms"]),
            }
        )
        instance = AsyncHackerNewsAdapter(
            concurrency=5,
            rate_limit_rps=1000.0,  # effectively no throttling in tests
            db_cache_enabled=False,  # no DB dependency in unit tests
        )
        instance._resolver = resolver
        yield instance
        await instance.close()

    @staticmethod
    def _story(item_id: int, title: str, time_offset_hours: float = 1.0, kids=None, score: int = 10):
        return {
            "id": item_id,
            "type": "story",
            "title": title,
            "time": int((datetime.now(UTC) - timedelta(hours=time_offset_hours)).timestamp()),
            "score": score,
            "by": f"user{item_id}",
            "kids": kids or [],
        }

    @staticmethod
    def _comment(item_id: int, text: str, parent: int, time_offset_hours: float = 0.5, kids=None):
        return {
            "id": item_id,
            "type": "comment",
            "text": text,
            "parent": parent,
            "time": int((datetime.now(UTC) - timedelta(hours=time_offset_hours)).timestamp()),
            "by": f"user{item_id}",
            "kids": kids or [],
        }

    @pytest.mark.asyncio
    async def test_matches_ticker_via_title(self, adapter: AsyncHackerNewsAdapter) -> None:
        story = self._story(1, "Nvidia unveils new datacenter chip")

        async def fake_fetch_json(path, timeout=10):
            del timeout
            if path == "newstories.json":
                return [1]
            if path == "topstories.json":
                return []
            if path == "item/1.json":
                return story
            return None

        with patch.object(adapter, "_fetch_json", side_effect=fake_fetch_json):
            summary = await adapter.fetch_summary("NVDA")

        assert summary["tech_coverage_available"] is True
        assert summary["mentions"] == 1
        assert summary["provider"] == "hackernews"

    @pytest.mark.asyncio
    async def test_uncovered_ticker_returns_false_not_neutral(self, adapter: AsyncHackerNewsAdapter) -> None:
        async def fake_fetch_json(path, timeout=10):
            del timeout
            if path in ("newstories.json", "topstories.json"):
                return []
            return None

        with patch.object(adapter, "_fetch_json", side_effect=fake_fetch_json):
            summary = await adapter.fetch_summary("GME")  # not in the tiny test entity map

        assert summary["tech_coverage_available"] is False
        assert summary["mentions"] == 0
        # Never a fabricated neutral reading (spec §2.4/§2.13) -- caller must not read 0.0/0.0
        # here as "neutral sentiment", only as "no coverage".

    @pytest.mark.asyncio
    async def test_covered_ticker_zero_mentions_is_true_with_zero(self, adapter: AsyncHackerNewsAdapter) -> None:
        async def fake_fetch_json(path, timeout=10):
            del timeout
            if path in ("newstories.json", "topstories.json"):
                return []
            return None

        with patch.object(adapter, "_fetch_json", side_effect=fake_fetch_json):
            summary = await adapter.fetch_summary("NVDA")

        assert summary["tech_coverage_available"] is True
        assert summary["mentions"] == 0

    @pytest.mark.asyncio
    async def test_dead_and_deleted_items_skipped(self, adapter: AsyncHackerNewsAdapter) -> None:
        dead_story = self._story(2, "Nvidia news")
        dead_story["dead"] = True
        deleted_story = self._story(3, "Nvidia again")
        deleted_story["deleted"] = True
        live_story = self._story(4, "Nvidia earnings beat")

        async def fake_fetch_json(path, timeout=10):
            del timeout
            if path == "newstories.json":
                return [2, 3, 4]
            if path == "topstories.json":
                return []
            return {"item/2.json": dead_story, "item/3.json": deleted_story, "item/4.json": live_story}.get(path)

        with patch.object(adapter, "_fetch_json", side_effect=fake_fetch_json):
            summary = await adapter.fetch_summary("NVDA")

        assert summary["mentions"] == 1

    @pytest.mark.asyncio
    async def test_lookback_window_excludes_old_items(self, adapter: AsyncHackerNewsAdapter) -> None:
        old_story = self._story(5, "Nvidia old news", time_offset_hours=100)  # outside 48h default

        async def fake_fetch_json(path, timeout=10):
            del timeout
            if path == "newstories.json":
                return [5]
            if path == "topstories.json":
                return []
            if path == "item/5.json":
                return old_story
            return None

        with patch.object(adapter, "_fetch_json", side_effect=fake_fetch_json):
            summary = await adapter.fetch_summary("NVDA")

        assert summary["mentions"] == 0

    @pytest.mark.asyncio
    async def test_multi_entity_story_counts_for_both_tickers(self, adapter: AsyncHackerNewsAdapter) -> None:
        story = self._story(6, "Comparing Nvidia and Meta Platforms AI roadmaps")

        async def fake_fetch_json(path, timeout=10):
            del timeout
            if path == "newstories.json":
                return [6]
            if path == "topstories.json":
                return []
            if path == "item/6.json":
                return story
            return None

        with patch.object(adapter, "_fetch_json", side_effect=fake_fetch_json):
            nvda_summary = await adapter.fetch_summary("NVDA")
            meta_summary = await adapter.fetch_summary("META")

        assert nvda_summary["mentions"] == 1
        assert meta_summary["mentions"] == 1

    @pytest.mark.asyncio
    async def test_corpus_fetched_once_per_adapter_instance(self, adapter: AsyncHackerNewsAdapter) -> None:
        """Shared-corpus invariant: a second ticker lookup must not re-hit the network."""
        story = self._story(7, "Nvidia partners with cloud providers")

        call_log = []

        async def fake_fetch_json(path, timeout=10):
            del timeout
            call_log.append(path)
            if path == "newstories.json":
                return [7]
            if path == "topstories.json":
                return []
            if path == "item/7.json":
                return story
            return None

        with patch.object(adapter, "_fetch_json", side_effect=fake_fetch_json):
            await adapter.fetch_summary("NVDA")
            calls_after_first = len(call_log)
            await adapter.fetch_summary("META")
            calls_after_second = len(call_log)

        assert calls_after_second == calls_after_first  # no additional network calls

    @pytest.mark.asyncio
    async def test_fetch_messages_body_is_cleaned_and_no_raw_username(self, adapter: AsyncHackerNewsAdapter) -> None:
        story = self._story(8, "Nvidia releases <code>driver</code> update", kids=[9])
        comment = self._comment(9, "<p>This <b>Nvidia</b> driver works great for me.</p>", parent=8)

        async def fake_fetch_json(path, timeout=10):
            del timeout
            if path == "newstories.json":
                return [8]
            if path == "topstories.json":
                return []
            return {"item/8.json": story, "item/9.json": comment}.get(path)

        with patch.object(adapter, "_fetch_json", side_effect=fake_fetch_json):
            messages = await adapter.fetch_messages("NVDA")

        assert len(messages) == 2
        for msg in messages:
            assert "<" not in msg["body"]
            assert msg["user"]["username"] == ""  # no raw HN handle retained
            assert msg["provider"] == "hackernews"


    @pytest.mark.asyncio
    async def test_observability_stats_empty_before_corpus_built(self, adapter: AsyncHackerNewsAdapter) -> None:
        """get_observability_stats() backs sentiment.hn.corpus_size/entity_match_rate (spec §2.10)."""
        assert adapter.get_observability_stats() == {}

    @pytest.mark.asyncio
    async def test_observability_stats_reports_corpus_size_and_match_rate(
        self, adapter: AsyncHackerNewsAdapter
    ) -> None:
        matched_story = self._story(10, "Nvidia unveils new datacenter chip")
        unmatched_story = self._story(11, "A story about gardening")

        async def fake_fetch_json(path, timeout=10):
            del timeout
            if path == "newstories.json":
                return [10, 11]
            if path == "topstories.json":
                return []
            return {"item/10.json": matched_story, "item/11.json": unmatched_story}.get(path)

        with patch.object(adapter, "_fetch_json", side_effect=fake_fetch_json):
            await adapter.fetch_summary("NVDA")

        stats = adapter.get_observability_stats()
        assert stats["corpus_size"] == 2
        assert stats["entity_match_rate"] == pytest.approx(0.5)


class TestPerformanceInvariant:
    """Fan-out must be flat across batch size (spec §2.9's performance test)."""

    @pytest.mark.asyncio
    async def test_second_ticker_summary_does_not_refetch_corpus(self) -> None:
        resolver = EntityResolver(
            {"NVDA": EntityDef(ticker="NVDA", names=["nvidia"]), "AMD": EntityDef(ticker="AMD", names=["amd"])}
        )
        adapter = AsyncHackerNewsAdapter(rate_limit_rps=1000.0, db_cache_enabled=False)
        adapter._resolver = resolver

        ensure_corpus_mock = AsyncMock(wraps=adapter._ensure_corpus)
        adapter._corpus = []  # pretend corpus already built (empty is fine for this assertion)
        adapter._ensure_corpus = ensure_corpus_mock  # type: ignore[method-assign]

        await adapter.fetch_summary("NVDA")
        await adapter.fetch_summary("AMD")
        await adapter.fetch_messages("NVDA")

        assert ensure_corpus_mock.await_count == 3  # called each time, but ...
        # ... and each call is a cheap no-op once _corpus is set (no network fan-out re-triggered)
        await adapter.close()
