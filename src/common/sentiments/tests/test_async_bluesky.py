"""
Unit tests for AsyncBlueskyAdapter.

Mocked entirely at the atproto client boundary (``client.app.bsky.feed.search_posts`` /
``client.login``) -- no live network calls, no real credentials required. Covers spec §2.9's
adapter-mock list (empty/partial/malformed/429/403), the cursor-pagination fallback loop-guard,
and ambiguous-ticker cashtag-only enforcement (spec §2.3).
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from atproto_client.exceptions import RequestException, UnauthorizedError
from atproto_client.request import Response as AtprotoResponse

from src.common.sentiments.adapters.async_bluesky import AsyncBlueskyAdapter
from src.common.sentiments.entity.resolver import EntityDef, EntityResolver


def _make_post(
    uri: str,
    text: str,
    did: str = "did:plc:author1",
    account_created_at: str = "2018-01-01T00:00:00Z",
    langs=("en",),
    likes: int = 1,
    replies: int = 0,
    reposts: int = 0,
    created_at: str | None = None,
) -> SimpleNamespace:
    """Build a lightweight duck-typed stand-in for a PostView, shaped like what the adapter reads."""
    return SimpleNamespace(
        uri=uri,
        like_count=likes,
        reply_count=replies,
        repost_count=reposts,
        author=SimpleNamespace(did=did, created_at=account_created_at),
        record=SimpleNamespace(
            text=text,
            created_at=created_at or datetime.now(UTC).isoformat(),
            langs=list(langs),
        ),
    )


def _search_response(posts, cursor: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(posts=posts, cursor=cursor, hits_total=len(posts))


def _unauthorized(status_code: int) -> UnauthorizedError:
    return UnauthorizedError(AtprotoResponse(success=False, status_code=status_code, content=None, headers={}))


def _mock_client(adapter: AsyncBlueskyAdapter) -> AsyncMock:
    """Narrow ``adapter._client`` (typed ``AsyncClient | None``) to the AsyncMock fixtures inject."""
    assert adapter._client is not None
    return adapter._client  # type: ignore[return-value]  # AsyncMock, not a real AsyncClient


@pytest_asyncio.fixture
async def adapter():
    resolver = EntityResolver(
        {"NVDA": EntityDef(ticker="NVDA", names=["nvidia"], products=["cuda"])},
        ambiguous_tickers=["ON", "ALL"],
    )
    instance = AsyncBlueskyAdapter(
        concurrency=5,
        rate_limit_delay=0.0,
        handle="test.bsky.social",
        app_password="app-password-123",
        client=AsyncMock(),
    )
    instance._resolver = resolver
    instance._logged_in = True  # skip real login in unit tests
    yield instance
    await instance.close()


class TestBuildQueries:
    def test_cashtag_and_company_name(self, adapter: AsyncBlueskyAdapter) -> None:
        assert adapter._build_queries("NVDA") == ["$NVDA", "nvidia"]

    def test_ambiguous_ticker_cashtag_only(self, adapter: AsyncBlueskyAdapter) -> None:
        """Ambiguous tickers ($ALL, $IT, $ON, ...) never get a bare-name query (spec §2.3)."""
        assert adapter._build_queries("ON") == ["$ON"]

    def test_uncovered_ticker_cashtag_only(self, adapter: AsyncBlueskyAdapter) -> None:
        assert adapter._build_queries("GME") == ["$GME"]

    def test_company_name_search_disabled(self, adapter: AsyncBlueskyAdapter) -> None:
        adapter.search_terms = ("cashtag",)
        assert adapter._build_queries("NVDA") == ["$NVDA"]


class TestFetchMessages:
    @pytest.mark.asyncio
    async def test_empty_response(self, adapter: AsyncBlueskyAdapter) -> None:
        _mock_client(adapter).app.bsky.feed.search_posts = AsyncMock(return_value=_search_response([]))
        messages = await adapter.fetch_messages("NVDA")
        assert messages == []

    @pytest.mark.asyncio
    async def test_partial_response_single_page(self, adapter: AsyncBlueskyAdapter) -> None:
        posts = [_make_post("at://1", "Nvidia earnings beat expectations, $NVDA mooning")]
        _mock_client(adapter).app.bsky.feed.search_posts = AsyncMock(return_value=_search_response(posts))
        messages = await adapter.fetch_messages("NVDA")
        assert len(messages) == 1
        assert messages[0]["provider"] == "bluesky"
        assert messages[0]["body"] == posts[0].record.text
        assert messages[0]["user"]["username"] == ""  # no raw handle retained

    @pytest.mark.asyncio
    async def test_malformed_post_missing_optional_fields_handled(self, adapter: AsyncBlueskyAdapter) -> None:
        post = _make_post("at://1", "", likes=0, replies=0, reposts=0)
        _mock_client(adapter).app.bsky.feed.search_posts = AsyncMock(return_value=_search_response([post]))
        messages = await adapter.fetch_messages("NVDA")
        assert len(messages) == 1
        assert messages[0]["body"] == ""

    @pytest.mark.asyncio
    async def test_generic_request_error_is_swallowed_per_query(self, adapter: AsyncBlueskyAdapter) -> None:
        """429/5xx-style generic failures degrade gracefully rather than crashing the batch."""
        _mock_client(adapter).app.bsky.feed.search_posts = AsyncMock(
            side_effect=RequestException(AtprotoResponse(success=False, status_code=429, content=None, headers={}))
        )
        messages = await adapter.fetch_messages("NVDA")
        assert messages == []

    @pytest.mark.asyncio
    async def test_401_raises_immediately(self, adapter: AsyncBlueskyAdapter) -> None:
        """401 (bad credentials) fails loudly rather than being swallowed like other errors."""
        _mock_client(adapter).app.bsky.feed.search_posts = AsyncMock(side_effect=_unauthorized(401))
        with pytest.raises(UnauthorizedError):
            await adapter.fetch_messages("NVDA")

    @pytest.mark.asyncio
    async def test_non_english_posts_dropped(self, adapter: AsyncBlueskyAdapter) -> None:
        posts = [
            _make_post("at://1", "Nvidia earnings", langs=("en",)),
            _make_post("at://2", "Nvidia ganancias", langs=("es",)),
        ]
        _mock_client(adapter).app.bsky.feed.search_posts = AsyncMock(return_value=_search_response(posts))
        messages = await adapter.fetch_messages("NVDA")
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_no_raw_author_identifier_in_any_message(self, adapter: AsyncBlueskyAdapter) -> None:
        posts = [_make_post("at://1", "Nvidia earnings", did="did:plc:sensitivehandle")]
        _mock_client(adapter).app.bsky.feed.search_posts = AsyncMock(return_value=_search_response(posts))
        messages = await adapter.fetch_messages("NVDA")
        assert "did:plc:sensitivehandle" not in str(messages)


class TestPaginationFallback:
    @pytest.mark.asyncio
    async def test_403_mid_pagination_falls_back_to_time_window(self, adapter: AsyncBlueskyAdapter) -> None:
        page1_posts = [_make_post(f"at://{i}", f"Nvidia post {i}") for i in range(2)]
        page2_posts = [_make_post(f"at://{i}", f"Nvidia post {i}") for i in range(2, 4)]

        call_count = 0

        async def fake_search_posts(params):
            del params
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _search_response(page1_posts, cursor="cursor-abc")
            if call_count == 2:
                # cursor-based page 2 fails with 403 (known unauthenticated-pagination behavior)
                raise _unauthorized(403)
            # time-window fallback (params.until set instead of cursor) succeeds
            return _search_response(page2_posts, cursor=None)

        _mock_client(adapter).app.bsky.feed.search_posts = AsyncMock(side_effect=fake_search_posts)
        posts, fallback_triggered = await adapter._search_posts("$NVDA", None, limit=10)

        assert fallback_triggered is True
        assert len(posts) == 4

    @pytest.mark.asyncio
    async def test_no_new_uris_stops_loop(self, adapter: AsyncBlueskyAdapter) -> None:
        """Guard: a page that yields no new URIs must stop pagination, not loop forever."""
        repeated_posts = [_make_post("at://1", "Nvidia post")]
        _mock_client(adapter).app.bsky.feed.search_posts = AsyncMock(
            return_value=_search_response(repeated_posts, cursor="same-cursor-forever")
        )
        posts, _ = await adapter._search_posts("$NVDA", None, limit=50)
        # The server keeps returning the same post + a non-empty cursor forever. The loop can't
        # know that in advance, so it costs exactly one "wasted" follow-up call to discover the
        # second page has no new URIs -- but it must then stop, not spin indefinitely.
        assert len(posts) == 1
        assert _mock_client(adapter).app.bsky.feed.search_posts.await_count == 2


class TestBotHeuristics:
    def test_high_volume_author_flagged(self, adapter: AsyncBlueskyAdapter) -> None:
        post = _make_post("at://1", "Nvidia", account_created_at="2015-01-01T00:00:00Z")
        assert adapter._is_suspected_bot(post, post_count_for_ticker=25) is True

    def test_new_account_high_activity_flagged(self, adapter: AsyncBlueskyAdapter) -> None:
        recent = (datetime.now(UTC) - timedelta(hours=12)).isoformat()
        post = _make_post("at://1", "Nvidia", account_created_at=recent)
        assert adapter._is_suspected_bot(post, post_count_for_ticker=6) is True

    def test_established_low_volume_author_not_flagged(self, adapter: AsyncBlueskyAdapter) -> None:
        post = _make_post("at://1", "Nvidia", account_created_at="2015-01-01T00:00:00Z")
        assert adapter._is_suspected_bot(post, post_count_for_ticker=2) is False

    def test_new_account_low_activity_not_flagged(self, adapter: AsyncBlueskyAdapter) -> None:
        recent = (datetime.now(UTC) - timedelta(hours=12)).isoformat()
        post = _make_post("at://1", "Nvidia", account_created_at=recent)
        assert adapter._is_suspected_bot(post, post_count_for_ticker=2) is False


class TestFetchSummary:
    @pytest.mark.asyncio
    async def test_empty_summary(self, adapter: AsyncBlueskyAdapter) -> None:
        _mock_client(adapter).app.bsky.feed.search_posts = AsyncMock(return_value=_search_response([]))
        summary = await adapter.fetch_summary("NVDA")
        assert summary["mentions"] == 0
        assert summary["provider"] == "bluesky"

    @pytest.mark.asyncio
    async def test_summary_counts_unique_authors(self, adapter: AsyncBlueskyAdapter) -> None:
        posts = [
            _make_post("at://1", "Nvidia moon", did="did:plc:a"),
            _make_post("at://2", "Nvidia moon again", did="did:plc:a"),
            _make_post("at://3", "Nvidia dump", did="did:plc:b"),
        ]
        _mock_client(adapter).app.bsky.feed.search_posts = AsyncMock(return_value=_search_response(posts))
        summary = await adapter.fetch_summary("NVDA")
        assert summary["mentions"] == 3
        assert summary["unique_authors"] == 2


class TestAuth:
    @pytest.mark.asyncio
    async def test_missing_credentials_raises(self) -> None:
        instance = AsyncBlueskyAdapter(handle=None, app_password=None, client=AsyncMock())
        instance.handle = None
        instance.app_password = None
        with patch("src.common.sentiments.adapters.async_bluesky.secrets") as mock_secrets:
            mock_secrets.BLUESKY_HANDLE = None
            mock_secrets.BLUESKY_APP_PASSWORD = None
            with pytest.raises(RuntimeError, match="BLUESKY_HANDLE"):
                await instance._ensure_login()
        await instance.close()

    @pytest.mark.asyncio
    async def test_401_on_login_raises_and_logs(self) -> None:
        mock_client = AsyncMock()
        mock_client.login = AsyncMock(side_effect=_unauthorized(401))
        instance = AsyncBlueskyAdapter(handle="bad", app_password="bad", client=mock_client)
        with pytest.raises(UnauthorizedError):
            await instance._ensure_login()
        assert instance._logged_in is False
        await instance.close()

    @pytest.mark.asyncio
    async def test_successful_login_only_happens_once(self) -> None:
        mock_client = AsyncMock()
        instance = AsyncBlueskyAdapter(handle="ok", app_password="ok", client=mock_client)
        await instance._ensure_login()
        await instance._ensure_login()
        assert mock_client.login.await_count == 1
        assert instance.auth_refresh_count == 1
        await instance.close()
