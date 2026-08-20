# src/common/sentiments/adapters/async_bluesky.py
"""
Async Bluesky adapter -- primary retail-chatter source, replacing Reddit (spec §2.3).

Gated off by default (``providers.bluesky: false``) until ``BLUESKY_HANDLE`` /
``BLUESKY_APP_PASSWORD`` (an app password, never the account password -- see
Settings -> App Passwords on bsky.app) are configured in ``config/donotshare/donotshare.py`` /
the environment.

Authenticates via the official ``atproto`` SDK. Does not rely on unauthenticated public access:
``public.api.bsky.app`` has returned 403 for search since mid-2026, and unauthenticated
``searchPosts`` returns 403 on any cursor-based pagination request even when the first page
succeeds -- an error that resembles rate limiting but isn't (spec §2.3).
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import config.donotshare.donotshare as secrets
from src.common.sentiments.adapters.base_adapter import BaseSentimentAdapter
from src.common.sentiments.entity.resolver import EntityResolver
from src.common.sentiments.processing.bot_detector import hash_author_id
from src.common.sentiments.processing.heuristic_analyzer import HeuristicSentimentAnalyzer
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

try:
    from atproto import AsyncClient
    from atproto_client.exceptions import UnauthorizedError
    from atproto_client.models.app.bsky.feed.search_posts import Params as SearchPostsParams

    ATPROTO_AVAILABLE = True
except ImportError:
    ATPROTO_AVAILABLE = False


class AsyncBlueskyAdapter(BaseSentimentAdapter):
    """Bluesky adapter for the `retail` signal class (cashtag + company-name search)."""

    signal_class = "retail"
    supports_cashtag = True

    def __init__(
        self,
        name: str = "bluesky",
        client: "AsyncClient | None" = None,
        concurrency: int = 5,
        rate_limit_delay: float = 0.2,
        handle: str | None = None,
        app_password: str | None = None,
        lang_filter: str = "en",
        max_posts_per_ticker: int = 200,
        search_terms: Tuple[str, ...] = ("cashtag", "company_name"),
        entity_map_path: str | Path | None = None,
        max_retries: int = 2,
    ):
        super().__init__(name, concurrency, rate_limit_delay)

        if not ATPROTO_AVAILABLE:
            raise RuntimeError("The 'atproto' package is required for AsyncBlueskyAdapter (pip install atproto)")

        self._provided_client = client is not None
        self._client = client
        self.handle = handle
        self.app_password = app_password
        self.lang_filter = lang_filter
        self.max_posts_per_ticker = max_posts_per_ticker
        self.search_terms = search_terms
        self.max_retries = max_retries

        self._analyzer = HeuristicSentimentAnalyzer(signal_class=self.signal_class)

        try:
            self._resolver: EntityResolver | None = EntityResolver.from_yaml(entity_map_path)
        except Exception as e:
            _logger.warning("Failed to load entity map for Bluesky company-name search: %s", e)
            self._resolver = None

        self._login_lock = asyncio.Lock()
        self._logged_in = False
        self.auth_refresh_count = 0
        self.pagination_fallback_count = 0

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    async def _ensure_login(self) -> None:
        """
        Authenticate once per adapter instance (i.e. once per batch run) and reuse the session.

        Distinguishes 401 (bad credentials -- fail loudly, do not retry) from 403 on pagination
        (known behavior, handled by ``_search_posts``'s time-window fallback). The SDK maps both
        401 and 403 to the same ``UnauthorizedError`` class, so the distinction is made on
        ``e.response.status_code``, not exception type (spec §2.8).
        """
        if self._logged_in:
            return
        async with self._login_lock:
            if self._logged_in:
                return

            if self._client is None:
                self._client = AsyncClient()

            handle = self.handle or getattr(secrets, "BLUESKY_HANDLE", None)
            app_password = self.app_password or getattr(secrets, "BLUESKY_APP_PASSWORD", None)
            if not handle or not app_password:
                raise RuntimeError(
                    "Bluesky adapter requires BLUESKY_HANDLE and BLUESKY_APP_PASSWORD (an app "
                    "password, never the account password -- Settings > App Passwords on bsky.app)"
                )

            try:
                await self._client.login(login=handle, password=app_password)
                self._logged_in = True
                self.auth_refresh_count += 1
                _logger.info("Bluesky adapter authenticated as %s", handle)
            except UnauthorizedError as e:
                status = e.response.status_code if e.response else None
                _logger.error(
                    "Bluesky authentication failed (status=%s) -- verify BLUESKY_APP_PASSWORD is a "
                    "valid app password, not the account password",
                    status,
                )
                raise

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------
    def _build_queries(self, ticker: str) -> List[str]:
        """
        Build the search queries for a ticker.

        Cashtag-only search badly undercounts on Bluesky (spec §2.3), so a company-name query is
        added when the ticker is in the entity map -- *unless* the ticker is flagged ambiguous
        (``$ALL``, ``$IT``, ``$ON``, ...), in which case only the ``$``-prefixed cashtag is used
        and bare-name matches are dropped entirely, guarding against ticker/word collisions.
        """
        queries = [f"${ticker}"]

        if "company_name" not in self.search_terms or self._resolver is None:
            return queries
        if ticker in self._resolver.ambiguous_tickers:
            return queries

        names = self._resolver.get_names(ticker)
        if names:
            queries.append(names[0])
        return queries

    # ------------------------------------------------------------------
    # Search + pagination
    # ------------------------------------------------------------------
    async def _search_posts(self, query: str, since_iso: str | None, limit: int) -> Tuple[List[Any], bool]:
        """
        Search posts for one query, paginating up to ``limit`` results.

        Prefers cursor-based pagination; falls back to the time-window strategy (``until`` set to
        the previous page's last post's ``createdAt``, deduped on ``uri``) on a 403 mid-pagination
        -- the known unauthenticated-pagination failure mode (spec §2.3). Stops as soon as a page
        yields no new URIs, in either mode, to avoid looping on a timestamp boundary.

        Returns:
            (posts, pagination_fallback_triggered)
        """
        posts: List[Any] = []
        seen_uris: set[str] = set()
        cursor: str | None = None
        until: str | None = None
        use_cursor = True
        fallback_triggered = False

        while len(posts) < limit:
            page_limit = min(100, limit - len(posts))
            params_kwargs: Dict[str, Any] = {
                "q": query,
                "limit": page_limit,
                "sort": "latest",
                "lang": self.lang_filter,
            }
            if since_iso:
                params_kwargs["since"] = since_iso
            if use_cursor:
                if cursor:
                    params_kwargs["cursor"] = cursor
            elif until:
                params_kwargs["until"] = until

            try:
                assert self._client is not None
                resp = await self._client.app.bsky.feed.search_posts(SearchPostsParams(**params_kwargs))
            except UnauthorizedError as e:
                status = e.response.status_code if e.response else None
                if status == 401:
                    _logger.error("Bluesky returned 401 mid-search -- credentials revoked or expired")
                    raise
                if status == 403 and use_cursor:
                    _logger.debug(
                        "Bluesky 403 on cursor pagination for query %r (known behavior) -- "
                        "falling back to time-window strategy",
                        query,
                    )
                    use_cursor = False
                    fallback_triggered = True
                    continue  # retry this same page with `until` instead of `cursor`
                raise

            await asyncio.sleep(self.rate_limit_delay)

            new_posts = [p for p in resp.posts if p.uri not in seen_uris]
            if not new_posts:
                break  # guard: no new URIs -- stop rather than loop on a timestamp boundary
            seen_uris.update(p.uri for p in new_posts)
            posts.extend(new_posts)

            if use_cursor:
                if not resp.cursor:
                    break
                cursor = resp.cursor
            else:
                # post.record is typed as UnknownType by the SDK (any AT-proto record shape) --
                # it's a feed.post.Record in practice here, but access it defensively.
                until = getattr(new_posts[-1].record, "created_at", None)
                if not until:
                    break  # no timestamp to page from -- stop rather than loop forever

        return posts[:limit], fallback_triggered

    # ------------------------------------------------------------------
    # Filtering + bot heuristics
    # ------------------------------------------------------------------
    @staticmethod
    def _is_english(post: Any) -> bool:
        """Drop posts whose langs excludes English even though ``lang=en`` was requested --
        the API's lang filter is best-effort, not a hard guarantee (spec §2.3).

        ``post`` is a PostView at runtime, but its ``record`` field is typed as ``UnknownType``
        by the SDK (any AT-proto record shape is technically legal there), so this reads it
        defensively via ``getattr`` rather than declaring a strict PostView-typed parameter.
        """
        langs = getattr(post.record, "langs", None) or []
        if not langs:
            return True  # no lang metadata -- don't drop on absence, only on an explicit mismatch
        return "en" in langs

    @staticmethod
    def _is_suspected_bot(post: Any, post_count_for_ticker: int) -> bool:
        """
        Per-source bot heuristic (spec §2.5.2, Bluesky-specific -- distinct from HN's, which
        skips bot detection entirely):
        - >20 posts/day on one ticker -> likely bot. The batch lookback is <=24h in practice, so
          "posts in this fetch" approximates "posts/day".
        - Account age <2 days AND >5 posts -> suspicious.
        """
        if post_count_for_ticker > 20:
            return True

        created_at = getattr(post.author, "created_at", None)
        if created_at:
            try:
                account_created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                age_days = (datetime.now(UTC) - account_created).total_seconds() / 86400.0
                if age_days < 2 and post_count_for_ticker > 5:
                    return True
            except ValueError:
                pass

        return False

    # ------------------------------------------------------------------
    # AsyncAdapter contract
    # ------------------------------------------------------------------
    async def fetch_messages(self, ticker: str, since_ts: int | None = None, limit: int = 200) -> List[Dict[str, Any]]:
        tk = ticker.upper().strip()
        if not tk:
            return []

        await self._ensure_login()

        since_iso = datetime.fromtimestamp(since_ts, tz=UTC).isoformat() if since_ts else None
        queries = self._build_queries(tk)

        # Union and dedupe on uri across the cashtag + company-name queries (spec §2.3).
        all_posts: Dict[str, Any] = {}
        for query in queries:
            try:
                posts, fallback_triggered = await self._search_posts(query, since_iso, limit)
            except UnauthorizedError:
                raise
            except Exception as e:
                _logger.warning("Bluesky search failed for query %r (ticker %s): %s", query, tk, e)
                continue
            if fallback_triggered:
                self.pagination_fallback_count += 1
            for post in posts:
                all_posts[post.uri] = post

        posts_list = [p for p in all_posts.values() if self._is_english(p)][:limit]

        author_post_counts: Dict[str, int] = {}
        for post in posts_list:
            author_post_counts[post.author.did] = author_post_counts.get(post.author.did, 0) + 1

        messages: List[Dict[str, Any]] = []
        for post in posts_list:
            post_count = author_post_counts.get(post.author.did, 1)
            messages.append(
                {
                    "id": post.uri,
                    "body": getattr(post.record, "text", "") or "",
                    "created_at": getattr(post.record, "created_at", None),
                    # No raw handle/DID retained -- author_hash is the salted hash (spec §2.11).
                    "user": {"username": "", "id": hash_author_id(post.author.did), "followers": 0},
                    "likes": int(post.like_count or 0),
                    "replies": int(post.reply_count or 0),
                    "retweets": int(post.repost_count or 0),
                    "provider": "bluesky",
                    "engagement": {
                        "likes": post.like_count or 0,
                        "reposts": post.repost_count or 0,
                        "replies": post.reply_count or 0,
                    },
                    "meta": {"is_bot": self._is_suspected_bot(post, post_count)},
                }
            )
        return messages

    async def fetch_summary(self, ticker: str, since_ts: int | None = None) -> Dict[str, Any]:
        base = {"provider": "bluesky", "timestamp": datetime.now(UTC).isoformat()}
        messages = await self.fetch_messages(ticker, since_ts=since_ts, limit=self.max_posts_per_ticker)

        mentions = len(messages)
        if mentions == 0:
            return {**base, "mentions": 0, "sentiment_score": 0.0, "unique_authors": 0, "bot_pct": 0.0}

        bullish = bearish = bot_count = 0
        authors: set[str] = set()
        for msg in messages:
            body = str(msg.get("body") or "")
            if body:
                result = self._analyzer.analyze_sentiment(body)
                if result.score > 0.1:
                    bullish += 1
                elif result.score < -0.1:
                    bearish += 1
            if msg.get("meta", {}).get("is_bot"):
                bot_count += 1
            authors.add(msg["user"]["id"])

        sentiment_score = max(-1.0, min(1.0, (bullish - bearish) / mentions))

        return {
            **base,
            "mentions": mentions,
            "sentiment_score": sentiment_score,
            "unique_authors": len(authors),
            "bot_pct": bot_count / mentions,
        }

    async def close(self) -> None:
        try:
            if self._client and not self._provided_client:
                await self._client.request.close()
            _logger.debug("Bluesky adapter closed successfully")
        except Exception as e:
            _logger.warning("Error closing Bluesky adapter: %s", e)
