# src/common/sentiments/adapters/async_hackernews.py
"""
Async Hacker News adapter -- the `tech_discourse` signal class.

Not a retail-hype source: HN discussion of a company is analytical (outage post-mortems, product
critique, technical comparison). Its sentiment correlates with engineering reputation, not retail
buying pressure -- it must never be blended with `retail` sentiment (see
sentiment-spec-rev2.md §2.4, §2.5.6).

Uses the HN Firebase API (no auth, no key): https://github.com/HackerNews/API

Critical design constraint (spec §2.4): **shared-corpus fetch, not per-ticker fetching.** Once
per batch run this adapter builds a single corpus (newstories + topstories, walked down bounded
by max_depth/max_items_per_thread) and entity-matches *every* ticker against that one corpus in
process. Cost is O(corpus size), independent of how many tickers are in the batch. The corpus is
also cached in Postgres (`ss_hn_corpus`) so a scan re-run within `hn_corpus_ttl_seconds` reuses it
wholesale instead of re-fetching from the Firebase API.
"""

from __future__ import annotations

import asyncio
import html
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import aiohttp

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.sentiments.adapters.base_adapter import BaseSentimentAdapter
from src.common.sentiments.entity.resolver import EntityResolver
from src.common.sentiments.processing.bot_detector import hash_author_id
from src.common.sentiments.processing.heuristic_analyzer import HeuristicSentimentAnalyzer
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

BASE_URL = "https://hacker-news.firebaseio.com/v0/"

# Strip <pre><code>...</code></pre> blocks entirely before scoring -- source code reads as
# strongly negative to most classifiers ("fail", "error", "abort", "kill", "dead") and HN comments
# are full of it. Leaving this in systematically biases every tech ticker negative, invisibly.
_CODE_BLOCK_RE = re.compile(r"<pre>\s*<code>.*?</code>\s*</pre>", re.IGNORECASE | re.DOTALL)
_PARAGRAPH_RE = re.compile(r"</?p>", re.IGNORECASE)
_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")


def clean_hn_text(raw: str | None, strip_code_blocks: bool = True) -> str:
    """
    Clean raw Hacker News HTML into plain text suitable for sentiment scoring.

    Args:
        raw: Raw HTML as returned by the HN API (``item.text`` / ``item.title``).
        strip_code_blocks: When True (the default, and the spec-mandated behavior), remove
            ``<pre><code>...</code></pre>`` blocks entirely rather than just stripping tags.

    Returns:
        Cleaned plain text with paragraph breaks preserved as ``\\n\\n``.
    """
    if not raw:
        return ""

    text = raw
    if strip_code_blocks:
        text = _CODE_BLOCK_RE.sub(" ", text)
    text = _PARAGRAPH_RE.sub("\n\n", text)
    text = _BREAK_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


class _ThreadBudget:
    """Mutable remaining-item counter shared across one story's recursive comment-tree walk."""

    __slots__ = ("remaining",)

    def __init__(self, remaining: int) -> None:
        self.remaining = remaining


class AsyncHackerNewsAdapter(BaseSentimentAdapter):
    """Shared-corpus Hacker News adapter for the `tech_discourse` signal class."""

    signal_class = "tech_discourse"
    supports_cashtag = False

    def __init__(
        self,
        name: str = "hackernews",
        session: aiohttp.ClientSession | None = None,
        concurrency: int = 10,
        rate_limit_rps: float = 10.0,
        rate_limit_delay: float | None = None,
        max_retries: int = 2,
        corpus_sources: tuple[str, ...] = ("newstories", "topstories"),
        corpus_lookback_hours: int = 48,
        max_depth: int = 4,
        max_items_per_thread: int = 300,
        max_story_candidates: int = 500,
        strip_code_blocks: bool = True,
        hn_corpus_ttl_seconds: int = 1800,
        entity_map_path: str | Path | None = None,
        db_cache_enabled: bool = True,
    ):
        # `rate_limit_delay` is accepted directly for compatibility with adapter_manager's
        # generic batching config (which only knows about the shared "concurrency" /
        # "rate_limit_delay" knobs); when given it takes precedence over `rate_limit_rps`.
        effective_delay = rate_limit_delay if rate_limit_delay is not None else (
            1.0 / rate_limit_rps if rate_limit_rps > 0 else 0.0
        )
        super().__init__(name, concurrency, rate_limit_delay=effective_delay)
        self._provided_session = session is not None
        self._session = session
        self.max_retries = max_retries
        self.corpus_sources = corpus_sources
        self.corpus_lookback_hours = corpus_lookback_hours
        self.max_depth = max_depth
        self.max_items_per_thread = max_items_per_thread
        self.max_story_candidates = max_story_candidates
        self.strip_code_blocks = strip_code_blocks
        self.hn_corpus_ttl_seconds = hn_corpus_ttl_seconds
        self.db_cache_enabled = db_cache_enabled

        # Reads config/sentiments.json's lexicons.tech_discourse bucket -- distinct from the
        # retail lexicon used by StockTwits/Bluesky/etc; sharing one lexicon across both is the
        # exact Rev 1 failure mode this module exists to fix (near-zero WSB-slang frequency on HN
        # silently returns neutral for every message). Spec §2.5.3.
        self._analyzer = HeuristicSentimentAnalyzer(signal_class=self.signal_class)

        try:
            self._resolver: EntityResolver | None = EntityResolver.from_yaml(entity_map_path)
        except Exception as e:
            _logger.error("Failed to load Hacker News entity map, tech_coverage will be unavailable: %s", e)
            self._resolver = None

        self._corpus: List[Dict[str, Any]] | None = None
        self._corpus_lock = asyncio.Lock()
        self._ticker_index: Dict[str, List[Dict[str, Any]]] = {}
        self._story_comment_counts: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    async def _get_session(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _fetch_json(self, path: str, timeout: int = 10) -> Any:
        """GET a Firebase endpoint, retrying on 429/5xx. Returns None on failure."""
        session = await self._get_session()
        url = f"{BASE_URL}{path}"

        async with self.semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                        if resp.status in (429,) or resp.status >= 500:
                            if attempt < self.max_retries:
                                backoff = max(self.rate_limit_delay, 0.05) * (2**attempt)
                                await asyncio.sleep(backoff)
                                continue
                            return None
                        if resp.status != 200:
                            return None
                        data = await resp.json(content_type=None)
                        if self.rate_limit_delay:
                            await asyncio.sleep(self.rate_limit_delay)
                        return data
                except (aiohttp.ClientError, TimeoutError) as e:
                    if attempt < self.max_retries:
                        await asyncio.sleep(max(self.rate_limit_delay, 0.05) * (2**attempt))
                        continue
                    _logger.debug("Hacker News request failed for %s: %s", url, e)
                    return None
        return None

    async def _fetch_items_bulk(self, ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch item.json for every id concurrently. Skips ids that fail or return null."""
        if not ids:
            return {}
        tasks = [self._fetch_json(f"item/{item_id}.json") for item_id in ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: Dict[int, Dict[str, Any]] = {}
        for item_id, res in zip(ids, results):
            if isinstance(res, dict):
                out[item_id] = res
        return out

    async def _fetch_story_ids(self) -> List[int]:
        endpoints = {"newstories": "newstories.json", "topstories": "topstories.json"}
        seen: set[int] = set()
        ordered: List[int] = []
        for source in self.corpus_sources:
            path = endpoints.get(source)
            if not path:
                _logger.warning("Unknown Hacker News corpus source %r, skipping", source)
                continue
            ids = await self._fetch_json(path)
            if not isinstance(ids, list):
                continue
            for item_id in ids:
                if isinstance(item_id, int) and item_id not in seen:
                    seen.add(item_id)
                    ordered.append(item_id)
        return ordered[: self.max_story_candidates]

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def _clean_and_normalize_story(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        title = clean_hn_text(str(raw.get("title") or ""), strip_code_blocks=self.strip_code_blocks)
        body = clean_hn_text(raw.get("text"), strip_code_blocks=self.strip_code_blocks)
        text = f"{title}\n\n{body}" if title and body else (title or body)
        author = raw.get("by")
        return {
            "item_id": raw["id"],
            "item_type": "story",
            "parent_id": None,
            "story_id": raw["id"],
            "author_hash": hash_author_id(str(author)) if author else None,
            "created_utc": datetime.fromtimestamp(raw.get("time", 0), tz=UTC),
            "text_clean": text,
            "score": raw.get("score"),
        }

    def _clean_and_normalize_comment(self, raw: Dict[str, Any], story_id: int) -> Dict[str, Any]:
        author = raw.get("by")
        return {
            "item_id": raw["id"],
            "item_type": "comment",
            "parent_id": raw.get("parent"),
            "story_id": story_id,
            "author_hash": hash_author_id(str(author)) if author else None,
            "created_utc": datetime.fromtimestamp(raw.get("time", 0), tz=UTC),
            "text_clean": clean_hn_text(raw.get("text"), strip_code_blocks=self.strip_code_blocks),
            "score": None,
        }

    @staticmethod
    def _is_skippable(raw: Dict[str, Any] | None) -> bool:
        return not raw or bool(raw.get("dead")) or bool(raw.get("deleted"))

    def _finalize_corpus(self, corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Derive per-item engagement fields uniformly, whether the corpus came from a live fetch
        or the Postgres cache (which does not persist HN's own ``descendants``/``kids`` fields).

        ``reply_count`` = number of direct children present *in this corpus* (a proxy for HN's
        ``descendants``, scoped to what was actually captured within max_depth/max_items_per_thread
        and the lookback window). ``story_score`` is looked up for comments so per-message
        engagement weighting has a thread-prominence value, matching spec §2.4's mapping (a
        comment's weight derives from its thread's prominence, not a nonexistent like count).
        """
        by_id = {item["item_id"]: item for item in corpus}
        children_count: Dict[int, int] = {}
        for item in corpus:
            parent_id = item.get("parent_id")
            if parent_id is not None:
                children_count[parent_id] = children_count.get(parent_id, 0) + 1

        for item in corpus:
            item["reply_count"] = children_count.get(item["item_id"], 0)
            if item["item_type"] == "comment":
                story = by_id.get(item.get("story_id"))
                item["story_score"] = story.get("score") if story else None
        return corpus

    # ------------------------------------------------------------------
    # Corpus fetch (live) and comment-tree walk
    # ------------------------------------------------------------------
    async def _walk_comments(
        self,
        kid_ids: List[int],
        story_id: int,
        depth: int,
        budget: _ThreadBudget,
        items: Dict[int, Dict[str, Any]],
        cutoff: datetime,
    ) -> None:
        if depth > self.max_depth or budget.remaining <= 0 or not kid_ids:
            return

        take = list(kid_ids)[: budget.remaining]
        raw_comments = await self._fetch_items_bulk(take)

        next_level: List[int] = []
        for comment_id, raw in raw_comments.items():
            if budget.remaining <= 0:
                break
            if self._is_skippable(raw) or raw.get("type") != "comment":
                continue
            created = datetime.fromtimestamp(raw.get("time", 0), tz=UTC)
            if created < cutoff:
                continue
            items[comment_id] = self._clean_and_normalize_comment(raw, story_id=story_id)
            budget.remaining -= 1
            next_level.extend(raw.get("kids") or [])

        if next_level:
            await self._walk_comments(next_level, story_id, depth + 1, budget, items, cutoff)

    async def _fetch_corpus_live(self, cutoff: datetime) -> List[Dict[str, Any]]:
        story_ids = await self._fetch_story_ids()
        if not story_ids:
            return []

        raw_stories = await self._fetch_items_bulk(story_ids)
        items: Dict[int, Dict[str, Any]] = {}

        for story_id, raw in raw_stories.items():
            if self._is_skippable(raw) or raw.get("type") != "story":
                continue
            created = datetime.fromtimestamp(raw.get("time", 0), tz=UTC)
            if created < cutoff:
                continue
            items[story_id] = self._clean_and_normalize_story(raw)
            budget = _ThreadBudget(self.max_items_per_thread)
            await self._walk_comments(raw.get("kids") or [], story_id, 1, budget, items, cutoff)

        return list(items.values())

    # ------------------------------------------------------------------
    # Postgres corpus cache (best-effort -- never blocks a live fetch on failure)
    # ------------------------------------------------------------------
    @staticmethod
    async def _run_db(fn: Any, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    async def _load_cached_corpus(self, cutoff: datetime) -> List[Dict[str, Any]] | None:
        """
        Return the cached corpus if it was refreshed within ``hn_corpus_ttl_seconds`` -- i.e. a
        whole-corpus cache hit, not a per-item one. Returns None on any cache miss/failure so the
        caller falls back to a live fetch.
        """
        if not self.db_cache_enabled:
            return None
        try:
            from src.data.db.services.short_squeeze_service import ShortSqueezeService

            service = ShortSqueezeService()
            fetched_after = datetime.now(UTC) - timedelta(seconds=self.hn_corpus_ttl_seconds)
            rows = await self._run_db(service.get_hn_corpus_items_since, cutoff, fetched_after)
        except Exception as e:
            _logger.debug("Hacker News corpus cache lookup failed, will fetch live: %s", e)
            return None

        if not rows:
            return None

        return [
            {
                "item_id": row["item_id"],
                "item_type": row["item_type"],
                "parent_id": row["parent_id"],
                "story_id": row["story_id"],
                "author_hash": row["author_hash"],
                "created_utc": row["created_utc"],
                "text_clean": row["text_clean"] or "",
                "score": row["score"],
            }
            for row in rows
        ]

    async def _persist_corpus(self, corpus: List[Dict[str, Any]]) -> None:
        if not self.db_cache_enabled or not corpus:
            return
        now = datetime.now(UTC)
        rows = [
            {
                "item_id": item["item_id"],
                "item_type": item["item_type"],
                "parent_id": item.get("parent_id"),
                "story_id": item.get("story_id"),
                "author_hash": item.get("author_hash"),
                "created_utc": item["created_utc"],
                "text_clean": item.get("text_clean") or "",
                "score": item.get("score"),
                "fetched_at": now,
            }
            for item in corpus
        ]
        try:
            from src.data.db.services.short_squeeze_service import ShortSqueezeService

            service = ShortSqueezeService()
            chunk_size = 500
            for start in range(0, len(rows), chunk_size):
                await self._run_db(service.upsert_hn_corpus_items, rows[start : start + chunk_size])
        except Exception as e:
            _logger.warning("Failed to persist Hacker News corpus cache (continuing without it): %s", e)

    # ------------------------------------------------------------------
    # Corpus lifecycle + entity index
    # ------------------------------------------------------------------
    async def _ensure_corpus(self) -> None:
        """Build the shared corpus exactly once per adapter instance (i.e. once per batch run)."""
        if self._corpus is not None:
            return
        async with self._corpus_lock:
            if self._corpus is not None:
                return

            cutoff = datetime.now(UTC) - timedelta(hours=self.corpus_lookback_hours)
            cached = await self._load_cached_corpus(cutoff)
            if cached is not None:
                _logger.info("Hacker News corpus cache hit: %d items", len(cached))
                corpus = cached
            else:
                _logger.info("Fetching fresh Hacker News corpus (lookback=%dh)", self.corpus_lookback_hours)
                corpus = await self._fetch_corpus_live(cutoff)
                _logger.info("Fetched Hacker News corpus: %d items", len(corpus))
                await self._persist_corpus(corpus)

            self._corpus = self._finalize_corpus(corpus)
            self._build_entity_index()

    def _build_entity_index(self) -> None:
        self._ticker_index = {}
        self._story_comment_counts = {}
        if not self._corpus:
            return

        for item in self._corpus:
            if item["item_type"] == "comment" and item.get("story_id") is not None:
                story_id = item["story_id"]
                self._story_comment_counts[story_id] = self._story_comment_counts.get(story_id, 0) + 1

        if self._resolver is None:
            return

        for item in self._corpus:
            text = item.get("text_clean") or ""
            if not text:
                continue
            match = self._resolver.match(text)
            for ticker in match.tickers:
                self._ticker_index.setdefault(ticker, []).append(item)

    # ------------------------------------------------------------------
    # AsyncAdapter contract
    # ------------------------------------------------------------------
    async def fetch_summary(self, ticker: str, since_ts: int | None = None) -> Dict[str, Any]:
        tk = ticker.upper().strip()
        await self._ensure_corpus()

        base = {"provider": "hackernews", "timestamp": datetime.now(UTC).isoformat()}
        if self._resolver is None or not self._resolver.is_covered(tk):
            # Not in the entity map at all -- "no data", never a fabricated neutral reading.
            return {**base, "mentions": 0, "sentiment_score": 0.0, "discussion_depth": 0.0, "tech_coverage_available": False}

        items = self._ticker_index.get(tk, [])
        if since_ts is not None:
            items = [i for i in items if i["created_utc"].timestamp() >= since_ts]

        if not items:
            return {**base, "mentions": 0, "sentiment_score": 0.0, "discussion_depth": 0.0, "tech_coverage_available": True}

        scores = [self._analyzer.analyze_sentiment(i["text_clean"]).score for i in items]
        sentiment_score = max(-1.0, min(1.0, sum(scores) / len(scores)))

        story_ids = {i["story_id"] for i in items if i.get("story_id") is not None}
        comment_counts = [self._story_comment_counts.get(sid, 0) for sid in story_ids]
        discussion_depth = (sum(comment_counts) / len(comment_counts)) if comment_counts else 0.0

        return {
            **base,
            "mentions": len(items),
            "sentiment_score": sentiment_score,
            "discussion_depth": discussion_depth,
            "tech_coverage_available": True,
        }

    async def fetch_messages(self, ticker: str, since_ts: int | None = None, limit: int = 200) -> List[Dict[str, Any]]:
        tk = ticker.upper().strip()
        await self._ensure_corpus()

        items = self._ticker_index.get(tk, [])
        if since_ts is not None:
            items = [i for i in items if i["created_utc"].timestamp() >= since_ts]
        items = sorted(items, key=lambda i: i["created_utc"], reverse=True)[:limit]

        messages: List[Dict[str, Any]] = []
        for item in items:
            if item["item_type"] == "story":
                score = int(item.get("score") or 0)
                reply_count = int(item.get("reply_count") or 0)
                engagement = {"score": item.get("score"), "descendants": item.get("reply_count", 0)}
            else:
                score = int(item.get("story_score") or 0)
                reply_count = int(item.get("reply_count") or 0)
                engagement = {"parent_story_score": item.get("story_score"), "reply_count": item.get("reply_count", 0)}

            messages.append(
                {
                    "id": str(item["item_id"]),
                    "body": item["text_clean"],
                    "created_at": item["created_utc"].isoformat(),
                    # No raw HN username retained -- author_hash is the salted hash (spec §2.11).
                    "user": {"username": "", "id": item.get("author_hash") or "", "followers": 0},
                    "likes": score,
                    "replies": reply_count,
                    "retweets": 0,
                    "provider": "hackernews",
                    "engagement": engagement,
                }
            )
        return messages

    async def close(self) -> None:
        try:
            if self._session and not self._provided_session:
                await self._session.close()
                self._session = None
            _logger.debug("Hacker News adapter closed successfully")
        except Exception as e:
            _logger.warning("Error closing Hacker News adapter: %s", e)
