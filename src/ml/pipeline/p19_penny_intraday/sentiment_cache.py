"""
P19 sentiment batch cache (spec v2 §10) — throttles the multi-provider
sentiment fetch to roughly once per TTL window regardless of shadow-poll
cadence.

The spec text says sentiment is "captured per poll", but the shadow loop
polls every 15 minutes during market hours (~32 times/day) while every other
pipeline that calls ``collect_sentiment_batch_sync`` (P04's daily deep scan)
does so once a day — sentiment (Reddit/StockTwits/news mention counts) also
doesn't meaningfully change on a 15-minute cadence. Calling the full batch on
every poll would make P19 by far the heaviest consumer of these provider
rate limits in the codebase for no signal benefit. One cache file for the
whole watchlist (not per-ticker like ``StructuralProfileCache``) since it's
always fetched as one batch call.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DEFAULT_CACHE_PATH = "results/p19_penny_intraday/sentiment_cache.json"


class SentimentCache:
    """Single-file batch cache: ``{"fetched_at": iso, "data": {ticker: {...}}}``."""

    def __init__(self, cache_path: str = DEFAULT_CACHE_PATH, ttl_minutes: int = 60) -> None:
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_minutes = ttl_minutes

    def _load(self) -> Optional[Dict[str, Any]]:
        if not self.cache_path.exists():
            return None
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            _logger.warning("Sentiment cache corrupt — treating as a cache miss")
            return None

    def is_fresh(self) -> bool:
        cached = self._load()
        if cached is None:
            return False
        fetched_at_str = cached.get("fetched_at")
        if not fetched_at_str:
            return False
        try:
            fetched_at = datetime.fromisoformat(fetched_at_str)
        except ValueError:
            return False
        age_minutes = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 60.0
        return age_minutes <= self.ttl_minutes

    def data(self) -> Dict[str, Any]:
        cached = self._load()
        return dict(cached.get("data", {})) if cached else {}

    def save(self, data: Dict[str, Any]) -> None:
        payload = {"fetched_at": datetime.now(timezone.utc).isoformat(), "data": data}
        self.cache_path.write_text(json.dumps(payload), encoding="utf-8")
