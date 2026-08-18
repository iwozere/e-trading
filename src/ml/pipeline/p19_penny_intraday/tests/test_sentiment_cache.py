"""Tests for the P19 sentiment batch cache (throttling, spec §10)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday.sentiment_cache import SentimentCache


def test_missing_cache_file_is_not_fresh(tmp_path):
    cache = SentimentCache(str(tmp_path / "sentiment.json"))
    assert cache.is_fresh() is False
    assert cache.data() == {}


def test_save_then_load_round_trips(tmp_path):
    cache = SentimentCache(str(tmp_path / "sentiment.json"))
    cache.save({"AAA": {"mentions_24h": 5}})
    assert cache.is_fresh() is True
    assert cache.data() == {"AAA": {"mentions_24h": 5}}


def test_stale_entry_past_ttl_is_not_fresh(tmp_path):
    cache = SentimentCache(str(tmp_path / "sentiment.json"), ttl_minutes=60)
    cache.save({"AAA": {"mentions_24h": 5}})
    # Rewrite fetched_at to be older than the TTL.
    import json
    from datetime import datetime, timedelta, timezone

    payload = json.loads(cache.cache_path.read_text(encoding="utf-8"))
    payload["fetched_at"] = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
    cache.cache_path.write_text(json.dumps(payload), encoding="utf-8")

    assert cache.is_fresh() is False
    assert cache.data() == {"AAA": {"mentions_24h": 5}}  # data() is still readable, just stale


def test_corrupt_cache_file_is_treated_as_a_miss(tmp_path):
    cache = SentimentCache(str(tmp_path / "sentiment.json"))
    cache.cache_path.write_text("not valid json{{{", encoding="utf-8")
    assert cache.is_fresh() is False
    assert cache.data() == {}
