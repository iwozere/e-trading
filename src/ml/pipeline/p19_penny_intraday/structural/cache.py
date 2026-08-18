"""
P19 Layer 0 — per-ticker structural profile cache.

Weekly full refresh (spec §4.0); the profiler also passes the latest known
filing date for a CIK so a name can be re-profiled early if new filing
activity landed since the cache was last written (the "daily delta check").
Deliberately separate from ``EdgarDownloader``'s own file cache, which has no
TTL at all (design-v2.md §0.1) — this class is what actually enforces
freshness for Layer 0.
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional

from src.ml.pipeline.p19_penny_intraday.models.structural_profile import StructuralProfile
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DEFAULT_CACHE_DIR = "results/p19_penny_intraday/structural_cache"


class StructuralProfileCache:
    """Per-ticker JSON cache: ``{cache_dir}/{TICKER}.json``."""

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, ttl_days: int = 7) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days

    def _path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}.json"

    def load(self, ticker: str) -> Optional[StructuralProfile]:
        """Return the cached profile, or None if absent or unreadable."""
        path = self._path(ticker)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return StructuralProfile.from_dict(data)
        except Exception:
            _logger.warning("Structural cache corrupt for %s — treating as a cache miss", ticker)
            return None

    def save(self, profile: StructuralProfile) -> None:
        path = self._path(profile.ticker)
        path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")

    def is_fresh(self, ticker: str, as_of: date, latest_filing_date: Optional[date] = None) -> bool:
        """
        True if the cached profile does not need a re-fetch: within the weekly
        TTL, AND (when ``latest_filing_date`` is known) no filing has landed for
        this CIK since the profile was last computed.

        Args:
            ticker: Watchlist ticker.
            as_of: Today's date.
            latest_filing_date: Most recent filing date seen for this ticker's
                CIK, if resolvable — drives the daily delta check.
        """
        cached = self.load(ticker)
        if cached is None or cached.as_of is None:
            return False
        age_days = (as_of - cached.as_of).days
        if age_days < 0:
            return False  # clock skew / manually edited cache file — treat as stale
        if age_days > self.ttl_days:
            return False
        if latest_filing_date is not None and latest_filing_date > cached.as_of:
            return False
        return True
