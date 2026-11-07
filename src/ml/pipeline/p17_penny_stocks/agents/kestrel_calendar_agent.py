"""
P17 Kestrel Calendar Agent

Stage 5.7: annotate each candidate with its next scheduled catalyst from P20
Kestrel's ``k20_catalysts`` table — earnings (Finnhub, Kestrel watchlist only),
PDUFA / AdCom / clinical readouts (pdufa.bio, whole Kestrel universe) and spin-offs.

Annotation only: the table is a current snapshot (rows are overwritten when dates
move, no "known since" column) so it cannot be back-tested; P17 records what it saw
each day in the candidates CSV so it can be evaluated later. Back-dated runs older
than ``kestrel_max_target_age_days`` are skipped, because today's snapshot would
leak future knowledge into them.

Kestrel never moves events to ``passed`` (2026-11-07: all 863 rows 'upcoming'), so
only events dated on/after the run day are used.
"""

import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p17_penny_stocks.config import P17FilingsConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

CatalystFetcher = Callable[[int], List[Dict[str, Any]]]


def _default_fetcher(days_ahead: int) -> List[Dict[str, Any]]:
    """Read upcoming catalysts for all tickers from the prod DB (imported lazily)."""
    from src.data.db.services.kestrel_service import KestrelService

    return KestrelService().get_all_upcoming_catalysts(days_ahead=days_ahead)


class KestrelCalendarAgent:
    """Populate ``next_catalyst_*`` fields from P20 Kestrel's catalyst calendar."""

    def __init__(
        self,
        config: P17FilingsConfig,
        target_date: str,
        fetcher: CatalystFetcher | None = None,
        today: date | None = None,
    ) -> None:
        self.cfg = config
        self.target = date.fromisoformat(target_date)
        self._fetch = fetcher or _default_fetcher
        self.today = today or date.today()

    def run(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Enrich candidates in place; never raises.

        Args:
            candidates: Candidate list.

        Returns:
            The same list.
        """
        if not self.cfg.kestrel_enabled or not candidates:
            return candidates
        age = (self.today - self.target).days
        if age > self.cfg.kestrel_max_target_age_days:
            _logger.info("Kestrel calendar skipped: target %s is %d days old (snapshot table)", self.target, age)
            return candidates
        try:
            rows = self._fetch(self.cfg.kestrel_days_ahead)
        except Exception:
            _logger.warning("Kestrel catalyst calendar unavailable — skipping annotation", exc_info=True)
            return candidates

        nearest = self._nearest_by_ticker(rows)
        hits = 0
        for c in candidates:
            ev = nearest.get(c.ticker.upper())
            if ev is None:
                continue
            event_date: date = ev["event_date"]
            c.next_catalyst_type = str(ev.get("event_type") or "")
            c.next_catalyst_date = event_date.isoformat()
            c.next_catalyst_days = (event_date - self.today).days
            c.next_catalyst_confidence = str(ev.get("confidence") or "")
            hits += 1
        _logger.info("Kestrel calendar: %d events loaded, %d/%d candidates annotated", len(rows), hits, len(candidates))
        return candidates

    def _nearest_by_ticker(self, rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Earliest event on/after today per ticker; confirmed beats estimated on the same date."""
        best: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            d = r.get("event_date")
            ticker = str(r.get("ticker") or "").upper()
            if not ticker or not isinstance(d, date) or d < self.today:
                continue
            cur = best.get(ticker)
            key = (d, 0 if r.get("confidence") == "confirmed" else 1)
            if cur is None or key < (cur["event_date"], 0 if cur.get("confidence") == "confirmed" else 1):
                best[ticker] = r
        return best
