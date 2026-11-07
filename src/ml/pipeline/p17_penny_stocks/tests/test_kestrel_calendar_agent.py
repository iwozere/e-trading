"""Tests for P17 KestrelCalendarAgent (next scheduled catalyst annotation)."""

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p17_penny_stocks.agents.kestrel_calendar_agent import KestrelCalendarAgent
from src.ml.pipeline.p17_penny_stocks.config import P17FilingsConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate

TODAY = date(2026, 11, 9)


def _rows() -> List[Dict[str, Any]]:
    return [
        {"ticker": "AAA", "event_type": "earnings", "event_date": date(2026, 11, 20), "confidence": "confirmed"},
        {"ticker": "AAA", "event_type": "pdufa", "event_date": date(2026, 11, 12), "confidence": "estimated"},
        {"ticker": "AAA", "event_type": "earnings", "event_date": date(2026, 11, 1), "confidence": "confirmed"},  # past
        {"ticker": "bbb", "event_type": "clinical_readout", "event_date": date(2026, 11, 12), "confidence": "estimated"},
        {"ticker": "BBB", "event_type": "pdufa", "event_date": date(2026, 11, 12), "confidence": "confirmed"},
    ]


def _agent(target: str = "2026-11-08", fetcher: Any = None, **cfg: Any) -> KestrelCalendarAgent:
    return KestrelCalendarAgent(P17FilingsConfig(**cfg), target, fetcher=fetcher or (lambda days: _rows()), today=TODAY)


def test_nearest_future_event_per_ticker():
    a, b, c = Candidate(ticker="AAA"), Candidate(ticker="BBB"), Candidate(ticker="CCC")
    _agent().run([a, b, c])
    assert (a.next_catalyst_type, a.next_catalyst_days, a.next_catalyst_confidence) == ("pdufa", 3, "estimated")
    assert a.next_catalyst_date == "2026-11-12"
    assert (b.next_catalyst_type, b.next_catalyst_confidence) == ("pdufa", "confirmed")  # confirmed wins a tie
    assert c.next_catalyst_type == "" and c.next_catalyst_days is None


def test_backdated_run_is_skipped():
    """The table is a current snapshot — never annotate old target dates with today's knowledge."""
    a = Candidate(ticker="AAA")
    _agent(target="2026-10-01").run([a])
    assert a.next_catalyst_type == ""


def test_db_failure_is_swallowed():
    def boom(days: int) -> List[Dict[str, Any]]:
        raise RuntimeError("db down")

    a = Candidate(ticker="AAA")
    _agent(fetcher=boom).run([a])
    assert a.next_catalyst_type == ""


def test_disabled():
    a = Candidate(ticker="AAA")
    _agent(kestrel_enabled=False).run([a])
    assert a.next_catalyst_type == ""


def test_days_ahead_passed_to_fetcher():
    seen: List[int] = []

    def fetch(days: int) -> List[Dict[str, Any]]:
        seen.append(days)
        return []

    _agent(fetcher=fetch, kestrel_days_ahead=21).run([Candidate(ticker="AAA")])
    assert seen == [21]
