"""Repository tests for retiring Kestrel catalysts (past -> passed, moved dates -> cancelled)."""

from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.data.db.models.model_kestrel import K20Catalyst
from src.data.db.repos.repo_kestrel import KestrelRepo

TODAY = date(2026, 11, 7)


def _add(session: Session, ticker: str, event_date: date, state: str = "upcoming", source: str = "finnhub",
         event_type: str = "earnings") -> K20Catalyst:
    row = K20Catalyst(ticker=ticker, event_type=event_type, event_date=event_date, source=source, state=state)
    session.add(row)
    session.flush()
    return row


def _states(session: Session) -> dict:
    rows = session.execute(select(K20Catalyst)).scalars().all()
    return {(r.ticker, r.event_type, r.event_date): r.state for r in rows}


def test_mark_past_catalysts_passed(db_session: Session) -> None:
    _add(db_session, "AAA", date(2026, 11, 4))
    _add(db_session, "BBB", date(2026, 11, 6), state="date_changed")
    _add(db_session, "CCC", date(2026, 11, 5), state="cancelled")
    _add(db_session, "DDD", TODAY)

    assert KestrelRepo(db_session).mark_past_catalysts_passed(TODAY) == 2

    states = _states(db_session)
    assert states[("AAA", "earnings", date(2026, 11, 4))] == "passed"
    assert states[("BBB", "earnings", date(2026, 11, 6))] == "passed"
    assert states[("CCC", "earnings", date(2026, 11, 5))] == "cancelled"
    assert states[("DDD", "earnings", TODAY)] == "upcoming"


def test_cancel_superseded_catalysts_only_touches_source_window(db_session: Session) -> None:
    _add(db_session, "ADMA", date(2026, 11, 9))  # moved -> cancel
    _add(db_session, "ADMA", date(2026, 11, 11))  # new date -> keep
    _add(db_session, "ADMA", date(2027, 3, 1))  # outside window -> keep
    _add(db_session, "ADMA", date(2026, 11, 20), source="pdufa.bio", event_type="pdufa")  # other source -> keep
    _add(db_session, "VNDA", date(2026, 11, 9))  # other ticker -> keep

    n = KestrelRepo(db_session).cancel_superseded_catalysts(
        "ADMA", "earnings", "finnhub", TODAY, date(2027, 2, 5), [date(2026, 11, 11)]
    )

    assert n == 1
    states = _states(db_session)
    assert states[("ADMA", "earnings", date(2026, 11, 9))] == "cancelled"
    assert states[("ADMA", "earnings", date(2026, 11, 11))] == "upcoming"
    assert states[("ADMA", "earnings", date(2027, 3, 1))] == "upcoming"
    assert states[("ADMA", "pdufa", date(2026, 11, 20))] == "upcoming"
    assert states[("VNDA", "earnings", date(2026, 11, 9))] == "upcoming"
