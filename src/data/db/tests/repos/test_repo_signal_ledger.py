"""Tests for SignalLedgerRepo (needs the Postgres test DB, see conftest.py)."""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from src.data.db.repos.repo_signal_ledger import SignalLedgerRepo


def _signal(ticker: str, rank: int) -> dict:
    return {"pipeline": "p05", "source": "p05_top_picks", "signal_date": date(2026, 10, 20), "ticker": ticker, "side": "long",
            "rank": rank, "score": 8.0, "horizon_days": 20, "meta": {"rank": rank}}


def test_upsert_counts_only_new_rows_and_keeps_outcomes(db_session: Session) -> None:
    repo = SignalLedgerRepo(db_session)
    assert repo.upsert_signals([_signal("AAA", 1), _signal("BBB", 2)]) == 2

    pending = {r["ticker"]: r["id"] for r in repo.get_pending_signals()}
    repo.update_outcomes([{"id": pending["AAA"], "status": "complete", "entry_date": date(2026, 10, 21), "entry_price": 10.0,
                           "ret_20d": 0.05, "data_flag": None}])

    assert repo.upsert_signals([_signal("AAA", 7), _signal("CCC", 3)]) == 1
    rows = {r["ticker"]: r for r in repo.list_signals(source="p05_top_picks")}
    assert rows["AAA"]["rank"] == 7
    assert rows["AAA"]["status"] == "complete"
    assert rows["AAA"]["ret_20d"] == 0.05
    assert {r["ticker"] for r in repo.get_pending_signals()} == {"BBB", "CCC"}


def test_replace_book_nav_overwrites_history(db_session: Session) -> None:
    repo = SignalLedgerRepo(db_session)
    nav = [{"date": date(2026, 10, d), "nav": 1.0 * d, "benchmark_nav": 1.0, "invested": 0.0, "positions": 0, "opened": 0, "closed": 0}
           for d in (20, 21)]
    repo.replace_book_nav("p05_top_picks", nav)
    repo.replace_book_nav("p05_top_picks", nav[:1])

    rows = repo.get_book_nav("p05_top_picks")
    assert [r["date"] for r in rows] == [date(2026, 10, 20)]
    assert rows[0]["nav"] == 20.0
