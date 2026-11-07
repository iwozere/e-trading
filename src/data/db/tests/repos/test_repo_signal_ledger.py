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


def test_book_nav_and_trades_are_kept_per_policy(db_session: Session) -> None:
    repo = SignalLedgerRepo(db_session)
    repo.upsert_signals([_signal("AAA", 1)])
    signal_id = repo.get_pending_signals()[0]["id"]
    nav = [{"date": date(2026, 10, 20), "nav": 1.0, "benchmark_nav": 1.0, "invested": 0.0, "positions": 0, "opened": 0, "closed": 0}]
    repo.replace_book_nav("p05_top_picks", nav)
    repo.replace_book_nav("p05_top_picks", [{**nav[0], "nav": 2.0}], policy="bracket")
    trade = {"signal_id": signal_id, "ticker": "AAA", "signal_date": date(2026, 10, 20), "entry_date": date(2026, 10, 21),
             "entry_price": 10.0, "exit_date": None, "exit_price": None, "exit_reason": "open", "sessions_held": 3,
             "net_return": 0.02, "last_price": 10.2, "mae": -0.01, "mfe": 0.03}
    repo.replace_book_trades("p05_top_picks", "bracket", [trade, {**trade, "ticker": "BBB", "exit_date": date(2026, 10, 23),
                                                               "exit_price": 9.0, "exit_reason": "stop"}])
    repo.replace_book_trades("p05_top_picks", "bracket", [trade])  # replace, not append

    assert [r["nav"] for r in repo.get_book_nav("p05_top_picks", "horizon")] == [1.0]
    assert {r["policy"] for r in repo.get_book_nav("p05_top_picks")} == {"horizon", "bracket"}
    trades = repo.get_book_trades(source="p05_top_picks", policy="bracket")
    assert [(t["ticker"], t["exit_reason"], t["signal_id"]) for t in trades] == [("AAA", "open", signal_id)]
    assert repo.get_book_trades(policy="bracket", open_only=True)[0]["mfe"] == 0.03
    assert repo.get_book_trades(policy="horizon") == []


def test_outcomes_store_long_horizons_and_excursions(db_session: Session) -> None:
    repo = SignalLedgerRepo(db_session)
    repo.upsert_signals([_signal("AAA", 1)])
    signal_id = repo.get_pending_signals()[0]["id"]
    repo.update_outcomes([{"id": signal_id, "status": "pending", "entry_date": date(2026, 10, 21), "entry_price": 10.0,
                           "ret_250d": 0.4, "vt_120d": 0.05, "mae_20d": -0.1, "mfe_20d": 0.2, "data_flag": None}])
    row = repo.list_signals()[0]
    assert (row["ret_250d"], row["vt_120d"], row["mae_20d"], row["mfe_20d"]) == (0.4, 0.05, -0.1, 0.2)


def test_book_stats_are_replaced_per_source(db_session: Session) -> None:
    repo = SignalLedgerRepo(db_session)
    repo.replace_book_stats("p05_top_picks", {"horizon": {"book": {"nav": 1.0}}, "bracket": {"book": {"nav": 2.0}}})
    repo.replace_book_stats("p05_top_picks", {"horizon": {"book": {"nav": 3.0}}})
    repo.replace_book_stats("p17_candidates", {"horizon": {"book": {"nav": 4.0}}})

    rows = repo.get_book_stats("p05_top_picks")
    assert [(r["policy"], r["stats"]["book"]["nav"]) for r in rows] == [("horizon", 3.0)]
    assert len(repo.get_book_stats()) == 2
