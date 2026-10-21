"""Tests for the signal ledger routes (scorecard, paper-book NAV, recent signals)."""

from datetime import date
from unittest.mock import MagicMock, patch


def _rows() -> list:
    rows = []
    for day in range(1, 26):
        row = {"source": "p05_top_picks", "signal_date": date(2026, 9, day), "side": "long", "meta": None, "status": "complete"}
        for h in (1, 5, 10, 20):
            row.update({f"ret_{h}d": 0.03 + 0.001 * (day % 4), f"vt_{h}d": 0.01, f"spy_{h}d": 0.01, f"iwm_{h}d": 0.0})
        rows.append(row)
    return rows


def _nav() -> list:
    return [
        {"source": "p05_top_picks", "date": date(2026, 9, 2), "nav": 200_000.0, "benchmark_nav": 200_000.0, "invested": 0.0, "positions": 0, "opened": 0, "closed": 0},
        {"source": "p05_top_picks", "date": date(2026, 9, 3), "nav": 202_000.0, "benchmark_nav": 201_000.0, "invested": 2_000.0, "positions": 1, "opened": 1, "closed": 0},
    ]


def _service() -> MagicMock:
    service = MagicMock()
    service.list_signals.return_value = _rows()
    service.get_book_nav.side_effect = lambda source=None: [r for r in _nav() if source in (None, r["source"])]
    return service


def test_scorecard_returns_streams_books_and_counts(authenticated_client_viewer):
    with patch("src.api.signal_ledger_routes._service", return_value=_service()):
        response = authenticated_client_viewer.get("/api/signals/scorecard?benchmark=spy")

    assert response.status_code == 200
    data = response.json()
    assert data["benchmark"] == "SPY"
    stream = data["streams"][0]
    assert stream["stream"] == "p05_top_picks"
    assert {h["horizon"] for h in stream["horizons"]} == {1, 5, 10, 20}
    assert stream["horizons"][-1]["verdict"] == "edge"
    assert data["books"]["p05_top_picks"]["excess_return"] > 0
    assert data["status_counts"] == {"p05_top_picks": {"complete": 25}}


def test_scorecard_rejects_unknown_benchmark(authenticated_client_viewer):
    response = authenticated_client_viewer.get("/api/signals/scorecard?benchmark=qqq")
    assert response.status_code == 400


def test_book_nav_404_for_unknown_source(authenticated_client_viewer):
    with patch("src.api.signal_ledger_routes._service", return_value=_service()):
        assert authenticated_client_viewer.get("/api/signals/books/p05_top_picks/nav").json()["nav"][1]["positions"] == 1
        assert authenticated_client_viewer.get("/api/signals/books/nope/nav").status_code == 404


def test_recent_signals_passes_filters(authenticated_client_viewer):
    service = _service()
    with patch("src.api.signal_ledger_routes._service", return_value=service):
        response = authenticated_client_viewer.get("/api/signals/recent?source=p05_top_picks&limit=5")

    assert response.status_code == 200
    service.list_signals.assert_called_once_with(source="p05_top_picks", since=None, limit=5, newest_first=True)
