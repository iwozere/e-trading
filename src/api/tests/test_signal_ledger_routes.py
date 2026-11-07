"""Tests for the signal ledger routes (scorecard, paper books per exit policy, trades, recent signals)."""

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
    rows = []
    for policy, gain in (("horizon", 2_000.0), ("bracket", -1_000.0)):
        rows += [
            {"source": "p05_top_picks", "policy": policy, "date": date(2026, 9, 2), "nav": 200_000.0, "benchmark_nav": 200_000.0,
             "invested": 0.0, "positions": 0, "opened": 0, "closed": 0},
            {"source": "p05_top_picks", "policy": policy, "date": date(2026, 9, 3), "nav": 200_000.0 + gain, "benchmark_nav": 201_000.0,
             "invested": 2_000.0, "positions": 1, "opened": 1, "closed": 0},
        ]
    return rows


def _trades() -> list:
    return [
        {"source": "p05_top_picks", "policy": "horizon", "ticker": "AAA", "exit_date": date(2026, 9, 20), "exit_reason": "time",
         "net_return": 0.04, "sessions_held": 20, "mae": -0.02, "mfe": 0.06},
        {"source": "p05_top_picks", "policy": "bracket", "ticker": "AAA", "exit_date": date(2026, 9, 8), "exit_reason": "stop",
         "net_return": -0.15, "sessions_held": 4, "mae": -0.15, "mfe": 0.01},
        {"source": "p05_top_picks", "policy": "horizon", "ticker": "BBB", "exit_date": None, "exit_reason": "open",
         "net_return": 0.01, "sessions_held": 2, "mae": 0.0, "mfe": 0.01},
    ]


def _service() -> MagicMock:
    service = MagicMock()
    service.list_signals.return_value = _rows()
    service.get_book_nav.side_effect = lambda source=None, policy=None: [
        r for r in _nav() if source in (None, r["source"]) and policy in (None, r["policy"])
    ]
    service.get_book_stats.side_effect = lambda source=None: [
        {"source": "p05_top_picks", "policy": name, "computed_at": None,
         "stats": {"book": {"excess_return": excess}, "trades": {"exit_reasons": reasons}, "all_signals": {"closed": closed}}}
        for name, excess, reasons, closed in (("bracket", -0.01, {"stop": 1}, 9), ("horizon", 0.01, {"time": 1}, 12))
        if source in (None, "p05_top_picks")
    ]
    service.get_book_trades.side_effect = lambda source=None, policy=None, open_only=False, limit=None: [
        t for t in _trades()
        if source in (None, t["source"]) and policy in (None, t["policy"]) and (not open_only or t["exit_date"] is None)
    ]
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
    assert data["books"]["p05_top_picks"]["excess_return"] > 0  # the default (horizon) book, not bracket
    assert data["status_counts"] == {"p05_top_picks": {"complete": 25}}
    assert "native" in {p["name"] for p in data["policies"]}


def test_scorecard_rejects_unknown_benchmark(authenticated_client_viewer):
    response = authenticated_client_viewer.get("/api/signals/scorecard?benchmark=qqq")
    assert response.status_code == 400


def test_book_nav_404_for_unknown_source(authenticated_client_viewer):
    with patch("src.api.signal_ledger_routes._service", return_value=_service()):
        assert authenticated_client_viewer.get("/api/signals/books/p05_top_picks/nav").json()["nav"][1]["positions"] == 1
        assert authenticated_client_viewer.get("/api/signals/books/nope/nav").status_code == 404
        bracket = authenticated_client_viewer.get("/api/signals/books/p05_top_picks/nav?policy=bracket").json()
        assert bracket["policy"] == "bracket" and bracket["nav"][1]["nav"] == 199_000.0
        assert authenticated_client_viewer.get("/api/signals/books/p05_top_picks/nav?policy=yolo").status_code == 400


def test_policy_comparison_has_book_and_trade_stats_per_policy(authenticated_client_viewer):
    with patch("src.api.signal_ledger_routes._service", return_value=_service()):
        data = authenticated_client_viewer.get("/api/signals/books/p05_top_picks/policies").json()
        missing = authenticated_client_viewer.get("/api/signals/books/nope/policies")

    by_name = {p["name"]: p for p in data["policies"]}
    assert list(by_name) == ["horizon", "bracket"]  # POLICIES order, only policies with stats
    assert by_name["bracket"]["trades"]["exit_reasons"] == {"stop": 1}
    assert by_name["bracket"]["book"]["excess_return"] < 0
    assert by_name["horizon"]["all_signals"]["closed"] == 12
    assert by_name["bracket"]["description"].startswith("Stop")
    assert missing.status_code == 404


def test_trades_and_open_positions(authenticated_client_viewer):
    with patch("src.api.signal_ledger_routes._service", return_value=_service()):
        trades = authenticated_client_viewer.get("/api/signals/books/p05_top_picks/trades?policy=horizon").json()["trades"]
        positions = authenticated_client_viewer.get("/api/signals/positions").json()["positions"]

    assert {t["ticker"] for t in trades} == {"AAA", "BBB"}
    assert [p["ticker"] for p in positions] == ["BBB"]


def test_recent_signals_passes_filters(authenticated_client_viewer):
    service = _service()
    with patch("src.api.signal_ledger_routes._service", return_value=service):
        response = authenticated_client_viewer.get("/api/signals/recent?source=p05_top_picks&limit=5")

    assert response.status_code == 200
    service.list_signals.assert_called_once_with(source="p05_top_picks", since=None, limit=5, newest_first=True)
