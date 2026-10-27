"""Tests for P20 Kestrel universe loader — chunked upsert / partial-persistence behavior."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.ingest import universe_loader
from src.model.telegram_bot import Fundamentals


def _fake_csv_df(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": tickers,
            "exchange": ["NASDAQ"] * len(tickers),
            "sector": ["Tech"] * len(tickers),
            "industry": ["Software"] * len(tickers),
        }
    )


@pytest.fixture
def upsert_calls(monkeypatch):
    """Patch upsert_universe_rows to record each chunk instead of hitting the DB."""
    calls: list[list[dict]] = []

    def _fake_upsert(rows):
        calls.append(list(rows))
        return len(rows)

    monkeypatch.setattr(universe_loader, "upsert_universe_rows", _fake_upsert)
    monkeypatch.setattr(universe_loader, "get_active_tickers", lambda: [])
    monkeypatch.setattr(universe_loader, "mark_tickers_delisted", lambda _tickers: 0)
    return calls


def test_run_upserts_in_chunks_not_once(monkeypatch, upsert_calls):
    """With 5 tickers and a chunk size of 2, upsert_universe_rows is called multiple times."""
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    monkeypatch.setattr(universe_loader, "_load_nasdaq_csv", lambda: _fake_csv_df(tickers))
    monkeypatch.setattr(universe_loader, "_UPSERT_CHUNK_SIZE", 2)
    monkeypatch.setattr(universe_loader, "_fetch_all_fundamentals", lambda tks: ((t, None) for t in tks))

    result = universe_loader.run()

    assert result["tickers_upserted"] == 5
    # 5 tickers at chunk size 2 -> chunks of [2, 2, 1] = 3 upsert calls, not 1.
    assert len(upsert_calls) == 3
    assert [len(c) for c in upsert_calls] == [2, 2, 1]


def test_run_partial_failure_keeps_earlier_chunks_persisted(monkeypatch, upsert_calls):
    """If fetching raises partway through, chunks already flushed stay upserted."""
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    monkeypatch.setattr(universe_loader, "_load_nasdaq_csv", lambda: _fake_csv_df(tickers))
    monkeypatch.setattr(universe_loader, "_UPSERT_CHUNK_SIZE", 2)

    def _flaky_fetch(tks):
        for i, t in enumerate(tks):
            if i == 3:
                raise RuntimeError("simulated scheduler kill")
            yield t, None

    monkeypatch.setattr(universe_loader, "_fetch_all_fundamentals", _flaky_fetch)

    with pytest.raises(RuntimeError):
        universe_loader.run()

    # The first full chunk (AAA, BBB) was flushed before the failure at ticker index 3.
    assert len(upsert_calls) == 1
    assert [row["ticker"] for row in upsert_calls[0]] == ["AAA", "BBB"]


def test_derive_balance_sheet_fields_from_yahoo_fields():
    """
    Regression guard: Fundamentals has no net_debt_ebitda / interest_coverage /
    revenue_yoy_growth attributes, so reading them by name returned None for
    every ticker. They must be derived from the fields Yahoo actually fills.
    """
    fund = Fundamentals(
        revenue_growth=0.08,
        gross_margin=0.45,
        total_debt=600.0,
        total_cash=100.0,
        enterprise_value=2_000.0,
        enterprise_value_to_ebitda=10.0,  # EBITDA = 200
    )

    fields = universe_loader._derive_balance_sheet_fields(fund)

    assert fields["revenue_yoy_growth"] == 0.08
    assert fields["gross_margin"] == 0.45
    assert fields["net_debt"] == 500.0
    assert fields["net_debt_ebitda"] == pytest.approx(2.5)
    assert fields["interest_coverage"] is None


def test_derive_balance_sheet_fields_net_cash_negative_ebitda():
    """Net cash is still recorded when EBITDA is negative (no meaningful ratio)."""
    fund = Fundamentals(total_debt=10.0, total_cash=90.0, enterprise_value=500.0, enterprise_value_to_ebitda=-4.0)

    fields = universe_loader._derive_balance_sheet_fields(fund)

    assert fields["net_debt"] == -80.0
    assert fields["net_debt_ebitda"] is None


def test_derive_balance_sheet_fields_all_missing():
    """No cash/debt data leaves every derived field None."""
    fields = universe_loader._derive_balance_sheet_fields(Fundamentals())

    assert fields["net_debt"] is None
    assert fields["net_debt_ebitda"] is None


def test_fetch_fundamentals_for_ticker_treats_empty_result_as_failure(monkeypatch):
    """
    Regression guard: get_fundamentals_unified returns an all-None Fundamentals
    when Yahoo rate-limits; that must count as a failure so the retry pass runs.
    """

    async def _empty(ticker, provider=None):
        return Fundamentals(ticker=ticker, sources={})

    monkeypatch.setattr(universe_loader, "get_fundamentals_unified", _empty)

    assert universe_loader._fetch_fundamentals_for_ticker("AAA") is None


def test_fetch_fundamentals_for_ticker_returns_populated_result(monkeypatch):
    """A Fundamentals with real data passes through unchanged."""
    fund = Fundamentals(ticker="AAA", market_cap=1e9)

    async def _populated(ticker, provider=None):
        return fund

    monkeypatch.setattr(universe_loader, "get_fundamentals_unified", _populated)

    assert universe_loader._fetch_fundamentals_for_ticker("AAA") is fund


def test_fetch_all_fundamentals_retries_empty_results(monkeypatch):
    """Tickers that come back empty in pass 1 are retried in pass 2."""
    attempts: dict[str, int] = {}

    async def _rate_limited_once(ticker, provider=None):
        attempts[ticker] = attempts.get(ticker, 0) + 1
        if ticker == "BBB" and attempts[ticker] == 1:
            return Fundamentals(ticker=ticker)
        return Fundamentals(ticker=ticker, market_cap=1e9)

    monkeypatch.setattr(universe_loader, "get_fundamentals_unified", _rate_limited_once)
    monkeypatch.setattr(universe_loader, "_RETRY_COOLDOWN_SECONDS", 0)

    results = list(universe_loader._fetch_all_fundamentals(["AAA", "BBB"]))

    assert [(t, f is None) for t, f in results] == [("AAA", False), ("BBB", True), ("BBB", False)]
    assert attempts == {"AAA": 1, "BBB": 2}


def test_run_failed_ticker_keeps_stored_balance_sheet_fields(monkeypatch, upsert_calls):
    """
    A ticker without fundamentals is upserted without balance-sheet keys (so the
    stored values survive), in a separate statement from tickers that have them.
    """
    tickers = ["AAA", "BBB"]
    monkeypatch.setattr(universe_loader, "_load_nasdaq_csv", lambda: _fake_csv_df(tickers))
    fund = Fundamentals(total_debt=10.0, total_cash=5.0)
    monkeypatch.setattr(
        universe_loader, "_fetch_all_fundamentals", lambda tks: iter([("AAA", fund), ("BBB", None)])
    )

    result = universe_loader.run()

    assert result["tickers_upserted"] == 2
    assert len(upsert_calls) == 2
    rows_by_ticker = {row["ticker"]: row for call in upsert_calls for row in call}
    assert rows_by_ticker["AAA"]["net_debt"] == 5.0
    assert "net_debt" not in rows_by_ticker["BBB"]
    assert "gross_margin" not in rows_by_ticker["BBB"]
