"""Tests for P20 Kestrel universe loader — chunked upsert / partial-persistence behavior."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.ingest import universe_loader


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
