"""
Tests for eod_ingest signal computation.

Regression test: signal values derived from pandas reductions (e.g. adv_20d)
were numpy.float64 scalars; under numpy 2.x psycopg2 rendered them as
"np.float64(...)" in SQL, failing the k20_signals upsert with
InvalidSchemaName: schema "np" does not exist.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("talib")

from src.ml.pipeline.p20_kestrel.ingest import eod_ingest
from src.ml.pipeline.p20_kestrel.ingest.eod_ingest import _compute_signals_for_ticker


def _make_ohlcv(rows: int = 250) -> pd.DataFrame:
    """Build a synthetic upward-trending OHLCV frame."""
    rng = np.random.default_rng(42)
    index = pd.bdate_range(end="2026-07-08", periods=rows)
    close = pd.Series(100.0 + np.arange(rows) * 0.1 + rng.normal(0, 0.5, rows), index=index)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, rows).astype("int64"),
        },
        index=index,
    )


def test_compute_signals_returns_expected_types():
    rows = _compute_signals_for_ticker("TEST", _make_ohlcv(), date(2026, 7, 8))

    assert rows, "expected signal rows for a 250-bar frame"
    signal_types = {r["signal_type"] for r in rows}
    assert {"close", "adv_20d", "return_3m", "return_6m", "two_yr_high"} <= signal_types

    for row in rows:
        # Must be builtin float, not numpy.float64 — psycopg2 binds float
        # subclasses via repr(), which is not a SQL literal under numpy 2.x.
        assert type(row["value"]) is float, f"{row['signal_type']} value is {type(row['value'])!r}"


def test_compute_signals_short_history_returns_empty():
    assert _compute_signals_for_ticker("TEST", _make_ohlcv(10), date(2026, 7, 8)) == []


class _FakeDataManager:
    """Stand-in for DataManager — run() only needs get_ohlcv_batch to exist."""

    def get_ohlcv_batch(self, tickers, timeframe, start_date=None, end_date=None):
        return {}


@pytest.fixture
def upsert_calls(monkeypatch):
    """Patch upsert_signals to record each chunk instead of hitting the DB."""
    calls: list[list[dict]] = []

    def _fake_upsert(rows):
        calls.append(list(rows))
        return len(rows)

    monkeypatch.setattr(eod_ingest, "upsert_signals", _fake_upsert)
    monkeypatch.setattr(eod_ingest, "start_job_run", lambda *a, **k: None)
    monkeypatch.setattr(eod_ingest, "finish_job_run", lambda *a, **k: None)
    monkeypatch.setattr(eod_ingest, "DataManager", _FakeDataManager)
    monkeypatch.setattr(eod_ingest, "_ingest_vix_signal", lambda target_date: [])
    return calls


def test_run_upserts_signals_in_chunks_not_once(monkeypatch, upsert_calls):
    """With 5 tickers and a chunk size of 2, upsert_signals is called multiple times."""
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    monkeypatch.setattr(eod_ingest, "get_active_tickers", lambda: tickers)
    monkeypatch.setattr(eod_ingest, "_UPSERT_CHUNK_SIZE", 2)

    def _fake_process(ticker, **kwargs):
        row = {"ticker": ticker, "date": kwargs["target_date"], "signal_type": "close", "value": 1.0}
        return ticker, [row], True

    monkeypatch.setattr(eod_ingest, "_process_ticker", _fake_process)

    result = eod_ingest.run(as_of_date=date(2026, 7, 8))

    assert result == {"tickers_ok": 5, "tickers_failed": 0, "signals_upserted": 5}
    # 5 tickers at chunk size 2 -> chunks of [2, 2, 1] = 3 upsert calls, not 1.
    assert len(upsert_calls) == 3
    assert [len(c) for c in upsert_calls] == [2, 2, 1]


def test_run_crash_mid_compute_keeps_earlier_chunks_persisted(monkeypatch, upsert_calls):
    """If a ticker's compute raises partway through, chunks already flushed stay upserted."""
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    monkeypatch.setattr(eod_ingest, "get_active_tickers", lambda: tickers)
    monkeypatch.setattr(eod_ingest, "_UPSERT_CHUNK_SIZE", 2)

    def _flaky_process(ticker, **kwargs):
        if ticker == "DDD":
            raise RuntimeError("simulated scheduler kill")
        row = {"ticker": ticker, "date": kwargs["target_date"], "signal_type": "close", "value": 1.0}
        return ticker, [row], True

    monkeypatch.setattr(eod_ingest, "_process_ticker", _flaky_process)

    with pytest.raises(RuntimeError):
        eod_ingest.run(as_of_date=date(2026, 7, 8))

    # The first full chunk (AAA, BBB) was flushed before the crash on DDD.
    assert len(upsert_calls) == 1
    assert [row["ticker"] for row in upsert_calls[0]] == ["AAA", "BBB"]


class _FakeVixDownloader:
    """Stand-in for VIXDataDownloader — records the path it was asked to refresh."""

    def __init__(self, raise_on_update: bool = False):
        self.raise_on_update = raise_on_update
        self.updated_file = None

    def update_vix(self, vix_file=None):
        if self.raise_on_update:
            raise RuntimeError("simulated Yahoo outage")
        self.updated_file = vix_file


def test_ingest_vix_signal_returns_latest_close_on_or_before_target(monkeypatch, tmp_path):
    vix_file = tmp_path / "vix.csv"
    vix_file.write_text("date,vix,regime\n2026-07-06,14.0,calm\n2026-07-07,15.5,calm\n2026-07-09,20.0,normal\n")

    monkeypatch.setattr(eod_ingest, "_VIX_FILE", vix_file)
    monkeypatch.setattr(eod_ingest, "_vix_downloader", _FakeVixDownloader())

    rows = eod_ingest._ingest_vix_signal(date(2026, 7, 8))

    assert rows == [{"ticker": "VIX", "date": date(2026, 7, 8), "signal_type": "close", "value": 15.5}]


def test_ingest_vix_signal_empty_when_download_fails(monkeypatch, tmp_path):
    vix_file = tmp_path / "vix.csv"
    monkeypatch.setattr(eod_ingest, "_VIX_FILE", vix_file)
    monkeypatch.setattr(eod_ingest, "_vix_downloader", _FakeVixDownloader(raise_on_update=True))

    assert eod_ingest._ingest_vix_signal(date(2026, 7, 8)) == []


def test_ingest_vix_signal_empty_when_no_data_before_target(monkeypatch, tmp_path):
    vix_file = tmp_path / "vix.csv"
    vix_file.write_text("date,vix,regime\n2026-07-09,20.0,normal\n")

    monkeypatch.setattr(eod_ingest, "_VIX_FILE", vix_file)
    monkeypatch.setattr(eod_ingest, "_vix_downloader", _FakeVixDownloader())

    assert eod_ingest._ingest_vix_signal(date(2026, 7, 8)) == []


def test_run_includes_vix_signal(monkeypatch, upsert_calls):
    """run() upserts the VIX close row alongside per-ticker signals."""
    monkeypatch.setattr(eod_ingest, "get_active_tickers", lambda: [])
    monkeypatch.setattr(
        eod_ingest,
        "_ingest_vix_signal",
        lambda target_date: [{"ticker": "VIX", "date": target_date, "signal_type": "close", "value": 16.2}],
    )

    result = eod_ingest.run(as_of_date=date(2026, 7, 8))

    assert result == {"tickers_ok": 0, "tickers_failed": 0, "signals_upserted": 1}
    assert upsert_calls == [[{"ticker": "VIX", "date": date(2026, 7, 8), "signal_type": "close", "value": 16.2}]]
