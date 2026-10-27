"""Tests for AlpacaDataDownloader timezone handling of request bounds."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader import alpaca_data_downloader
from src.data.downloader.alpaca_data_downloader import AlpacaDataDownloader, _to_naive_utc


def _bar(ts: datetime) -> SimpleNamespace:
    return SimpleNamespace(timestamp=ts, open=1.0, high=2.0, low=0.5, close=1.5, volume=100)


class _FakeDataClient:
    """Returns one tz-aware daily bar per day in the request range, like alpaca-py does."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def get_stock_bars(self, request: dict) -> dict:
        self.requests.append(request)
        start, end = request["start"], request["end"]
        bars = []
        day = start.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        while day.replace(tzinfo=None) <= end:
            bars.append(_bar(day))
            day += timedelta(days=1)
        return {request["symbol_or_symbols"]: bars}


@pytest.fixture
def downloader(monkeypatch):
    """An AlpacaDataDownloader wired to a fake data client (no credentials, no network)."""
    monkeypatch.setattr(alpaca_data_downloader, "StockBarsRequest", lambda **kwargs: kwargs)
    dl = AlpacaDataDownloader.__new__(AlpacaDataDownloader)
    dl.interval_mapping = {"1d": "1Day"}
    dl.data_client = _FakeDataClient()
    return dl


def test_to_naive_utc_converts_aware_datetime():
    """A tz-aware datetime is converted to UTC before the tzinfo is dropped."""
    aware = datetime(2026, 10, 25, 9, 0, tzinfo=timezone(timedelta(hours=2)))

    assert _to_naive_utc(aware) == datetime(2026, 10, 25, 7, 0)


def test_to_naive_utc_keeps_naive_datetime():
    """A naive datetime is taken to be UTC already and returned unchanged."""
    naive = datetime(2026, 10, 25, 9, 0)

    assert _to_naive_utc(naive) is naive


def test_get_ohlcv_accepts_tz_aware_bounds(downloader):
    """
    Regression guard: tz-aware start/end raised "Cannot compare tz-naive and
    tz-aware datetime-like objects" when filtering the naive bar timestamps.
    """
    start = datetime(2026, 10, 20, tzinfo=timezone.utc)
    end = datetime(2026, 10, 23, tzinfo=timezone.utc)

    df = downloader.get_ohlcv("CTRA", "1d", start, end)

    assert len(df) == 4
    assert df.index.tz is None
    assert df.index.min() == datetime(2026, 10, 20)
    assert downloader.data_client.requests[0]["start"].tzinfo is None


def test_get_ohlcv_single_chunk_accepts_tz_aware_bounds(downloader):
    """The non-chunked path filters with the same converted bounds."""
    start = datetime(2026, 10, 20, tzinfo=timezone.utc)
    end = datetime(2026, 10, 21, tzinfo=timezone.utc)

    df = downloader.get_ohlcv("CTRA", "1d", start, end, enable_chunking=False, limit=10)

    assert len(df) == 2
