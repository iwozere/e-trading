"""Tests for OpenFigiMapper CUSIP resolution rules and cache versioning."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader import openfigi_mapper
from src.data.downloader.openfigi_mapper import OpenFigiMapper, _select_equity_ticker


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self.status_code = 200
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    """Answers each mapping job from a {cusip: data_list} table and records the jobs sent."""

    def __init__(self, answers: Dict[str, List[Dict[str, Any]]]) -> None:
        self.answers = answers
        self.jobs: List[List[Dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        del url
        jobs = json.loads(kwargs["data"])
        self.jobs.append(jobs)
        items = []
        for job in jobs:
            hits = self.answers.get(job["idValue"], [])
            items.append({"data": hits} if hits else {"warning": "No identifier found."})
        return _FakeResponse(items)


def _hit(ticker: str, security_type: str = "Common Stock", exch: str = "US") -> Dict[str, Any]:
    return {"ticker": ticker, "exchCode": exch, "securityType": security_type}


@pytest.fixture
def mapper(tmp_path, monkeypatch):
    """A keyed-tier mapper with a temp cache dir and no request throttling."""
    monkeypatch.setattr(openfigi_mapper, "_MIN_INTERVAL_KEYED", 0.0)
    return OpenFigiMapper(cache_dir=tmp_path, api_key="test-key")


def test_select_equity_ticker_converts_share_class_slash():
    """OpenFIGI's BRK/B becomes BRK-B, the form the OHLCV providers use."""
    assert _select_equity_ticker([_hit("BRK/B")]) == "BRK-B"


def test_select_equity_ticker_accepts_adr():
    """ADRs (e.g. TSM) are equities and resolve to their ticker."""
    assert _select_equity_ticker([_hit("TSM", security_type="ADR")]) == "TSM"


@pytest.mark.parametrize(
    "hit",
    [
        _hit("AAPL 3.25 02/23/26", security_type="GLOBAL"),
        _hit("BAC 5 PERP LL", security_type="PUBLIC"),
        _hit("ACHR/WS", security_type="Equity WRT"),
    ],
)
def test_select_equity_ticker_rejects_non_equity(hit):
    """Regression guard: bonds, preferreds and warrants used to fall back to data[0]."""
    assert _select_equity_ticker([hit]) is None


def test_map_cusips_requests_us_composite_only(mapper, monkeypatch):
    """Every mapping job carries exchCode=US so foreign lines (6D8, DWDPEUR) are never returned."""
    session = _FakeSession({"084670702": [_hit("BRK/B")]})
    monkeypatch.setattr(mapper, "_session", session)

    result = mapper.map_cusips(["084670702", "26614N102"])

    assert result == {"084670702": "BRK-B", "26614N102": None}
    assert all(job["exchCode"] == "US" for job in session.jobs[0])


def test_map_cusips_free_tier_batches_ten(tmp_path, monkeypatch):
    """Without a key OpenFIGI rejects more than 10 jobs per request (HTTP 413)."""
    monkeypatch.setattr(openfigi_mapper, "_MIN_INTERVAL_FREE", 0.0)
    monkeypatch.setattr(OpenFigiMapper, "_get_config_value", staticmethod(lambda *_args, **_kwargs: None))
    free = OpenFigiMapper(cache_dir=tmp_path)
    session = _FakeSession({})
    monkeypatch.setattr(free, "_session", session)

    free.map_cusips([f"{i:09d}" for i in range(25)])

    assert [len(jobs) for jobs in session.jobs] == [10, 10, 5]


def test_map_cusips_re_resolves_rows_from_older_resolver(mapper, tmp_path, monkeypatch):
    """Cache rows without the current resolver tag are fetched again; current rows are reused."""
    cache_file = tmp_path / "openfigi" / "cusip_map.csv.gz"
    cache_file.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"cusip": "26614N102", "ticker": "6D8", "resolved_at": "2026-06-26T00:00:00Z"},
            {"cusip": "037833100", "ticker": "AAPL", "resolved_at": "2026-06-26T00:00:00Z"},
        ]
    ).to_csv(cache_file, index=False, compression="gzip")
    session = _FakeSession({"037833100": [_hit("AAPL")]})
    monkeypatch.setattr(mapper, "_session", session)

    first = mapper.map_cusips(["26614N102", "037833100"])

    assert first == {"26614N102": None, "037833100": "AAPL"}
    assert sorted(job["idValue"] for job in session.jobs[0]) == ["037833100", "26614N102"]
    saved = pd.read_csv(cache_file, compression="gzip", dtype=str)
    assert set(saved["resolver"]) == {openfigi_mapper._RESOLVER_VERSION}

    fresh = OpenFigiMapper(cache_dir=tmp_path, api_key="test-key")
    fresh_session = _FakeSession({})
    monkeypatch.setattr(fresh, "_session", fresh_session)
    assert fresh.map_cusips(["26614N102", "037833100"]) == first
    assert fresh_session.jobs == []
