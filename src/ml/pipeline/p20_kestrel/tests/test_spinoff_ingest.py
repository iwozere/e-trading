"""Tests for the Sleeve B2 spin-off registration monitor (gap 10.2)."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.ml.pipeline.p20_kestrel.ingest import spinoff_ingest
from src.ml.pipeline.p20_kestrel.ingest.spinoff_ingest import _build_cik_to_ticker, run

_TODAY = date(2026, 8, 27)


class _FakeEdgarDownloader:
    """Stands in for EdgarDownloader — no network, no real cache dir."""

    def __init__(self, filings: pd.DataFrame, tickers_file: str):
        self._filings = filings
        self._tickers_file = tickers_file

    def download_form10_filings(self, as_of_date=None):
        return self._filings

    def download_company_tickers(self) -> Path:
        return Path(self._tickers_file)


@pytest.fixture
def company_tickers_file(tmp_path):
    path = tmp_path / "company_tickers.json"
    path.write_text(
        json.dumps(
            {
                "0": {"cik_str": 1659166, "ticker": "FTV", "title": "Fortive Corp"},
                "1": {"cik_str": 1653477, "ticker": "NGVT", "title": "Ingevity Corp"},
            }
        ),
        encoding="utf-8",
    )
    return str(path)


# ---------------------------------------------------------------------------
# _build_cik_to_ticker
# ---------------------------------------------------------------------------


def test_build_cik_to_ticker_maps_and_uppercases(company_tickers_file):
    edgar = _FakeEdgarDownloader(pd.DataFrame(), company_tickers_file)

    mapping = _build_cik_to_ticker(edgar)

    assert mapping == {"1659166": "FTV", "1653477": "NGVT"}


def test_build_cik_to_ticker_returns_empty_on_missing_file(tmp_path):
    edgar = _FakeEdgarDownloader(pd.DataFrame(), str(tmp_path / "does_not_exist.json"))

    assert _build_cik_to_ticker(edgar) == {}


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


@pytest.fixture
def run_env(monkeypatch):
    monkeypatch.setattr(spinoff_ingest, "start_job_run", lambda *a, **kw: None)
    monkeypatch.setattr(spinoff_ingest, "finish_job_run", lambda *a, **kw: None)

    upserted: list[dict] = []
    monkeypatch.setattr(spinoff_ingest, "upsert_catalyst", lambda row: upserted.append(row))
    return upserted


def _patch_edgar(monkeypatch, filings: pd.DataFrame, tickers_file: str):
    monkeypatch.setattr(
        spinoff_ingest,
        "EdgarDownloader",
        lambda: _FakeEdgarDownloader(filings, tickers_file),
    )


def test_run_upserts_catalyst_for_resolvable_ticker(monkeypatch, run_env, company_tickers_file):
    filings = pd.DataFrame(
        [
            {
                "cik": "1659166",
                "entity_name": "Fortive Corp",
                "accession_number": "0001193125-15-394365",
                "filed_date": "2026-08-26",
                "form_type": "10-12B",
            }
        ]
    )
    _patch_edgar(monkeypatch, filings, company_tickers_file)

    result = run(as_of_date=_TODAY)

    assert result["filings_seen"] == 1
    assert result["tickers_resolved"] == 1
    assert result["catalysts_upserted"] == 1
    row = run_env[0]
    assert row["ticker"] == "FTV"
    assert row["event_type"] == "spinoff"
    assert row["event_date"] == "2026-08-26"
    assert row["confidence"] == "estimated"
    assert row["source"] == "edgar_form10"


def test_run_skips_unresolvable_cik(monkeypatch, run_env, company_tickers_file):
    filings = pd.DataFrame(
        [
            {
                "cik": "9999999",  # not in company_tickers.json fixture
                "entity_name": "Some Newco Not Yet Listed",
                "accession_number": "0000000000-26-000001",
                "filed_date": "2026-08-26",
                "form_type": "10-12B",
            }
        ]
    )
    _patch_edgar(monkeypatch, filings, company_tickers_file)

    result = run(as_of_date=_TODAY)

    assert result["filings_seen"] == 1
    assert result["tickers_resolved"] == 0
    assert result["catalysts_upserted"] == 0


def test_run_flags_amendments_in_notes(monkeypatch, run_env, company_tickers_file):
    filings = pd.DataFrame(
        [
            {
                "cik": "1659166",
                "entity_name": "Fortive Corp",
                "accession_number": "0001193125-15-400000",
                "filed_date": "2026-08-26",
                "form_type": "10-12B/A",
            }
        ]
    )
    _patch_edgar(monkeypatch, filings, company_tickers_file)

    run(as_of_date=_TODAY)

    assert "[amendment]" in run_env[0]["notes"]


def test_run_handles_no_filings(monkeypatch, run_env, company_tickers_file):
    _patch_edgar(monkeypatch, pd.DataFrame(), company_tickers_file)

    result = run(as_of_date=_TODAY)

    assert result == {"filings_seen": 0, "tickers_resolved": 0, "catalysts_upserted": 0}
