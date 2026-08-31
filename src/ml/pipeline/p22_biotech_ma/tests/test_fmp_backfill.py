"""Tests for ingest/fmp_backfill.py. No live DB/network — repo and FMPClient are mocked/faked."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.ingest.fmp_backfill import (
    RAW_PRICE_SOURCE,
    build_backfill_targets,
    land_historical_prices,
    resolve_ticker_by_name,
)
from src.ml.pipeline.p22_biotech_ma.ingest.fmp_universe import TickerTarget, UnresolvedCompany


def _repo(companies):
    repo = MagicMock()
    repo.list_companies_full.return_value = companies
    return repo


def test_resolve_ticker_by_name_exact_match():
    company = UnresolvedCompany(cik="0000000100", name="Example Biotech Inc")
    client = MagicMock()
    client.search_company_by_name.return_value = [
        {"symbol": "EXBI", "name": "Example Biotech Inc"},
        {"symbol": "OTHR", "name": "Something Unrelated Corp"},
    ]

    result = resolve_ticker_by_name(company, client)

    assert result is not None
    assert result.ticker == "EXBI"
    assert result.cik == "0000000100"


def test_resolve_ticker_by_name_no_exact_match_returns_none():
    """A fuzzy-but-not-exact candidate must NOT be auto-accepted — see module docstring."""
    company = UnresolvedCompany(cik="0000000100", name="Example Biotech Inc")
    client = MagicMock()
    client.search_company_by_name.return_value = [{"symbol": "EXBI", "name": "Example Biotech Incorporated"}]

    assert resolve_ticker_by_name(company, client) is None


def test_resolve_ticker_by_name_prefers_usd_us_exchange_among_exact_matches():
    """Live-discovered 2026-08-31: searching a real company can return multiple exact-name
    matches across exchanges (e.g. Moderna's real NASDAQ listing AND a Frankfurt cross-listing,
    both literally named "Moderna, Inc."). The German one appeared FIRST in the real response —
    picking "the first exact match" would have been wrong."""
    company = UnresolvedCompany(cik="0001682852", name="Moderna, Inc.")
    client = MagicMock()
    client.search_company_by_name.return_value = [
        {"symbol": "0QF.F", "name": "Moderna, Inc.", "currency": "EUR", "exchange": "FSX"},
        {"symbol": "MRNA", "name": "Moderna, Inc.", "currency": "USD", "exchange": "NASDAQ"},
    ]

    result = resolve_ticker_by_name(company, client)

    assert result is not None
    assert result.ticker == "MRNA"


def test_resolve_ticker_by_name_falls_back_to_first_when_no_us_listing():
    company = UnresolvedCompany(cik="0001682852", name="Foreign Only Inc")
    client = MagicMock()
    client.search_company_by_name.return_value = [
        {"symbol": "FGN.L", "name": "Foreign Only Inc", "currency": "GBP", "exchange": "LSE"},
        {"symbol": "FGN.PA", "name": "Foreign Only Inc", "currency": "EUR", "exchange": "PAR"},
    ]

    result = resolve_ticker_by_name(company, client)

    assert result is not None
    assert result.ticker == "FGN.L"  # first exact match, since no US listing exists


def test_resolve_ticker_by_name_no_candidates_returns_none():
    company = UnresolvedCompany(cik="0000000100", name="Totally Obscure Co")
    client = MagicMock()
    client.search_company_by_name.return_value = []

    assert resolve_ticker_by_name(company, client) is None


def test_build_backfill_targets_known_only_skips_search():
    repo = _repo([{"company_id": 1, "cik": "0001", "ticker": "ABCD", "name": "Known Co"}])
    client = MagicMock()

    result = build_backfill_targets(repo, include_unresolved=False, client=client)

    assert len(result["targets"]) == 1
    assert result["resolved_via_search"] == 0
    assert result["still_unresolved"] == []
    client.search_company_by_name.assert_not_called()


def test_build_backfill_targets_includes_resolved_unresolved_companies(monkeypatch):
    repo = _repo([{"company_id": 1, "cik": "0001", "ticker": "ABCD", "name": "Known Co"}])
    monkeypatch.setattr(
        "src.ml.pipeline.p22_biotech_ma.ingest.fmp_backfill.build_unresolved_universe",
        lambda repo: [UnresolvedCompany(cik="0002", name="Newly Found Inc")],
    )
    client = MagicMock()
    client.search_company_by_name.return_value = [{"symbol": "NFI", "name": "Newly Found Inc"}]

    result = build_backfill_targets(repo, include_unresolved=True, client=client)

    tickers = {t.ticker for t in result["targets"]}
    assert tickers == {"ABCD", "NFI"}
    assert result["resolved_via_search"] == 1
    assert result["still_unresolved"] == []


def test_build_backfill_targets_logs_unresolved_when_no_match(monkeypatch):
    repo = _repo([])
    monkeypatch.setattr(
        "src.ml.pipeline.p22_biotech_ma.ingest.fmp_backfill.build_unresolved_universe",
        lambda repo: [UnresolvedCompany(cik="0002", name="Cannot Find This Inc")],
    )
    client = MagicMock()
    client.search_company_by_name.return_value = []

    result = build_backfill_targets(repo, include_unresolved=True, client=client)

    assert result["targets"] == []
    assert len(result["still_unresolved"]) == 1
    assert result["still_unresolved"][0].name == "Cannot Find This Inc"


def test_land_historical_prices_writes_raw_zone_and_counts(tmp_path):
    targets = [TickerTarget(company_id=1, cik="0001", ticker="ABCD", name="Known Co")]
    client = MagicMock()
    client.fetch_historical_price_full.return_value = [{"symbol": "ABCD", "date": "2024-01-02", "close": 100.0}]

    result = land_historical_prices(
        targets, start_date=date(2000, 1, 1), end_date=date(2024, 1, 1), client=client, root=tmp_path
    )

    assert result == {"landed": 1, "skipped_already_landed": 0, "failed": []}
    landed = raw_zone.read_latest_partition(RAW_PRICE_SOURCE, root=tmp_path)
    assert len(landed) == 1


def test_land_historical_prices_records_failed_ticker_on_none_response(tmp_path):
    targets = [TickerTarget(company_id=1, cik="0001", ticker="NODATA", name="No Data Co")]
    client = MagicMock()
    client.fetch_historical_price_full.return_value = None

    result = land_historical_prices(
        targets, start_date=date(2000, 1, 1), end_date=date(2024, 1, 1), client=client, root=tmp_path
    )

    assert result["landed"] == 0
    assert result["failed"] == ["NODATA"]


def test_land_historical_prices_skips_already_landed(tmp_path):
    raw_zone.write(
        source=RAW_PRICE_SOURCE, entity="ABCD", as_of_date=date(2024, 1, 1),
        payload=[{"symbol": "ABCD", "date": "2023-01-01", "close": 50.0}], root=tmp_path,
    )
    targets = [TickerTarget(company_id=1, cik="0001", ticker="ABCD", name="Known Co")]
    client = MagicMock()

    result = land_historical_prices(
        targets, start_date=date(2000, 1, 1), end_date=date(2024, 1, 1), client=client, root=tmp_path
    )

    assert result == {"landed": 0, "skipped_already_landed": 1, "failed": []}
    client.fetch_historical_price_full.assert_not_called()


def test_land_historical_prices_force_bypasses_skip(tmp_path):
    raw_zone.write(
        source=RAW_PRICE_SOURCE, entity="ABCD", as_of_date=date(2024, 1, 1),
        payload=[{"symbol": "ABCD", "date": "2023-01-01", "close": 50.0}], root=tmp_path,
    )
    targets = [TickerTarget(company_id=1, cik="0001", ticker="ABCD", name="Known Co")]
    client = MagicMock()
    client.fetch_historical_price_full.return_value = [{"symbol": "ABCD", "date": "2024-06-01", "close": 120.0}]

    result = land_historical_prices(
        targets, start_date=date(2000, 1, 1), end_date=date(2024, 1, 1), client=client,
        skip_already_landed=False, root=tmp_path,
    )

    assert result["landed"] == 1
    client.fetch_historical_price_full.assert_called_once()
