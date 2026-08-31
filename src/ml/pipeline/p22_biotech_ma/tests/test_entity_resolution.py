"""
Tests for ingest/entity_resolution.py (spec §2.0.2, §2.0.3, §3.3, M2 slice).
No live DB; `write_universe` is exercised against a `MagicMock` repo.
"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.entity_resolution import (
    build_universe,
    build_universe_history,
    fetch_ticker_exchange_map,
    is_likely_spac,
    normalize_cik,
    normalize_company_name,
    write_universe,
)


@pytest.fixture(autouse=True)
def _no_real_sleep():
    with patch("src.ml.pipeline.p22_biotech_ma.ingest.http_retry.time.sleep"):
        yield


def _mock_json_response(payload, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


def test_fetch_ticker_exchange_map_parses_and_normalizes_cik():
    client = MagicMock()
    client.get.return_value = _mock_json_response(
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"], [59478, "Eli Lilly & Co", "LLY", "NYSE"]],
        }
    )

    mapping = fetch_ticker_exchange_map(client)

    assert mapping["0000320193"] == ("AAPL", "Nasdaq")
    assert mapping["0000059478"] == ("LLY", "NYSE")


def test_fetch_ticker_exchange_map_returns_empty_on_failure():
    client = MagicMock()
    client.get.return_value = _mock_json_response({}, status_code=500)

    mapping = fetch_ticker_exchange_map(client)

    assert mapping == {}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Acme Therapeutics, Inc.", "acme"),
        ("ACME PHARMACEUTICALS CORP", "acme"),
        ("Acme  Biosciences   Ltd.", "acme"),
        ("Acme Holdings PLC", "acme"),
    ],
)
def test_normalize_company_name_strips_suffixes_and_collapses_whitespace(raw, expected):
    assert normalize_company_name(raw) == expected


def test_normalize_company_name_different_companies_stay_distinct():
    assert normalize_company_name("Acme Inc") != normalize_company_name("Beta Corp")


@pytest.mark.parametrize(
    "name",
    [
        "Global Healthcare Acquisition Corp",
        "Biotech Special Purpose Acquisition Company",
        "Frontier Blank Check Holdings",
    ],
)
def test_is_likely_spac_positive_cases(name):
    assert is_likely_spac(name) is True


@pytest.mark.parametrize("name", ["Acme Therapeutics Inc", "Beta Pharmaceuticals Corp", "Gamma Biosciences Ltd"])
def test_is_likely_spac_negative_cases(name):
    assert is_likely_spac(name) is False


def test_normalize_cik_zero_pads_and_strips_leading_zeros_first():
    assert normalize_cik("320193") == "0000320193"
    assert normalize_cik("0000320193") == "0000320193"


def _dera_row(cik, name, sic="2836", form="10-K", filed="20240115"):
    return {"cik": cik, "name": name, "sic": sic, "form": form, "filed": filed}


def test_build_universe_recent_filing_is_reporting_eligible():
    rows = [_dera_row("320193", "Acme Therapeutics Inc", filed="20240201")]
    candidates = build_universe(rows, {}, as_of=date(2024, 3, 1))
    assert len(candidates) == 1
    assert candidates[0].eligible_reporting is True


def test_build_universe_stale_filing_is_not_reporting_eligible():
    rows = [_dera_row("320193", "Acme Therapeutics Inc", filed="20200101")]
    candidates = build_universe(rows, {}, as_of=date(2024, 3, 1))
    assert candidates[0].eligible_reporting is False


def test_build_universe_no_10k_10q_filing_is_not_reporting_eligible():
    rows = [_dera_row("320193", "Acme Therapeutics Inc", form="8-K", filed="20240201")]
    candidates = build_universe(rows, {}, as_of=date(2024, 3, 1))
    assert candidates[0].eligible_reporting is False
    assert candidates[0].most_recent_filed is None


def test_build_universe_takes_latest_of_multiple_filings():
    rows = [
        _dera_row("320193", "Acme Therapeutics Inc", form="10-Q", filed="20230101"),
        _dera_row("320193", "Acme Therapeutics Inc", form="10-K", filed="20240201"),
    ]
    candidates = build_universe(rows, {}, as_of=date(2024, 3, 1))
    assert candidates[0].most_recent_filed == date(2024, 2, 1)
    assert candidates[0].most_recent_form == "10-K"


def test_build_universe_exchange_eligible_true_for_nasdaq():
    rows = [_dera_row("320193", "Acme Therapeutics Inc")]
    ticker_map = {"0000320193": ("ACME", "Nasdaq")}
    candidates = build_universe(rows, ticker_map, as_of=date(2024, 3, 1))
    assert candidates[0].eligible_exchange is True
    assert candidates[0].ticker == "ACME"


def test_build_universe_exchange_eligible_false_for_otc():
    rows = [_dera_row("320193", "Acme Therapeutics Inc")]
    ticker_map = {"0000320193": ("ACME", "OTC")}
    candidates = build_universe(rows, ticker_map, as_of=date(2024, 3, 1))
    assert candidates[0].eligible_exchange is False


def test_build_universe_exchange_unknown_is_none_not_false():
    """A CIK missing from the current-snapshot map (e.g. delisted) must be None, not False."""
    rows = [_dera_row("320193", "Acme Therapeutics Inc")]
    candidates = build_universe(rows, {}, as_of=date(2024, 3, 1))
    assert candidates[0].eligible_exchange is None


def test_build_universe_flags_spac_by_name():
    rows = [_dera_row("999", "Healthcare Acquisition Corp")]
    candidates = build_universe(rows, {}, as_of=date(2024, 3, 1))
    assert candidates[0].flagged_spac is True


def test_build_universe_size_and_asset_floor_not_computed():
    rows = [_dera_row("320193", "Acme Therapeutics Inc")]
    candidates = build_universe(rows, {}, as_of=date(2024, 3, 1))
    assert candidates[0].size_floor_met is None
    assert candidates[0].asset_floor_met is None


def test_write_universe_writes_non_spac_companies():
    repo = MagicMock()
    rows = [_dera_row("320193", "Acme Therapeutics Inc", filed="20240201")]
    candidates = build_universe(rows, {}, as_of=date(2024, 3, 1))

    stats = write_universe(candidates, repo)

    repo.upsert_company.assert_called_once()
    repo.add_review_item.assert_not_called()
    assert stats == {"companies_written": 1, "spac_flagged_for_review": 0, "total_candidates": 1}


def test_write_universe_routes_spac_flagged_to_review_queue_not_company_table():
    repo = MagicMock()
    rows = [_dera_row("999", "Healthcare Acquisition Corp")]
    candidates = build_universe(rows, {}, as_of=date(2024, 3, 1))

    stats = write_universe(candidates, repo)

    repo.upsert_company.assert_not_called()
    repo.add_review_item.assert_called_once()
    call_kwargs = repo.add_review_item.call_args.kwargs
    assert call_kwargs["item_type"] == "entity_match"
    payload = call_kwargs["payload"]
    assert payload["reason"] == "spac_name_heuristic"
    # Carries everything a later confirm (ingest/review_queue.confirm_item) needs to write
    # p22_company without re-deriving it from scratch.
    assert payload["cik"] == "0000000999"
    assert payload["name"] == "Healthcare Acquisition Corp"
    assert "ticker" in payload and "exchange" in payload and "eligible_reporting" in payload
    assert stats == {"companies_written": 0, "spac_flagged_for_review": 1, "total_candidates": 1}


def test_build_universe_history_recomputes_per_quarter_not_against_today():
    """A company reporting only in an early quarter should be eligible for that quarter's as_of,
    and ineligible (lookback expired) by a much later quarter — not judged against today."""
    quarters_rows = {
        "2019q1": [_dera_row("111", "Acme Therapeutics Inc", filed="20190115")],
    }
    history = build_universe_history(quarters_rows, {})

    assert list(history.keys()) == ["2019q1"]
    candidates = history["2019q1"]
    assert len(candidates) == 1
    # as_of = 2019-03-31, filed 2019-01-15 -> well within the 6-month lookback.
    assert candidates[0].eligible_reporting is True


def test_build_universe_history_uses_cumulative_union_across_quarters():
    """A company that only filed in Q1 must still appear (and be reporting-eligible) in Q2's
    universe, since Q2's as_of is only ~3 months after the Q1 filing — well within lookback."""
    quarters_rows = {
        "2024q1": [_dera_row("111", "Acme Therapeutics Inc", filed="20240115")],
        "2024q2": [_dera_row("222", "Beta Pharmaceuticals Corp", filed="20240415")],
    }
    history = build_universe_history(quarters_rows, {})

    q2_ciks = {c.cik for c in history["2024q2"]}
    assert "0000000111" in q2_ciks
    assert "0000000222" in q2_ciks
    # By Q2 (as_of 2024-06-30), Acme's Q1 filing (2024-01-15) is still within 183 days.
    acme = next(c for c in history["2024q2"] if c.cik == "0000000111")
    assert acme.eligible_reporting is True


def test_build_universe_history_reporting_eligibility_expires_in_a_later_quarter():
    """A company that stops filing should become reporting-ineligible once enough quarters
    pass for the trailing-6-month lookback to expire — proving this isn't computed once."""
    quarters_rows = {
        "2019q1": [_dera_row("111", "Acme Therapeutics Inc", filed="20190115")],
        "2020q1": [_dera_row("222", "Beta Pharmaceuticals Corp", filed="20200115")],
    }
    history = build_universe_history(quarters_rows, {})

    acme_2019 = next(c for c in history["2019q1"] if c.cik == "0000000111")
    acme_2020 = next(c for c in history["2020q1"] if c.cik == "0000000111")
    assert acme_2019.eligible_reporting is True
    assert acme_2020.eligible_reporting is False  # over a year since Acme's last filing


def test_build_universe_history_skips_malformed_quarter_keys():
    quarters_rows = {
        "not-a-quarter": [_dera_row("111", "Acme Therapeutics Inc")],
        "2024q1": [_dera_row("222", "Beta Pharmaceuticals Corp", filed="20240115")],
    }
    history = build_universe_history(quarters_rows, {})
    assert list(history.keys()) == ["2024q1"]
