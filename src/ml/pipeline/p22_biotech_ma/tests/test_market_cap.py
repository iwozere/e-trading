"""Tests for ingest/market_cap.py. No live DB — repo is a MagicMock/fake."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.market_cap import MarketCapResult, compute_market_cap, run

_AS_OF = date(2026, 9, 5)


def _repo(price=None, shares_facts=None):
    repo = MagicMock()
    repo.get_latest_raw_close_as_of.return_value = price
    repo.get_financial_facts_as_of.return_value = shares_facts or []
    return repo


def test_compute_market_cap_multiplies_raw_close_by_shares_outstanding():
    repo = _repo(
        price={"trade_date": date(2026, 9, 4), "close_raw": 20.0, "vendor": "yfinance"},
        shares_facts=[{"value": 1_000_000.0}],
    )

    result, reason = compute_market_cap(repo, company_id=7, as_of=_AS_OF)

    assert reason is None
    assert result == MarketCapResult(
        company_id=7, market_cap=20_000_000.0, price_trade_date=date(2026, 9, 4), shares_outstanding=1_000_000.0,
    )


def test_compute_market_cap_none_when_no_price():
    repo = _repo(price=None, shares_facts=[{"value": 1_000_000.0}])

    result, reason = compute_market_cap(repo, company_id=7, as_of=_AS_OF)

    assert result is None
    assert reason == "no_price"


def test_compute_market_cap_none_when_no_shares_outstanding():
    repo = _repo(price={"trade_date": date(2026, 9, 4), "close_raw": 20.0, "vendor": "yfinance"}, shares_facts=[])

    result, reason = compute_market_cap(repo, company_id=7, as_of=_AS_OF)

    assert result is None
    assert reason == "no_shares_outstanding"


def test_compute_market_cap_none_when_shares_value_is_none():
    repo = _repo(
        price={"trade_date": date(2026, 9, 4), "close_raw": 20.0, "vendor": "yfinance"},
        shares_facts=[{"value": None}],
    )

    result, reason = compute_market_cap(repo, company_id=7, as_of=_AS_OF)

    assert result is None
    assert reason == "no_shares_outstanding"


def test_compute_market_cap_none_when_shares_non_positive():
    repo = _repo(
        price={"trade_date": date(2026, 9, 4), "close_raw": 20.0, "vendor": "yfinance"},
        shares_facts=[{"value": 0.0}],
    )

    result, reason = compute_market_cap(repo, company_id=7, as_of=_AS_OF)

    assert result is None
    assert reason == "non_positive_shares_outstanding"


def test_run_writes_market_cap_fact_for_each_computable_company():
    repo = MagicMock()
    repo.get_latest_raw_close_as_of.return_value = {
        "trade_date": date(2026, 9, 4), "close_raw": 10.0, "vendor": "yfinance",
    }
    repo.get_financial_facts_as_of.return_value = [{"value": 500_000.0}]

    summary = run(repo, company_ids=[1, 2], as_of=_AS_OF)

    assert summary["computed"] == 2
    assert summary["skipped"] == 0
    assert repo.upsert_financial_fact_bitemporal.call_count == 2
    _, kwargs = repo.upsert_financial_fact_bitemporal.call_args
    assert kwargs["metric"] == "market_cap"
    assert kwargs["value"] == 5_000_000.0
    assert kwargs["period_end"] == date(2026, 9, 4)


def test_run_reports_rejection_breakdown_when_every_company_fails():
    repo = MagicMock()
    repo.get_latest_raw_close_as_of.return_value = None
    repo.get_financial_facts_as_of.return_value = []

    summary = run(repo, company_ids=[1, 2, 3], as_of=_AS_OF)

    assert summary["computed"] == 0
    assert summary["skipped"] == 3
    assert summary["rejection_breakdown"] == {"no_price": 3}
    repo.upsert_financial_fact_bitemporal.assert_not_called()


def test_run_mixed_pass_and_fail():
    repo = MagicMock()

    def fake_price(company_id, as_of):
        del as_of
        return {"trade_date": date(2026, 9, 4), "close_raw": 10.0, "vendor": "yfinance"} if company_id == 1 else None

    repo.get_latest_raw_close_as_of.side_effect = fake_price
    repo.get_financial_facts_as_of.return_value = [{"value": 100.0}]

    summary = run(repo, company_ids=[1, 2], as_of=_AS_OF)

    assert summary["computed"] == 1
    assert summary["skipped"] == 1
    assert summary["rejection_breakdown"] == {"no_price": 1}
