"""Tests for ingest/size_floor.py and ingest/price_quality.py against an in-memory repo with as-of semantics."""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.cli.size_floor_report_cli import is_suspect
from src.ml.pipeline.p22_biotech_ma.ingest.entity_resolution import build_universe_history
from src.ml.pipeline.p22_biotech_ma.ingest.price_quality import (
    float_consistency_history,
    latest_public_float,
    summarize_consistency,
)
from src.ml.pipeline.p22_biotech_ma.ingest.size_floor import evaluate_size_floor, make_size_floor_lookup
from src.ml.pipeline.p22_biotech_ma.tests.fake_market_repo import FakeMarketRepo

_CID = 7
_FLOAT_DATE = date(2025, 6, 30)
_AS_OF = date(2026, 3, 31)


def _repo(*, float_price: float, as_of_price: float, public_float: float, shares: float = 10_000_000.0) -> FakeMarketRepo:
    """One company: share counts before both dates, a bar on the float date and on `_AS_OF`, a 10-K float."""
    repo = FakeMarketRepo()
    repo.add_fact(_CID, "shares_outstanding", shares, period_end=date(2025, 5, 1), filed=date(2025, 5, 10))
    repo.add_fact(_CID, "shares_outstanding", shares, period_end=date(2026, 2, 1), filed=date(2026, 2, 20))
    repo.add_price(_CID, _FLOAT_DATE, float_price)
    repo.add_price(_CID, _AS_OF, as_of_price)
    repo.add_fact(_CID, "public_float", public_float, period_end=_FLOAT_DATE, filed=date(2026, 3, 1))
    return repo


def test_size_floor_uses_market_cap_when_consistent_with_float():
    repo = _repo(float_price=5.0, as_of_price=2.0, public_float=40_000_000.0)  # 50M vs 40M float: ratio 1.25

    decision = evaluate_size_floor(repo, _CID, _AS_OF)

    assert decision.met is False  # 2.0 x 10M = 20M today, below $25M despite the earlier float
    assert decision.basis == "market_cap"
    assert decision.market_cap == 20_000_000.0


def test_size_floor_falls_back_to_float_when_price_series_is_corrupt():
    """ACOR-style: vintage price ~1000x too low on every date; the float alone proves the floor."""
    repo = _repo(float_price=0.005, as_of_price=0.004, public_float=700_000_000.0)

    decision = evaluate_size_floor(repo, _CID, _AS_OF)

    assert decision.met is True
    assert decision.basis == "public_float"


def test_size_floor_unknown_when_inconsistent_and_float_below_floor():
    """ADR-style: share count far larger than the listed security; small float cannot decide either way."""
    repo = _repo(float_price=5.0, as_of_price=5.0, public_float=2_000_000.0)  # ratio 25

    decision = evaluate_size_floor(repo, _CID, _AS_OF)

    assert decision.met is None
    assert decision.basis == "inconsistent_with_public_float"


def test_size_floor_market_cap_decides_without_any_float():
    repo = FakeMarketRepo()
    repo.add_fact(_CID, "shares_outstanding", 10_000_000.0, period_end=date(2026, 2, 1), filed=date(2026, 2, 20))
    repo.add_price(_CID, _AS_OF, 3.0)

    decision = evaluate_size_floor(repo, _CID, _AS_OF)

    assert decision.met is True
    assert decision.basis == "market_cap"


def test_size_floor_unknown_without_price_reports_reason():
    repo = FakeMarketRepo()
    repo.add_fact(_CID, "shares_outstanding", 10_000_000.0, period_end=date(2026, 2, 1), filed=date(2026, 2, 20))

    decision = evaluate_size_floor(repo, _CID, _AS_OF)

    assert decision.met is None
    assert decision.basis == "no_price"


def test_public_float_not_visible_before_10k_filed():
    repo = _repo(float_price=5.0, as_of_price=5.0, public_float=40_000_000.0)

    assert latest_public_float(repo, _CID, date(2026, 2, 28)) is None
    found = latest_public_float(repo, _CID, _AS_OF)
    assert found is not None and found.value == 40_000_000.0


def test_public_float_ignored_when_too_old():
    repo = _repo(float_price=5.0, as_of_price=5.0, public_float=40_000_000.0)

    assert latest_public_float(repo, _CID, date(2027, 6, 30)) is None


def test_float_consistency_history_and_summary():
    repo = _repo(float_price=0.005, as_of_price=0.004, public_float=700_000_000.0)
    repo.add_price(_CID, date(2024, 6, 28), 0.006)
    repo.add_fact(_CID, "shares_outstanding", 10_000_000.0, period_end=date(2024, 5, 1), filed=date(2024, 5, 10))
    repo.add_fact(_CID, "public_float", 600_000_000.0, period_end=date(2024, 6, 30), filed=date(2025, 3, 1))

    checks = float_consistency_history(repo, _CID, _AS_OF)
    summary = summarize_consistency(checks)

    assert [c.public_float.period_end for c in checks] == [_FLOAT_DATE, date(2024, 6, 30)]
    assert summary["checked"] == 2 and summary["inconsistent"] == 2 and summary["too_low"] == 2
    assert is_suspect(summary)
    assert not is_suspect({"checked": 3, "inconsistent": 1})


def test_build_universe_history_fills_size_floor_point_in_time():
    repo = _repo(float_price=5.0, as_of_price=2.0, public_float=40_000_000.0)
    repo.companies["0000000111"] = {"company_id": _CID}
    quarters_rows = {
        "2026q1": [
            {"cik": "111", "name": "Acme Therapeutics Inc", "sic": "2836", "form": "10-K", "filed": "20260301"},
            {"cik": "222", "name": "Unknown Bio Inc", "sic": "2836", "form": "10-K", "filed": "20260301"},
        ],
    }

    history = build_universe_history(quarters_rows, {}, size_floor=make_size_floor_lookup(repo))

    by_cik = {c.cik: c for c in history["2026q1"]}
    assert by_cik["0000000111"].size_floor_met is False  # 2.0 x 10M on 2026-03-31
    assert by_cik["0000000222"].size_floor_met is None  # no p22_company row
