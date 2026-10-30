"""Tests for ingest/deal_entry.py (spec §2.5, M6). Fake repo, no live DB."""

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.deal_entry import (
    DealEntry,
    DealEntryError,
    DuplicateDealError,
    ReviewFlagsNotAcknowledgedError,
    entry_from_mapping,
    evaluate_flags,
    find_link_target,
    gather_context,
    link_candidate,
    parse_fraction,
    record_deal,
)
from src.ml.pipeline.p22_biotech_ma.ingest.review_queue import UnknownReviewItemReasonError, confirm_item

_ANNOUNCED = date(2024, 1, 8)
_TARGET = 7


class _FakeRepo:
    def __init__(
        self,
        closes: Optional[Dict[date, float]] = None,
        splits: Optional[List[Dict[str, Any]]] = None,
        shares: Optional[float] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
    ):
        self.closes = closes or {}
        self.splits = splits or []
        self.shares = shares
        self.actions = actions or []
        self.deals: Dict[int, Dict[str, Any]] = {}
        self.resolved: List[Dict[str, Any]] = []
        self.companies = {"0000310158": {"company_id": 99}}

    def get_latest_raw_close_as_of(self, company_id: int, as_of: date, vendor: Optional[str] = None):
        del company_id, vendor
        eligible = [d for d in self.closes if d <= as_of]
        if not eligible:
            return None
        trade_date = max(eligible)
        return {"trade_date": trade_date, "close_raw": self.closes[trade_date], "vendor": "fmp"}

    def get_splits_as_of(self, company_id: int, as_of: date):
        del company_id
        return [s for s in self.splits if s["ex_date"] <= as_of]

    def get_financial_facts_as_of(self, company_id: int, metric: str, as_of: date):
        del company_id, as_of
        if metric == "shares_outstanding" and self.shares is not None:
            return [{"value": self.shares, "period_end": _ANNOUNCED - timedelta(days=60)}]
        return []

    def get_corporate_actions_between(self, company_id: int, start: date, end: date):
        del company_id
        return [a for a in self.actions if start <= a["ex_date"] <= end]

    def get_company_by_cik(self, cik: str):
        return self.companies.get(cik)

    def find_deal(self, target_id: int, announcement_date: date):
        return next(
            (d for d in self.deals.values() if d["target_id"] == target_id and d["announcement_date"] == announcement_date),
            None,
        )

    def get_deal(self, deal_id: int):
        return self.deals.get(deal_id)

    def list_deals(self, target_id: Optional[int] = None):
        deals = [d for d in self.deals.values() if target_id is None or d["target_id"] == target_id]
        return sorted(deals, key=lambda d: d["announcement_date"])

    def insert_deal(self, **fields: Any) -> int:
        deal_id = len(self.deals) + 1
        self.deals[deal_id] = {"deal_id": deal_id, **fields}
        return deal_id

    def resolve_review_item(self, **kwargs: Any) -> None:
        self.resolved.append(kwargs)


def _item(item_id: int = 1, status: str = "pending", filed: str = "2024-01-10") -> Dict[str, Any]:
    return {
        "item_id": item_id,
        "item_type": "deal_candidate",
        "status": status,
        "payload": {"reason": "deal_candidate", "company_id": _TARGET, "accession_no": f"0001-24-00000{item_id}",
                    "filed_date": filed},
    }


def _entry(**overrides: Any) -> DealEntry:
    fields: Dict[str, Any] = {
        "announcement_date": _ANNOUNCED,
        "deal_type": "strategic_acquisition",
        "acquirer_name": "Merck & Co Inc",
        "acquirer_cik": "0000310158",
        "upfront_per_share": 115.0,
        "status": "announced",
    }
    fields.update(overrides)
    return DealEntry(**fields)


# ---------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------

def test_entry_from_mapping_parses_all_field_types():
    entry = entry_from_mapping({
        "announcement_date": "2024-01-08", "deal_type": "strategic_acquisition", "acquirer_name": "Merck",
        "acquirer_cik": "310158", "upfront_per_share": "$1,115.50", "has_cvr": "yes", "cvr_max_per_share": "4",
        "stock_consideration_pct": "45%", "target_holders_pct": "0.1", "status": "completed",
        "completion_date": "2024-03-01", "acquirer_is_private": "", "note": "  ",
    })
    assert entry.acquirer_cik == "0000310158"
    assert entry.upfront_per_share == 1115.5
    assert entry.has_cvr is True
    assert entry.stock_consideration_pct == pytest.approx(0.45)
    assert entry.acquirer_is_private is None
    assert entry.note is None


def test_entry_from_mapping_requires_core_fields():
    with pytest.raises(DealEntryError, match="required"):
        entry_from_mapping({"announcement_date": "2024-01-08", "deal_type": "strategic_acquisition"})


def test_parse_fraction_rejects_bare_number_above_one():
    with pytest.raises(DealEntryError, match="%"):
        parse_fraction("45")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"deal_type": "merger"}, "deal_type"),
        ({"status": "pending"}, "status"),
        ({"completion_date": _ANNOUNCED - timedelta(days=1), "status": "completed"}, "before announcement"),
        ({"completion_date": _ANNOUNCED, "status": "terminated"}, "completion_date given"),
        ({"has_cvr": False, "cvr_max_per_share": 2.0}, "has_cvr is no"),
        ({"upfront_per_share": 0.0}, "positive"),
    ],
)
def test_entry_validation_rejects_inconsistent_fields(overrides, message):
    with pytest.raises(DealEntryError, match=message):
        entry_from_mapping({
            "announcement_date": str(overrides.pop("announcement_date", _ANNOUNCED)),
            "deal_type": overrides.pop("deal_type", "strategic_acquisition"),
            "acquirer_name": "X",
            **{k: ("yes" if v is True else "no" if v is False else str(v)) for k, v in overrides.items()},
        })


# ---------------------------------------------------------------------
# Heuristics and premiums
# ---------------------------------------------------------------------

def test_evaluate_flags_fires_each_heuristic():
    entry = _entry(acquirer_is_private=True, acquirer_has_approved_product=False,
                   stock_consideration_pct=0.9, target_holders_pct=0.1)
    fired, unchecked = evaluate_flags(entry, market_cap=30e6, reverse_splits=[{"ex_date": _ANNOUNCED, "ratio": 0.1}])
    assert set(fired) == {"acquirer_private_no_product", "target_market_cap_lt_50m",
                          "majority_stock_low_retention", "reverse_split_within_90d"}
    assert unchecked == ["pipeline_discontinuation_12m"]


def test_evaluate_flags_reports_missing_inputs_as_unchecked():
    fired, unchecked = evaluate_flags(_entry(), market_cap=None, reverse_splits=[])
    assert fired == []
    assert {"acquirer_private_no_product", "target_market_cap_lt_50m", "majority_stock_low_retention"} <= set(unchecked)


def test_gather_context_computes_premiums_on_announcement_share_basis():
    # 2:1 forward split between the 30-day-ago close and announcement: the old $200 close is $100 on today's basis.
    repo = _FakeRepo(
        closes={_ANNOUNCED - timedelta(days=31): 200.0, _ANNOUNCED - timedelta(days=3): 100.0},
        splits=[{"ex_date": _ANNOUNCED - timedelta(days=10), "ratio": 2.0}],
    )
    context = gather_context(repo, _TARGET, _entry(upfront_per_share=125.0))
    assert context.premium_1d == pytest.approx(0.25)
    assert context.premium_30d == pytest.approx(0.25)
    assert context.close_30d is not None and context.close_30d["split_factor"] == 2.0


def test_gather_context_ignores_stale_prices():
    repo = _FakeRepo(closes={_ANNOUNCED - timedelta(days=200): 10.0})
    context = gather_context(repo, _TARGET, _entry())
    assert context.premium_1d is None and context.prior_close is None


def test_gather_context_flags_reverse_split_and_small_market_cap():
    repo = _FakeRepo(
        closes={_ANNOUNCED - timedelta(days=3): 2.0},
        shares=10e6,
        actions=[{"ex_date": _ANNOUNCED - timedelta(days=30), "action_type": "reverse_split", "ratio": 0.05}],
    )
    context = gather_context(repo, _TARGET, _entry())
    assert "reverse_split_within_90d" in context.flags_fired
    assert "target_market_cap_lt_50m" in context.flags_fired


# ---------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------

def test_record_deal_writes_row_and_confirms_item():
    repo = _FakeRepo(closes={_ANNOUNCED - timedelta(days=1): 100.0})
    result = record_deal(repo, _item(), _entry(), entered_by="ak")
    deal = repo.deals[result["deal_id"]]
    assert deal["target_id"] == _TARGET
    assert deal["acquirer_id"] == 99
    assert deal["premium_1d"] == pytest.approx(0.15)
    assert deal["review_item_id"] == 1
    assert deal["review_flags"]["fired"] == []
    assert repo.resolved == [{"item_id": 1, "status": "confirmed", "reviewed_by": "ak", "note": f"deal_id={result['deal_id']}"}]


def test_record_deal_keeps_unmatched_acquirer_by_name_only():
    repo = _FakeRepo()
    result = record_deal(repo, _item(), _entry(acquirer_cik=None, acquirer_name="Private Bio Inc"), entered_by="ak")
    deal = repo.deals[result["deal_id"]]
    assert deal["acquirer_id"] is None
    assert deal["acquirer_name"] == "Private Bio Inc"


def test_record_deal_blocks_flagged_strategic_acquisition_until_acknowledged():
    repo = _FakeRepo(actions=[{"ex_date": _ANNOUNCED - timedelta(days=5), "action_type": "reverse_split", "ratio": 0.1}])
    with pytest.raises(ReviewFlagsNotAcknowledgedError):
        record_deal(repo, _item(), _entry(), entered_by="ak")
    assert repo.deals == {}

    result = record_deal(repo, _item(), _entry(), entered_by="ak", acknowledge_flags=True)
    assert repo.deals[result["deal_id"]]["review_flags"]["fired"] == ["reverse_split_within_90d"]


def test_record_deal_allows_flags_on_excluded_deal_type():
    repo = _FakeRepo(actions=[{"ex_date": _ANNOUNCED - timedelta(days=5), "action_type": "reverse_split", "ratio": 0.1}])
    result = record_deal(repo, _item(), _entry(deal_type="reverse_merger"), entered_by="ak")
    assert repo.deals[result["deal_id"]]["deal_type"] == "reverse_merger"


def test_record_deal_refuses_duplicate_and_resolved_items():
    repo = _FakeRepo()
    record_deal(repo, _item(1), _entry(), entered_by="ak")
    with pytest.raises(DuplicateDealError):
        record_deal(repo, _item(2), _entry(), entered_by="ak")
    with pytest.raises(DealEntryError, match="already confirmed"):
        record_deal(repo, _item(3, status="confirmed"), _entry(), entered_by="ak")


def test_link_candidate_picks_latest_deal_announced_before_filing():
    repo = _FakeRepo()
    record_deal(repo, _item(1), _entry(announcement_date=date(2020, 5, 1)), entered_by="ak")
    second = record_deal(repo, _item(2), _entry(), entered_by="ak")["deal_id"]

    latest = find_link_target(repo, _item(3, filed="2024-02-01"))
    earlier = find_link_target(repo, _item(3, filed="2020-06-01"))
    assert latest is not None and latest["deal_id"] == second
    assert earlier is not None and earlier["deal_id"] == 1
    assert link_candidate(repo, _item(3, filed="2024-02-01"), None, reviewed_by="ak") == second
    assert repo.resolved[-1]["note"] == f"filing of deal_id={second}"


def test_link_candidate_without_any_deal_raises():
    with pytest.raises(DealEntryError, match="enter the deal first"):
        link_candidate(_FakeRepo(), _item(), None, reviewed_by="ak")


def test_review_queue_confirm_points_deal_candidates_at_deal_entry_cli():
    with pytest.raises(UnknownReviewItemReasonError, match="deal_entry_cli"):
        confirm_item(_item(), _FakeRepo(), reviewed_by="ak")
