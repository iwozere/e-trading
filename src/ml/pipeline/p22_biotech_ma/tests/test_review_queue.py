"""Tests for ingest/review_queue.py (spec §3.4). No live DB — repo is a MagicMock."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.review_queue import (
    UnknownReviewItemReasonError,
    confirm_item,
    queue_depth_report,
    reject_item,
)

_NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def test_confirm_spac_item_upserts_company_using_payload_fields():
    repo = MagicMock()
    repo.upsert_company.return_value = 42
    item = {
        "item_id": 1,
        "item_type": "entity_match",
        "payload": {
            "reason": "spac_name_heuristic",
            "cik": "0000000999",
            "name": "Actually A Real Biotech Inc",
            "sic": "2836",
            "ticker": "ABRC",
            "exchange": "Nasdaq",
            "eligible_reporting": True,
        },
    }

    outcome = confirm_item(item, repo, reviewed_by="alex")

    repo.upsert_company.assert_called_once_with(
        cik="0000000999",
        name="Actually A Real Biotech Inc",
        ticker="ABRC",
        exchange="Nasdaq",
        sic_code="2836",
        is_active=True,
        role="target",
    )
    repo.resolve_review_item.assert_called_once_with(item_id=1, status="confirmed", reviewed_by="alex", note=None)
    assert "42" in outcome


def test_confirm_fuzzy_alias_item_writes_alias_with_known_from_not_now():
    known_from = datetime(2024, 1, 15, tzinfo=timezone.utc)
    repo = MagicMock()
    item = {
        "item_id": 2,
        "item_type": "entity_match",
        "payload": {
            "reason": "fuzzy_alias_candidate",
            "candidate_name": "Acme Therapuetics Inc",
            "matched_company_id": 7,
            "matched_name": "Acme Therapeutics Inc",
            "score": 95.2,
            "source": "clinicaltrials",
            "known_from": known_from.isoformat(),
        },
    }

    confirm_item(item, repo, reviewed_by="alex", note="looks right")

    repo.add_company_alias.assert_called_once_with(
        company_id=7,
        alias="Acme Therapuetics Inc",
        source="clinicaltrials",
        is_verified=True,
        known_from=known_from,
    )
    repo.resolve_review_item.assert_called_once_with(
        item_id=2, status="confirmed", reviewed_by="alex", note="looks right"
    )


def test_confirm_unknown_reason_raises_and_does_not_resolve():
    repo = MagicMock()
    item = {"item_id": 3, "item_type": "entity_match", "payload": {"reason": "something_new"}}

    with pytest.raises(UnknownReviewItemReasonError):
        confirm_item(item, repo, reviewed_by="alex")

    repo.resolve_review_item.assert_not_called()


def test_reject_item_only_updates_status_no_downstream_write():
    repo = MagicMock()
    reject_item(5, repo, reviewed_by="alex", note="not a match")

    repo.resolve_review_item.assert_called_once_with(item_id=5, status="rejected", reviewed_by="alex", note="not a match")
    repo.add_company_alias.assert_not_called()
    repo.upsert_company.assert_not_called()


def test_queue_depth_report_groups_by_item_type_and_computes_age():
    items = [
        {"item_id": 1, "item_type": "entity_match", "created_at": _NOW - timedelta(hours=10)},
        {"item_id": 2, "item_type": "entity_match", "created_at": _NOW - timedelta(hours=2)},
        {"item_id": 3, "item_type": "process_event", "created_at": _NOW - timedelta(hours=30)},
    ]

    report = queue_depth_report(items, now=_NOW)

    assert report["entity_match"]["count"] == 2
    assert report["entity_match"]["oldest_age_hours"] == pytest.approx(10.0)
    assert report["process_event"]["count"] == 1
    assert report["process_event"]["oldest_age_hours"] == pytest.approx(30.0)


def test_queue_depth_report_handles_missing_created_at():
    items = [{"item_id": 1, "item_type": "entity_match", "created_at": None}]
    report = queue_depth_report(items, now=_NOW)
    assert report["entity_match"]["count"] == 1
    assert report["entity_match"]["median_age_hours"] is None
    assert report["entity_match"]["oldest_age_hours"] is None


def test_queue_depth_report_empty_input():
    assert queue_depth_report([], now=_NOW) == {}
