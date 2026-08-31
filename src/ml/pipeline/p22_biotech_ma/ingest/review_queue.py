"""
P22 — review-queue confirmation logic (spec §3.4).

`entity_resolution.write_universe` and `alias_matching.resolve_aliases` are
the *producers* of `p22_review_item` rows; this module is the *consumer* —
turning a human's confirm/reject decision into the actual downstream write
(or into a plain status update, for a reject). Kept separate from the CLI
(`cli/review_queue_cli.py`) so the dispatch logic is unit-testable without an
interactive terminal or a real DB.

Confirmation dispatch is keyed on `payload['reason']`, matching the two
producers that exist so far:

- `'spac_name_heuristic'` (from `entity_resolution.write_universe`) — confirm
  writes the candidate to `p22_company` via `upsert_company`, using the
  ticker/exchange/reporting-eligibility already computed at candidate-build
  time (carried in the payload) rather than re-deriving it.
- `'fuzzy_alias_candidate'` (from `alias_matching.resolve_aliases`) — confirm
  writes `p22_company_alias` via `add_company_alias`, using
  `payload['known_from']` (the raw-zone ingestion timestamp of the source
  payload the candidate name came from), **never** the review timestamp —
  spec §3.4: "confirmation writes back with `known_from` set to the
  underlying filing date, not the review date."

A reject never writes anything beyond the review item's own status — the
candidate is simply dropped, which is the whole point of the queue existing
(spec §3.3: "never auto-accepted").
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_KNOWN_REASONS = frozenset({"spac_name_heuristic", "fuzzy_alias_candidate"})


class UnknownReviewItemReasonError(ValueError):
    """Raised when a review item's `payload['reason']` has no registered confirm handler."""


def confirm_item(item: Dict[str, Any], repo: Any, *, reviewed_by: str, note: str | None = None) -> str:
    """
    Apply a human's "yes, this is correct" decision: perform the downstream
    write the item's `reason` calls for, then mark the item confirmed.

    Args:
        item: A row from `P22Repo.get_pending_review_items`/`get_review_item`
            (has `item_id`, `item_type`, `payload`, ...).
        repo: A `P22Repo`-shaped object.
        reviewed_by: Human identifier for the audit trail.
        note: Optional free-text reviewer note.

    Returns:
        A short human-readable description of what was written.

    Raises:
        UnknownReviewItemReasonError: if `payload['reason']` isn't one of the
            reasons this module knows how to confirm — surfaced rather than
            silently marking the item confirmed with no write, which would
            quietly lose the candidate.
    """
    payload = item["payload"]
    reason = payload.get("reason")

    if reason == "spac_name_heuristic":
        company_id = repo.upsert_company(
            cik=payload["cik"],
            name=payload["name"],
            ticker=payload.get("ticker"),
            exchange=payload.get("exchange"),
            sic_code=payload.get("sic"),
            is_active=payload.get("eligible_reporting"),
            role="target",
        )
        outcome = f"upserted company_id={company_id} (cik={payload['cik']}, name={payload['name']!r})"
    elif reason == "fuzzy_alias_candidate":
        known_from = datetime.fromisoformat(payload["known_from"])
        repo.add_company_alias(
            company_id=payload["matched_company_id"],
            alias=payload["candidate_name"],
            source=payload["source"],
            is_verified=True,
            known_from=known_from,
        )
        outcome = (
            f"added alias {payload['candidate_name']!r} -> company_id={payload['matched_company_id']} "
            f"(known_from={known_from.isoformat()})"
        )
    else:
        raise UnknownReviewItemReasonError(
            f"No confirm handler for reason={reason!r} (item_id={item['item_id']}, "
            f"item_type={item['item_type']}). Known reasons: {sorted(_KNOWN_REASONS)}"
        )

    repo.resolve_review_item(item_id=item["item_id"], status="confirmed", reviewed_by=reviewed_by, note=note)
    _logger.info("Confirmed review item %s: %s", item["item_id"], outcome)
    return outcome


def reject_item(item_id: int, repo: Any, *, reviewed_by: str, note: str | None = None) -> None:
    """Mark a review item rejected. No downstream write — the candidate is dropped."""
    repo.resolve_review_item(item_id=item_id, status="rejected", reviewed_by=reviewed_by, note=note)
    _logger.info("Rejected review item %s", item_id)


def queue_depth_report(pending_items: list[Dict[str, Any]], *, now: datetime) -> Dict[str, Dict[str, Any]]:
    """
    Queue depth and median age by `item_type` (spec §3.4: "Queue depth and
    median age by `item_type` are reported in every run").

    Args:
        pending_items: Rows from `P22Repo.get_pending_review_items()` (needs
            `created_at`, added in migration 005 — see that file's docstring
            for why the spec's own §3.4 schema sketch didn't have it).
        now: Current time (injected, not `datetime.now()`, for deterministic
            tests and so a caller can pin the report to a job's start time).

    Returns:
        `{item_type: {"count": int, "median_age_hours": float, "oldest_age_hours": float}}`.
        An item with no `created_at` (a pre-migration row, or a test double
        that omits it) is excluded from the age stats but still counted.
    """
    by_type: Dict[str, list[Dict[str, Any]]] = {}
    for item in pending_items:
        by_type.setdefault(item["item_type"], []).append(item)

    report: Dict[str, Dict[str, Any]] = {}
    for item_type, items in by_type.items():
        ages_hours = sorted(
            (now - i["created_at"]).total_seconds() / 3600.0 for i in items if i.get("created_at") is not None
        )
        report[item_type] = {
            "count": len(items),
            "median_age_hours": ages_hours[len(ages_hours) // 2] if ages_hours else None,
            "oldest_age_hours": ages_hours[-1] if ages_hours else None,
        }

    return report
