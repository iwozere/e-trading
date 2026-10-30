"""
P22 — hand-verified deal labels (spec §2.5, M6), 2026-10-29.

`deal_candidates.py` queues one `deal_candidate` review item per (company, SC 14D9/DEFM14A/S-4
filing). This module turns a human's reading of that filing into a `p22_deal` row, or links a
further filing of an already-entered deal, or rejects the candidate. The CLI
(`cli/deal_entry_cli.py`) is thin glue over these functions.

What the human supplies (`DealEntry`): announcement date, `deal_type`, acquirer, upfront
consideration per share, CVR terms, status/completion date, and the consideration mix. What this
module derives instead of asking for it:

- `premium_1d` / `premium_30d` — upfront per share over the raw close of the last trading day before
  announcement / on or before 30 calendar days before it, both rescaled for any split between the
  price date and the announcement (`market_cap.split_factor_between`), and rejected if the close is
  older than `MARKET_CAP_MAX_PRICE_AGE_DAYS` (a halted or delisted series keeps its last bar forever).
- the spec's reverse-merger detection heuristics, as `review_flags` (spec §2.5: "flag, don't
  auto-exclude"):
    * `acquirer_private_no_product` — acquirer privately held and has no approved product (hand input),
    * `target_market_cap_lt_50m` — point-in-time market cap the day before announcement
      (`market_cap.compute_market_cap`),
    * `majority_stock_low_retention` — more than half stock and target holders keep < 20% (hand input),
    * `reverse_split_within_90d` — a reverse split with ex-date in the 90 days up to announcement.
  The fifth heuristic ("pipeline discontinuation announced within 12 months") has no data source in
  this pipeline and is always reported as unchecked.

Entering a `strategic_acquisition` (the only positive label) while any flag fires requires an explicit
acknowledgement — those flags describe exactly the reverse mergers the spec says "look like
acquisitions in the filings."
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import MARKET_CAP_MAX_PRICE_AGE_DAYS
from src.ml.pipeline.p22_biotech_ma.ingest.market_cap import compute_market_cap, split_factor_between
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# spec §2.5's `deal_type` table — only `strategic_acquisition` is a positive label.
DEAL_TYPES = (
    "strategic_acquisition", "reverse_merger", "shell_transaction", "liquidation", "asset_sale", "pe_take_private",
)
POSITIVE_DEAL_TYPE = "strategic_acquisition"
DEAL_STATUSES = ("announced", "completed", "terminated")

REVERSE_MERGER_MARKET_CAP_USD = 50_000_000.0
REVERSE_SPLIT_LOOKBACK_DAYS = 90
PREMIUM_30D_LOOKBACK_DAYS = 30
UNCHECKABLE_HEURISTICS = ("pipeline_discontinuation_12m",)


class DealEntryError(ValueError):
    """The hand-entered fields are invalid, or the candidate cannot take this action."""


class ReviewFlagsNotAcknowledgedError(DealEntryError):
    """A `strategic_acquisition` was entered while reverse-merger heuristics fired, without acknowledgement."""


class DuplicateDealError(DealEntryError):
    """A deal already exists for this target and announcement date — link the candidate to it instead."""


@dataclass(frozen=True)
class DealEntry:
    """The fields a human reads off the filing. Percentages are fractions in [0, 1]."""

    announcement_date: date
    deal_type: str
    acquirer_name: str
    acquirer_cik: Optional[str] = None
    status: Optional[str] = None
    completion_date: Optional[date] = None
    upfront_per_share: Optional[float] = None
    has_cvr: Optional[bool] = None
    cvr_max_per_share: Optional[float] = None
    stock_consideration_pct: Optional[float] = None
    target_holders_pct: Optional[float] = None
    acquirer_is_private: Optional[bool] = None
    acquirer_has_approved_product: Optional[bool] = None
    note: Optional[str] = None


@dataclass(frozen=True)
class DealContext:
    """Everything derived from the database for one entry, shown to the reviewer before writing."""

    premium_1d: Optional[float]
    premium_30d: Optional[float]
    prior_close: Optional[Dict[str, Any]]
    close_30d: Optional[Dict[str, Any]]
    market_cap: Optional[float]
    market_cap_reason: Optional[str]
    reverse_splits: List[Dict[str, Any]]
    flags_fired: List[str] = field(default_factory=list)
    flags_unchecked: List[str] = field(default_factory=list)

    def review_flags_json(self) -> Dict[str, Any]:
        """The `p22_deal.review_flags` payload."""
        return {
            "fired": self.flags_fired,
            "unchecked": self.flags_unchecked,
            "market_cap_at_announcement": self.market_cap,
            "market_cap_reason": self.market_cap_reason,
            "reverse_splits_90d": [
                {"ex_date": s["ex_date"].isoformat(), "ratio": s["ratio"]} for s in self.reverse_splits
            ],
        }


# ---------------------------------------------------------------------------
# Parsing (shared by CLI flags and CSV import)
# ---------------------------------------------------------------------------


def parse_bool(value: Optional[str]) -> Optional[bool]:
    """`y/yes/true/1` and `n/no/false/0` (any case); blank is None."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip().lower()
    if text in ("y", "yes", "true", "1"):
        return True
    if text in ("n", "no", "false", "0"):
        return False
    raise DealEntryError(f"Not a yes/no value: {value!r}")


def parse_float(value: Optional[str]) -> Optional[float]:
    """A plain number (`$` and `,` allowed); blank is None."""
    if value is None or not str(value).strip():
        return None
    try:
        return float(str(value).strip().replace("$", "").replace(",", ""))
    except ValueError as exc:
        raise DealEntryError(f"Not a number: {value!r}") from exc


def parse_fraction(value: Optional[str]) -> Optional[float]:
    """A fraction in [0, 1], or a percentage written with `%` (`45%` -> 0.45); blank is None.

    A bare number above 1 is rejected rather than guessed to be a percentage."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    is_percent = text.endswith("%")
    number = parse_float(text.rstrip("%"))
    assert number is not None
    if is_percent:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        raise DealEntryError(f"Fraction out of [0, 1] (write percentages with '%'): {value!r}")
    return number


def parse_date(value: Optional[str]) -> Optional[date]:
    """ISO `YYYY-MM-DD`; blank is None."""
    if value is None or not str(value).strip():
        return None
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise DealEntryError(f"Not an ISO date (YYYY-MM-DD): {value!r}") from exc


def _blank_to_none(value: Optional[str]) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def entry_from_mapping(fields: Mapping[str, Optional[str]]) -> DealEntry:
    """
    Build a `DealEntry` from string fields (a CSV row or CLI arguments) and validate it.

    Raises:
        DealEntryError: on a missing required field or any invalid value.
    """
    announcement_date = parse_date(fields.get("announcement_date"))
    deal_type = _blank_to_none(fields.get("deal_type"))
    acquirer_name = _blank_to_none(fields.get("acquirer_name"))
    if announcement_date is None or deal_type is None or acquirer_name is None:
        raise DealEntryError("announcement_date, deal_type and acquirer_name are required")
    acquirer_cik = _blank_to_none(fields.get("acquirer_cik"))
    entry = DealEntry(
        announcement_date=announcement_date,
        deal_type=deal_type,
        acquirer_name=acquirer_name,
        acquirer_cik=acquirer_cik.zfill(10) if acquirer_cik else None,
        status=_blank_to_none(fields.get("status")),
        completion_date=parse_date(fields.get("completion_date")),
        upfront_per_share=parse_float(fields.get("upfront_per_share")),
        has_cvr=parse_bool(fields.get("has_cvr")),
        cvr_max_per_share=parse_float(fields.get("cvr_max_per_share")),
        stock_consideration_pct=parse_fraction(fields.get("stock_consideration_pct")),
        target_holders_pct=parse_fraction(fields.get("target_holders_pct")),
        acquirer_is_private=parse_bool(fields.get("acquirer_is_private")),
        acquirer_has_approved_product=parse_bool(fields.get("acquirer_has_approved_product")),
        note=_blank_to_none(fields.get("note")),
    )
    validate_entry(entry)
    return entry


def validate_entry(entry: DealEntry) -> None:
    """
    Check the hand-entered fields for internal consistency.

    Raises:
        DealEntryError: describing every problem found.
    """
    problems: List[str] = []
    if entry.deal_type not in DEAL_TYPES:
        problems.append(f"deal_type must be one of {list(DEAL_TYPES)}, got {entry.deal_type!r}")
    if entry.status is not None and entry.status not in DEAL_STATUSES:
        problems.append(f"status must be one of {list(DEAL_STATUSES)}, got {entry.status!r}")
    if entry.completion_date is not None:
        if entry.completion_date < entry.announcement_date:
            problems.append("completion_date is before announcement_date")
        if entry.status not in (None, "completed"):
            problems.append(f"completion_date given but status is {entry.status!r}")
    if entry.upfront_per_share is not None and entry.upfront_per_share <= 0:
        problems.append("upfront_per_share must be positive")
    if entry.cvr_max_per_share is not None:
        if entry.has_cvr is False:
            problems.append("cvr_max_per_share given but has_cvr is no")
        if entry.cvr_max_per_share <= 0:
            problems.append("cvr_max_per_share must be positive")
    if entry.announcement_date > date.today():
        problems.append("announcement_date is in the future")
    if problems:
        raise DealEntryError("; ".join(problems))


# ---------------------------------------------------------------------------
# Derived context
# ---------------------------------------------------------------------------


def _close_on_basis(repo: Any, company_id: int, on_or_before: date, basis_date: date) -> Optional[Dict[str, Any]]:
    """Latest raw close on or before `on_or_before`, rescaled to `basis_date`'s share basis.

    None if there is no close or it is older than `MARKET_CAP_MAX_PRICE_AGE_DAYS`."""
    price = repo.get_latest_raw_close_as_of(company_id, on_or_before)
    if price is None or (on_or_before - price["trade_date"]).days > MARKET_CAP_MAX_PRICE_AGE_DAYS:
        return None
    factor = split_factor_between(repo.get_splits_as_of(company_id, basis_date), price["trade_date"], basis_date)
    return {"trade_date": price["trade_date"], "close": price["close_raw"] / factor, "split_factor": factor}


def premium(upfront_per_share: Optional[float], close: Optional[Dict[str, Any]]) -> Optional[float]:
    """`upfront / close - 1`, or None when either side is unknown."""
    if upfront_per_share is None or close is None or close["close"] <= 0:
        return None
    return upfront_per_share / close["close"] - 1.0


def evaluate_flags(
    entry: DealEntry, *, market_cap: Optional[float], reverse_splits: List[Dict[str, Any]]
) -> tuple[List[str], List[str]]:
    """
    Spec §2.5's reverse-merger heuristics.

    Returns:
        `(fired, unchecked)` — heuristic names that fired, and those that could not be evaluated
        because an input is missing.
    """
    fired: List[str] = []
    unchecked: List[str] = list(UNCHECKABLE_HEURISTICS)

    if entry.acquirer_is_private is None or (entry.acquirer_is_private and entry.acquirer_has_approved_product is None):
        unchecked.append("acquirer_private_no_product")
    elif entry.acquirer_is_private and entry.acquirer_has_approved_product is False:
        fired.append("acquirer_private_no_product")

    if market_cap is None:
        unchecked.append("target_market_cap_lt_50m")
    elif market_cap < REVERSE_MERGER_MARKET_CAP_USD:
        fired.append("target_market_cap_lt_50m")

    if entry.stock_consideration_pct is None or entry.target_holders_pct is None:
        unchecked.append("majority_stock_low_retention")
    elif entry.stock_consideration_pct > 0.5 and entry.target_holders_pct < 0.2:
        fired.append("majority_stock_low_retention")

    if reverse_splits:
        fired.append("reverse_split_within_90d")

    return fired, unchecked


def gather_context(repo: Any, target_id: int, entry: DealEntry) -> DealContext:
    """Prices, premiums, point-in-time market cap and reverse splits for one entry."""
    day_before = entry.announcement_date - timedelta(days=1)
    prior_close = _close_on_basis(repo, target_id, day_before, entry.announcement_date)
    close_30d = _close_on_basis(
        repo, target_id, entry.announcement_date - timedelta(days=PREMIUM_30D_LOOKBACK_DAYS), entry.announcement_date
    )
    market_cap_result, market_cap_reason = compute_market_cap(repo, target_id, day_before)
    market_cap = market_cap_result.market_cap if market_cap_result is not None else None

    actions = repo.get_corporate_actions_between(
        target_id, entry.announcement_date - timedelta(days=REVERSE_SPLIT_LOOKBACK_DAYS), entry.announcement_date
    )
    reverse_splits = [
        a for a in actions
        if a["action_type"] == "reverse_split" or (a["action_type"] == "split" and (a["ratio"] or 1.0) < 1.0)
    ]
    fired, unchecked = evaluate_flags(entry, market_cap=market_cap, reverse_splits=reverse_splits)
    return DealContext(
        premium_1d=premium(entry.upfront_per_share, prior_close),
        premium_30d=premium(entry.upfront_per_share, close_30d),
        prior_close=prior_close,
        close_30d=close_30d,
        market_cap=market_cap,
        market_cap_reason=market_cap_reason,
        reverse_splits=reverse_splits,
        flags_fired=fired,
        flags_unchecked=unchecked,
    )


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def _require_pending_candidate(item: Dict[str, Any]) -> int:
    if item["item_type"] != "deal_candidate":
        raise DealEntryError(f"Item {item['item_id']} is a {item['item_type']!r} item, not a deal_candidate")
    if item["status"] != "pending":
        raise DealEntryError(f"Item {item['item_id']} is already {item['status']}")
    return int(item["payload"]["company_id"])


def _resolve_acquirer_id(repo: Any, acquirer_cik: Optional[str]) -> Optional[int]:
    if not acquirer_cik:
        return None
    company = repo.get_company_by_cik(acquirer_cik)
    return company["company_id"] if company is not None else None


def record_deal(
    repo: Any,
    item: Dict[str, Any],
    entry: DealEntry,
    *,
    entered_by: str,
    acknowledge_flags: bool = False,
) -> Dict[str, Any]:
    """
    Write a hand-verified `p22_deal` row from a pending `deal_candidate` item and confirm the item.

    Args:
        repo: A `P22Repo`-shaped object.
        item: The review item (`get_review_item` row).
        entry: The hand-entered fields.
        entered_by: Reviewer identifier.
        acknowledge_flags: Required to enter a `strategic_acquisition` while heuristics fire.

    Returns:
        `{"deal_id": int, "context": DealContext}`.

    Raises:
        DealEntryError: invalid entry or item; `DuplicateDealError` if the deal already exists;
            `ReviewFlagsNotAcknowledgedError` if flags fired on a positive label without acknowledgement.
    """
    validate_entry(entry)
    target_id = _require_pending_candidate(item)

    existing = repo.find_deal(target_id, entry.announcement_date)
    if existing is not None:
        raise DuplicateDealError(
            f"Deal {existing['deal_id']} already exists for company {target_id} announced "
            f"{entry.announcement_date} — link this candidate to it instead"
        )

    context = gather_context(repo, target_id, entry)
    if entry.deal_type == POSITIVE_DEAL_TYPE and context.flags_fired and not acknowledge_flags:
        raise ReviewFlagsNotAcknowledgedError(
            f"Reverse-merger heuristics fired for a strategic_acquisition: {context.flags_fired}. "
            "Re-read the filing; if it really is a strategic acquisition, acknowledge the flags."
        )

    payload = item["payload"]
    deal_id = repo.insert_deal(
        target_id=target_id,
        acquirer_id=_resolve_acquirer_id(repo, entry.acquirer_cik),
        acquirer_name=entry.acquirer_name,
        announcement_date=entry.announcement_date,
        completion_date=entry.completion_date,
        upfront_per_share=entry.upfront_per_share,
        has_cvr=entry.has_cvr,
        cvr_max_per_share=entry.cvr_max_per_share,
        premium_1d=context.premium_1d,
        premium_30d=context.premium_30d,
        status=entry.status,
        deal_type=entry.deal_type,
        accession_no=payload.get("accession_no"),
        review_item_id=item["item_id"],
        stock_consideration_pct=entry.stock_consideration_pct,
        target_holders_pct=entry.target_holders_pct,
        review_flags=context.review_flags_json(),
        entered_by=entered_by,
        note=entry.note,
    )
    repo.resolve_review_item(item_id=item["item_id"], status="confirmed", reviewed_by=entered_by, note=f"deal_id={deal_id}")
    _logger.info(
        "Recorded deal %s: company %s, %s, announced %s, flags %s",
        deal_id, target_id, entry.deal_type, entry.announcement_date, context.flags_fired,
    )
    return {"deal_id": deal_id, "context": context}


def find_link_target(repo: Any, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The deal a further filing most plausibly belongs to: the same company's latest deal announced on
    or before the filing date. None if the company has no deal yet."""
    company_id = int(item["payload"]["company_id"])
    filed = parse_date(item["payload"].get("filed_date"))
    deals = repo.list_deals(target_id=company_id)
    if filed is not None:
        deals = [d for d in deals if d["announcement_date"] <= filed]
    return deals[-1] if deals else None


def link_candidate(
    repo: Any, item: Dict[str, Any], deal_id: Optional[int], *, reviewed_by: str, note: Optional[str] = None
) -> int:
    """
    Confirm a candidate as another filing (amendment, S-4 alongside an SC 14D9, ...) of an existing deal.

    Args:
        deal_id: The deal to link to; None picks `find_link_target`.

    Returns:
        The linked `deal_id`.
    """
    _require_pending_candidate(item)
    deal = repo.get_deal(deal_id) if deal_id is not None else find_link_target(repo, item)
    if deal is None:
        raise DealEntryError(
            f"No deal to link item {item['item_id']} to"
            + (f" (deal_id={deal_id} not found)" if deal_id is not None else " — enter the deal first")
        )
    text = f"filing of deal_id={deal['deal_id']}" + (f"; {note}" if note else "")
    repo.resolve_review_item(item_id=item["item_id"], status="confirmed", reviewed_by=reviewed_by, note=text)
    _logger.info("Linked review item %s to deal %s", item["item_id"], deal["deal_id"])
    return int(deal["deal_id"])
