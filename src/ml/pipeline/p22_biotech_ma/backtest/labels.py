"""
P22 — deal labels for the backtest (spec §2.5, §5.3, §5.5), M6, 2026-10-30.

**`P(deal)` label:** a company is positive at `as_of` when a `strategic_acquisition` naming it as
target is announced in `(as_of, as_of + horizon]` (spec §5.5: "announced within 24 months"; labels
key on announcement, §2.5, so a later-terminated deal still counts). `pe_take_private` is kept as
its own label and never pooled (§2.5). The excluded types (reverse merger, shell, liquidation,
asset sale) are not positives; the company stays in the universe as a negative, which is what it
was for a strategic buyer.

**Pending deals:** a company whose own deal was already announced on or before `as_of` and has not
closed or been terminated is in a known process, not a prediction problem, so the harness drops it
from that snapshot. Without a `completion_date` the deal counts as pending for
`PENDING_WITHOUT_COMPLETION_DAYS` after announcement.

**Censoring:** a snapshot is only usable when its whole window is covered by the label set, i.e.
`as_of + horizon <= labels_complete_through` (the day the deal review was last brought current).

**Return label (`E[return | deal]`, §5.3, for M7):** `consideration / close(as_of) - 1`, with the
`as_of` close rescaled to the announcement's share basis and the consideration valued under
`p22_cvr_policy.yaml` (v1: CVRs at zero, upfront only). `None` when the company's price series is
suspect (`price_quality.is_price_suspect`, e.g. corrupt vintage prices or ADRs without a ratio) or
no recent close exists.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import CVR_POLICY_YAML
from src.ml.pipeline.p22_biotech_ma.ingest.deal_entry import POSITIVE_DEAL_TYPE, close_on_basis
from src.ml.pipeline.p22_biotech_ma.ingest.price_quality import is_price_suspect

PE_DEAL_TYPE = "pe_take_private"
LABEL_HORIZON_MONTHS = 24
PENDING_WITHOUT_COMPLETION_DAYS = 365

_CVR_METHODS = ("zero",)


class CvrPolicyError(ValueError):
    """`p22_cvr_policy.yaml` names a valuation method this code does not implement."""


@dataclass(frozen=True)
class CvrPolicy:
    """Parsed `p22_cvr_policy.yaml`."""

    valuation_method: str
    return_basis: str


def load_cvr_policy(path: Path = CVR_POLICY_YAML) -> CvrPolicy:
    """Read and validate the CVR policy. Only `zero` is implemented; `probability_weighted` is M7+."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    policy = CvrPolicy(valuation_method=str(raw.get("valuation_method")), return_basis=str(raw.get("return_basis")))
    if policy.valuation_method not in _CVR_METHODS:
        raise CvrPolicyError(f"Unsupported CVR valuation_method {policy.valuation_method!r}; implemented: {_CVR_METHODS}")
    if policy.return_basis != "upfront_per_share":
        raise CvrPolicyError(f"Unsupported return_basis {policy.return_basis!r}")
    return policy


def add_months(day: date, months: int) -> date:
    """`day` shifted by whole months, clamped to the target month's last day (Jan 31 + 1 -> Feb 28/29)."""
    month_index = day.month - 1 + months
    year, month = day.year + month_index // 12, month_index % 12 + 1
    for candidate_day in (day.day, 30, 29, 28):
        try:
            return date(year, month, candidate_day)
        except ValueError:
            continue
    raise ValueError(f"Cannot shift {day} by {months} months")


class DealLabels:
    """Label lookups over the entered `p22_deal` rows."""

    def __init__(self, deals: Sequence[Dict[str, Any]], horizon_months: int = LABEL_HORIZON_MONTHS) -> None:
        self.deals = [d for d in deals if d.get("target_id") is not None and d.get("announcement_date") is not None]
        self.horizon_months = horizon_months

    def window_end(self, as_of: date) -> date:
        """Last announcement date that still counts for a snapshot at `as_of`."""
        return add_months(as_of, self.horizon_months)

    def _announced_in_window(self, as_of: date, deal_types: Sequence[str]) -> set[int]:
        end = self.window_end(as_of)
        return {
            d["target_id"] for d in self.deals
            if d.get("deal_type") in deal_types and as_of < d["announcement_date"] <= end
        }

    def positives(self, as_of: date) -> set[int]:
        """Targets of a `strategic_acquisition` announced in `(as_of, as_of + horizon]`."""
        return self._announced_in_window(as_of, (POSITIVE_DEAL_TYPE,))

    def pe_positives(self, as_of: date) -> set[int]:
        """Targets of a `pe_take_private` in the same window — the separate label, never pooled."""
        return self._announced_in_window(as_of, (PE_DEAL_TYPE,))

    def pending(self, as_of: date) -> set[int]:
        """Targets with a deal of any type announced on or before `as_of` that is still open at `as_of`."""
        pending: set[int] = set()
        for d in self.deals:
            announced = d["announcement_date"]
            if announced > as_of or d.get("status") == "terminated":
                continue
            completion = d.get("completion_date")
            if completion is not None:
                if completion > as_of:
                    pending.add(d["target_id"])
            elif as_of <= announced + timedelta(days=PENDING_WITHOUT_COMPLETION_DAYS):
                pending.add(d["target_id"])
        return pending

    def first_positive_deal(self, company_id: int, as_of: date) -> Optional[Dict[str, Any]]:
        """The earliest positive deal for `company_id` inside the snapshot window, or None."""
        end = self.window_end(as_of)
        matches = [
            d for d in self.deals
            if d["target_id"] == company_id and d.get("deal_type") == POSITIVE_DEAL_TYPE and as_of < d["announcement_date"] <= end
        ]
        return min(matches, key=lambda d: d["announcement_date"]) if matches else None


def is_censored(as_of: date, labels_complete_through: date, horizon_months: int = LABEL_HORIZON_MONTHS) -> bool:
    """True when part of the snapshot's label window lies after the label set's coverage."""
    return add_months(as_of, horizon_months) > labels_complete_through


def consideration_value(deal: Dict[str, Any], policy: CvrPolicy) -> Optional[float]:
    """Per-share consideration under the CVR policy (`zero`: the upfront amount only)."""
    upfront = deal.get("upfront_per_share")
    if upfront is None:
        return None
    if policy.valuation_method == "zero":
        return float(upfront)
    raise CvrPolicyError(f"Unsupported CVR valuation_method {policy.valuation_method!r}")


def deal_return(repo: Any, deal: Dict[str, Any], as_of: date, policy: CvrPolicy) -> Optional[float]:
    """
    Return from the `as_of` close to the deal consideration (spec §5.3's `E[return | deal]` target).

    The close is the latest raw close on or before `as_of`, rescaled to the announcement date's share
    basis so a split between the two cannot fake a return. None for a suspect price series or a
    missing/stale close.
    """
    consideration = consideration_value(deal, policy)
    if consideration is None:
        return None
    target_id = deal["target_id"]
    if is_price_suspect(repo, target_id, as_of):
        return None
    close = close_on_basis(repo, target_id, as_of, deal["announcement_date"])
    if close is None or close["close"] <= 0:
        return None
    return consideration / close["close"] - 1.0


def return_labels(repo: Any, labels: DealLabels, company_ids: Sequence[int], as_of: date, policy: CvrPolicy) -> List[Dict[str, Any]]:
    """`deal_return` for every company in `company_ids` with a positive deal in the window."""
    rows: List[Dict[str, Any]] = []
    for company_id in company_ids:
        deal = labels.first_positive_deal(company_id, as_of)
        if deal is None:
            continue
        rows.append({
            "as_of": as_of,
            "company_id": company_id,
            "deal_id": deal.get("deal_id"),
            "announcement_date": deal["announcement_date"],
            "return_if_deal": deal_return(repo, deal, as_of, policy),
        })
    return rows
