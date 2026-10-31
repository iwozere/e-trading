"""
P22 — roster disappearances vs `p22_deal` (spec §2.0.1, M6), 2026-10-30.

A company that stops filing 10-K/10-Q was acquired, delisted, went dark or liquidated. Spec §2.0.1
asks for the disappearances to be classified against the deal labels. Each disappeared company falls
into one of:

- `deal_entered` — a `p22_deal` row exists (the type says acquired, reverse merger, liquidation, ...);
- `candidates_pending` — candidate filings are queued but not reviewed yet;
- `candidates_reviewed_no_deal` — every candidate was reviewed and none became a deal (rejected as
  acquirer side, say). Most are genuine non-deals; worth a glance when the company vanished soon after;
- `no_candidate` — nothing in the queue. Either a non-deal exit (bankruptcy, delisting, going dark)
  or a deal the SC 14D9/DEFM14A/S-4 scan cannot see (a scheme of arrangement for a foreign issuer, a
  merger approved by written consent with only an 8-K and information statement). These are the label
  gaps to check by hand.

Disappeared means: no 10-K/10-Q within the reporting lookback (183 days) of the latest landed DERA
quarter, the same rule as universe eligibility.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.entity_resolution import UniverseCandidate, build_universe, quarter_end_date

CLASSIFICATIONS = ("deal_entered", "candidates_pending", "candidates_reviewed_no_deal", "no_candidate")


@dataclass
class Disappearance:
    """One company that stopped filing, and what the label set knows about it."""

    cik: str
    name: str
    company_id: Optional[int]
    ticker: Optional[str]
    last_filed: Optional[date]
    last_form: Optional[str]
    classification: str
    deal_types: List[str] = field(default_factory=list)
    candidate_statuses: Dict[str, int] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        """Flat dict for CSV output."""
        return {
            "classification": self.classification,
            "cik": self.cik,
            "company_id": self.company_id,
            "ticker": self.ticker,
            "name": self.name,
            "last_filed": self.last_filed,
            "last_form": self.last_form,
            "deal_types": ";".join(self.deal_types),
            "candidate_statuses": ";".join(f"{k}={v}" for k, v in sorted(self.candidate_statuses.items())),
        }


def latest_candidates(quarters_rows: Mapping[str, List[Dict[str, Any]]]) -> tuple[Optional[date], List[UniverseCandidate]]:
    """Every CIK in the landed DERA quarters, evaluated as of the latest quarter end."""
    ends = [end for q in quarters_rows if (end := quarter_end_date(q)) is not None]
    if not ends:
        return None, []
    as_of = max(ends)
    all_rows = [row for rows in quarters_rows.values() for row in rows]
    return as_of, build_universe(all_rows, {}, as_of)


def classify(
    candidates: Sequence[UniverseCandidate],
    companies_by_cik: Mapping[str, Dict[str, Any]],
    deals: Sequence[Dict[str, Any]],
    review_items: Sequence[Dict[str, Any]],
    since: date,
) -> List[Disappearance]:
    """
    Classify every candidate that stopped reporting on or after `since`.

    Args:
        candidates: `latest_candidates` output.
        companies_by_cik: `p22_company` rows keyed by 10-digit CIK.
        deals: Every `p22_deal` row.
        review_items: Every `deal_candidate` review item, any status.
        since: Ignore companies whose last 10-K/10-Q is older than this.
    """
    deal_types_by_target: Dict[int, List[str]] = {}
    for d in deals:
        if d.get("target_id") is not None:
            deal_types_by_target.setdefault(d["target_id"], []).append(str(d.get("deal_type")))
    statuses_by_company: Dict[int, Dict[str, int]] = {}
    for item in review_items:
        company_id = (item.get("payload") or {}).get("company_id")
        if company_id is None:
            continue
        counts = statuses_by_company.setdefault(int(company_id), {})
        counts[item["status"]] = counts.get(item["status"], 0) + 1

    out: List[Disappearance] = []
    for c in candidates:
        if c.eligible_reporting or c.most_recent_filed is None or c.most_recent_filed < since:
            continue
        company = companies_by_cik.get(c.cik)
        company_id = company["company_id"] if company is not None else None
        deal_types = deal_types_by_target.get(company_id, []) if company_id is not None else []
        statuses = statuses_by_company.get(company_id, {}) if company_id is not None else {}
        if deal_types:
            classification = "deal_entered"
        elif statuses.get("pending"):
            classification = "candidates_pending"
        elif statuses:
            classification = "candidates_reviewed_no_deal"
        else:
            classification = "no_candidate"
        out.append(Disappearance(
            cik=c.cik, name=c.name, company_id=company_id, ticker=company.get("ticker") if company else c.ticker,
            last_filed=c.most_recent_filed, last_form=c.most_recent_form, classification=classification,
            deal_types=deal_types, candidate_statuses=statuses,
        ))
    out.sort(key=lambda d: (CLASSIFICATIONS.index(d.classification), d.last_filed or date.min, d.name))
    return out
