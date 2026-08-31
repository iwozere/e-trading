"""
P22 — entity alias matching (spec §3.3).

The hardest data problem in this build: ClinicalTrials.gov sponsor strings,
FDA applicant names, and SEC registrant names for the same company routinely
differ. Two-step resolution against the already-resolved `p22_company`
roster:

1. **Deterministic** — exact match on `entity_resolution.normalize_company_name`.
   Written straight to `p22_company_alias` with `is_verified=True`.
2. **Fuzzy** — highest rapidfuzz `token_set_ratio` >= `FUZZY_MATCH_THRESHOLD`
   (88, per spec). Routed to the review queue, **never auto-accepted**
   (spec §3.3) — nothing is written to `p22_company_alias` until a human
   confirms it.

Neither: logged as unresolved, not silently dropped (spec §3.3: "Unresolved
sponsor strings are logged and reported weekly").

O(candidates x known_companies) per call — fine at this universe's scale
(~700-900 companies); revisit if the candidate list itself grows into the
same order of magnitude.

`extract_ctgov_sponsor_names`/`extract_openfda_sponsor_names` pull candidate
strings out of the raw landed payloads. Field paths live-verified against the
real APIs 2026-08-30: CT.gov's `leadSponsor.name` occasionally isn't a clean
company name at all — e.g. a merger-notice sentence like "Pfizer's Upjohn has
merged with Mylan to form Viatris Inc." was observed live for a real sponsor
query. That's not handled specially here; it just flows into `match_alias`
like any other candidate and, appropriately, fails to match anything and
lands in `unresolved` rather than being force-fit to a company. CT.gov
`collaborators` (also fetched by the ingest job) are deliberately not
extracted as alias candidates in this slice — spec §3.3/§2.2 point at
`leadSponsor` specifically, and a collaborator on a trial is not necessarily
the company being matched to a filer.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.entity_resolution import normalize_company_name
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

FUZZY_MATCH_THRESHOLD = 88  # spec §3.3: "Fuzzy match (token-set ratio >= 88)"


@dataclass(frozen=True)
class MatchResult:
    """One candidate name's resolution outcome."""

    candidate_name: str
    match_type: str  # 'deterministic' | 'fuzzy' | 'none'
    company_id: Optional[int]
    matched_name: Optional[str]
    score: Optional[float]


def match_alias(candidate_name: str, known_companies: Dict[int, str]) -> MatchResult:
    """
    Resolve one external sponsor/applicant string against `known_companies`
    (`company_id -> name`). Deterministic match wins outright; otherwise the
    best fuzzy match is returned (whether or not it clears the threshold —
    callers decide what to do with a sub-threshold best match).
    """
    normalized_candidate = normalize_company_name(candidate_name)

    for company_id, name in known_companies.items():
        if normalize_company_name(name) == normalized_candidate:
            return MatchResult(candidate_name, "deterministic", company_id, name, 100.0)

    best_company_id: Optional[int] = None
    best_name: Optional[str] = None
    best_score = 0.0
    for company_id, name in known_companies.items():
        score = float(fuzz.token_set_ratio(candidate_name, name))
        if score > best_score:
            best_score = score
            best_company_id = company_id
            best_name = name

    if best_company_id is not None and best_score >= FUZZY_MATCH_THRESHOLD:
        return MatchResult(candidate_name, "fuzzy", best_company_id, best_name, best_score)

    return MatchResult(candidate_name, "none", None, None, best_score if best_company_id is not None else None)


def resolve_aliases(
    candidates: List[Tuple[str, datetime]],
    known_companies: Dict[int, str],
    repo: Any,
    source: str,
) -> Dict[str, int]:
    """
    Run `match_alias` over every candidate name and write the outcome:
      - deterministic -> `p22_company_alias` (`is_verified=True`)
      - fuzzy -> review queue (`item_type='entity_match'`), not written to
        `p22_company_alias` until a human confirms (spec §3.3)
      - none -> logged only

    Args:
        candidates: `(candidate_name, known_from)` pairs — external
            sponsor/applicant strings (e.g. CT.gov `leadSponsor.name` values)
            paired with when *we* learned that string existed (the raw-zone
            landing timestamp of the payload it came from). Required for both
            outcomes that eventually write a `known_from`-bearing row: the
            deterministic path writes it immediately; the fuzzy path stores it
            in the review-item payload so a later CLI confirmation can write
            it correctly (spec §3.4: "confirmation writes back with
            `known_from` set to the underlying filing date, not the review
            date" — for a non-filing source like CT.gov/openFDA, the closest
            analog to a filing date is when we actually observed the string).
        known_companies: `company_id -> name`, the resolved roster to match against.
        repo: A `P22Repo`-shaped object (duck-typed for test doubles).
        source: Tag recorded on the alias/review item (e.g. `'clinicaltrials'`, `'openfda'`).

    Returns:
        Counters: `deterministic`, `fuzzy_flagged`, `unresolved`.
    """
    counts = {"deterministic": 0, "fuzzy_flagged": 0, "unresolved": 0}

    for candidate_name, known_from in candidates:
        result = match_alias(candidate_name, known_companies)

        if result.match_type == "deterministic" and result.company_id is not None:
            repo.add_company_alias(
                company_id=result.company_id,
                alias=candidate_name,
                source=source,
                is_verified=True,
                known_from=known_from,
            )
            counts["deterministic"] += 1
        elif result.match_type == "fuzzy" and result.company_id is not None:
            repo.add_review_item(
                item_type="entity_match",
                payload={
                    "reason": "fuzzy_alias_candidate",
                    "candidate_name": candidate_name,
                    "matched_company_id": result.company_id,
                    "matched_name": result.matched_name,
                    "score": result.score,
                    "source": source,
                    "known_from": known_from.isoformat(),
                },
                priority=0,
            )
            counts["fuzzy_flagged"] += 1
        else:
            _logger.warning("Unresolved %s sponsor/applicant name: %s", source, candidate_name)
            counts["unresolved"] += 1

    _logger.info("Alias resolution for source=%s: %s", source, counts)
    return counts


def extract_ctgov_sponsor_names(studies: List[Dict[str, Any]]) -> List[str]:
    """
    Pull `leadSponsor.name` out of landed CT.gov study payloads (field path
    live-verified 2026-08-30, see module docstring). Blank/missing names are
    dropped; duplicates are not — callers that want a unique candidate list
    should dedupe (`resolve_aliases` doesn't require it, but re-checking the
    same string N times is wasted work at scale).
    """
    names: List[str] = []
    for study in studies:
        lead_sponsor = study.get("protocolSection", {}).get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
        name = lead_sponsor.get("name")
        if name:
            names.append(name)
    return names


def extract_openfda_sponsor_names(applications: List[Dict[str, Any]]) -> List[str]:
    """
    Pull `sponsor_name` out of landed openFDA Drugs@FDA application payloads
    (field path live-verified 2026-08-30, see module docstring). openFDA
    stores this uppercased (e.g. "PFIZER") — passed through as-is;
    `normalize_company_name` lowercases in the deterministic-match path, and
    `rapidfuzz.token_set_ratio` is case-insensitive, so no extra normalization
    is needed here.
    """
    return [app["sponsor_name"] for app in applications if app.get("sponsor_name")]
