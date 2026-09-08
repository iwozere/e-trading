"""
P22 — CT.gov single-intervention-trial asset linkage (spec §3.2, M3 Block B input).

Resolves `docs/Tasks.md` "Decisions needed" item 8's safe subset, per user
decision 2026-08-31: a trial with **exactly one** DRUG/BIOLOGICAL-type
intervention has no ambiguity about which intervention is the sponsor's own
asset (there's only one candidate) — unlike the Vertex/Moderna VX-522+IVA
example that motivated leaving `asset_id` unconditionally `None`
(`ingest/trial_normalization.py`'s docstring), where a comparator/partner
drug made "which intervention is the asset" a real question. Multi-
intervention trials are still left unlinked here — this module does not
attempt to solve that harder case.

Assets are deduplicated per `(company_id, name)` — repeated trials for the
same drug program resolve to the same `p22_asset` row rather than creating a
duplicate per trial. `therapeutic_area` (`NOT NULL` in the schema) comes from
`therapeutic_area_classifier.classify_therapeutic_area`, a best-effort
keyword heuristic over the trial's `conditions` — see that module's docstring
for its disclosed limitations. `modality` (added 2026-09-08) comes from the
same style of best-effort keyword classifier
(`modality_classifier.classify_modality`, over the intervention's own
name/type) — same "candidate classification pending review, not ground
truth" disclosure. `target_protein`/`is_lead` are still always `None`: neither
is derivable from CT.gov's intervention name/type at all (`is_lead` needs
company-level pipeline context this trial-by-trial view doesn't have).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.modality_classifier import classify_modality
from src.ml.pipeline.p22_biotech_ma.ingest.therapeutic_area_classifier import classify_therapeutic_area
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_ASSET_INTERVENTION_TYPES = frozenset({"DRUG", "BIOLOGICAL"})


def extract_single_intervention_name(study: Dict[str, Any]) -> Optional[str]:
    """
    Return the trial's sole DRUG/BIOLOGICAL intervention name, or `None` if
    there are zero or multiple such interventions (the ambiguous case this
    module deliberately does not handle — see module docstring).
    """
    interventions = study.get("protocolSection", {}).get("armsInterventionsModule", {}).get("interventions", [])
    drug_like = [i for i in interventions if i.get("type") in _ASSET_INTERVENTION_TYPES]
    if len(drug_like) != 1:
        return None
    name = drug_like[0].get("name")
    return name.strip() if name and name.strip() else None


def extract_single_intervention_type(study: Dict[str, Any]) -> Optional[str]:
    """
    Return the trial's sole DRUG/BIOLOGICAL intervention's CT.gov `type`, mirroring
    `extract_single_intervention_name`'s exact selection logic. Kept as a separate function
    rather than changing that one to return a tuple — `intervention_type` is only needed by
    `resolve_or_create_asset`'s modality classification, added 2026-09-08 well after
    `single_intervention_name`'s original call sites were written; a tuple return would have
    forced updating every existing caller for a field most of them don't need.
    """
    interventions = study.get("protocolSection", {}).get("armsInterventionsModule", {}).get("interventions", [])
    drug_like = [i for i in interventions if i.get("type") in _ASSET_INTERVENTION_TYPES]
    if len(drug_like) != 1:
        return None
    return drug_like[0].get("type")


def extract_conditions(study: Dict[str, Any]) -> List[str]:
    """The trial's free-text `conditions` list, for `therapeutic_area` classification and `indication`."""
    return study.get("protocolSection", {}).get("conditionsModule", {}).get("conditions", []) or []


def resolve_or_create_asset(
    *,
    company_id: int,
    intervention_name: str,
    conditions: List[str],
    repo: Any,
    intervention_type: Optional[str] = None,
) -> int:
    """
    Look up an existing `p22_asset` for `(company_id, intervention_name)`, or
    create one, classifying `therapeutic_area` from `conditions` via
    `therapeutic_area_classifier` and `modality` from the intervention's own
    name/type via `modality_classifier` (added 2026-09-08 — `intervention_type`
    defaults to `None` so existing callers that don't have it in scope, e.g.
    tests exercising the therapeutic_area path alone, are unaffected; a
    missing type doesn't block classification since `classify_modality`
    currently keys off `intervention_name` alone anyway). Idempotent —
    repeated calls for the same (company, asset name) return the same
    `asset_id`, never a duplicate row.

    Returns:
        The (new or pre-existing) `asset_id`.
    """
    existing = repo.get_asset_by_company_and_name(company_id, intervention_name)
    if existing is not None:
        return existing["asset_id"]

    therapeutic_area = classify_therapeutic_area(conditions)
    if therapeutic_area == "unclassified":
        _logger.info(
            "No keyword match classifying therapeutic_area for asset=%r company_id=%s conditions=%s "
            "— written as 'unclassified', see therapeutic_area_classifier.py docstring",
            intervention_name, company_id, conditions,
        )

    modality = classify_modality(intervention_type, intervention_name)
    if modality == "unclassified":
        _logger.info(
            "No keyword match classifying modality for asset=%r company_id=%s — written as "
            "'unclassified', see modality_classifier.py docstring",
            intervention_name, company_id,
        )

    return repo.upsert_asset(
        company_id=company_id,
        name=intervention_name,
        therapeutic_area=therapeutic_area,
        modality=modality,
        target_protein=None,
        indication="; ".join(conditions) if conditions else None,
        is_lead=None,
    )
