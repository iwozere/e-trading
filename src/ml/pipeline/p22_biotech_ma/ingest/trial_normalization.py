"""
P22 — ClinicalTrials.gov trial normalizer (spec §2.2, §3.2, M3 Block B input).

Turns landed `clinicaltrials_studies` raw-zone payloads (the CT.gov API v2
study records already landed by `run_clinicaltrials_ingest.py`, one partition
per company keyed by CIK — see that job's `raw_zone.write(entity=cik or name,
...)`) into `p22_trial` rows via `P22Repo.upsert_trial`.

**Scope of this pass**, field paths live-verified 2026-08-30 against a real
CT.gov response (a Moderna-sponsored study): every `p22_trial` column that can
be read directly off `CLINICALTRIALS_FIELDS` is populated. Three columns are
always written as `None` because the data to fill them honestly isn't fetched
yet, not because they were overlooked:

- `uses_biomarker_selection` — needs eligibility-criteria free-text parsing;
  `eligibilityModule` isn't in `CLINICALTRIALS_FIELDS` today.
- `has_active_comparator` — needs each arm group's own `type`
  (`EXPERIMENTAL`/`ACTIVE_COMPARATOR`/`PLACEBO_COMPARATOR`); CT.gov exposes
  this under `armGroupsModule.armGroups[].type`, a field not currently fetched
  (`armsInterventionsModule.interventions[].armGroupLabels` names the arms but
  not their type).
- `endpoint_changed_midtrial` — this is exactly what
  `clinicaltrials_client.fetch_study_version_history` lands
  (`clinicaltrials_history` raw-zone source), but diffing that history into a
  boolean is separate, not-yet-built feature-engineering work (see that
  client's module docstring).

**`asset_id` is linked ONLY for single-intervention trials, per user decision
2026-08-31 (`docs/Tasks.md` "Decisions needed" item 8).** A trial's
`armsInterventionsModule.interventions` can list multiple DRUG/BIOLOGICAL
entries (a company's own asset plus a combination/comparator partner drug —
see the Vertex/Moderna VX-522+IVA study checked live for this module), and
CT.gov gives no field marking which intervention is "this sponsor's own
asset" versus a comparator or partner compound. A trial with exactly ONE such
intervention has no such ambiguity — there's only one candidate — so
`ingest/asset_normalization.py` resolves/creates a `p22_asset` for those and
this module wires the resulting `asset_id` into `p22_trial`. Multi-
intervention trials still get `asset_id=None`, unchanged; solving that harder
case (a maintained drug-name-to-company mapping or a real entity-linking
step) remains open.

This module owns parsing only. It does not read the raw zone or open a DB
session — `jobs/run_trial_normalization.py` does that.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.asset_normalization import (
    extract_conditions,
    extract_single_intervention_name,
    resolve_or_create_asset,
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# CT.gov's `designInfo.allocation` values that unambiguously answer
# "was this trial randomized" — `NA` means "not applicable" (e.g. a
# single-arm or observational study), which is a real "we don't know /
# doesn't apply" case, not a `False`, so it maps to `None`.
_ALLOCATION_TO_IS_RANDOMIZED: Dict[str, Optional[bool]] = {
    "RANDOMIZED": True,
    "NON_RANDOMIZED": False,
    "NA": None,
}


@dataclass(frozen=True)
class TrialRecord:
    """One CT.gov study, normalized to `p22_trial` shape. See module docstring for the always-`None` fields."""

    nct_id: str
    phase: Optional[str]
    status: Optional[str]
    enrollment: Optional[int]
    primary_completion_date: Optional[date]
    uses_biomarker_selection: Optional[bool]  # always None this pass — see module docstring
    is_randomized: Optional[bool]
    has_active_comparator: Optional[bool]  # always None this pass — see module docstring
    primary_endpoint_text: Optional[str]
    endpoint_changed_midtrial: Optional[bool]  # always None this pass — see module docstring
    countries: Optional[List[str]]
    known_from: datetime
    single_intervention_name: Optional[str]  # set only for single-DRUG/BIOLOGICAL-intervention trials
    conditions: List[str]  # for asset therapeutic_area classification and indication text


def _parse_ctgov_date(raw: Optional[str]) -> Optional[date]:
    """
    CT.gov date-struct `date` values are usually `YYYY-MM-DD` but can be the
    coarser `YYYY-MM` (seen live on `ESTIMATED`-type dates) — treated as the
    first of that month rather than dropped, since a month-granularity date is
    still meaningfully more precise than `None`.
    """
    if not raw:
        return None
    parts = raw.split("-")
    try:
        if len(parts) == 3:
            return date.fromisoformat(raw)
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
    except (ValueError, TypeError):
        pass
    _logger.warning("Unparseable CT.gov date value: %r", raw)
    return None


def _extract_countries(study: Dict[str, Any]) -> Optional[List[str]]:
    locations = study.get("protocolSection", {}).get("contactsLocationsModule", {}).get("locations", [])
    countries: List[str] = []
    for loc in locations:
        country = loc.get("country")
        if country and country not in countries:
            countries.append(country)
    return countries or None


def extract_trial_record(study: Dict[str, Any], known_from: datetime) -> Optional[TrialRecord]:
    """
    Normalize one landed CT.gov study record. Returns `None` (logged) if the
    study has no `nctId` — shouldn't happen for a real API response, but a
    malformed/partial payload must not crash the whole batch.
    """
    protocol = study.get("protocolSection", {})
    nct_id = protocol.get("identificationModule", {}).get("nctId")
    if not nct_id:
        _logger.warning("CT.gov study payload missing nctId, skipping: %s", str(study)[:200])
        return None

    design = protocol.get("designModule", {})
    status_module = protocol.get("statusModule", {})
    outcomes = protocol.get("outcomesModule", {}).get("primaryOutcomes", [])

    phases = design.get("phases") or []
    phase = "/".join(phases) if phases else None

    enrollment_raw = design.get("enrollmentInfo", {}).get("count")
    enrollment = int(enrollment_raw) if enrollment_raw is not None else None

    allocation = design.get("designInfo", {}).get("allocation")
    is_randomized = _ALLOCATION_TO_IS_RANDOMIZED.get(allocation) if allocation else None

    endpoint_text = "; ".join(o["measure"] for o in outcomes if o.get("measure")) or None

    return TrialRecord(
        nct_id=nct_id,
        phase=phase,
        status=status_module.get("overallStatus"),
        enrollment=enrollment,
        primary_completion_date=_parse_ctgov_date(
            status_module.get("primaryCompletionDateStruct", {}).get("date")
        ),
        uses_biomarker_selection=None,
        is_randomized=is_randomized,
        has_active_comparator=None,
        primary_endpoint_text=endpoint_text,
        endpoint_changed_midtrial=None,
        countries=_extract_countries(study),
        known_from=known_from,
        single_intervention_name=extract_single_intervention_name(study),
        conditions=extract_conditions(study),
    )


def extract_trial_records(studies: List[Dict[str, Any]], known_from: datetime) -> List[TrialRecord]:
    """Normalize every study in a landed `clinicaltrials_studies` payload, dropping unparseable ones."""
    records = [extract_trial_record(study, known_from) for study in studies]
    return [r for r in records if r is not None]


def write_trial_records(records: List[TrialRecord], repo: Any, *, company_id: Optional[int] = None) -> int:
    """
    Write a list of `TrialRecord`s via `P22Repo.upsert_trial`. Returns the
    number written.

    Args:
        records: Trials to write.
        repo: A `P22Repo`-shaped object.
        company_id: The company these trials belong to. When given, and a
            record has a `single_intervention_name` (single-DRUG/BIOLOGICAL-
            intervention trial), resolves/creates a `p22_asset` and links it
            via `asset_id` (see module docstring). `None` (the default)
            preserves the old behavior of never linking — callers that don't
            have a company_id in scope, or tests exercising trial-only
            writes, are unaffected.
    """
    for record in records:
        asset_id = None
        if company_id is not None and record.single_intervention_name:
            asset_id = resolve_or_create_asset(
                company_id=company_id,
                intervention_name=record.single_intervention_name,
                conditions=record.conditions,
                repo=repo,
            )
        repo.upsert_trial(
            nct_id=record.nct_id,
            asset_id=asset_id,
            phase=record.phase,
            status=record.status,
            enrollment=record.enrollment,
            primary_completion_date=record.primary_completion_date,
            uses_biomarker_selection=record.uses_biomarker_selection,
            is_randomized=record.is_randomized,
            has_active_comparator=record.has_active_comparator,
            primary_endpoint_text=record.primary_endpoint_text,
            endpoint_changed_midtrial=record.endpoint_changed_midtrial,
            countries=record.countries,
            known_from=record.known_from,
        )
    return len(records)
