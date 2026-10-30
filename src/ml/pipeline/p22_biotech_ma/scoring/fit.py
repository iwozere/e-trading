"""
P22 — Block D: pairwise strategic fit (spec §4.4), M4, 2026-10-29.

    fit(acquirer, target) = preference × size_feasibility × (1 − fis_penalty) × (1 − antitrust_penalty)

Preference terms are additive and weighted; the three gates are multiplicative so a blocked deal can't
be out-voted by how badly the acquirer wants the asset (spec §4.4). Everything here is pure: profiles
are assembled by `scoring/score_run.py`, so the formula is unit-testable without a database.

**Missing inputs.** Spec §4: `None` propagates as missing, never as zero.
- A preference term whose input is unknown is left out and the remaining weights are renormalized;
  when no term is computable the pair's fit is `None`.
- A gate whose input is unknown is treated as open (1.0) and recorded in `unknown_gates`, because
  closing it would silently drop every pair. That is the dominant case today: `dry_powder` needs
  `ebitda`/`currency_quality` (see `features/block_a.py`), so `size_feasibility` is unknown for every
  acquirer until those exist.

**Term definitions the spec leaves open** (recorded in docs/Design.md "Phase 1 scoring"):
- `ta_overlap` — 1.0 when the acquirer's `pipeline_gap_by_ta` for the target lead asset's TA is
  positive (it needs to buy into that area), 0.0 when ≤ 0, `None` when the gap is unknown.
- `modality_capability` — 1.0 when the acquirer owns an asset of the target lead asset's modality,
  else 0.0; `None` when the target's modality is unknown/`unclassified` or the acquirer has no
  classified assets on file.
- `geographic_fit` — share of the target lead asset's trial countries inside the acquirer's commercial
  footprint. No footprint data exists, so this is `None` for every pair today.
- `antitrust_penalty` — the configured penalty when the acquirer already has a Phase III+ asset in the
  target lead asset's exact indication (spec §4.4.2's "me-too asset gated against the incumbent").
  Exact normalized-string match on `p22_asset.indication`, so it under-detects.
- `fis_penalty` — the acquirer bloc's penalty. The spec's target-side increment (BARDA/NIH funding,
  US genomic data, critical-infrastructure manufacturing) has no data source and is not applied.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.scoring.weights_config import ScoringConfig

UNCLASSIFIED_MODALITY = "unclassified"


def normalize_indication(indication: Optional[str]) -> Optional[str]:
    """Lower-cased, whitespace-collapsed indication string, or None when blank."""
    if not indication or not indication.strip():
        return None
    return " ".join(indication.lower().split())


@dataclass(frozen=True)
class AcquirerProfile:
    """Everything `fit()` needs about one acquirer as of the run date."""

    company_id: int
    name: str
    bloc: str
    dry_powder: Optional[float] = None
    pipeline_gap_by_ta: Dict[str, Optional[float]] = field(default_factory=dict)
    modalities: FrozenSet[str] = frozenset()
    late_stage_indications: FrozenSet[str] = frozenset()
    commercial_countries: Optional[FrozenSet[str]] = None


@dataclass(frozen=True)
class TargetProfile:
    """Everything `fit()` needs about one target's lead asset as of the run date."""

    company_id: int
    enterprise_value: Optional[float] = None
    lead_therapeutic_area: Optional[str] = None
    lead_modality: Optional[str] = None
    lead_indication: Optional[str] = None
    lead_trial_countries: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class FitResult:
    """One (acquirer, target) pair with its per-term breakdown for the dossier (spec §4.4)."""

    acquirer_id: int
    acquirer_name: str
    fit: Optional[float]
    preference: Optional[float]
    terms: Dict[str, Optional[float]]
    gates: Dict[str, float]
    unknown_gates: List[str]

    def as_json(self) -> Dict[str, Any]:
        return {
            "acquirer_id": self.acquirer_id,
            "acquirer_name": self.acquirer_name,
            "fit": self.fit,
            "preference": self.preference,
            "terms": self.terms,
            "gates": self.gates,
            "unknown_gates": self.unknown_gates,
        }


def ta_overlap(acquirer: AcquirerProfile, target: TargetProfile) -> Optional[float]:
    if target.lead_therapeutic_area is None:
        return None
    gap = acquirer.pipeline_gap_by_ta.get(target.lead_therapeutic_area)
    if gap is None:
        return None
    return 1.0 if gap > 0 else 0.0


def modality_capability(acquirer: AcquirerProfile, target: TargetProfile) -> Optional[float]:
    if target.lead_modality in (None, UNCLASSIFIED_MODALITY) or not acquirer.modalities:
        return None
    return 1.0 if target.lead_modality in acquirer.modalities else 0.0


def geographic_fit(acquirer: AcquirerProfile, target: TargetProfile) -> Optional[float]:
    if acquirer.commercial_countries is None or not target.lead_trial_countries:
        return None
    inside = len(target.lead_trial_countries & acquirer.commercial_countries)
    return inside / len(target.lead_trial_countries)


def size_feasibility(dry_powder: Optional[float], enterprise_value: Optional[float], full_below: float) -> Optional[float]:
    """1.0 while EV < `full_below` × dry powder, linear down to 0 at EV = dry powder (spec §4.4). None if unknown."""
    if dry_powder is None or enterprise_value is None:
        return None
    if enterprise_value <= full_below * dry_powder:
        return 1.0
    if dry_powder <= 0 or enterprise_value >= dry_powder:
        return 0.0
    return (dry_powder - enterprise_value) / (dry_powder * (1.0 - full_below))


def antitrust_penalty(acquirer: AcquirerProfile, target: TargetProfile, penalty: float) -> Optional[float]:
    indication = normalize_indication(target.lead_indication)
    if indication is None:
        return None
    return penalty if indication in acquirer.late_stage_indications else 0.0


_PREFERENCE_TERMS = {
    "ta_overlap": ta_overlap,
    "modality_capability": modality_capability,
    "geographic_fit": geographic_fit,
}


def fit(acquirer: AcquirerProfile, target: TargetProfile, config: ScoringConfig) -> FitResult:
    """Score one pair (spec §4.4). See module docstring for how unknown terms and gates are handled."""
    terms = {name: fn(acquirer, target) for name, fn in _PREFERENCE_TERMS.items()}
    available = {name: value for name, value in terms.items() if value is not None}
    weight_total = sum(config.fit_preference.get(name, 0.0) for name in available)
    preference: Optional[float] = None
    if available and weight_total > 0:
        preference = sum(config.fit_preference.get(name, 0.0) * value for name, value in available.items()) / weight_total

    unknown_gates: List[str] = []
    size = size_feasibility(acquirer.dry_powder, target.enterprise_value, config.size_feasibility_full_below)
    if size is None:
        unknown_gates.append("size_feasibility")
    antitrust = antitrust_penalty(acquirer, target, config.antitrust_same_indication_penalty)
    if antitrust is None:
        unknown_gates.append("antitrust")
    gates = {
        "size_feasibility": size if size is not None else 1.0,
        "fis": 1.0 - config.fis_penalty_by_bloc[acquirer.bloc],
        "antitrust": 1.0 - (antitrust or 0.0),
    }

    value: Optional[float] = None
    if preference is not None:
        value = preference * gates["size_feasibility"] * gates["fis"] * gates["antitrust"]
    return FitResult(
        acquirer_id=acquirer.company_id,
        acquirer_name=acquirer.name,
        fit=value,
        preference=preference,
        terms=terms,
        gates=gates,
        unknown_gates=unknown_gates,
    )


def best_fits(acquirers: List[AcquirerProfile], target: TargetProfile, config: ScoringConfig) -> List[FitResult]:
    """Every pair for one target, best first (pairs with unknown fit last, ties by acquirer id)."""
    results = [fit(acquirer, target, config) for acquirer in acquirers]
    return sorted(results, key=lambda r: (r.fit is None, -(r.fit or 0.0), r.acquirer_id))
