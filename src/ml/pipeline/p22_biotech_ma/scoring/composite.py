"""
P22 — Phase 1 rule-based composite (spec §5.1–§5.2), M4, 2026-10-29. Pure functions, no database.

    fundamental = ( 0.30·acquirer_pressure_matched + 0.25·target_asset_quality + 0.20·strategic_fit
                  + 0.15·financial_attractiveness − 0.10·friction_penalty ) × dilution_gate
    composite   = tier + fundamental                                   (spec §5.2, tier ∈ 0..3)

Every block score is a cross-sectional rank percentile within the as-of universe (spec §5.1). A block
score is built in two steps: each feature is rank-percentiled (flipped for `direction: lower`), the
block's raw value is the weighted mean of its available feature percentiles, and that raw value is
rank-percentiled again so every block lands on the same [0, 1] scale.

**Missing blocks** (spec §4: missing is never zero). When some positive blocks are unknown for a
company, the known ones are scaled up to the full positive weight (0.90 with spec weights):

    positive = Σ_known w_i·s_i × (Σ_all w_i / Σ_known w_i)

which equals the spec formula exactly once every block is known, and is a weighted mean of what is
known before then. `coverage` (= Σ_known w_i / Σ_all w_i) is stored in `subscores` so a high score
built on one block is visible as such. An unknown friction penalty subtracts nothing; an unknown
`dilution_risk` leaves the gate open. Both are recorded. No known positive block → no fundamental
score, and the company is left unranked.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.scoring.weights_config import (
    COMPOSITE_POSITIVE_TERMS,
    FRICTION_TERM,
    FeatureWeight,
    ScoringConfig,
)


def rank_percentile(values: Mapping[int, Optional[float]]) -> Dict[int, float]:
    """
    Share of the other known values strictly below each value, in [0, 1].

    Ties share the lower rank, so a binary flag maps its zeros to exactly 0.0 (a friction nobody has
    costs nobody anything). Unknown values are left out. A single known value scores 0.5.
    """
    known = {key: value for key, value in values.items() if value is not None}
    if not known:
        return {}
    if len(known) == 1:
        return {key: 0.5 for key in known}
    ordered = sorted(known.values())
    result: Dict[int, float] = {}
    for key, value in known.items():
        below = _count_below(ordered, value)
        result[key] = below / (len(known) - 1)
    return result


def _count_below(ordered: List[float], value: float) -> int:
    lo, hi = 0, len(ordered)
    while lo < hi:
        mid = (lo + hi) // 2
        if ordered[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def block_scores(
    features: Mapping[int, Mapping[str, Optional[float]]], weights: Mapping[str, FeatureWeight]
) -> Dict[int, Optional[float]]:
    """
    One block's score per company from its raw feature values.

    Args:
        features: `{company_id: {feature_name: raw value or None}}`.
        weights: The block's `FeatureWeight`s; zero-weight features are ignored.

    Returns:
        `{company_id: score in [0, 1] or None}` for every company in `features`.
    """
    percentiles: Dict[str, Dict[int, float]] = {}
    for name, fw in weights.items():
        if fw.weight <= 0:
            continue
        pct = rank_percentile({cid: f.get(name) for cid, f in features.items()})
        percentiles[name] = {cid: (p if fw.direction == "higher" else 1.0 - p) for cid, p in pct.items()}

    raw: Dict[int, Optional[float]] = {}
    for company_id in features:
        parts = [(weights[name].weight, pct[company_id]) for name, pct in percentiles.items() if company_id in pct]
        total = sum(w for w, _ in parts)
        raw[company_id] = sum(w * p for w, p in parts) / total if total > 0 else None
    normalized = rank_percentile(raw)
    return {company_id: normalized.get(company_id) for company_id in features}


@dataclass(frozen=True)
class Fundamental:
    """One company's §5.1 fundamental score and how it was assembled."""

    value: Optional[float]
    coverage: float
    missing_blocks: List[str]
    contributions: Dict[str, float]
    dilution_gate: float
    notes: List[str] = field(default_factory=list)


def fundamental_score(
    blocks: Mapping[str, Optional[float]], dilution_risk: Optional[float], config: ScoringConfig
) -> Fundamental:
    """
    The spec §5.1 formula for one company.

    Args:
        blocks: `{term: block score or None}` for the four positive terms and `friction_penalty`.
        dilution_risk: `block_c.dilution_risk` (1.0 / 0.0 / None).
        config: Weights.
    """
    weights = config.composite
    full_weight = sum(weights[t] for t in COMPOSITE_POSITIVE_TERMS)
    known: Dict[str, float] = {}
    missing: List[str] = []
    for term in COMPOSITE_POSITIVE_TERMS:
        score = blocks.get(term)
        if score is None:
            missing.append(term)
        else:
            known[term] = score
    known_weight = sum(weights[t] for t in known)
    notes: List[str] = []

    gate = config.dilution_gate_factor if dilution_risk == 1.0 else 1.0
    if dilution_risk is None:
        notes.append("dilution_risk_unknown")
    if not known or known_weight <= 0:
        return Fundamental(None, 0.0, missing, {}, gate, notes)

    scale = full_weight / known_weight
    contributions = {t: weights[t] * score * scale for t, score in known.items()}
    friction = blocks.get(FRICTION_TERM)
    if friction is None:
        missing.append(FRICTION_TERM)
    else:
        contributions[FRICTION_TERM] = -weights[FRICTION_TERM] * friction
    value = sum(contributions.values()) * gate
    return Fundamental(value, known_weight / full_weight, missing, contributions, gate, notes)


def rank_desc(values: Mapping[int, Optional[float]]) -> Dict[int, int]:
    """1-based rank by value descending (ties broken by company id); unknown values get no rank."""
    known = sorted(((v, cid) for cid, v in values.items() if v is not None), key=lambda x: (-x[0], x[1]))
    return {cid: i + 1 for i, (_, cid) in enumerate(known)}
