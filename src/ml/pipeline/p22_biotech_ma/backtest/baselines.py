"""
P22 — the spec §0.3 baseline rankings, plus the model's own ranking and one per composite block
(for attribution: which block carries signal), as ordered company-id lists.

Every ranking puts unknown values last and breaks remaining ties by `company_id`, so a run is
reproducible. Random is not a list: its expectations come straight from `metrics.PooledMetrics`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.block_b import PHASE_RANK_LABELS
from src.ml.pipeline.p22_biotech_ma.scoring.weights_config import FRICTION_TERM

PHASE2_RANK = PHASE_RANK_LABELS.index("PHASE2")


def _desc_key(value: Optional[float]) -> tuple[int, float]:
    """Sort key putting larger values first and None last."""
    return (1, 0.0) if value is None else (0, -value)


def naive_informed(features: Mapping[int, Mapping[str, Optional[float]]]) -> List[int]:
    """
    Spec §0.3 "naive-informed": `phase_max` descending, tie-broken by `cash_runway_months` descending.
    Companies with no known phase are ranked after every company with one.
    """
    return sorted(
        features,
        key=lambda cid: (_desc_key(features[cid].get("phase_max")), _desc_key(features[cid].get("cash_runway_months")), cid),
    )


def catalyst_proximity(features: Mapping[int, Mapping[str, Optional[float]]]) -> List[int]:
    """
    Spec §0.3 "catalyst-proximity": inverse days to the next known catalyst, restricted to Phase II+.
    Only companies at Phase II or later with an upcoming catalyst are ranked; the list can be shorter
    than k.
    """
    eligible: List[tuple[float, int]] = []
    for cid, f in features.items():
        phase, days = f.get("phase_max"), f.get("days_to_next_catalyst")
        if phase is not None and phase >= PHASE2_RANK and days is not None:
            eligible.append((days, cid))
    return [cid for _, cid in sorted(eligible)]


def block_ranking(rows: Sequence[Dict[str, Any]], block: str, company_ids: Optional[set[int]] = None) -> List[int]:
    """
    One block's score alone, as the composite uses it: descending, except the subtracted friction
    term, which ranks ascending. Names without the block are left out, so the list can be short.
    """
    sign = 1.0 if block == FRICTION_TERM else -1.0
    scored: List[tuple[float, int]] = []
    for r in rows:
        value = (r.get("subscores") or {}).get("blocks", {}).get(block)
        if value is not None and (company_ids is None or r["company_id"] in company_ids):
            scored.append((sign * value, r["company_id"]))
    return [cid for _, cid in sorted(scored)]


def model_ranking(rows: Sequence[Dict[str, Any]], company_ids: Optional[set[int]] = None) -> List[int]:
    """
    The Phase 1 model's order: `composite` descending, unranked (None) names left out. With
    `company_ids` the ranking is restricted to that population (e.g. Tier 0).
    """
    scored = [
        r for r in rows
        if r.get("composite") is not None and (company_ids is None or r["company_id"] in company_ids)
    ]
    return [r["company_id"] for r in sorted(scored, key=lambda r: (-r["composite"], r["company_id"]))]
