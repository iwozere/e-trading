"""
P22 — Block E: Feasibility Frictions (spec §4.5, M3), 2026-09-08.

Spec's own invariant: "every feature in this block must be single-valued per company" — no
acquirer dimension (that's `fit()`'s job, §4.4). `is_foreign_domiciled` is the only feature
implemented this pass; see module docstring below and `docs/Tasks.md` for why the rest
(`has_poison_pill`, `staggered_board`, `dual_class_shares`, `has_controlling_holder`,
`recent_failed_process`, `royalty_encumbrance`) aren't attempted — all six need governance-document
(DEF 14A/charter) or 8-K text-parsing infrastructure that doesn't exist, and unlike
`process_events.py`'s strategic-alternatives phrases, spec gives no ready-made phrase list for any
of them — inventing keyword heuristics for a "poison pill" or "dual-class shares" detector without
one would be exactly the kind of fabricated business logic this codebase's discipline forbids.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext
from src.ml.pipeline.p22_biotech_ma.features.registry import register_feature


@register_feature("block_e.is_foreign_domiciled")
def is_foreign_domiciled(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """
    Target-side domicile friction (spec §4.5): inversion mechanics, cross-border tender-offer
    complexity. **Not** foreign-investment/CFIUS screening — that's the acquirer-side, pairwise
    gate in Block D (spec §4.4.1). Sourced from `ingest/domicile_normalization.py`, normalizing
    SEC submissions' own business-address data (live-verified against real foreign and domestic
    filers — see that module's docstring for a real EDGAR data-quality gap it works around).
    """
    del as_of
    return ctx.get_latest_fact(company_id, "is_foreign_domiciled")
