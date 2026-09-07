"""
P22 — Block B: Target Asset Quality (spec §4.2, M3), 2026-09-08.

Reuses `FeatureContext.get_trials_for_company` for every feature here — all four need the
company's whole trial/asset portfolio at once, not a single fact lookup, so unlike Block A/C they
don't go through `get_latest_fact`.

**"Lead asset" is a proxy, not spec's real field.** Spec's features are named `lead_asset_poa`
etc., implying a designated lead program, but `p22_asset.is_lead` is always `None`
(`ingest/asset_normalization.py`'s disclosed gap — "needs company-level ... judgment," not built).
Absent that flag, this module treats the asset with the FURTHEST-PROGRESSED trial (highest phase
reached) as the lead-asset proxy — a defensible operational reading of "the company's most
advanced/important program," not a guess at which specific asset a human would pick, and disclosed
as a proxy everywhere it's used.

**Status as of 2026-09-08**: `phase_max` and `asset_count_ph2plus` are real today (only need
`p22_trial`/`p22_asset`, both already flowing). `catalyst_window` is real for the ordinary
forward-looking buckets; its `post_positive_0-180` bucket can never fire (needs a positive-readout
detection this repo doesn't have). `lead_asset_poa` is real ONLY for a lead asset still at
Phase 1 or earlier — `docs/Tasks.md` "Decisions needed" item 13 explains why later phases return
`None` rather than a guessed combination formula. `has_positive_ph3`, `pdufa_pending`,
`endpoint_stability`, `trial_design_quality`, and `ev_to_risk_adjusted_npv` are not implemented at
all this pass — each needs a data source or extraction step that doesn't exist yet (see each
function's docstring below and `docs/Tasks.md` for the full accounting); no stub functions are
added for them, matching Block A's "don't build a function that can only ever return None"
precedent for `deal_cadence_3y`/`stock_deal_propensity`.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext
from src.ml.pipeline.p22_biotech_ma.features.registry import register_feature
from src.ml.pipeline.p22_biotech_ma.ingest.base_rates_config import load_base_rates
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# CT.gov's phase vocabulary, ranked. A combined value ("PHASE2/PHASE3") ranks at its highest part.
PHASE_RANK_LABELS: List[str] = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]
_PHASE_RANK = {label: i for i, label in enumerate(PHASE_RANK_LABELS)}
_PHASE2_RANK = _PHASE_RANK["PHASE2"]
_PHASE1_MAX_RANK = _PHASE_RANK["PHASE1"]  # lead_asset_poa's "clean" case: EARLY_PHASE1 or PHASE1

# spec §4.2's catalyst_window buckets, ranked nearest-first. "post_positive_0-180" is omitted —
# see module docstring for why it can never be computed with current data.
CATALYST_WINDOW_LABELS: List[str] = ["<60", "60-180", "180-365", "365-730", ">730"]


def _phase_rank(phase: Optional[str]) -> Optional[int]:
    """Highest ranked part of a possibly-combined CT.gov phase string, or `None` if empty/unknown."""
    if not phase:
        return None
    ranks = [_PHASE_RANK[p] for p in phase.split("/") if p in _PHASE_RANK]
    return max(ranks) if ranks else None


def _lead_asset(trials: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The furthest-progressed asset's therapeutic_area + phase rank, or `None` if no trial has a
    recognizable phase. See module docstring for why this is a proxy for spec's `is_lead` field."""
    best: Optional[Dict[str, Any]] = None
    best_rank = -1
    for trial in trials:
        rank = _phase_rank(trial.get("phase"))
        if rank is None:
            continue
        if rank > best_rank:
            best_rank = rank
            best = trial
    if best is None:
        return None
    return {"therapeutic_area": best.get("therapeutic_area"), "phase_rank": best_rank}


@register_feature("block_b.phase_max")
def phase_max(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """Furthest phase reached by any asset (spec §4.2), encoded as a float ordinal —
    `PHASE_RANK_LABELS[int(phase_max)]` maps back to the label, same convention as
    `features/block_c.py`'s `size_band`. `None` if the company has no trial with a recognizable
    phase on file (not the same as `0.0`, which means "confirmed EARLY_PHASE1, nothing further")."""
    del as_of
    trials = ctx.get_trials_for_company(company_id)
    ranks = [r for r in (_phase_rank(t.get("phase")) for t in trials) if r is not None]
    return float(max(ranks)) if ranks else None


@register_feature("block_b.asset_count_ph2plus")
def asset_count_ph2plus(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """Count of distinct assets with at least one trial reaching Phase II or later (spec §4.2:
    "portfolio breadth — single-asset companies are riskier but also cheaper targets"). `0.0` is a
    real, meaningful answer (every asset is pre-Phase-II) distinct from `None` (no trial data on
    file for this company at all)."""
    del as_of
    trials = ctx.get_trials_for_company(company_id)
    if not trials:
        return None
    ph2plus_assets = {
        t["asset_id"] for t in trials if (_phase_rank(t.get("phase")) or -1) >= _PHASE2_RANK
    }
    return float(len(ph2plus_assets))


@register_feature("block_b.catalyst_window")
def catalyst_window(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """
    Days to the next value-inflecting readout, bucketed (spec §4.2), encoded as a float ordinal
    into `CATALYST_WINDOW_LABELS` (nearest-first: `0.0` = "<60", ... `4.0` = ">730"). Takes the
    NEAREST upcoming `primary_completion_date` across every trial for the company (any asset, any
    phase) — the next event that could move the name, not restricted to the lead-asset proxy.

    `None` if no trial has a future `primary_completion_date` on file. The `post_positive_0-180`
    bucket spec also names is never reachable here — it needs to know a readout was POSITIVE, and
    this repo has no results/outcome data (`hasResults`, topline direction) for any trial yet.
    """
    trials = ctx.get_trials_for_company(company_id)
    upcoming = [
        t["primary_completion_date"] for t in trials
        if t.get("primary_completion_date") is not None and t["primary_completion_date"] >= as_of
    ]
    if not upcoming:
        return None
    days = (min(upcoming) - as_of).days
    if days < 60:
        return 0.0
    if days < 180:
        return 1.0
    if days < 365:
        return 2.0
    if days < 730:
        return 3.0
    return 4.0


@register_feature("block_b.lead_asset_poa")
def lead_asset_poa(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """
    Probability of approval for the lead-asset proxy (spec §4.2: "base rate x modifiers, from
    phase + TA + biomarker + CRL history"). **Feeds the return model (§5.3), not the
    deal-probability composite** — see spec's own warning against folding raw PoA into `fit()`.

    Computed ONLY when the lead asset is still at Phase I or earlier: `p22_base_rates.yaml`'s
    `by_therapeutic_area` figures are LOA-FROM-PHASE-1 (i.e., already conditional on being no
    further than Phase 1) — applying that same number to an asset already in Phase 2/3 would be
    wrong (double-counts/ignores the progress already made), and spec doesn't specify how to
    recompose a later-phase estimate from the generic (non-TA-specific)
    `phase_2_success`/`phase_3_to_filing`/`filing_to_approval` rates alongside a TA-specific
    Phase-1 figure, nor how `orphan_by_phase` (itself a full alternate rate table, not a simple
    multiplier) is meant to combine with either — see `docs/Tasks.md` "Decisions needed" item 13.
    Returns `None` for any later-phase lead asset rather than guessing a combination.

    Falls back to `loa_from_phase_1_overall` (logged, per spec §4.2's `base_rate_fallback`
    requirement) when the lead asset's own therapeutic_area has no source-backed figure yet (9 of
    ~21 areas are still `null`, `p22_base_rates.yaml`'s own header explains why). The biomarker
    modifier is applied only when `uses_biomarker_selection` is known (always `None` today,
    `ingest/trial_normalization.py`'s disclosed gap) — capped at 1.0 since this is a probability.
    """
    del as_of
    trials = ctx.get_trials_for_company(company_id)
    lead = _lead_asset(trials)
    if lead is None or lead["phase_rank"] > _PHASE1_MAX_RANK:
        return None

    base_rates = load_base_rates()
    ta = lead["therapeutic_area"]
    rate = base_rates.by_therapeutic_area.get(ta)
    if rate is None:
        _logger.info(
            "lead_asset_poa: base_rate_fallback for company_id=%s (therapeutic_area=%r has no "
            "source-backed rate) — using loa_from_phase_1_overall=%s",
            company_id, ta, base_rates.loa_from_phase_1_overall,
        )
        rate = base_rates.loa_from_phase_1_overall

    # uses_biomarker_selection is always None today (ingest/trial_normalization.py) — the modifier
    # is simply never applied yet, not defaulted to "biomarker-selected" (that would inflate PoA).
    return min(1.0, rate)
