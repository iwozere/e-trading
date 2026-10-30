"""
P22 — Phase 1 score run (spec §5, §3.2), M4, 2026-10-29.

Scores every eligible target as of one date and writes one `p22_score_run` plus a `p22_score` row per
target. Reads only through `FeatureContext` and the block feature functions, so the run sees exactly
what those functions see as of `as_of` (financial facts are `known_from`-gated; trials are not
bitemporal — a known gap, see `features/context.py`).

**Universe:** `p22_company` rows that are active, not in the acquirer universe (big pharma resolved
from DERA as `role='both'` is a buyer here, not a target), and not known to fail the $25M size floor
(`size_floor.evaluate_size_floor`; an unknown floor is kept and counted in the summary).

**Acquirers:** `p22_acquirers.yaml` entries whose `entry_date <= as_of < exit_date`, matched to
`p22_company` by CIK and then by ticker.

**What is stored:** `composite`/`rank_by_composite`, `tier`/`tier_reason`, block scores and coverage
in `subscores`, and per-term contributions plus the top acquirer pairs in `contributions`.
`p_deal_24m`, `expected_return_if_deal`, `expected_value` and `rank_by_expected_value` stay NULL:
they need the M7 return model and the M6 labels (spec §5.3). The spec makes `rank_by_expected_value`
the default view, so until M7 lands the only ranking that exists is `rank_by_composite`; there is
deliberately no bare `rank` column (spec §5.4).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features import block_a, block_b, block_c, block_e
from src.ml.pipeline.p22_biotech_ma.features.block_g import build_block_g, process_tier
from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext
from src.ml.pipeline.p22_biotech_ma.ingest.acquirer_config import AcquirerConfig
from src.ml.pipeline.p22_biotech_ma.ingest.size_floor import evaluate_size_floor
from src.ml.pipeline.p22_biotech_ma.scoring.composite import block_scores, fundamental_score, rank_desc, rank_percentile
from src.ml.pipeline.p22_biotech_ma.scoring.fit import (
    UNCLASSIFIED_MODALITY,
    AcquirerProfile,
    FitResult,
    TargetProfile,
    best_fits,
    normalize_indication,
)
from src.ml.pipeline.p22_biotech_ma.scoring.weights_config import ScoringConfig
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_PHASE3_RANK = block_b.PHASE_RANK_LABELS.index("PHASE3")

_TARGET_FEATURES = {
    "target_asset_quality": {
        "phase_max": block_b.phase_max,
        "catalyst_window": block_b.catalyst_window,
        "asset_count_ph2plus": block_b.asset_count_ph2plus,
    },
    "financial_attractiveness": {
        "ev_to_cash": block_c.ev_to_cash,
        "size_band": block_c.size_band,
    },
    "friction_penalty": {
        "is_foreign_domiciled": block_e.is_foreign_domiciled,
    },
}
_ACQUIRER_FEATURES = {
    "revenue_at_risk_3y": block_a.revenue_at_risk_3y,
    "revenue_at_risk_5y": block_a.revenue_at_risk_5y,
}


@dataclass(frozen=True)
class ScoreRunResult:
    """Scored rows (p22_score columns plus `ticker`/`name` for reports) and a run summary."""

    rows: List[Dict[str, Any]]
    summary: Dict[str, Any]


def resolve_acquirers(
    acquirers: Sequence[AcquirerConfig], companies: Sequence[Dict[str, Any]], as_of: date
) -> List[tuple[AcquirerConfig, int]]:
    """Acquirers in the universe on `as_of`, matched to `company_id` by CIK, then ticker. Unmatched are logged."""
    by_cik = {c["cik"]: c["company_id"] for c in companies if c.get("cik")}
    by_ticker = {c["ticker"]: c["company_id"] for c in companies if c.get("ticker")}
    resolved: List[tuple[AcquirerConfig, int]] = []
    for acquirer in acquirers:
        if acquirer.entry_date > as_of or (acquirer.exit_date is not None and as_of >= acquirer.exit_date):
            continue
        company_id = by_cik.get(acquirer.cik.zfill(10)) if acquirer.cik else None
        if company_id is None:
            company_id = by_ticker.get(acquirer.ticker)
        if company_id is None:
            _logger.warning("Acquirer %s (%s) has no p22_company row — run the acquirer roster load", acquirer.name, acquirer.ticker)
            continue
        resolved.append((acquirer, company_id))
    return resolved


def _late_stage_asset_ids(trials: List[Dict[str, Any]]) -> set[int]:
    ids = set()
    for trial in trials:
        phases = [block_b.PHASE_RANK_LABELS.index(p) for p in (trial.get("phase") or "").split("/") if p in block_b.PHASE_RANK_LABELS]
        if phases and max(phases) >= _PHASE3_RANK and trial.get("asset_id") is not None:
            ids.add(trial["asset_id"])
    return ids


def build_acquirer_profile(repo: Any, ctx: FeatureContext, acquirer: AcquirerConfig, company_id: int) -> AcquirerProfile:
    """Assemble one acquirer's `fit()` inputs from the store."""
    assets = repo.get_assets_for_company(company_id)
    late_stage_ids = _late_stage_asset_ids(ctx.get_trials_for_company(company_id))
    return AcquirerProfile(
        company_id=company_id,
        name=acquirer.name,
        bloc=acquirer.bloc,
        dry_powder=block_a.dry_powder(company_id, ctx.as_of, ctx),
        pipeline_gap_by_ta=block_a.pipeline_gap_by_ta(company_id, ctx.as_of, ctx),
        modalities=frozenset(a["modality"] for a in assets if a.get("modality") not in (None, UNCLASSIFIED_MODALITY)),
        late_stage_indications=frozenset(
            ind for a in assets if a["asset_id"] in late_stage_ids and (ind := normalize_indication(a.get("indication")))
        ),
    )


def build_target_profile(repo: Any, ctx: FeatureContext, company_id: int) -> TargetProfile:
    """Assemble one target's lead-asset `fit()` inputs (lead asset = furthest-progressed, see block_b)."""
    lead = block_b.lead_asset(ctx.get_trials_for_company(company_id))
    asset: Dict[str, Any] = {}
    if lead is not None and lead.get("asset_id") is not None:
        asset = next((a for a in repo.get_assets_for_company(company_id) if a["asset_id"] == lead["asset_id"]), {})
    return TargetProfile(
        company_id=company_id,
        enterprise_value=block_c.enterprise_value(company_id, ctx.as_of, ctx),
        lead_therapeutic_area=lead.get("therapeutic_area") if lead else None,
        lead_modality=asset.get("modality"),
        lead_indication=asset.get("indication"),
    )


def select_targets(repo: Any, companies: Sequence[Dict[str, Any]], acquirer_ids: set[int], as_of: date) -> tuple[List[int], Dict[str, int]]:
    """The scoring universe (see module docstring) and a count of why companies were left out."""
    excluded = {"inactive": 0, "acquirer": 0, "below_size_floor": 0, "size_floor_unknown_kept": 0}
    targets: List[int] = []
    for company in companies:
        company_id = company["company_id"]
        if company_id in acquirer_ids or company.get("role") == "acquirer":
            excluded["acquirer"] += 1
            continue
        if company.get("is_active") is not True:
            excluded["inactive"] += 1
            continue
        decision = evaluate_size_floor(repo, company_id, as_of)
        if decision.met is False:
            excluded["below_size_floor"] += 1
            continue
        if decision.met is None:
            excluded["size_floor_unknown_kept"] += 1
        targets.append(company_id)
    return targets, excluded


def score_universe(
    repo: Any,
    as_of: date,
    config: ScoringConfig,
    acquirers: Sequence[AcquirerConfig],
    target_ids: Optional[Sequence[int]] = None,
) -> ScoreRunResult:
    """
    Score every target as of `as_of`. Does not write anything (see `persist_score_run`).

    Args:
        repo: A `P22Repo`-shaped object.
        as_of: Score date; every read is bounded by it.
        config: Parsed weights.
        acquirers: Parsed `p22_acquirers.yaml`.
        target_ids: Restrict to these companies (skips universe selection) — for spot checks and tests.
    """
    ctx = FeatureContext(as_of=as_of, repo=repo)
    companies = repo.list_companies_full()
    company_by_id = {c["company_id"]: c for c in companies}

    resolved = resolve_acquirers(acquirers, companies, as_of)
    acquirer_profiles = [build_acquirer_profile(repo, ctx, cfg, cid) for cfg, cid in resolved]
    acquirer_ids = {cid for _, cid in resolved}

    excluded: Dict[str, int] = {}
    if target_ids is None:
        targets, excluded = select_targets(repo, companies, acquirer_ids, as_of)
    else:
        targets = list(target_ids)

    # Block A per acquirer, rank-percentiled across the acquirer universe.
    acquirer_features = {cid: {name: fn(cid, as_of, ctx) for name, fn in _ACQUIRER_FEATURES.items()} for cid in acquirer_ids}
    pressure_by_acquirer = block_scores(acquirer_features, config.blocks.get("acquirer_pressure", {}))

    raw_features: Dict[int, Dict[str, Optional[float]]] = {}
    block_inputs: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {block: {} for block in _TARGET_FEATURES}
    fits: Dict[int, List[FitResult]] = {}
    likely: Dict[int, Optional[Dict[str, Any]]] = {}
    strategic_fit_raw: Dict[int, Optional[float]] = {}
    pressure_raw: Dict[int, Optional[float]] = {}
    tiers: Dict[int, tuple[int, str]] = {}
    dilution: Dict[int, Optional[float]] = {}

    for company_id in targets:
        features: Dict[str, Optional[float]] = {}
        for block, fns in _TARGET_FEATURES.items():
            values = {name: fn(company_id, as_of, ctx) for name, fn in fns.items()}
            block_inputs[block][company_id] = values
            features.update(values)
        dilution[company_id] = block_c.dilution_risk(company_id, as_of, ctx)
        features["dilution_risk"] = dilution[company_id]
        raw_features[company_id] = features

        pairs = best_fits(acquirer_profiles, build_target_profile(repo, ctx, company_id), config)
        fits[company_id] = pairs
        top = pairs[0] if pairs and pairs[0].fit is not None else None
        strategic_fit_raw[company_id] = top.fit if top is not None else None

        g = build_block_g(company_id, as_of, ctx)
        tiers[company_id] = process_tier(g)
        # Spec §5.2: a disclosed partner overrides the fit argmax as likely_acquirer.
        if g.partner_identity is not None and g.partner_identity in acquirer_ids:
            likely_id, source = g.partner_identity, "partner_identity"
        elif top is not None:
            likely_id, source = top.acquirer_id, "fit_argmax"
        else:
            likely_id, source = None, None
        if likely_id is not None:
            likely[company_id] = {"company_id": likely_id, "name": company_by_id.get(likely_id, {}).get("name"), "source": source}
            pressure_raw[company_id] = pressure_by_acquirer.get(likely_id)
        else:
            likely[company_id] = None
            pressure_raw[company_id] = None

    blocks: Dict[str, Dict[int, Optional[float]]] = {
        block: block_scores(block_inputs[block], config.blocks.get(block, {})) for block in _TARGET_FEATURES
    }
    strategic_pct = rank_percentile(strategic_fit_raw)
    pressure_pct = rank_percentile(pressure_raw)
    blocks["strategic_fit"] = {cid: strategic_pct.get(cid) for cid in targets}
    blocks["acquirer_pressure_matched"] = {cid: pressure_pct.get(cid) for cid in targets}

    rows: List[Dict[str, Any]] = []
    composites: Dict[int, Optional[float]] = {}
    for company_id in targets:
        company_blocks = {block: scores.get(company_id) for block, scores in blocks.items()}
        fundamental = fundamental_score(company_blocks, dilution[company_id], config)
        tier, tier_reason = tiers[company_id]
        composite = tier + fundamental.value if fundamental.value is not None else None
        composites[company_id] = composite
        company = company_by_id.get(company_id, {})
        rows.append({
            "company_id": company_id,
            "ticker": company.get("ticker"),
            "name": company.get("name"),
            "composite": composite,
            "tier": tier,
            "tier_reason": tier_reason,
            "subscores": {
                "fundamental": fundamental.value,
                "blocks": company_blocks,
                "coverage": fundamental.coverage,
                "missing_blocks": fundamental.missing_blocks,
                "dilution_gate": fundamental.dilution_gate,
                "notes": fundamental.notes,
                "likely_acquirer": likely[company_id],
                "strategic_fit_raw": strategic_fit_raw[company_id],
                "features": raw_features[company_id],
            },
            "contributions": {
                "composite_terms": fundamental.contributions,
                "top_pairs": [p.as_json() for p in fits[company_id][: config.top_pairs]],
            },
        })

    ranks = rank_desc(composites)
    for row in rows:
        row["rank_by_composite"] = ranks.get(row["company_id"])
    rows.sort(key=lambda r: (r["rank_by_composite"] is None, r["rank_by_composite"] or 0, r["company_id"]))

    summary = {
        "as_of": as_of.isoformat(),
        "acquirers": len(acquirer_profiles),
        "targets": len(targets),
        "ranked": len(ranks),
        "unranked_no_fundamental": len(targets) - len(ranks),
        "by_tier": {t: sum(1 for r in rows if r["tier"] == t) for t in range(4)},
        "block_coverage": {block: sum(1 for v in scores.values() if v is not None) for block, scores in blocks.items()},
        "excluded": excluded,
    }
    _logger.info("Scored universe as of %s: %s", as_of, summary)
    return ScoreRunResult(rows=rows, summary=summary)


_SCORE_COLUMNS = ("company_id", "composite", "rank_by_composite", "tier", "tier_reason", "subscores", "contributions")


def persist_score_run(repo: Any, as_of: date, config: ScoringConfig, config_hash: str, rows: List[Dict[str, Any]]) -> int:
    """Write one `p22_score_run` and its `p22_score` rows. Returns the `run_id`."""
    run_id = repo.create_score_run(as_of_date=as_of, model_version=config.model_version, config_hash=config_hash)
    repo.insert_scores(run_id, [{column: row[column] for column in _SCORE_COLUMNS} for row in rows])
    _logger.info("Persisted score run %s (%d scores, config_hash=%s)", run_id, len(rows), config_hash)
    return run_id
