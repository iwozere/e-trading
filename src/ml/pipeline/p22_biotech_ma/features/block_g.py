"""
P22 — Block G: Revealed Process Signals (spec §2.6, §4.7, §5.2, M5).

Categorically different from Blocks A-F (spec §4.7's own framing): those
infer *latent* attractiveness from fundamentals, while Block G captures
*revealed* process — a sale already underway or contractually pre-arranged.
Accordingly this module does NOT use the `@register_feature` /
`(company_id, as_of, ctx) -> float | None` contract every other block uses
— `BlockG` is a structured snapshot (state machine + activist + partnership
fields), and `apply_process_tier` folds it into the composite via TIERING
(spec §5.2), never a weighted sum: "a 0.05 weight would dissolve them."

**Verification gate, enforced by construction, not by convention.**
`build_block_g` reads exclusively through `FeatureContext.get_verified_*`
(`features/context.py`), and every one of those already filters
`is_verified = TRUE` (or, for `activist_position`, which spec's own schema
gives no such column — a real SEC 13D/G filing IS the verification) AND
`known_from <= as_of` at the repo layer (spec §4.7's bitemporal caution).
`build_block_g` itself has no way to see an unverified or not-yet-known row.

**Status as of 2026-09-08**: `process_state`/`process_scope`/
`days_since_process_open` are real today (`ingest/process_events.py` +
`jobs/run_process_events_ingest.py` are live) once a candidate clears
review. `has_13d_activist`/`activist_intent_max`/`activist_escalation`/
`has_strategic_toehold`/`strategic_toehold_pct` and
`partner_structure_max`/`partner_equity_pct`/`partner_identity` correctly
default to "none seen" for every company today — `activist_position`/
`partnership_structure` ingest (spec §2.6.2/§2.6.3) isn't built yet (tracked
in `docs/Tasks.md` as follow-on M5 work), not a bug in this module.
`apply_process_tier` is a direct, tested port of spec's own §5.2 pseudocode.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext

# spec §5.2's tiering order for stated_intent (weakest to strongest).
_INTENT_RANK = {"passive": 0, "engagement": 1, "board_seats": 2, "sale_demand": 3}
# spec §5.2's tiering order for structure_type (weakest to strongest) — matches Tier
# 2 (rofn_rofr, equity_plus_commercial) vs. Tier 3 (acquisition_option) exactly.
_STRUCTURE_RANK = {"license_only": 0, "equity_plus_commercial": 1, "rofn_rofr": 1, "acquisition_option": 2}


@dataclass(frozen=True)
class BlockG:
    """One company's Block G snapshot, as of one `as_of` date (spec §4.7's feature table)."""

    process_state: str = "none"  # none|rumored|disclosed_open|concluded_deal|concluded_no_deal
    process_scope: Optional[str] = None  # whole_company|asset_only|unclear
    days_since_process_open: Optional[int] = None
    has_13d_activist: bool = False
    activist_intent_max: Optional[str] = None
    activist_escalation: int = 0
    has_strategic_toehold: bool = False
    strategic_toehold_pct: Optional[float] = None
    partner_structure_max: Optional[str] = None
    partner_equity_pct: Optional[float] = None
    partner_identity: Optional[int] = None  # company_id of the partner, spec §5.2's likely_acquirer override


def build_block_g(company_id: int, as_of: date, ctx: FeatureContext) -> BlockG:
    """Assemble one company's `BlockG` snapshot from the three verification-gated,
    lookahead-safe reads (`FeatureContext.get_verified_process_events`/
    `get_verified_activist_positions`/`get_verified_partnership_structures`)."""
    process_events = ctx.get_verified_process_events(company_id)
    process_state = "none"
    process_scope = None
    days_since_process_open = None
    if process_events:
        latest = process_events[0]  # already ordered event_date DESC by the repo method
        process_state = latest["state"]
        process_scope = latest["scope"]
        if process_state == "disclosed_open":
            days_since_process_open = (as_of - latest["event_date"]).days

    activist_positions = ctx.get_verified_activist_positions(company_id)
    has_13d_activist = any(
        p["filer_type"] == "activist" and p["form_type"] in ("SC 13D", "SC 13D/A") for p in activist_positions
    )
    activist_intents = [p["stated_intent"] for p in activist_positions if p.get("stated_intent")]
    activist_intent_max = (
        max(activist_intents, key=lambda i: _INTENT_RANK.get(i, -1)) if activist_intents else None
    )
    activist_escalation = sum(1 for p in activist_positions if p["form_type"] == "SC 13D/A")
    strategic_toeholds = [p for p in activist_positions if p["filer_type"] == "strategic_corporate"]
    has_strategic_toehold = bool(strategic_toeholds)
    strategic_toehold_pct = (
        max((p["pct_of_class"] for p in strategic_toeholds if p.get("pct_of_class") is not None), default=None)
        if strategic_toeholds
        else None
    )

    partnerships = ctx.get_verified_partnership_structures(company_id)
    partner_structure_max = None
    partner_equity_pct = None
    partner_identity = None
    if partnerships:
        strongest = max(partnerships, key=lambda s: _STRUCTURE_RANK.get(s["structure_type"], -1))
        partner_structure_max = strongest["structure_type"]
        partner_equity_pct = strongest.get("partner_equity_pct")
        partner_identity = strongest.get("partner_id")

    return BlockG(
        process_state=process_state,
        process_scope=process_scope,
        days_since_process_open=days_since_process_open,
        has_13d_activist=has_13d_activist,
        activist_intent_max=activist_intent_max,
        activist_escalation=activist_escalation,
        has_strategic_toehold=has_strategic_toehold,
        strategic_toehold_pct=strategic_toehold_pct,
        partner_structure_max=partner_structure_max,
        partner_equity_pct=partner_equity_pct,
        partner_identity=partner_identity,
    )


def apply_process_tier(fundamental: float, g: BlockG) -> float:
    """
    Tiers are disjoint bands; fundamental score orders WITHIN a tier (spec §5.2). Direct port of
    spec's own pseudocode — see that section for the rationale (Block G's conditional
    probabilities are "an order of magnitude above what fundamentals produce," so a linear weight
    would either dissolve or swamp the signal).
    """
    if g.partner_structure_max == "acquisition_option":
        return 3.0 + fundamental  # Tier 3 — contractually pre-arranged
    if (
        g.process_state == "disclosed_open"
        and g.process_scope == "whole_company"
        and g.days_since_process_open is not None
        and g.days_since_process_open < 365
    ):
        return 3.0 + fundamental  # Tier 3 — sale process underway
    if (
        g.activist_intent_max == "sale_demand"
        or g.has_strategic_toehold
        or g.partner_structure_max in ("rofn_rofr", "equity_plus_commercial")
    ):
        return 2.0 + fundamental  # Tier 2 — structural pressure or pre-positioning
    if g.has_13d_activist or g.process_state == "rumored" or g.process_scope == "asset_only":
        return 1.0 + fundamental  # Tier 1 — elevated
    return fundamental  # Tier 0 — fundamentals only
