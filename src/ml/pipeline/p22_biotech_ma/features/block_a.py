"""
P22 — Block A: Acquirer Pressure Index (spec §4.1, M3), 2026-09-08.

Computed per large-cap acquirer (`p22_company.role in ('acquirer', 'both')`),
annually and forward 5 years per spec — this module computes one `as_of`
snapshot at a time, same signature convention as every other feature block;
the annual/forward-5-year sweep is an orchestration concern for whatever
calls these, not something a single feature function does itself.

**Status as of 2026-09-08: every function here is spec-correct and unit-
tested (including the real-computation path, spec §8.1), but most return
`None` in production today** — this is the same "scaffolding ahead of the
blocking input" pattern Block C used successfully for `market_cap`
(`features/block_c.py`'s history): these start returning real values the
moment their inputs exist, with no change needed here.

**What's real today:**
- `existing_net_debt` (`cash_capacity`'s subtraction term) — `total_debt`
  and `cash_and_equivalents`/`short_term_investments` are all already
  normalized (`ingest/financial_facts.py`), so this leg needs no new data.
- `market_cap` (`equity_capacity`'s multiplicand) — see `ingest/market_cap.py`.
- `get_phase3_asset_count_by_ta` (`pipeline_gap_by_ta`'s subtracted term's
  count leg) — `p22_trial`/`p22_asset` already support this for
  single-intervention-trial-linked assets (`docs/Tasks.md` item 8).

**What's still blocked, and why (see `docs/Tasks.md` "Decisions needed" for
the newly-added items this pass surfaced):**
- `ebitda` — not normalized into `p22_financial_fact` by anything yet.
  **Live-verified 2026-09-08**: FMP's `/stable/ratios`/`/key-metrics`/
  `/analyst-estimates` endpoints (see `ingest/fmp_client.py`) would supply
  this (plus forward EPS estimates for `fwd_pe`), but the account's CURRENT
  tier 402s for 22 of the 25 Block A acquirers (only PFE, ABBV, JNJ work) —
  a much narrower entitlement than the historical-price gap item 1 already
  described. Not wired to a normalizer this pass: building one now would
  produce Block A output that's real for 3 acquirers and silently `None`
  for the other 22, which is worse than uniformly `None` (indistinguishable
  from "the feature isn't ready" vs. "this acquirer specifically has no
  data" — exactly the kind of misleading partial-coverage result this
  codebase's discipline avoids elsewhere, e.g. `base_rate_fallback`'s
  explicit flagging requirement, spec §4.2).
- `target_leverage_ratio` (`cash_capacity`) — spec gives no number; a real
  leverage-tolerance assumption per the M&A financing convention this
  screen models, needs a domain call, not a guess. `TARGET_LEVERAGE_RATIO`
  below is `None` until curated (same "don't fabricate a base rate"
  discipline as `p22_base_rates.yaml`'s null areas).
- `currency_quality` (`equity_capacity`) needs BOTH `fwd_pe` (blocked above)
  AND a `stability_factor(trailing 12m realized vol)` formula spec §4.1
  names but never defines mathematically — genuinely undefined, not merely
  unbuilt. `percentile_rank` below implements the unambiguous half
  (percentile rank of `fwd_pe` vs. a peer cohort) as real, tested, generic
  math, ready for whenever both gaps close; it is deliberately NOT wired
  into `equity_capacity` itself, which instead reads `currency_quality` as
  an already-computed fact — see that function's docstring for why a
  peer-cohort computation doesn't fit the single-company feature signature.
- `assumed_peak_sales` per therapeutic area (`pipeline_gap_by_ta`) — a
  business assumption spec never quantifies, same character as
  `target_leverage_ratio` above. `pipeline_gap_by_ta` takes it as an
  explicit parameter (default `{}`, meaning "nothing curated yet") rather
  than reading a config file no loader exists for yet (`docs/Tasks.md`:
  "no config-loader/config_hash mechanism built" — M4 scope).
- `revenue_at_risk_3y`/`revenue_at_risk_5y`'s numerator needs
  `p22_patent_expiry.ttm_revenue_usd`, always `None` today
  (`ingest/patent_expiry_normalization.py`'s deliberately-deferred
  classification gap) — unrelated to the vendor/curation gaps above.
- `deal_cadence_3y`/`stock_deal_propensity` are NOT implemented here at
  all (not even as an always-`None` stub) — both read `p22_deal`, a table
  that doesn't exist until M6. A stub function reading a nonexistent table
  would be pure theater; tracked in `docs/Tasks.md` instead, same treatment
  as Blocks D/E/F.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext
from src.ml.pipeline.p22_biotech_ma.features.registry import register_feature

# Spec-given default (§4.1: "~15% share issuance before boards balk") — not a curated business
# assumption like TARGET_LEVERAGE_RATIO below, so no domain review is needed to use it as-is.
MAX_DILUTION_TOLERANCE = 0.15

# Spec names this input (§4.1's cash_capacity formula) but gives no value — a real M&A-financing
# leverage-tolerance assumption needs a domain call, not a guess (docs/Tasks.md "Decisions needed").
# `cash_capacity` returns None whenever this is None, exactly like a null p22_base_rates.yaml entry.
TARGET_LEVERAGE_RATIO: Optional[float] = None


def _revenue_at_risk(company_id: int, as_of: date, ctx: FeatureContext, window_months: int) -> Optional[float]:
    """
    Shared implementation for `revenue_at_risk_3y`/`_5y` (spec §4.1): sum of TTM revenue of
    products with `loe_date` within `window_months` of `as_of`, divided by the acquirer's total
    TTM revenue.

    Returns `None`, never a falsely-precise partial sum, if:
      - `total_ttm_revenue` isn't known (the denominator; not normalized anywhere yet, see module
        docstring) — makes the whole ratio undefined regardless of the numerator, or
      - any patent within the window has an unknown `ttm_revenue_usd` — a numerator computed by
        silently treating unknown per-product revenue as zero would UNDERSTATE risk while looking
        like a real, precise number. Every `p22_patent_expiry` row's `ttm_revenue_usd` is `None`
        today (see module docstring), so this always returns `None` in production for now.
    """
    total_ttm_revenue = ctx.get_latest_fact(company_id, "total_ttm_revenue")
    if total_ttm_revenue is None or total_ttm_revenue <= 0:
        return None

    patents = ctx.repo.get_patent_expiries_for_acquirer(company_id)
    window_end = date(as_of.year, as_of.month, as_of.day)
    # Simple month-based window (no calendar-library dependency for this): step forward
    # `window_months` months from as_of, clamping the day to stay a valid date.
    total_add_months = window_end.month - 1 + window_months
    window_end = date(
        window_end.year + total_add_months // 12,
        total_add_months % 12 + 1,
        min(window_end.day, 28),  # avoids Feb 30-style overflow; loe_date-month granularity is what matters
    )

    at_risk_revenue = 0.0
    for patent in patents:
        loe_date = patent.get("loe_date")
        if loe_date is None or not (as_of <= loe_date <= window_end):
            continue
        ttm_revenue_usd = patent.get("ttm_revenue_usd")
        if ttm_revenue_usd is None:
            return None  # an in-window product with unknown revenue makes the numerator unreliable
        at_risk_revenue += ttm_revenue_usd

    return at_risk_revenue / total_ttm_revenue


@register_feature("block_a.revenue_at_risk_3y")
def revenue_at_risk_3y(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """Share of TTM revenue at risk from patent/exclusivity expiry within 36 months (spec §4.1)."""
    return _revenue_at_risk(company_id, as_of, ctx, window_months=36)


@register_feature("block_a.revenue_at_risk_5y")
def revenue_at_risk_5y(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """Share of TTM revenue at risk from patent/exclusivity expiry within 60 months (spec §4.1)."""
    return _revenue_at_risk(company_id, as_of, ctx, window_months=60)


@register_feature("block_a.cash_capacity")
def cash_capacity(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """
    `(cash + marketable_securities) + (target_leverage_ratio × EBITDA − existing_net_debt)`,
    floored at 0 (spec §4.1). `existing_net_debt = total_debt − cash − marketable_securities`
    (standard definition) — real today, since every one of its inputs is already normalized.
    `ebitda` and `TARGET_LEVERAGE_RATIO` are not (see module docstring); either missing makes the
    whole expression undefined, not partially computable.
    """
    del as_of  # part of the shared feature-function signature (spec §4); this metric has no as-of branching
    cash = ctx.get_latest_fact(company_id, "cash_and_equivalents")
    ebitda = ctx.get_latest_fact(company_id, "ebitda")
    if cash is None or ebitda is None or TARGET_LEVERAGE_RATIO is None:
        return None

    marketable_securities = ctx.get_latest_fact(company_id, "short_term_investments") or 0.0
    total_debt = ctx.get_latest_fact(company_id, "total_debt") or 0.0
    existing_net_debt = total_debt - cash - marketable_securities

    capacity = (cash + marketable_securities) + (TARGET_LEVERAGE_RATIO * ebitda - existing_net_debt)
    return max(0.0, capacity)


@register_feature("block_a.equity_capacity")
def equity_capacity(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """
    `market_cap × max_dilution_tolerance × currency_quality` (spec §4.1). `market_cap` is real
    (`ingest/market_cap.py`); `currency_quality` is read as an already-computed fact rather than
    derived here — see module docstring for why (needs a peer-cohort pass + an undefined
    `stability_factor` formula, neither of which fits this function's single-company signature).
    Always `None` in production today until both of `currency_quality`'s own inputs exist.
    """
    del as_of
    market_cap = ctx.get_latest_fact(company_id, "market_cap")
    currency_quality = ctx.get_latest_fact(company_id, "currency_quality")
    if market_cap is None or currency_quality is None:
        return None
    return market_cap * MAX_DILUTION_TOLERANCE * currency_quality


@register_feature("block_a.dry_powder")
def dry_powder(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """`cash_capacity + equity_capacity` (spec §4.1). `None` if either leg is `None` — a partial
    sum (e.g. cash_capacity alone) would understate true dry powder, not a safe lower bound to
    report as-is without saying so."""
    cc = cash_capacity(company_id, as_of, ctx)
    ec = equity_capacity(company_id, as_of, ctx)
    if cc is None or ec is None:
        return None
    return cc + ec


def percentile_rank(value: float, peers: List[float]) -> Optional[float]:
    """
    Percentile rank of `value` within `peers` (inclusive of `value` itself if already present in
    the list, exclusive otherwise), in `[0, 1]`. The unambiguous half of `currency_quality`'s
    `percentile_rank(acquirer.fwd_pe vs peer group)` term (spec §4.1) — real, generic, and
    independent of the still-undefined `stability_factor` half (see module docstring). Not wired
    into `equity_capacity`/`currency_quality` yet: computing this for real needs one pass over
    every acquirer's `fwd_pe` first (a cohort-level computation, not a per-company one), which has
    no caller until `fwd_pe` itself is normalized (see module docstring's FMP entitlement finding).

    Returns:
        `None` if `peers` is empty (no peer group to rank against).
    """
    if not peers:
        return None
    rank = sum(1 for p in peers if p <= value)
    return rank / len(peers)


def pipeline_gap_by_ta(
    company_id: int,
    as_of: date,
    ctx: FeatureContext,
    assumed_peak_sales_by_ta: Optional[Dict[str, float]] = None,
) -> Dict[str, Optional[float]]:
    """
    Per therapeutic area: `revenue_at_risk in that TA − (count of own Phase III assets in that TA
    × assumed_peak_sales)` (spec §4.1) — "the key output... which therapeutic areas does this
    acquirer need to buy into?"

    **Deliberately NOT a `@register_feature`**: spec's own table already marks this "Per TA," i.e.
    one value per therapeutic area, not the single `float | None` every other feature function
    returns (spec §4). Forcing it into that signature would mean either picking one TA arbitrarily
    or flattening real per-TA structure the dossier needs to show — same class of spec/infra
    mismatch already called out for Block D's pairwise `fit()` (`docs/pipeline-specification.md`
    §4.4). Callers needing this (M4's scoring layer) should call it directly, keyed by company.

    `assumed_peak_sales_by_ta` is a genuine business assumption spec never quantifies (same
    character as `TARGET_LEVERAGE_RATIO` above) — taken as an explicit parameter (default `{}`)
    rather than read from a config file, since no config-loader mechanism exists yet for anything
    in `config/pipeline/*.yaml` (`docs/Tasks.md`; that's M4 scoring infra).

    Returns:
        `{therapeutic_area: gap_or_None}` for every TA with at least one Phase III asset on file
        for this company — a TA the acquirer has no Phase III presence in at all isn't "gapped,"
        it's simply not represented here (spec doesn't define a gap value for a zero-count TA, and
        inventing one would be a guess, not a derivation).
    """
    del as_of  # part of the shared feature-function signature (spec §4); ctx is already as-of-bound
    assumed_peak_sales_by_ta = assumed_peak_sales_by_ta or {}
    phase3_counts = ctx.get_phase3_asset_count_by_ta(company_id)

    # `_revenue_at_risk`'s per-TA breakdown isn't built (its own numerator/denominator are both
    # unavailable today — see module docstring); read the same not-yet-normalized per-TA fact name
    # by convention so this starts working the moment that upstream piece exists.
    result: Dict[str, Optional[float]] = {}
    for therapeutic_area, count in phase3_counts.items():
        revenue_at_risk_in_ta = ctx.get_latest_fact(company_id, f"revenue_at_risk_ta:{therapeutic_area}")
        peak_sales = assumed_peak_sales_by_ta.get(therapeutic_area)
        if revenue_at_risk_in_ta is None or peak_sales is None:
            result[therapeutic_area] = None
            continue
        result[therapeutic_area] = revenue_at_risk_in_ta - (count * peak_sales)

    return result
