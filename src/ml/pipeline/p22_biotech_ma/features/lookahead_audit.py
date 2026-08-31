"""
P22 — lookahead audit (spec §8.3, "mandatory... any failure blocks the build").

This module builds the audit's **sampling and assertion logic** — pure,
DB-free, fully unit-tested against synthetic fixtures now, ready to wire
against real data once it exists.

**What is NOT done here, and why:** spec §8.3 requires the 200-sample audit
be *stratified to guarantee coverage of the three highest-risk sources* —
vendor-sourced facts, 13F holdings, and 13D/process events. As of 2026-08-30,
this repo has **zero** rows in any of those three categories: the
market-data vendor isn't selected yet (spec §2.4/§2.0.6), and 13F/13D
ingestion is Block F/G scope (M5/M6+), not built. Wiring
`stratified_sample`/`assert_lookahead_safe` against a real `(company, as_of)`
population drawn from the DB and running it in CI as spec's "blocks the
build" gate would be **vacuous** right now — a pass with nothing at risk to
catch is worse than no gate at all, because it would look like the safety
property is verified when it has never actually been exercised against the
categories that matter. Do this once vendor/13F/13D ingestion lands, not
before. Tracked in `docs/Tasks.md`.

The one general-purpose piece already usable today: `assert_lookahead_safe`
can run against `p22_financial_fact` rows right now (the SEC-sourced facts
that already exist) — it just can't yet fulfill the *stratified* coverage
requirement, since the three named risk categories have no data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The three categories spec §8.3 explicitly requires coverage of, plus a
# catch-all for everything else (e.g. SEC financial facts) that the sample
# should still include some of, just not be dominated by.
HIGH_RISK_CATEGORIES: Tuple[str, ...] = ("vendor_fact", "13f_holding", "13d_process_event")


@dataclass(frozen=True)
class AuditSample:
    """One `(company, as_of)` pair drawn for the audit, tagged with which source category it probes."""

    company_id: int
    as_of: date
    source_category: str


def stratified_sample(
    population: Sequence[AuditSample],
    *,
    total: int = 200,
    min_per_high_risk_category: int = 20,
    rng: Optional[random.Random] = None,
) -> List[AuditSample]:
    """
    Draw `total` samples from `population`, guaranteeing at least
    `min_per_high_risk_category` from each of `HIGH_RISK_CATEGORIES` present
    in the population (spec §8.3: "must be stratified... not drawn
    uniformly — uniform sampling over a universe dominated by SEC facts will
    rarely hit the cases that actually leak").

    Args:
        population: Every candidate `(company, as_of, source_category)` to
            draw from, e.g. one entry per fact row currently in the store.
        total: Target sample size (spec §8.3: 200).
        min_per_high_risk_category: Floor per high-risk category, provided
            the population has at least that many in that category — a
            category with fewer available samples than the floor
            contributes everything it has, not a padded/duplicated set.
        rng: Injected `random.Random` for deterministic tests; a fresh one otherwise.

    Returns:
        Up to `total` samples (fewer if the population itself is smaller).
    """
    rng = rng or random.Random()
    by_category: Dict[str, List[AuditSample]] = {}
    for item in population:
        by_category.setdefault(item.source_category, []).append(item)

    selected: List[AuditSample] = []
    selected_ids: set = set()

    for category in HIGH_RISK_CATEGORIES:
        candidates = by_category.get(category, [])
        take = min(min_per_high_risk_category, len(candidates))
        for item in rng.sample(candidates, take) if take else []:
            selected.append(item)
            selected_ids.add(id(item))

    remaining_slots = total - len(selected)
    if remaining_slots > 0:
        remaining_pool = [item for item in population if id(item) not in selected_ids]
        take = min(remaining_slots, len(remaining_pool))
        selected.extend(rng.sample(remaining_pool, take) if take else [])

    return selected[:total]


def assert_lookahead_safe(fact_rows: Sequence[Dict[str, Any]]) -> None:
    """
    spec §8.3's core assertion: every fact used in scoring must satisfy
    `known_from <= as_of`.

    Args:
        fact_rows: Each dict needs `known_from` (a `datetime`) and `as_of`
            (a `date`) — the caller is responsible for joining a sampled
            `AuditSample` to the actual fact row(s) read for that
            `(company, as_of)` pair before calling this.

    Raises:
        AssertionError: listing every violating row, if any exist. This is
            the "any failure blocks the build" gate — never soften this to a
            warning or a partial-pass.
    """
    violations = [row for row in fact_rows if row["known_from"].date() > row["as_of"]]
    if violations:
        raise AssertionError(
            f"{len(violations)} lookahead violation(s) — a fact's known_from is after its as_of "
            f"(spec §8.3, mandatory, blocks the build): {violations[:5]}"
            + (" ..." if len(violations) > 5 else "")
        )


def assert_known_from_is_filing_date_not_period_or_crossing_date(
    rows: Sequence[Dict[str, Any]], *, filing_date_field: str = "filed_date"
) -> None:
    """
    spec §8.3, sample categories 2 and 3 specifically: for 13F holdings and
    13D/process events, `known_from` must equal the **filing** date, never
    `period_end` (13F) or the beneficial-ownership crossing date (13D) —
    "A 45-day error here is invisible in output and fatal in backtest."

    Args:
        rows: Each dict needs `known_from` (`datetime`) and
            `filing_date_field` (`date`) — the row's actual SEC filing date,
            independent of whatever period/crossing date it also carries.
        filing_date_field: Name of the filing-date field to compare against
            (rows from different sources may name it differently upstream;
            the caller normalizes before calling this).

    Raises:
        AssertionError: listing every row whose `known_from` doesn't match its filing date.
    """
    violations = [row for row in rows if row["known_from"].date() != row[filing_date_field]]
    if violations:
        raise AssertionError(
            f"{len(violations)} row(s) have known_from != {filing_date_field} (spec §8.3: must be the "
            f"filing date, never period_end or the crossing date): {violations[:5]}"
            + (" ..." if len(violations) > 5 else "")
        )
