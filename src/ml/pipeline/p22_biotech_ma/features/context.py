"""
P22 — feature-computation context (spec §4).

Every feature function has the signature spec §4 mandates:

    def feature(company_id: int, as_of: date, ctx: FeatureContext) -> float | None

`FeatureContext` is the one object every feature function reads the
bitemporal store through — it exists so a feature function never touches
`P22Repo`/SQLAlchemy directly (keeping feature logic pure and testable
against a fake `repo`) and so lookahead safety is enforced in ONE place:
every read here goes through `P22Repo.get_financial_facts_as_of`, which
already filters on `known_from <= as_of` before a feature function ever sees
a value — a feature function has no way to accidentally see the future.

Returning `None` is meaningful (spec §4: "must propagate as missing, never
as zero") — `get_latest_fact` returns `None` whenever nothing is known yet,
never a stand-in zero. Every Block C function (`features/block_c.py`) relies
on this: reading a metric that hasn't been normalized into the store yet
(e.g. `market_cap`, blocked on the vendor decision) returns `None`
automatically, with no special-casing needed in the feature function itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FeatureContext:
    """Bound to one `as_of` date. `repo` is a `P22Repo`-shaped object (duck-typed for test doubles)."""

    as_of: date
    repo: Any

    def get_latest_fact(self, company_id: int, metric: str) -> Optional[float]:
        """
        The most recently *known* value of `metric` for `company_id`, as of
        `self.as_of` — i.e. the value a backtest snapshot taken on `as_of`
        would actually have seen. `None` if nothing is known yet.
        """
        row = self.get_latest_fact_row(company_id, metric)
        if row is None or row.get("value") is None:
            return None
        return float(row["value"])

    def get_latest_fact_row(self, company_id: int, metric: str) -> Optional[Dict[str, Any]]:
        """Like `get_latest_fact`, but returns the full fact row (value, period_end, known_from, ...)."""
        facts = self.repo.get_financial_facts_as_of(company_id, metric, self.as_of)
        return facts[0] if facts else None

    def get_trailing_average(self, company_id: int, metric: str, periods: int = 4) -> Optional[float]:
        """
        Average of the most recent `periods` known values of `metric`, as of
        `self.as_of` (spec §4.3: `cash_runway_months` uses a "trailing-4Q
        average quarterly operating burn"). Reuses the same lookahead-safe
        read as `get_latest_fact` (`P22Repo.get_financial_facts_as_of`,
        already ordered by `known_from` descending) and just averages more
        than one row instead of taking the first.

        Averages over however many periods ARE known if fewer than `periods`
        exist (e.g. a company with only 2 quarters of history) rather than
        requiring a full window — a partial-window average is still
        meaningful, and requiring the full window would delay every young
        company's derived features by a year for no reason. `None` only if
        nothing is known at all.
        """
        rows = self.repo.get_financial_facts_as_of(company_id, metric, self.as_of)
        values = [float(r["value"]) for r in rows[:periods] if r.get("value") is not None]
        if not values:
            return None
        return sum(values) / len(values)
