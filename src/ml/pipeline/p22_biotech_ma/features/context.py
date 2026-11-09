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

import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import PUBLIC_FLOAT_MAX_VALUE_USD, PUBLIC_FLOAT_TO_MARKET_CAP
from src.ml.pipeline.p22_biotech_ma.ingest.market_cap import compute_market_cap
from src.ml.pipeline.p22_biotech_ma.ingest.price_quality import latest_public_float


@dataclass(frozen=True)
class FeatureContext:
    """Bound to one `as_of` date. `repo` is a `P22Repo`-shaped object (duck-typed for test doubles)."""

    as_of: date
    repo: Any
    _market_caps: Dict[int, Optional[float]] = field(default_factory=dict, compare=False, repr=False)

    def get_market_cap(self, company_id: int) -> Optional[float]:
        """
        Market cap as of `self.as_of`, derived from the raw close and share count known then
        (`ingest.market_cap.compute_market_cap`), not read from stored `market_cap` facts.

        The stored facts carry `known_from` = the day the daily job ran, so a backtest snapshot
        before the job existed (2026-09) saw none and Block C scored nobody in 2016-2024.
        Deriving here gives live and historical runs the same definition and the same guards
        (stale price, stale share count, splits); both inputs are read as of `self.as_of`.
        Cached per company because Block C asks for it several times per company.

        Without a usable price or share count, falls back to the latest 10-K public float known on
        `as_of` (at most `PUBLIC_FLOAT_MAX_AGE_DAYS` old) times `PUBLIC_FLOAT_TO_MARKET_CAP`. The price
        history mostly lacks companies that later delisted, so leaving them unknown made "no market
        cap" a marker of the names about to be acquired (2026-11-09 backtest: 13% vs 6.5%).
        """
        if company_id not in self._market_caps:
            result = compute_market_cap(self.repo, company_id, self.as_of)[0]
            self._market_caps[company_id] = result.market_cap if result is not None else self._float_market_cap(company_id)
        return self._market_caps[company_id]

    def _float_market_cap(self, company_id: int) -> Optional[float]:
        public_float = latest_public_float(self.repo, company_id, self.as_of)
        if public_float is None or public_float.value > PUBLIC_FLOAT_MAX_VALUE_USD:
            return None
        return public_float.value * PUBLIC_FLOAT_TO_MARKET_CAP

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

    def get_phase3_asset_count_by_ta(self, company_id: int) -> Dict[str, int]:
        """
        Pass-through to `P22Repo.count_phase3_assets_by_therapeutic_area` — kept here (rather than
        called directly on `ctx.repo`) so `features/block_a.py`'s `pipeline_gap_by_ta` never touches
        the repo/SQLAlchemy layer directly, same discipline as every other feature-function read
        (module docstring above). Counts only trials public by `as_of` (first posted, or started);
        each trial's phase is its current one, since `p22_trial` is a plain upsert, not bitemporal.
        """
        return self.repo.count_phase3_assets_by_therapeutic_area(company_id, as_of=self.as_of)

    def get_trials_for_company(self, company_id: int) -> List[Dict[str, Any]]:
        """Pass-through to `P22Repo.get_trials_for_company` — spec §4.2's Block B, read by
        `features/block_b.py`. Only trials public by `as_of` (first posted, or started); their
        phase and primary completion date are the current ones (same caveat as
        `get_phase3_asset_count_by_ta`)."""
        return self.repo.get_trials_for_company(company_id, as_of=self.as_of)

    def get_verified_process_events(self, company_id: int) -> List[Dict[str, Any]]:
        """Pass-through to `P22Repo.get_verified_process_events`, bound to `self.as_of` — spec
        §4.7's Block G, read by `features/block_g.py`. Already lookahead-gated AND
        verification-gated by the repo method itself (see that method's docstring)."""
        return self.repo.get_verified_process_events(company_id, self.as_of)

    def get_verified_activist_positions(self, company_id: int) -> List[Dict[str, Any]]:
        """Pass-through to `P22Repo.get_verified_activist_positions`, bound to `self.as_of`."""
        return self.repo.get_verified_activist_positions(company_id, self.as_of)

    def get_verified_partnership_structures(self, company_id: int) -> List[Dict[str, Any]]:
        """Pass-through to `P22Repo.get_verified_partnership_structures`, bound to `self.as_of`."""
        return self.repo.get_verified_partnership_structures(company_id, self.as_of)

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
