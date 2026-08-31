"""
P22 — feature data-quality checks (spec §8.2, M3).

Pandera schemas for the exact bounds spec §8.2 lists, plus one repo-level
check (`assert_every_company_has_a_verified_alias`) that isn't a per-row
column bound and so doesn't fit a `DataFrameSchema` — it's a set-membership
assertion across two tables instead.

Usage: build a one-row-per-(company_id, as_of) `pandas.DataFrame` from
computed features (Block C so far; more columns as later blocks land) and
call `validate()`. A validation failure raises `pandera.errors.SchemaError`
with the specific row/column that violated a bound — this is meant to run in
the same job/CI step that computes a batch of features, not as a standalone
pass over already-trusted data.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pandera.pandas as pa

# spec §8.2: "cash_runway_months ∈ [0, 120] or null"
CASH_RUNWAY_MONTHS_SCHEMA = pa.Column(float, pa.Check.in_range(0, 120), nullable=True, coerce=True)

# spec §8.2: "enterprise_value may be negative (legitimate); market cap may not" — EV itself has no
# stated bound (any real number, or null); the constraint is on market_cap, checked separately below.
ENTERPRISE_VALUE_SCHEMA = pa.Column(float, nullable=True, coerce=True)
MARKET_CAP_SCHEMA = pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True)

# spec §8.2: "lead_asset_poa ∈ [0, 1]"
LEAD_ASSET_POA_SCHEMA = pa.Column(float, pa.Check.in_range(0, 1), nullable=True, coerce=True)


def loe_date_schema(as_of: date) -> pa.Column:
    """
    spec §8.2: "loe_date ∈ [1990-01-01, as_of + 25 years]" — bound depends on
    `as_of`, so this is a function, not a module-level constant like the
    others above. Bounds are `pandas.Timestamp`, not `datetime.date` — a
    `pa.DateTime` column coerces to `datetime64[ns]`, which pandas refuses to
    compare against a plain `date` (raises `TypeError`, not just a failed
    check), so the bounds must already be timestamps.
    """
    lower = pd.Timestamp(1990, 1, 1)
    try:
        upper = pd.Timestamp(as_of.year + 25, as_of.month, as_of.day)
    except ValueError:  # as_of is Feb 29 and +25 years isn't a leap year
        upper = pd.Timestamp(as_of) + pd.Timedelta(days=25 * 365)
    return pa.Column(pa.DateTime, pa.Check.in_range(lower, upper), nullable=True, coerce=True)


def block_c_schema() -> pa.DataFrameSchema:
    """
    One row per (company_id, as_of). Columns match `features/block_c.py`'s
    registered feature names, minus the block prefix. Only the two bounds
    spec §8.2 states explicitly (`cash_runway_months`, `enterprise_value`)
    are covered here — `ev_to_cash`/`dilution_risk`/`atm_capacity_pct`/
    `size_band` have no stated bound in the spec and are left unvalidated
    rather than inventing one.
    """
    return pa.DataFrameSchema(
        {
            "cash_runway_months": CASH_RUNWAY_MONTHS_SCHEMA,
            "enterprise_value": ENTERPRISE_VALUE_SCHEMA,
        },
        strict=False,  # extra columns (company_id, as_of, other block-C features) are fine
    )


def assert_every_company_has_a_verified_alias(repo: Any) -> None:
    """
    spec §8.2: "No `company` row without at least one verified alias." Not a
    pandera column check (it's a join/set-membership constraint across
    `p22_company` and `p22_company_alias`, not a bound on one column's
    values), so it's a plain assertion function instead.

    Raises:
        AssertionError: listing every company_id with no verified alias, if any exist.
    """
    unaliased = repo.get_companies_without_verified_alias()
    if unaliased:
        raise AssertionError(
            f"{len(unaliased)} company row(s) have no verified alias (spec §8.2): {sorted(unaliased)[:20]}"
            + (" ..." if len(unaliased) > 20 else "")
        )
