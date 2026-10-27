"""
P22 — point-in-time size floor (market cap > $25M, spec §2.0.3/§2.0.6), 2026-10-27.

Decision order for one company on one `as_of`:

1. Derived market cap (`market_cap.compute_market_cap`), when it exists AND the
   company's price/share series passes `price_quality.check_float_consistency`
   on the newest known public float (or there is no float to check against):
   `met = market_cap >= floor`.
2. Otherwise, a public float no older than `PUBLIC_FLOAT_MAX_AGE_DAYS` that is
   itself >= the floor proves the floor (market cap >= non-affiliate float):
   `met = True`. This rescues exactly the cases the consistency check rejects —
   a corrupt price series (ACOR) and a float a filer tagged 1000x too large
   (Sarepta) both still pass correctly.
3. Otherwise `met = None` — unknown, never silently False. Same convention as
   `UniverseCandidate.eligible_exchange`: a company is not dropped for a reason
   the code could not actually check.

Known residual errors (cross-check, `docs/Design.md` "Size floor"): ~3% of
company-quarters whose float alone is above $25M still get a derived market
cap below it, the same rate as FMP's own historical market cap. Step 2 can
pass a company whose stock collapsed since its float date, but only when its
market-cap series is already unusable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import SIZE_FLOOR_USD
from src.ml.pipeline.p22_biotech_ma.ingest.market_cap import compute_market_cap
from src.ml.pipeline.p22_biotech_ma.ingest.price_quality import check_float_consistency, latest_public_float
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass(frozen=True)
class SizeFloorDecision:
    """
    Outcome of `evaluate_size_floor`.

    `basis` is "market_cap" or "public_float" when `met` is decided, otherwise the reason it is not
    (a `compute_market_cap` rejection reason, or "inconsistent_with_public_float").
    """

    met: Optional[bool]
    basis: str
    market_cap: Optional[float] = None
    public_float: Optional[float] = None


def evaluate_size_floor(repo: Any, company_id: int, as_of: date, floor_usd: float = SIZE_FLOOR_USD) -> SizeFloorDecision:
    """
    Decide whether `company_id` met the size floor on `as_of`, using only data known on `as_of`.

    Args:
        repo: A `P22Repo`-shaped object.
        company_id: The company.
        as_of: Decision date.
        floor_usd: Minimum market cap.
    """
    result, reason = compute_market_cap(repo, company_id, as_of)
    public_float = latest_public_float(repo, company_id, as_of)
    float_value = public_float.value if public_float is not None else None

    if result is not None:
        check = check_float_consistency(repo, company_id, public_float) if public_float is not None else None
        if check is None or check.consistent:
            return SizeFloorDecision(result.market_cap >= floor_usd, "market_cap", result.market_cap, float_value)
        reason = "inconsistent_with_public_float"

    market_cap = result.market_cap if result is not None else None
    if float_value is not None and float_value >= floor_usd:
        return SizeFloorDecision(True, "public_float", market_cap, float_value)
    assert reason is not None
    return SizeFloorDecision(None, reason, market_cap, float_value)


def make_size_floor_lookup(repo: Any, floor_usd: float = SIZE_FLOOR_USD) -> Callable[[str, date], Optional[bool]]:
    """
    Adapt `evaluate_size_floor` to the `(cik, as_of) -> Optional[bool]` callable
    `entity_resolution.build_universe_history` takes.

    A CIK with no `p22_company` row yields None (unknown). CIK -> company_id lookups are cached,
    since a history build asks about the same CIKs once per quarter.
    """
    company_ids: Dict[str, Optional[int]] = {}

    def lookup(cik: str, as_of: date) -> Optional[bool]:
        if cik not in company_ids:
            company = repo.get_company_by_cik(cik)
            company_ids[cik] = company["company_id"] if company is not None else None
        company_id = company_ids[cik]
        if company_id is None:
            return None
        return evaluate_size_floor(repo, company_id, as_of, floor_usd).met

    return lookup
