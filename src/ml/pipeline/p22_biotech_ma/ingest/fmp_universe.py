"""
P22 — target-ticker universe for the FMP historical bulk backfill (spec
§2.0.6/§2.4, docs/Tasks.md item 1, M3, 2026-08-31).

Splits the universe by what we already know, because FMP's API is
ticker-keyed and `p22_company.ticker` is only reliably populated for
companies resolved off the CURRENT SEC ticker snapshot (a known limitation —
see `ingest/entity_resolution.py`'s docstring on historical ticker
resolution). Concretely:

- **`build_known_universe`** — every `p22_company` row that already has a
  `ticker` (the current/still-listed biotech roster, plus the 25 acquirers,
  all of which are currently-listed mega-caps). Safe to hand straight to FMP.
- **`build_unresolved_universe`** — every CIK seen across ALL historically
  landed SEC DERA quarters (`universe_snapshot.all_landed_quarters()`,
  already SIC-filtered to biotech at landing time by
  `sec_universe_ingest.py`) that has NO `ticker` on file. These are the
  companies most likely to matter for `E[return | deal]` labeling (delisted
  before this repo ever resolved a ticker for them — plausibly because they
  were acquired), and the ones this whole vendor decision exists to cover,
  but they need a NAME-based ticker lookup before FMP can be queried at all.
  `jobs/run_fmp_historical_backfill.py` is responsible for attempting that
  resolution (via `fmp_client.search_company_by_name`, itself unverified —
  see that module's docstring) and logging what it can't resolve, not this
  module — this module only identifies WHO needs resolving.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.universe_snapshot import all_landed_quarters
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass(frozen=True)
class TickerTarget:
    """A company we can query FMP for directly — a real ticker is already on file."""

    company_id: Optional[int]
    cik: Optional[str]
    ticker: str
    name: str


@dataclass(frozen=True)
class UnresolvedCompany:
    """A company with SEC filing history but no ticker on file — needs name-based resolution first."""

    cik: str
    name: str


def build_known_universe(repo: Any) -> List[TickerTarget]:
    """Every `p22_company` row with a `ticker` already on file."""
    companies = repo.list_companies_full()
    targets = [
        TickerTarget(company_id=c["company_id"], cik=c.get("cik"), ticker=c["ticker"], name=c["name"])
        for c in companies
        if c.get("ticker")
    ]
    _logger.info("Known universe (ticker already on file): %d companies", len(targets))
    return targets


def build_unresolved_universe(repo: Any, *, root: Optional[Path] = None) -> List[UnresolvedCompany]:
    """
    Every CIK across all historically landed DERA quarters with no `ticker`
    on file in `p22_company` — see module docstring. Deduped by CIK, keeping
    the most-recently-filed row's `name` (a company's registered name can
    change over its history; the latest is the most useful for a name search).

    Args:
        repo: A `P22Repo`-shaped object.
        root: Raw-zone root override, passed through to
            `all_landed_quarters` (used by tests).
    """
    known_ciks = {c["cik"] for c in repo.list_companies_full() if c.get("cik")}

    by_cik: Dict[str, Dict[str, Any]] = {}
    for rows in all_landed_quarters(root=root).values():
        for row in rows:
            cik = row.get("cik")
            if not cik or cik in known_ciks:
                continue
            existing = by_cik.get(cik)
            if existing is None or row.get("filed", "") > existing.get("filed", ""):
                by_cik[cik] = row

    unresolved = [UnresolvedCompany(cik=cik, name=row.get("name", "")) for cik, row in by_cik.items()]
    _logger.info("Unresolved universe (no ticker on file, needs name search): %d companies", len(unresolved))
    return unresolved
