"""
P22 — acquirer-config loader (spec §2.0.4, §3.5, M3).

Parses `config/pipeline/p22_acquirers.yaml` into `AcquirerConfig` records and
upserts each acquirer's *identity* into `p22_company` (role `acquirer`, or
`both` if the company already has a resolved `target`-side row from the DERA
universe — see `upsert_acquirer_roster`). This is deliberately narrower than
"load the acquirer config": `bloc`, `entry_date`, and `exit_date` are read
here as data but are **not written to any DB column** — `p22_company` has no
such columns, and the spec's own §3.2 SQL sketch doesn't define one. They stay
in the config file and are meant to be read directly by Block A's capacity
model once it's built (spec §4.1), keyed off the same company identity this
module creates.

**This intentionally does not wait on `docs/Tasks.md` "Decisions needed" item
3** (real entry/exit-date curation, real CIKs). That item is about whether the
*dates in the file* are accurate — a domain-curation question. Whether the
~21 companies the file already names exist as `p22_company` rows with the
correct identity so other tables (patent expiry, financial facts, price
history) can attach to them is a separate, purely mechanical question, and
this module answers only that one. Anything that reads `entry_date`/
`exit_date` for real capacity-window logic (i.e. Block A itself) must treat
today's placeholder dates as exactly that — placeholders — until item 3 is
resolved; this module does not gate on it because it doesn't use those values
for anything beyond passing them through.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import ACQUIRERS_YAML
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# spec §4.4.1's foreign-investment-screening gate vocabulary.
VALID_BLOCS = frozenset({"us", "allied", "neutral", "elevated_scrutiny"})


@dataclass(frozen=True)
class AcquirerConfig:
    """One row of `p22_acquirers.yaml`, parsed and validated but not yet resolved to a company_id."""

    name: str
    ticker: str
    cik: Optional[str]
    bloc: str
    entry_date: date
    exit_date: Optional[date]


def load_acquirers(path: Path = ACQUIRERS_YAML) -> List[AcquirerConfig]:
    """
    Parse and validate `p22_acquirers.yaml`. Raises `ValueError` on a
    structurally invalid entry (unknown `bloc`, missing `name`/`ticker`) —
    this is config the whole downstream chain trusts, so a bad row should
    fail loudly at load time, not surface as a confusing normalization bug
    later.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("acquirers", [])

    acquirers: List[AcquirerConfig] = []
    for entry in entries:
        name = entry.get("name")
        ticker = entry.get("ticker")
        bloc = entry.get("bloc")
        if not name or not ticker:
            raise ValueError(f"Acquirer entry missing name/ticker: {entry!r}")
        if bloc not in VALID_BLOCS:
            raise ValueError(f"Acquirer {name!r} has invalid bloc {bloc!r}; must be one of {sorted(VALID_BLOCS)}")

        acquirers.append(
            AcquirerConfig(
                name=name,
                ticker=ticker,
                cik=entry.get("cik"),
                bloc=bloc,
                entry_date=entry["entry_date"],
                exit_date=entry.get("exit_date"),
            )
        )

    _logger.info("Loaded %d acquirer config entries from %s", len(acquirers), path)
    return acquirers


def upsert_acquirer_roster(acquirers: List[AcquirerConfig], repo: Any) -> int:
    """
    Upsert each acquirer's identity into `p22_company` via
    `P22Repo.upsert_acquirer_company` (ticker-keyed merge — see that method's
    docstring for why this can't be a plain `cik`-keyed upsert like the DERA
    roster path, and for how it merges into an existing `target`-role row
    instead of creating a duplicate identity for a company DERA already
    resolved).

    Returns:
        Number of acquirer entries processed.
    """
    for acquirer in acquirers:
        repo.upsert_acquirer_company(name=acquirer.name, ticker=acquirer.ticker, cik=acquirer.cik)
    return len(acquirers)
