"""
P22 — company domicile normalizer, for Block E's `is_foreign_domiciled` (spec §4.5), 2026-09-08.

`ingest/sec_raw_ingest.py` already lands the FULL SEC `/submissions/CIK##########.json` payload
verbatim for every CIK (`source="sec_submissions"`) — this is the first normalizer to actually
read it for anything beyond entity resolution's ticker/exchange snapshot.

**Live-verified 2026-09-08 that a single field isn't reliable.** `addresses.business.isForeignLocation`
looks like the obvious signal, but a real foreign acquirer (Novo Nordisk A/S, Danish) has it
`null` even though `addresses.business.stateOrCountry` correctly shows `"G7"` (SEC's own code for
Denmark, confirmed via `stateOrCountryDescription: "Denmark"`) — `isForeignLocation` is populated
inconsistently in EDGAR's own data, not something to trust alone. A second real check (AstraZeneca
PLC, UK) DOES have `isForeignLocation: 1`. This module treats EITHER signal as sufficient: a
company is foreign-domiciled if `isForeignLocation == 1` OR `stateOrCountry` isn't a real US
state/territory postal code — the two checks catch each other's gaps.

Explicitly NOT the same thing as Block A's foreign-investment-screening bloc tiers (spec §4.4.1,
`p22_acquirers.yaml`'s `bloc` field) — that's an ACQUIRER-side, pairwise CFIUS concept. This is
the separate, mild, TARGET-side friction spec §4.5 describes (inversion mechanics, cross-border
tender-offer complexity) and applies to any company, not just the curated acquirer roster.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

# Metric name for the p22_financial_fact row this normalizes into (see jobs/run_domicile_normalization.py).
METRIC = "is_foreign_domiciled"
SOURCE_ID = "sec_submissions"

# US state/territory postal codes SEC's stateOrCountry field uses for a domestic address.
# Anything else (a 2-letter/alphanumeric SEC country code, e.g. "X0"=UK, "G7"=Denmark) is foreign.
_US_STATE_OR_TERRITORY_CODES = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
    "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "PR", "VI", "GU", "AS", "MP",
})


def extract_is_foreign_domiciled(submissions: Dict[str, Any]) -> Optional[bool]:
    """
    Derive `is_foreign_domiciled` from a landed `sec_submissions` payload. `None` if the payload
    has no usable business-address data at all (not the same as a confirmed domestic address).
    """
    business = (submissions.get("addresses") or {}).get("business") or {}
    if business.get("isForeignLocation") == 1:
        return True
    state_or_country = business.get("stateOrCountry")
    if state_or_country:
        return state_or_country.upper() not in _US_STATE_OR_TERRITORY_CODES
    return None
