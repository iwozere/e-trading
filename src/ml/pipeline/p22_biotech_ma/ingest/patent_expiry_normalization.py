"""
P22 — Orange Book patent-expiry normalizer (spec §2.3, §4.1 Block A input).

Turns landed `orange_book` raw-zone payloads (`products.txt` + `patent.txt`,
already landed by `run_orange_book_ingest.py` via
`orange_book_client.fetch_and_land_orange_book`) into `p22_patent_expiry`
rows via `P22Repo.upsert_patent_expiry`. File format and column names
live-verified 2026-08-30 against the real, current Orange Book ZIP:

```
products.txt: Ingredient~DF;Route~Trade_Name~Applicant~Strength~Appl_Type~
              Appl_No~Product_No~TE_Code~Approval_Date~RLD~RS~Type~
              Applicant_Full_Name
patent.txt:   Appl_Type~Appl_No~Product_No~Patent_No~Patent_Expire_Date_Text~
              Drug_Substance_Flag~Drug_Product_Flag~Patent_Use_Code~
              Delist_Flag~Submission_Date
```

`Patent_Expire_Date_Text` is `"Aug 24, 2026"` (`%b %d, %Y`), matching the
spec's own naming.

**Scope of this pass:**

- Only `patent.txt` is normalized. `exclusivity.txt` (regulatory exclusivity
  grants — `Exclusivity_Code` values like `NCE`, `ODE-###`, `PED`, `GAIN`,
  ...) is NOT — collapsing that code space onto `p22_patent_expiry`'s four-
  value `exclusivity_type` enum (`patent|orphan|ped|bla_12yr`) is itself a
  domain-classification decision (which codes count as which bucket, and
  whether a code implies orphan-drug status vs. pediatric-extension vs.
  something else), the same character of decision as the therapeutic-area
  classifier this build has deliberately deferred elsewhere. Every row this
  module writes is a real small-molecule PATENT (not an FDA exclusivity
  grant), so `exclusivity_type="patent"` is a safe constant here, not a
  guess — see `docs/Tasks.md` "Decisions needed" for the exclusivity.txt gap.
- `therapeutic_area` is always written `None` — same reasoning as
  `p22_asset.therapeutic_area` elsewhere in this build (see
  `ingest/trial_normalization.py`'s docstring): classifying a product's
  therapeutic area from Orange Book's `Ingredient`/`DF;Route` fields needs the
  same not-yet-made mapping decision, and `patent_expiry.therapeutic_area` is
  nullable, so `None` is the honest answer, not a fabricated guess.
- `ttm_revenue_usd` is always `None` — spec §2.3 calls this the "highest-value
  and highest-effort" join in the whole build (product revenue by year,
  mostly requiring manual 10-K exhibit extraction for large-cap pharma), out
  of scope for this pass entirely.
- **Applicant -> acquirer resolution is deterministic-only, not fuzzy.**
  `alias_matching.match_alias` is reused for the lookup, but only an exact
  (post-`normalize_company_name`) match is written; a sub-100 fuzzy match is
  logged (with its score) and NOT written, and NOT queued for review either —
  unlike CT.gov/openFDA sponsor-alias matching, this module does not extend
  `review_queue.py`'s confirm dispatch to know how to write a
  `p22_patent_expiry` row from a confirmed review item, which would be needed
  to queue these safely. That's a real, contained gap, not an oversight —
  logged in `docs/Tasks.md`. In practice this means the hit rate here will be
  low at first (most Orange Book applicants are generic manufacturers or
  companies outside the curated ~21-name acquirer roster, and legal-entity
  applicant names like "PADAGIS ISRAEL PHARMACEUTICALS LTD" often differ from
  a parent company's roster name even for a real acquirer's own subsidiary)
  — expected and safe, not a bug: only unambiguous matches get written.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.alias_matching import match_alias
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_PRODUCT_KEY = Tuple[Optional[str], Optional[str], Optional[str]]


@dataclass(frozen=True)
class PatentExpiryRecord:
    """One Orange Book patent row, joined to its product and ready for `P22Repo.upsert_patent_expiry`."""

    applicant_full_name: str
    product_name: Optional[str]
    application_no: str
    loe_date: date
    exclusivity_type: str
    source: str = "orange_book"


def _parse_patent_expire_date(raw: Optional[str]) -> Optional[date]:
    """`Patent_Expire_Date_Text` is `"Aug 24, 2026"` — live-verified 2026-08-30."""
    if not raw or not raw.strip():
        return None
    try:
        return datetime.strptime(raw.strip(), "%b %d, %Y").date()
    except ValueError:
        _logger.warning("Unparseable Orange Book Patent_Expire_Date_Text: %r", raw)
        return None


def _product_key(row: Dict[str, Any]) -> _PRODUCT_KEY:
    return (row.get("Appl_Type"), row.get("Appl_No"), row.get("Product_No"))


def build_product_lookup(products_rows: List[Dict[str, Any]]) -> Dict[_PRODUCT_KEY, Dict[str, Any]]:
    """`(Appl_Type, Appl_No, Product_No) -> products.txt row`, for joining `patent.txt` to its applicant."""
    return {_product_key(row): row for row in products_rows}


def extract_patent_expiry_records(
    products_rows: List[Dict[str, Any]], patent_rows: List[Dict[str, Any]]
) -> List[PatentExpiryRecord]:
    """
    Join `patent.txt` rows to `products.txt` on `(Appl_Type, Appl_No, Product_No)` and normalize.
    A patent row with no parseable expiry date, or no matching product (so no applicant name to
    resolve an acquirer from later), is dropped and logged — not written with a guessed value.
    """
    lookup = build_product_lookup(products_rows)
    records: List[PatentExpiryRecord] = []

    for patent_row in patent_rows:
        loe_date = _parse_patent_expire_date(patent_row.get("Patent_Expire_Date_Text"))
        if loe_date is None:
            continue

        product = lookup.get(_product_key(patent_row))
        applicant = product.get("Applicant_Full_Name") if product else None
        if not applicant:
            continue

        records.append(
            PatentExpiryRecord(
                applicant_full_name=applicant,
                product_name=product.get("Trade_Name") if product else None,
                application_no=patent_row.get("Appl_No", ""),
                loe_date=loe_date,
                exclusivity_type="patent",
            )
        )

    return records


def resolve_applicant_to_acquirer(applicant_full_name: str, acquirer_companies: Dict[int, str]) -> Optional[int]:
    """
    Deterministic-only match of an Orange Book applicant name against the acquirer roster (see
    module docstring for why fuzzy matches are logged, not written). Returns `company_id` or `None`.
    """
    result = match_alias(applicant_full_name, acquirer_companies)
    if result.match_type == "deterministic" and result.company_id is not None:
        return result.company_id
    if result.match_type == "fuzzy":
        _logger.info(
            "Unresolved (fuzzy, not auto-written) patent applicant->acquirer candidate: "
            "%r ~ %r (score=%.0f) — see module docstring",
            applicant_full_name, result.matched_name, result.score,
        )
    return None


def write_patent_expiry_records(
    records: List[PatentExpiryRecord], acquirer_companies: Dict[int, str], repo: Any
) -> Dict[str, int]:
    """
    Resolve each record's applicant to an acquirer and write matched ones via
    `P22Repo.upsert_patent_expiry`.

    Returns:
        Counters: `written`, `unresolved`.
    """
    counts = {"written": 0, "unresolved": 0}
    for record in records:
        acquirer_id = resolve_applicant_to_acquirer(record.applicant_full_name, acquirer_companies)
        if acquirer_id is None:
            counts["unresolved"] += 1
            continue
        repo.upsert_patent_expiry(
            acquirer_id=acquirer_id,
            application_no=record.application_no,
            loe_date=record.loe_date,
            source=record.source,
            product_name=record.product_name,
            therapeutic_area=None,
            ttm_revenue_usd=None,
            exclusivity_type=record.exclusivity_type,
        )
        counts["written"] += 1

    _logger.info("Patent-expiry write: %s", counts)
    return counts
