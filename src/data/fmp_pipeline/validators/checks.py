"""Import every check module so the checks register in spec §11 order."""

from __future__ import annotations

from src.data.fmp_pipeline.validators import checks_events, checks_pipeline, checks_prices, checks_reference, checks_siblings

#: Spec §11 table order, used by ``run_checks``.
ORDER = ("V-SCHEMA", "V-DUP", "V-ORDER", "V-OHLC", "V-CAL", "V-SPLIT-KNOWN", "V-SPLIT-CONSISTENCY", "V-OVERRIDES",
         "V-SIBLING-BASIS", "V-DIV-SANITY", "V-ADJ-RECON", "V-YF-XCHECK", "V-DELIST", "V-UNIVERSE", "V-IDENT", "V-FLAGS",
         "V-INDEX", "V-FUND-PIT", "V-FUND-COV", "V-INTRADAY-ALIGN", "V-INTRADAY-DAILY", "V-COMPLETE", "V-RAW-ARCHIVE",
         "V-REPRO", "V-ADJ-UNIT")
MODULES = (checks_prices, checks_events, checks_siblings, checks_reference, checks_pipeline)
