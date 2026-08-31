"""
P22 — FMP historical bulk backfill orchestration (spec §2.0.6/§2.4,
docs/Tasks.md item 1, M3, 2026-08-31).

Business logic behind `cli/fmp_backfill_cli.py` — a human-run, ONE-TIME
operation meant for the window a Premium-tier FMP subscription is active
(30-year history, vs. Starter/Basic's 5-year cap): land the widest
reasonable set of historical daily price payloads into the raw zone before
the subscription lapses back to a lower tier. Not a `jobs/register_jobs.py`
scheduled job — see the CLI's docstring for why.

**What this does NOT do: write `p22_price_daily`/`p22_corporate_action`
rows.** That normalization step is deliberately separate and NOT time-boxed
to the Premium month — the raw zone is immutable and content-addressed, so
once a payload is landed here, deciding how to turn it into bitemporal rows
(in particular: resolving whether FMP's `close` field is truly raw/
unadjusted or already split/dividend-adjusted — unverified, see
`ingest/fmp_client.py`'s docstring) can happen calmly afterward, against
real data, with no time pressure. Land now, normalize later.

**Ticker resolution for delisted-before-we-resolved-them companies** uses
FMP's name-search endpoint (`ingest/fmp_client.search_company_by_name` —
itself unverified, see that module) with **deterministic-only** matching —
same caution as `alias_matching.py`'s deterministic path: a fuzzy/uncertain
match is logged for manual review, never auto-accepted, even though the
consequence of a wrong match here (a wasted API call, some harmless
unrelated-company data landed under the wrong ticker) is lower-stakes than a
wrong entry in `p22_company_alias`. Consistency of caution across the
codebase was judged more valuable than exploiting this lower-stakes case.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.ingest.entity_resolution import normalize_company_name
from src.ml.pipeline.p22_biotech_ma.ingest.fmp_client import FMPClient
from src.ml.pipeline.p22_biotech_ma.ingest.fmp_universe import (
    TickerTarget,
    UnresolvedCompany,
    build_known_universe,
    build_unresolved_universe,
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

RAW_PRICE_SOURCE = "fmp_historical_price"


_US_EXCHANGES = frozenset({"NASDAQ", "NYSE", "AMEX", "NYSE AMERICAN", "NYSEAMERICAN", "BATS", "CBOE"})


def resolve_ticker_by_name(company: UnresolvedCompany, client: FMPClient) -> Optional[TickerTarget]:
    """
    Deterministic-only name-search resolution — see module docstring.
    `None` (logged) if no candidate's name normalizes to an exact match.

    **Live-verified 2026-08-31 finding, fixed here:** a name search can
    return MULTIPLE exact-name matches for the same company across different
    exchanges/currencies (confirmed live: searching "Moderna" returns both
    `MRNA` on NASDAQ, USD, AND `0QF.F` on the Frankfurt Stock Exchange, EUR
    — both literally named "Moderna, Inc."). Picking the first exact match
    in API response order is NOT safe — it happened to return the German
    cross-listing before the US one in the observed response. Among exact
    matches, this prefers a USD-denominated US-exchange listing (`p22` is
    explicitly "US-listed biotech companies," spec §0), and only falls back
    to the first exact match if no candidate looks like a US listing.
    """
    candidates = client.search_company_by_name(company.name)
    normalized_target = normalize_company_name(company.name)
    exact_matches = [
        c for c in candidates
        if c.get("symbol") and c.get("name") and normalize_company_name(c["name"]) == normalized_target
    ]

    if not exact_matches:
        if candidates:
            _logger.info(
                "No exact name match for unresolved CIK=%s name=%r among %d FMP search candidate(s) — "
                "not written, needs manual review",
                company.cik, company.name, len(candidates),
            )
        return None

    us_matches = [
        c for c in exact_matches
        if c.get("currency") == "USD" and str(c.get("exchange", "")).upper() in _US_EXCHANGES
    ]
    if len(exact_matches) > 1 and not us_matches:
        _logger.info(
            "Multiple exact name matches for CIK=%s name=%r, none look like a US listing "
            "(exchanges=%s) — using the first one; verify manually if this matters",
            company.cik, company.name, [c.get("exchange") for c in exact_matches],
        )
    chosen = (us_matches or exact_matches)[0]
    return TickerTarget(company_id=None, cik=company.cik, ticker=chosen["symbol"], name=company.name)


def build_backfill_targets(
    repo: Any, *, include_unresolved: bool, client: Optional[FMPClient] = None
) -> Dict[str, Any]:
    """
    Assemble the full list of `TickerTarget`s to fetch.

    Returns:
        `{"targets": List[TickerTarget], "resolved_via_search": int,
          "still_unresolved": List[UnresolvedCompany]}`
    """
    targets: List[TickerTarget] = list(build_known_universe(repo))
    resolved_via_search = 0
    still_unresolved: List[UnresolvedCompany] = []

    if include_unresolved:
        unresolved = build_unresolved_universe(repo)
        owns_client = client is None
        active_client = client or FMPClient()
        try:
            for company in unresolved:
                match = resolve_ticker_by_name(company, active_client)
                if match is not None:
                    targets.append(match)
                    resolved_via_search += 1
                else:
                    still_unresolved.append(company)
        finally:
            if owns_client:
                active_client.close()

    _logger.info(
        "Backfill target assembly: %d targets (%d resolved via name search), %d still unresolved",
        len(targets), resolved_via_search, len(still_unresolved),
    )
    return {"targets": targets, "resolved_via_search": resolved_via_search, "still_unresolved": still_unresolved}


def land_historical_prices(
    targets: List[TickerTarget],
    *,
    start_date: date,
    end_date: date,
    client: Optional[FMPClient] = None,
    skip_already_landed: bool = True,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Fetch and land full raw historical-price JSON for each target.

    Args:
        targets: Tickers to fetch.
        start_date: Earliest date to request (FMP silently returns whatever
            it actually has within the plan's window — no need to guess the
            plan's exact cutoff, just ask wide).
        end_date: Latest date to request.
        client: Reuse an existing `FMPClient` (its own connection/rate
            limiter), or `None` to open and close one for this call.
        skip_already_landed: Skip a ticker if the raw zone already has ANY
            prior landing for it (any date partition) — makes this safe to
            interrupt and re-run without re-spending API quota. Set `False`
            to force a re-fetch (e.g. to pick up newer data for a ticker
            landed early in the month).
        root: Raw-zone root override (used by tests).

    Returns:
        `{"landed": int, "skipped_already_landed": int, "failed": List[str]}`
    """
    owns_client = client is None
    active_client = client or FMPClient()
    landed = 0
    skipped = 0
    failed: List[str] = []

    try:
        for i, target in enumerate(targets, 1):
            if skip_already_landed and raw_zone.has_any_landed(RAW_PRICE_SOURCE, target.ticker, root=root):
                skipped += 1
                continue

            payload = active_client.fetch_historical_price_full(target.ticker, start_date, end_date)
            if payload is None:
                failed.append(target.ticker)
                continue

            raw_zone.write(
                source=RAW_PRICE_SOURCE, entity=target.ticker, as_of_date=date.today(), payload=payload, root=root
            )
            landed += 1

            if i % 25 == 0:
                _logger.info(
                    "Backfill progress: %d/%d (landed=%d skipped=%d failed=%d)",
                    i, len(targets), landed, skipped, len(failed),
                )
    finally:
        if owns_client:
            active_client.close()

    _logger.info("Backfill complete: landed=%d skipped=%d failed=%d", landed, skipped, len(failed))
    return {"landed": landed, "skipped_already_landed": skipped, "failed": failed}
