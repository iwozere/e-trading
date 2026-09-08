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

**Added 2026-09-08: `land_ratios_and_estimates`, the same land-now-normalize-
later pattern for `/stable/ratios`/`/stable/analyst-estimates`** — Block A's
still-blocked `ebitda`/forward-P/E (`docs/Tasks.md` items 9-10). The user
plans to buy Premium for one month starting ~2026-09-15; at the CURRENT
tier's observed throughput (5 req/s, well under Premium's published 750
req/min — see `config.FMP_RATE_LIMIT_RPS`) a full universe pass across all
three endpoints (price + ratios + estimates) is on the order of an hour, not
a month — the month is buffer for re-runs, fixing whatever the first real
pass surfaces, and picking up newly name-search-resolved tickers, not a
throughput requirement. Run `cli/fmp_backfill_cli.py backfill` and
`backfill-fundamentals` (ideally more than once over the month — both are
fully resumable) rather than treating this as a single make-or-break session.

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
# Added 2026-09-08, ahead of the user's planned FMP Premium purchase (~1 week out, for one
# month) — for Block A's still-blocked ebitda/forward-P/E (docs/Tasks.md items 9-10). Live-verified
# 2026-09-08 on the CURRENT (non-Premium) tier that these endpoints 402 for 22 of 25 acquirers —
# `test-fundamentals` in `cli/fmp_backfill_cli.py` re-checks a handful of those specific tickers
# first, so a real Premium key's actual entitlement is confirmed before spending a full run on it.
RAW_RATIOS_SOURCE = "fmp_ratios"
RAW_ANALYST_ESTIMATES_SOURCE = "fmp_analyst_estimates"


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
    limit: Optional[int] = None,
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
        limit: Stop after actually FETCHING this many tickers (already-landed
            skips don't count against it) — lets a caller cap one session's
            API spend without needing to pre-slice `targets` (which would
            keep re-considering the same already-landed prefix every time).
            `None` (default) fetches every not-yet-landed target.
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

            if limit is not None and landed >= limit:
                _logger.info("Backfill limit of %d reached — stopping for this run, %d target(s) untouched", limit, len(targets) - i + 1)
                break

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


def land_ratios_and_estimates(
    targets: List[TickerTarget],
    *,
    client: Optional[FMPClient] = None,
    skip_already_landed: bool = True,
    limit: Optional[int] = None,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Fetch and land `/stable/ratios` + `/stable/analyst-estimates` for each target — the EBITDA
    and forward-EPS data Block A's `cash_capacity`/`currency_quality` need (`docs/Tasks.md` items
    9-10). Added 2026-09-08 ahead of the user's planned FMP Premium purchase: live-verified on the
    CURRENT tier that both 402 for 22 of the 25 acquirers, so this is landing nothing useful yet —
    ready the moment the account is upgraded, same "built ahead of the decision" precedent as
    `land_historical_prices` itself.

    A ticker counts as "already landed" only when BOTH endpoints have a prior landing — a ticker
    that landed ratios but failed estimates (or vice versa) is retried, not silently left
    half-complete. Same `limit`/resumability contract as `land_historical_prices`.

    Returns:
        `{"landed_ratios": int, "landed_estimates": int, "skipped_already_landed": int,
          "failed": List[str]}` — `failed` lists tickers where BOTH endpoints came back `None`
          (a 402/404/error for both, not just one).
    """
    owns_client = client is None
    active_client = client or FMPClient()
    landed_ratios = 0
    landed_estimates = 0
    skipped = 0
    failed: List[str] = []
    fetched_count = 0

    try:
        for i, target in enumerate(targets, 1):
            already_landed = skip_already_landed and (
                raw_zone.has_any_landed(RAW_RATIOS_SOURCE, target.ticker, root=root)
                and raw_zone.has_any_landed(RAW_ANALYST_ESTIMATES_SOURCE, target.ticker, root=root)
            )
            if already_landed:
                skipped += 1
                continue

            if limit is not None and fetched_count >= limit:
                _logger.info(
                    "Fundamentals backfill limit of %d reached — stopping for this run, %d target(s) untouched",
                    limit, len(targets) - i + 1,
                )
                break
            fetched_count += 1

            ratios = active_client.fetch_ratios(target.ticker)
            if ratios is not None:
                raw_zone.write(source=RAW_RATIOS_SOURCE, entity=target.ticker, as_of_date=date.today(), payload=ratios, root=root)
                landed_ratios += 1

            estimates = active_client.fetch_analyst_estimates(target.ticker, period="annual")
            if estimates is not None:
                raw_zone.write(
                    source=RAW_ANALYST_ESTIMATES_SOURCE, entity=target.ticker, as_of_date=date.today(),
                    payload=estimates, root=root,
                )
                landed_estimates += 1

            if ratios is None and estimates is None:
                failed.append(target.ticker)

            if i % 25 == 0:
                _logger.info(
                    "Fundamentals backfill progress: %d/%d (ratios=%d estimates=%d skipped=%d failed=%d)",
                    i, len(targets), landed_ratios, landed_estimates, skipped, len(failed),
                )
    finally:
        if owns_client:
            active_client.close()

    _logger.info(
        "Fundamentals backfill complete: ratios=%d estimates=%d skipped=%d failed=%d",
        landed_ratios, landed_estimates, skipped, len(failed),
    )
    return {
        "landed_ratios": landed_ratios, "landed_estimates": landed_estimates,
        "skipped_already_landed": skipped, "failed": failed,
    }
