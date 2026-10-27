"""
P22 — derived `market_cap` normalizer (spec §2.0.6/§2.4, resolved 2026-09-07).

`market_cap` was blocked for months on a dedicated market-data vendor
(`docs/Tasks.md` "Decisions needed" item 1) — but for any CURRENTLY-LISTED
company, no vendor purchase is actually needed: `p22_price_daily` already
gets a genuinely raw daily close from `ingest/price_ingest.py` (yfinance),
and `p22_financial_fact` already gets `shares_outstanding` from
`ingest/financial_facts.py` (SEC XBRL `dei:EntityCommonStockSharesOutstanding`).
`market_cap(t) = raw_close(t) x shares_outstanding(t)` needs nothing else.

**Both inputs must stay RAW/as-filed, never split-adjusted** —
`ingest/price_archive.py`'s module docstring explains why: a retro-adjusted
price times an as-filed share count is wrong by exactly the split factor.
`P22Repo.get_latest_raw_close_as_of` deliberately reads the unadjusted
column, not `get_adjusted_close`.

**What this does NOT solve**: a company that is delisted (acquired, or
otherwise no longer trading) has no yfinance ticker and therefore no
`p22_price_daily` rows going forward — this derivation naturally returns
`None` for those, same as before. That gap is still the FMP-Premium/
delisted-ticker-history question (`docs/Tasks.md` item 1's "Not yet
root-caused" framing narrowed to just that case), needed for the M6
backtest's historical labeling, not for live Block A/C scoring.

**Guards added 2026-10-27** (cross-check against 10-K public float, see
`docs/Design.md` "Size floor"):

- A close older than `MARKET_CAP_MAX_PRICE_AGE_DAYS` or a share count whose
  `period_end` is older than `MARKET_CAP_MAX_SHARES_AGE_DAYS` is rejected
  (`stale_price`/`stale_shares_outstanding`) — `get_latest_raw_close_as_of`
  happily returns a delisted company's last bar years later, and a few filers'
  share series simply stop (Amarin's last tagged count is from 2013).
- The share count is rescaled by every split/reverse split with an ex-date
  after its `period_end` and on or before the price's trade date. The cover-
  page count is as of `period_end`; a reverse split between that date and the
  price day otherwise inflates market cap by the split factor — 42% of the
  >8x outliers in the cross-check were exactly this.

The result is written as an ordinary `p22_financial_fact` row (metric
`"market_cap"`) via `P22Repo.upsert_financial_fact_bitemporal` — every
existing Block A/C feature function that reads `market_cap` via
`FeatureContext.get_latest_fact` needs no change to start seeing real
values once this job has run.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import MARKET_CAP_MAX_PRICE_AGE_DAYS, MARKET_CAP_MAX_SHARES_AGE_DAYS
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_METRIC = "market_cap"
_SOURCE_ID = "derived:raw_close_x_shares_outstanding"


@dataclass(frozen=True)
class MarketCapResult:
    """One company's derived market cap, ready for `upsert_financial_fact_bitemporal`."""

    company_id: int
    market_cap: float
    price_trade_date: date
    shares_outstanding: float  # split-adjusted to price_trade_date
    shares_period_end: Optional[date] = None
    split_factor: float = 1.0


def split_factor_between(splits: List[Dict[str, Any]], after: Optional[date], through: date) -> float:
    """
    Product of split ratios with `after < ex_date <= through` — what a share count as of `after`
    must be multiplied by to be comparable with a price on `through`.

    Args:
        splits: `P22Repo.get_splits_as_of` rows (`ex_date`, `ratio`).
        after: The share count's `period_end`; None means unknown, so no rescaling.
        through: The price's trade date.

    Returns:
        1.0 when no split falls in the window.
    """
    if after is None:
        return 1.0
    factor = 1.0
    for split in splits:
        if after < split["ex_date"] <= through:
            factor *= split["ratio"]
    return factor


def compute_market_cap(repo: Any, company_id: int, as_of: date) -> Tuple[Optional[MarketCapResult], Optional[str]]:
    """
    Derive one company's `market_cap` as of `as_of`.

    Both reads are already lookahead-safe (`get_latest_raw_close_as_of`,
    `get_financial_facts_as_of`) — this function just combines them.

    Returns:
        `(result, None)` on success, or `(None, reason)` where `reason` is a
        short category string suitable for aggregating via `collections.Counter`
        (same diagnostic discipline as P20's `sleeve_a.py`/`sleeve_c.py`
        rejection-breakdown instrumentation) — so a systematic gap (e.g. "every
        company is missing shares_outstanding") is visible in the run summary
        instead of silently producing an all-`None` funnel again.
    """
    price = repo.get_latest_raw_close_as_of(company_id, as_of)
    if price is None:
        return None, "no_price"
    if (as_of - price["trade_date"]).days > MARKET_CAP_MAX_PRICE_AGE_DAYS:
        return None, "stale_price"

    shares_facts = repo.get_financial_facts_as_of(company_id, "shares_outstanding", as_of)
    if not shares_facts or shares_facts[0].get("value") is None:
        return None, "no_shares_outstanding"

    shares_outstanding = float(shares_facts[0]["value"])
    if shares_outstanding <= 0:
        return None, "non_positive_shares_outstanding"

    shares_period_end = shares_facts[0].get("period_end")
    if shares_period_end is not None and (as_of - shares_period_end).days > MARKET_CAP_MAX_SHARES_AGE_DAYS:
        return None, "stale_shares_outstanding"

    split_factor = split_factor_between(repo.get_splits_as_of(company_id, as_of), shares_period_end, price["trade_date"])
    shares_outstanding *= split_factor
    market_cap = price["close_raw"] * shares_outstanding
    return (
        MarketCapResult(
            company_id=company_id,
            market_cap=market_cap,
            price_trade_date=price["trade_date"],
            shares_outstanding=shares_outstanding,
            shares_period_end=shares_period_end,
            split_factor=split_factor,
        ),
        None,
    )


def run(repo: Any, company_ids: List[int], as_of: Optional[date] = None) -> Dict[str, Any]:
    """
    Compute and write `market_cap` for every company in `company_ids`.

    Args:
        repo: A `P22Repo`-shaped object.
        company_ids: Companies to compute for (targets and acquirers alike —
            Block A needs acquirer market caps too, same as `run_price_ingest.py`).
        as_of: Defaults to today. The `known_from`/`period_end` written is
            "now" and `price_trade_date` respectively — see inline comment.

    Returns:
        Summary dict with `computed`/`skipped` counts and a `rejection_breakdown`
        (top skip reasons), logged as a warning when nothing computes at all.
    """
    as_of = as_of or date.today()
    computed = 0
    rejection_reasons: Counter[str] = Counter()

    for company_id in company_ids:
        result, reason = compute_market_cap(repo, company_id, as_of)
        if result is None:
            assert reason is not None
            rejection_reasons[reason] += 1
            continue

        repo.upsert_financial_fact_bitemporal(
            company_id=result.company_id,
            metric=_METRIC,
            value=result.market_cap,
            # "Now", not the price's trade_date: this fact is only known once
            # the derivation actually runs (spec §3.1 — the pipeline didn't
            # know a company's market cap on the trade date itself, only once
            # this job computed it), matching `price_ingest.py`'s VENDOR_PRICE_LAG_DAYS=0
            # convention of stamping known_from at ingest time, not trade time.
            known_from=datetime.now(timezone.utc),
            source_id=_SOURCE_ID,
            period_end=result.price_trade_date,
            valid_from=result.price_trade_date,
        )
        computed += 1

    total_skipped = sum(rejection_reasons.values())
    top_rejections = rejection_reasons.most_common(10)
    if computed == 0 and top_rejections:
        _logger.warning(
            "market_cap: 0/%d companies computed — top rejection reasons: %s",
            len(company_ids), top_rejections,
        )

    return {
        "companies_considered": len(company_ids),
        "computed": computed,
        "skipped": total_skipped,
        "rejection_breakdown": dict(top_rejections),
    }
