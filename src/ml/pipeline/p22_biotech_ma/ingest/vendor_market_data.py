"""
P22 — market-data vendor adapter interface (spec §2.4, narrowed by §2.0.5-2.0.6).

This module defines the bitemporal contract any future FUNDAMENTALS adapter
(market cap, shares outstanding, segment revenue — point-in-time facts) must
satisfy. Daily OHLCV price ingest does NOT go through this Protocol — it has
its own dedicated, already-implemented path (`P22Repo.upsert_price_daily`/
`upsert_corporate_action`, `ingest/price_ingest.py`, `ingest/fmp_backfill.py`)
directly against `p22_price_daily`'s own `vendor` column, which was already
designed to be multi-source (`'ibkr'|'fmp'|'yfinance'`).

Per §2.0's source-capability matrix, the fundamentals gap this Protocol
exists for is narrower than it first looked: EDGAR (already integrated via
EdgarDownloader) covers most of what §2.4 originally asked for on the
fundamentals side. **Price data resolved differently and is DONE, not
deferred** (2026-09-01): ongoing/current daily prices are covered by
yfinance (`ingest/yfinance_client.py`, `jobs/run_price_ingest.py`, running
daily in production) rather than IBKR — sidesteps IBKR's still-unverified
raw-vs-adjusted question (`docs/Tasks.md` item 6) and needs no live
TWS/Gateway session. Deep historical prices for delisted tickers (needed for
`E[return | deal]` labeling, M6/M7) are FMP's role
(`ingest/fmp_backfill.py`) — see `docs/Tasks.md` item 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Protocol


@dataclass(frozen=True)
class VendorFact:
    """
    One bitemporal vendor-sourced fact, matching the `financial_fact` table
    contract (spec §2.4, §3.2).

    `known_from` must be the date the vendor first published this value —
    never the period it describes, never the date we happened to fetch it.
    If the vendor cannot supply that, callers fall back to
    `period_end + VENDOR_FUNDAMENTALS_LAG_DAYS` (or `VENDOR_PRICE_LAG_DAYS`
    for prices), per config.py and `config/vendor_lag.yaml`.
    """

    company_id: int
    metric: str
    value: float
    unit: str
    period_end: date
    known_from: datetime
    source_id: str
    source_url: Optional[str] = None


class MarketDataProvider(Protocol):
    """
    Bitemporal market-data vendor contract. A conforming implementation must
    NEVER update a fact in place — restatements are new `VendorFact` rows
    with a later `known_from`; the caller (via `P22Repo.
    upsert_financial_fact_bitemporal`) is responsible for closing the prior
    row's `valid_to`, not this provider.
    """

    def get_market_cap(self, ticker: str, as_of: date) -> Optional[VendorFact]:
        """Point-in-time market cap, survivorship-bias-free (queryable for delisted tickers)."""
        ...

    def get_shares_outstanding(self, ticker: str, as_of: date) -> Optional[VendorFact]:
        """Point-in-time shares outstanding."""
        ...

    def get_segment_revenue(self, ticker: str, as_of: date, therapeutic_area: str) -> Optional[VendorFact]:
        """Segment/product-level revenue for a large-cap acquirer, by therapeutic area."""
        ...

    def get_historical_price(self, ticker: str, as_of: date) -> Optional[VendorFact]:
        """
        Historical close price as of a given date, including for delisted
        tickers. This is the one gap spec §2.0.6 identifies as genuinely
        unmet by existing repo integrations.
        """
        ...


class NullMarketDataProvider:
    """
    No-op `MarketDataProvider`. Every method raises `NotImplementedError` —
    this makes "no vendor is wired in yet" a loud failure at the call site
    rather than a silent `None` that could be mistaken for "no data available"
    (which would corrupt a gate like `dilution_gate` into always passing or
    always failing, rather than surfacing "we can't compute this").
    """

    def get_market_cap(self, ticker: str, as_of: date) -> Optional[VendorFact]:
        raise NotImplementedError("No market-data vendor is configured (see docs/Tasks.md)")

    def get_shares_outstanding(self, ticker: str, as_of: date) -> Optional[VendorFact]:
        raise NotImplementedError("No market-data vendor is configured (see docs/Tasks.md)")

    def get_segment_revenue(self, ticker: str, as_of: date, therapeutic_area: str) -> Optional[VendorFact]:
        raise NotImplementedError("No market-data vendor is configured (see docs/Tasks.md)")

    def get_historical_price(self, ticker: str, as_of: date) -> Optional[VendorFact]:
        raise NotImplementedError("No market-data vendor is configured (see docs/Tasks.md)")
