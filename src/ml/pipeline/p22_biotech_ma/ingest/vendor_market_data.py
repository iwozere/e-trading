"""
P22 — market-data vendor adapter interface (spec §2.4, narrowed by §2.0.5-2.0.6).

No vendor has been selected yet (deferred decision, 2026-08-30). This module
defines the bitemporal contract any future adapter must satisfy, so call
sites that will eventually need vendor data can be written now against a
stable interface.

Per §2.0's source-capability matrix, this gap is narrower than it first
looked: EDGAR (fundamentals, already integrated via EdgarDownloader) and IBKR
(daily prices for currently-listed names, already integrated) cover most of
what §2.4 originally asked for. What's actually missing is historical prices
for *delisted* tickers, needed for `E[return | deal]` labeling (M6/M7) — see
docs/Tasks.md for the recommended path (FMP Starter, ~$15/mo).
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
