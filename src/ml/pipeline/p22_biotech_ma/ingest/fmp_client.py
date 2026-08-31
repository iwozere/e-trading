"""
P22 — Financial Modeling Prep (FMP) client for the vendor market-data
decision (spec §2.0.6/§2.4, docs/Tasks.md item 1, decided 2026-08-31: FMP).

This is a NEW, narrow client for the two things P22 needs that
`src.data.downloader.fmp_data_downloader.FMPDataDownloader` (already used
elsewhere in this repo) doesn't already provide in the shape P22 needs:

1. **Full raw historical daily price JSON.** `FMPDataDownloader.get_ohlcv`
   calls `/stable/historical-price-full` — **live-verified 2026-08-31 that
   this endpoint is now DEAD (404, even for an obviously-valid symbol like
   MRNA)**, not just lossy. The correct current endpoint is
   `/stable/historical-price-eod/full`, confirmed live (200, real data back
   to a company's IPO date). Its response is a **bare list** of
   `{symbol, date, open, high, low, close, volume, change, changePercent,
   vwap}` — no `historical` wrapper key, and notably **no separate
   `adjClose` field** — so whether `close` is truly raw/unadjusted or
   already split/dividend-adjusted is still an open question this client
   does not resolve; it lands the response verbatim either way (spec
   §2.0.7's bitemporal price archive wants the complete raw payload, not a
   pre-judged one).
2. **Company name search** — for resolving a ticker for a company we only
   know by CIK/name (delisted before this repo's current-snapshot-only
   ticker resolution ever saw it — see `docs/Tasks.md` item 1's ticker-gap
   note). `FMPDataDownloader` has no equivalent method. **Live-verified
   2026-08-31**: `/stable/search-name` is real and returns candidates shaped
   `{symbol, name, currency, exchange, exchangeFullName}` — but a single
   real query ("Moderna") returned TWO exact-name matches across different
   exchanges (NASDAQ `MRNA` and a Frankfurt cross-listing `0QF.F`, both
   literally named "Moderna, Inc."), which is why
   `ingest/fmp_backfill.resolve_ticker_by_name` prefers a USD/US-exchange
   candidate rather than trusting API response order.

**Also live-discovered 2026-08-31, not yet understood — flag for the user:**
the current key can fetch MRNA and PFE but gets `402 Payment Required`
("this value set for 'symbol' is not available under your current
subscription") for AMGN, GILD, and SRPT — across every date range tried,
`/full` and `/light` endpoints alike. This looks like a **per-symbol
entitlement list**, not a date-depth cap — a materially different kind of
plan restriction than the "Basic/Starter capped at 5 years of history"
finding from the web-search-based plan comparison earlier in `docs/Tasks.md`
item 1. Needs checking directly against the FMP account dashboard, not
guessable from outside.

Both functions reuse `FMPDataDownloader`'s own API-key resolution (env var ->
`config.donotshare.donotshare.FMP_API_KEY` -> default), so there is exactly
one place in the repo that decides where the key comes from.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.ml.pipeline.p22_biotech_ma.config import FMP_STABLE_URL
from src.ml.pipeline.p22_biotech_ma.ingest.http_retry import get_with_retry
from src.ml.pipeline.p22_biotech_ma.ingest.rate_limits import fmp_limiter
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class FMPClient:
    """Thin client over the FMP `/stable` endpoints P22 needs directly (see module docstring)."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0) -> None:
        # Delegates key resolution to FMPDataDownloader rather than re-implementing
        # the env-var/donotshare lookup priority a second time — one source of truth.
        self._api_key = api_key or FMPDataDownloader().api_key
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FMPClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        del _exc_info
        self.close()

    def fetch_historical_price_full(self, symbol: str, start_date: date, end_date: date) -> Optional[List[Dict[str, Any]]]:
        """
        Full raw historical daily price list from `/stable/historical-price-eod/full`
        for one symbol — live-verified 2026-08-31 (see module docstring for
        the endpoint-name correction vs. `FMPDataDownloader.get_ohlcv`).

        Returns:
            The raw response JSON — a list of daily row dicts (every field
            FMP sends, e.g. `open`/`high`/`low`/`close`/`volume`/`vwap`/
            `change`/`changePercent` — not just the 5 `get_ohlcv`
            standardizes to), or `None` on failure / no data / a symbol not
            covered by the current plan (a 402 or 404 here is logged at
            INFO, not WARNING/ERROR — both are expected, common outcomes for
            this exact bulk-backfill use case, not errors to alarm on).
        """
        params = {
            "apikey": self._api_key,
            "symbol": symbol,
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
        }
        resp = get_with_retry(
            self._client, f"{FMP_STABLE_URL}/historical-price-eod/full", params=params, rate_limiter=fmp_limiter
        )
        if resp is None:
            return None
        if resp.status_code == 402:
            _logger.info("FMP historical price for %s not covered by the current plan (402)", symbol)
            return None
        if resp.status_code == 404:
            _logger.info("No FMP historical price data for %s (404)", symbol)
            return None
        if resp.status_code != 200:
            _logger.error("FMP historical price request for %s failed: status %d", symbol, resp.status_code)
            return None
        data = resp.json()
        if not isinstance(data, list):
            _logger.warning(
                "FMP historical price for %s returned unexpected shape %s — endpoint may have changed again",
                symbol, type(data).__name__,
            )
            return None
        return data

    def search_company_by_name(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Company name search, for resolving a ticker when only a CIK/name is
        on file. Endpoint and response shape live-verified 2026-08-31 (see
        module docstring) — real candidates include `symbol`, `name`,
        `currency`, `exchange`, `exchangeFullName`.

        Returns:
            List of candidate match dicts, possibly empty. Never raises; a
            malformed/unexpected response is logged and treated as "no
            candidates" rather than crashing a bulk backfill run over it.
        """
        params = {"apikey": self._api_key, "query": query, "limit": limit}
        resp = get_with_retry(self._client, f"{FMP_STABLE_URL}/search-name", params=params, rate_limiter=fmp_limiter)
        if resp is None or resp.status_code != 200:
            if resp is not None:
                _logger.warning(
                    "FMP name search for %r failed: status %d — endpoint may need updating, see module docstring",
                    query, resp.status_code,
                )
            return []
        data = resp.json()
        if not isinstance(data, list):
            _logger.warning(
                "FMP name search for %r returned unexpected shape %s — endpoint may need updating, "
                "see module docstring", query, type(data).__name__,
            )
            return []
        return data
