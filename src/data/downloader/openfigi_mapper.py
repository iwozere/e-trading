"""
OpenFIGI CUSIP-to-Ticker Mapper

Maps CUSIP identifiers to stock tickers using the OpenFIGI v3 API.
Maintains a persistent append-only CSV.gz cache so each CUSIP is resolved
only once.

Cache layout:
    DATA_CACHE_DIR/openfigi/
        cusip_map.csv.gz   ← accumulated mapping, columns: cusip, ticker,
                               resolved_at, resolver

Only US-composite listings of equity securities are accepted, and share-class
tickers use the dash form (BRK/B → BRK-B). Cache rows written by an older
resolver (no or different ``resolver`` value) are treated as unknown and
re-resolved on the next lookup.

Free tier: 25 requests/minute, 10 CUSIPs per POST (no key required).
With OPENFIGI_API_KEY env var: 250 requests/minute, 100 CUSIPs per POST.

Classes:
- OpenFigiMapper: CUSIP→ticker mapper with persistent CSV.gz cache
"""

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Union

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

_OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
_BATCH_SIZE_FREE = 10
_BATCH_SIZE_KEYED = 100
# Free tier: 25 req/min → 1 req per 2.4s; keyed: 250/min → 1 per 0.24s
_MIN_INTERVAL_FREE = 2.5
_MIN_INTERVAL_KEYED = 0.25
# Bump when the resolution rules change so cached rows from older rules are re-resolved.
# us_equity_v2: exchCode=US filter, equity security types only, BRK/B → BRK-B.
_RESOLVER_VERSION = "us_equity_v2"
_EQUITY_SECURITY_TYPES = frozenset({"Common Stock", "ADR", "ETP", "REIT", "Closed-End Fund", "MLP"})


class OpenFigiMapper(BaseDataDownloader):
    """
    Maps CUSIPs to stock tickers via the OpenFIGI API v3.

    Maintains a persistent CSV.gz cache at DATA_CACHE_DIR/openfigi/cusip_map.csv.gz.
    Only unknown CUSIPs (not already in the cache) are sent to the API.
    CUSIPs that resolve to non-equity instruments (bonds, options) are stored
    with ticker=None to prevent repeated API calls.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the OpenFIGI mapper.

        Args:
            cache_dir: Root cache directory. Defaults to DATA_CACHE_DIR.
                       Cache is stored at <cache_dir>/openfigi/cusip_map.csv.gz.
            api_key: OpenFIGI API key (optional). If None, uses free tier.
                     Can also be set via OPENFIGI_API_KEY env var or donotshare config.
        """
        super().__init__()
        root = Path(cache_dir) if cache_dir else Path(DATA_CACHE_DIR)
        self._cache_file = root / "openfigi" / "cusip_map.csv.gz"
        self._session = requests.Session()

        key = api_key or self._get_config_value("OPENFIGI_API_KEY")
        if key:
            self._session.headers.update({"X-OPENFIGI-APIKEY": key})
            self._min_interval = _MIN_INTERVAL_KEYED
            self._batch_size = _BATCH_SIZE_KEYED
            _logger.debug("OpenFIGI: using keyed tier (250 req/min)")
        else:
            self._min_interval = _MIN_INTERVAL_FREE
            self._batch_size = _BATCH_SIZE_FREE
            _logger.debug("OpenFIGI: using free tier (25 req/min)")

        self._last_request_time: float = 0.0
        self._cache: Dict[str, str | None] | None = None

    # ------------------------------------------------------------------
    # BaseDataDownloader interface
    # ------------------------------------------------------------------

    def get_provider_name(self) -> str:
        """Return the canonical provider name."""
        return "openfigi"

    def get_supported_intervals(self) -> List[str]:
        """OpenFIGI does not provide OHLCV data."""
        return []

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_date: Any,
        end_date: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Not supported — returns an empty DataFrame."""
        del symbol, interval, start_date, end_date, kwargs
        _logger.warning("OpenFigiMapper does not provide OHLCV data. Use map_cusips() instead.")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def map_cusips(
        self,
        cusips: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, str | None]:
        """
        Resolve a list of CUSIPs to ticker symbols.

        Only fetches CUSIPs that are not already in the local cache. Results are
        appended to the persistent cache so future calls are instant.

        Args:
            cusips: List of 9-character CUSIP strings.
            force_refresh: If True, re-query all CUSIPs regardless of cache.

        Returns:
            Mapping of ``{cusip: ticker_or_None}``. Non-equity instruments and
            unresolvable CUSIPs map to None.
        """
        if not cusips:
            return {}

        cache = self._load_cache() if not force_refresh else {}
        unknown = [c for c in cusips if c not in cache] if not force_refresh else list(cusips)

        if unknown:
            _logger.info("Resolving %d unknown CUSIPs via OpenFIGI ...", len(unknown))
            new_mappings = self._fetch_all(unknown)
            cache.update(new_mappings)
            self._save_cache(new_mappings)

        return {c: cache.get(c) for c in cusips}

    def load_cache(self) -> pd.DataFrame:
        """
        Load the full CSV.gz mapping table as a DataFrame.

        Returns:
            DataFrame with columns: cusip, ticker, resolved_at, resolver.
            Empty DataFrame if no cache exists.
        """
        if not self._cache_file.exists():
            return pd.DataFrame()
        return pd.read_csv(self._cache_file, compression="gzip")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> Dict[str, str | None]:
        """Load the cache dict from CSV.gz, skipping rows written by an older resolver."""
        if self._cache is not None:
            return self._cache
        if not self._cache_file.exists():
            self._cache = {}
            return self._cache
        df = pd.read_csv(self._cache_file, compression="gzip", dtype=str).fillna("")
        resolvers = df["resolver"] if "resolver" in df.columns else pd.Series("", index=df.index)
        current = df[resolvers == _RESOLVER_VERSION]
        mapping: Dict[str, str | None] = {}
        for cusip, ticker in zip(current["cusip"].tolist(), current["ticker"].tolist()):
            mapping[str(cusip)] = str(ticker) if ticker else None
        self._cache = mapping
        stale = len(df) - len(current)
        if stale:
            _logger.info("Ignoring %d CUSIP mappings from an older resolver; they will be re-resolved", stale)
        _logger.debug("Loaded %d CUSIPs from cache %s", len(self._cache), self._cache_file)
        return self._cache

    def _save_cache(self, new_mappings: Dict[str, str | None]) -> None:
        """Append new mappings to the CSV.gz cache."""
        if not new_mappings:
            return

        resolved_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = [
            {"cusip": cusip, "ticker": ticker, "resolved_at": resolved_at, "resolver": _RESOLVER_VERSION}
            for cusip, ticker in new_mappings.items()
        ]

        new_df = pd.DataFrame(rows)

        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        if self._cache_file.exists():
            existing = pd.read_csv(self._cache_file, compression="gzip")
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=["cusip"], keep="last", inplace=True)
        else:
            combined = new_df

        combined.to_csv(self._cache_file, index=False, compression="gzip")
        _logger.info("Saved %d new CUSIP mappings to %s (total: %d)", len(new_df), self._cache_file, len(combined))

    def _fetch_all(self, cusips: List[str]) -> Dict[str, str | None]:
        """Batch-fetch all CUSIPs from OpenFIGI in chunks of the tier's batch size."""
        result: Dict[str, str | None] = {}
        for i in range(0, len(cusips), self._batch_size):
            batch = cusips[i : i + self._batch_size]
            batch_result = self._fetch_batch(batch)
            result.update(batch_result)
            _logger.debug("OpenFIGI: resolved %d/%d CUSIPs", i + len(batch), len(cusips))
        return result

    def _fetch_batch(self, cusips: List[str]) -> Dict[str, str | None]:
        """
        POST up to one batch of CUSIPs to OpenFIGI and return resolved tickers.

        Only US-composite listings are requested: without the exchCode filter
        OpenFIGI's first hit is often a foreign line (DWDPEUR, 6D8, APLSUSD).

        Args:
            cusips: List of CUSIP strings, at most the tier's batch size.

        Returns:
            Mapping of {cusip: ticker_or_None}.
        """
        payload = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in cusips]

        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        try:
            resp = self._session.post(
                _OPENFIGI_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30,
            )
            self._last_request_time = time.monotonic()

            if resp.status_code == 429:
                _logger.warning("OpenFIGI rate limit hit — sleeping 60s")
                time.sleep(60)
                return self._fetch_batch(cusips)

            resp.raise_for_status()
            items = resp.json()
        except requests.HTTPError as exc:
            _logger.warning("OpenFIGI HTTP error: %s", exc)
            return {c: None for c in cusips}
        except Exception:
            _logger.exception("OpenFIGI request failed")
            return {c: None for c in cusips}

        result: Dict[str, str | None] = {}
        for cusip, item in zip(cusips, items):
            result[cusip] = _select_equity_ticker(item.get("data", []))

        return result


def _select_equity_ticker(data: List[Dict[str, Any]]) -> str | None:
    """
    Pick the equity ticker from one OpenFIGI mapping result.

    Bonds, preferreds, warrants and rights map to None (falling back to the
    first hit used to cache bond descriptions like "AAPL 3.25 02/23/26" as
    tickers). Share-class slashes become dashes (BRK/B → BRK-B), the form the
    OHLCV providers use.

    Args:
        data: The ``data`` list of one OpenFIGI mapping job.

    Returns:
        The ticker, or None when no equity listing is present.
    """
    for entry in data:
        ticker = entry.get("ticker")
        if ticker and entry.get("securityType") in _EQUITY_SECURITY_TYPES:
            return str(ticker).replace("/", "-")
    return None
