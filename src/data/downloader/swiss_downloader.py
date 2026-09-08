"""
Swiss Market Data Downloader (SIX Exchange Regulation + Zefix)

Swiss equivalent of ``edgar_downloader.py``. Two free data sources are wired up:

SIX Exchange Regulation (SER AG) — the Swiss counterpart of SEC filings for
SIX-listed companies:
- Significant Shareholders  → Swiss equivalent of SEC Schedule 13D/13G
                              (Art. 120 FinfraG stake disclosures)
- Management Transactions   → Swiss equivalent of SEC Form 4 (insider trades)
- Official Notices          → exchange notices (delistings, sanctions, etc.)

SER publishes no official public API or documented schema for this data —
its search UI (ser-ag.com) is a React SPA. The three feeds above are pulled
from that SPA's own backing JSON endpoints (host "sheldon", found via
browser network-tab inspection 2026-09-08), e.g.:
    GET https://www.ser-ag.com/sheldon/significant_shareholders/v1/overview.json
        ?pageSize=100&pageNumber=0&sortAttribute=byDate&fromDate=YYYYMMDD&toDate=YYYYMMDD
No authentication is required and the response is full structured JSON (the
actual crossed-ownership-threshold voting-rate percentages included) — no
RSS/regex-scraping needed. IMPORTANT: this is an undocumented internal API
with no stability contract; SER could change field names or paths without
notice. The two lookup tables below (``_BUY_SELL_INDICATOR``,
``_OBLIGOR_FUNCTION``) were reverse-engineered by cross-checking these codes
against the SER RSS feed's plain-English descriptions for the same
notification IDs (verified against 4 live notifications, 2026-09-08) — an
unrecognized code degrades to the raw code string (logged), never raises.

Zefix (Central Business Name Index) — Switzerland's official company
registry (run by the Federal Office of Justice's Federal Commercial Registry
Office / EHRA), the Swiss equivalent of EDGAR's company_tickers.json
ticker→CIK mapping. Free REST API, but unlike EDGAR (which only requires a
descriptive User-Agent header) Zefix requires a registered account and HTTP
Basic auth — there is no self-service signup; request access by emailing
zefix@bj.admin.ch (confirmed via https://www.zefix.admin.ch/en/contact,
2026-09-08) and ask for both the integration and production environments in
one message. Credentials are read from ZEFIX_USERNAME / ZEFIX_PASSWORD (env
var or config.donotshare.donotshare), matching the _get_config_value pattern
used by other providers' API keys.

Cache layout (one immutable file per calendar day, mirroring EdgarDownloader's
Form4/13D-G convention — SER's date-range query makes arbitrary-date backfill
possible, unlike a rolling RSS feed):
    DATA_CACHE_DIR/swiss/
        ser/
            significant_shareholders/{date}.csv.gz
            management_transactions/{date}.csv.gz
            official_notices/{date}.csv.gz
        zefix/
            company_<uid>.json            ← one file per looked-up company

Classes:
- SwissDownloader: Main downloader class for SIX/SER disclosure feeds and
  Zefix company-registry lookups.
"""

import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.data.utils.atomic_write import atomic_to_csv
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

# SER's SPA backing JSON API ("sheldon") — undocumented, no auth required.
_SHELDON_BASE = "https://www.ser-ag.com/sheldon"
_SIG_SHAREHOLDERS_URL = f"{_SHELDON_BASE}/significant_shareholders/v1/overview.json"
_MGMT_TXN_URL = f"{_SHELDON_BASE}/management_transactions/v1/overview.json"
_OFFICIAL_NOTICES_URL = f"{_SHELDON_BASE}/official_notices/v2/find.json"
_SHELDON_PAGE_SIZE = 100

# management_transactions "buySellIndicator" / "obligorFunctionCode" — reverse-engineered
# by cross-checking against the SER RSS feed's plain-English descriptions for the same
# notification IDs (T1Q9700014, T1Q9700022, T1Q9400011, T1Q9400045 — verified 2026-09-08).
_BUY_SELL_INDICATOR = {"1": "Purchase", "2": "Sale"}
_OBLIGOR_FUNCTION = {
    "1": "executive member of the board of directors / member of senior management",
    "2": "non-executive member of the board of directors",
}

# Zefix (Central Business Name Index) — Swiss company registry.
_ZEFIX_BASE_URL = "https://www.zefix.admin.ch/ZefixPublicREST/api/v1"

# Neither SER nor Zefix document a request-rate limit; throttle politely.
_MIN_REQUEST_INTERVAL = 1.0

_SIG_SHAREHOLDERS_COLS = [
    "filing_id",
    "company",
    "company_id",
    "category",
    "publication_date",
    "transaction_date",
    "purchase_total_voting_rate",
    "sale_total_voting_rate",
    "below_threshold_voting_rate",
    "beneficial_names",
    "shareholder_names",
    "trigger_comment",
]
_MGMT_TXN_COLS = [
    "filing_id",
    "company",
    "company_id",
    "isin",
    "transaction_date",
    "action",
    "quantity",
    "price_per_security_chf",
    "total_value_chf",
    "actor_role",
    "security_type_code",
    "security_description",
]
_OFFICIAL_NOTICES_COLS = ["filing_id", "date", "notice_type", "contact", "title", "isin"]


class SwissDownloader(BaseDataDownloader):
    """
    Swiss market data downloader covering SIX Exchange Regulation disclosure
    feeds and Zefix company-registry lookups.

    Caches everything under DATA_CACHE_DIR/swiss/, one immutable file per
    calendar day for the SER feeds (like EdgarDownloader's Form4/13D-G
    caching), so arbitrary historical dates can be backfilled on demand.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] | None = None,
        zefix_username: Optional[str] = None,
        zefix_password: Optional[str] = None,
    ):
        """
        Initialize the Swiss downloader.

        Args:
            cache_dir: Root cache directory. Defaults to DATA_CACHE_DIR.
                       Swiss files are stored under <cache_dir>/swiss/.
            zefix_username: Zefix API account username. Falls back to the
                             ZEFIX_USERNAME env var / donotshare config.
            zefix_password: Zefix API account password. Falls back to the
                             ZEFIX_PASSWORD env var / donotshare config.
        """
        super().__init__()
        self._cache_dir = Path(cache_dir or DATA_CACHE_DIR) / "swiss"
        self._ser_dir = self._cache_dir / "ser"
        self._zefix_dir = self._cache_dir / "zefix"
        self._ser_dir.mkdir(parents=True, exist_ok=True)
        self._zefix_dir.mkdir(parents=True, exist_ok=True)

        self._zefix_username = zefix_username or self._get_config_value("ZEFIX_USERNAME")
        self._zefix_password = zefix_password or self._get_config_value("ZEFIX_PASSWORD")
        self._last_request_time = 0.0

    def get_provider_name(self) -> str:
        """Return the canonical provider name for this downloader."""
        return "swiss"

    def get_supported_intervals(self) -> List[str]:
        """This downloader serves disclosure filings and registry lookups, not OHLCV bars."""
        return []

    def get_ohlcv(
        self, symbol: str, interval: str, start_date: datetime, end_date: datetime, **kwargs: Any
    ) -> Optional[pd.DataFrame]:
        """
        Not applicable — SwissDownloader has no price-data source. Always returns None.

        Args:
            symbol: Unused.
            interval: Unused.
            start_date: Unused.
            end_date: Unused.
            **kwargs: Unused.
        """
        return None

    def _throttle(self) -> None:
        """Enforce a minimum gap between outbound requests (polite default; no documented limit)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # SIX Exchange Regulation (SER AG) — "sheldon" JSON API
    # ------------------------------------------------------------------

    def _fetch_all_sheldon_items(self, url: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Page through a sheldon ``overview.json``/``find.json`` endpoint and
        return the concatenated ``itemList`` across all pages.

        Args:
            url: Full endpoint URL (one of the module-level _*_URL constants).
            params: Query params EXCLUDING pageNumber/pageSize (added per page).

        Returns:
            All items across every page. Stops early (with a warning) if the
            response shape is unexpected, rather than looping forever.
        """
        items: List[Dict[str, Any]] = []
        page = 0
        total_count: Optional[int] = None
        while total_count is None or len(items) < total_count:
            self._throttle()
            query = {**params, "pageNumber": page, "pageSize": _SHELDON_PAGE_SIZE}
            resp = requests.get(url, params=query, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            if body.get("status") != "Ok":
                _logger.warning("Unexpected status from %s: %r", url, body.get("status"))
                break
            total_count = body.get("totalCount", 0)
            page_items = body.get("itemList", [])
            if not page_items:
                break
            items.extend(page_items)
            page += 1
        return items

    def download_significant_shareholders(self, as_of_date: date | None = None, force: bool = False) -> pd.DataFrame:
        """
        Download significant-shareholder disclosures (Art. 120 FinfraG — the
        Swiss equivalent of SEC Schedule 13D/13G) published on a given date,
        including the actual crossed-ownership voting-rate percentages.

        Cached as DATA_CACHE_DIR/swiss/ser/significant_shareholders/{date}.csv.gz.

        Args:
            as_of_date: Publication date to fetch. Defaults to yesterday.
            force: Re-download even if cached.

        Returns:
            DataFrame with columns: filing_id, company, company_id, category,
            publication_date, transaction_date, purchase_total_voting_rate,
            sale_total_voting_rate, below_threshold_voting_rate,
            beneficial_names, shareholder_names, trigger_comment.
        """
        target_date = as_of_date or (datetime.now(timezone.utc).date() - timedelta(days=1))
        date_str = str(target_date)
        dest = self._ser_dir / "significant_shareholders" / f"{date_str}.csv.gz"

        if dest.exists() and not force:
            _logger.info("Significant shareholders for %s already cached at %s", date_str, dest)
            return pd.read_csv(dest, compression="gzip")

        ymd = target_date.strftime("%Y%m%d")
        _logger.info("Downloading significant shareholder disclosures for %s ...", date_str)
        raw_items = self._fetch_all_sheldon_items(
            _SIG_SHAREHOLDERS_URL, {"sortAttribute": "byDate", "fromDate": ymd, "toDate": ymd}
        )
        records = [_flatten_significant_shareholder(item) for item in raw_items]
        df = (
            pd.DataFrame(records, columns=_SIG_SHAREHOLDERS_COLS)  # type: ignore[arg-type]
            if records
            else pd.DataFrame(columns=_SIG_SHAREHOLDERS_COLS)  # type: ignore[arg-type]
        )
        atomic_to_csv(df, dest, index=False, compression="gzip")
        _logger.info("Cached %d significant shareholder disclosure(s) for %s -> %s", len(df), date_str, dest)
        return df

    def download_management_transactions(self, as_of_date: date | None = None, force: bool = False) -> pd.DataFrame:
        """
        Download management-transaction disclosures (the Swiss equivalent of
        SEC Form 4 insider transactions) filed on a given date.

        Cached as DATA_CACHE_DIR/swiss/ser/management_transactions/{date}.csv.gz.

        Args:
            as_of_date: Transaction date to fetch. Defaults to yesterday.
            force: Re-download even if cached.

        Returns:
            DataFrame with columns: filing_id, company, company_id, isin,
            transaction_date, action, quantity, price_per_security_chf,
            total_value_chf, actor_role, security_type_code,
            security_description. An unrecognized action/role code comes back
            as its raw code string (logged), never raises.
        """
        target_date = as_of_date or (datetime.now(timezone.utc).date() - timedelta(days=1))
        date_str = str(target_date)
        dest = self._ser_dir / "management_transactions" / f"{date_str}.csv.gz"

        if dest.exists() and not force:
            _logger.info("Management transactions for %s already cached at %s", date_str, dest)
            return pd.read_csv(dest, compression="gzip")

        ymd = target_date.strftime("%Y%m%d")
        _logger.info("Downloading management transactions for %s ...", date_str)
        raw_items = self._fetch_all_sheldon_items(
            _MGMT_TXN_URL, {"sortAttribute": "byDate", "fromDate": ymd, "toDate": ymd}
        )
        records = [_flatten_management_transaction(item) for item in raw_items]
        df = (
            pd.DataFrame(records, columns=_MGMT_TXN_COLS) if records else pd.DataFrame(columns=_MGMT_TXN_COLS)  # type: ignore[arg-type]
        )
        atomic_to_csv(df, dest, index=False, compression="gzip")
        _logger.info("Cached %d management transaction(s) for %s -> %s", len(df), date_str, dest)
        return df

    def download_official_notices(self, as_of_date: date | None = None, force: bool = False) -> pd.DataFrame:
        """
        Download official exchange notices (delistings, sanctions, rule-based
        parameter adjustments, etc.) published on a given date.

        Cached as DATA_CACHE_DIR/swiss/ser/official_notices/{date}.csv.gz.

        Args:
            as_of_date: Publication date to fetch. Defaults to yesterday.
            force: Re-download even if cached.

        Returns:
            DataFrame with columns: filing_id, date, notice_type, contact,
            title, isin.
        """
        target_date = as_of_date or (datetime.now(timezone.utc).date() - timedelta(days=1))
        date_str = str(target_date)
        dest = self._ser_dir / "official_notices" / f"{date_str}.csv.gz"

        if dest.exists() and not force:
            _logger.info("Official notices for %s already cached at %s", date_str, dest)
            return pd.read_csv(dest, compression="gzip")

        ymd = target_date.strftime("%Y%m%d")
        _logger.info("Downloading official notices for %s ...", date_str)
        raw_items = self._fetch_all_sheldon_items(
            _OFFICIAL_NOTICES_URL,
            {"firstDate": ymd, "lastDate": ymd, "sortAttribute": "dateTime", "sortDirection": "desc"},
        )
        records = [_flatten_official_notice(item) for item in raw_items]
        df = (
            pd.DataFrame(records, columns=_OFFICIAL_NOTICES_COLS)  # type: ignore[arg-type]
            if records
            else pd.DataFrame(columns=_OFFICIAL_NOTICES_COLS)  # type: ignore[arg-type]
        )
        atomic_to_csv(df, dest, index=False, compression="gzip")
        _logger.info("Cached %d official notice(s) for %s -> %s", len(df), date_str, dest)
        return df

    # ------------------------------------------------------------------
    # Zefix (Central Business Name Index) — Swiss company registry
    # ------------------------------------------------------------------

    def _zefix_request(self, method: str, path: str, **kwargs: Any) -> Any:
        """
        Issue an authenticated Zefix API request.

        Args:
            method: HTTP method ("GET" or "POST").
            path: Path under _ZEFIX_BASE_URL (e.g. "/company/uid/{uid}").
            **kwargs: Forwarded to requests.request (e.g. json=... for POST).

        Returns:
            Parsed JSON response body.

        Raises:
            RuntimeError: If no Zefix credentials are configured.
            requests.HTTPError: On a non-2xx response.
        """
        if not (self._zefix_username and self._zefix_password):
            raise RuntimeError(
                "Zefix credentials not configured — set ZEFIX_USERNAME / ZEFIX_PASSWORD "
                "(request an account by emailing zefix@bj.admin.ch — no self-service signup exists)"
            )
        self._throttle()
        resp = requests.request(
            method,
            f"{_ZEFIX_BASE_URL}{path}",
            auth=(self._zefix_username, self._zefix_password),
            timeout=30,
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json()

    def search_company(self, name: str, active_only: bool = True) -> pd.DataFrame:
        """
        Search Zefix by company name — the Swiss equivalent of EDGAR's
        company_tickers.json ticker→CIK lookup (Zefix has no ticker concept;
        it indexes by legal name / UID / canton instead).

        Args:
            name: Company name (or fragment) to search for.
            active_only: Restrict to currently active companies.

        Returns:
            DataFrame with columns: name, ehraid, uid, chid, legalSeatId,
            legalSeat, registryOfCommerceId, legalForm, status, sogcDate,
            deletionDate (one row per match).
        """
        data = self._zefix_request(
            "POST", "/company/search", json={"name": name, "activeOnly": active_only}
        )
        return pd.DataFrame(data)

    def get_company_by_uid(self, uid: str, force: bool = False) -> Dict[str, Any]:
        """
        Fetch the full company record for a Zefix UID (format CHE-xxx.xxx.xxx)
        — the Swiss equivalent of EDGAR's per-CIK submissions.json lookup.

        Cached as DATA_CACHE_DIR/swiss/zefix/company_{uid}.json (dots
        stripped from the filename).

        Args:
            uid: Company UID, e.g. "CHE-106.588.217".
            force: Re-fetch even if a cached copy exists.

        Returns:
            Full company record dict (name, address, purpose, capital,
            legal form, audit company, etc.).
        """
        safe_uid = uid.replace(".", "").replace(" ", "")
        dest = self._zefix_dir / f"company_{safe_uid}.json"
        if dest.exists() and not force:
            import json

            return json.loads(dest.read_text(encoding="utf-8"))

        data = self._zefix_request("GET", f"/company/uid/{uid}")

        import json

        dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        _logger.info("Cached Zefix company record for %s -> %s", uid, dest)
        return data


def _yyyymmdd_to_iso(value: Any) -> Optional[str]:
    """Convert a sheldon-style YYYYMMDD int (e.g. 20260907, or falsy/0 for "unset") to ISO date string."""
    if not value:
        return None
    digits = str(int(value))
    return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"


def _flatten_significant_shareholder(item: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one sheldon significant_shareholders itemList entry into a cache row."""
    pub = item.get("publication", {}) or {}
    return {
        "filing_id": pub.get("notificationId"),
        "company": pub.get("notificationSubmitter"),
        "company_id": pub.get("notificationSubmitterId"),
        "category": pub.get("category"),
        "publication_date": _yyyymmdd_to_iso(pub.get("publicationDate")),
        "transaction_date": _yyyymmdd_to_iso(pub.get("transactionDate")),
        "purchase_total_voting_rate": pub.get("purchaseTotalVotingRate"),
        "sale_total_voting_rate": pub.get("saleTotalVotingRate"),
        "below_threshold_voting_rate": pub.get("belowThresholdVotingRate"),
        "beneficial_names": "; ".join(item.get("beneficialNames") or []),
        "shareholder_names": "; ".join(item.get("shareholderNames") or []),
        "trigger_comment": " ".join(pub.get("triggerComment") or []),
    }


def _flatten_management_transaction(item: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one sheldon management_transactions itemList entry into a cache row."""
    buy_sell_code = item.get("buySellIndicator", "")
    obligor_code = item.get("obligorFunctionCode", "")
    notification_id = item.get("notificationId")
    if buy_sell_code not in _BUY_SELL_INDICATOR:
        _logger.warning("Unknown buySellIndicator %r in management transaction %s", buy_sell_code, notification_id)
    if obligor_code not in _OBLIGOR_FUNCTION:
        _logger.warning("Unknown obligorFunctionCode %r in management transaction %s", obligor_code, notification_id)
    return {
        "filing_id": notification_id,
        "company": item.get("notificationSubmitter"),
        "company_id": item.get("notificationSubmitterId"),
        "isin": item.get("ISIN"),
        "transaction_date": _yyyymmdd_to_iso(item.get("transactionDate")),
        "action": _BUY_SELL_INDICATOR.get(buy_sell_code, buy_sell_code),
        "quantity": item.get("transactionSize"),
        "price_per_security_chf": item.get("transactionAmountPerSecurityCHF"),
        "total_value_chf": item.get("transactionAmountCHF"),
        "actor_role": _OBLIGOR_FUNCTION.get(obligor_code, obligor_code),
        "security_type_code": item.get("securityTypeCode"),
        "security_description": item.get("securityDescription"),
    }


def _flatten_official_notice(item: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one sheldon official_notices itemList entry into a cache row."""
    return {
        "filing_id": item.get("noticeId"),
        "date": _yyyymmdd_to_iso(item.get("date")),
        "notice_type": item.get("noticeType"),
        "contact": item.get("contact"),
        "title": item.get("title"),
        "isin": item.get("isin"),
    }


if __name__ == "__main__":
    import argparse
    import json as json_module

    parser = argparse.ArgumentParser(description="Download Swiss SER/Zefix data to local cache.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_shareholders = subparsers.add_parser("significant-shareholders", help="Download for one publication date")
    p_shareholders.add_argument("--date", type=str, default=None, help="ISO date, e.g. 2026-09-07 (default: yesterday)")
    p_shareholders.add_argument("--force", action="store_true")

    p_mgmt = subparsers.add_parser("management-transactions", help="Download for one transaction date")
    p_mgmt.add_argument("--date", type=str, default=None, help="ISO date, e.g. 2026-09-07 (default: yesterday)")
    p_mgmt.add_argument("--force", action="store_true")

    p_notices = subparsers.add_parser("official-notices", help="Download for one publication date")
    p_notices.add_argument("--date", type=str, default=None, help="ISO date, e.g. 2026-09-07 (default: yesterday)")
    p_notices.add_argument("--force", action="store_true")

    p_search = subparsers.add_parser("zefix-search", help="Search Zefix by company name")
    p_search.add_argument("name", type=str, help="Company name or fragment")

    p_uid = subparsers.add_parser("zefix-company", help="Fetch a Zefix company record by UID")
    p_uid.add_argument("uid", type=str, help="Company UID, e.g. CHE-106.588.217")
    p_uid.add_argument("--force", action="store_true")

    parser.add_argument("--cache-dir", type=str, default=None, help=f"Cache root (default: {DATA_CACHE_DIR})")

    args = parser.parse_args()
    dl = SwissDownloader(cache_dir=args.cache_dir)

    if args.command in ("significant-shareholders", "management-transactions", "official-notices"):
        as_of = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
        method = {
            "significant-shareholders": dl.download_significant_shareholders,
            "management-transactions": dl.download_management_transactions,
            "official-notices": dl.download_official_notices,
        }[args.command]
        df = method(as_of_date=as_of, force=args.force)
        print(f"__SCHEDULER_RESULT__:{json_module.dumps({'success': True, 'row_count': len(df)})}")

    elif args.command == "zefix-search":
        df = dl.search_company(args.name)
        result = {"success": True, "count": len(df), "companies": df.to_dict(orient="records")}
        print(f"__SCHEDULER_RESULT__:{json_module.dumps(result)}")

    elif args.command == "zefix-company":
        record = dl.get_company_by_uid(args.uid, force=args.force)
        print(f"__SCHEDULER_RESULT__:{json_module.dumps({'success': True, 'company': record})}")
