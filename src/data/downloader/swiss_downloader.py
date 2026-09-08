"""
Swiss Market Data Downloader (SIX Exchange Regulation + Zefix)

Swiss equivalent of ``edgar_downloader.py``. Two free data sources are wired up:

SIX Exchange Regulation (SER AG) — the Swiss counterpart of SEC filings for
SIX-listed companies. Unlike EDGAR there is no bulk JSON/XBRL API; SER's
disclosure platform (OLSDigital) is a web UI. The one free, structured,
no-auth surface is its RSS feeds (2-minute TTL), which this module polls
incrementally:
- Significant Shareholders  → Swiss equivalent of SEC Schedule 13D/13G
                              (Art. 120 FinfraG stake disclosures)
- Management Transactions   → Swiss equivalent of SEC Form 4 (insider trades)
- Official Notices          → exchange notices (delistings, sanctions, etc.)

IMPORTANT LIMITATION: the Significant Shareholders feed only carries the
company name + a link to the disclosure detail page — SER does not publish
the crossed-ownership-threshold percentage in any machine-readable feed. The
detail page is a client-side SPA route (``#/shareholder-details/{id}``)
backed by a JSON API this module has not yet located (needs live browser
network-tab inspection, e.g. via the claude-in-chrome skill, to confirm the
call before it can be wired up). Management Transactions, by contrast, embed
everything needed (action/quantity/price/total/role) directly in the RSS
``description`` as plain text, which this module parses via regex.

Zefix (Central Business Name Index) — Switzerland's official company
registry (run by the Federal Office of the Commercial Registry / EHRA), the
Swiss equivalent of EDGAR's company_tickers.json ticker→CIK mapping. Free
REST API, but unlike EDGAR (which only requires a descriptive User-Agent
header) Zefix requires a registered account and HTTP Basic auth — see
https://www.zefix.admin.ch/ZefixPublicREST/swagger-ui/index.html for the live
spec and registration link. Credentials are read from ZEFIX_USERNAME /
ZEFIX_PASSWORD (env var or config.donotshare.donotshare), matching the
_get_config_value pattern used by other providers' API keys.

Cache layout:
    DATA_CACHE_DIR/swiss/
        ser/
            significant_shareholders.csv  ← incremental, deduped by filing_id
            management_transactions.csv   ← incremental, deduped by filing_id
            official_notices.csv          ← incremental, deduped by filing_id
        zefix/
            company_<uid>.json            ← one file per looked-up company

Classes:
- SwissDownloader: Main downloader class for SIX/SER disclosure feeds and
  Zefix company-registry lookups.

Verified against live endpoints 2026-09-07; SER has no documented API
stability guarantee for the RSS feeds or the description text template, so
treat both as best-effort and re-verify if parsing starts silently dropping
fields.
"""

import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from xml.etree import ElementTree as ET

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

# SIX Exchange Regulation (SER AG) RSS feeds — free, no auth required.
_SER_FEEDS = {
    "significant_shareholders": "https://www.ser-ag.com/itf-data/significant-shareholders/rss-en.xml",
    "management_transactions": "https://www.ser-ag.com/itf-data/management-transactions/rss-en.xml",
    "official_notices": "https://www.ser-ag.com/itf-data/official-notices/rss-en.xml",
}

# guid/link format: ".../{feed-page}.html#/{route}/{FILING_ID}" — the trailing
# path segment is the stable per-filing identifier used for dedup.
_FILING_ID_RE = re.compile(r"/([^/#]+)$")

# Management Transactions <description> follows a fixed template, verified
# against live feed items 2026-09-07, e.g.:
#   "Purchase of 128 securities amounting to CHF 32,103.76 (CHF 250.81 /
#    security) by a non-executive member of the board of directors"
#   "Sale of 14,885 securities amounting to CHF 46,217.93 (CHF 3.11 /
#    security) by an executive member of the board of directors / member of
#    senior management"
# Numbers use "," as thousands separator and "." as decimal point.
_MGMT_TXN_PATTERN = re.compile(
    r"^(?P<action>\w+)\s+of\s+(?P<quantity>[\d,]+)\s+securities\s+amounting\s+to\s+CHF\s+"
    r"(?P<total_chf>[\d,]+\.\d+)\s+\(CHF\s+(?P<price_chf>[\d,]+\.\d+)\s*/\s*security\)\s+"
    r"by\s+(?P<actor_role>.+)$"
)

# Zefix (Central Business Name Index) — Swiss company registry.
_ZEFIX_BASE_URL = "https://www.zefix.admin.ch/ZefixPublicREST/api/v1"

# Neither SER nor Zefix document a request-rate limit; throttle politely.
_MIN_REQUEST_INTERVAL = 1.0

_SER_COLS = ["filing_id", "company", "pub_date", "link", "description", "fetched_at"]
_MGMT_TXN_COLS = [
    "filing_id",
    "company",
    "pub_date",
    "link",
    "action",
    "quantity",
    "price_per_security_chf",
    "total_value_chf",
    "actor_role",
    "fetched_at",
]


class SwissDownloader(BaseDataDownloader):
    """
    Swiss market data downloader covering SIX Exchange Regulation disclosure
    feeds and Zefix company-registry lookups.

    Caches everything under DATA_CACHE_DIR/swiss/. The SER feeds are polled
    incrementally (each call fetches the current rolling feed and appends only
    new items, deduped by filing_id) since SER exposes no historical query —
    unlike EDGAR's full-text search, a missed poll window means those items
    are gone for good, so this should run frequently (the feed TTL is 2
    minutes) if genuine completeness matters.
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
    # SIX Exchange Regulation (SER AG) RSS feeds
    # ------------------------------------------------------------------

    def _fetch_ser_feed_items(self, feed_name: str) -> List[Dict[str, str]]:
        """
        Fetch and parse the raw RSS items for one SER feed.

        Args:
            feed_name: Key into _SER_FEEDS.

        Returns:
            List of dicts with keys: title, link, category, description,
            pub_date, guid, filing_id.
        """
        url = _SER_FEEDS[feed_name]
        self._throttle()
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        items = []
        for item in root.findall("./channel/item"):
            guid = (item.findtext("guid") or "").strip()
            link = (item.findtext("link") or "").strip()
            match = _FILING_ID_RE.search(guid or link)
            items.append(
                {
                    "title": (item.findtext("title") or "").strip(),
                    "link": link,
                    "category": (item.findtext("category") or "").strip(),
                    "description": (item.findtext("description") or "").strip(),
                    "pub_date": (item.findtext("pubDate") or "").strip(),
                    "guid": guid,
                    "filing_id": match.group(1) if match else guid,
                }
            )
        return items

    def _download_ser_feed_incremental(self, feed_name: str, columns: List[str]) -> pd.DataFrame:
        """
        Fetch a SER feed and append any not-yet-seen items to its running cache.

        Args:
            feed_name: Key into _SER_FEEDS.
            columns: Column order for the cached CSV (feed-specific).

        Returns:
            Full accumulated DataFrame (existing cache + any new rows).
        """
        dest = self._ser_dir / f"{feed_name}.csv"
        existing = pd.read_csv(dest) if dest.exists() else pd.DataFrame(columns=columns)  # type: ignore[arg-type]
        existing_ids = set(existing["filing_id"]) if not existing.empty else set()

        items = self._fetch_ser_feed_items(feed_name)
        new_items = [it for it in items if it["filing_id"] not in existing_ids]
        if not new_items:
            _logger.info(
                "No new %s items (feed returned %d, all already cached)", feed_name, len(items)
            )
            return existing

        fetched_at = datetime.now(timezone.utc).isoformat()
        new_rows = [self._ser_item_to_row(feed_name, it, fetched_at) for it in new_items]
        new_df = pd.DataFrame(new_rows, columns=columns)  # type: ignore[arg-type]
        combined = pd.concat([existing, new_df], ignore_index=True)
        atomic_to_csv(combined, dest, index=False)
        _logger.info("Cached %d new %s item(s) -> %s", len(new_rows), feed_name, dest)
        return combined

    @staticmethod
    def _ser_item_to_row(feed_name: str, item: Dict[str, str], fetched_at: str) -> Dict[str, Any]:
        """Map one raw RSS item to its feed-specific output row (mgmt transactions get parsed)."""
        base: Dict[str, Any] = {
            "filing_id": item["filing_id"],
            "company": item["title"],
            "pub_date": item["pub_date"],
            "link": item["link"],
            "fetched_at": fetched_at,
        }
        if feed_name == "management_transactions":
            parsed = _parse_management_transaction(item["description"])
            base.update(parsed)
        else:
            base["description"] = item["description"]
        return base

    def download_significant_shareholders(self) -> pd.DataFrame:
        """
        Download recent significant-shareholder disclosures (Art. 120
        FinfraG — the Swiss equivalent of SEC Schedule 13D/13G) from the SER
        RSS feed, appending any not-yet-cached items.

        NOTE: the feed does not carry the crossed-ownership-threshold
        percentage — only company name + a link to the SPA detail page. Pull
        that page's underlying JSON API (not yet identified) if the actual
        stake percentage is needed downstream.

        Cached as DATA_CACHE_DIR/swiss/ser/significant_shareholders.csv.

        Returns:
            DataFrame with columns: filing_id, company, pub_date, link,
            description, fetched_at.
        """
        return self._download_ser_feed_incremental("significant_shareholders", _SER_COLS)

    def download_management_transactions(self) -> pd.DataFrame:
        """
        Download recent management-transaction disclosures (the Swiss
        equivalent of SEC Form 4 insider transactions) from the SER RSS feed,
        appending any not-yet-cached items with action/quantity/price parsed
        out of the description text.

        Cached as DATA_CACHE_DIR/swiss/ser/management_transactions.csv.

        Returns:
            DataFrame with columns: filing_id, company, pub_date, link,
            action, quantity, price_per_security_chf, total_value_chf,
            actor_role, fetched_at. Any field the regex fails to match on a
            future description-template change comes back as None rather
            than raising.
        """
        return self._download_ser_feed_incremental("management_transactions", _MGMT_TXN_COLS)

    def download_official_notices(self) -> pd.DataFrame:
        """
        Download recent official exchange notices (delistings, sanctions,
        etc.) from the SER RSS feed, appending any not-yet-cached items.

        Cached as DATA_CACHE_DIR/swiss/ser/official_notices.csv.

        Returns:
            DataFrame with columns: filing_id, company, pub_date, link,
            description, fetched_at.
        """
        return self._download_ser_feed_incremental("official_notices", _SER_COLS)

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
                "(register a free account at https://www.zefix.admin.ch/ZefixPublicREST/)"
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


def _parse_management_transaction(description: str) -> Dict[str, Optional[str]]:
    """
    Best-effort parse of a Management Transactions RSS description into
    structured fields. Returns all-None values (rather than raising) if the
    description doesn't match the known template, since SER publishes no
    formal schema for this text and the template could change silently.

    Args:
        description: Raw <description> text, e.g. "Purchase of 128 securities
            amounting to CHF 32,103.76 (CHF 250.81 / security) by a
            non-executive member of the board of directors".

    Returns:
        Dict with keys: action, quantity, price_per_security_chf,
        total_value_chf, actor_role.
    """
    match = _MGMT_TXN_PATTERN.match(description)
    if not match:
        _logger.warning("Management transaction description did not match known template: %r", description)
        return {
            "action": None,
            "quantity": None,
            "price_per_security_chf": None,
            "total_value_chf": None,
            "actor_role": None,
        }
    return {
        "action": match.group("action"),
        "quantity": match.group("quantity").replace(",", ""),
        "price_per_security_chf": match.group("price_chf").replace(",", ""),
        "total_value_chf": match.group("total_chf").replace(",", ""),
        "actor_role": match.group("actor_role"),
    }


if __name__ == "__main__":
    import argparse
    import json as json_module

    parser = argparse.ArgumentParser(description="Download Swiss SER/Zefix data to local cache.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("significant-shareholders", help="Poll the Significant Shareholders RSS feed")
    subparsers.add_parser("management-transactions", help="Poll the Management Transactions RSS feed")
    subparsers.add_parser("official-notices", help="Poll the Official Notices RSS feed")

    p_search = subparsers.add_parser("zefix-search", help="Search Zefix by company name")
    p_search.add_argument("name", type=str, help="Company name or fragment")

    p_uid = subparsers.add_parser("zefix-company", help="Fetch a Zefix company record by UID")
    p_uid.add_argument("uid", type=str, help="Company UID, e.g. CHE-106.588.217")
    p_uid.add_argument("--force", action="store_true")

    parser.add_argument("--cache-dir", type=str, default=None, help=f"Cache root (default: {DATA_CACHE_DIR})")

    args = parser.parse_args()
    dl = SwissDownloader(cache_dir=args.cache_dir)

    if args.command == "significant-shareholders":
        df = dl.download_significant_shareholders()
        print(f"__SCHEDULER_RESULT__:{json_module.dumps({'success': True, 'row_count': len(df)})}")

    elif args.command == "management-transactions":
        df = dl.download_management_transactions()
        print(f"__SCHEDULER_RESULT__:{json_module.dumps({'success': True, 'row_count': len(df)})}")

    elif args.command == "official-notices":
        df = dl.download_official_notices()
        print(f"__SCHEDULER_RESULT__:{json_module.dumps({'success': True, 'row_count': len(df)})}")

    elif args.command == "zefix-search":
        df = dl.search_company(args.name)
        result = {"success": True, "count": len(df), "companies": df.to_dict(orient="records")}
        print(f"__SCHEDULER_RESULT__:{json_module.dumps(result)}")

    elif args.command == "zefix-company":
        record = dl.get_company_by_uid(args.uid, force=args.force)
        print(f"__SCHEDULER_RESULT__:{json_module.dumps({'success': True, 'company': record})}")
