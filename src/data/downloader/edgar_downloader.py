"""
SEC EDGAR Data Downloader

This module provides functionality to download company tickers, company facts,
and company submissions from the SEC EDGAR API and cache them locally in
DATA_CACHE_DIR/edgar/.

Main Features:
- Download company_tickers.json (maps tickers to CIK numbers)
- Download individual company facts JSON files (XBRL financial data)
- Download company submissions JSON files (filing history, useful for 8-K tracking)
- Bulk download with rate-limiting (SEC allows max 10 req/sec)
- Skip already-cached files to support incremental updates

Cache layout:
    DATA_CACHE_DIR/edgar/
        company_tickers.json              ← ticker → CIK mapping
        companyfacts/
            CIK0000320193.json            ← XBRL financial facts (large, ~MB)
        submissions/
            CIK0000320193.json            ← filing history (light, updated daily)

Classes:
- EdgarDownloader: Main downloader class for SEC EDGAR data
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import requests
import pandas as pd

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

# SEC EDGAR API endpoints
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANY_FACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# SEC Fair Access Policy: no more than 10 requests per second
_MIN_REQUEST_INTERVAL = 0.11


class EdgarDownloader(BaseDataDownloader):
    """
    SEC EDGAR Data Downloader.

    Downloads company tickers, XBRL company facts, and filing submissions
    from the SEC EDGAR API and caches them to DATA_CACHE_DIR/edgar/.

    SEC Fair Access Policy requires a descriptive User-Agent header and limits
    requests to 10 per second — this class enforces that automatically.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        user_agent: str = "e-trading-research akossyrev@gmail.com",
    ):
        """
        Initialize the EDGAR downloader.

        Args:
            cache_dir: Root cache directory. Defaults to DATA_CACHE_DIR.
                       EDGAR files are stored under <cache_dir>/edgar/.
            user_agent: Value for the HTTP User-Agent header.
                        SEC requires a descriptive string with a contact e-mail.
        """
        super().__init__()
        root = Path(cache_dir) if cache_dir else Path(DATA_CACHE_DIR)
        self._edgar_dir = root / "edgar"
        self._companyfacts_dir = self._edgar_dir / "companyfacts"
        self._submissions_dir = self._edgar_dir / "submissions"
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # BaseDataDownloader interface
    # ------------------------------------------------------------------

    def get_provider_name(self) -> str:
        """Return the canonical provider name."""
        return "edgar"

    def get_supported_intervals(self) -> List[str]:
        """EDGAR provides fundamentals/filings, not interval-based OHLCV data."""
        return []

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Not supported by EDGAR — returns an empty DataFrame.

        Args:
            symbol: Ticker symbol (unused).
            interval: Data interval (unused).
            start_date: Start date (unused).
            end_date: End date (unused).
            **kwargs: Additional parameters (unused).

        Returns:
            Empty DataFrame.
        """
        del symbol, interval, start_date, end_date, kwargs
        _logger.warning("EDGAR does not provide OHLCV data. Use download_company_facts() instead.")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # company_tickers
    # ------------------------------------------------------------------

    def download_company_tickers(self, force: bool = False) -> Path:
        """
        Download company_tickers.json from SEC EDGAR and cache it locally.

        The file maps sequential index keys to objects containing:
        - ``cik_str``: Zero-padded 10-digit CIK string
        - ``ticker``: Uppercase ticker symbol
        - ``title``: Company name

        Args:
            force: If True, re-download even when the cached file already exists.

        Returns:
            Path to the cached company_tickers.json file.
        """
        dest = self._edgar_dir / "company_tickers.json"
        if dest.exists() and not force:
            _logger.info("company_tickers.json already cached at %s", dest)
            return dest

        _logger.info("Downloading company_tickers.json from SEC EDGAR ...")
        data = self._get(_COMPANY_TICKERS_URL)
        self._write_json(data, dest)
        _logger.info("Saved company_tickers.json (%d companies) to %s", len(data), dest)
        return dest

    def load_company_tickers(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Load company_tickers.json from the local cache (downloading if absent).

        Args:
            force_refresh: If True, re-download before loading.

        Returns:
            Parsed JSON dict mapping index strings to ticker/CIK/title objects.
        """
        path = self.download_company_tickers(force=force_refresh)
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # companyfacts
    # ------------------------------------------------------------------

    def download_company_facts(self, cik: Union[int, str], force: bool = False) -> Optional[Path]:
        """
        Download the XBRL company facts JSON for a single CIK.

        Saved as DATA_CACHE_DIR/edgar/companyfacts/<CIK10>.json.

        Args:
            cik: CIK number as int or string (leading zeros are accepted).
            force: If True, re-download even when the cached file already exists.

        Returns:
            Path to the cached JSON file, or None if the download failed.
        """
        cik_int = _parse_cik(cik)
        dest = self._companyfacts_dir / f"{cik_int:010d}.json"

        if dest.exists() and not force:
            _logger.debug("Company facts for CIK %010d already cached at %s", cik_int, dest)
            return dest

        url = _COMPANY_FACTS_URL_TEMPLATE.format(cik=cik_int)
        data = self._fetch(url, cik_int, label="companyfacts")
        if data is None:
            return None

        self._write_json(data, dest)
        _logger.debug("Saved company facts for CIK %010d to %s", cik_int, dest)
        return dest

    def download_all_company_facts(
        self,
        cik_list: Optional[List[Union[int, str]]] = None,
        force: bool = False,
        max_errors: int = 50,
    ) -> Dict[str, Any]:
        """
        Bulk-download company facts for a list of CIKs (or all tickers if omitted).

        When ``cik_list`` is None the method downloads company_tickers.json first
        (if not already cached) and uses every CIK found there.

        Args:
            cik_list: Explicit list of CIK numbers to download.
                      If None, downloads facts for all CIKs in company_tickers.json.
            force: If True, re-download files that already exist in the cache.
            max_errors: Stop after this many cumulative errors.

        Returns:
            Summary dict with keys ``total``, ``downloaded``, ``skipped``, ``errors``.
        """
        resolved = self._resolve_cik_list(cik_list, label="company_tickers.json")
        return self._bulk_download(
            resolved,
            dest_dir=self._companyfacts_dir,
            download_fn=self.download_company_facts,
            force=force,
            max_errors=max_errors,
        )

    def load_company_facts(
        self,
        cik: Union[int, str],
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Load company facts JSON for a given CIK from the local cache.

        Downloads the file first if it is not already cached.

        Args:
            cik: CIK number as int or string.
            force_refresh: If True, re-download before loading.

        Returns:
            Parsed JSON dict, or None if the file could not be retrieved.
        """
        path = self.download_company_facts(cik, force=force_refresh)
        if path is None or not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # submissions (8-K tracking)
    # ------------------------------------------------------------------

    def download_submissions(self, cik: Union[int, str], force: bool = False) -> Optional[Path]:
        """
        Download the submissions JSON for a single CIK.

        The submissions endpoint returns lightweight metadata about all filings
        including a ``filings.recent`` object with up to ~1,000 most recent filings
        (form type, filing date, accession number, etc.).  It is suitable for daily
        incremental updates and 8-K event tracking.

        Saved as DATA_CACHE_DIR/edgar/submissions/CIK<CIK10>.json.

        Args:
            cik: CIK number as int or string (leading zeros are accepted).
            force: If True, re-download even when the cached file already exists.

        Returns:
            Path to the cached JSON file, or None if the download failed.
        """
        cik_int = _parse_cik(cik)
        dest = self._submissions_dir / f"CIK{cik_int:010d}.json"

        if dest.exists() and not force:
            _logger.debug("Submissions for CIK %010d already cached at %s", cik_int, dest)
            return dest

        url = _SUBMISSIONS_URL_TEMPLATE.format(cik=cik_int)
        data = self._fetch(url, cik_int, label="submissions")
        if data is None:
            return None

        self._write_json(data, dest)
        _logger.debug("Saved submissions for CIK %010d to %s", cik_int, dest)
        return dest

    def download_all_submissions(
        self,
        cik_list: Optional[List[Union[int, str]]] = None,
        force: bool = False,
        max_errors: int = 50,
    ) -> Dict[str, Any]:
        """
        Bulk-download submissions for a list of CIKs (or all tickers if omitted).

        Submissions files are small and intended for daily refresh, so this method
        is suitable for scheduled overnight runs.

        Args:
            cik_list: Explicit list of CIK numbers to download.
                      If None, downloads submissions for all CIKs in company_tickers.json.
            force: If True, re-download files that already exist in the cache.
            max_errors: Stop after this many cumulative errors.

        Returns:
            Summary dict with keys ``total``, ``downloaded``, ``skipped``, ``errors``.
        """
        resolved = self._resolve_cik_list(cik_list, label="submissions")

        def _download_submissions_fn(cik: Union[int, str], force: bool = False) -> Optional[Path]:
            return self.download_submissions(cik, force=force)

        return self._bulk_download(
            resolved,
            dest_dir=self._submissions_dir,
            download_fn=_download_submissions_fn,
            force=force,
            max_errors=max_errors,
            filename_prefix="CIK",
        )

    def load_submissions(
        self,
        cik: Union[int, str],
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Load submissions JSON for a given CIK from the local cache.

        Downloads the file first if it is not already cached.

        Args:
            cik: CIK number as int or string.
            force_refresh: If True, re-download before loading.

        Returns:
            Parsed submissions dict, or None if the file could not be retrieved.
        """
        path = self.download_submissions(cik, force=force_refresh)
        if path is None or not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def get_recent_filings(
        self,
        cik: Union[int, str],
        form_type: Optional[str] = None,
        since: Optional[datetime] = None,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return recent filings for a CIK, optionally filtered by form type and date.

        Reads from the ``filings.recent`` section of the submissions JSON which
        contains up to ~1,000 most recent filings.

        Args:
            cik: CIK number as int or string.
            form_type: Filter by SEC form type, e.g. ``"8-K"``, ``"10-K"``, ``"10-Q"``.
                       If None, all form types are returned.
            since: Only return filings on or after this date (UTC-aware or naive).
            force_refresh: If True, re-download the submissions file before reading.

        Returns:
            List of filing dicts, each containing at minimum:
            ``form``, ``filingDate``, ``accessionNumber``, ``primaryDocument``.
        """
        data = self.load_submissions(cik, force_refresh=force_refresh)
        if data is None:
            return []

        try:
            recent: Dict[str, List[Any]] = data["filings"]["recent"]
        except (KeyError, TypeError):
            _logger.warning("Unexpected submissions structure for CIK %s", cik)
            return []

        # The recent block is column-oriented: each key maps to a list of equal length
        keys = list(recent.keys())
        n = len(recent.get("form", []))
        if n == 0:
            return []

        filings: List[Dict[str, Any]] = [
            {k: recent[k][i] for k in keys} for i in range(n)
        ]

        if form_type is not None:
            filings = [f for f in filings if f.get("form") == form_type]

        if since is not None:
            since_naive = since.replace(tzinfo=None) if since.tzinfo else since
            filings = [
                f for f in filings
                if _parse_filing_date(f.get("filingDate", "")) >= since_naive
            ]

        return filings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_cik_list(
        self,
        cik_list: Optional[List[Union[int, str]]],
        label: str = "",
    ) -> List[int]:
        """Resolve an optional CIK list to a concrete list of ints."""
        if cik_list is not None:
            return [_parse_cik(c) for c in cik_list]
        tickers_path = self.download_company_tickers(force=False)
        resolved = _extract_ciks(tickers_path)
        _logger.info("Loaded %d CIKs from company_tickers.json for %s", len(resolved), label)
        return resolved

    def _bulk_download(
        self,
        cik_ints: List[int],
        dest_dir: Path,
        download_fn: Any,
        force: bool,
        max_errors: int,
        filename_prefix: str = "",
    ) -> Dict[str, Any]:
        """Generic bulk-download loop shared by facts and submissions."""
        total = len(cik_ints)
        downloaded = skipped = errors = 0

        _logger.info("Starting bulk download for %d CIKs ...", total)

        for i, cik_int in enumerate(cik_ints, start=1):
            dest = dest_dir / f"{filename_prefix}{cik_int:010d}.json"

            if dest.exists() and not force:
                skipped += 1
            else:
                result = download_fn(cik_int, force=force)
                if result is None:
                    errors += 1
                    if errors >= max_errors:
                        _logger.error("Reached max_errors limit (%d). Stopping.", max_errors)
                        break
                else:
                    downloaded += 1

            if i % 100 == 0:
                _logger.info(
                    "Progress: %d/%d — downloaded=%d skipped=%d errors=%d",
                    i, total, downloaded, skipped, errors,
                )

        summary: Dict[str, Any] = {
            "total": total,
            "downloaded": downloaded,
            "skipped": skipped,
            "errors": errors,
        }
        _logger.info(
            "Bulk download complete: downloaded=%d skipped=%d errors=%d / %d total",
            downloaded, skipped, errors, total,
        )
        return summary

    def _fetch(self, url: str, cik_int: int, label: str) -> Optional[Any]:
        """Rate-limited GET with standard error handling. Returns parsed JSON or None."""
        try:
            return self._get(url)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                _logger.debug("No %s for CIK %010d (404)", label, cik_int)
            else:
                _logger.warning("HTTP error downloading %s CIK %010d: %s", label, cik_int, exc)
            return None
        except Exception:
            _logger.exception("Failed to download %s for CIK %010d", label, cik_int)
            return None

    def _get(self, url: str) -> Any:
        """
        Perform a rate-limited GET request and return parsed JSON.

        Enforces the SEC Fair Access Policy (<=10 req/s).

        Args:
            url: URL to fetch.

        Returns:
            Parsed JSON response.

        Raises:
            requests.HTTPError: On non-2xx responses.
        """
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

        _logger.debug("GET %s", url)
        response = self._session.get(url, timeout=30)
        self._last_request_time = time.monotonic()
        response.raise_for_status()
        return response.json()

    def _write_json(self, data: Any, dest: Path) -> None:
        """Write JSON data to dest, creating parent directories as needed."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_cik(cik: Union[int, str]) -> int:
    """Normalise a CIK value (int or zero-padded string) to a plain int."""
    return int(str(cik).lstrip("0") or "0")


def _extract_ciks(tickers_path: Path) -> List[int]:
    """Extract sorted unique CIK ints from a company_tickers.json file."""
    with tickers_path.open("r", encoding="utf-8") as fh:
        data: Dict[str, Dict[str, Any]] = json.load(fh)

    ciks: List[int] = []
    for entry in data.values():
        try:
            ciks.append(_parse_cik(entry["cik_str"]))
        except (KeyError, ValueError) as exc:
            _logger.warning("Could not parse CIK from entry %s: %s", entry, exc)
    return sorted(set(ciks))


def _parse_filing_date(date_str: str) -> datetime:
    """Parse a filing date string (YYYY-MM-DD) to a naive datetime. Returns epoch on failure."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return datetime(1970, 1, 1)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download SEC EDGAR data to local cache.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_tickers = subparsers.add_parser("tickers", help="Download company_tickers.json")
    p_tickers.add_argument("--force", action="store_true", help="Re-download even if already cached")

    p_facts = subparsers.add_parser("facts", help="Download XBRL company facts JSON files")
    p_facts.add_argument("--cik", type=str, nargs="*", help="Specific CIK(s) (default: all)")
    p_facts.add_argument("--force", action="store_true")
    p_facts.add_argument("--max-errors", type=int, default=50)

    p_subs = subparsers.add_parser("submissions", help="Download submissions JSON files (8-K tracking)")
    p_subs.add_argument("--cik", type=str, nargs="*", help="Specific CIK(s) (default: all)")
    p_subs.add_argument("--force", action="store_true")
    p_subs.add_argument("--max-errors", type=int, default=50)

    p_8k = subparsers.add_parser("recent-filings", help="Print recent filings for a CIK")
    p_8k.add_argument("cik", type=str, help="CIK number")
    p_8k.add_argument("--form", type=str, default="8-K", help="Form type filter (default: 8-K)")
    p_8k.add_argument("--since", type=str, default=None, help="ISO date filter, e.g. 2024-01-01")

    parser.add_argument("--cache-dir", type=str, default=None, help=f"Cache root (default: {DATA_CACHE_DIR})")
    parser.add_argument("--user-agent", type=str, default="e-trading-research akossyrev@gmail.com")

    args = parser.parse_args()
    dl = EdgarDownloader(cache_dir=args.cache_dir, user_agent=args.user_agent)

    if args.command == "tickers":
        path = dl.download_company_tickers(force=args.force)
        result = {"success": True, "path": str(path)}
        print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")

    elif args.command == "facts":
        summary = dl.download_all_company_facts(
            cik_list=args.cik or None, force=args.force, max_errors=args.max_errors,
        )
        print(f"__SCHEDULER_RESULT__:{json.dumps({'success': True, **summary})}")

    elif args.command == "submissions":
        summary = dl.download_all_submissions(
            cik_list=args.cik or None, force=args.force, max_errors=args.max_errors,
        )
        print(f"__SCHEDULER_RESULT__:{json.dumps({'success': True, **summary})}")

    elif args.command == "recent-filings":
        since_dt = datetime.strptime(args.since, "%Y-%m-%d") if args.since else None
        filings = dl.get_recent_filings(args.cik, form_type=args.form, since=since_dt)
        result = {"success": True, "count": len(filings), "filings": filings}
        print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
