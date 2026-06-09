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
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
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
_EDGAR_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
_EDGAR_EFTS_SEARCH = "https://efts.sec.gov/LATEST/search-index"

# SEC Fair Access Policy: no more than 10 requests per second
_MIN_REQUEST_INTERVAL = 0.11

# 13F filing window: institutions have up to 45 days after quarter-end to file
_QUARTER_END_MONTH_DAY = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
_13F_FILING_WINDOW_DAYS = 50  # search for filings up to 50 days after quarter-end

# Candidate infotable filenames in 13F-HR filings (tried in order)
_INFOTABLE_FILENAMES = [
    "infotable.xml",
    "InfoTable.xml",
    "form13fInfoTable.xml",
    "xslForm13F_X02.xml",
    "xslForm13F_X01.xml",
]

# EFTS pagination page size (max 100)
_EFTS_PAGE_SIZE = 100


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
        self._13f_dir = self._edgar_dir / "13f"
        self._13f_index_dir = self._13f_dir / "index"
        self._13f_holdings_dir = self._13f_dir / "holdings"
        self._form4_dir = self._13f_dir / "form4"
        self._13dg_dir = self._13f_dir / "13dg"
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
    # 13F-HR institutional holdings
    # ------------------------------------------------------------------

    def download_13f_index(self, year: int, quarter: int, force: bool = False) -> pd.DataFrame:
        """
        Download the index of all 13F-HR filings for a calendar quarter.

        Queries EDGAR EFTS for 13F-HR filings filed within 50 days of the
        quarter-end date and caches the result as
        DATA_CACHE_DIR/edgar/13f/index/{year}_Q{quarter}.csv.gz.

        Args:
            year: Calendar year (e.g., 2024).
            quarter: Quarter number 1–4.
            force: Re-download even if the cache file exists.

        Returns:
            DataFrame with columns: cik, institution_name, accession_number, filed_date.
            Empty DataFrame on failure.
        """
        dest = self._13f_index_dir / f"{year}_Q{quarter}.csv.gz"
        if dest.exists() and not force:
            _logger.info("13F index for %d Q%d already cached at %s", year, quarter, dest)
            return pd.read_csv(dest, compression="gzip", dtype=str)

        start_dt, end_dt = _13f_filing_window(year, quarter)
        _logger.info("Downloading 13F index for %d Q%d (filing window %s → %s)", year, quarter, start_dt, end_dt)

        hits = self._efts_search(forms="13F-HR", start_dt=str(start_dt), end_dt=str(end_dt))
        if not hits:
            _logger.warning("No 13F-HR filings found for %d Q%d", year, quarter)
            return pd.DataFrame()

        records = []
        for hit in hits:
            src = hit.get("_source", {})
            records.append({
                "cik": str(src.get("entity_id", "")).lstrip("0") or None,
                "institution_name": src.get("entity_name", ""),
                "accession_number": src.get("accession_no", ""),
                "filed_date": src.get("file_date", ""),
                "period_of_report": src.get("period_of_report", ""),
            })

        df = pd.DataFrame(records).dropna(subset=["cik"])
        df["cik"] = df["cik"].astype(str)

        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False, compression="gzip")
        _logger.info("Cached 13F index: %d filers for %d Q%d → %s", len(df), year, quarter, dest)
        return df

    def download_13f_infotable(
        self,
        cik: Union[int, str],
        accession_number: str,
        year: int,
        quarter: int,
        institution_name: str = "",
        force: bool = False,
    ) -> Optional[Path]:
        """
        Download, parse, and cache the holdings infotable for one 13F-HR filing.

        Tries candidate infotable filenames in order, falls back to parsing the
        EDGAR filing index HTML if all candidates return 404.
        Result is saved as DATA_CACHE_DIR/edgar/13f/holdings/{year}_Q{quarter}/{cik:010d}.csv.gz.

        Args:
            cik: Institution CIK.
            accession_number: Accession number, e.g. ``"0001234567-24-000123"``.
            year: Calendar year of the reporting quarter.
            quarter: Quarter number 1–4.
            institution_name: Human-readable institution name (stored in output).
            force: Re-download even if cached.

        Returns:
            Path to the saved CSV.gz, or None on failure.
        """
        cik_int = _parse_cik(cik)
        quarter_dir = self._13f_holdings_dir / f"{year}_Q{quarter}"
        dest = quarter_dir / f"{cik_int:010d}.csv.gz"

        if dest.exists() and not force:
            _logger.debug("13F holdings for CIK %010d Q%d/%d already cached", cik_int, year, quarter)
            return dest

        acc_norm = accession_number.replace("-", "")
        xml_content = self._fetch_filing_xml(cik_int, acc_norm)
        if xml_content is None:
            _logger.warning("Could not fetch infotable XML for CIK %010d acc %s", cik_int, accession_number)
            return None

        quarter_str = f"{year}Q{quarter}"
        df = self.parse_13f_infotable(xml_content, cik_int, institution_name, quarter_str)
        if df.empty:
            _logger.warning("Empty infotable for CIK %010d acc %s", cik_int, accession_number)
            return None

        # Compute portfolio percentage weights
        total_value = df["value_usd"].sum()
        df["pct_of_portfolio"] = df["value_usd"] / total_value if total_value > 0 else 0.0

        quarter_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False, compression="gzip")
        _logger.debug(
            "Cached 13F holdings for CIK %010d: %d positions, $%.0fM total → %s",
            cik_int, len(df), total_value / 1_000_000, dest,
        )
        return dest

    def parse_13f_infotable(
        self,
        xml_content: str,
        cik: int,
        institution_name: str,
        quarter: str,
    ) -> pd.DataFrame:
        """
        Parse a 13F infotable XML string into a holdings DataFrame.

        Handles both namespaced and bare XML variants used across different filers.

        Args:
            xml_content: Raw XML text of the infotable document.
            cik: Institution CIK (added to every row).
            institution_name: Institution name (added to every row).
            quarter: Quarter string, e.g. ``"2024Q1"`` (added to every row).

        Returns:
            DataFrame with columns: cik, institution_name, quarter, name_of_issuer,
            cusip, value_usd, shares, investment_discretion, put_call.
            Empty DataFrame on parse failure or no positions found.
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            _logger.warning("XML parse error for CIK %d, quarter %s", cik, quarter)
            return pd.DataFrame()

        # Strip namespaces so tag matching works regardless of ns variant
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}")[-1]

        records = []
        for info in root.findall(".//infoTable"):
            records.append({
                "cik": cik,
                "institution_name": institution_name,
                "quarter": quarter,
                "name_of_issuer": _xml_text(info, "nameOfIssuer"),
                "cusip": _xml_text(info, "cusip"),
                "value_usd": _safe_int(_xml_text(info, "value")) * 1000,
                "shares": _safe_int(_xml_text(info, ".//sshPrnamt")),
                "investment_discretion": _xml_text(info, "investmentDiscretion"),
                "put_call": _xml_text(info, "putCall"),
            })

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)

    def load_13f_holdings(
        self,
        cik: Union[int, str],
        year: int,
        quarter: int,
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Load cached 13F holdings for a CIK and quarter from CSV.gz.

        Args:
            cik: Institution CIK.
            year: Calendar year.
            quarter: Quarter number 1–4.
            force_refresh: If True, re-download before loading.

        Returns:
            DataFrame of holdings, or None if the file is absent and cannot be fetched.
        """
        cik_int = _parse_cik(cik)
        dest = self._13f_holdings_dir / f"{year}_Q{quarter}" / f"{cik_int:010d}.csv.gz"
        if dest.exists() and not force_refresh:
            return pd.read_csv(dest, compression="gzip")

        # Try to obtain from the index
        index_df = self.download_13f_index(year, quarter, force=False)
        row = index_df[index_df["cik"] == str(cik_int)]
        if row.empty:
            _logger.warning("CIK %d not found in 13F index for %d Q%d", cik_int, year, quarter)
            return None

        acc = row.iloc[0]["accession_number"]
        name = row.iloc[0].get("institution_name", "")
        path = self.download_13f_infotable(cik_int, acc, year, quarter, name, force=force_refresh)
        if path is None or not path.exists():
            return None

        return pd.read_csv(path, compression="gzip")

    def get_new_13f_filings_today(self, as_of_date: Optional[date] = None) -> pd.DataFrame:
        """
        Return 13F-HR filings submitted on a given date (default: today).

        Used by the daily scheduler job to detect new filings incrementally.
        Does NOT cache — always queries EDGAR live.

        Args:
            as_of_date: Date to check. Defaults to today (UTC).

        Returns:
            DataFrame with columns: cik, institution_name, accession_number, filed_date.
        """
        check_date = as_of_date or datetime.now().date()
        date_str = str(check_date)
        _logger.info("Checking EDGAR for new 13F-HR filings on %s", date_str)

        hits = self._efts_search(forms="13F-HR", start_dt=date_str, end_dt=date_str)
        if not hits:
            return pd.DataFrame(columns=["cik", "institution_name", "accession_number", "filed_date"])

        records = [
            {
                "cik": str(h.get("_source", {}).get("entity_id", "")).lstrip("0"),
                "institution_name": h.get("_source", {}).get("entity_name", ""),
                "accession_number": h.get("_source", {}).get("accession_no", ""),
                "filed_date": h.get("_source", {}).get("file_date", ""),
                "period_of_report": h.get("_source", {}).get("period_of_report", ""),
            }
            for h in hits
        ]
        _logger.info("Found %d new 13F-HR filings on %s", len(records), date_str)
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Form 4 and Schedule 13D/G daily monitoring
    # ------------------------------------------------------------------

    def download_form4_filings(
        self,
        as_of_date: Optional[date] = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Download and parse Form 4 insider transaction filings for a given date.

        Only sale transactions (codes S, S-) are retained. Results are cached as
        DATA_CACHE_DIR/edgar/13f/form4/{date}.csv.gz.

        Args:
            as_of_date: Filing date to fetch. Defaults to yesterday (markets are
                        closed when the pipeline runs at 07:00 UTC).
            force: Re-download even if cached.

        Returns:
            DataFrame with columns: ticker, issuer_cik, insider_name, transaction_code,
            shares, price_per_share, total_value_usd, filed_date.
        """
        target_date = as_of_date or (datetime.now().date() - timedelta(days=1))
        date_str = str(target_date)
        dest = self._form4_dir / f"{date_str}.csv.gz"

        if dest.exists() and not force:
            _logger.info("Form 4 filings for %s already cached at %s", date_str, dest)
            return pd.read_csv(dest, compression="gzip")

        _logger.info("Downloading Form 4 filings for %s ...", date_str)
        hits = self._efts_search(forms="4", start_dt=date_str, end_dt=date_str)

        records = []
        for hit in hits:
            src = hit.get("_source", {})
            acc = src.get("accession_no", "").replace("-", "")
            cik_str = str(src.get("entity_id", "")).lstrip("0")
            if not acc or not cik_str:
                continue

            xml_content = self._fetch_filing_xml(
                int(cik_str) if cik_str else 0,
                acc,
                candidate_names=["primary-doc.xml", "form4.xml", "doc4.xml"],
            )
            if xml_content is None:
                continue

            for row in _parse_form4_xml(xml_content, filed_date=date_str):
                records.append(row)

        df = pd.DataFrame(records) if records else pd.DataFrame()

        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False, compression="gzip")
        _logger.info("Cached %d Form 4 sale transactions for %s → %s", len(df), date_str, dest)
        return df

    def download_13dg_filings(
        self,
        as_of_date: Optional[date] = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Download Schedule 13D and 13G amendments filed on a given date.

        Results are cached as DATA_CACHE_DIR/edgar/13f/13dg/{date}.csv.gz.

        Args:
            as_of_date: Filing date to fetch. Defaults to yesterday.
            force: Re-download even if cached.

        Returns:
            DataFrame with columns: cik, entity_name, accession_number, filed_date,
            form_type (13D or 13G).
        """
        target_date = as_of_date or (datetime.now().date() - timedelta(days=1))
        date_str = str(target_date)
        dest = self._13dg_dir / f"{date_str}.csv.gz"

        if dest.exists() and not force:
            _logger.info("13D/G filings for %s already cached at %s", date_str, dest)
            return pd.read_csv(dest, compression="gzip")

        _logger.info("Downloading 13D/G filings for %s ...", date_str)
        hits = self._efts_search(forms="SC 13D,SC 13G,SC 13D/A,SC 13G/A", start_dt=date_str, end_dt=date_str)

        records = [
            {
                "cik": str(h.get("_source", {}).get("entity_id", "")).lstrip("0"),
                "entity_name": h.get("_source", {}).get("entity_name", ""),
                "accession_number": h.get("_source", {}).get("accession_no", ""),
                "filed_date": date_str,
                "form_type": h.get("_source", {}).get("form_type", ""),
            }
            for h in hits
        ]

        df = pd.DataFrame(records) if records else pd.DataFrame()
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False, compression="gzip")
        _logger.info("Cached %d 13D/G filings for %s → %s", len(df), date_str, dest)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def resolve_tickers_to_ciks(self, tickers: List[str]) -> List[Union[int, str]]:
        """
        Resolve a list of ticker symbols to CIK integers using company_tickers.json.

        Tickers not found in the mapping are skipped with a warning.

        Args:
            tickers: Ticker symbols (e.g. ['JPM', 'AAPL', 'BRK-B']).

        Returns:
            Sorted list of unique CIK integers for the matched tickers.
        """
        tickers_path = self.download_company_tickers(force=False)
        with tickers_path.open("r", encoding="utf-8") as fh:
            data: Dict[str, Dict[str, Any]] = json.load(fh)

        ticker_to_cik: Dict[str, int] = {}
        for entry in data.values():
            try:
                ticker_to_cik[str(entry["ticker"]).upper()] = _parse_cik(entry["cik_str"])
            except (KeyError, ValueError):
                pass

        ciks: List[Union[int, str]] = []
        for ticker in tickers:
            cik = ticker_to_cik.get(ticker.upper())
            if cik is not None:
                ciks.append(cik)
            else:
                _logger.warning("Ticker %s not found in company_tickers.json — skipped", ticker)

        return sorted(set(ciks))

    def _efts_search(self, forms: str, start_dt: str, end_dt: str) -> List[Dict]:
        """
        Paginate through EDGAR EFTS full-text search results for given form types.

        Args:
            forms: Comma-separated form type filter, e.g. ``"13F-HR"`` or ``"4"``.
            start_dt: Start date string ``"YYYY-MM-DD"``.
            end_dt: End date string ``"YYYY-MM-DD"``.

        Returns:
            List of ``hits`` dicts from the EFTS response.
        """
        all_hits: List[Dict] = []
        offset = 0

        while True:
            params = {
                "forms": forms,
                "dateRange": "custom",
                "startdt": start_dt,
                "enddt": end_dt,
                "from": offset,
            }
            try:
                elapsed = time.monotonic() - self._last_request_time
                if elapsed < _MIN_REQUEST_INTERVAL:
                    time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

                resp = self._session.get(_EDGAR_EFTS_SEARCH, params=params, timeout=30)
                self._last_request_time = time.monotonic()
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                _logger.exception("EFTS search failed (forms=%s start=%s end=%s offset=%d)", forms, start_dt, end_dt, offset)
                break

            hits = data.get("hits", {}).get("hits", [])
            all_hits.extend(hits)

            total = data.get("hits", {}).get("total", {}).get("value", 0)
            offset += len(hits)
            if not hits or offset >= total:
                break

        return all_hits

    def _fetch_filing_xml(
        self,
        cik_int: int,
        acc_norm: str,
        candidate_names: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Try candidate filenames inside an EDGAR filing folder and return XML text.

        Args:
            cik_int: CIK as integer.
            acc_norm: Accession number with dashes removed, e.g. ``"000123456724001234"``.
            candidate_names: Ordered list of filenames to try. Defaults to
                             ``_INFOTABLE_FILENAMES``.

        Returns:
            Raw XML text, or None if no candidate succeeds.
        """
        names = candidate_names or _INFOTABLE_FILENAMES
        base = f"{_EDGAR_ARCHIVES_BASE}/{cik_int}/{acc_norm}"

        for filename in names:
            url = f"{base}/{filename}"
            try:
                elapsed = time.monotonic() - self._last_request_time
                if elapsed < _MIN_REQUEST_INTERVAL:
                    time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

                resp = self._session.get(url, timeout=30)
                self._last_request_time = time.monotonic()

                if resp.status_code == 200:
                    _logger.debug("Found filing document at %s", url)
                    return resp.text
                if resp.status_code != 404:
                    _logger.warning("Unexpected status %d for %s", resp.status_code, url)
            except Exception:
                _logger.exception("Error fetching %s", url)

        # Fallback: fetch the filing index HTML and look for any .xml link
        index_url = f"{_EDGAR_ARCHIVES_BASE}/{cik_int}/{acc_norm}/{acc_norm}-index.htm"
        try:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < _MIN_REQUEST_INTERVAL:
                time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
            resp = self._session.get(index_url, timeout=30)
            self._last_request_time = time.monotonic()

            if resp.status_code == 200:
                import re as _re
                xml_files = re.findall(r'href="([^"]+\.xml)"', resp.text, _re.IGNORECASE)
                infotable_candidates = [f for f in xml_files if any(kw in f.lower() for kw in ("form", "info", "table"))]
                for xml_file in (infotable_candidates or xml_files):
                    xml_url = f"{_EDGAR_ARCHIVES_BASE}/{cik_int}/{acc_norm}/{xml_file}"
                    r2 = self._session.get(xml_url, timeout=30)
                    self._last_request_time = time.monotonic()
                    if r2.status_code == 200:
                        return r2.text
        except Exception:
            _logger.exception("Fallback index fetch failed for CIK %d acc %s", cik_int, acc_norm)

        return None

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

def _13f_filing_window(year: int, quarter: int) -> tuple:
    """Return (start_date, end_date) of the 13F filing window for a quarter."""
    qe_month, qe_day = _QUARTER_END_MONTH_DAY[quarter]
    quarter_end = date(year, qe_month, qe_day)
    start = quarter_end + timedelta(days=1)
    end = quarter_end + timedelta(days=_13F_FILING_WINDOW_DAYS)
    return start, end


def _xml_text(element: ET.Element, path: str) -> str:
    """Return stripped text of the first matching sub-element, or empty string."""
    found = element.find(path)
    if found is not None and found.text:
        return found.text.strip()
    return ""


def _safe_int(value: str) -> int:
    """Convert a string to int, returning 0 on failure."""
    try:
        return int(value.replace(",", "").strip()) if value else 0
    except (ValueError, AttributeError):
        return 0


def _parse_form4_xml(xml_content: str, filed_date: str) -> List[Dict[str, Any]]:
    """
    Parse a Form 4 XML document and return sale transaction rows.

    Only returns rows for open-market sale codes: S (sale) and S- (sale short).

    Args:
        xml_content: Raw XML text of the Form 4 filing.
        filed_date: Date string ``"YYYY-MM-DD"`` added to every row.

    Returns:
        List of row dicts (may be empty if no qualifying transactions found).
    """
    _SALE_CODES = {"S", "S-"}
    rows: List[Dict[str, Any]] = []

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return rows

    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}")[-1]

    ticker = _xml_text(root, ".//issuerTradingSymbol")
    issuer_cik = _xml_text(root, ".//issuerCik")
    insider_name = _xml_text(root, ".//rptOwnerName")

    for txn in root.findall(".//nonDerivativeTransaction"):
        code = _xml_text(txn, ".//transactionCode")
        if code not in _SALE_CODES:
            continue

        shares_str = _xml_text(txn, ".//transactionShares/value")
        price_str = _xml_text(txn, ".//transactionPricePerShare/value")
        shares = _safe_int(shares_str)
        try:
            price = float(price_str) if price_str else 0.0
        except ValueError:
            price = 0.0

        rows.append({
            "ticker": ticker,
            "issuer_cik": issuer_cik,
            "insider_name": insider_name,
            "transaction_code": code,
            "shares": shares,
            "price_per_share": price,
            "total_value_usd": shares * price,
            "filed_date": filed_date,
        })

    return rows


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
