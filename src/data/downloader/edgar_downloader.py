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

import gzip
import html
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
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

# SEC EDGAR API endpoints
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANY_FACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_EDGAR_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
_EDGAR_EFTS_SEARCH = "https://efts.sec.gov/LATEST/search-index"


class EftsUnavailableError(Exception):
    """Raised when the EDGAR EFTS search endpoint fails before returning any results."""


# SEC Fair Access Policy: no more than 10 requests per second
_MIN_REQUEST_INTERVAL = 0.5

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

# XBRL fact candidates for shares outstanding (preference order)
_SHARES_FACT_CANDIDATES = [
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstandingPeriod"),
    ("us-gaap", "WeightedAverageNumberOfShareOutstandingBasic"),
    ("us-gaap", "SharesOutstanding"),
    ("dei", "EntityCommonStockSharesOutstanding"),
]

# EDGAR quarterly full-index (covers all form types including SC 13D/G which EFTS does not index)
_EDGAR_FULL_INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/form.gz"
_13DG_FORM_TYPES = frozenset({"SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A"})
# Form 10 / 10-12B / 10-12G registration statements — the filing a company makes
# to register the stock being distributed in a spin-off (P20 Kestrel Sleeve B2,
# spec gap 10.2). Same EDGAR quarterly-index approach as 13D/G: verified against
# a real quarterly form.idx (2015 QTR4) that this form-type string is exactly
# how EDGAR labels it there (e.g. Fortive Corp, Ingevity Corp spin-offs).
_FORM10_FORM_TYPES = frozenset({"10-12B", "10-12B/A", "10-12G", "10-12G/A"})
# Number of header lines to skip in the quarterly form.idx file
_FORM_IDX_HEADER_LINES = 9

# Form 4 non-derivative transaction columns (all codes, not just sales — see
# download_form4_filings docstring).
_FORM4_COLS = [
    "ticker",
    "issuer_cik",
    "insider_name",
    "transaction_code",
    "acquired_disposed_code",
    "shares",
    "price_per_share",
    "total_value_usd",
    "filed_date",
    "transaction_date",
    "is_director",
    "is_officer",
    "is_ten_percent_owner",
    "officer_title",
    "is_10b5_1_plan",
    "is_derivative",
]


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
        cache_dir: Union[str, Path] | None = None,
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
        self._form10_dir = self._13f_dir / "form10"
        self._8k_dir = self._edgar_dir / "8k"
        self._8k_index_dir = self._8k_dir / "index"
        self._full_index_dir = self._edgar_dir / "full-index"
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

    def download_company_facts(self, cik: Union[int, str], force: bool = False) -> Path | None:
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
        cik_list: List[Union[int, str]] | None = None,
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
    ) -> Dict[str, Any] | None:
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

    def download_submissions(self, cik: Union[int, str], force: bool = False) -> Path | None:
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
        cik_list: List[Union[int, str]] | None = None,
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

        def _download_submissions_fn(cik: Union[int, str], force: bool = False) -> Path | None:
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
    ) -> Dict[str, Any] | None:
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
        form_type: str | None = None,
        since: datetime | None = None,
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

        filings: List[Dict[str, Any]] = [{k: recent[k][i] for k in keys} for i in range(n)]

        if form_type is not None:
            filings = [f for f in filings if f.get("form") == form_type]

        if since is not None:
            since_naive = since.replace(tzinfo=None) if since.tzinfo else since
            filings = [f for f in filings if _parse_filing_date(f.get("filingDate", "")) >= since_naive]

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
            _logger.debug("13F index for %d Q%d already cached at %s", year, quarter, dest)
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
            records.append(
                {
                    "cik": _efts_first_cik(src) or None,
                    "institution_name": _efts_company(src),
                    "accession_number": str(src.get("adsh", "")),
                    "filed_date": src.get("file_date", ""),
                    "period_of_report": src.get("period_ending", ""),
                }
            )

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
    ) -> Path | None:
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
            cik_int,
            len(df),
            total_value / 1_000_000,
            dest,
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
            records.append(
                {
                    "cik": cik,
                    "institution_name": institution_name,
                    "quarter": quarter,
                    "name_of_issuer": _xml_text(info, "nameOfIssuer"),
                    "cusip": _xml_text(info, "cusip"),
                    "value_usd": _safe_int(_xml_text(info, "value")) * 1000,
                    "shares": _safe_int(_xml_text(info, ".//sshPrnamt")),
                    "investment_discretion": _xml_text(info, "investmentDiscretion"),
                    "put_call": _xml_text(info, "putCall"),
                }
            )

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)

    def load_13f_holdings(
        self,
        cik: Union[int, str],
        year: int,
        quarter: int,
        force_refresh: bool = False,
    ) -> pd.DataFrame | None:
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

    def get_new_13f_filings_today(self, as_of_date: date | None = None) -> pd.DataFrame:
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
            return pd.DataFrame(columns=["cik", "institution_name", "accession_number", "filed_date"])  # type: ignore[arg-type]

        records = [
            {
                "cik": _efts_first_cik(h.get("_source", {})),
                "institution_name": _efts_company(h.get("_source", {})),
                "accession_number": str(h.get("_source", {}).get("adsh", "")),
                "filed_date": h.get("_source", {}).get("file_date", ""),
                "period_of_report": h.get("_source", {}).get("period_ending", ""),
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
        as_of_date: date | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Download and parse Form 4 insider transaction filings for a given date.

        All transaction codes are retained (not just sales) — callers filter to
        what they need. This used to drop everything but sale codes {"S", "S-"}
        before caching; P18's ``Form4Monitor.get_significant_sells`` already
        re-filters to sale codes itself so it is unaffected by the wider set, and
        P20 Kestrel's ``filings_ingest.py`` reads this same cache file directly
        expecting buy codes {"P", "A"} that could never have appeared under the
        old sale-only filter — this also fixes that. Results are cached as
        DATA_CACHE_DIR/edgar/13f/form4/{date}.csv.gz.

        Args:
            as_of_date: Filing date to fetch. Defaults to yesterday (markets are
                        closed when the pipeline runs at 07:00 UTC).
            force: Re-download even if cached.

        Returns:
            DataFrame with columns: ticker, issuer_cik, insider_name, transaction_code,
            acquired_disposed_code, shares, price_per_share, total_value_usd, filed_date,
            transaction_date, is_director, is_officer, is_ten_percent_owner,
            officer_title, is_10b5_1_plan, is_derivative.
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
            acc = str(src.get("adsh", ""))
            if not acc:
                continue
            acc_norm = acc.replace("-", "")
            # A Form 4 EFTS hit lists BOTH the reporting owner and the issuer in
            # `ciks`; the filing lives under either entity's CIK directory, and the
            # issuer ticker is read from the parsed XML. The primary-document name
            # varies per filing (e.g. edgardoc.xml, primary_doc.xml, wf-form4_*.xml),
            # so use the exact filename the EFTS `_id` carries as the first candidate.
            cik_str = _efts_first_cik(src)
            cik_int = int(cik_str) if cik_str else int(acc_norm[:10])
            primary_doc = _primary_doc_from_efts_id(str(hit.get("_id", "")))
            candidate_names = [primary_doc] if primary_doc else []
            candidate_names += ["primary-doc.xml", "primary_doc.xml", "form4.xml", "doc4.xml"]

            xml_content = self._fetch_filing_xml(cik_int, acc_norm, candidate_names=candidate_names)
            if xml_content is None:
                continue

            for row in _parse_form4_xml(xml_content, filed_date=date_str):
                records.append(row)

        df = pd.DataFrame(records, columns=_FORM4_COLS) if records else pd.DataFrame(columns=_FORM4_COLS)  # type: ignore[arg-type]

        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False, compression="gzip")
        _logger.info("Cached %d Form 4 transactions for %s → %s", len(df), date_str, dest)
        return df

    def download_13dg_filings(
        self,
        as_of_date: date | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Download Schedule 13D and 13G filings submitted on a given date.

        Results are cached as DATA_CACHE_DIR/edgar/13f/13dg/{date}.csv.gz.

        EDGAR's full-text search (EFTS) does not index SC 13D/G filings, so this
        method uses the official quarterly form.idx file cached under
        DATA_CACHE_DIR/edgar/full-index/. The quarterly index is refreshed daily
        for the current quarter and cached permanently for past quarters.

        Args:
            as_of_date: Filing date to fetch. Defaults to yesterday.
            force: Re-download even if cached (also forces quarterly index refresh).

        Returns:
            DataFrame with columns: cik, entity_name, accession_number, filed_date,
            form_type (SC 13D, SC 13G, SC 13D/A, or SC 13G/A).
        """
        target_date = as_of_date or (datetime.now().date() - timedelta(days=1))
        date_str = str(target_date)
        dest = self._13dg_dir / f"{date_str}.csv.gz"

        if dest.exists() and not force:
            _logger.info("13D/G filings for %s already cached at %s", date_str, dest)
            return pd.read_csv(dest, compression="gzip")

        # EDGAR EFTS does not index SC 13D/G filings — use the quarterly form.idx instead.
        _logger.info("Fetching 13D/G filings for %s from EDGAR quarterly form index ...", date_str)
        quarter = (target_date.month - 1) // 3 + 1
        idx_lines = self._fetch_quarterly_form_idx(target_date.year, quarter, force=force)

        records = []
        for line in idx_lines:
            if not (line.startswith("SC 13D") or line.startswith("SC 13G")):
                continue
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) < 5:
                continue
            form_type, entity_name, cik_str, filed_date, filename = parts[:5]
            if form_type not in _13DG_FORM_TYPES or filed_date != date_str:
                continue
            # Accession number lives in the filename stem: edgar/data/{cik}/XXXXXXXXXX-YY-NNNNNN.txt
            acc_no = Path(filename).stem
            records.append(
                {
                    "cik": cik_str.strip(),
                    "entity_name": entity_name.strip(),
                    "accession_number": acc_no,
                    "filed_date": filed_date,
                    "form_type": form_type,
                }
            )

        _13DG_COLS = ["cik", "entity_name", "accession_number", "filed_date", "form_type"]
        df = pd.DataFrame(records, columns=_13DG_COLS) if records else pd.DataFrame(columns=_13DG_COLS)  # type: ignore[arg-type]
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False, compression="gzip")
        _logger.info("Cached %d 13D/G filings for %s → %s", len(df), date_str, dest)
        return df

    def download_form10_filings(
        self,
        as_of_date: date | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Download Form 10 / 10-12B / 10-12G registration statements filed on a given date.

        This is the filing a company makes to register the stock being distributed
        in a spin-off (P20 Kestrel Sleeve B2, spec gap 10.2). Results are cached as
        DATA_CACHE_DIR/edgar/13f/form10/{date}.csv.gz.

        Like SC 13D/G, EDGAR's full-text search (EFTS) does not reliably index
        every Form 10 variant, so this uses the same official quarterly form.idx
        file as ``download_13dg_filings``.

        Note: a spin-off's ticker frequently does not exist yet in
        company_tickers.json at Form 10 filing time (the entity is still
        pre-listing) — resolving CIK to ticker is the caller's responsibility and
        may need to be retried on a later run once the ticker is assigned.

        Args:
            as_of_date: Filing date to fetch. Defaults to yesterday.
            force: Re-download even if cached (also forces quarterly index refresh).

        Returns:
            DataFrame with columns: cik, entity_name, accession_number, filed_date,
            form_type (10-12B, 10-12G, or their /A amendments).
        """
        target_date = as_of_date or (datetime.now().date() - timedelta(days=1))
        date_str = str(target_date)
        dest = self._form10_dir / f"{date_str}.csv.gz"

        if dest.exists() and not force:
            _logger.info("Form 10 filings for %s already cached at %s", date_str, dest)
            return pd.read_csv(dest, compression="gzip")

        _logger.info("Fetching Form 10 filings for %s from EDGAR quarterly form index ...", date_str)
        quarter = (target_date.month - 1) // 3 + 1
        idx_lines = self._fetch_quarterly_form_idx(target_date.year, quarter, force=force)

        records = []
        for line in idx_lines:
            if not line.startswith("10-12"):
                continue
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) < 5:
                continue
            form_type, entity_name, cik_str, filed_date, filename = parts[:5]
            if form_type not in _FORM10_FORM_TYPES or filed_date != date_str:
                continue
            acc_no = Path(filename).stem
            records.append(
                {
                    "cik": cik_str.strip(),
                    "entity_name": entity_name.strip(),
                    "accession_number": acc_no,
                    "filed_date": filed_date,
                    "form_type": form_type,
                }
            )

        _FORM10_COLS = ["cik", "entity_name", "accession_number", "filed_date", "form_type"]
        df = pd.DataFrame(records, columns=_FORM10_COLS) if records else pd.DataFrame(columns=_FORM10_COLS)  # type: ignore[arg-type]
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False, compression="gzip")
        _logger.info("Cached %d Form 10 filings for %s → %s", len(df), date_str, dest)
        return df

    def download_8k_filings(
        self,
        as_of_date: date | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Download the index of all 8-K filings submitted on a given date.

        Queries EDGAR EFTS for form 8-K on the date and caches a lightweight
        per-day index — one row per filing — as
        ``DATA_CACHE_DIR/edgar/8k/index/{date}.csv.gz``. This is the universe-wide
        daily catalyst feed the P17 CatalystAgent reads from (8-K item codes per
        filing) and the seed for later 8-K body fetching.

        Args:
            as_of_date: Filing date to fetch. Defaults to yesterday (markets/EDGAR
                are quiet when the daily bundle runs).
            force: Re-download even if cached.

        Returns:
            DataFrame with columns: cik, company, accession_number, items,
            description, filed_date, primary_document. Empty if none were filed.
        """
        target_date = as_of_date or (datetime.now().date() - timedelta(days=1))
        date_str = str(target_date)
        dest = self._8k_index_dir / f"{date_str}.csv.gz"

        if dest.exists() and not force:
            _logger.info("8-K index for %s already cached at %s", date_str, dest)
            return pd.read_csv(dest, compression="gzip", dtype=str)

        # NB: EFTS treats the `forms` param as an exact match; passing "8-K,8-K/A"
        # paradoxically returns only the /A amendments, so query the plain form
        # (which is the bulk of catalyst-bearing filings). Amendments can be added
        # later via a second query if needed.
        _logger.info("Downloading 8-K index for %s ...", date_str)
        hits = self._efts_search(forms="8-K", start_dt=date_str, end_dt=date_str)

        records = []
        for hit in hits:
            src = hit.get("_source", {})
            # EFTS _source is list-oriented: ciks / display_names are arrays, the
            # accession number is `adsh`. (Field names verified against the live
            # endpoint — NOT entity_id/accession_no, which do not exist here.)
            cik_str = _efts_first_cik(src)
            acc = str(src.get("adsh", ""))
            if not cik_str or not acc:
                continue
            records.append(
                {
                    "cik": cik_str,
                    "company": _efts_company(src),
                    "accession_number": acc,
                    "items": _normalize_8k_items(src.get("items")),
                    "description": str(src.get("file_description", "") or src.get("file_type", "")),
                    "filed_date": str(src.get("file_date", "") or date_str),
                    "primary_document": _primary_doc_from_efts_id(str(hit.get("_id", ""))),
                }
            )

        _8K_COLS = ["cik", "company", "accession_number", "items", "description", "filed_date", "primary_document"]
        df = pd.DataFrame(records, columns=_8K_COLS) if records else pd.DataFrame(columns=_8K_COLS)  # type: ignore[arg-type]
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False, compression="gzip")
        _logger.info("Cached %d 8-K filings for %s → %s", len(df), date_str, dest)
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

    def get_shares_outstanding_at_date(
        self,
        cik: Union[int, str],
        settlement_date: date,
    ) -> Dict[str, Any] | None:
        """
        Return the best shares-outstanding fact for a CIK on or before a given date.

        Loads company facts from cache (fetching from SEC EDGAR if absent) and
        searches XBRL facts in preference order for the most recent filing whose
        end date is <= settlement_date.

        Args:
            cik: CIK number as int or string.
            settlement_date: Upper bound for the effective date (e.g. a settlement date).

        Returns:
            Dict with keys value (int), effective_date (date), fact (str), unit (str);
            or None if no matching fact is found.
        """
        cf_json = self.load_company_facts(cik)
        if cf_json is None:
            _logger.warning("No company facts available for CIK %s", cik)
            return None
        return _pick_best_shares_fact(cf_json, settlement_date)

    def _fetch_quarterly_form_idx(self, year: int, quarter: int, force: bool = False) -> List[str]:
        """
        Download and cache the EDGAR quarterly form.idx for a given year/quarter.

        For the current quarter the cache is refreshed if older than 24 hours because
        EDGAR appends new filings daily. Past quarters are cached permanently.

        Args:
            year: Calendar year.
            quarter: Quarter number 1–4.
            force: Re-download even if a valid cache exists.

        Returns:
            Data lines from form.idx (9-line header stripped).
        """
        dest = self._full_index_dir / f"{year}_Q{quarter}_form.idx.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)

        today = datetime.now().date()
        current_quarter = (today.month - 1) // 3 + 1
        is_current = year == today.year and quarter == current_quarter
        cache_stale = is_current and dest.exists() and (time.time() - dest.stat().st_mtime) > 86400

        if not dest.exists() or force or cache_stale:
            url = _EDGAR_FULL_INDEX_URL.format(year=year, quarter=quarter)
            _logger.info("Downloading EDGAR form index for %dQ%d ...", year, quarter)
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < _MIN_REQUEST_INTERVAL:
                time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
            resp = self._session.get(url, timeout=120)
            self._last_request_time = time.monotonic()
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            _logger.info("Cached EDGAR form index → %s (%d bytes)", dest, len(resp.content))

        with gzip.open(dest, "rt", encoding="latin-1") as fh:
            lines = fh.readlines()

        return [ln.rstrip("\n") for ln in lines[_FORM_IDX_HEADER_LINES:]]

    def _efts_search(
        self,
        forms: str,
        start_dt: str,
        end_dt: str,
        q: str | None = None,
        ciks: str | None = None,
    ) -> List[Dict]:
        """
        Paginate through EDGAR EFTS full-text search results for given form types.

        Args:
            forms: Comma-separated form type filter, e.g. ``"13F-HR"`` or ``"4"``.
            start_dt: Start date string ``"YYYY-MM-DD"``.
            end_dt: End date string ``"YYYY-MM-DD"``.
            q: Optional full-text search phrase (quote it yourself for an exact
               phrase match, e.g. ``'"going concern"'``). Used by P19's Layer 0
               text-scoped signals (N5/N6, design-v2.md §3.2) — omitted by every
               other existing caller, which searches by form+date only.
            ciks: Optional comma-separated 10-digit CIKs to scope the search to
                  (EFTS param name is ``ciks``, not ``cik``). Used by the P19
                  intraday filings poll to scope a single query to the whole
                  watchlist instead of one query per ticker.

        Returns:
            List of ``hits`` dicts from the EFTS response.
        """
        all_hits: List[Dict] = []
        offset = 0

        while True:
            params: Dict[str, Any] = {
                "forms": forms,
                "dateRange": "custom",
                "startdt": start_dt,
                "enddt": end_dt,
                "from": offset,
            }
            if q:
                params["q"] = q
            if ciks:
                params["ciks"] = ciks
            last_exc: Exception | None = None
            data: Dict[str, Any] = {}
            for attempt in range(3):
                try:
                    elapsed = time.monotonic() - self._last_request_time
                    if elapsed < _MIN_REQUEST_INTERVAL:
                        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

                    resp = self._session.get(_EDGAR_EFTS_SEARCH, params=params, timeout=30)
                    self._last_request_time = time.monotonic()
                    resp.raise_for_status()
                    data = resp.json()
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status is not None and status < 500:
                        break  # 4xx — retrying won't help
                    backoff = 2**attempt
                    _logger.warning(
                        "EFTS attempt %d/3 failed (forms=%s start=%s status=%s) — retrying in %ds",
                        attempt + 1,
                        forms,
                        start_dt,
                        status,
                        backoff,
                    )
                    time.sleep(backoff)

            if last_exc is not None:
                _logger.exception(
                    "EFTS search failed after 3 attempts (forms=%s start=%s end=%s offset=%d)",
                    forms,
                    start_dt,
                    end_dt,
                    offset,
                    exc_info=last_exc,
                )
                if not all_hits:
                    raise EftsUnavailableError(f"EFTS unavailable for forms={forms!r} on {start_dt}") from last_exc
                break  # partial results already collected — stop gracefully

            hits = data.get("hits", {}).get("hits", [])
            all_hits.extend(hits)

            total = data.get("hits", {}).get("total", {}).get("value", 0)
            offset += len(hits)
            if not hits or offset >= total:
                break

        return all_hits

    def efts_filings_search(
        self,
        ciks: List[str],
        forms: str,
        start_dt: str,
        end_dt: str,
    ) -> List[Dict]:
        """
        Multi-CIK EFTS filing lookup (no text phrase) — "did any of these
        issuers file this form type in this date range". Used by P19's
        intraday filings poll (spec §9) to scope one query per form type to
        the whole watchlist at once, rather than one query per ticker.

        Verified live against the EFTS endpoint (2026-08-18): ``ciks`` accepts
        a comma-list and ORs it (an Elasticsearch ``terms`` filter, unlike
        ``forms`` which is an exact match — see ``efts_text_search``'s
        docstring for that quirk). Chunked at 100 CIKs per request — the
        matching P19 watchlist cap (spec §4.1), so realistically always one
        request per form type in practice.

        Args:
            ciks: Issuer CIKs (any digit-string form; zero-padded internally).
            forms: A single form type, e.g. ``"424B5"``.
            start_dt: Start date ``"YYYY-MM-DD"``.
            end_dt: End date ``"YYYY-MM-DD"``.

        Returns:
            De-duplicated list of EFTS hit dicts across all CIK chunks.
        """
        padded = [f"{int(c):010d}" for c in ciks if str(c).strip()]
        if not padded:
            return []
        seen_ids: set = set()
        out: List[Dict] = []
        chunk_size = 100
        for i in range(0, len(padded), chunk_size):
            chunk = padded[i : i + chunk_size]
            try:
                hits = self._efts_search(forms=forms, start_dt=start_dt, end_dt=end_dt, ciks=",".join(chunk))
            except EftsUnavailableError:
                _logger.warning("efts_filings_search: EFTS unavailable for forms=%s chunk %d", forms, i // chunk_size)
                continue
            for h in hits:
                hid = h.get("_id")
                if hid and hid not in seen_ids:
                    seen_ids.add(hid)
                    out.append(h)
        return out

    def efts_text_search(
        self,
        cik: str,
        phrases: List[str],
        forms: str,
        start_dt: str,
        end_dt: str,
    ) -> List[Dict]:
        """
        CIK-scoped EFTS full-text phrase search (design-v2.md §3.2), for P19
        Layer 0's text-parse-required signals: N5 (floating-rate/toxic
        convertible), N6 (going-concern qualification), N16 (auditor quality,
        via the EX-23.1 consent exhibit — see ``_fetch_filing_document``).

        Verified live against the EFTS endpoint (2026-08-18): ``q`` does an
        Elasticsearch ``match_phrase`` (quote the phrase yourself is
        unnecessary — this method quotes it), ``ciks`` filters by CIK, and the
        response contains no highlighted snippet — only match metadata
        (``_source``: ciks/display_names/adsh/file_date/root_forms/file_type/
        file_description), same shape as every other ``_efts_search`` caller
        in this file.

        Args:
            cik: Issuer CIK (any digit-string form; zero-padded internally).
            phrases: Exact phrases to search for (OR'd — hits are unioned and
                     de-duplicated by ``_id``). Each phrase is sent as its own
                     query since EFTS ``q`` is a single phrase match, not a
                     boolean OR of several.
            forms: A single form type, e.g. ``"10-K"``. **Do not pass a
                   comma-list** — EFTS treats ``forms`` as an exact match, not
                   an OR, so ``"10-K,10-K/A"`` paradoxically returns only the
                   amendments (verified live, 2026-08-18; same quirk already
                   documented on the 8-K catalyst path in this file). Callers
                   needing several form types must issue one call per form and
                   union the results — see ``get_auditor_name`` for the pattern.
            start_dt: Start date ``"YYYY-MM-DD"`` — callers scope this
                      themselves (StructuralSignals.md's N5 recency-scoping
                      requirement: latest annual + latest interim only, not an
                      unscoped all-history search).
            end_dt: End date ``"YYYY-MM-DD"``.

        Returns:
            De-duplicated list of EFTS hit dicts across all phrases.
        """
        try:
            cik_padded = f"{int(cik):010d}"
        except (TypeError, ValueError):
            return []
        seen_ids: set = set()
        out: List[Dict] = []
        for phrase in phrases:
            try:
                hits = self._efts_search(forms=forms, start_dt=start_dt, end_dt=end_dt, q=f'"{phrase}"', ciks=cik_padded)
            except EftsUnavailableError:
                _logger.warning("efts_text_search: EFTS unavailable for CIK %s phrase %r — skipping", cik, phrase)
                continue
            for h in hits:
                hid = h.get("_id")
                if hid and hid not in seen_ids:
                    seen_ids.add(hid)
                    out.append(h)
        return out

    def _fetch_filing_document(self, cik_int: int, acc_norm: str, filename: str) -> str | None:
        """
        Fetch one specific document from a filing folder by its exact filename
        (as opposed to ``_fetch_filing_xml``'s candidate-name guessing, which
        is for filings whose XML document name isn't already known). Used to
        pull the primary document an ``efts_text_search`` hit's ``_id``
        already names.

        Returns:
            Raw document text (HTML or XML), or None if the fetch failed.
        """
        url = f"{_EDGAR_ARCHIVES_BASE}/{cik_int}/{acc_norm}/{filename}"
        for attempt in range(3):
            try:
                elapsed = time.monotonic() - self._last_request_time
                if elapsed < _MIN_REQUEST_INTERVAL:
                    time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
                resp = self._session.get(url, timeout=30)
                self._last_request_time = time.monotonic()
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code == 404:
                    return None
                if resp.status_code == 503:
                    sleep_sec = 30 * (attempt + 1)
                    _logger.warning("EDGAR rate-limited (503) for %s, sleeping %ds (attempt %d/3)", url, sleep_sec, attempt + 1)
                    time.sleep(sleep_sec)
                    continue
                _logger.warning("Unexpected status %d for %s", resp.status_code, url)
                return None
            except Exception as e:
                sleep_sec = 15 * (attempt + 1)
                _logger.warning("Network error fetching %s (attempt %d/3), sleeping %ds: %s", url, attempt + 1, sleep_sec, e)
                time.sleep(sleep_sec)
        return None

    def get_auditor_name(
        self,
        cik: str,
        start_dt: str,
        end_dt: str,
        forms: str = "10-K,10-K/A,20-F,20-F/A",
    ) -> str | None:
        """
        Best-effort auditor firm name for a CIK (StructuralSignals.md N16), via
        the EX-23.1 "Consent of Independent Registered Public Accounting Firm"
        exhibit filed alongside the latest annual report in the window.

        This is a text-extraction heuristic (``_extract_auditor_name``), not a
        structured field — no XBRL tag exists for auditor identity. Callers
        (P19's ``structural/profiler.py``) scope ``start_dt``/``end_dt`` to
        the issuer's latest annual filing window, matching the recency-scoping
        convention ``efts_text_search`` callers already follow.

        Returns:
            The extracted firm name, or None if no consent exhibit was found
            or extraction failed — never guesses.
        """
        # EFTS treats `forms` as an exact match, not an OR — a comma-list like
        # "10-K,10-K/A" paradoxically returns only amendments (verified live,
        # 2026-08-18; same quirk already documented for the 8-K catalyst path
        # above). Query each form type separately and union the hits.
        hits: List[Dict] = []
        seen_ids: set = set()
        for form in forms.split(","):
            for h in self.efts_text_search(
                cik=cik,
                phrases=["CONSENT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM"],
                forms=form.strip(),
                start_dt=start_dt,
                end_dt=end_dt,
            ):
                hid = h.get("_id")
                if hid and hid not in seen_ids:
                    seen_ids.add(hid)
                    hits.append(h)
        if not hits:
            return None
        # Most recent filing first (file_date descending) — an issuer can
        # change auditors, and only the current one is relevant to N16.
        hits.sort(key=lambda h: str(h.get("_source", {}).get("file_date", "")), reverse=True)
        hit = hits[0]
        acc_norm = str(hit.get("_source", {}).get("adsh", "")).replace("-", "")
        filename = _primary_doc_from_efts_id(str(hit.get("_id", "")))
        cik_str = _efts_first_cik(hit.get("_source", {}))
        if not acc_norm or not filename or not cik_str:
            return None
        doc_text = self._fetch_filing_document(int(cik_str), acc_norm, filename)
        if doc_text is None:
            return None
        return _extract_auditor_name(doc_text)

    def _fetch_filing_xml(
        self,
        cik_int: int,
        acc_norm: str,
        candidate_names: List[str] | None = None,
    ) -> str | None:
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
            for attempt in range(3):
                try:
                    elapsed = time.monotonic() - self._last_request_time
                    if elapsed < _MIN_REQUEST_INTERVAL:
                        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

                    resp = self._session.get(url, timeout=30)
                    self._last_request_time = time.monotonic()

                    if resp.status_code == 200:
                        _logger.debug("Found filing document at %s", url)
                        return resp.text
                    if resp.status_code == 404:
                        break  # file does not exist; try next candidate name
                    if resp.status_code == 503:
                        sleep_sec = 30 * (attempt + 1)
                        _logger.warning(
                            "EDGAR rate-limited (503) for %s, sleeping %ds (attempt %d/3)", url, sleep_sec, attempt + 1
                        )
                        time.sleep(sleep_sec)
                        continue  # retry same URL
                    _logger.warning("Unexpected status %d for %s", resp.status_code, url)
                    break
                except Exception as e:
                    sleep_sec = 15 * (attempt + 1)
                    _logger.warning(
                        "Network error fetching %s (attempt %d/3), sleeping %ds: %s", url, attempt + 1, sleep_sec, e
                    )
                    time.sleep(sleep_sec)

        # Fallback: fetch the filing index HTML and look for any .xml link.
        # Try both the institution CIK path and the filing-agent CIK path.
        # EDGAR stores filings under the submitter's CIK (often a filing agent),
        # which is the first 10 digits of the accession number.
        acc_filer_cik = int(acc_norm[:10])
        cik_candidates = [cik_int]
        if acc_filer_cik != cik_int:
            cik_candidates.append(acc_filer_cik)

        for path_cik in cik_candidates:
            # Try candidate filenames at the alternate CIK path first
            if path_cik != cik_int:
                alt_base = f"{_EDGAR_ARCHIVES_BASE}/{path_cik}/{acc_norm}"
                for filename in names:
                    url = f"{alt_base}/{filename}"
                    for attempt in range(3):
                        try:
                            elapsed = time.monotonic() - self._last_request_time
                            if elapsed < _MIN_REQUEST_INTERVAL:
                                time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
                            resp = self._session.get(url, timeout=30)
                            self._last_request_time = time.monotonic()
                            if resp.status_code == 200:
                                _logger.debug("Found filing document at %s (agent CIK path)", url)
                                return resp.text
                            if resp.status_code == 404:
                                break
                            if resp.status_code == 503:
                                sleep_sec = 30 * (attempt + 1)
                                _logger.warning(
                                    "EDGAR rate-limited (503) for %s, sleeping %ds (attempt %d/3)",
                                    url,
                                    sleep_sec,
                                    attempt + 1,
                                )
                                time.sleep(sleep_sec)
                                continue
                            break
                        except Exception as e:
                            sleep_sec = 15 * (attempt + 1)
                            _logger.warning(
                                "Network error fetching %s (attempt %d/3), sleeping %ds: %s",
                                url,
                                attempt + 1,
                                sleep_sec,
                                e,
                            )
                            time.sleep(sleep_sec)

            index_url = f"{_EDGAR_ARCHIVES_BASE}/{path_cik}/{acc_norm}/{acc_norm}-index.htm"
            try:
                elapsed = time.monotonic() - self._last_request_time
                if elapsed < _MIN_REQUEST_INTERVAL:
                    time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
                resp = self._session.get(index_url, timeout=30)
                self._last_request_time = time.monotonic()

                if resp.status_code == 200:
                    import re as _re

                    xml_files = re.findall(r'href="([^"]+\.xml)"', resp.text, _re.IGNORECASE)
                    infotable_candidates = [
                        f for f in xml_files if any(kw in f.lower() for kw in ("form", "info", "table"))
                    ]
                    for xml_file in infotable_candidates or xml_files:
                        xml_url = f"{_EDGAR_ARCHIVES_BASE}/{path_cik}/{acc_norm}/{xml_file}"
                        r2 = self._session.get(xml_url, timeout=30)
                        self._last_request_time = time.monotonic()
                        if r2.status_code == 200:
                            return r2.text
            except Exception as e:
                _logger.warning("Fallback index fetch failed for CIK %d acc %s: %s", path_cik, acc_norm, e)

        return None

    def _resolve_cik_list(
        self,
        cik_list: List[Union[int, str]] | None,
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
                    i,
                    total,
                    downloaded,
                    skipped,
                    errors,
                )

        summary: Dict[str, Any] = {
            "total": total,
            "downloaded": downloaded,
            "skipped": skipped,
            "errors": errors,
        }
        _logger.info(
            "Bulk download complete: downloaded=%d skipped=%d errors=%d / %d total",
            downloaded,
            skipped,
            errors,
            total,
        )
        return summary

    def _fetch(self, url: str, cik_int: int, label: str) -> Any | None:
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


def _pick_best_shares_fact(
    cf_json: Dict[str, Any],
    settlement_date: date,
) -> Dict[str, Any] | None:
    """
    Select the most recent shares-outstanding fact from EDGAR company facts JSON
    whose end date is on or before settlement_date.

    Iterates _SHARES_FACT_CANDIDATES in preference order.

    Args:
        cf_json: Parsed company facts JSON from SEC EDGAR.
        settlement_date: Upper bound for the effective date of the fact.

    Returns:
        Dict with keys value (int), effective_date (date), fact (str), unit (str);
        or None if no match found.
    """
    facts = cf_json.get("facts", {})
    best: tuple | None = None  # (end_date, value, fact_name, unit_name)
    for space, fname in _SHARES_FACT_CANDIDATES:
        fact_obj = facts.get(space, {}).get(fname)
        if not fact_obj:
            continue
        for unit_name, entries in fact_obj.get("units", {}).items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                end = entry.get("end") or entry.get("instant")
                if not end:
                    continue
                try:
                    end_date = datetime.strptime(end[:10], "%Y-%m-%d").date()
                except ValueError:
                    continue
                if end_date > settlement_date:
                    continue
                val = entry.get("val")
                if val is None:
                    continue
                try:
                    val_int = int(float(val))
                except (ValueError, TypeError):
                    continue
                if best is None or end_date > best[0]:
                    best = (end_date, val_int, f"{space}:{fname}", unit_name)
    if best is None:
        return None
    return {"value": best[1], "effective_date": best[0], "fact": best[2], "unit": best[3]}


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


def _footnotes_mentioning_10b5_1(root: ET.Element) -> set:
    """
    Return the footnote ids whose text references a Rule 10b5-1 trading plan.

    SEC's 2022 Rule 10b5-1 amendments added a structured plan-adoption indicator
    to the Form 4 XML schema, but its exact element name has not been verified
    against a live filing here (design-v2.md §3.1) — this falls back to the text
    convention filers used before, and continue to use alongside, the structured
    field: a footnote whose text mentions "10b5-1". Revisit once verified.
    """
    ids = set()
    for fn in root.findall(".//footnote"):
        if "10b5-1" in (fn.text or "").lower():
            fid = fn.get("id")
            if fid:
                ids.add(fid)
    return ids


def _safe_bool(value: str) -> bool:
    """Parse a Form 4 XML boolean flag (``"1"``/``"0"``, occasionally ``"true"``/``"false"``)."""
    return value.strip().lower() in {"1", "true"}


def _parse_form4_xml(xml_content: str, filed_date: str) -> List[Dict[str, Any]]:
    """
    Parse a Form 4 XML document and return all non-derivative transaction rows.

    Every transaction code is returned (not just sales) — callers filter to what
    they need; see ``download_form4_filings``'s docstring for why the filtering
    moved to the caller. Derivative transactions (options, warrants) are not
    parsed — no current consumer needs them, and non-derivative open-market buys
    (code P) / sales (code S) are what P17/P18/P19/P20 all actually use.

    Args:
        xml_content: Raw XML text of the Form 4 filing.
        filed_date: Date string ``"YYYY-MM-DD"`` added to every row (when the
            filing was received by SEC — up to 2 business days after the trade).

    Returns:
        List of row dicts (may be empty if no transactions found).
    """
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
    plan_footnote_ids = _footnotes_mentioning_10b5_1(root)

    relationship = root.find(".//reportingOwnerRelationship")
    is_director = _safe_bool(_xml_text(relationship, ".//isDirector")) if relationship is not None else False
    is_officer = _safe_bool(_xml_text(relationship, ".//isOfficer")) if relationship is not None else False
    is_ten_pct_owner = (
        _safe_bool(_xml_text(relationship, ".//isTenPercentOwner")) if relationship is not None else False
    )
    officer_title = _xml_text(relationship, ".//officerTitle") if relationship is not None else ""

    for txn in root.findall(".//nonDerivativeTransaction"):
        code = _xml_text(txn, ".//transactionCode")
        if not code:
            continue

        shares_str = _xml_text(txn, ".//transactionShares/value")
        price_str = _xml_text(txn, ".//transactionPricePerShare/value")
        acquired_disposed = _xml_text(txn, ".//transactionAcquiredDisposedCode/value")
        shares = _safe_int(shares_str)
        try:
            price = float(price_str) if price_str else 0.0
        except ValueError:
            price = 0.0

        # Per-transaction trade date, distinct from `filed_date` (the SEC receipt
        # date). Falls back to filed_date on a malformed/missing element rather
        # than leaving it blank — the two are never more than a couple of
        # business days apart by law.
        transaction_date = _xml_text(txn, ".//transactionDate/value") or filed_date

        txn_footnote_ids = {fn.get("id") for fn in txn.findall(".//footnoteId") if fn.get("id")}

        rows.append(
            {
                "ticker": ticker,
                "issuer_cik": issuer_cik,
                "insider_name": insider_name,
                "transaction_code": code,
                "acquired_disposed_code": acquired_disposed,
                "shares": shares,
                "price_per_share": price,
                "total_value_usd": shares * price,
                "filed_date": filed_date,
                "transaction_date": transaction_date,
                "is_director": is_director,
                "is_officer": is_officer,
                "is_ten_percent_owner": is_ten_pct_owner,
                "officer_title": officer_title,
                "is_10b5_1_plan": bool(txn_footnote_ids & plan_footnote_ids),
                "is_derivative": False,
            }
        )

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


def _normalize_8k_items(items: Any) -> str:
    """Normalise an EFTS 8-K ``items`` field (list or string) to a comma-joined string."""
    if items is None:
        return ""
    if isinstance(items, (list, tuple)):
        return ",".join(str(i).strip() for i in items if str(i).strip())
    return str(items).strip()


def _primary_doc_from_efts_id(efts_id: str) -> str:
    """Extract the primary document filename from an EFTS hit ``_id`` (``"{accession}:{doc}"``)."""
    return efts_id.split(":", 1)[1] if ":" in efts_id else ""


def _company_from_display_name(display_name: str) -> str:
    """Strip the ``" (CIK 0001234567)"`` suffix from an EFTS display_names entry."""
    return display_name.split("(CIK", 1)[0].strip()


def _efts_first_cik(src: Dict[str, Any]) -> str:
    """
    Return the first CIK (leading zeros stripped) from an EFTS ``_source``, or "".

    EFTS ``_source`` is list-oriented: the CIK(s) live in ``ciks`` (an array), NOT
    in ``entity_id`` — which does not exist on this endpoint.
    """
    ciks = src.get("ciks") or []
    return str(ciks[0]).lstrip("0") if ciks else ""


def _efts_company(src: Dict[str, Any]) -> str:
    """Return the display name (CIK suffix stripped) from an EFTS ``_source``, or ""."""
    names = src.get("display_names") or []
    return _company_from_display_name(str(names[0])) if names else ""


# Firm-name designation suffixes/markers that identify which *line* of an
# EX-23.1 "Consent of Independent Registered Public Accounting Firm" exhibit
# is the firm-name line, once split on block boundaries. Matched against the
# whole line (not captured piecemeal) so odd separator characters real filers
# use (``+``, ``&``) don't break extraction the way a bounded capture group
# would — verified against a live filing (2026-08-18, DeltaSoft Corp CIK
# 0002020919, accession 0001683168-26-005450): "/S/ Boladale lawal" is
# followed by its own line, "BOLADALE LAWAL & CO".
_AUDITOR_SUFFIX_MARKER = re.compile(
    r"LLP|LLC|P\.?C\.?\b|CPAs?\b|&|\+|AND COMPANY|CHARTERED ACCOUNTANTS?",
    re.IGNORECASE,
)


def _extract_auditor_name(doc_text: str) -> str | None:
    """
    Best-effort auditor firm-name extraction from an EX-23.1 consent exhibit's
    HTML text (StructuralSignals.md N16). Returns None rather than guessing
    when no confident match is found — an unresolved N16 must depress
    ``coverage``, never be silently read as "no auditor flag" (N17 rule).

    Text-parsing heuristic, not a structured field (no XBRL tag exists for
    this) — precision is necessarily short of what a maintained PCAOB Form AP
    integration would give (design-v2.md §Roadmap's documented simplification).
    """
    # Keep block-level tags as line breaks before stripping the rest, so the
    # firm-name line isn't glued onto the signature line above it.
    text = re.sub(r"<(p|div|br|tr)\b[^>]*>", "\n", doc_text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = [re.sub(r"\s+", " ", ln).strip(" ,.") for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]

    sig_idx = next((i for i, ln in enumerate(lines) if ln.lower().startswith("/s/")), None)
    if sig_idx is None:
        # No signature line at all -- searching the whole document risks
        # matching an unrelated "&"/"+" elsewhere in the boilerplate (e.g. the
        # addressee line, "Shareholders & Board of Directors"). A genuine
        # EX-23.1 consent always has a signature; if this doesn't, it's
        # unresolved, not a document worth guessing from.
        return None

    for ln in lines[sig_idx + 1 : sig_idx + 8]:
        if _AUDITOR_SUFFIX_MARKER.search(ln):
            return ln
    return None


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
            cik_list=args.cik or None,
            force=args.force,
            max_errors=args.max_errors,
        )
        print(f"__SCHEDULER_RESULT__:{json.dumps({'success': True, **summary})}")

    elif args.command == "submissions":
        summary = dl.download_all_submissions(
            cik_list=args.cik or None,
            force=args.force,
            max_errors=args.max_errors,
        )
        print(f"__SCHEDULER_RESULT__:{json.dumps({'success': True, **summary})}")

    elif args.command == "recent-filings":
        since_dt = datetime.strptime(args.since, "%Y-%m-%d") if args.since else None
        filings = dl.get_recent_filings(args.cik, form_type=args.form, since=since_dt)
        result = {"success": True, "count": len(filings), "filings": filings}
        print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
