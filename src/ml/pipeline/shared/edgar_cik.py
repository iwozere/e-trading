"""
Shared EDGAR CIK-resolution helpers.

Used by any pipeline stage that needs to map tickers to SEC CIK numbers before
querying per-company EDGAR endpoints (e.g. P17's DilutionAgent and CatalystAgent).
"""

from datetime import datetime
from typing import Any, Dict, List, Union

from src.data.downloader.edgar_downloader import EdgarDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def build_cik_map(edgar: EdgarDownloader, tickers: List[str]) -> Dict[str, Union[int, str]]:
    """
    Resolve a list of tickers to CIK numbers via EDGAR's company_tickers.json.

    Args:
        edgar: An EdgarDownloader instance (its own cache/session is reused).
        tickers: Tickers to resolve (case-insensitive).

    Returns:
        Dict[ticker (upper) → CIK], containing only tickers that resolved.
    """
    try:
        raw_map: Dict[str, Any] = edgar.load_company_tickers()
        ticker_to_cik: Dict[str, Union[int, str]] = {}
        for entry in raw_map.values():
            t = str(entry.get("ticker", "")).upper()
            c_str = entry.get("cik_str")
            if t and c_str:
                ticker_to_cik[t] = int(c_str) if str(c_str).isdigit() else c_str

        wanted = {t.upper() for t in tickers}
        result = {t: ticker_to_cik[t] for t in wanted if t in ticker_to_cik}
        _logger.info("CIK map: resolved %d/%d tickers", len(result), len(wanted))
        return result
    except Exception:
        _logger.exception("Failed to build CIK map")
        return {}


def parse_filing_date(date_str: str) -> datetime | None:
    """Parse an EDGAR filing date in either ``YYYY-MM-DD`` or ``YYYYMMDD`` form."""
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None
