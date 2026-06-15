"""Earnings calendar — fetches upcoming earnings dates from FMP and caches monthly."""

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests

from src.ml.pipeline.p05_ai_selector.config import EARNINGS_CACHE_DIR
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_FMP_STABLE_URL = "https://financialmodelingprep.com/stable/earning-calendar"
_CACHE_TTL_HOURS = 24

try:
    from config.donotshare.donotshare import FMP_API_KEY
except ImportError:
    FMP_API_KEY = None


class EarningsCalendar:
    """Fetches and caches the FMP earnings calendar."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        self._api_key = api_key or FMP_API_KEY
        self._cache_dir = cache_dir or EARNINGS_CACHE_DIR

    def get_earnings_within_days(
        self,
        tickers: List[str],
        as_of_date: date,
        window_days: int = 7,
    ) -> Dict[str, date]:
        """
        Return {ticker: earnings_date} for tickers with earnings within window_days.

        Args:
            tickers: List of ticker symbols to check.
            as_of_date: Reference date.
            window_days: Number of calendar days ahead to scan.

        Returns:
            Dict mapping ticker → nearest earnings date (may be empty on failure).
        """
        from datetime import timedelta

        end_date = as_of_date + timedelta(days=window_days)
        calendar_df = self._get_calendar(as_of_date, end_date)
        if calendar_df.empty:
            return {}

        ticker_set = set(t.upper() for t in tickers)
        result: Dict[str, date] = {}
        for _, row in calendar_df.iterrows():
            symbol = str(row.get("symbol", "")).upper()
            if symbol not in ticker_set:
                continue
            try:
                earnings_date = date.fromisoformat(str(row["date"])[:10])
                if as_of_date <= earnings_date <= end_date:
                    if symbol not in result or earnings_date < result[symbol]:
                        result[symbol] = earnings_date
            except (ValueError, TypeError):
                pass

        _logger.info(
            "EarningsCalendar: %d tickers have earnings within %d days of %s",
            len(result),
            window_days,
            as_of_date,
        )
        return result

    def _get_calendar(self, start: date, end: date) -> pd.DataFrame:
        """Return cached or freshly fetched calendar for the given date range."""
        cache_key = f"{start.strftime('%Y-%m')}.csv.gz"
        cache_file = self._cache_dir / cache_key

        if self._is_cache_fresh(cache_file):
            _logger.debug("EarningsCalendar: cache hit for %s", cache_key)
            try:
                return pd.read_csv(cache_file, compression="gzip")
            except Exception:
                _logger.exception("EarningsCalendar: error reading cache %s", cache_file)

        df = self._fetch_from_fmp(start, end)
        if not df.empty:
            self._write_cache(cache_file, df)
        return df

    def _fetch_from_fmp(self, start: date, end: date) -> pd.DataFrame:
        """Fetch earnings calendar from FMP stable API."""
        if not self._api_key:
            _logger.warning("EarningsCalendar: no FMP API key — returning empty")
            return pd.DataFrame()
        try:
            resp = requests.get(
                _FMP_STABLE_URL,
                params={
                    "apikey": self._api_key,
                    "from": start.isoformat(),
                    "to": end.isoformat(),
                },
                timeout=30,
            )
            if resp.status_code in (402, 404):
                _logger.warning("EarningsCalendar: FMP returned %d — returning empty", resp.status_code)
                return pd.DataFrame()
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(data)
        except requests.exceptions.RequestException:
            _logger.exception("EarningsCalendar: HTTP error fetching calendar")
            return pd.DataFrame()
        except Exception:
            _logger.exception("EarningsCalendar: unexpected error")
            return pd.DataFrame()

    def _is_cache_fresh(self, cache_file: Path) -> bool:
        """Return True if the cache file exists and was written within the TTL."""
        if not cache_file.exists():
            return False
        import time
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        return age_hours < _CACHE_TTL_HOURS

    def _write_cache(self, cache_file: Path, df: pd.DataFrame) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False, compression="gzip")
        _logger.debug("EarningsCalendar: cached %d rows to %s", len(df), cache_file)
