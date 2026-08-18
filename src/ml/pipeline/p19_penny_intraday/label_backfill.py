"""
P19 label backfill — T+10 cron (spec v2 §12.2, §16 item 4).

Fills the forward-looking outcome labels ``eod_backfill`` cannot know same-day:
forward close-to-close returns (``ret_t1/t3/t5/t10``) and the two structural-
decay labels (``dilution_event_within_5d``, ``reverse_split_within_180d``).
Runs once a shadow-store date is old enough that T+10 session data exists;
re-running on a date still too young is a safe no-op (nothing gets written).
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, cast

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p19_penny_intraday.shadow_store import DEFAULT_DB_PATH, ShadowStore
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Same offering-form set as structural/grading.py's N8 (design-v2.md keeps this
# consistent across Layer 0 and label backfill).
_OFFERING_FORMS = frozenset({"424B5", "424B4", "424B3", "424B2", "424B1", "S-1", "S-1/A"})
_UNREGISTERED_SALE_ITEM = "3.02"
# Calendar-day proxy for "T+10 trading sessions have definitely passed" --
# generous on purpose (weekends + a holiday or two); a date that's actually
# still short on session data just yields fewer non-None ret_t* labels, not a
# wrong one, since _forward_closes only returns offsets it actually has data for.
_MIN_AGE_CALENDAR_DAYS = 16
# Window fetched per ticker to cover 10 trading sessions with slack for holidays.
_FORWARD_FETCH_CALENDAR_DAYS = 20


class LabelBackfill:
    """Fills forward-return + structural-decay labels for old-enough shadow dates."""

    def __init__(
        self,
        store: Optional[ShadowStore] = None,
        edgar: Optional[EdgarDownloader] = None,
        data_manager: Optional[Any] = None,
    ) -> None:
        self._store = store or ShadowStore(DEFAULT_DB_PATH)
        self._edgar = edgar or EdgarDownloader()
        self._data_manager = data_manager  # lazily constructed if not injected

    def run(self, min_age_days: int = _MIN_AGE_CALENDAR_DAYS) -> Dict[str, Any]:
        """Backfill every shadow date old enough to plausibly have T+10 data."""
        today = datetime.now().date()
        dates_filled = 0
        tickers_filled = 0
        for date_str in self._store.dates_needing_label_backfill():
            as_of = _parse_date(date_str)
            if as_of is None or (today - as_of).days < min_age_days:
                continue
            n = self._backfill_date(date_str, as_of)
            if n:
                dates_filled += 1
                tickers_filled += n
        _logger.info("Label backfill: %d dates, %d tickers labelled", dates_filled, tickers_filled)
        return {"dates": dates_filled, "tickers": tickers_filled}

    def _backfill_date(self, date_str: str, as_of: date) -> int:
        filled = 0
        for ticker in self._store.tickers_needing_label_backfill(date_str):
            labels = self._compute_labels(ticker, as_of)
            if labels is None:
                continue
            self._store.update_forward_labels(date_str, ticker, labels)
            filled += 1
        return filled

    # ── Per-ticker label computation ────────────────────────────────────────

    def _compute_labels(self, ticker: str, as_of: date) -> Optional[Dict[str, Any]]:
        closes = self._forward_closes(ticker, as_of)
        base = closes.get(0)
        if base is None or base <= 0:
            return None
        return {
            "ret_t1": _ret(base, closes.get(1)),
            "ret_t3": _ret(base, closes.get(3)),
            "ret_t5": _ret(base, closes.get(5)),
            "ret_t10": _ret(base, closes.get(10)),
            "dilution_event_within_5d": self._dilution_event_within(ticker, as_of, 5),
            "reverse_split_within_180d": self._reverse_split_within(ticker, as_of, 180),
        }

    def _forward_closes(self, ticker: str, as_of: date) -> Dict[int, float]:
        """{trading-session offset from as_of: close}. Offset 0 is as_of's own
        session (the "base" ret_t* are measured from), assuming DataManager's
        daily bars start there — true for any actual trading day."""
        dm = self._get_data_manager()
        start = datetime.combine(as_of, datetime.min.time())
        end = start + timedelta(days=_FORWARD_FETCH_CALENDAR_DAYS)
        try:
            df = dm.get_ohlcv(ticker, "1d", start, end)
        except Exception:
            _logger.warning("Forward-closes fetch failed for %s as_of=%s", ticker, as_of)
            return {}
        if df is None or df.empty:
            return {}
        return {i: float(row["close"]) for i, (_idx, row) in enumerate(df.iterrows())}

    def _dilution_event_within(self, ticker: str, as_of: date, sessions: int) -> Optional[bool]:
        cik = self._resolve_cik(ticker)
        if cik is None:
            return None
        window_end = as_of + timedelta(days=sessions * 2)  # generous calendar-day proxy for `sessions` trading days
        try:
            filings = self._edgar.get_recent_filings(cik, since=datetime.combine(as_of, datetime.min.time()))
        except Exception:
            _logger.warning("Filings fetch failed for %s (CIK %s)", ticker, cik)
            return None
        for f in filings:
            form = str(f.get("form") or "").upper().strip()
            filed = _parse_date(f.get("filingDate", ""))
            if filed is None or filed > window_end:
                continue
            if form in _OFFERING_FORMS:
                return True
            if form == "8-K" and _UNREGISTERED_SALE_ITEM in [i.strip() for i in str(f.get("items") or "").split(",")]:
                return True
        return False

    def _reverse_split_within(self, ticker: str, as_of: date, days: int) -> Optional[bool]:
        try:
            import yfinance as yf

            splits = yf.Ticker(ticker).splits
        except Exception:
            _logger.warning("yfinance splits fetch failed for %s", ticker)
            return None
        if splits is None:
            return False
        window_end = as_of + timedelta(days=days)
        for ts, ratio in splits.items():
            ts_any: Any = ts
            d = cast(date, ts_any.date() if hasattr(ts_any, "date") else ts_any)
            if as_of < d <= window_end and float(cast(Any, ratio)) < 1.0:
                return True
        return False

    def _resolve_cik(self, ticker: str) -> Optional[str]:
        try:
            ciks = self._edgar.resolve_tickers_to_ciks([ticker])
        except Exception:
            return None
        return str(ciks[0]) if ciks else None

    def _get_data_manager(self) -> Any:
        if self._data_manager is None:
            from src.data.data_manager import DataManager

            self._data_manager = DataManager()
        return self._data_manager


def _ret(base: Optional[float], val: Optional[float]) -> Optional[float]:
    if base is None or val is None or base <= 0:
        return None
    return val / base - 1.0


def _parse_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None
