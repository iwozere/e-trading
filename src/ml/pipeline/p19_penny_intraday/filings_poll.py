"""
P19 intraday EDGAR filings poll (spec v2 §9).

Same-session dilution/catalyst filing detection, scoped to the day's
watchlist CIKs. v2 extends the watched form set beyond the daily 8-K index:
``424B5``, ``S-1``, ``S-3`` (dilution takedowns/authorisations) plus 8-K
items **3.01** (exchange deficiency) and **3.02** (unregistered equity sale)
— the direct trigger for Phase 2's disposition-escalation rule (spec §8.2),
once that engine exists.

**Log-only for now** — there is no Alert Manager yet (Phase 2), so hits are
written to a small dedicated SQLite table for awareness and later
calibration/escalation wiring, not alerts (decision #3: shadow logging before
alerting). Cheap to run every N minutes during market hours: one EFTS query
per watched form type, scoped to the whole watchlist's CIKs at once via
``EdgarDownloader.efts_filings_search`` — not one query per ticker.

Deliberately calls ``efts_filings_search`` directly rather than
``EdgarDownloader.download_8k_filings`` for the 8-K leg: that method writes to
the **shared, universe-wide** daily 8-K index cache
(``edgar/8k/index/{date}.csv.gz``) that P17's CatalystAgent reads once daily
expecting a complete end-of-day snapshot. Calling it intraday with
``force=True`` would overwrite that shared file with a partial same-day
snapshot and (because the file would then already exist) suppress P17's own
next-day refresh — silently starving P17 of any 8-K filed after P19's last
intraday poll. This poll keeps its own separate table instead.
"""

import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p19_penny_intraday.watchlist_builder import load_watchlist
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DEFAULT_OUTPUT_DIR = "results/p19_penny_intraday"
DEFAULT_DB_PATH = "results/p19_penny_intraday/filings_events.sqlite"

# spec §9 / §8.2's watched form set.
_DILUTION_FORMS = ("424B5", "S-1", "S-3")
_WATCHED_8K_ITEMS = frozenset({"3.01", "3.02"})


def _normalize_items(items: Any) -> str:
    """Mirrors edgar_downloader.py's ``_normalize_8k_items`` (list or string,
    both real EFTS shapes seen in production) without reaching into that
    module's private helper."""
    if items is None:
        return ""
    if isinstance(items, (list, tuple)):
        return ",".join(str(i).strip() for i in items if str(i).strip())
    return str(items).strip()


def _hit_cik(src: Dict[str, Any]) -> str:
    ciks = src.get("ciks") or []
    return str(ciks[0]).lstrip("0") if ciks else ""


def _hit_accession(hit: Dict[str, Any]) -> str:
    return str(hit.get("_source", {}).get("adsh", ""))


class FilingsPoll:
    """One market-hours run: EFTS-scan the watchlist's CIKs for same-session
    dilution/catalyst filings and log any new ones."""

    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        target_date: Optional[str] = None,
        edgar: Optional[EdgarDownloader] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self.output_dir = output_dir
        self.target_date = target_date or datetime.now().date().isoformat()
        self._edgar = edgar or EdgarDownloader()
        self.db_path = db_path or DEFAULT_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._ensure_schema()
        self._cik_map: Optional[Dict[str, str]] = None

    def _ensure_schema(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS filings_events (\n"
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "  date TEXT, ts TEXT, ticker TEXT, cik TEXT,\n"
            "  form_type TEXT, item TEXT, accession TEXT, is_dilution INTEGER,\n"
            "  UNIQUE(date, ticker, accession, item)\n"
            ")"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS ix_filings_events_date_ticker ON filings_events(date, ticker)")
        self._conn.commit()

    # ── Public API ────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        entries = load_watchlist(self.output_dir, self.target_date)
        if not entries:
            return {"date": self.target_date, "tickers": 0, "new_hits": 0, "reason": "no watchlist"}

        cik_map = self._build_cik_map([e.ticker for e in entries])
        cik_to_ticker = {cik: ticker for ticker, cik in cik_map.items()}
        ciks = list(cik_map.values())
        if not ciks:
            return {"date": self.target_date, "tickers": len(entries), "new_hits": 0, "reason": "no CIKs resolved"}

        new_hits = 0
        for form in _DILUTION_FORMS:
            hits = self._safe_search(ciks, form)
            new_hits += self._record_dilution_hits(hits, cik_to_ticker, form)

        eightk_hits = self._safe_search(ciks, "8-K")
        new_hits += self._record_8k_hits(eightk_hits, cik_to_ticker)

        _logger.info("Filings poll %s: %d tickers, %d new hits", self.target_date, len(entries), new_hits)
        return {"date": self.target_date, "tickers": len(entries), "new_hits": new_hits}

    def events_for_date(self, target_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """All logged events for a date, most recent first — for shadow_report
        and, eventually, the Phase 2 escalation reader."""
        cur = self._conn.execute(
            "SELECT ts, ticker, form_type, item, is_dilution FROM filings_events WHERE date=? ORDER BY ts DESC",
            (target_date or self.target_date,),
        )
        cols = ["ts", "ticker", "form_type", "item", "is_dilution"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── CIK resolution (same pattern as structural/profiler.py) ─────────────

    def _build_cik_map(self, tickers: List[str]) -> Dict[str, str]:
        if self._cik_map is None:
            try:
                raw = self._edgar.load_company_tickers()
            except Exception:
                _logger.warning("company_tickers.json load failed — filings poll skipped this run")
                self._cik_map = {}
            else:
                self._cik_map = {}
                for entry in raw.values():
                    t = str(entry.get("ticker", "")).upper()
                    cik = entry.get("cik_str")
                    if t and cik:
                        self._cik_map[t] = str(cik)
        wanted = {t.upper() for t in tickers}
        return {t: cik for t, cik in self._cik_map.items() if t in wanted}

    # ── EFTS ──────────────────────────────────────────────────────────────

    def _safe_search(self, ciks: List[str], form: str) -> List[Dict[str, Any]]:
        try:
            return self._edgar.efts_filings_search(
                ciks=ciks, forms=form, start_dt=self.target_date, end_dt=self.target_date
            )
        except Exception:
            _logger.warning("Filings poll: EFTS search failed for form %s", form)
            return []

    # ── Recording ─────────────────────────────────────────────────────────

    def _record_dilution_hits(self, hits: List[Dict[str, Any]], cik_to_ticker: Dict[str, str], form: str) -> int:
        recorded = 0
        for h in hits:
            src = h.get("_source", {})
            cik = _hit_cik(src)
            ticker = cik_to_ticker.get(cik)
            acc = _hit_accession(h)
            if not ticker or not acc:
                continue
            recorded += self._insert(ticker, cik, form, item="", accession=acc, is_dilution=True)
        return recorded

    def _record_8k_hits(self, hits: List[Dict[str, Any]], cik_to_ticker: Dict[str, str]) -> int:
        recorded = 0
        for h in hits:
            src = h.get("_source", {})
            cik = _hit_cik(src)
            ticker = cik_to_ticker.get(cik)
            acc = _hit_accession(h)
            if not ticker or not acc:
                continue
            items = {i.strip() for i in _normalize_items(src.get("items")).split(",") if i.strip()}
            watched = items & _WATCHED_8K_ITEMS
            for item in watched:
                recorded += self._insert(ticker, cik, "8-K", item=item, accession=acc, is_dilution=(item == "3.02"))
        return recorded

    def _insert(self, ticker: str, cik: str, form: str, item: str, accession: str, is_dilution: bool) -> int:
        try:
            self._conn.execute(
                "INSERT INTO filings_events (date, ts, ticker, cik, form_type, item, accession, is_dilution) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.target_date,
                    datetime.now().isoformat(),
                    ticker,
                    cik,
                    form,
                    item,
                    accession,
                    int(is_dilution),
                ),
            )
            self._conn.commit()
            return 1
        except sqlite3.IntegrityError:
            return 0  # already logged this (date, ticker, accession, item) — not new


def run_filings_poll(output_dir: str = DEFAULT_OUTPUT_DIR, target_date: Optional[date] = None) -> Dict[str, Any]:
    """CLI entry point (``run_p19.py filings-poll``)."""
    poll = FilingsPoll(output_dir=output_dir, target_date=(target_date or datetime.now().date()).isoformat())
    return poll.run()
