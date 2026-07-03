"""
KestrelService — service layer for all k20_* database operations (P20 Kestrel pipeline).

Each method opens its own Unit-of-Work via @with_uow and delegates to KestrelRepo.
Module-level aliases at the bottom expose the same API as standalone functions
so that P20 modules can import them by name and tests can patch them per-module.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from src.data.db.services.base_service import BaseDBService, with_uow


class KestrelService(BaseDBService):
    """Unit-of-Work service for all k20_* tables."""

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    @with_uow
    def upsert_universe_rows(self, rows: List[Dict[str, Any]]) -> int:
        """Upsert universe rows (PK: ticker). Returns row count."""
        return self.repos.kestrel.upsert_universe_rows(rows)

    @with_uow
    def get_active_tickers(self) -> List[str]:
        """Return all tickers with status='active'."""
        return self.repos.kestrel.get_active_tickers()

    @with_uow
    def get_universe_row(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return a single universe row as a dict, or None."""
        return self.repos.kestrel.get_universe_row(ticker)

    @with_uow
    def mark_tickers_delisted(self, tickers: List[str]) -> int:
        """Set status='delisted' for all given tickers. Returns row count."""
        return self.repos.kestrel.mark_tickers_delisted(tickers)

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    @with_uow
    def upsert_signals(self, rows: List[Dict[str, Any]]) -> int:
        """Upsert signal rows (PK: ticker, date, signal_type). Returns row count."""
        return self.repos.kestrel.upsert_signals(rows)

    @with_uow
    def get_signals(
        self,
        ticker: str,
        signal_type: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Return signal rows for a ticker/type over an optional date range."""
        return self.repos.kestrel.get_signals(ticker, signal_type, start_date, end_date)

    @with_uow
    def get_latest_signal(self, ticker: str, signal_type: str) -> Optional[float]:
        """Return the most recent value for a (ticker, signal_type) pair."""
        return self.repos.kestrel.get_latest_signal(ticker, signal_type)

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------

    @with_uow
    def upsert_sentiment(self, rows: List[Dict[str, Any]]) -> int:
        """Upsert sentiment rows (PK: ticker, date, source). Returns row count."""
        return self.repos.kestrel.upsert_sentiment(rows)

    @with_uow
    def get_latest_sentiment(self, ticker: str, source: str) -> Optional[Dict[str, Any]]:
        """Return the most recent sentiment row for a (ticker, source) pair."""
        return self.repos.kestrel.get_latest_sentiment(ticker, source)

    @with_uow
    def get_sentiment_history(
        self,
        ticker: str,
        source: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Return sentiment history. Date range takes precedence over days limit."""
        return self.repos.kestrel.get_sentiment_history(ticker, source, start, end, days)

    # ------------------------------------------------------------------
    # Catalysts
    # ------------------------------------------------------------------

    @with_uow
    def upsert_catalyst(self, row: Dict[str, Any]) -> int:
        """Upsert a catalyst (natural key: ticker, event_type, event_date). Returns id."""
        return self.repos.kestrel.upsert_catalyst(row)

    @with_uow
    def get_upcoming_catalysts(self, ticker: str, days_ahead: int = 14) -> List[Dict[str, Any]]:
        """Return upcoming catalyst events for a ticker within a horizon."""
        return self.repos.kestrel.get_upcoming_catalysts(ticker, days_ahead)

    @with_uow
    def get_catalysts_in_window(self, days_ahead: int = 5) -> List[Dict[str, Any]]:
        """Return all upcoming catalysts for any ticker within days_ahead days."""
        return self.repos.kestrel.get_catalysts_in_window(days_ahead)

    @with_uow
    def get_all_upcoming_catalysts(self, days_ahead: int = 15) -> List[Dict[str, Any]]:
        """Return upcoming catalysts for ALL tickers within days_ahead days."""
        return self.repos.kestrel.get_all_upcoming_catalysts(days_ahead)

    @with_uow
    def get_past_spinoffs(self, days_min: int = 20, days_max: int = 60) -> List[Dict[str, Any]]:
        """Return spinoff catalysts whose event_date fell in [today-days_max, today-days_min]."""
        return self.repos.kestrel.get_past_spinoffs(days_min, days_max)

    @with_uow
    def stamp_catalyst_alert(self, catalyst_id: int, column: str) -> None:
        """Stamp an alert-time column (t10_alerted_at / t3_alerted_at) to now."""
        self.repos.kestrel.stamp_catalyst_alert(catalyst_id, column)

    # ------------------------------------------------------------------
    # LLM runs
    # ------------------------------------------------------------------

    @with_uow
    def get_llm_run_cached(self, task_type: str, input_ref: str) -> Optional[Dict[str, Any]]:
        """Return a previously cached LLM output, or None if not found."""
        return self.repos.kestrel.get_llm_run_cached(task_type, input_ref)

    @with_uow
    def insert_llm_run(self, row: Dict[str, Any]) -> int:
        """Insert / update an LLM run record. Returns id."""
        return self.repos.kestrel.insert_llm_run(row)

    @with_uow
    def get_pending_llm_runs(self, task_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Return LLM runs with no output yet (verdict IS NULL)."""
        return self.repos.kestrel.get_pending_llm_runs(task_type, limit)

    @with_uow
    def get_llm_monthly_spend(self) -> float:
        """Return the total LLM spend (USD) for the current calendar month."""
        return self.repos.kestrel.get_llm_monthly_spend()

    # ------------------------------------------------------------------
    # Watchlist
    # ------------------------------------------------------------------

    @with_uow
    def upsert_watchlist(self, row: Dict[str, Any]) -> None:
        """Upsert a watchlist entry (PK: ticker, sleeve)."""
        self.repos.kestrel.upsert_watchlist(row)

    @with_uow
    def get_watchlist(
        self, state: Optional[str] = None, sleeve: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return watchlist rows, optionally filtered by state and sleeve."""
        return self.repos.kestrel.get_watchlist(state, sleeve)

    @with_uow
    def get_watchlist_tickers(self) -> List[str]:
        """Return all watchlist tickers that are not rejected/expired."""
        return self.repos.kestrel.get_watchlist_tickers()

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    @with_uow
    def insert_position(self, row: Dict[str, Any]) -> int:
        """Insert a new position and return its generated id."""
        return self.repos.kestrel.insert_position(row)

    @with_uow
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Return all open positions."""
        return self.repos.kestrel.get_open_positions()

    @with_uow
    def update_position(self, position_id: int, updates: Dict[str, Any]) -> None:
        """Update fields on an existing position by id."""
        self.repos.kestrel.update_position(position_id, updates)

    # ------------------------------------------------------------------
    # Request budget
    # ------------------------------------------------------------------

    @with_uow
    def get_or_create_budget(self, source: str, run_date: date, quota: int) -> Dict[str, Any]:
        """Return today's budget row, creating it with used=0 if absent."""
        return self.repos.kestrel.get_or_create_budget(source, run_date, quota)

    @with_uow
    def increment_budget_used(self, source: str, run_date: date, amount: int = 1) -> int:
        """Increment the used count and return the new value."""
        return self.repos.kestrel.increment_budget_used(source, run_date, amount)

    @with_uow
    def update_budget_notes(self, source: str, run_date: date, notes: dict) -> None:
        """Store carry-over metadata in the notes JSON column."""
        self.repos.kestrel.update_budget_notes(source, run_date, notes)

    # ------------------------------------------------------------------
    # Job runs
    # ------------------------------------------------------------------

    @with_uow
    def start_job_run(self, job: str, run_date: date) -> None:
        """Mark a job as running (upsert)."""
        self.repos.kestrel.start_job_run(job, run_date)

    @with_uow
    def finish_job_run(
        self,
        job: str,
        run_date: date,
        status: str,
        rows_out: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark a job as finished (ok/failed/skipped)."""
        self.repos.kestrel.finish_job_run(job, run_date, status, rows_out, error)

    @with_uow
    def get_job_run(self, job: str, run_date: date) -> Optional[str]:
        """Return the status of a job run, or None if no row exists."""
        return self.repos.kestrel.get_job_run(job, run_date)

    # ------------------------------------------------------------------
    # Alerts log
    # ------------------------------------------------------------------

    @with_uow
    def log_alert(
        self,
        trigger: str,
        ticker: Optional[str] = None,
        payload: Optional[dict] = None,
        channel: Optional[str] = None,
    ) -> None:
        """Append an entry to the alerts log."""
        self.repos.kestrel.log_alert(trigger, ticker, payload, channel)

    @with_uow
    def get_today_alerts(self, trigger: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return alerts logged today, optionally filtered by trigger."""
        return self.repos.kestrel.get_today_alerts(trigger)

    # ------------------------------------------------------------------
    # Company aliases
    # ------------------------------------------------------------------

    @with_uow
    def upsert_aliases(self, rows: List[Dict[str, Any]]) -> int:
        """Upsert company alias rows (PK: ticker, alias). Returns row count."""
        return self.repos.kestrel.upsert_aliases(rows)

    @with_uow
    def get_all_aliases(self) -> List[Dict[str, Any]]:
        """Return all alias rows as dicts."""
        return self.repos.kestrel.get_all_aliases()

    @with_uow
    def get_blocklist(self) -> List[Dict[str, Any]]:
        """Return all alias blocklist entries."""
        return self.repos.kestrel.get_blocklist()
