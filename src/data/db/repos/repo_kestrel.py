"""
Repository layer for all k20_* tables (P20 Kestrel pipeline).

Accepts a SQLAlchemy Session in __init__ — never opens its own session.
All writes call self.session.flush() to materialise generated keys/counts
within the current Unit-of-Work; the surrounding service commits on success.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import extract, func as sa_func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.data.db.models.model_kestrel import (
    K20AlertsLog,
    K20AliasBlocklist,
    K20Catalyst,
    K20CompanyAlias,
    K20JobRun,
    K20LLMRun,
    K20Position,
    K20RequestBudget,
    K20SentimentDaily,
    K20Signal,
    K20Universe,
    K20Watchlist,
)


class KestrelRepo:
    """All repository operations for k20_* tables, bound to a single Session."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def upsert_universe_rows(self, rows: List[Dict[str, Any]]) -> int:
        """Upsert universe rows (PK: ticker). Returns row count."""
        if not rows:
            return 0
        stmt = pg_insert(K20Universe).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker"],
            set_={k: stmt.excluded[k] for k in rows[0] if k != "ticker"},
        )
        self.session.execute(stmt)
        return len(rows)

    def get_active_tickers(self) -> List[str]:
        """Return all tickers with status='active'."""
        rows = self.session.execute(
            select(K20Universe.ticker).where(K20Universe.status == "active")
        ).scalars().all()
        return list(rows)

    def get_universe_row(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return a single universe row as a dict, or None."""
        row = self.session.execute(
            select(K20Universe).where(K20Universe.ticker == ticker)
        ).scalars().first()
        if row is None:
            return None
        return {c.key: getattr(row, c.key) for c in K20Universe.__table__.columns}

    def mark_tickers_delisted(self, tickers: List[str]) -> int:
        """Set status='delisted' for all given tickers. Returns row count."""
        if not tickers:
            return 0
        result = self.session.execute(
            update(K20Universe)
            .where(K20Universe.ticker.in_(tickers))
            .values(status="delisted")
        )
        return result.rowcount

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def upsert_signals(self, rows: List[Dict[str, Any]]) -> int:
        """Upsert signal rows (PK: ticker, date, signal_type). Returns row count."""
        if not rows:
            return 0
        stmt = pg_insert(K20Signal).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "date", "signal_type"],
            set_={"value": stmt.excluded.value, "sleeve": stmt.excluded.sleeve},
        )
        self.session.execute(stmt)
        return len(rows)

    def get_signals(
        self,
        ticker: str,
        signal_type: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Return signal rows for a ticker/type over an optional date range."""
        q = select(K20Signal).where(
            K20Signal.ticker == ticker, K20Signal.signal_type == signal_type
        )
        if start_date:
            q = q.where(K20Signal.date >= start_date)
        if end_date:
            q = q.where(K20Signal.date <= end_date)
        q = q.order_by(K20Signal.date)
        rows = self.session.execute(q).scalars().all()
        return [{"date": r.date, "value": r.value, "sleeve": r.sleeve} for r in rows]

    def get_signals_for_date(self, ticker: str, on_date: date) -> Dict[str, float]:
        """Return all signals for a ticker on a date as {signal_type: value}."""
        rows = self.session.execute(
            select(K20Signal.signal_type, K20Signal.value).where(
                K20Signal.ticker == ticker, K20Signal.date == on_date
            )
        ).all()
        return {
            str(signal_type): float(value)
            for signal_type, value in rows
            if value is not None
        }

    def get_latest_signal(self, ticker: str, signal_type: str) -> Optional[float]:
        """Return the most recent value for a (ticker, signal_type) pair."""
        row = self.session.execute(
            select(K20Signal.value)
            .where(K20Signal.ticker == ticker, K20Signal.signal_type == signal_type)
            .order_by(K20Signal.date.desc())
            .limit(1)
        ).scalars().first()
        return float(row) if row is not None else None

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------

    def upsert_sentiment(self, rows: List[Dict[str, Any]]) -> int:
        """Upsert sentiment rows (PK: ticker, date, source). Returns row count."""
        if not rows:
            return 0
        update_cols = [
            "mentions", "avg_tone", "tone_std", "pos_score", "neg_score",
            "bullish_ratio", "top_domains", "mention_z20", "tone_z20",
        ]
        stmt = pg_insert(K20SentimentDaily).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "date", "source"],
            set_={c: stmt.excluded[c] for c in update_cols},
        )
        self.session.execute(stmt)
        return len(rows)

    def get_latest_sentiment(self, ticker: str, source: str) -> Optional[Dict[str, Any]]:
        """Return the most recent sentiment row for a (ticker, source) pair."""
        row = self.session.execute(
            select(K20SentimentDaily)
            .where(K20SentimentDaily.ticker == ticker, K20SentimentDaily.source == source)
            .order_by(K20SentimentDaily.date.desc())
            .limit(1)
        ).scalars().first()
        if row is None:
            return None
        return {c.key: getattr(row, c.key) for c in K20SentimentDaily.__table__.columns}

    def get_sentiment_history(
        self,
        ticker: str,
        source: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Return sentiment history. date range takes precedence over days limit."""
        q = select(K20SentimentDaily).where(
            K20SentimentDaily.ticker == ticker,
            K20SentimentDaily.source == source,
        )
        if start is not None:
            q = q.where(K20SentimentDaily.date >= start)
        if end is not None:
            q = q.where(K20SentimentDaily.date <= end)
        q = q.order_by(K20SentimentDaily.date.desc())
        if start is None and end is None:
            q = q.limit(days)
        rows = self.session.execute(q).scalars().all()
        return [
            {c.key: getattr(r, c.key) for c in K20SentimentDaily.__table__.columns}
            for r in reversed(rows)
        ]

    # ------------------------------------------------------------------
    # Catalysts
    # ------------------------------------------------------------------

    def upsert_catalyst(self, row: Dict[str, Any]) -> int:
        """Upsert a catalyst (natural key: ticker, event_type, event_date). Returns id."""
        existing = self.session.execute(
            select(K20Catalyst).where(
                K20Catalyst.ticker == row["ticker"],
                K20Catalyst.event_type == row["event_type"],
                K20Catalyst.event_date == row.get("event_date"),
            )
        ).scalars().first()
        if existing:
            for k, v in row.items():
                if k != "id":
                    setattr(existing, k, v)
            self.session.flush()
            return existing.id
        obj = K20Catalyst(**row)
        self.session.add(obj)
        self.session.flush()
        return obj.id

    def get_upcoming_catalysts(self, ticker: str, days_ahead: int = 14) -> List[Dict[str, Any]]:
        """Return upcoming catalyst events for a ticker within a horizon."""
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)
        rows = self.session.execute(
            select(K20Catalyst)
            .where(
                K20Catalyst.ticker == ticker,
                K20Catalyst.event_date >= today,
                K20Catalyst.event_date <= cutoff,
                K20Catalyst.state.in_(["upcoming", "date_changed"]),
            )
            .order_by(K20Catalyst.event_date)
        ).scalars().all()
        return [{c.key: getattr(r, c.key) for c in K20Catalyst.__table__.columns} for r in rows]

    def get_catalysts_in_window(self, days_ahead: int = 5) -> List[Dict[str, Any]]:
        """Return all upcoming catalysts for any ticker within days_ahead days."""
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)
        rows = self.session.execute(
            select(K20Catalyst)
            .where(
                K20Catalyst.event_date.between(today, cutoff),
                K20Catalyst.state.in_(["upcoming", "date_changed"]),
            )
            .order_by(K20Catalyst.event_date, K20Catalyst.ticker)
        ).scalars().all()
        return [{c.key: getattr(r, c.key) for c in K20Catalyst.__table__.columns} for r in rows]

    def get_all_upcoming_catalysts(self, days_ahead: int = 15) -> List[Dict[str, Any]]:
        """Return upcoming catalysts for ALL tickers within days_ahead days."""
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)
        rows = self.session.execute(
            select(K20Catalyst)
            .where(
                K20Catalyst.event_date.between(today, cutoff),
                K20Catalyst.state.in_(["upcoming", "date_changed"]),
            )
            .order_by(K20Catalyst.event_date)
        ).scalars().all()
        return [{c.key: getattr(r, c.key) for c in K20Catalyst.__table__.columns} for r in rows]

    def get_past_spinoffs(self, days_min: int = 20, days_max: int = 60) -> List[Dict[str, Any]]:
        """Return spinoff catalysts whose event_date fell in [today-days_max, today-days_min]."""
        today = date.today()
        window_start = today - timedelta(days=days_max)
        window_end = today - timedelta(days=days_min)
        rows = self.session.execute(
            select(K20Catalyst)
            .where(
                K20Catalyst.event_type == "spinoff",
                K20Catalyst.event_date.between(window_start, window_end),
            )
            .order_by(K20Catalyst.event_date)
        ).scalars().all()
        return [{c.key: getattr(r, c.key) for c in K20Catalyst.__table__.columns} for r in rows]

    def stamp_catalyst_alert(self, catalyst_id: int, column: str) -> None:
        """Stamp an alert-time column (t10_alerted_at / t3_alerted_at) to now."""
        allowed = {"t10_alerted_at", "t3_alerted_at", "datechange_alerted_at"}
        if column not in allowed:
            raise ValueError("column must be one of %s" % allowed)
        self.session.execute(
            update(K20Catalyst)
            .where(K20Catalyst.id == catalyst_id)
            .values(**{column: datetime.now(timezone.utc)})
        )

    # ------------------------------------------------------------------
    # LLM runs
    # ------------------------------------------------------------------

    def get_llm_run_cached(self, task_type: str, input_ref: str) -> Optional[Dict[str, Any]]:
        """Return a previously cached LLM output, or None if not found."""
        row = self.session.execute(
            select(K20LLMRun).where(
                K20LLMRun.task_type == task_type,
                K20LLMRun.input_ref == input_ref,
            )
        ).scalars().first()
        if row is None:
            return None
        return {c.key: getattr(row, c.key) for c in K20LLMRun.__table__.columns}

    def insert_llm_run(self, row: Dict[str, Any]) -> int:
        """Insert / update an LLM run record on (task_type, input_ref) conflict. Returns id."""
        stmt = pg_insert(K20LLMRun).values(row)
        update_cols = ["output_json", "model", "tokens_in", "tokens_out", "cost_usd", "verdict", "ts"]
        stmt = stmt.on_conflict_do_update(
            constraint="uq_k20_llm_runs_task_ref",
            set_={c: stmt.excluded[c] for c in update_cols if c in row},
        )
        result = self.session.execute(stmt)
        self.session.flush()
        pk = result.inserted_primary_key
        return pk[0] if pk else 0

    def get_pending_llm_runs(self, task_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Return LLM runs with no output yet (verdict IS NULL)."""
        rows = self.session.execute(
            select(K20LLMRun)
            .where(K20LLMRun.task_type == task_type, K20LLMRun.verdict.is_(None))
            .order_by(K20LLMRun.ts)
            .limit(limit)
        ).scalars().all()
        return [{c.key: getattr(r, c.key) for c in K20LLMRun.__table__.columns} for r in rows]

    def get_llm_monthly_spend(self) -> float:
        """Return the total LLM spend (USD) for the current calendar month."""
        today = date.today()
        result = self.session.execute(
            select(sa_func.coalesce(sa_func.sum(K20LLMRun.cost_usd), 0)).where(
                extract("year", K20LLMRun.ts) == today.year,
                extract("month", K20LLMRun.ts) == today.month,
            )
        ).scalar()
        return float(result or 0)

    # ------------------------------------------------------------------
    # Watchlist
    # ------------------------------------------------------------------

    def upsert_watchlist(self, row: Dict[str, Any]) -> None:
        """Upsert a watchlist entry (PK: ticker, sleeve)."""
        stmt = pg_insert(K20Watchlist).values(row)
        update_cols = ["score", "llm_verdict", "dossier_run_id", "thesis_short", "state"]
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "sleeve"],
            set_={c: stmt.excluded[c] for c in update_cols if c in row},
        )
        self.session.execute(stmt)

    def get_watchlist(
        self, state: Optional[str] = None, sleeve: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return watchlist rows, optionally filtered by state and sleeve."""
        q = select(K20Watchlist)
        if state:
            q = q.where(K20Watchlist.state == state)
        if sleeve:
            q = q.where(K20Watchlist.sleeve == sleeve)
        q = q.order_by(K20Watchlist.score.desc().nullslast())
        rows = self.session.execute(q).scalars().all()
        return [{c.key: getattr(r, c.key) for c in K20Watchlist.__table__.columns} for r in rows]

    def get_watchlist_tickers(self) -> List[str]:
        """Return all tickers on the watchlist that are not rejected/expired."""
        rows = self.session.execute(
            select(K20Watchlist.ticker).where(
                K20Watchlist.state.notin_(["rejected", "expired"])
            )
        ).scalars().all()
        return list(set(rows))

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def insert_position(self, row: Dict[str, Any]) -> int:
        """Insert a new position and return its generated id."""
        obj = K20Position(**row)
        self.session.add(obj)
        self.session.flush()
        return obj.id

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Return all open positions."""
        rows = self.session.execute(
            select(K20Position).where(K20Position.status == "open")
        ).scalars().all()
        return [{c.key: getattr(r, c.key) for c in K20Position.__table__.columns} for r in rows]

    def update_position(self, position_id: int, updates: Dict[str, Any]) -> None:
        """Update fields on an existing position by id."""
        self.session.execute(
            update(K20Position).where(K20Position.id == position_id).values(**updates)
        )

    # ------------------------------------------------------------------
    # Request budget
    # ------------------------------------------------------------------

    def get_or_create_budget(self, source: str, run_date: date, quota: int) -> Dict[str, Any]:
        """Return today's budget row, creating it with used=0 if absent."""
        stmt = pg_insert(K20RequestBudget).values(
            source=source, date=run_date, quota=quota, used=0
        ).on_conflict_do_nothing()
        self.session.execute(stmt)
        self.session.flush()
        row = self.session.execute(
            select(K20RequestBudget).where(
                K20RequestBudget.source == source, K20RequestBudget.date == run_date
            )
        ).scalars().first()
        return {c.key: getattr(row, c.key) for c in K20RequestBudget.__table__.columns}

    def increment_budget_used(self, source: str, run_date: date, amount: int = 1) -> int:
        """Increment the used count and return the new value."""
        self.session.execute(
            update(K20RequestBudget)
            .where(K20RequestBudget.source == source, K20RequestBudget.date == run_date)
            .values(used=K20RequestBudget.used + amount)
        )
        self.session.flush()
        row = self.session.execute(
            select(K20RequestBudget.used).where(
                K20RequestBudget.source == source, K20RequestBudget.date == run_date
            )
        ).scalars().first()
        return int(row or 0)

    def update_budget_notes(self, source: str, run_date: date, notes: dict) -> None:
        """Store carry-over metadata in the notes JSON column."""
        self.session.execute(
            update(K20RequestBudget)
            .where(K20RequestBudget.source == source, K20RequestBudget.date == run_date)
            .values(notes=notes)
        )

    # ------------------------------------------------------------------
    # Job runs
    # ------------------------------------------------------------------

    def start_job_run(self, job: str, run_date: date) -> None:
        """Mark a job as running (upsert; overwrites a previous failed/skipped row)."""
        stmt = pg_insert(K20JobRun).values(
            job=job, run_date=run_date, status="running",
            started_at=datetime.now(timezone.utc),
        ).on_conflict_do_update(
            index_elements=["job", "run_date"],
            set_={"status": "running", "started_at": datetime.now(timezone.utc), "error": None},
        )
        self.session.execute(stmt)

    def finish_job_run(
        self,
        job: str,
        run_date: date,
        status: str,
        rows_out: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark a job as finished (ok/failed/skipped)."""
        self.session.execute(
            update(K20JobRun)
            .where(K20JobRun.job == job, K20JobRun.run_date == run_date)
            .values(
                status=status,
                finished_at=datetime.now(timezone.utc),
                rows_out=rows_out,
                error=error,
            )
        )

    def get_job_run(self, job: str, run_date: date) -> Optional[str]:
        """Return the status of a job run, or None if no row exists."""
        row = self.session.execute(
            select(K20JobRun.status).where(
                K20JobRun.job == job, K20JobRun.run_date == run_date
            )
        ).scalars().first()
        return row

    # ------------------------------------------------------------------
    # Alerts log
    # ------------------------------------------------------------------

    def log_alert(
        self,
        trigger: str,
        ticker: Optional[str] = None,
        payload: Optional[dict] = None,
        channel: Optional[str] = None,
    ) -> None:
        """Append an entry to the alerts log."""
        self.session.add(K20AlertsLog(
            ticker=ticker, trigger=trigger, payload=payload, channel=channel,
        ))

    def get_today_alerts(self, trigger: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return alerts logged today, optionally filtered by trigger."""
        today = date.today()
        q = select(K20AlertsLog).where(
            K20AlertsLog.ts >= datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)
        )
        if trigger:
            q = q.where(K20AlertsLog.trigger == trigger)
        rows = self.session.execute(q.order_by(K20AlertsLog.ts)).scalars().all()
        return [{c.key: getattr(r, c.key) for c in K20AlertsLog.__table__.columns} for r in rows]

    # ------------------------------------------------------------------
    # Company aliases
    # ------------------------------------------------------------------

    def upsert_aliases(self, rows: List[Dict[str, Any]]) -> int:
        """Upsert company alias rows (PK: ticker, alias). Returns row count."""
        if not rows:
            return 0
        update_cols = ["alias_type", "normalized_alias"]
        stmt = pg_insert(K20CompanyAlias).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "alias"],
            set_={c: stmt.excluded[c] for c in update_cols},
        )
        self.session.execute(stmt)
        return len(rows)

    def get_all_aliases(self) -> List[Dict[str, Any]]:
        """Return all alias rows as dicts."""
        rows = self.session.execute(select(K20CompanyAlias)).scalars().all()
        return [
            {
                "ticker": r.ticker,
                "alias": r.alias,
                "alias_type": r.alias_type,
                "normalized_alias": r.normalized_alias,
            }
            for r in rows
        ]

    def get_blocklist(self) -> List[Dict[str, Any]]:
        """Return all alias blocklist entries."""
        rows = self.session.execute(select(K20AliasBlocklist)).scalars().all()
        return [
            {"alias": r.alias, "ticker": r.ticker, "match_policy": r.match_policy}
            for r in rows
        ]
