"""
SQLite task manifest (spec §6.2): the single source of truth for what has been downloaded.

- One row per atomic request; ``task_id = sha1(dump_id|endpoint|symbol|canonical_params)``, so re-planning the
  same work is a no-op (``INSERT OR IGNORE``).
- Status machine: ``pending`` → ``running`` → ``ok`` | ``ok_empty`` | ``plan_restricted`` | ``api_error`` |
  ``failed`` | ``possibly_truncated``; ``skipped`` is set by the planner. ``possibly_truncated`` is an intraday window
  that hit the row cap; its two halves are planned instead (spec §6.5). On open, ``running`` rows revert to
  ``pending``: a run killed mid-request simply redoes that request.
- Durability: WAL journal with ``synchronous=FULL``, one transaction per state change. SQLite's own journal gives
  the crash atomicity the spec asks for; the temp-file + fsync + rename rule applies to the plain files the archive
  writes (``fsutil.atomic_write_bytes``), not to the database.
- ``bandwidth_log`` records every response's wire and decoded bytes for the rolling 30-day cap (spec §6.3).
- ``meta`` holds per-dump settings that must not change between plans (the ``as_of`` date); ``symbol_bounds``
  the first and last daily bar seen per symbol, which bounds the intraday windows (spec §6.5).

The API key never appears here: ``params_json`` holds only query parameters.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from src.data.fmp_pipeline.client.fmp_client import FetchResult

STATUSES = ("pending", "running", "ok", "ok_empty", "api_error", "plan_restricted", "failed", "skipped",
            "possibly_truncated")
FINAL_STATUSES = ("ok", "ok_empty", "api_error", "plan_restricted", "failed", "skipped", "possibly_truncated")

#: Final classification of a FetchResult -> manifest status.
STATUS_BY_CLASSIFICATION = {
    "ok": "ok",
    "ok_empty": "ok_empty",
    "plan_restricted": "plan_restricted",
    "api_error": "api_error",
    "not_found": "api_error",
    "unauthorized": "failed",
    "rate_limited": "failed",
    "server_error": "failed",
    "network_error": "failed",
    "possibly_truncated": "possibly_truncated",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks(
  task_id       TEXT PRIMARY KEY,
  dump_id       TEXT NOT NULL,
  endpoint      TEXT NOT NULL,
  symbol        TEXT NULL,
  params_json   TEXT NOT NULL,
  priority      INTEGER NOT NULL,
  tier          TEXT NULL,
  status        TEXT NOT NULL,
  attempts      INTEGER NOT NULL DEFAULT 0,
  http_status   INTEGER NULL,
  rows          INTEGER NULL,
  bytes_wire    INTEGER NULL,
  bytes_decoded INTEGER NULL,
  sha256        TEXT NULL,
  fetched_at    TEXT NULL,
  error         TEXT NULL
);
CREATE INDEX IF NOT EXISTS ix_tasks_queue ON tasks(status, priority, task_id);
CREATE TABLE IF NOT EXISTS bandwidth_log(ts TEXT NOT NULL, bytes_wire INTEGER NOT NULL, bytes_decoded INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS ix_bandwidth_ts ON bandwidth_log(ts);
CREATE TABLE IF NOT EXISTS run_events(ts TEXT NOT NULL, event TEXT NOT NULL, detail TEXT NULL);
CREATE TABLE IF NOT EXISTS meta(dump_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                                PRIMARY KEY(dump_id, key));
CREATE TABLE IF NOT EXISTS symbol_bounds(dump_id TEXT NOT NULL, symbol TEXT NOT NULL, first_date TEXT NOT NULL,
                                         last_date TEXT NOT NULL, PRIMARY KEY(dump_id, symbol));
"""


def canonical_params(params: dict[str, Any]) -> str:
    """Return a deterministic JSON encoding of request parameters (sorted keys, no whitespace)."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def task_id(dump_id: str, endpoint: str, symbol: Optional[str], params: dict[str, Any]) -> str:
    """Return ``sha1(dump_id|endpoint|symbol|canonical_params)`` (spec §6.2)."""
    raw = f"{dump_id}|{endpoint}|{symbol or ''}|{canonical_params(params)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskSpec:
    """A request to plan. ``params`` excludes ``symbol``, which is stored in its own column."""

    endpoint: str
    symbol: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    priority: int = 100
    tier: Optional[str] = None


@dataclass(frozen=True)
class Task:
    """A manifest row as claimed by a worker."""

    task_id: str
    dump_id: str
    endpoint: str
    symbol: Optional[str]
    params: dict[str, Any]
    priority: int
    tier: Optional[str]
    attempts: int

    @property
    def request_params(self) -> dict[str, Any]:
        """Query parameters to send: ``symbol`` (if any) plus the stored params."""
        return ({"symbol": self.symbol} if self.symbol is not None else {}) | self.params


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Manifest:
    """The task manifest of one dump, backed by one SQLite file."""

    def __init__(self, path: Path, dump_id: str, recover: bool = True) -> None:
        """
        Open (or create) the manifest and revert interrupted ``running`` tasks to ``pending``.

        Args:
            path: ``manifest.sqlite`` location; parent directories are created.
            dump_id: Vintage id (e.g. ``2026-10``); every task belongs to exactly one dump.
            recover: Revert ``running`` tasks to ``pending``. Only the dump's writer (holding the dump lock) may do
                this; a reader such as ``fmp status`` passes ``False`` so it never disturbs a live run.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.dump_id = dump_id
        self._db = sqlite3.connect(str(path), isolation_level=None, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(_SCHEMA)
        self.recovered = self.reset_running() if recover else 0

    def close(self) -> None:
        """Close the connection."""
        self._db.close()

    def __enter__(self) -> "Manifest":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        del _exc_info
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    # --- planning --------------------------------------------------------------------------------------------

    def add_tasks(self, specs: Iterable[TaskSpec]) -> int:
        """
        Insert tasks as ``pending``; tasks already present (same ``task_id``) are left untouched.

        Args:
            specs: Tasks to plan.

        Returns:
            Number of new tasks inserted.
        """
        rows = [(task_id(self.dump_id, s.endpoint, s.symbol, s.params), self.dump_id, s.endpoint, s.symbol,
                 canonical_params(s.params), s.priority, s.tier) for s in specs]
        with self._tx() as db:
            before = db.total_changes
            db.executemany(
                "INSERT OR IGNORE INTO tasks(task_id, dump_id, endpoint, symbol, params_json, priority, tier, status)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')", rows)
            return db.total_changes - before

    def reset_running(self) -> int:
        """Revert ``running`` tasks to ``pending`` (crash recovery); return how many were reverted."""
        with self._tx() as db:
            return db.execute("UPDATE tasks SET status='pending' WHERE status='running' AND dump_id=?",
                              (self.dump_id,)).rowcount

    # --- execution -------------------------------------------------------------------------------------------

    def claim(self, limit: int = 1, max_priority: Optional[int] = None, min_priority: Optional[int] = None,
              tier: Optional[str] = None) -> list[Task]:
        """
        Atomically move the highest-priority ``pending`` tasks to ``running`` and increment their attempts.

        Args:
            limit: Maximum number of tasks.
            max_priority: If given, only tasks with ``priority <= max_priority`` are claimed.
            min_priority: If given, only tasks with ``priority >= min_priority`` are claimed.
            tier: If given, only tasks of this tier are claimed.

        Returns:
            The claimed tasks, lowest ``priority`` first (ties by ``task_id``); empty when nothing is pending.
        """
        sql = "SELECT * FROM tasks WHERE status='pending' AND dump_id=? AND priority BETWEEN ? AND ?"
        args: list[Any] = [self.dump_id, min_priority if min_priority is not None else -(2**62),
                           max_priority if max_priority is not None else 2**62]
        if tier is not None:
            sql += " AND tier=?"
            args.append(tier)
        with self._tx() as db:
            rows = db.execute(sql + " ORDER BY priority, task_id LIMIT ?", (*args, limit)).fetchall()
            db.executemany("UPDATE tasks SET status='running', attempts=attempts+1 WHERE task_id=?",
                           [(r["task_id"],) for r in rows])
        return [Task(r["task_id"], r["dump_id"], r["endpoint"], r["symbol"], json.loads(r["params_json"]),
                     r["priority"], r["tier"], r["attempts"] + 1) for r in rows]

    def complete(self, task: Task, result: FetchResult, sha256: Optional[str] = None) -> str:
        """
        Record a task's final outcome.

        Args:
            task: The claimed task.
            result: The client's final result for it.
            sha256: Checksum of the stored body (from the archive), if stored.

        Returns:
            The status written.
        """
        status = STATUS_BY_CLASSIFICATION.get(result.classification, "failed")
        error = result.error if status not in ("ok", "ok_empty") else None
        if status == "failed" and not error:
            error = result.classification
        with self._tx() as db:
            db.execute(
                "UPDATE tasks SET status=?, http_status=?, rows=?, bytes_wire=?, bytes_decoded=?, sha256=?,"
                " fetched_at=?, error=? WHERE task_id=?",
                (status, result.http_status, result.rows, result.bytes_wire, result.bytes_decoded, sha256,
                 result.fetched_at, error, task.task_id))
        return status

    def fail(self, task: Task, error: str) -> None:
        """Mark a task ``failed`` with an error message (e.g. the sink raised)."""
        with self._tx() as db:
            db.execute("UPDATE tasks SET status='failed', error=?, fetched_at=? WHERE task_id=?",
                       (error[:1000], _now().isoformat(), task.task_id))

    def release(self, task: Task) -> None:
        """Return a claimed task to ``pending`` without counting the attempt (the run stopped before finishing it)."""
        with self._tx() as db:
            db.execute("UPDATE tasks SET status='pending', attempts=MAX(attempts-1, 0) WHERE task_id=?",
                       (task.task_id,))

    def requeue_failed(self) -> int:
        """Move every ``failed`` task back to ``pending`` for another run; return how many."""
        with self._tx() as db:
            return db.execute("UPDATE tasks SET status='pending' WHERE status='failed' AND dump_id=?",
                              (self.dump_id,)).rowcount

    # --- bandwidth and reporting -----------------------------------------------------------------------------

    def record_bandwidth(self, bytes_wire: int, bytes_decoded: int, when: Optional[datetime] = None) -> None:
        """Append one response's sizes to ``bandwidth_log``."""
        ts = (when or _now()).isoformat()
        with self._tx() as db:
            db.execute("INSERT INTO bandwidth_log(ts, bytes_wire, bytes_decoded) VALUES (?, ?, ?)",
                       (ts, bytes_wire, bytes_decoded))

    def bandwidth_used(self, window_days: int = 30, now: Optional[datetime] = None) -> tuple[int, int]:
        """
        Return ``(bytes_wire, bytes_decoded)`` logged within the trailing window.

        The log is shared by every dump in this file, as FMP's cap is per account.
        """
        since = ((now or _now()) - timedelta(days=window_days)).isoformat()
        row = self._db.execute("SELECT COALESCE(SUM(bytes_wire), 0), COALESCE(SUM(bytes_decoded), 0)"
                               " FROM bandwidth_log WHERE ts >= ?", (since,)).fetchone()
        return int(row[0]), int(row[1])

    def log_event(self, event: str, detail: str = "") -> None:
        """Record a run-level event (start, stop reason, quota hit time)."""
        with self._tx() as db:
            db.execute("INSERT INTO run_events(ts, event, detail) VALUES (?, ?, ?)",
                       (_now().isoformat(), event, detail))

    def events(self) -> list[tuple[str, str, str]]:
        """Return all run events, oldest first."""
        return [(r[0], r[1], r[2]) for r in self._db.execute("SELECT ts, event, detail FROM run_events ORDER BY rowid")]

    def counts(self) -> dict[str, int]:
        """Return the number of tasks per status for this dump (statuses with no tasks are omitted)."""
        return {r[0]: r[1] for r in self._db.execute(
            "SELECT status, COUNT(*) FROM tasks WHERE dump_id=? GROUP BY status", (self.dump_id,))}

    def task(self, task_id_: str) -> Optional[dict[str, Any]]:
        """Return one task row as a dict, or ``None``."""
        row = self._db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id_,)).fetchone()
        return dict(row) if row is not None else None

    def task_rows(self, endpoints: Optional[Iterable[str]] = None) -> list[dict[str, Any]]:
        """
        Return this dump's task rows as dicts, optionally only for some endpoints.

        Args:
            endpoints: Endpoint names to include; ``None`` means all.

        Returns:
            Rows ordered by priority, then ``task_id``.
        """
        sql, args = "SELECT * FROM tasks WHERE dump_id=?", [self.dump_id]
        if endpoints is not None:
            names = list(endpoints)
            sql += f" AND endpoint IN ({','.join('?' * len(names))})"
            args.extend(names)
        return [dict(r) for r in self._db.execute(sql + " ORDER BY priority, task_id", args)]

    def finished_ids(self) -> set[str]:
        """Return the ids of this dump's tasks in a final status."""
        marks = ",".join("?" * len(FINAL_STATUSES))
        return {r[0] for r in self._db.execute(
            f"SELECT task_id FROM tasks WHERE dump_id=? AND status IN ({marks})", (self.dump_id, *FINAL_STATUSES))}

    def totals(self) -> list[tuple[int, Optional[str], str, int, int, int]]:
        """Return ``(priority, tier, status, tasks, bytes_wire, bytes_decoded)`` groups for this dump."""
        return [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in self._db.execute(
            "SELECT priority, tier, status, COUNT(*), COALESCE(SUM(bytes_wire), 0), COALESCE(SUM(bytes_decoded), 0)"
            " FROM tasks WHERE dump_id=? GROUP BY priority, tier, status", (self.dump_id,))]

    def failures(self, limit: int = 10) -> list[tuple[str, Optional[str], str]]:
        """Return up to ``limit`` ``(endpoint, symbol, error)`` of ``failed`` tasks, most recent first."""
        return [(r[0], r[1], r[2] or "") for r in self._db.execute(
            "SELECT endpoint, symbol, error FROM tasks WHERE dump_id=? AND status='failed'"
            " ORDER BY fetched_at DESC LIMIT ?", (self.dump_id, limit))]

    def pending_symbols(self) -> set[str]:
        """Return the symbols that still have ``pending`` tasks."""
        return {r[0] for r in self._db.execute(
            "SELECT DISTINCT symbol FROM tasks WHERE dump_id=? AND status='pending' AND symbol IS NOT NULL",
            (self.dump_id,))}

    def skip_symbols(self, symbols: Iterable[str], reason: str) -> int:
        """
        Mark the ``pending`` tasks of ``symbols`` as ``skipped`` (the planner dropped them from the universe).

        Args:
            symbols: Symbols to skip.
            reason: Stored in ``error``.

        Returns:
            Number of tasks skipped.
        """
        rows = [(reason, self.dump_id, s) for s in symbols]
        with self._tx() as db:
            before = db.total_changes
            db.executemany("UPDATE tasks SET status='skipped', error=? WHERE dump_id=? AND symbol=? AND"
                           " status='pending'", rows)
            return db.total_changes - before

    def symbols_finished(self, endpoint: str) -> set[str]:
        """Return the symbols whose ``endpoint`` tasks (at least one) are all in a final status."""
        marks = ",".join("?" * len(FINAL_STATUSES))
        return {r[0] for r in self._db.execute(
            f"SELECT symbol FROM tasks WHERE dump_id=? AND endpoint=? AND symbol IS NOT NULL GROUP BY symbol"
            f" HAVING SUM(status NOT IN ({marks}))=0", (self.dump_id, endpoint, *FINAL_STATUSES))}

    # --- per-dump metadata -----------------------------------------------------------------------------------

    def meta(self, key: str) -> Optional[str]:
        """Return a stored per-dump value, or ``None``."""
        row = self._db.execute("SELECT value FROM meta WHERE dump_id=? AND key=?", (self.dump_id, key)).fetchone()
        return str(row[0]) if row is not None else None

    def meta_setdefault(self, key: str, value: str) -> str:
        """
        Store ``value`` under ``key`` unless a value is already stored; return the stored value.

        Used for settings that are part of task identity (``as_of``): the first plan fixes them for the dump.
        """
        with self._tx() as db:
            db.execute("INSERT OR IGNORE INTO meta(dump_id, key, value) VALUES (?, ?, ?)", (self.dump_id, key, value))
        stored = self.meta(key)
        assert stored is not None
        return stored

    def meta_set(self, key: str, value: str) -> None:
        """Store (or replace) a per-dump value."""
        with self._tx() as db:
            db.execute("INSERT INTO meta(dump_id, key, value) VALUES (?, ?, ?) ON CONFLICT(dump_id, key)"
                       " DO UPDATE SET value=excluded.value", (self.dump_id, key, value))

    def update_bounds(self, symbol: str, first: date, last: date) -> None:
        """Widen the recorded first/last daily bar dates of ``symbol`` to include ``first`` … ``last``."""
        with self._tx() as db:
            db.execute(
                "INSERT INTO symbol_bounds(dump_id, symbol, first_date, last_date) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(dump_id, symbol) DO UPDATE SET first_date=MIN(first_date, excluded.first_date),"
                " last_date=MAX(last_date, excluded.last_date)",
                (self.dump_id, symbol, first.isoformat(), last.isoformat()))

    def bounds(self) -> dict[str, tuple[date, date]]:
        """Return ``{symbol: (first daily bar date, last daily bar date)}`` for this dump."""
        return {r[0]: (date.fromisoformat(r[1]), date.fromisoformat(r[2])) for r in self._db.execute(
            "SELECT symbol, first_date, last_date FROM symbol_bounds WHERE dump_id=?", (self.dump_id,))}
