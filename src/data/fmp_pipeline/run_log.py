"""
Structured run log of ``fmp run`` (spec §13): one JSON object per line, plus a progress summary every minute.

``<data_root>/logs/<dump_id>/run-<UTC start>.jsonl`` gets:

- ``run_start`` / ``block_start`` / ``run_stop`` records;
- one ``request`` record per finished task: ``request_id`` (the task id), ``worker``, ``endpoint``, ``symbol``,
  ``params`` (canonical, never the API key: the key travels only in a header), ``tier``, ``priority``,
  ``status``, ``http_status``, ``rows``, ``bytes_wire``, ``bytes_decoded``, ``attempts``, ``duration_ms`` and a
  shortened ``error``. Response bodies are never logged;
- one ``progress`` record per ``interval_s`` (and a final one): tasks finished this run by status, requests per
  minute over the last interval and over the run, the dump's pending tasks, bytes in the rolling bandwidth window
  against the hard stop, and an ETA at the run's average rate. The same summary goes to the normal logger, so a
  console watching the run sees it once a minute even when the run is stalled (circuit breaker, slow server).

Every line is flushed immediately, so the log of a killed run is complete up to the kill. Timestamps are UTC.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Optional

from src.notification.logger import setup_logger

if TYPE_CHECKING:
    from src.data.fmp_pipeline.client.fmp_client import FetchResult
    from src.data.fmp_pipeline.manifest.manifest import Manifest, Task
    from src.data.fmp_pipeline.runner import BandwidthGuard

_logger = setup_logger(__name__)

GB = 1024**3
ERROR_CHARS = 300


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def run_log_path(data_root: Path, dump_id: str, started: Optional[datetime] = None) -> Path:
    """``<data_root>/logs/<dump_id>/run-<YYYYmmddTHHMMSSZ>.jsonl`` for a run started at ``started`` (default now)."""
    stamp = (started or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return data_root / "logs" / dump_id / f"run-{stamp}.jsonl"


class RunLog:
    """Writes the JSON-lines run log and the once-a-minute progress summary."""

    def __init__(self, path: Path, dump_id: str, interval_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """
        Args:
            path: The ``.jsonl`` file (created with its parents; appended to if it exists).
            dump_id: Written into every record.
            interval_s: Seconds between progress summaries.
            clock: Monotonic clock (tests inject a fake one).
        """
        self.path = path
        self.dump_id = dump_id
        self.interval_s = interval_s
        self._clock = clock
        self._started = clock()
        self._last_progress = self._started
        self._finished_at_last = 0
        self.finished: Counter[str] = Counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: Optional[IO[str]] = path.open("a", encoding="utf-8")

    # --- records ------------------------------------------------------------------------------------------------

    def write(self, event: str, **fields: Any) -> None:
        """Append one record ``{"ts", "event", "dump_id", **fields}`` and flush it."""
        if self._fh is None:
            return
        record = {"ts": _utc_now(), "event": event, "dump_id": self.dump_id, **fields}
        self._fh.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
        self._fh.flush()

    def request(self, worker: int, task: "Task", status: str, duration_s: float,
                result: Optional["FetchResult"] = None, error: str = "") -> None:
        """
        Record one finished task.

        Args:
            worker: Worker number.
            task: The claimed task.
            status: The status the manifest recorded (``ok``, ``ok_empty``, ``failed`` …).
            duration_s: Wall time from claim to completion.
            result: The final response, if one arrived.
            error: Error text when there is no response (or in addition to it).
        """
        self.finished[status] += 1
        text = error or (result.error if result is not None else "")
        self.write("request", request_id=task.task_id, worker=worker, endpoint=task.endpoint, symbol=task.symbol,
                   params=task.params, tier=task.tier, priority=task.priority, status=status,
                   http_status=result.http_status if result is not None else None,
                   rows=result.rows if result is not None else None,
                   bytes_wire=result.bytes_wire if result is not None else 0,
                   bytes_decoded=result.bytes_decoded if result is not None else 0,
                   attempts=result.attempts if result is not None else task.attempts,
                   duration_ms=round(duration_s * 1000), error=text[:ERROR_CHARS] or None)

    # --- progress -----------------------------------------------------------------------------------------------

    def progress(self, manifest: "Manifest", guard: "BandwidthGuard", final: bool = False) -> dict[str, Any]:
        """
        Write a ``progress`` record and log the same summary.

        Args:
            manifest: The dump's manifest (pending counts, bandwidth window).
            guard: The run's bandwidth guard (basis, cap, safety, bytes this run).
            final: Marks the summary written at the end of the run.

        Returns:
            The record's fields.
        """
        now = self._clock()
        elapsed_min = max(now - self._started, 1e-9) / 60
        interval_min = max(now - self._last_progress, 1e-9) / 60
        done = sum(self.finished.values())
        recent = done - self._finished_at_last
        counts = manifest.counts()
        pending = counts.get("pending", 0) + counts.get("running", 0)
        wire, decoded = manifest.bandwidth_used(guard.window_days)
        used = wire if guard.basis == "wire" else decoded
        limit = guard.limit_bytes
        rate = done / elapsed_min
        eta_h = pending / rate / 60 if rate > 0 else None
        fields: dict[str, Any] = {
            "final": final, "finished": done, "by_status": dict(self.finished),
            "rate_per_min": round(recent / interval_min, 1), "avg_rate_per_min": round(rate, 1),
            "pending": pending, "window_bytes": used, "window_limit_bytes": int(limit), "basis": guard.basis,
            "run_bytes": guard.run_used, "eta_hours": round(eta_h, 2) if eta_h is not None else None,
        }
        self.write("progress", **fields)
        self._last_progress = now
        self._finished_at_last = done
        eta = f"{eta_h:.1f} h" if eta_h is not None else "unknown"
        statuses = ", ".join(f"{k} {v}" for k, v in sorted(self.finished.items())) or "none"
        _logger.info("Progress: %d tasks this run (%s), %.0f/min now, %.0f/min average; %d pending, ETA %s; "
                     "bandwidth %.2f of %.2f GB (%s, %d d)", done, statuses, fields["rate_per_min"], rate, pending,
                     eta, used / GB, limit / GB, guard.basis, guard.window_days)
        return fields

    def close(self) -> None:
        """Close the file; later writes are ignored."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None
