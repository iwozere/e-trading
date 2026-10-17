"""
Resumable download loop (spec §6.2–§6.4): workers claim tasks from the manifest, fetch them, hand every
response to a sink, then record the outcome.

Order per task is **review, follow-ups, sink, then manifest**. A ``Reviewer`` (the planner's paging and intraday
truncation rules) may relabel the result and plan follow-up tasks; those are inserted before the task is completed,
so a crash can only re-plan them (idempotent), never lose them. A crash between sink and manifest re-fetches the task
on the next run, so the sink can see a response twice but never loses one. The raw archive (WP-5) is append-only
and its reader keeps the newest line per request, which makes duplicates harmless (spec §6.6).

Run-level stops, each recorded in ``run_events`` and leaving unfinished tasks ``pending``:

- ``QuotaExhausted``: stop immediately (spec §6.2).
- ``AuthAbort``: every recent request was 401/403.
- Bandwidth: before each claim, if the rolling-window bytes reach ``safety × cap`` (default 0.90 × 50 GB,
  counted on ``decoded`` bytes until the paid probe proves the cap counts wire bytes; spec §6.3), stop cleanly.
- ``ArchiveUnavailable`` from the sink: the archive drive vanished, filled up or failed. The task goes back to
  ``pending`` (its response is fetched again next run) instead of every remaining task failing (spec §13).
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from src.data.fmp_pipeline.client.fmp_client import FetchResult, FmpClient
from src.data.fmp_pipeline.errors import ArchiveUnavailable, AuthAbort, QuotaExhausted
from src.data.fmp_pipeline.manifest.manifest import Manifest, Task, TaskSpec
from src.data.fmp_pipeline.run_log import RunLog
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

#: Receives every final response. Returns the stored body's sha256 (or ``None``); may be sync or async.
Sink = Callable[[Task, FetchResult], Union[Optional[str], Awaitable[Optional[str]]]]
#: Inspects a final response before it is stored: returns the (possibly relabelled) result and follow-up tasks.
Reviewer = Callable[[Task, FetchResult], tuple[FetchResult, list[TaskSpec]]]
BandwidthBasis = Literal["wire", "decoded"]
GB = 1024**3


def sha256_sink(task: Task, result: FetchResult) -> str:
    """Default sink: store nothing, return the body's sha256 (for tests and dry runs)."""
    del task
    return hashlib.sha256(result.body).hexdigest()


@dataclass
class BandwidthGuard:
    """
    Hard stop at ``safety × cap_bytes`` over the trailing ``window_days`` (spec §6.3).

    The window total is read from ``bandwidth_log`` at most every ``refresh_s`` seconds; in between, bytes reported
    through ``add`` are added in memory, so the check before each claim costs no query.
    """

    cap_bytes: int = 50 * GB
    safety: float = 0.90
    basis: BandwidthBasis = "decoded"
    window_days: int = 30
    refresh_s: float = 60.0
    clock: Callable[[], float] = time.monotonic
    run_limit_bytes: Optional[int] = None
    scale: float = 1.0
    _used: Optional[int] = field(default=None, init=False, repr=False)
    _synced_at: float = field(default=0.0, init=False, repr=False)
    run_used: int = field(default=0, init=False)

    def add(self, bytes_wire: int, bytes_decoded: int) -> None:
        """Account for one response between database syncs."""
        counted = bytes_wire if self.basis == "wire" else bytes_decoded
        self.run_used += counted
        if self._used is not None:
            self._used += counted

    def exceeded(self, manifest: Manifest) -> Optional[str]:
        """Return a stop message if the window limit or the per-run limit (``fmp run --max-gb``) is reached."""
        if self.run_limit_bytes is not None and self.run_used >= self.run_limit_bytes:
            return (f"this run used {self.run_used / GB:.2f} GB ({self.basis}), reaching its --max-gb limit of "
                    f"{self.run_limit_bytes / GB:.2f} GB")
        if self._used is None or self.clock() - self._synced_at >= self.refresh_s:
            wire, decoded = manifest.bandwidth_used(self.window_days)
            self._used = wire if self.basis == "wire" else decoded
            self._synced_at = self.clock()
        used = self._used
        if used >= self.limit_bytes:
            scaled = f" at {self.scale:g} dashboard bytes per local byte" if self.scale != 1 else ""
            return (f"bandwidth {used / GB:.2f} GB ({self.basis}, {self.window_days}d) reached "
                    f"{self.safety:.0%} of the {self.cap_bytes / GB:.0f} GB cap{scaled}")
        return None

    @property
    def limit_bytes(self) -> float:
        """The window hard stop in locally counted ``basis`` bytes: ``safety × cap / scale``."""
        return self.safety * self.cap_bytes / self.scale


@dataclass
class RunSummary:
    """What one ``Downloader.run`` did."""

    processed: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "done"


class Downloader:
    """Runs a worker pool over the manifest's pending tasks."""

    def __init__(
        self,
        manifest: Manifest,
        client: FmpClient,
        sink: Sink = sha256_sink,
        workers: int = 12,
        guard: Optional[BandwidthGuard] = None,
        reviewer: Optional[Reviewer] = None,
        max_priority: Optional[int] = None,
        min_priority: Optional[int] = None,
        tier: Optional[str] = None,
        run_log: Optional[RunLog] = None,
    ) -> None:
        """
        Args:
            manifest: Task manifest.
            client: Shared FMP client. Its ``on_bytes`` hook is set here, so every response (retries included) is
                written to ``bandwidth_log`` and counted by the guard.
            sink: Called with every final response before the manifest records it.
            workers: Concurrent workers (spec: 8–16).
            guard: Bandwidth hard stop; defaults to 0.90 × 50 GB decoded over 30 days.
            reviewer: Called with every final response before the sink; see ``Reviewer``.
            max_priority: Only claim tasks up to this priority (``fmp plan`` fetches its source lists this way).
            min_priority: Only claim tasks from this priority on (``fmp run`` works one block at a time).
            tier: Only claim tasks of this tier (``fmp run --tier``).
            run_log: Structured run log (spec §13): a record per finished task and a progress summary every
                ``run_log.interval_s`` seconds while the run lasts.
        """
        self._manifest = manifest
        self._client = client
        self._sink = sink
        self._workers = workers
        self._guard = guard or BandwidthGuard()
        self._reviewer = reviewer
        self._max_priority = max_priority
        self._min_priority = min_priority
        self._tier = tier
        self._stop: Optional[str] = None
        self._summary = RunSummary()
        self._run_log = run_log
        client.on_bytes = self._on_bytes

    def _on_bytes(self, bytes_wire: int, bytes_decoded: int) -> None:
        self._manifest.record_bandwidth(bytes_wire, bytes_decoded)
        self._guard.add(bytes_wire, bytes_decoded)

    async def run(self) -> RunSummary:
        """
        Process pending tasks until none are left or a run-level stop occurs.

        Returns:
            Counts and the stop reason (``done``, ``quota``, ``auth``, ``bandwidth``, ``archive``).
        """
        self._manifest.log_event("run_start", f"workers={self._workers} recovered={self._manifest.recovered}")
        ticker = asyncio.create_task(self._ticker()) if self._run_log is not None else None
        try:
            await asyncio.gather(*(self._worker(i) for i in range(self._workers)))
        finally:
            if ticker is not None:
                ticker.cancel()
        self._summary.stop_reason = self._stop or "done"
        self._manifest.log_event("run_stop", self._summary.stop_reason)
        _logger.info("Download run stopped (%s): %d tasks, %s", self._summary.stop_reason, self._summary.processed,
                     self._summary.by_status)
        return self._summary

    async def _ticker(self) -> None:
        """Write a progress summary every ``interval_s`` seconds, also while no task finishes (a stalled run)."""
        assert self._run_log is not None
        while True:
            await asyncio.sleep(self._run_log.interval_s)
            self._run_log.progress(self._manifest, self._guard)

    async def _worker(self, worker_id: int) -> None:
        while self._stop is None:
            reason = self._guard.exceeded(self._manifest)
            if reason is not None:
                self._halt("bandwidth", reason)
                return
            claimed = self._manifest.claim(1, self._max_priority, self._min_priority, self._tier)
            if not claimed:
                return
            await self._process(worker_id, claimed[0])

    async def _process(self, worker_id: int, task: Task) -> None:
        started = time.monotonic()
        try:
            result = await self._client.fetch(task.endpoint, task.request_params)
        except QuotaExhausted as exc:
            self._manifest.release(task)
            self._halt("quota", str(exc))
            return
        except AuthAbort as exc:
            self._manifest.release(task)
            self._halt("auth", str(exc))
            return
        except ValueError as exc:
            self._manifest.fail(task, f"invalid request: {exc}")
            self._finish(worker_id, task, "failed", started, error=f"invalid request: {exc}")
            return

        if self._reviewer is not None:
            result, follow_ups = self._reviewer(task, result)
            if follow_ups:
                self._manifest.add_tasks(follow_ups)
        try:
            stored = self._sink(task, result)
            sha256 = await stored if inspect.isawaitable(stored) else stored
        except ArchiveUnavailable as exc:
            self._manifest.release(task)
            self._halt("archive", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - a sink failure must fail this task, not the whole run
            _logger.exception("Worker %d: sink failed for %s %s", worker_id, task.endpoint, task.symbol)
            self._manifest.fail(task, f"sink error: {exc}")
            self._finish(worker_id, task, "failed", started, result, f"sink error: {exc}")
            return
        self._finish(worker_id, task, self._manifest.complete(task, result, sha256), started, result)

    def _finish(self, worker_id: int, task: Task, status: str, started: float,
                result: Optional[FetchResult] = None, error: str = "") -> None:
        """Count a finished task and log it to the run log."""
        self._summary.processed += 1
        self._summary.by_status[status] = self._summary.by_status.get(status, 0) + 1
        if self._run_log is not None:
            self._run_log.request(worker_id, task, status, time.monotonic() - started, result, error)

    def _halt(self, reason: str, detail: str) -> None:
        if self._stop is None:
            self._stop = reason
            self._manifest.log_event(f"stop_{reason}", detail)
            if self._run_log is not None:
                self._run_log.write("stop", reason=reason, detail=detail)
            _logger.error("Stopping download run: %s", detail)
