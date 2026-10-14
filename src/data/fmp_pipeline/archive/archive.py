"""
Append-only raw archive of FMP responses (spec §6.6, plan WP-5); the ``Downloader`` sink.

Layout: ``<raw_root>/<dump_id>/<endpoint path>/<encoded symbol>.jsonl.gz``. Every response becomes one JSON line,
compressed as its **own gzip member** and appended to the file, so the file stays valid gzip (``gzip -dc`` reads
all lines) while many windows of one symbol share a file instead of making hundreds of thousands of tiny files.

Line: ``{"task_id", "request": {"endpoint", "params"}, "fetched_at", "http_status", "classification", "rows",
"sha256", "body"}``. ``body`` is the parsed JSON; a non-JSON body (FMP's plain-text 402s) is ``null`` with the text
in ``body_text``. ``sha256`` is of the response bytes as received and is also what the manifest stores, so the two
can be cross-checked. The key is never part of ``params`` (it travels in a header) and any ``apikey`` parameter is
dropped defensively.

Durability: each member is written with one ``write`` call, flushed and fsynced before ``append`` returns, and the
runner marks a task done only after that (D28). A crash or a vanished drive can leave a partial member at the end
of one file. The first append to a file in each process therefore checks that the file ends on a complete member
and truncates a torn tail first (D33). Every write error, a missing archive root and low free space raise
``ArchiveUnavailable``, which pauses the run instead of failing tasks one by one.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from src.data.fmp_pipeline.archive.reader import good_length
from src.data.fmp_pipeline.archive.symbol_path import encode_symbol, endpoint_dir
from src.data.fmp_pipeline.client.fmp_client import FetchResult
from src.data.fmp_pipeline.errors import ArchiveUnavailable
from src.data.fmp_pipeline.manifest.manifest import Task
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

GB = 1024**3
SUFFIX = ".jsonl.gz"
_DUMP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def build_record(task: Task, result: FetchResult) -> dict[str, Any]:
    """
    Build the archive line for one response.

    Args:
        task: The manifest task (its ``request_params`` are what was sent).
        result: The final response.

    Returns:
        The record as a JSON-serializable dict.
    """
    params = {k: v for k, v in task.request_params.items() if k.lower() != "apikey"}
    record: dict[str, Any] = {
        "task_id": task.task_id,
        "request": {"endpoint": task.endpoint, "params": params},
        "fetched_at": result.fetched_at,
        "http_status": result.http_status,
        "classification": result.classification,
        "rows": result.rows,
        "sha256": hashlib.sha256(result.body).hexdigest(),
        "body": None,
    }
    if result.body:
        try:
            record["body"] = json.loads(result.body)
        except ValueError:
            record["body_text"] = result.body.decode("utf-8", errors="replace")
    return record


class RawArchive:
    """Writer for one dump's raw archive. Single writer per dump; not thread-safe (the runner calls it inline)."""

    def __init__(
        self,
        raw_root: Path,
        dump_id: str,
        compresslevel: int = 5,
        min_free_bytes: int = 2 * GB,
        space_check_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            raw_root: ``<data_root>/raw``; created if missing (later appends require it to still exist).
            dump_id: Dump directory name (``2026-10``).
            compresslevel: gzip level; spec suggests 4–6.
            min_free_bytes: Pause the run when the volume has less free space than this.
            space_check_s: Interval between free-space checks.
            clock: Monotonic clock (injectable for tests).

        Raises:
            ValueError: If ``dump_id`` is not a safe directory name.
        """
        if not _DUMP_ID_RE.match(dump_id):
            raise ValueError(f"dump_id {dump_id!r} is not a safe directory name")
        self.raw_root = raw_root
        self.dump_dir = raw_root / dump_id
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self._level = compresslevel
        self._min_free = min_free_bytes
        self._space_check_s = space_check_s
        self._clock = clock
        self._space_checked_at: Optional[float] = None
        self._known_len: dict[Path, int] = {}
        self.appended = 0
        self.bytes_written = 0
        self.repaired: list[Path] = []

    def path_for(self, endpoint: str, symbol: Optional[str]) -> Path:
        """
        Return the archive file for an endpoint and symbol.

        Args:
            endpoint: FMP endpoint path (``historical-price-eod/non-split-adjusted``).
            symbol: Ticker, or ``None`` for endpoints without one.

        Returns:
            ``<dump_dir>/<endpoint>/<encoded symbol>.jsonl.gz``.
        """
        return self.dump_dir / endpoint_dir(endpoint) / (encode_symbol(symbol) + SUFFIX)

    def __call__(self, task: Task, result: FetchResult) -> str:
        """Sink interface for ``Downloader``: same as ``append``."""
        return self.append(task, result)

    def append(self, task: Task, result: FetchResult) -> str:
        """
        Append one response as a new gzip member and make it durable.

        Args:
            task: The manifest task.
            result: The final response.

        Returns:
            sha256 of the response body (stored in the manifest).

        Raises:
            ArchiveUnavailable: Archive root missing, free space below ``min_free_bytes``, or any write error. The
                file is left ending on a complete member whenever the volume still allows a truncate.
        """
        self._check_available()
        path = self.path_for(task.endpoint, task.symbol)
        record = build_record(task, result)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        member = gzip.compress(line, compresslevel=self._level, mtime=0)
        start: Optional[int] = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            start = self._prepare(path)
            with path.open("ab") as fh:
                fh.write(member)
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            self._known_len.pop(path, None)
            if start is not None:
                _truncate_quietly(path, start)
            raise ArchiveUnavailable(f"cannot write {path}: {exc}") from exc
        self._known_len[path] = start + len(member)
        self.appended += 1
        self.bytes_written += len(member)
        return str(record["sha256"])

    def _check_available(self) -> None:
        if not self.raw_root.is_dir():
            raise ArchiveUnavailable(f"archive root {self.raw_root} is not available")
        now = self._clock()
        if self._space_checked_at is not None and now - self._space_checked_at < self._space_check_s:
            return
        try:
            free = shutil.disk_usage(self.raw_root).free
        except OSError as exc:
            raise ArchiveUnavailable(f"cannot read free space of {self.raw_root}: {exc}") from exc
        self._space_checked_at = now
        if free < self._min_free:
            raise ArchiveUnavailable(f"only {free / GB:.2f} GB free on {self.raw_root} "
                                     f"(minimum {self._min_free / GB:.2f} GB)")

    def _prepare(self, path: Path) -> int:
        """Return the length to append at, truncating a torn tail the first time a file is seen."""
        size = path.stat().st_size if path.exists() else 0
        if self._known_len.get(path) == size:
            return size
        good = good_length(path) if size else 0
        if good < size:
            _logger.warning("Archive %s ends in a torn write: truncating %d bytes to %d", path, size - good, good)
            with path.open("r+b") as fh:
                fh.truncate(good)
                fh.flush()
                os.fsync(fh.fileno())
            self.repaired.append(path)
        self._known_len[path] = good
        return good


def _truncate_quietly(path: Path, length: int) -> None:
    """Best-effort rollback of a failed append; the next run's tail check covers the case where it fails too."""
    try:
        with path.open("r+b") as fh:
            fh.truncate(length)
    except OSError:
        _logger.warning("Could not roll back %s to %d bytes; it is repaired on the next append", path, length)
