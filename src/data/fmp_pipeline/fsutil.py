"""
Atomic file writes that survive Windows antivirus and indexers (spec §13, plan §9.3).

On Windows, ``os.replace`` fails with ``PermissionError`` while another process (Defender, the search indexer,
a backup tool) briefly holds the target open. The replace is retried with backoff instead of failing the run.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def replace_with_retry(
    src: Path,
    dst: Path,
    attempts: int = 8,
    first_delay_s: float = 0.05,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """
    ``os.replace(src, dst)``, retrying on ``PermissionError`` with exponential backoff.

    Args:
        src: Source file (usually a temp file next to ``dst``).
        dst: Destination; replaced atomically.
        attempts: Maximum attempts.
        first_delay_s: Delay before the second attempt; doubles each time (0.05 s … ~6 s over 8 attempts).
        sleep: Sleep function (injectable for tests).

    Raises:
        PermissionError: If the last attempt still fails.
    """
    delay = first_delay_s
    for attempt in range(1, attempts + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            _logger.warning("Replace of %s blocked (attempt %d/%d), retrying in %.2fs", dst, attempt, attempts, delay)
            sleep(delay)
            delay *= 2


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """
    Write ``data`` to ``path`` atomically: temp file in the same directory, fsync, then replace.

    A crash leaves either the old file or the new one, never a partial file.

    Args:
        path: Target path; its directory is created if missing.
        data: Content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
