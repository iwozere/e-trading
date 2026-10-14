"""
Atomic file writes that survive Windows antivirus and indexers (spec §13, plan §9.3), and the free-disk check
(spec §6.8).

On Windows, ``os.replace`` fails with ``PermissionError`` while another process (Defender, the search indexer,
a backup tool) briefly holds the target open. The replace is retried with backoff instead of failing the run.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from src.data.fmp_pipeline.errors import InsufficientDiskSpace
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


def ensure_free_space(path: Path, need_bytes: int, margin: float = 0.20) -> int:
    """
    Fail early if the volume holding ``path`` has less than ``need_bytes × (1 + margin)`` free (spec §6.8).

    Args:
        path: Any existing path on the volume (the nearest existing parent is used if ``path`` does not exist).
        need_bytes: Bytes the phase expects to write.
        margin: Safety margin on top of ``need_bytes`` (spec: 20%).

    Returns:
        The free bytes on the volume.

    Raises:
        InsufficientDiskSpace: If free space is below the requirement.
    """
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    required = int(need_bytes * (1 + margin))
    if free < required:
        raise InsufficientDiskSpace(
            f"{probe}: {free / 1024**3:.1f} GB free, need {required / 1024**3:.1f} GB "
            f"({need_bytes / 1024**3:.1f} GB + {margin:.0%} margin)")
    return free
