"""
Atomic file writes that survive Windows antivirus and indexers (spec §13, plan §9.3), and the free-disk check
(spec §6.8), and the per-dump writer lock.

On Windows, ``os.replace`` fails with ``PermissionError`` while another process (Defender, the search indexer,
a backup tool) briefly holds the target open. The replace is retried with backoff instead of failing the run.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from src.data.fmp_pipeline.errors import DumpLocked, InsufficientDiskSpace
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


def swap_directory(new: Path, final: Path) -> None:
    """
    Replace directory ``final`` with ``new`` by renames: the old one is moved aside, ``new`` renamed in, the old
    one removed. A reader sees the old or the new tree, never a mix (vintage activation, ``fmp sync``).

    Args:
        new: The complete replacement (on the same volume as ``final``).
        final: The directory to replace (need not exist).
    """
    old = final.with_name(f".{final.name}.old")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        replace_with_retry(final, old)
    replace_with_retry(new, final)
    if old.exists():
        shutil.rmtree(old)


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


def existing_parent(path: Path) -> Path:
    """Return ``path`` or its nearest existing parent (a drive root at worst)."""
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def free_bytes(path: Path) -> Optional[int]:
    """Return the free bytes on the volume holding ``path`` (or its nearest existing parent); ``None`` if unknown."""
    try:
        return shutil.disk_usage(existing_parent(path)).free
    except OSError:
        return None


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
    probe = existing_parent(path)
    free = shutil.disk_usage(probe).free
    required = int(need_bytes * (1 + margin))
    if free < required:
        raise InsufficientDiskSpace(
            f"{probe}: {free / 1024**3:.1f} GB free, need {required / 1024**3:.1f} GB "
            f"({need_bytes / 1024**3:.1f} GB + {margin:.0%} margin)")
    return free


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """
    Hold an exclusive, non-blocking OS lock on ``path`` for the duration of the ``with`` block.

    The lock is a byte-range lock (``msvcrt.locking`` on Windows, ``flock`` elsewhere) on an open file, so the OS
    releases it when the process exits or is killed: there is no stale lock file to clean up after a crash.

    Args:
        path: Lock file; created (with parents) if missing. Its content is irrelevant.

    Raises:
        DumpLocked: Another process holds the lock.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as fh:
        try:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DumpLocked(f"{path} is held by another fmp process") from exc
        try:
            yield
        finally:
            if sys.platform == "win32":
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
