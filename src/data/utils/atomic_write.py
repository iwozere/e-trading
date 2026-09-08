"""
Atomic CSV Cache Writes

Several P15/downloader caches are incrementally maintained files (one row or one
day appended per run) that a watermark function later treats as "already
cached" purely by checking existence — it never validates content. If a writer
is killed mid-``to_csv`` (e.g. the scheduler's hard ``process.kill()`` on a
timed-out job), a direct write-in-place leaves a truncated/corrupt gzip file
that then reads as permanently cached and is never retried.

``atomic_to_csv`` closes that gap: it writes to a temp file in the same
directory and ``os.replace()``s it into place, so any reader (or a killed
writer) only ever observes the previous complete version or the new complete
version — never a partial one.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Union

import pandas as pd

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def atomic_write_bytes(data: bytes, path: Union[str, Path]) -> None:
    """
    Write raw bytes to ``path`` atomically via a same-directory temp file + rename.

    Args:
        data: Bytes to write (already compressed/encoded as needed — this does
            no encoding of its own).
        path: Final destination path (parent directory is created if missing).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, dest)
    except BaseException:
        # BaseException (not just Exception) so Ctrl-C / SystemExit during the
        # write still clean up the temp file instead of littering the cache dir.
        # This is best-effort tidiness only — it cannot run at all under a real
        # SIGKILL/TerminateProcess, but that case is already safe regardless:
        # `dest` is only ever touched by the atomic os.replace() below, so an
        # untimely hard kill just leaves an orphaned .tmp file, never a partial
        # `dest`.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            _logger.warning("atomic_write_bytes: failed to clean up temp file %s", tmp_path)
        raise


def atomic_to_csv(df: pd.DataFrame, path: Union[str, Path], **to_csv_kwargs: Any) -> None:
    """
    Write ``df`` to ``path`` atomically via a same-directory temp file + rename.

    Args:
        df: DataFrame to write.
        path: Final destination path (parent directory is created if missing).
        **to_csv_kwargs: Forwarded verbatim to ``DataFrame.to_csv`` (e.g.
            ``compression="gzip"``, ``index=False``, ``sep="\\t"``). Pass
            ``compression`` explicitly rather than relying on extension
            inference — the temp file's suffix does not match the final name.

    Raises:
        Exception: Whatever ``DataFrame.to_csv`` raises; the temp file is
            cleaned up before re-raising so failed writes don't leak files.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_csv(tmp_path, **to_csv_kwargs)
        os.replace(tmp_path, dest)
    except BaseException:
        # See atomic_write_bytes for why this is BaseException, not Exception.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            _logger.warning("atomic_to_csv: failed to clean up temp file %s", tmp_path)
        raise
