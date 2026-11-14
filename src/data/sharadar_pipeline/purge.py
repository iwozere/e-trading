"""
Delete the Sharadar data root and prove it is gone.

The personal licence requires deleting every copy within 30 days of the subscription ending. Run this on every
machine that holds a copy (dev machine, Pi) and remove the folder from backups by hand. The report is written
outside the data root and holds file names and sizes only, no data.
"""

from __future__ import annotations

import json
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.data.sharadar_pipeline import config
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class PurgeRefused(RuntimeError):
    """The target does not look like a Sharadar data root, or the purge was not confirmed."""


def inventory(data_root: Path) -> list[dict[str, Any]]:
    """List every file under ``data_root`` with its size."""
    if not data_root.exists():
        return []
    return [{"path": str(path.relative_to(data_root)).replace("\\", "/"), "size": path.stat().st_size}
            for path in sorted(data_root.rglob("*")) if path.is_file()]


def purge(data_root: Path, confirm: bool = False, report_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Delete ``data_root`` and everything in it.

    Args:
        data_root: Sharadar data root. Its folder name must be ``sharadar-data`` — a guard against deleting
            anything else.
        confirm: Must be True; otherwise nothing is deleted and the inventory is returned as a dry run.
        report_path: Where to write the report (must lie outside ``data_root``).

    Returns:
        Report with host, time, file count, bytes, the file list and ``deleted``.

    Raises:
        PurgeRefused: Wrong folder name, or a report path inside the data root.
        RuntimeError: Files remain after deletion.
    """
    if data_root.name != config.DATA_DIR_NAME:
        raise PurgeRefused(f"{data_root} is not named {config.DATA_DIR_NAME}; refusing to delete it")
    if report_path is not None and data_root.resolve() in report_path.resolve().parents:
        raise PurgeRefused("the purge report must be written outside the data root")
    files = inventory(data_root)
    report: dict[str, Any] = {"host": socket.gethostname(), "data_root": str(data_root),
                              "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "files": len(files), "bytes": sum(f["size"] for f in files), "deleted": False, "file_list": files}
    if not confirm:
        return report
    if data_root.exists():
        shutil.rmtree(data_root)
    if data_root.exists():
        raise RuntimeError(f"{data_root} still exists after the purge: {len(inventory(data_root))} files remain")
    report["deleted"] = True
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _logger.info("Purged %s: %d files, %.1f MB", data_root, report["files"], report["bytes"] / 1e6)
    return report
