"""
Manifest of the Sharadar data root: which files were downloaded, when, and the deletion deadline.

Stored as ``<data_root>/manifest.json`` and deleted with the data root. The purge report, written outside the
data root, is what remains as proof of deletion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.data.sharadar_pipeline.config import MANIFEST_FILE, RETENTION_DAYS


@dataclass
class FileEntry:
    """One raw file in the data root."""

    table: str
    name: str
    size: int
    sha256: str
    downloaded_at: str
    rows: Optional[int] = None


@dataclass
class Manifest:
    """Download state of one data root."""

    mode: str = "sample"
    subscription_start: Optional[str] = None
    subscription_end: Optional[str] = None
    files: dict[str, FileEntry] = field(default_factory=dict)

    @classmethod
    def load(cls, data_root: Path) -> "Manifest":
        """Read the manifest of ``data_root``; a missing file gives an empty manifest."""
        path = data_root / MANIFEST_FILE
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = {name: FileEntry(**entry) for name, entry in payload.pop("files", {}).items()}
        return cls(files=files, **payload)

    def save(self, data_root: Path) -> None:
        """Write the manifest atomically."""
        data_root.mkdir(parents=True, exist_ok=True)
        path = data_root / MANIFEST_FILE
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(path)

    def record(self, table: str, name: str, size: int, sha256: str, rows: Optional[int] = None) -> None:
        """Add or replace the entry of one table."""
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.files[table] = FileEntry(table, name, size, sha256, stamp, rows)

    def deletion_deadline(self) -> Optional[date]:
        """
        Return the last day the data may exist.

        ``subscription_end`` plus the retention period; if only the start is known, a one-month subscription is
        assumed. None in sample mode or when no date is recorded.
        """
        if self.mode != "paid":
            return None
        if self.subscription_end:
            return date.fromisoformat(self.subscription_end) + timedelta(days=RETENTION_DAYS)
        if self.subscription_start:
            return date.fromisoformat(self.subscription_start) + timedelta(days=30 + RETENTION_DAYS)
        return None

    def days_left(self, today: Optional[date] = None) -> Optional[int]:
        """Days until the deletion deadline (negative when overdue), or None."""
        deadline = self.deletion_deadline()
        if deadline is None:
            return None
        return (deadline - (today or datetime.now(timezone.utc).date())).days
