"""
Archive-wide checksum files (spec §14 item 3) in ``sha256sum -c`` format, and the gzip decompression test.

``<data_root>/checksums/<dump_id>/`` holds ``ARCHIVE.sha256`` (every file under ``raw/<dump_id>/``, paths relative
to it), ``VINTAGE.sha256`` (every file under ``vintages/<dump_id>/``) and ``checksums.json`` (file count and total
bytes of each). A backup that mirrors the ``data_root`` layout is checked against the same files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.data.fmp_pipeline.archive.archive import SUFFIX
from src.data.fmp_pipeline.archive.reader import ArchiveCorrupt, iter_members
from src.data.fmp_pipeline.builders.parquet_io import file_sha256
from src.data.fmp_pipeline.fsutil import atomic_write_bytes

ARCHIVE_FILE = "ARCHIVE.sha256"
VINTAGE_FILE = "VINTAGE.sha256"
SUMMARY_FILE = "checksums.json"


@dataclass(frozen=True)
class ChecksumSummary:
    """What one checksum file covers."""

    files: int
    bytes: int
    checksum_file_sha256: str


def checksum_dir(data_root: Path, dump_id: str) -> Path:
    """``<data_root>/checksums/<dump_id>``."""
    return data_root / "checksums" / dump_id


def list_files(root: Path) -> list[str]:
    """Every file under ``root`` as a POSIX path relative to it, sorted."""
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def write_checksums(root: Path, out: Path) -> ChecksumSummary:
    """
    Write ``<sha256>  <relative path>`` for every file under ``root`` (``sha256sum -c`` compatible).

    Returns:
        File count, total bytes and the sha256 of the written checksum file itself.
    """
    lines, total = [], 0
    for rel in list_files(root):
        path = root / rel
        total += path.stat().st_size
        lines.append(f"{file_sha256(path)}  {rel}\n")
    data = "".join(lines).encode("utf-8")
    atomic_write_bytes(out, data)
    return ChecksumSummary(len(lines), total, hashlib.sha256(data).hexdigest())


def write_summary(directory: Path, summaries: dict[str, ChecksumSummary]) -> None:
    """``checksums.json``: count, bytes and checksum-file hash per covered tree."""
    atomic_write_bytes(directory / SUMMARY_FILE,
                       json.dumps({k: asdict(v) for k, v in summaries.items()}, indent=1).encode("utf-8"))


def verify_checksums(root: Path, checksum_file: Path, expected: ChecksumSummary | None = None) -> list[str]:
    """
    Check every listed file under ``root`` (and that nothing else is there).

    Args:
        root: The tree to check (the original, or its copy on the backup volume).
        checksum_file: An ``ARCHIVE.sha256``/``VINTAGE.sha256``.
        expected: Count and bytes to compare as well.

    Returns:
        Error messages; empty means identical.
    """
    if not checksum_file.exists():
        return [f"{checksum_file} missing"]
    if not root.is_dir():
        return [f"{root} missing"]
    listed: dict[str, str] = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, _, rel = line.partition("  ")
        listed[rel] = digest
    errors = []
    present = set(list_files(root))
    for rel in sorted(set(listed) - present):
        errors.append(f"missing: {rel}")
    for rel in sorted(present - set(listed)):
        errors.append(f"not in {checksum_file.name}: {rel}")
    total = 0
    for rel in sorted(set(listed) & present):
        path = root / rel
        total += path.stat().st_size
        if file_sha256(path) != listed[rel]:
            errors.append(f"checksum mismatch: {rel}")
    if expected is not None and (len(present) != expected.files or total != expected.bytes):
        errors.append(f"{root}: {len(present)} files / {total} bytes, expected {expected.files} / {expected.bytes}")
    return errors


def read_summary(directory: Path) -> dict[str, ChecksumSummary]:
    """Read ``checksums.json``."""
    path = directory / SUMMARY_FILE
    if not path.exists():
        return {}
    return {k: ChecksumSummary(**v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def gzip_test(dump_dir: Path) -> tuple[int, int, list[str]]:
    """
    Decompress every gzip member of every archive file end to end (spec §14: a copy can carry a corrupt file).

    Returns:
        ``(files, members, errors)``.
    """
    files = members = 0
    errors = []
    for path in sorted(dump_dir.rglob("*" + SUFFIX)):
        files += 1
        try:
            for _ in iter_members(path):
                members += 1
        except ArchiveCorrupt as exc:
            errors.append(str(exc))
    return files, members, errors
