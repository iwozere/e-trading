"""
``fmp backup``, ``fmp publish`` and ``fmp sync`` (spec §6.1, §10.4, §14; owner decisions Q3', Q3'').

- **backup** (writer → backup volume, e.g. the NAS or a USB disk): packs ``raw/<dump_id>`` and
  ``vintages/<dump_id>`` into ``<dest>/<dump_id>/`` with the checksum files, then verifies the copy there member by
  member. ``fmp verify --final --backup <dest>`` accepts this layout.
- **publish** (writer → hand-off folder, Q3'': directly on ``R:``, the Pi's SSD): packs the vintage (and with
  ``--increments`` the increment layer) into ``<dest>/publish/<dump_id>/``, verifies it, and writes ``READY`` last,
  so a reader never takes a half-written publication.
- **sync** (reader, e.g. the Pi): takes the newest ``READY`` publication (or ``--dump-id``), verifies it, unpacks
  into a temporary directory on the local disk, re-checks every file, swaps it in and then switches
  ``current.txt`` — local-OS atomicity, never network-share atomicity (Q3''). With ``--increments`` it replaces the
  local increment layer the same way; a reader never writes increments otherwise (§10.4).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.data.fmp_pipeline.builders.builder import CURRENT_FILE
from src.data.fmp_pipeline.fsutil import atomic_write_bytes, swap_directory
from src.data.fmp_pipeline.increments.store import INCREMENTS_DIR, IncrementStore
from src.data.fmp_pipeline.transfer.packing import READY, pack_tree, unpack, verify_packed, write_manifest
from src.data.fmp_pipeline.validators.checksums import (
    ARCHIVE_FILE,
    SUMMARY_FILE,
    VINTAGE_FILE,
    checksum_dir,
    verify_checksums,
    write_checksums,
    write_summary,
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

INCREMENTS_FILE = "INCREMENTS.sha256"
PUBLISH_DIR = "publish"


@dataclass
class TransferResult:
    """Where a transfer went and what went wrong (empty ``errors`` = verified)."""

    destination: Path
    dump_id: str
    errors: list[str] = field(default_factory=list)
    files: int = 0


def _require(path: Path, what: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"no {what} at {path}")


def backup(data_root: Path, dump_id: str, dest: Path) -> TransferResult:
    """
    Pack the dump's raw archive and vintage into ``<dest>/<dump_id>/`` and verify the copy there.

    Raises:
        FileNotFoundError: The dump has no archive or no vintage.
    """
    raw, vintage = data_root / "raw" / dump_id, data_root / "vintages" / dump_id
    _require(raw, "raw archive")
    _require(vintage, "vintage")
    sums = checksum_dir(data_root, dump_id)
    write_summary(sums, {"archive": write_checksums(raw, sums / ARCHIVE_FILE),
                         "vintage": write_checksums(vintage, sums / VINTAGE_FILE)})
    out = dest / dump_id
    packs = [pack_tree(raw, out, "raw"), pack_tree(vintage, out, "vintage")]
    (out / "checksums").mkdir(parents=True, exist_ok=True)
    for name in (ARCHIVE_FILE, VINTAGE_FILE, SUMMARY_FILE):
        shutil.copy2(sums / name, out / "checksums" / name)
    write_manifest(out, dump_id, packs)
    result = TransferResult(out, dump_id, files=sum(p.files for p in packs))
    result.errors = (verify_packed(out, "raw", out / "checksums" / ARCHIVE_FILE)
                     + verify_packed(out, "vintage", out / "checksums" / VINTAGE_FILE))
    _logger.info("Backup of %s to %s: %d files in %d tars, %s", dump_id, out, result.files,
                 sum(len(p.tars) for p in packs), "verified" if not result.errors else f"{len(result.errors)} errors")
    return result


def publish(data_root: Path, dump_id: str, dest: Path, increments: bool = False) -> TransferResult:
    """
    Stage the vintage (and optionally the increments) in ``<dest>/publish/<dump_id>/``; ``READY`` only if verified.

    Raises:
        FileNotFoundError: The dump has no vintage (or no increment layer when ``increments``).
    """
    vintage = data_root / "vintages" / dump_id
    _require(vintage, "vintage")
    out = dest / PUBLISH_DIR / dump_id
    (out / READY).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    write_checksums(vintage, out / VINTAGE_FILE)
    packs = [pack_tree(vintage, out, "vintage")]
    extra: dict[str, object] = {}
    if increments:
        store = IncrementStore(data_root)
        _require(store.root, "increment layer")
        write_checksums(store.root, out / INCREMENTS_FILE)
        packs.append(pack_tree(store.root, out, "increments"))
        extra["increment_seq"] = store.meta().get("increment_seq", 0)
    write_manifest(out, dump_id, packs, extra)
    result = TransferResult(out, dump_id, files=sum(p.files for p in packs))
    result.errors = verify_packed(out, "vintage", out / VINTAGE_FILE)
    if increments:
        result.errors += verify_packed(out, "increments", out / INCREMENTS_FILE)
    if not result.errors:
        atomic_write_bytes(out / READY, datetime.now(timezone.utc).isoformat(timespec="seconds").encode("utf-8"))
    _logger.info("Publish of %s to %s: %s", dump_id, out, "READY" if not result.errors else f"{result.errors[:3]}")
    return result


def ready_publications(source: Path) -> list[str]:
    """Dump ids under ``<source>/publish/`` that carry ``READY``, oldest first."""
    folder = source / PUBLISH_DIR
    return sorted(p.name for p in folder.iterdir() if (p / READY).exists()) if folder.is_dir() else []


def sync(data_root: Path, source: Path, dump_id: Optional[str] = None, increments: bool = False,
         switch: bool = True) -> TransferResult:
    """
    Install a publication on this machine: verify, unpack locally, re-check, swap in, switch ``current.txt``.

    Raises:
        FileNotFoundError: No ``READY`` publication (or not the requested one).
    """
    available = ready_publications(source)
    chosen = dump_id or (available[-1] if available else None)
    if chosen is None or chosen not in available:
        raise FileNotFoundError(f"no READY publication{' ' + dump_id if dump_id else ''} under {source / PUBLISH_DIR}")
    folder = source / PUBLISH_DIR / chosen
    result = TransferResult(data_root / "vintages" / chosen, chosen)
    result.errors = verify_packed(folder, "vintage", folder / VINTAGE_FILE)
    if increments:
        result.errors += verify_packed(folder, "increments", folder / INCREMENTS_FILE)
    if result.errors:
        return result
    staging = data_root / "vintages" / f".{chosen}.syncing"
    if staging.exists():
        shutil.rmtree(staging)
    result.files = unpack(folder, "vintage", staging)
    result.errors = verify_checksums(staging, folder / VINTAGE_FILE)
    inc_staging = data_root / f".{INCREMENTS_DIR}.syncing"
    if increments and not result.errors:
        if inc_staging.exists():
            shutil.rmtree(inc_staging)
        result.files += unpack(folder, "increments", inc_staging)
        result.errors = verify_checksums(inc_staging, folder / INCREMENTS_FILE)
    if result.errors:
        _logger.error("Sync of %s rejected after unpacking: %s", chosen, result.errors[:3])
        return result
    swap_directory(staging, data_root / "vintages" / chosen)
    if increments:
        swap_directory(inc_staging, data_root / INCREMENTS_DIR)
    if switch:
        atomic_write_bytes(data_root / CURRENT_FILE, f"{chosen}\n".encode("utf-8"))
    _logger.info("Synced %s from %s (%d files)%s", chosen, folder, result.files,
                 "; current.txt switched" if switch else "")
    return result
