"""
Pack a directory tree into a few large tar files, verify them anywhere, and unpack them safely (spec §13, §14).

A network share copies a few large files fast and hundreds of thousands of small ones slowly, so ``fmp backup``
and ``fmp publish`` write one **uncompressed** tar per top-level entry of a tree (the archive's gzip members and
the vintage's zstd Parquet are already compressed). ``MANIFEST.json`` lists every tar with its sha256, size and
member count. ``verify_packed`` re-hashes the tars and then hashes every member's content against the tree's
``sha256sum``-format checksum file, so the copy on the destination volume is checked file by file without
unpacking it.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from src.data.fmp_pipeline.builders.parquet_io import HASH_CHUNK, file_sha256
from src.data.fmp_pipeline.fsutil import atomic_write_bytes, replace_with_retry

MANIFEST = "MANIFEST.json"
READY = "READY"
ROOT_GROUP = "_root"


@dataclass(frozen=True)
class PackedFile:
    """One tar of a pack."""

    name: str
    sha256: str
    bytes: int
    members: int


@dataclass
class Pack:
    """What ``pack_tree`` wrote."""

    tree: str
    tars: list[PackedFile] = field(default_factory=list)
    files: int = 0
    bytes: int = 0


def _groups(root: Path) -> dict[str, list[Path]]:
    """Files of ``root`` grouped by their top-level entry (files directly in ``root`` form one group)."""
    groups: dict[str, list[Path]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        groups.setdefault(rel.parts[0] if len(rel.parts) > 1 else ROOT_GROUP, []).append(path)
    return groups


def pack_tree(root: Path, out_dir: Path, prefix: str) -> Pack:
    """
    Write ``<prefix>-<group>.tar`` files for ``root`` into ``out_dir`` (member names relative to ``root``).

    Each tar is written under a temporary name and renamed when complete, so an interrupted copy never leaves a
    tar that looks finished.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pack = Pack(prefix)
    for group, files in _groups(root).items():
        name = f"{prefix}-{group}.tar"
        tmp = out_dir / f".{name}.partial"
        with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tar:
            for path in files:
                tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
                pack.files += 1
                pack.bytes += path.stat().st_size
        with open(tmp, "rb+") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        replace_with_retry(tmp, out_dir / name)
        pack.tars.append(PackedFile(name, file_sha256(out_dir / name), (out_dir / name).stat().st_size, len(files)))
    return pack


def write_manifest(out_dir: Path, dump_id: str, packs: list[Pack], extra: Optional[dict[str, Any]] = None) -> Path:
    """``MANIFEST.json``: the packs and their tars."""
    payload = {"dump_id": dump_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "packs": [asdict(p) for p in packs], **(extra or {})}
    path = out_dir / MANIFEST
    atomic_write_bytes(path, json.dumps(payload, indent=1).encode("utf-8"))
    return path


def read_manifest(directory: Path) -> dict[str, Any]:
    """Read ``MANIFEST.json`` (raises ``FileNotFoundError`` when absent)."""
    return json.loads((directory / MANIFEST).read_text(encoding="utf-8"))


def _listed(checksum_file: Path) -> dict[str, str]:
    listed = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, _, rel = line.partition("  ")
        if rel:
            listed[rel] = digest
    return listed


def verify_packed(directory: Path, tree: str, checksum_file: Path) -> list[str]:
    """
    Check one pack in ``directory``: tar hashes against ``MANIFEST.json``, every member against ``checksum_file``.

    Args:
        directory: Where the pack was written (the backup or publish folder).
        tree: The pack's ``tree`` name in the manifest (``raw``, ``vintage``, ``increments``).
        checksum_file: ``sha256sum``-format list of the original tree.

    Returns:
        Error messages; empty means the copy is identical to the original, file by file.
    """
    try:
        manifest = read_manifest(directory)
    except FileNotFoundError:
        return [f"{directory / MANIFEST} missing"]
    pack = next((p for p in manifest["packs"] if p["tree"] == tree), None)
    if pack is None:
        return [f"{MANIFEST} has no {tree} pack"]
    if not checksum_file.exists():
        return [f"{checksum_file} missing"]
    listed = _listed(checksum_file)
    seen: set[str] = set()
    errors = []
    for entry in pack["tars"]:
        path = directory / entry["name"]
        if not path.exists():
            errors.append(f"missing: {entry['name']}")
            continue
        if file_sha256(path) != entry["sha256"]:
            errors.append(f"tar checksum mismatch: {entry['name']}")
            continue
        with tarfile.open(path, "r") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                handle = tar.extractfile(member)
                assert handle is not None
                digest = hashlib.sha256()
                while chunk := handle.read(HASH_CHUNK):
                    digest.update(chunk)
                seen.add(member.name)
                if listed.get(member.name) != digest.hexdigest():
                    errors.append(f"member mismatch: {member.name}" if member.name in listed
                                  else f"not in {checksum_file.name}: {member.name}")
    errors += [f"missing from the pack: {rel}" for rel in sorted(set(listed) - seen)]
    return errors


def _safe(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def unpack(directory: Path, tree: str, target: Path) -> int:
    """
    Extract one pack into ``target`` (created); refuses members with absolute paths or ``..``.

    Returns:
        Files extracted.
    """
    pack = next(p for p in read_manifest(directory)["packs"] if p["tree"] == tree)
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for entry in pack["tars"]:
        with tarfile.open(directory / entry["name"], "r") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            bad = [m.name for m in members if not _safe(m.name)]
            if bad:
                raise ValueError(f"{entry['name']}: unsafe member paths {bad[:3]}")
            tar.extractall(target, members=members, filter="data")
            count += len(members)
    return count
