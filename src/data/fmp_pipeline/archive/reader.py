"""
Read the raw archive (spec §6.6): stream gzip members, find a torn tail, pick the newest successful response.

An archive file is a sequence of independent gzip members, one JSON line each. Each member is decompressed on its
own, so the reader knows exactly where every member ends. That is how a write cut off by a crash or a vanished
drive is found: the bytes after the last complete member (CRC and length checked by zlib) are the torn tail.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.fmp_pipeline.manifest.manifest import canonical_params
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

CHUNK_SIZE = 1 << 20
SUCCESS_CLASSIFICATIONS = ("ok", "ok_empty")


class ArchiveCorrupt(Exception):
    """An archive file does not end on a complete gzip member."""

    def __init__(self, path: Path, good_end: int, size: int, reason: str) -> None:
        """
        Args:
            path: The file.
            good_end: Byte offset just after the last complete member (the length to truncate to).
            size: File size when read.
            reason: What was wrong with the bytes after ``good_end``.
        """
        super().__init__(f"{path}: {size - good_end} bytes after offset {good_end} are not a complete gzip member "
                         f"({reason})")
        self.path = path
        self.good_end = good_end
        self.size = size


@dataclass(frozen=True)
class Member:
    """One gzip member: its byte range in the file and its decompressed content."""

    start: int
    end: int
    data: bytes


def iter_members(path: Path, chunk_size: int = CHUNK_SIZE) -> Iterator[Member]:
    """
    Stream the gzip members of ``path`` without loading the file.

    Args:
        path: An archive file.
        chunk_size: Read size.

    Yields:
        Every complete member, in file order.

    Raises:
        ArchiveCorrupt: After the last complete member, if bytes remain that are not a complete, valid member.
    """
    start = pos = 0
    decomp = zlib.decompressobj(wbits=31)
    out: list[bytes] = []
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk_size)
            if not buf:
                break
            pos += len(buf)
            while buf:
                try:
                    out.append(decomp.decompress(buf))
                except zlib.error as exc:
                    raise ArchiveCorrupt(path, start, _size(path), str(exc)) from None
                if not decomp.eof:
                    break
                buf = decomp.unused_data
                end = pos - len(buf)
                yield Member(start, end, b"".join(out))
                start, out = end, []
                decomp = zlib.decompressobj(wbits=31)
    if start != pos:
        raise ArchiveCorrupt(path, start, pos, "truncated member")


def _size(path: Path) -> int:
    return path.stat().st_size


def good_length(path: Path) -> int:
    """
    Return the length of the valid prefix of ``path``: its size if intact, else the end of the last complete member.

    Args:
        path: An existing archive file.

    Returns:
        Byte length to keep.
    """
    try:
        for _ in iter_members(path):
            pass
    except ArchiveCorrupt as exc:
        return exc.good_end
    return _size(path)


def iter_records(path: Path, strict: bool = True) -> Iterator[dict[str, Any]]:
    """
    Yield every archived response of one file as a dict.

    Args:
        path: An archive file.
        strict: If True, a torn tail raises ``ArchiveCorrupt``; if False it is logged and skipped (the response was
            never committed to the manifest, so it is re-fetched on the next run).

    Yields:
        Records in file order: ``task_id``, ``request``, ``fetched_at``, ``http_status``, ``classification``,
        ``rows``, ``sha256``, ``body`` (and ``body_text`` for non-JSON bodies).

    Raises:
        ArchiveCorrupt: On a torn tail when ``strict``.
    """
    try:
        for member in iter_members(path):
            for line in member.data.splitlines():
                if line.strip():
                    yield json.loads(line)
    except ArchiveCorrupt:
        if strict:
            raise
        _logger.warning("Skipping torn tail of %s", path)


def request_key(record: dict[str, Any]) -> str:
    """
    Identify the request a record answers: endpoint plus canonical parameters (symbol included).

    Args:
        record: An archive record.

    Returns:
        ``"<endpoint>?<canonical params JSON>"``.
    """
    request = record["request"]
    return f"{request['endpoint']}?{canonical_params(request['params'])}"


def latest_successful(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Keep the newest successful response per request (spec §6.6: re-fetches append, the builder takes the newest).

    Args:
        records: Records from one or more files, in file order.

    Returns:
        ``{request_key: record}``; on equal ``fetched_at`` the later record wins.
    """
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("classification") not in SUCCESS_CLASSIFICATIONS:
            continue
        key = request_key(record)
        current = latest.get(key)
        if current is None or record["fetched_at"] >= current["fetched_at"]:
            latest[key] = record
    return latest
