"""
Read what the builder and the planner need back out of the raw archive (spec §6.6, §7.4).

Two shapes of endpoint exist in the archive: **global** ones (one file, ``_.jsonl.gz``, possibly several pages)
and **per-symbol** ones (one file per symbol, possibly several date windows or periods). In both, every request
key may have been fetched more than once; only the newest successful line per request key counts.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.data.fmp_pipeline.archive.archive import SUFFIX
from src.data.fmp_pipeline.archive.reader import iter_records, latest_successful, request_key
from src.data.fmp_pipeline.archive.symbol_path import decode_symbol, endpoint_dir
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def body_rows(body: Any) -> list[dict[str, Any]]:
    """Return the dict rows of a response body: a list of dicts, or a single dict as a one-row list."""
    if isinstance(body, dict):
        body = [body]
    return [r for r in body if isinstance(r, dict)] if isinstance(body, list) else []


def _records(path: Path) -> Iterable[dict[str, Any]]:
    return iter_records(path, strict=False) if path.exists() else ()


def latest_records(path: Path) -> list[dict[str, Any]]:
    """
    Return the newest successful record per request key of one archive file, in a deterministic order.

    Args:
        path: Archive file (missing is fine: no records).

    Returns:
        Records sorted by their request key.
    """
    latest = latest_successful(_records(path))
    return [latest[key] for key in sorted(latest)]


def _page(record: dict[str, Any]) -> int:
    try:
        return int(record["request"]["params"].get("page", 0))
    except (TypeError, ValueError):
        return 0


def merged_pages(path: Path) -> tuple[Optional[list[dict[str, Any]]], list[dict[str, Any]]]:
    """
    Concatenate the pages of a global endpoint, dropping a page whose body repeats the previous one (D38).

    Args:
        path: The endpoint's ``_.jsonl.gz``.

    Returns:
        ``(rows or None if nothing successful was archived, the records used)``.
    """
    records = sorted(latest_records(path), key=_page)
    rows: Optional[list[dict[str, Any]]] = None
    used: list[dict[str, Any]] = []
    previous: Any = None
    for record in records:
        body = record.get("body")
        if rows is not None and body == previous:
            continue  # an endpoint that ignores `page` repeats itself
        previous = body
        rows = (rows or []) + body_rows(body)
        used.append(record)
    return rows, used


@dataclass(frozen=True)
class SymbolFile:
    """One per-symbol archive file."""

    symbol: str
    path: Path


def symbol_files(dump_dir: Path, endpoint: str) -> list[SymbolFile]:
    """
    List the per-symbol files of an endpoint, sorted by symbol.

    Args:
        dump_dir: ``<raw_root>/<dump_id>``.
        endpoint: Endpoint path.

    Returns:
        One entry per file whose name decodes to a symbol; non-canonical names are logged and skipped.
    """
    folder = dump_dir / endpoint_dir(endpoint)
    files: list[SymbolFile] = []
    for path in sorted(folder.glob("*" + SUFFIX)) if folder.is_dir() else []:
        try:
            symbol = decode_symbol(path.name[: -len(SUFFIX)])
        except ValueError:
            _logger.warning("Ignoring archive file with a non-canonical name: %s", path)
            continue
        if symbol is not None:
            files.append(SymbolFile(symbol, path))
    return sorted(files, key=lambda f: f.symbol)


def iter_symbol_records(dump_dir: Path, endpoint: str) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """
    Yield ``(symbol, newest successful records)`` for every per-symbol file of an endpoint, sorted by symbol.

    Args:
        dump_dir: ``<raw_root>/<dump_id>``.
        endpoint: Endpoint path.
    """
    for entry in symbol_files(dump_dir, endpoint):
        yield entry.symbol, latest_records(entry.path)


def first_row(records: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return the first row of the newest record (``profile``-style single-object endpoints), or ``None``."""
    if not records:
        return None
    newest = max(records, key=lambda r: (r["fetched_at"], request_key(r)))
    rows = body_rows(newest.get("body"))
    return rows[0] if rows and rows[0] else None
