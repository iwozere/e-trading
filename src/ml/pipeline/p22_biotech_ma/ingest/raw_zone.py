"""
P22 raw-zone writer.

Single write path for landing external payloads (spec §1, §7.3):

- **Partitioned** by source and date: ``<root>/<source>/<YYYY-MM-DD>/<hash>.json.gz``
- **Immutable**: files are never overwritten; the content hash is part of the path.
- **Idempotent / content-addressed**: an identical payload for the same
  ``(source, entity, as_of_date)`` re-fetch hashes to the same path and is a
  no-op write, per spec §7.3 ("identical payloads are deduplicated by hash").
- **known_from-stamped**: every write records when *we* learned the fact, in a
  companion manifest row, rather than relying on filesystem mtime (which a
  restore/rsync can silently change).

Every other P22 ingest client writes through this module rather than touching
files directly.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import RAW_ZONE_ROOT
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass(frozen=True)
class RawZoneWriteResult:
    """Outcome of a single raw-zone write."""

    path: Path
    content_hash: str
    known_from: datetime
    was_new: bool  # False if this content hash already existed (deduped, not re-written)


def _content_hash(payload: Any) -> str:
    """SHA-256 of the normalized (sorted-key) JSON payload."""
    normalized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def write(
    source: str,
    entity: str,
    as_of_date: date,
    payload: Any,
    *,
    root: Path | None = None,
) -> RawZoneWriteResult:
    """
    Land a payload in the raw zone, idempotently.

    Args:
        source: Data source name, e.g. "sec_submissions", "clinicaltrials", "openfda",
            "orange_book", "purple_book". Used as the top-level partition directory.
        entity: A stable identifier for the thing this payload describes (CIK, NCT ID,
            application number, ...). Recorded in the manifest for lookup; not part of
            the path (the content hash is what dedupes, not the entity).
        as_of_date: The date partition this write belongs to (typically "today", i.e.
            the date we fetched it — this is the ingest date, not `valid_from`).
        payload: JSON-serializable payload to land.
        root: Override the raw-zone root (used by tests).

    Returns:
        RawZoneWriteResult with the path, hash, known_from timestamp, and whether this
        was a new write or a dedup no-op.
    """
    zone_root = root if root is not None else RAW_ZONE_ROOT
    content_hash = _content_hash(payload)
    partition_dir = zone_root / source / as_of_date.isoformat()
    file_path = partition_dir / f"{content_hash}.json.gz"
    manifest_path = partition_dir / f"{content_hash}.manifest.json"

    known_from = datetime.now(timezone.utc)

    if file_path.exists():
        _logger.debug(
            "Raw-zone dedup hit: source=%s entity=%s hash=%s — not re-written",
            source,
            entity,
            content_hash,
        )
        return RawZoneWriteResult(
            path=file_path,
            content_hash=content_hash,
            known_from=known_from,
            was_new=False,
        )

    partition_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, default=str).encode("utf-8")
    with gzip.open(file_path, "wb") as f:
        f.write(body)

    manifest = {
        "source": source,
        "entity": entity,
        "as_of_date": as_of_date.isoformat(),
        "content_hash": content_hash,
        "known_from": known_from.isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _logger.info(
        "Raw-zone write: source=%s entity=%s date=%s hash=%s",
        source,
        entity,
        as_of_date.isoformat(),
        content_hash,
    )
    return RawZoneWriteResult(
        path=file_path,
        content_hash=content_hash,
        known_from=known_from,
        was_new=True,
    )


def has_any_landed(source: str, entity: str, *, root: Path | None = None) -> bool:
    """
    True if ANY partition (any ingest date) already has a landed payload for
    this `(source, entity)` pair — for a resumable bulk job to skip
    re-fetching something a prior, possibly-interrupted run already landed
    (e.g. `ingest/fmp_backfill.py`'s historical-price backfill).

    O(files under `source`) per call — fine for a one-time/occasional bulk
    job checking a few hundred/thousand entities once each; not meant for a
    hot path called per-request in a high-frequency job.
    """
    zone_root = root if root is not None else RAW_ZONE_ROOT
    source_dir = zone_root / source
    if not source_dir.is_dir():
        return False
    for date_dir in source_dir.iterdir():
        if not date_dir.is_dir():
            continue
        for manifest_path in date_dir.glob("*.manifest.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("entity") == entity:
                return True
    return False


def read(path: Path) -> Any:
    """Read back a gzipped JSON payload previously written by `write()`."""
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _latest_partition_dir(source: str, root: Path | None) -> Path | None:
    zone_root = root if root is not None else RAW_ZONE_ROOT
    source_dir = zone_root / source
    if not source_dir.is_dir():
        _logger.warning("No data landed yet for source=%s at %s", source, source_dir)
        return None
    date_dirs = sorted((d for d in source_dir.iterdir() if d.is_dir()), reverse=True)
    return date_dirs[0] if date_dirs else None


def read_latest_partition(source: str, *, root: Path | None = None) -> list[Any]:
    """
    Read every payload landed under `source`'s most recent date partition.

    Mirrors `universe_snapshot.latest_universe_rows`'s "most recent date
    directory" convention, generalized for any source. Each file in that
    partition is one entity's landed payload (e.g. one company's list of
    CT.gov studies, or one company's list of openFDA applications) — callers
    get back the list of payloads, not a flattened/deduped row set, since
    what "dedup" means is source-specific.

    Returns:
        List of payloads (whatever shape `write()` was called with),
        possibly empty if nothing has been landed yet for this source.
    """
    latest_dir = _latest_partition_dir(source, root)
    if latest_dir is None:
        return []

    payloads = [read(f) for f in latest_dir.glob("*.json.gz")]
    _logger.info("Loaded %d payloads from latest %s snapshot (%s)", len(payloads), source, latest_dir.name)
    return payloads


def read_latest_partition_with_manifest(source: str, *, root: Path | None = None) -> "list[tuple[Any, dict[str, Any]]]":
    """
    Like `read_latest_partition`, but pairs each payload with its full
    manifest dict (`source`, `entity`, `as_of_date`, `content_hash`,
    `known_from`) instead of discarding it. Two manifest fields matter most
    to callers so far: `known_from` — when *we* learned that payload's
    contents, per spec §3.4's bitemporal requirement that any downstream
    write derived from a landed payload use that timestamp, never "now" (the
    read/review time) — and `entity` (e.g. a CIK), needed by any normalizer
    that has to know which company a payload belongs to.

    Returns:
        List of `(payload, manifest)` tuples, possibly empty. A manifest
        missing `known_from` (shouldn't happen for anything `write()`
        produced) is skipped rather than passed through with a `None`.
    """
    latest_dir = _latest_partition_dir(source, root)
    if latest_dir is None:
        return []

    results: "list[tuple[Any, dict[str, Any]]]" = []
    for manifest_path in latest_dir.glob("*.manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload_path = manifest_path.with_suffix("").with_suffix(".json.gz")
        if not payload_path.exists() or not manifest.get("known_from"):
            continue
        results.append((read(payload_path), manifest))

    _logger.info("Loaded %d (payload, manifest) pairs from latest %s snapshot (%s)", len(results), source, latest_dir.name)
    return results
