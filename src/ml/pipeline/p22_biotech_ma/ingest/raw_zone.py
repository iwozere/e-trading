"""
Backward-compatible re-export shim.

The raw-zone writer moved to `src.data.pipeline.raw_zone` — a
pipeline-agnostic version of this module, promoted so other pipelines can
reuse the same content-hashed, immutable landing pattern (see that module's
docstring for the full contract). This shim keeps every existing P22 call
site working unchanged by defaulting `root` to P22's own
`RAW_ZONE_ROOT` (`src.ml.pipeline.p22_biotech_ma.config.RAW_ZONE_ROOT`),
since P22 code never passes `root=` explicitly except in tests.

New code should import `src.data.pipeline.raw_zone` directly and pass its
own `root` — do not add new callers to this shim.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from src.data.pipeline import raw_zone as _raw_zone
from src.data.pipeline.raw_zone import RawZoneWriteResult  # re-exported for existing imports
from src.ml.pipeline.p22_biotech_ma.config import RAW_ZONE_ROOT

__all__ = [
    "RawZoneWriteResult",
    "write",
    "has_any_landed",
    "read",
    "read_latest_partition",
    "read_partition_before",
    "read_latest_partition_with_manifest",
]


def write(
    source: str,
    entity: str,
    as_of_date: date,
    payload: Any,
    *,
    root: Path | None = None,
) -> RawZoneWriteResult:
    """See `src.data.pipeline.raw_zone.write`. Defaults `root` to P22's `RAW_ZONE_ROOT`."""
    return _raw_zone.write(source, entity, as_of_date, payload, root=root if root is not None else RAW_ZONE_ROOT)


def has_any_landed(source: str, entity: str, *, root: Path | None = None) -> bool:
    """See `src.data.pipeline.raw_zone.has_any_landed`. Defaults `root` to P22's `RAW_ZONE_ROOT`."""
    return _raw_zone.has_any_landed(source, entity, root=root if root is not None else RAW_ZONE_ROOT)


def read(path: Path) -> Any:
    """See `src.data.pipeline.raw_zone.read`."""
    return _raw_zone.read(path)


def read_latest_partition(source: str, *, root: Path | None = None) -> list[Any]:
    """See `src.data.pipeline.raw_zone.read_latest_partition`. Defaults `root` to P22's `RAW_ZONE_ROOT`."""
    return _raw_zone.read_latest_partition(source, root=root if root is not None else RAW_ZONE_ROOT)


def read_partition_before(source: str, before_date: date, *, root: Path | None = None) -> list[Any]:
    """See `src.data.pipeline.raw_zone.read_partition_before`. Defaults `root` to P22's `RAW_ZONE_ROOT`."""
    return _raw_zone.read_partition_before(source, before_date, root=root if root is not None else RAW_ZONE_ROOT)


def read_latest_partition_with_manifest(source: str, *, root: Path | None = None) -> "list[tuple[Any, dict[str, Any]]]":
    """See `src.data.pipeline.raw_zone.read_latest_partition_with_manifest`. Defaults `root` to P22's `RAW_ZONE_ROOT`."""
    return _raw_zone.read_latest_partition_with_manifest(source, root=root if root is not None else RAW_ZONE_ROOT)
