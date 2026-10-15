"""
Deterministic Parquet writing for the vintage (spec §7.3): zstd, ~100k-row row groups, statistics on, byte-stream
split for float columns, and content hashes for ``build_info.json`` (V-REPRO).

The same input and the same library versions give byte-identical files: no timestamps are written into the file
metadata, and rows arrive in a deterministic order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq

HASH_CHUNK = 1 << 20


@dataclass(frozen=True)
class ParquetSettings:
    """Writer settings, recorded in ``build_info.json``."""

    zstd_level: int = 6
    row_group_rows: int = 100_000

    def as_dict(self) -> dict[str, Any]:
        """Return the settings for ``build_info.json``."""
        return {"compression": "zstd", "compression_level": self.zstd_level, "row_group_rows": self.row_group_rows,
                "statistics": True, "byte_stream_split": "float columns", "pyarrow": pa.__version__}


def _float_columns(schema: pa.Schema) -> list[str]:
    return [f.name for f in schema if pa.types.is_floating(f.type)]


def writer_options(schema: pa.Schema, settings: ParquetSettings) -> dict[str, Any]:
    """Return ``pq.ParquetWriter`` keyword arguments for ``schema``."""
    floats = _float_columns(schema)
    others = [f.name for f in schema if f.name not in floats]
    return {
        "compression": "zstd",
        "compression_level": settings.zstd_level,
        "write_statistics": True,
        "use_dictionary": others,
        "use_byte_stream_split": floats or False,
    }


class TableWriter:
    """Streams record batches of one schema into one Parquet file."""

    def __init__(self, path: Path, schema: pa.Schema, settings: ParquetSettings) -> None:
        """
        Args:
            path: Output file; its directory is created.
            schema: Arrow schema of every batch.
            settings: Writer settings.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.schema = schema
        self.rows = 0
        self._settings = settings
        self._writer: Optional[pq.ParquetWriter] = pq.ParquetWriter(path, schema, **writer_options(schema, settings))

    def write(self, table: pa.Table) -> None:
        """Append ``table`` (cast to the writer's schema) in row groups of ``row_group_rows``."""
        if table.num_rows == 0:
            return
        assert self._writer is not None, "writer already closed"
        self._writer.write_table(table.cast(self.schema), row_group_size=self._settings.row_group_rows)
        self.rows += table.num_rows

    def close(self) -> None:
        """Finish the file (an empty table still gets a valid file with the schema)."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def write_table(path: Path, table: pa.Table, settings: ParquetSettings) -> int:
    """
    Write one table to one file.

    Args:
        path: Output file.
        table: Data.
        settings: Writer settings.

    Returns:
        Rows written.
    """
    writer = TableWriter(path, table.schema, settings)
    try:
        writer.write(table)
    finally:
        writer.close()
    return writer.rows


def file_sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()
