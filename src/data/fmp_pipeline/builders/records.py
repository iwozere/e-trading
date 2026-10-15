"""
Generic record tables (spec §7.1, §7.4): every non-price endpoint becomes one Parquet table, losslessly.

A ``TableSpec`` names the endpoints feeding a table and the fields that get **typed columns** (snake_case, spec
names). Everything else in a row is preserved:

- with ``numeric=True`` (statements, metrics, recon prices, macro series), fields whose values are numbers in the
  first ``schema_sample_files`` files become ``float64`` columns under their FMP name;
- every remaining field, and any value that did not fit its typed column, goes into ``_extra`` (compact JSON,
  sorted keys), so a later need for an unused field never forces a re-download.

Per request key only the newest successful response counts (spec §6.6). Rows dated outside the ``from``/``to`` of
their own request are dropped and counted (FMP silently ignores a bad ``from``, probe F7). With ``dedup`` columns,
the newest row per key wins.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date
from itertools import chain, islice
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa

from src.data.fmp_pipeline.archive.archive import SUFFIX
from src.data.fmp_pipeline.archive.lists import body_rows, iter_symbol_records, latest_records
from src.data.fmp_pipeline.archive.reader import request_key
from src.data.fmp_pipeline.archive.symbol_path import NO_SYMBOL, endpoint_dir
from src.data.fmp_pipeline.builders.parquet_io import ParquetSettings, TableWriter

EXTRA = "_extra"
ARROW_TYPES = {"str": pa.string(), "date": pa.date32(), "int": pa.int64(), "float": pa.float64(), "bool": pa.bool_()}

#: ``(symbol, date) -> security_id``; ``None`` when the symbol is not a security of the vintage.
Resolver = Callable[[Optional[str], Optional[date]], Optional[int]]


@dataclass(frozen=True)
class Field:
    """
    A typed column: FMP field ``source`` stored as ``column`` of ``kind`` (str, date, int, float, bool).

    A ``source`` starting with ``@`` is taken from the request's parameters when the row lacks it
    (``@name`` for ``economic-indicators``, whose rows do not repeat the indicator name).
    """

    source: str
    column: str
    kind: str = "str"


@dataclass(frozen=True)
class TableSpec:
    """One output table."""

    name: str
    endpoints: tuple[str, ...]
    per_symbol: bool
    fields: tuple[Field, ...] = (Field("symbol", "symbol"), Field("date", "date", "date"))
    dedup: tuple[str, ...] = ()
    numeric: bool = False
    label: Optional[str] = None
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def path(self) -> str:
        """Relative output path inside the vintage."""
        return f"{self.name}.parquet"


def parse_date(value: Any) -> Optional[date]:
    """Return the date of an ``YYYY-MM-DD…`` string, else ``None`` (empty strings included, probe: dividends)."""
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _convert(value: Any, kind: str) -> tuple[Any, bool]:
    """Return ``(converted value, fits)``; ``fits`` is False when a non-null value could not be converted."""
    if value is None or value == "":
        return None, True  # FMP sends "" for missing values (probe: dividends.declarationDate)
    if kind == "str":
        return (value, True) if isinstance(value, str) else (str(value), True)
    if kind == "date":
        parsed = parse_date(value)
        return parsed, parsed is not None
    if kind == "bool":
        return (value, True) if isinstance(value, bool) else (None, False)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if isinstance(value, str):
            try:
                number = float(value)
            except ValueError:
                return None, False
            if kind == "int":
                return (int(number), True) if number.is_integer() else (None, False)
            return number, True
        return None, False
    if kind == "int":
        return (int(value), True) if float(value).is_integer() else (None, False)
    return float(value), True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def in_window(record: dict[str, Any], row_date: Optional[date]) -> bool:
    """True unless the row's date lies outside the ``from``/``to`` its request asked for."""
    if row_date is None:
        return True
    params = record["request"]["params"]
    start, end = parse_date(params.get("from")), parse_date(params.get("to"))
    return not ((start is not None and row_date < start) or (end is not None and row_date > end))


def _page(record: dict[str, Any]) -> int:
    try:
        return int(record["request"]["params"].get("page", 0))
    except (TypeError, ValueError):
        return 0


def global_records(path: Path) -> list[dict[str, Any]]:
    """Newest records of a global endpoint by page, without pages that repeat the previous one (D38)."""
    out: list[dict[str, Any]] = []
    previous: Any = object()
    for record in sorted(latest_records(path), key=lambda r: (_page(r), request_key(r))):
        paged = "page" in record["request"]["params"]
        if paged and record.get("body") == previous:
            continue
        previous = record.get("body")
        out.append(record)
    return out


@dataclass
class TableStats:
    """What building one table did; goes into ``build_info.json``."""

    rows: int = 0
    responses: int = 0
    out_of_window: int = 0
    duplicates: int = 0
    unresolved: int = 0
    input_digest: str = ""


class RecordTableBuilder:
    """Builds one ``TableSpec`` from the archive into one Parquet file."""

    def __init__(self, spec: TableSpec, dump_dir: Path, resolver: Resolver, settings: ParquetSettings,
                 batch_rows: int = 50_000, schema_sample_files: int = 500) -> None:
        """
        Args:
            spec: The table.
            dump_dir: ``<raw_root>/<dump_id>``.
            resolver: Maps a row's symbol and date to a ``security_id``.
            settings: Parquet settings.
            batch_rows: Rows buffered before a batch is written.
            schema_sample_files: Files inspected to choose the numeric passthrough columns.
        """
        self.spec = spec
        self.dump_dir = dump_dir
        self.resolver = resolver
        self.settings = settings
        self.batch_rows = batch_rows
        self.sample_files = schema_sample_files
        self.stats = TableStats()
        self.numeric: list[str] = []
        self._digest = hashlib.sha256()
        self._date_column = next((f.column for f in spec.fields if f.kind == "date"), None)

    def _groups(self) -> Iterator[tuple[str, Optional[str], list[dict[str, Any]]]]:
        """Yield ``(endpoint, file symbol, records)`` per archive file, in a deterministic order."""
        for endpoint in self.spec.endpoints:
            if self.spec.per_symbol:
                for symbol, records in iter_symbol_records(self.dump_dir, endpoint):
                    yield endpoint, symbol, records
            else:
                path = self.dump_dir / endpoint_dir(endpoint) / (NO_SYMBOL + SUFFIX)
                yield endpoint, None, global_records(path)

    def _numeric_columns(self, sample: list[tuple[str, Optional[str], list[dict[str, Any]]]]) -> list[str]:
        if not self.spec.numeric:
            return []
        typed = {f.source for f in self.spec.fields}
        numeric: set[str] = set()
        other: set[str] = set()
        for _, _, records in sample:
            for record in records:
                for row in body_rows(record.get("body")):
                    for key, value in row.items():
                        if key in typed or value is None:
                            continue
                        (numeric if _is_number(value) else other).add(key)
        return sorted(numeric - other)

    def schema(self, numeric: list[str]) -> pa.Schema:
        """Return the output schema: ``security_id``, label, typed fields, numeric passthrough, ``_extra``."""
        columns = [pa.field("security_id", pa.int64())]
        if self.spec.label:
            columns.append(pa.field(self.spec.label, pa.string()))
        columns += [pa.field(f.column, ARROW_TYPES[f.kind]) for f in self.spec.fields]
        columns += [pa.field(name, pa.float64()) for name in numeric]
        columns.append(pa.field(EXTRA, pa.string()))
        return pa.schema(columns)

    def _rows(self, endpoint: str, file_symbol: Optional[str], records: list[dict[str, Any]],
              numeric: list[str]) -> list[dict[str, Any]]:
        """Convert one file's records to output rows (window check, dedup, deterministic order)."""
        out: dict[Any, dict[str, Any]] = {}
        ordered = sorted(records, key=lambda r: (r["fetched_at"], request_key(r)))
        serial = 0
        for record in ordered:
            self.stats.responses += 1
            self._digest.update(f"{request_key(record)} {record['sha256']}\n".encode())
            for row in body_rows(record.get("body")):
                built = self._row(endpoint, file_symbol, row, numeric, record["request"]["params"])
                if not in_window(record, self._when(built)):
                    self.stats.out_of_window += 1
                    continue
                if self.spec.dedup:
                    key: Any = tuple(built.get(c) for c in self.spec.dedup)
                    if key in out:
                        self.stats.duplicates += 1
                        del out[key]  # re-insert so the newest row also takes the newest position
                else:
                    key, serial = serial, serial + 1
                out[key] = built
        rows = list(out.values())
        sort_columns = self.spec.dedup or tuple(f.column for f in self.spec.fields)
        rows.sort(key=lambda r: tuple("" if r.get(c) is None else str(r.get(c)) for c in sort_columns))
        return rows

    def _row(self, endpoint: str, file_symbol: Optional[str], row: dict[str, Any], numeric: list[str],
             params: dict[str, Any]) -> dict[str, Any]:
        built: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        used: set[str] = set()
        for f in self.spec.fields:
            source = f.source.lstrip("@")
            used.add(source)
            raw = row.get(source)
            if raw is None and f.source.startswith("@"):
                raw = params.get(source)
            value, fits = _convert(raw, f.kind)
            built[f.column] = value
            if not fits:
                extra[source] = raw
        for name in numeric:
            used.add(name)
            value = row.get(name)
            if value is None or _is_number(value):
                built[name] = None if value is None else float(value)
            else:
                built[name] = None
                extra[name] = value
        for key, value in row.items():
            if key not in used:
                extra[key] = value
        symbol = built.get("symbol") if isinstance(built.get("symbol"), str) else file_symbol
        if "symbol" in built and built["symbol"] is None:
            built["symbol"] = file_symbol
        built["security_id"] = self.resolver(symbol, self._when(built))
        if built["security_id"] is None:
            self.stats.unresolved += 1
        if self.spec.label:
            built[self.spec.label] = self.spec.labels.get(endpoint, endpoint)
        built[EXTRA] = json.dumps(extra, sort_keys=True, ensure_ascii=False, separators=(",", ":")) if extra else None
        return built

    def _when(self, built: dict[str, Any]) -> Optional[date]:
        """The row's date: its first date-typed column (used for the window check and segment resolution)."""
        value = built.get(self._date_column) if self._date_column else None
        return value if isinstance(value, date) else None

    def iter_rows(self) -> Iterator[list[dict[str, Any]]]:
        """
        Yield the output rows of each archive file in turn; ``self.numeric`` is set before the first yield.

        Rows are dicts keyed by the schema's column names.
        """
        groups = self._groups()
        sample = list(islice(groups, self.sample_files))
        self.numeric = self._numeric_columns(sample)
        for endpoint, symbol, records in chain(sample, groups):
            yield self._rows(endpoint, symbol, records, self.numeric)
        self.stats.input_digest = self._digest.hexdigest()

    def build(self, out_dir: Path) -> TableStats:
        """
        Write the table to ``out_dir / spec.path``.

        Args:
            out_dir: The vintage directory being built.

        Returns:
            Row and input statistics.
        """
        files = self.iter_rows()
        buffer: list[dict[str, Any]] = next(files, [])
        writer = TableWriter(out_dir / self.spec.path, self.schema(self.numeric), self.settings)
        try:
            for rows in files:
                buffer += rows
                if len(buffer) >= self.batch_rows:
                    self._flush(writer, buffer)
            self._flush(writer, buffer)
        finally:
            writer.close()
        self.stats.rows = writer.rows
        self.stats.input_digest = self._digest.hexdigest()
        return self.stats

    @staticmethod
    def _flush(writer: TableWriter, buffer: list[dict[str, Any]]) -> None:
        if buffer:
            columns = {name: [row.get(name) for row in buffer] for name in writer.schema.names}
            writer.write(pa.table(columns, schema=writer.schema))
            buffer.clear()


def build_tables(specs: Iterable[TableSpec], dump_dir: Path, out_dir: Path, resolver: Resolver,
                 settings: ParquetSettings, batch_rows: int, sample_files: int) -> dict[str, TableStats]:
    """Build every table whose endpoints have anything archived; return ``{relative path: stats}``."""
    results: dict[str, TableStats] = {}
    for spec in specs:
        if not any((dump_dir / endpoint_dir(e)).is_dir() for e in spec.endpoints):
            continue
        results[spec.path] = RecordTableBuilder(spec, dump_dir, resolver, settings, batch_rows,
                                                sample_files).build(out_dir)
    return results
