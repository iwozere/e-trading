"""
Corporate events tables (spec §7.3): ``events/splits.parquet`` and ``events/dividends.parquet``.

Splits get ``ratio = numerator / denominator`` (4.0 for 4-for-1, 0.1 for 1-for-10); a row whose ratio cannot be
computed keeps a null ratio and is counted. Dividends keep FMP's raw ``dividend`` as ``amount_raw`` and
``adjDividend`` as ``amount_split_adjusted`` (probe F1); an empty ``declarationDate`` is null (probe). Both
tables are small enough to hold in memory, and the price builder needs them per security for the flags.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from src.data.fmp_pipeline.builders.parquet_io import ParquetSettings, write_table
from src.data.fmp_pipeline.builders.records import (
    EXTRA,
    Field,
    RecordTableBuilder,
    Resolver,
    TableSpec,
    TableStats,
)

SOURCE = "fmp"
SPLITS = TableSpec("events/splits", ("splits",), per_symbol=True,
                   fields=(Field("symbol", "symbol"), Field("date", "date", "date"),
                           Field("numerator", "numerator", "float"), Field("denominator", "denominator", "float")),
                   dedup=("symbol", "date"))
DIVIDENDS = TableSpec("events/dividends", ("dividends",), per_symbol=True,
                      fields=(Field("symbol", "symbol"), Field("date", "ex_date", "date"),
                              Field("recordDate", "record_date", "date"), Field("paymentDate", "payment_date", "date"),
                              Field("declarationDate", "declaration_date", "date"),
                              Field("dividend", "amount_raw", "float"),
                              Field("adjDividend", "amount_split_adjusted", "float"),
                              Field("frequency", "frequency")),
                      dedup=("symbol", "ex_date"))

SPLITS_SCHEMA = pa.schema([pa.field("security_id", pa.int64()), pa.field("symbol", pa.string()),
                           pa.field("date", pa.date32()), pa.field("numerator", pa.float64()),
                           pa.field("denominator", pa.float64()), pa.field("ratio", pa.float64()),
                           pa.field("source", pa.string()), pa.field(EXTRA, pa.string())])
DIVIDENDS_SCHEMA = pa.schema([pa.field("security_id", pa.int64()), pa.field("symbol", pa.string()),
                              pa.field("ex_date", pa.date32()), pa.field("record_date", pa.date32()),
                              pa.field("payment_date", pa.date32()), pa.field("declaration_date", pa.date32()),
                              pa.field("amount_raw", pa.float64()), pa.field("amount_split_adjusted", pa.float64()),
                              pa.field("frequency", pa.string()), pa.field("source", pa.string()),
                              pa.field(EXTRA, pa.string())])


@dataclass
class EventsResult:
    """Event dates per security (for the flags) and the build statistics."""

    split_dates: dict[int, np.ndarray] = field(default_factory=dict)
    dividend_dates: dict[int, np.ndarray] = field(default_factory=dict)
    stats: dict[str, TableStats] = field(default_factory=dict)
    invalid_ratios: int = 0


def _collect(builder: RecordTableBuilder) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part in builder.iter_rows():
        rows += part
    rows.sort(key=lambda r: (r["security_id"] is None, r["security_id"] or 0, str(r.get("symbol")),
                             str(r.get("date") or r.get("ex_date"))))
    return rows


def _dates(values: list[Any]) -> np.ndarray:
    return np.array(sorted(v for v in values if v is not None), dtype="datetime64[D]")


def build_events(dump_dir: Path, out_dir: Path, resolver: Resolver, settings: ParquetSettings) -> EventsResult:
    """
    Build both event tables and return the event dates per security.

    Args:
        dump_dir: ``<raw_root>/<dump_id>``.
        out_dir: Vintage directory being built.
        resolver: Symbol and date to ``security_id``.
        settings: Parquet settings.

    Returns:
        Event dates per ``security_id`` and statistics per table.
    """
    result = EventsResult()
    split_dates: dict[int, list[Any]] = defaultdict(list)
    dividend_dates: dict[int, list[Any]] = defaultdict(list)

    splits_builder = RecordTableBuilder(SPLITS, dump_dir, resolver, settings)
    splits = _collect(splits_builder)
    for row in splits:
        num, den = row["numerator"], row["denominator"]
        row["ratio"] = num / den if num and den and num > 0 and den > 0 else None
        row["source"] = SOURCE
        if row["ratio"] is None:
            result.invalid_ratios += 1
        elif row["security_id"] is not None and row["date"] is not None:
            split_dates[row["security_id"]].append(row["date"])
    splits_builder.stats.rows = write_table(out_dir / SPLITS.path, _table(splits, SPLITS_SCHEMA), settings)
    result.stats[SPLITS.path] = splits_builder.stats

    dividends_builder = RecordTableBuilder(DIVIDENDS, dump_dir, resolver, settings)
    dividends = _collect(dividends_builder)
    for row in dividends:
        row["source"] = SOURCE
        if row["security_id"] is not None and row["ex_date"] is not None:
            dividend_dates[row["security_id"]].append(row["ex_date"])
    dividends_builder.stats.rows = write_table(out_dir / DIVIDENDS.path, _table(dividends, DIVIDENDS_SCHEMA), settings)
    result.stats[DIVIDENDS.path] = dividends_builder.stats

    result.split_dates = {sid: _dates(v) for sid, v in split_dates.items()}
    result.dividend_dates = {sid: _dates(v) for sid, v in dividend_dates.items()}
    return result


def _table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    return pa.table({name: [row.get(name) for row in rows] for name in schema.names}, schema=schema)
