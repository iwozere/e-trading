"""
The increment layer on disk (spec §10.2, §10.4): yfinance bars and events newer than the vintage, stored **raw**.

Layout under ``<data_root>/increments/``::

    1d/<bucket>/<security_id>.parquet       raw daily bars   (vintage daily schema + ingested_at)
    1h/<bucket>/<security_id>.parquet       raw hourly bars  (vintage intraday schema + ingested_at)
    events/<bucket>/<security_id>.parquet   splits and dividends found in yfinance actions
    _meta.json                              increment_seq, writer host, last run summary

Files are per security and per timeframe and replaced atomically (temp file + ``os.replace``); a reader sees the
old or the new file, never a partial one. Every bar row carries ``source = 'yfinance'``, ``ingested_at`` and the
``vintage`` it extends (rule 6). ``IncrementStore`` is also the adapter's ``IncrementSource``.
"""

from __future__ import annotations

import io
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.fmp_pipeline.builders.parquet_io import ParquetSettings, writer_options
from src.data.fmp_pipeline.builders.prices import DAILY_SCHEMA, INTRADAY_SCHEMA, bucket_path
from src.data.fmp_pipeline.fsutil import atomic_write_bytes

INCREMENTS_DIR = "increments"
META_FILE = "_meta.json"
SOURCE = "yfinance"
INGESTED_AT = pa.field("ingested_at", pa.timestamp("us", tz="UTC"))
SCHEMAS = {"1d": DAILY_SCHEMA.append(INGESTED_AT), "1h": INTRADAY_SCHEMA.append(INGESTED_AT)}
EVENTS_SCHEMA = pa.schema([
    pa.field("security_id", pa.int64()), pa.field("type", pa.string()), pa.field("date", pa.date32()),
    pa.field("numerator", pa.float64()), pa.field("denominator", pa.float64()), pa.field("amount", pa.float64()),
    pa.field("source", pa.string()), INGESTED_AT,
])
KIND = {"1d": "1d", "1h": "1h", "events": "events"}


class IncrementStore:
    """Reads and (for the writer) writes the increment layer."""

    def __init__(self, data_root: Path, settings: Optional[ParquetSettings] = None) -> None:
        """
        Args:
            data_root: Pipeline data root; the layer lives in ``<data_root>/increments``.
            settings: Parquet settings for writes.
        """
        self.root = data_root / INCREMENTS_DIR
        self.settings = settings or ParquetSettings()

    # --- paths and meta ---------------------------------------------------------------------------------------

    def path(self, kind: str, security_id: int) -> Path:
        """Return the file of one security: ``kind`` is ``1d``, ``1h`` or ``events``."""
        return self.root / bucket_path(KIND[kind], security_id).removeprefix("prices/")

    def meta(self) -> dict[str, Any]:
        """Return ``_meta.json`` (empty when the layer was never written)."""
        path = self.root / META_FILE
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def write_meta(self, meta: dict[str, Any]) -> None:
        """Replace ``_meta.json`` atomically."""
        atomic_write_bytes(self.root / META_FILE, json.dumps(meta, indent=1, sort_keys=True, default=str).encode())

    # --- bars -------------------------------------------------------------------------------------------------

    def read_table(self, timeframe: str, security_id: int) -> Optional[pa.Table]:
        """Return a security's stored increment bars as Arrow, or ``None``."""
        path = self.path(timeframe, security_id)
        return pq.read_table(path) if path.exists() else None

    def last_bar(self, timeframe: str, security_id: int) -> Optional[pd.Timestamp]:
        """Return the newest stored bar time (daily: midnight of the date; hourly: UTC, tz-naive)."""
        frame = self.bars(security_id, "", timeframe, None)
        return None if frame is None or frame.empty else pd.Timestamp(frame.index.max())

    def write_bars(self, timeframe: str, security_id: int, table: pa.Table) -> None:
        """Replace a security's increment file with ``table`` (already merged, sorted, unique)."""
        self._write(self.path(timeframe, security_id), table.cast(SCHEMAS[timeframe]))

    def bars(self, security_id: int, symbol: str, timeframe: str, after: Optional[date]) -> Optional[pd.DataFrame]:
        """
        ``IncrementSource.bars``: raw bars on or after ``after`` (the adapter keeps those after its last bar).

        Returns:
            Frame on a tz-naive ``DatetimeIndex`` (daily date or UTC bar start) with ``open`` … ``volume``,
            ``flags``, ``source``; ``None`` if there is no file for this timeframe.
        """
        if timeframe not in SCHEMAS:
            return None
        table = self.read_table(timeframe, security_id)
        if table is None:
            return None
        df = table.to_pandas()
        key = "date" if timeframe == "1d" else "ts_utc"
        index = pd.DatetimeIndex(pd.to_datetime(df.pop(key)))
        if index.tz is not None:
            index = index.tz_convert("UTC").tz_localize(None)
        df.index = index.rename("timestamp")
        df = df.drop(columns=["security_id", "vintage", "ingested_at"] + (["date_et"] if key == "ts_utc" else []))
        if after is not None:
            df = df[df.index >= pd.Timestamp(after)]
        return df.sort_index()

    # --- events -----------------------------------------------------------------------------------------------

    def read_events(self, security_id: int) -> Optional[pa.Table]:
        """Return a security's stored events as Arrow, or ``None``."""
        path = self.path("events", security_id)
        return pq.read_table(path) if path.exists() else None

    def write_events(self, security_id: int, table: pa.Table) -> None:
        """Replace a security's events file."""
        self._write(self.path("events", security_id), table.cast(EVENTS_SCHEMA))

    def events(self, security_id: int, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """``IncrementSource.events``: splits ``[date, numerator, denominator]``, dividends ``[date, amount]``."""
        table = self.read_events(security_id)
        df = table.to_pandas() if table is not None else pd.DataFrame(columns=EVENTS_SCHEMA.names)
        splits = df.loc[df["type"] == "split", ["date", "numerator", "denominator"]].reset_index(drop=True)
        dividends = df.loc[df["type"] == "dividend", ["date", "amount"]].reset_index(drop=True)
        return splits, dividends

    # --- internals ----------------------------------------------------------------------------------------------

    def _write(self, path: Path, table: pa.Table) -> None:
        buffer = io.BytesIO()
        pq.write_table(table, buffer, row_group_size=self.settings.row_group_rows,
                       **writer_options(table.schema, self.settings))
        atomic_write_bytes(path, buffer.getvalue())


def utc_now() -> datetime:
    """Current UTC time (the ``ingested_at`` stamp)."""
    return datetime.now(timezone.utc)
