"""
Build the local Parquet store from the raw Sharadar files.

Raw files are ``<data_root>/raw/<table>.csv.zip`` (bulk download) or ``<table>.csv`` (free sample); both hold the
same CSV columns. The build is repeatable: it replaces ``<data_root>/store`` and never changes ``raw``.

Store layout (under ``<data_root>/store``)::

    reference/securities.parquet      one row per permaticker, FMP-vintage column names
    reference/symbol_changes.parquet  former symbols
    events/splits.parquet             adjust() form, by security_id
    events/dividends.parquet          raw-basis cash and spinoff dividends
    events/delistings.parquet         last trade date, reason, consideration
    prices/1d_raw/bucket=NN.parquet   raw daily bars, sorted by (security_id, date)
    fundamentals/fundamentals_ar.parquet   as-reported dimensions only
    tables/<table>.parquet            every other table as delivered
    _meta/build_info.json
"""

from __future__ import annotations

import json
import shutil
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.sharadar_pipeline import config
from src.data.sharadar_pipeline.convert import (
    delistings_frame,
    dividends_frame,
    fundamentals_ar,
    raw_prices,
    securities_frame,
    splits_frame,
    symbol_changes_frame,
)
from src.data.sharadar_pipeline.mapping import map_to_fmp
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

CHUNK_ROWS = 1_000_000
_PRICE_COLUMNS = ["security_id", "date", "open", "high", "low", "close", "volume", "closeadj_vendor"]
_PRICE_SCHEMA = pa.schema([pa.field("security_id", pa.int64()), pa.field("date", pa.date32()),
                           *[pa.field(name, pa.float64()) for name in _PRICE_COLUMNS[2:]]])


def raw_path(data_root: Path, table: str) -> Optional[Path]:
    """Return the raw file of ``table`` (zip preferred), or None if it was not downloaded."""
    for name in (f"{table}.csv.zip", f"{table}.csv"):
        path = data_root / config.RAW_DIR / name
        if path.exists():
            return path
    return None


@contextmanager
def _open_csv(path: Path) -> Iterator[IO[bytes]]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"{path} must hold exactly one CSV, found {archive.namelist()}")
            with archive.open(members[0]) as handle:
                yield handle
    else:
        with path.open("rb") as handle:
            yield handle


def read_raw(data_root: Path, table: str) -> pd.DataFrame:
    """Read a whole raw table as strings (small tables only)."""
    path = raw_path(data_root, table)
    if path is None:
        raise FileNotFoundError(f"raw table {table} is missing under {data_root / config.RAW_DIR}")
    with _open_csv(path) as handle:
        return pd.read_csv(handle, dtype=str, keep_default_na=False)


def read_raw_chunks(data_root: Path, table: str, chunk_rows: int = CHUNK_ROWS) -> Iterator[pd.DataFrame]:
    """Yield a raw table in chunks of strings (large tables)."""
    path = raw_path(data_root, table)
    if path is None:
        raise FileNotFoundError(f"raw table {table} is missing under {data_root / config.RAW_DIR}")
    with _open_csv(path) as handle:
        yield from pd.read_csv(handle, dtype=str, keep_default_na=False, chunksize=chunk_rows)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _with_ids(frame: pd.DataFrame, ticker_ids: dict[str, int]) -> pd.DataFrame:
    """Replace ``ticker`` by ``security_id``; rows of unknown tickers are dropped."""
    out = frame.copy()
    out["security_id"] = out["ticker"].map(ticker_ids)
    out = out.dropna(subset=["security_id"])
    out["security_id"] = out["security_id"].astype("int64")
    return out


def _build_prices(data_root: Path, store: Path, ticker_ids: dict[str, int]) -> dict[str, Any]:
    """Convert ``stocks`` to raw daily bars in bucket files; returns row and security counts."""
    staging = store / "_staging_prices"
    staging.mkdir(parents=True, exist_ok=True)
    writers: dict[int, pq.ParquetWriter] = {}
    rows = unknown = 0
    seen: set[int] = set()
    try:
        for chunk in read_raw_chunks(data_root, "stocks"):
            unknown += int((~chunk["ticker"].isin(ticker_ids.keys())).sum())
            prices = raw_prices(_with_ids(chunk, ticker_ids))
            prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.date
            prices = prices.dropna(subset=["date"])[_PRICE_COLUMNS]
            rows += len(prices)
            seen.update(prices["security_id"].unique().tolist())
            for bucket, part in prices.groupby(prices["security_id"] % config.PRICE_BUCKETS):
                if bucket not in writers:
                    writers[bucket] = pq.ParquetWriter(staging / f"{bucket:02d}.parquet", _PRICE_SCHEMA)
                writers[bucket].write_table(pa.Table.from_pandas(part, schema=_PRICE_SCHEMA, preserve_index=False))
    finally:
        for writer in writers.values():
            writer.close()
    target = store / config.PRICES_DIR
    target.mkdir(parents=True, exist_ok=True)
    for bucket in sorted(writers):
        frame = pq.read_table(staging / f"{bucket:02d}.parquet").to_pandas()
        frame = frame.drop_duplicates(["security_id", "date"], keep="last").sort_values(["security_id", "date"])
        pq.write_table(pa.Table.from_pandas(frame, schema=_PRICE_SCHEMA, preserve_index=False),
                       target / f"bucket={bucket:02d}.parquet", row_group_size=200_000)
    shutil.rmtree(staging)
    if unknown:
        _logger.warning("stocks: %d rows of tickers missing from the tickers table were dropped", unknown)
    return {"price_rows": rows, "price_securities": len(seen), "price_rows_unknown_ticker": unknown}


def _build_passthrough(data_root: Path, store: Path, table: str) -> int:
    """Copy a table to Parquet as delivered (strings), chunk by chunk; returns the row count."""
    path = store / "tables" / f"{table}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[pq.ParquetWriter] = None
    rows = 0
    try:
        for chunk in read_raw_chunks(data_root, table):
            batch = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, batch.schema)
            writer.write_table(batch.cast(writer.schema))
            rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    return rows


def build_store(data_root: Path, fmp_securities: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    """
    Build ``<data_root>/store`` from ``<data_root>/raw``.

    Args:
        data_root: Sharadar data root.
        fmp_securities: FMP vintage securities table; when given, each security gets its ``fmp_security_id``.

    Returns:
        Build summary (also written to ``_meta/build_info.json``).

    Raises:
        FileNotFoundError: A core table (tickers, actions, stocks) is missing.
    """
    store = data_root / config.STORE_DIR
    if store.exists():
        shutil.rmtree(store)
    store.mkdir(parents=True)
    info: dict[str, Any] = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tables": {}}

    tickers = read_raw(data_root, "tickers")
    securities = securities_frame(tickers)
    securities["has_fundamentals"] = securities["sharadar_ticker"].isin(
        set(tickers.loc[tickers["table"].str.lower().isin({"fundamentals", "sf1"}), "ticker"]))
    securities["fmp_security_id"] = (map_to_fmp(securities, fmp_securities) if fmp_securities is not None
                                     else pd.array([pd.NA] * len(securities), dtype="Int64"))
    _write(securities, store / config.SECURITIES)
    ticker_ids = dict(zip(securities["sharadar_ticker"], securities["security_id"]))
    info.update(securities=len(securities), delisted=int((~securities["is_actively_trading"]).sum()),
                mapped_to_fmp=int(securities["fmp_security_id"].notna().sum()))

    actions = read_raw(data_root, "actions")
    splits = splits_frame(actions)
    dividends = dividends_frame(actions, splits)
    frames = {config.SPLITS: splits, config.DIVIDENDS: dividends, config.DELISTINGS: delistings_frame(actions),
              config.SYMBOL_CHANGES: symbol_changes_frame(actions)}
    for relative, frame in frames.items():
        with_ids = _with_ids(frame, ticker_ids).drop(columns=["ticker"]).sort_values(["security_id", "date"])
        _write(with_ids, store / relative)
        info["tables"][relative] = len(with_ids)

    info.update(_build_prices(data_root, store, ticker_ids))

    if raw_path(data_root, "fundamentals") is not None:
        parts = [_with_ids(fundamentals_ar(chunk), ticker_ids) for chunk in read_raw_chunks(data_root, "fundamentals")]
        fundamentals = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        _write(fundamentals.sort_values(["security_id", "dimension", "date", "reportperiod"]), store / config.FUNDAMENTALS)
        info["tables"][config.FUNDAMENTALS] = len(fundamentals)

    done = {"tickers", "actions", "stocks", "fundamentals"}
    for table in config.BULK_TABLES:
        if table not in done and raw_path(data_root, table) is not None:
            info["tables"][f"tables/{table}.parquet"] = _build_passthrough(data_root, store, table)
    _write(tickers, store / "tables" / "tickers.parquet")

    meta = store / config.BUILD_INFO
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
    _logger.info("Sharadar store built: %d securities (%d delisted), %d price rows", info["securities"], info["delisted"],
                 info["price_rows"])
    return info
