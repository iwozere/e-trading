"""
Build the local Parquet store from the raw EODHD files.

The build is repeatable: it replaces ``<data_root>/store`` and never changes ``raw``. Store layout::

    reference/securities.parquet      one row per security, FMP-vintage column names plus EODHD extras
    events/splits.parquet             adjust() form, by security_id
    events/dividends.parquet          raw-basis cash dividends
    prices/1d_raw/bucket=NN.parquet   raw daily bars, sorted by (security_id, date)
    _meta/build_info.json             counts, FMP mapping and delisted coverage by last-trade year
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.eodhd_pipeline import config
from src.data.eodhd_pipeline.convert import BAR_COLUMNS, bars_frame, dividends_frame, splits_frame
from src.data.eodhd_pipeline.download import read_history
from src.data.eodhd_pipeline.mapping import map_to_fmp
from src.data.eodhd_pipeline.universe import load_universe
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_PRICE_COLUMNS = ["security_id", *BAR_COLUMNS]
_PRICE_SCHEMA = pa.schema([pa.field("security_id", pa.int64()), pa.field("date", pa.date32()),
                           *[pa.field(name, pa.float64()) for name in BAR_COLUMNS[1:]]])
#: Last-trade-year buckets of the coverage table in ``build_info``.
YEAR_BUCKETS = ((1990, 1997), (1998, 2002), (2003, 2007), (2008, 2011), (2012, 2016), (2017, 2020), (2021, 2100))


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _bucket_label(year: int) -> Optional[str]:
    for low, high in YEAR_BUCKETS:
        if low <= year <= high:
            return f"{low}+" if high >= 2100 else f"{low}-{high}"
    return None


def coverage_by_year(securities: pd.DataFrame) -> list[dict[str, Any]]:
    """Return, per last-trade-year bucket, the delisted securities with bars and how many the FMP vintage also has."""
    delisted = securities[~securities["is_actively_trading"] & securities["last_price_date"].notna()].copy()
    delisted["bucket"] = pd.to_datetime(delisted["last_price_date"]).dt.year.map(_bucket_label)
    rows = []
    for bucket, part in delisted.dropna(subset=["bucket"]).groupby("bucket", sort=True):
        in_fmp = int(part["fmp_security_id"].notna().sum())
        rows.append({"bucket": bucket, "delisted_with_bars": len(part), "also_in_fmp": in_fmp, "new_vs_fmp": len(part) - in_fmp})
    return rows


def build_store(data_root: Path, fmp_securities: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    """
    Build ``<data_root>/store`` from the universe and the raw history files.

    Args:
        data_root: EODHD data root.
        fmp_securities: FMP vintage securities table; when given, each security gets its ``fmp_security_id``.

    Returns:
        Build summary (also written to ``_meta/build_info.json``).

    Raises:
        FileNotFoundError: The universe is missing.
    """
    universe = load_universe(data_root)
    store = data_root / config.STORE_DIR
    if store.exists():
        shutil.rmtree(store)
    staging = store / "_staging_prices"
    staging.mkdir(parents=True)
    writers: dict[int, pq.ParquetWriter] = {}
    split_parts: list[pd.DataFrame] = []
    dividend_parts: list[pd.DataFrame] = []
    ranges: dict[int, tuple[Any, Any, int]] = {}
    restricted = limited = 0
    try:
        for sid, code in zip(universe["security_id"].astype(int), universe["eodhd_code"]):
            payload = read_history(data_root, sid)
            if payload is None or payload.get("code") != code:
                continue
            restricted += bool(payload.get("events_restricted"))
            limited += bool(payload.get("plan_limited"))
            splits = splits_frame(payload.get("splits") or [])
            bars = bars_frame(payload.get("eod") or [], splits, config.VOLUME_SPLIT_ADJUSTED)
            if not splits.empty:
                split_parts.append(splits.assign(security_id=sid))
            dividends = dividends_frame(payload.get("dividends") or [], splits)
            if not dividends.empty:
                dividend_parts.append(dividends.assign(security_id=sid))
            if bars.empty:
                continue
            ranges[sid] = (bars["date"].iloc[0], bars["date"].iloc[-1], len(bars))
            bars.insert(0, "security_id", sid)
            bucket = sid % config.PRICE_BUCKETS
            if bucket not in writers:
                writers[bucket] = pq.ParquetWriter(staging / f"{bucket:02d}.parquet", _PRICE_SCHEMA)
            writers[bucket].write_table(pa.Table.from_pandas(bars[_PRICE_COLUMNS], schema=_PRICE_SCHEMA, preserve_index=False))
    finally:
        for writer in writers.values():
            writer.close()
    target = store / config.PRICES_DIR
    target.mkdir(parents=True, exist_ok=True)
    for bucket in sorted(writers):
        frame = pq.read_table(staging / f"{bucket:02d}.parquet").to_pandas().sort_values(["security_id", "date"])
        pq.write_table(pa.Table.from_pandas(frame, schema=_PRICE_SCHEMA, preserve_index=False),
                       target / f"bucket={bucket:02d}.parquet", row_group_size=200_000)
    shutil.rmtree(staging)

    securities = _securities(universe, ranges)
    securities["fmp_security_id"] = (map_to_fmp(securities, fmp_securities) if fmp_securities is not None
                                     else pd.array([pd.NA] * len(securities), dtype="Int64"))
    _write(securities, store / config.SECURITIES)
    splits_all = _events(split_parts, ["security_id", "date", "numerator", "denominator"])
    dividends_all = _events(dividend_parts, ["security_id", "date", "amount", "kind"])
    _write(splits_all, store / config.SPLITS)
    _write(dividends_all, store / config.DIVIDENDS)

    info: dict[str, Any] = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe": len(universe), "securities_with_bars": len(ranges), "price_rows": sum(r[2] for r in ranges.values()),
        "delisted_with_bars": int((~securities["is_actively_trading"] & securities["last_price_date"].notna()).sum()),
        "splits": len(splits_all), "dividends": len(dividends_all),
        "mapped_to_fmp": int(securities["fmp_security_id"].notna().sum()),
        "plan_limited_files": limited, "events_restricted_files": restricted,
        "coverage_by_last_trade_year": coverage_by_year(securities) if fmp_securities is not None else [],
    }
    meta = store / config.BUILD_INFO
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(info, indent=2, sort_keys=True, default=str), encoding="utf-8")
    if limited or restricted:
        _logger.warning("%d files have a plan-cut history and %d lack splits/dividends: this store is a dry run only", limited, restricted)
    _logger.info("EODHD store built: %d securities with bars of %d, %d price rows, %d mapped to FMP", info["securities_with_bars"],
                 info["universe"], info["price_rows"], info["mapped_to_fmp"])
    return info


def _securities(universe: pd.DataFrame, ranges: dict[int, tuple[Any, Any, int]]) -> pd.DataFrame:
    """Reference table in the FMP vintage's column names (P24 ``load_securities`` reads it as it is)."""
    ids = universe["security_id"].astype(int)
    first = ids.map({sid: r[0] for sid, r in ranges.items()}).astype(object).where(ids.isin(list(ranges)), None)
    last = ids.map({sid: r[1] for sid, r in ranges.items()}).astype(object).where(ids.isin(list(ranges)), None)
    counts = ids.map({sid: r[2] for sid, r in ranges.items()}).fillna(0)
    listed = universe["listed"].astype(bool)
    return pd.DataFrame({
        "security_id": ids.astype("int64"),
        "symbol": universe["symbol"],
        "name": universe["name"],
        "exchange": universe["exchange"],
        "instrument_type": universe["eodhd_type"].map(config.INSTRUMENT_TYPES).fillna("other"),
        "first_price_date": first,
        "last_price_date": last,
        "delisted_date": last.where(~listed, None),
        "is_actively_trading": listed,
        "valid_from": None,
        "valid_to": None,
        "identity_uncertain": False,
        "isin": universe["isin"],
        "eodhd_code": universe["eodhd_code"],
        "eodhd_type": universe["eodhd_type"],
        "bars": counts.astype("int64"),
    }).sort_values("security_id").reset_index(drop=True)


def _events(parts: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame({column: pd.Series([], dtype="int64" if column == "security_id" else "object") for column in columns})
    return pd.concat(parts, ignore_index=True)[columns].sort_values(["security_id", "date"]).reset_index(drop=True)
