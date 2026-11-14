"""
Fill ``<data_root>/raw``: bulk files with a paid key, or a small sample with the public key.

Both paths leave one file per table with the same CSV columns, so the build does not care which one ran.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from src.data.sharadar_pipeline import config
from src.data.sharadar_pipeline.client import FreeTierExceeded, SharadarClient, SharadarError
from src.data.sharadar_pipeline.manifest import Manifest
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

#: Earliest date asked of the API; the default would be one year ago.
HISTORY_START = "1990-01-01"


def _save_csv(frame: pd.DataFrame, path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = frame.to_csv(index=False).encode("utf-8")
    path.write_bytes(data)
    return len(data), hashlib.sha256(data).hexdigest()


def query_tickers(client: SharadarClient) -> pd.DataFrame:
    """
    Read the whole ``tickers`` table with paged queries.

    Paging the unfiltered table loses and repeats rows (the default order is not stable across pages), so each
    ``table`` value is read separately in ``permaticker`` order and exact duplicates are dropped.
    """
    parts = [client.query("tickers", {"table": table, "sort": "permaticker.asc"}) for table in config.TICKER_TABLES]
    out = pd.concat([part for part in parts if len(part)], ignore_index=True)
    return out.drop_duplicates().reset_index(drop=True)


def download_sample(client: SharadarClient, data_root: Path, tickers: Iterable[str] = config.SAMPLE_TICKERS) -> Manifest:
    """
    Download the free sample: the whole ``tickers`` and ``descriptions`` tables and, for each sample ticker the
    free tier serves, its prices, actions, fundamentals and daily metrics.

    Args:
        client: Client holding the public sample key (a paid key works too).
        data_root: Sharadar data root.
        tickers: Tickers to try; those outside the free tier are skipped.

    Returns:
        The saved manifest (``mode`` = ``sample``).
    """
    raw = data_root / config.RAW_DIR
    manifest = Manifest.load(data_root)
    manifest.mode = "sample"
    whole = {"descriptions": client.query("descriptions"), "tickers": query_tickers(client)}
    for table, frame in whole.items():
        size, digest = _save_csv(frame, raw / f"{table}.csv")
        manifest.record(table, f"{table}.csv", size, digest, len(frame))
        _logger.info("Sample %s: %d rows", table, len(frame))
    served: list[str] = []
    parts: dict[str, list[pd.DataFrame]] = {table: [] for table in config.SAMPLE_TICKER_TABLES}
    for ticker in tickers:
        try:
            pulled = {table: client.query(table, {"ticker": ticker, "from": HISTORY_START}) for table in config.SAMPLE_TICKER_TABLES}
        except FreeTierExceeded:
            _logger.info("Sample ticker %s is outside the free tier, skipped", ticker)
            continue
        served.append(ticker)
        for table, frame in pulled.items():
            if len(frame):
                parts[table].append(frame)
    if not served:
        raise SharadarError("the free tier served none of the sample tickers")
    for table, frames in parts.items():
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        size, digest = _save_csv(frame, raw / f"{table}.csv")
        manifest.record(table, f"{table}.csv", size, digest, len(frame))
        _logger.info("Sample %s: %d rows for %d tickers", table, len(frame), len(served))
    manifest.save(data_root)
    return manifest


def bulk_status(client: SharadarClient, tables: Iterable[str] = config.BULK_TABLES) -> dict[str, Any]:
    """Return the bulk status of each table; a failing table maps to its error text."""
    out: dict[str, Any] = {}
    for table in tables:
        try:
            out[table] = client.bulk_status(table)
        except SharadarError as exc:
            out[table] = {"error": str(exc)}
    return out


def download_bulk(client: SharadarClient, data_root: Path, tables: Iterable[str] = config.BULK_TABLES, force: bool = False,
                  subscription_start: Optional[str] = None) -> Manifest:
    """
    Download the full-history bulk file of each table (paid key).

    Tables already in the manifest whose file still has the recorded size are skipped unless ``force``; the manifest
    is saved after every table, so an interrupted run resumes.

    Args:
        client: Client holding the paid key.
        data_root: Sharadar data root.
        tables: Tables to download.
        force: Download again even if present.
        subscription_start: ISO date the subscription began; default today on the first paid download.

    Returns:
        The saved manifest (``mode`` = ``paid``).
    """
    raw = data_root / config.RAW_DIR
    manifest = Manifest.load(data_root)
    if manifest.mode != "paid":
        manifest = Manifest(mode="paid")
    manifest.subscription_start = (subscription_start or manifest.subscription_start
                                   or datetime.now(timezone.utc).date().isoformat())
    manifest.save(data_root)
    for table in tables:
        name = f"{table}.csv.zip"
        known = manifest.files.get(table)
        if not force and known is not None and (raw / name).exists() and (raw / name).stat().st_size == known.size:
            _logger.info("Bulk %s already downloaded, skipped", table)
            continue
        done = client.download_bulk(table, raw / name)
        manifest.record(table, name, done.size, done.sha256)
        manifest.save(data_root)
    return manifest
