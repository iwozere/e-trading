"""
The download universe: EODHD's live and delisted US symbol lists, filtered, each code given a stable security id.

EODHD has no permanent id. A delisted symbol whose ticker was later reused is renamed ``ABC_old`` (``ABC_old1``, …),
so the code is unique at the time of the list. Ids are assigned once per code and kept in ``<data_root>/ids.json``;
running ``universe`` again only adds codes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.eodhd_pipeline import config
from src.data.eodhd_pipeline.client import EodhdClient, EodhdError
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

UNIVERSE_COLUMNS = ["security_id", "eodhd_code", "symbol", "name", "exchange", "eodhd_type", "isin", "currency", "listed"]


def base_symbol(code: str) -> str:
    """Return the ticker as traded: EODHD's ``_old`` / ``_oldN`` reuse suffix removed."""
    return code.split("_", 1)[0].upper()


def listing_frame(live: pd.DataFrame, delisted: pd.DataFrame, types: Iterable[str] = config.DEFAULT_TYPES,
                  exchanges: Iterable[str] | None = config.MAIN_EXCHANGES) -> pd.DataFrame:
    """
    Combine the two symbol lists into one row per code.

    Args:
        live: ``exchange-symbol-list`` of trading symbols.
        delisted: The same list with ``delisted=1``.
        types: EODHD ``Type`` values to keep.
        exchanges: ``Exchange`` values to keep; None keeps every venue.

    Returns:
        ``eodhd_code``, ``symbol``, ``name``, ``exchange``, ``eodhd_type``, ``isin``, ``currency``, ``listed``;
        delisted codes first, then live, each sorted by code. A code on both lists counts as live.
    """
    parts = []
    for frame, listed in ((delisted, False), (live, True)):
        if frame.empty:
            continue
        part = frame.reindex(columns=["Code", "Name", "Exchange", "Type", "Isin", "Currency"]).copy()
        part["listed"] = listed
        parts.append(part)
    columns = UNIVERSE_COLUMNS[1:]
    if not parts:
        return pd.DataFrame(columns=columns)
    rows = pd.concat(parts, ignore_index=True).dropna(subset=["Code"])
    rows = rows[rows["Type"].isin(list(types))]
    if exchanges is not None:
        rows = rows[rows["Exchange"].isin(list(exchanges))]
    rows = rows.sort_values(["Code", "listed"]).drop_duplicates("Code", keep="last")
    out = pd.DataFrame({
        "eodhd_code": rows["Code"].astype(str),
        "symbol": rows["Code"].astype(str).map(base_symbol),
        "name": rows["Name"].where(rows["Name"].notna(), None),
        "exchange": rows["Exchange"],
        "eodhd_type": rows["Type"],
        "isin": rows["Isin"].where(rows["Isin"].notna() & (rows["Isin"] != ""), None),
        "currency": rows["Currency"],
        "listed": rows["listed"].astype(bool),
    })
    return out.sort_values(["listed", "eodhd_code"]).reset_index(drop=True)[columns]


class IdRegistry:
    """Code -> security id, persisted in ``<data_root>/ids.json``; ids are never reused or changed."""

    def __init__(self, ids: dict[str, int]) -> None:
        self.ids = dict(ids)

    @classmethod
    def load(cls, data_root: Path) -> "IdRegistry":
        """Read the registry (empty if the file does not exist)."""
        path = data_root / config.IDS_FILE
        return cls(json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})

    def save(self, data_root: Path) -> None:
        """Write the registry atomically."""
        path = data_root / config.IDS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.ids, indent=0, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def assign(self, codes: Iterable[str]) -> list[int]:
        """Return the id of each code, giving new codes the next free ids in the order given."""
        next_id = max(self.ids.values(), default=config.ID_OFFSET) + 1
        out = []
        for code in codes:
            if code not in self.ids:
                self.ids[code] = next_id
                next_id += 1
            out.append(self.ids[code])
        return out


def build_universe(client: EodhdClient, data_root: Path, types: Iterable[str] = config.DEFAULT_TYPES,
                   exchanges: Iterable[str] | None = config.MAIN_EXCHANGES) -> pd.DataFrame:
    """
    Download both symbol lists (two API calls), keep them under ``raw/symbols`` and write ``universe.parquet``.

    Returns:
        The universe (``UNIVERSE_COLUMNS``).

    Raises:
        EodhdError: EODHD returned no delisted symbols.
    """
    delisted = client.symbols(delisted=True)
    live = client.symbols()
    if delisted.empty:
        raise EodhdError("EODHD returned no delisted symbols")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    symbols_dir = data_root / config.SYMBOLS_DIR
    symbols_dir.mkdir(parents=True, exist_ok=True)
    delisted.to_json(symbols_dir / f"delisted_{stamp}.json", orient="records")
    live.to_json(symbols_dir / f"live_{stamp}.json", orient="records")
    universe = listing_frame(live, delisted, types, exchanges)
    registry = IdRegistry.load(data_root)
    universe.insert(0, "security_id", registry.assign(universe["eodhd_code"]))
    registry.save(data_root)
    universe.to_parquet(data_root / config.UNIVERSE_FILE, index=False)
    _logger.info("Universe: %d codes (%d delisted, %d live) from %d delisted and %d live symbols", len(universe),
                 int((~universe["listed"]).sum()), int(universe["listed"].sum()), len(delisted), len(live))
    return universe


def load_universe(data_root: Path) -> pd.DataFrame:
    """
    Read ``universe.parquet``.

    Raises:
        FileNotFoundError: ``universe`` has not been run.
    """
    path = data_root / config.UNIVERSE_FILE
    if not path.exists():
        raise FileNotFoundError(f"no universe under {data_root}; run the universe command first")
    return pd.read_parquet(path)
