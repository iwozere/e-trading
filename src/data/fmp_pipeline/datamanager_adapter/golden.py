"""
Golden-output regression snapshot for ``DataManager.get_ohlcv`` (plan §8, WP-8).

Instead of auditing the ~147 calling files, hash what ``get_ohlcv`` returns for every symbol and timeframe in the
legacy cache, store the hashes, and compare after every change: with the FMP adapter **disabled**, any difference
is a bug; with it enabled, differences are expected only for symbols the FMP vintage serves.

The snapshot runs **offline**: the DataManager gets no providers, so a gap in the cache is returned as-is instead
of being downloaded (and written back into the cache). Each symbol is requested over its whole cached range.
The snapshot file holds hashes and row counts only, no prices; it lives under ``<data_root>/reports/golden/``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.data.fmp_pipeline.fsutil import atomic_write_bytes
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

#: The whole range of any cached file; with no providers, DataManager returns what the cache has.
RANGE_START = datetime(1970, 1, 1)
RANGE_END = datetime(2100, 1, 1)
VERSION = 1

#: ``(symbol, timeframe) -> frame``; raises when nothing can be returned.
Fetch = Callable[[str, str], pd.DataFrame]


def frame_hash(df: pd.DataFrame) -> str:
    """
    Hash everything a caller can observe of a frame: columns, dtypes, index type and name, index and values.

    Args:
        df: An OHLCV frame.

    Returns:
        sha256 hex digest.
    """
    digest = hashlib.sha256()
    digest.update(repr([str(c) for c in df.columns]).encode("utf-8"))
    digest.update(repr([str(t) for t in df.dtypes]).encode("utf-8"))
    digest.update(f"{type(df.index).__name__}|{df.index.dtype}|{df.index.name}".encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def cached_series(ohlcv_dir: Path) -> list[tuple[str, str]]:
    """
    List ``(symbol, timeframe)`` pairs present in the legacy cache.

    Args:
        ohlcv_dir: ``<DATA_CACHE_DIR>/ohlcv``.

    Returns:
        Pairs whose directory holds at least one ``<year>.csv.gz``, sorted.
    """
    pairs = []
    for symbol_dir in sorted(p for p in ohlcv_dir.iterdir() if p.is_dir()) if ohlcv_dir.is_dir() else []:
        for tf_dir in sorted(p for p in symbol_dir.iterdir() if p.is_dir()):
            if any(tf_dir.glob("*.csv.gz")):
                pairs.append((symbol_dir.name, tf_dir.name))
    return pairs


@dataclass
class Snapshot:
    """Hashes per ``symbol/timeframe``: ``{"hash", "rows"}`` or ``{"error"}``."""

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Write the snapshot as JSON (atomically)."""
        payload = {"version": VERSION, "meta": self.meta, "entries": dict(sorted(self.entries.items()))}
        atomic_write_bytes(path, json.dumps(payload, indent=1).encode("utf-8"))

    @classmethod
    def load(cls, path: Path) -> "Snapshot":
        """Read a snapshot written by ``save``."""
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != VERSION:
            raise ValueError(f"{path}: unsupported snapshot version {data.get('version')!r}")
        return cls(dict(data["entries"]), dict(data.get("meta", {})))


def capture(pairs: Iterable[tuple[str, str]], fetch: Fetch, meta: Optional[dict[str, Any]] = None) -> Snapshot:
    """
    Hash ``fetch(symbol, timeframe)`` for every pair.

    Args:
        pairs: ``(symbol, timeframe)`` to request.
        fetch: Returns the frame (``DataManager.get_ohlcv`` over the whole range, offline).
        meta: Free-form context stored with the snapshot (adapter state, cache dir).

    Returns:
        The snapshot; a pair whose fetch raises is recorded with the exception type and message.
    """
    snapshot = Snapshot(meta=dict(meta or {}))
    for i, (symbol, timeframe) in enumerate(pairs, 1):
        key = f"{symbol}/{timeframe}"
        try:
            df = fetch(symbol, timeframe)
            snapshot.entries[key] = {"hash": frame_hash(df), "rows": int(len(df))}
        except Exception as exc:  # noqa: BLE001 - every failure is part of the observed behaviour
            snapshot.entries[key] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        if i % 100 == 0:
            _logger.info("Golden snapshot: %d series hashed", i)
    return snapshot


@dataclass(frozen=True)
class Comparison:
    """Differences between a baseline and a new snapshot."""

    changed: list[str]
    missing: list[str]
    added: list[str]
    same: int

    @property
    def clean(self) -> bool:
        """True when nothing changed, disappeared or appeared."""
        return not (self.changed or self.missing or self.added)


def compare(baseline: Snapshot, current: Snapshot) -> Comparison:
    """Return which series changed, disappeared or appeared since ``baseline``."""
    base, cur = baseline.entries, current.entries
    changed = sorted(k for k in base.keys() & cur.keys() if base[k] != cur[k])
    return Comparison(changed, sorted(base.keys() - cur.keys()), sorted(cur.keys() - base.keys()),
                      len(base.keys() & cur.keys()) - len(changed))


def offline_fetch(data_manager: Any) -> Fetch:
    """
    Turn a ``DataManager`` into an offline fetch: no providers, so nothing is downloaded or written back.

    Args:
        data_manager: A ``DataManager`` instance (modified in place).

    Returns:
        ``fetch(symbol, timeframe)`` calling ``get_ohlcv`` over ``RANGE_START`` … ``RANGE_END``.
    """
    data_manager.provider_selector.get_provider_with_failover = lambda *args, **kwargs: []

    def fetch(symbol: str, timeframe: str) -> pd.DataFrame:
        return data_manager.get_ohlcv(symbol, timeframe, RANGE_START, RANGE_END)

    return fetch
