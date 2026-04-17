"""Append-only JSONL signal log and simple deduplication store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from src.notification.logger import setup_logger
from src.strategy_pack.models import PackSignal

_logger = setup_logger(__name__)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    rows_list = list(rows)
    with path.open("a", encoding="utf-8") as f:
        for row in rows_list:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    _logger.info("Appended %d JSONL records to %s", len(rows_list), path)


def append_signals(path: Path, signals: List[PackSignal]) -> None:
    append_jsonl(path, [s.to_jsonl_dict() for s in signals])


class DedupStore:
    """Tracks idempotency keys already notified (JSON sidecar)."""

    def __init__(self, path: Path):
        self.path = path
        self._keys: Set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._keys = set(data)
            elif isinstance(data, dict) and "keys" in data:
                self._keys = set(data["keys"])
        except (json.JSONDecodeError, OSError) as e:
            _logger.warning("DedupStore load failed (%s), starting empty: %s", self.path, e)

    def should_notify(self, key: str) -> bool:
        return key not in self._keys

    def mark_sent(self, key: str) -> None:
        self._keys.add(key)
        self._persist()

    def mark_many(self, keys: Iterable[str]) -> None:
        self._keys.update(keys)
        self._persist()

    def _persist(self) -> None:
        ensure_dir(self.path.parent)
        payload = {"keys": sorted(self._keys)}
        self.path.write_text(json.dumps(payload, indent=0), encoding="utf-8")
