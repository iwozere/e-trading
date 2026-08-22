"""
P21 Momentum — Ledger and current-positions I/O (docs/pipeline-specification.md §3).

``_state/ledger.jsonl`` is append-only — never truncated, never rewritten,
one line per simulated trade, ever. ``_state/current_positions.json`` is the
one file in ``_state/`` that *does* get overwritten each run, since it is a
mutable pointer ("what do I hold right now"), not a log.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import List

from src.ml.pipeline.p21_momentum.config import CURRENT_POSITIONS_PATH, LEDGER_PATH
from src.ml.pipeline.p21_momentum.schemas import LedgerEntry, Position
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def append_ledger_entries(entries: List[LedgerEntry], path: Path = LEDGER_PATH) -> None:
    """
    Append entries to _state/ledger.jsonl. Never truncates or rewrites existing lines.

    Args:
        entries: New trade entries for this run.
        path: Path to _state/ledger.jsonl.
    """
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict(), sort_keys=True))
            f.write("\n")
    _logger.info("Appended %d ledger entries to %s", len(entries), path)


def read_ledger_entries_for_month(year: int, month: int, path: Path = LEDGER_PATH) -> List[LedgerEntry]:
    """
    Filter the ledger for entries whose ``ts`` falls within (year, month).

    Per spec §3: "A given month's trades are not duplicated into that
    month's dated folder — they are the rows of _state/ledger.jsonl with a
    matching ts. Filter by date rather than maintaining two copies."

    Args:
        year, month: Target month.
        path: Path to _state/ledger.jsonl.

    Returns:
        List of LedgerEntry, in file order. Empty list if the file does not
        exist yet (first-ever run).
    """
    if not path.exists():
        return []
    entries: List[LedgerEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ts = datetime.fromisoformat(d["ts"])
            if ts.year == year and ts.month == month:
                entries.append(LedgerEntry.from_dict(d))
    return entries


def read_all_ledger_entries(path: Path = LEDGER_PATH) -> List[LedgerEntry]:
    """Read the entire ledger, in file order. Empty list if the file does not exist."""
    if not path.exists():
        return []
    entries: List[LedgerEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(LedgerEntry.from_dict(json.loads(line)))
    return entries


def write_current_positions(
    positions: List[Position],
    as_of: date,
    track: str,
    nav_total: float,
    cash: float,
    sleeve_market_value: float,
    regime_scalar: float,
    path: Path = CURRENT_POSITIONS_PATH,
) -> None:
    """
    Overwrite _state/current_positions.json with this run's holdings.

    Unlike the ledger, this file IS overwritten every run that changes
    positions — it is a mutable pointer to "what do I hold right now", not
    a log (spec §3).
    """
    payload = {
        "as_of": as_of.isoformat(),
        "track": track,
        "nav_total": nav_total,
        "cash": cash,
        "sleeve_market_value": sleeve_market_value,
        "regime_scalar": regime_scalar,
        "positions": [p.to_dict() for p in positions],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    _logger.info("Wrote %d current positions to %s", len(positions), path)


def read_current_positions(path: Path = CURRENT_POSITIONS_PATH) -> List[Position]:
    """
    Read _state/current_positions.json.

    Returns:
        List of Position. Empty list if the file does not exist yet (no
        positions held, e.g. the very first run).
    """
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [Position.from_dict(p) for p in payload.get("positions", [])]
