"""
P21 Momentum — Manual exclusion list (F5, docs/pipeline-specification.md §5).

Reads ``config/pipeline/p21_exclusions.json``. Read-only — the pipeline
never writes to this file; it is operator-maintained input, checked into
git, same convention as P20's ``config/pipeline/activists.json``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Set

from src.ml.pipeline.p21_momentum.config import EXCLUSIONS_PATH
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def load_exclusions(path: Path = EXCLUSIONS_PATH, as_of: date | None = None) -> Set[str]:
    """
    Return the set of currently-excluded tickers, skipping expired entries.

    Args:
        path: Path to config/pipeline/p21_exclusions.json.
        as_of: Date to evaluate expiry against (defaults to today).

    Returns:
        Set of excluded ticker symbols. An entry with no "expires" field
        never expires. Missing file -> empty set (valid starting state, per
        spec §5: "empty {"exclusions": []} is a valid starting state").
    """
    if not path.exists():
        _logger.info("No exclusions file at %s — treating as empty", path)
        return set()

    today = as_of or date.today()
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        _logger.exception("Failed to read exclusions file %s — treating as empty", path)
        return set()

    excluded: Set[str] = set()
    for entry in payload.get("exclusions", []):
        ticker = entry.get("ticker")
        if not ticker:
            continue
        expires_str = entry.get("expires")
        if expires_str:
            try:
                expires = date.fromisoformat(expires_str)
                if today > expires:
                    continue  # expired, ignore
            except ValueError:
                _logger.warning("Malformed 'expires' date for %s: %r — treating as non-expiring", ticker, expires_str)
        excluded.add(ticker)

    _logger.info("Loaded %d active exclusions from %s", len(excluded), path)
    return excluded
