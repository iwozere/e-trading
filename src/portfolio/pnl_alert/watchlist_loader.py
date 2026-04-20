"""
Watchlist YAML loader.

Parses and validates the user-editable watchlist for the PnL alert pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass(frozen=True)
class WatchlistEntry:
    """
    One manually tracked position.

    Attributes:
        symbol: Ticker symbol (uppercased on load).
        avg_price: Average buy price in USD (must be > 0).
        notes: Optional free-text note.
    """

    symbol: str
    avg_price: float
    notes: Optional[str] = None


class WatchlistValidationError(ValueError):
    """Raised when the watchlist YAML fails schema validation."""


def load_watchlist(path: str) -> List[WatchlistEntry]:
    """
    Load and validate a watchlist YAML file.

    The expected schema is::

        entries:
          - symbol: NVDA
            avg_price: 120.00
          - symbol: AAPL
            avg_price: 150.00
            notes: "optional"

    Args:
        path: Path to the YAML file, relative to the current working directory
            or absolute.

    Returns:
        List of validated, de-duplicated `WatchlistEntry` objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        WatchlistValidationError: If the YAML is invalid or fails validation.
    """
    wl_path = Path(path)
    if not wl_path.exists():
        raise FileNotFoundError(f"Watchlist file not found: {wl_path}")

    try:
        with wl_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise WatchlistValidationError(
            f"Watchlist YAML is not parseable ({wl_path}): {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise WatchlistValidationError(
            f"Watchlist must be a mapping at the top level, got {type(raw).__name__}"
        )

    raw_entries = raw.get("entries") or []
    if not isinstance(raw_entries, list):
        raise WatchlistValidationError(
            f"Watchlist 'entries' must be a list, got {type(raw_entries).__name__}"
        )

    entries: List[WatchlistEntry] = []
    seen_symbols: set[str] = set()

    for idx, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            raise WatchlistValidationError(
                f"Watchlist entry #{idx} must be a mapping, got {type(item).__name__}"
            )

        symbol_raw = item.get("symbol")
        if not symbol_raw or not isinstance(symbol_raw, str):
            raise WatchlistValidationError(
                f"Watchlist entry #{idx} is missing a non-empty 'symbol' string"
            )
        symbol = symbol_raw.strip().upper()

        avg_price_raw = item.get("avg_price")
        if not isinstance(avg_price_raw, (int, float)):
            raise WatchlistValidationError(
                f"Watchlist entry {symbol!r} has a non-numeric avg_price: {avg_price_raw!r}"
            )
        avg_price = float(avg_price_raw)
        if avg_price <= 0:
            raise WatchlistValidationError(
                f"Watchlist entry {symbol!r} has a non-positive avg_price: {avg_price}"
            )

        if symbol in seen_symbols:
            raise WatchlistValidationError(
                f"Watchlist has duplicate entries for symbol {symbol!r}"
            )
        seen_symbols.add(symbol)

        notes_raw = item.get("notes")
        notes = str(notes_raw) if notes_raw is not None else None

        entries.append(WatchlistEntry(symbol=symbol, avg_price=avg_price, notes=notes))

    _logger.info("Loaded %d watchlist entries from %s", len(entries), wl_path)
    return entries
