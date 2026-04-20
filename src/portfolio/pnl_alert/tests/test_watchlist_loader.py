"""Unit tests for `watchlist_loader.load_watchlist`."""

from pathlib import Path

import pytest

from src.portfolio.pnl_alert.watchlist_loader import (
    WatchlistValidationError,
    load_watchlist,
)


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_valid_entries(tmp_path: Path):
    """A well-formed YAML produces the expected entries (uppercased symbols)."""
    yaml_path = _write(
        tmp_path / "wl.yaml",
        """
entries:
  - symbol: nvda
    avg_price: 120.00
  - symbol: AAPL
    avg_price: 150
    notes: long-term hold
""",
    )

    entries = load_watchlist(str(yaml_path))

    assert len(entries) == 2
    assert entries[0].symbol == "NVDA"
    assert entries[0].avg_price == 120.0
    assert entries[1].symbol == "AAPL"
    assert entries[1].avg_price == 150.0
    assert entries[1].notes == "long-term hold"


def test_empty_file_is_valid(tmp_path: Path):
    """An empty-entries YAML loads as an empty list."""
    yaml_path = _write(tmp_path / "wl.yaml", "entries: []\n")

    entries = load_watchlist(str(yaml_path))

    assert entries == []


def test_missing_file_raises(tmp_path: Path):
    """A non-existent path raises `FileNotFoundError`."""
    with pytest.raises(FileNotFoundError):
        load_watchlist(str(tmp_path / "does-not-exist.yaml"))


def test_duplicate_symbols_rejected(tmp_path: Path):
    """Duplicate symbols (case-insensitive) fail validation."""
    yaml_path = _write(
        tmp_path / "wl.yaml",
        """
entries:
  - symbol: NVDA
    avg_price: 120
  - symbol: nvda
    avg_price: 130
""",
    )

    with pytest.raises(WatchlistValidationError):
        load_watchlist(str(yaml_path))


def test_non_positive_avg_price_rejected(tmp_path: Path):
    """A non-positive `avg_price` is rejected."""
    yaml_path = _write(
        tmp_path / "wl.yaml",
        """
entries:
  - symbol: NVDA
    avg_price: 0
""",
    )

    with pytest.raises(WatchlistValidationError):
        load_watchlist(str(yaml_path))


def test_missing_symbol_rejected(tmp_path: Path):
    """A missing `symbol` field is rejected."""
    yaml_path = _write(
        tmp_path / "wl.yaml",
        """
entries:
  - avg_price: 100
""",
    )

    with pytest.raises(WatchlistValidationError):
        load_watchlist(str(yaml_path))


def test_non_mapping_top_level_rejected(tmp_path: Path):
    """A YAML that isn't a mapping at the top level is rejected."""
    yaml_path = _write(tmp_path / "wl.yaml", "- just a list\n")

    with pytest.raises(WatchlistValidationError):
        load_watchlist(str(yaml_path))
