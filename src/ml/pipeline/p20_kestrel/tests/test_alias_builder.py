"""Tests for P20 Kestrel alias builder — normalize_alias."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.sentiment.alias_builder import normalize_alias


def test_normalize_alias_strips_inc():
    """Strips 'Inc.' suffix."""
    assert normalize_alias("Apple Inc.") == "apple"


def test_normalize_alias_strips_corp():
    """Strips 'Corp' suffix."""
    assert normalize_alias("Tesla Corp") == "tesla"


def test_normalize_alias_strips_corporation():
    """Strips 'Corporation' suffix."""
    assert normalize_alias("NVIDIA Corporation") == "nvidia"


def test_normalize_alias_lowercases():
    """Always returns lowercase."""
    assert normalize_alias("MICROSOFT") == "microsoft"


def test_normalize_alias_collapses_whitespace():
    """Collapses internal whitespace and strips edges."""
    assert normalize_alias("  Devon  Energy  Corp  ") == "devon energy"


def test_normalize_alias_handles_empty():
    """Returns empty string for empty input."""
    assert normalize_alias("") == ""
    assert normalize_alias("   ") == ""


def test_normalize_alias_strips_ltd():
    """Strips 'Ltd' and 'Ltd.' suffixes."""
    assert normalize_alias("Acme Ltd") == "acme"
    assert normalize_alias("Foo Ltd.") == "foo"


def test_normalize_alias_ticker_unchanged():
    """Ticker symbols (no legal suffixes) pass through."""
    assert normalize_alias("AAPL") == "aapl"


def test_normalize_alias_multi_word():
    """Multi-word names without suffixes are preserved."""
    result = normalize_alias("Meta Platforms Inc")
    assert "meta" in result
    assert "platforms" in result
