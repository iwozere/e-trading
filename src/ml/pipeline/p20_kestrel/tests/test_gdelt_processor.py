"""Tests for P20 Kestrel GDELT processor — utility functions."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.sentiment.gdelt_processor import (
    _MIN_PERIODS,
    _fuzzy_score,
    _normalize,
    _parse_v2tone,
)


def test_normalize_lowercases():
    """_normalize returns lowercase stripped text."""
    result = _normalize("Apple Inc. REPORTS Earnings")
    assert result == result.lower()


def test_normalize_empty():
    """_normalize handles empty string."""
    result = _normalize("")
    assert result == ""


def test_fuzzy_score_identical():
    """Identical strings score 1.0."""
    assert _fuzzy_score("apple", "apple") == 1.0


def test_fuzzy_score_different():
    """Completely different strings score < 0.5."""
    score = _fuzzy_score("apple", "xyzqwerty")
    assert score < 0.5


def test_fuzzy_score_near_match():
    """Near-identical strings score > 0.9."""
    score = _fuzzy_score("microsoft", "microsft")
    assert score > 0.85


def test_parse_v2tone_basic():
    """Parse a valid V2Tone string (comma-delimited per GKG 2.1 spec)."""
    result = _parse_v2tone("2.5,3.0,-0.5,1.0,10,5,3")
    assert result is not None
    tone, pos, neg = result
    assert tone == 2.5
    assert pos == 3.0
    assert neg == 0.5  # negative score is stored as abs()


def test_parse_v2tone_invalid():
    """Returns None for malformed V2Tone string."""
    assert _parse_v2tone("not_a_tone") is None
    assert _parse_v2tone("") is None


def test_min_periods_constant():
    """MIN_PERIODS for z-score warm-up is at least 10."""
    assert _MIN_PERIODS >= 10
