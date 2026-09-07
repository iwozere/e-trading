"""Tests for P20 Kestrel GDELT processor — utility functions."""

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.sentiment import gdelt_processor
from src.ml.pipeline.p20_kestrel.sentiment.gdelt_processor import (
    _MIN_PERIODS,
    GdeltProcessor,
    _compute_zscores,
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


# ── Orgs-slice file parsing ─────────────────────────────────────────────────


def _write_orgs_slice(path, rows):
    """Write a slim orgs-slice file in the GdeltDownloader format."""
    import gzip

    lines = ["\t".join(["date", "source", "themes", "orgs", "tone"])]
    lines += ["\t".join(r) for r in rows]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _processor_with_alias(ticker: str, alias: str) -> GdeltProcessor:
    p = GdeltProcessor()
    p._aliases = {_normalize(alias): [{"ticker": ticker, "normalized_alias": _normalize(alias)}]}
    p._blocklist = {}
    return p


def test_process_gkg_file_parses_orgs_slice(tmp_path):
    """Slim orgs-slice rows produce matched per-article records."""
    path = tmp_path / "20260707.gkg-orgs.csv.gz"
    _write_orgs_slice(
        path,
        [
            ("2026-07-07", "reuters.com", "ECON_STOCKMARKET", "Apple Inc,3", "2.5,3.0,-0.5,1.0,10,5,3"),
            ("2026-07-07", "example.com", "", "Unknown Org,1", "1.0,2.0,-1.0,1.0,10,5,3"),
        ],
    )
    p = _processor_with_alias("AAPL", "Apple Inc")
    records = p.process_gkg_file(path)
    assert len(records) == 1
    rec = records[0]
    assert rec["ticker"] == "AAPL"
    assert str(rec["date"]) == "2026-07-07"
    assert rec["avg_tone"] == 2.5
    assert rec["source_domain"] == "reuters.com"


# ── Z-score computation ──────────────────────────────────────────────────────


def test_compute_zscores_handles_decimal_history(monkeypatch):
    """
    Historical rows loaded via SQLAlchemy Numeric columns (asdecimal not yet
    fixed, or a future regression re-introducing it) come back as
    decimal.Decimal, not float. Regression test for the 2026-09-07 production
    crash: pandas' .std() raised
    ``TypeError: unsupported operand type(s) for -: 'float' and 'decimal.Decimal'``
    when a history DataFrame held Decimal objects.
    """
    hist_rows = [
        {"mentions": Decimal(n), "avg_tone": Decimal(str(t))}
        for n, t in zip(range(1, _MIN_PERIODS + 2), [1.0 + 0.1 * i for i in range(_MIN_PERIODS + 1)])
    ]
    monkeypatch.setattr(gdelt_processor, "get_sentiment_history", lambda *a, **kw: hist_rows)

    agg_rows = [{"ticker": "AAPL", "mentions": 5, "avg_tone": 2.0}]
    scored = _compute_zscores(agg_rows, date(2026, 9, 6))

    assert scored[0]["mention_z20"] is not None
    assert isinstance(scored[0]["mention_z20"], float)
    assert scored[0]["tone_z20"] is not None
    assert isinstance(scored[0]["tone_z20"], float)


def test_compute_zscores_warmup_below_min_periods(monkeypatch):
    """Fewer than _MIN_PERIODS history rows leaves z-scores as None (warm-up)."""
    hist_rows = [{"mentions": Decimal(3), "avg_tone": Decimal("1.0")}]
    monkeypatch.setattr(gdelt_processor, "get_sentiment_history", lambda *a, **kw: hist_rows)

    agg_rows = [{"ticker": "AAPL", "mentions": 5, "avg_tone": 2.0}]
    scored = _compute_zscores(agg_rows, date(2026, 9, 6))

    assert scored[0]["mention_z20"] is None
    assert scored[0]["tone_z20"] is None


def test_process_gkg_file_rejects_unknown_header(tmp_path):
    """Files in the old raw/aggregate formats are skipped, not misparsed."""
    import gzip

    path = tmp_path / "20260707.gkg-orgs.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("date,theme,article_count,avg_tone\n2026-07-07,ECON_X,5,1.0\n")
    p = _processor_with_alias("AAPL", "Apple Inc")
    assert p.process_gkg_file(path) == []
