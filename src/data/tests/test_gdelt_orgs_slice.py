"""Tests for the GDELT downloader's slim per-article orgs slice (P20 sentiment input)."""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.downloader.gdelt_downloader import _GKG_COLS, _ORGS_SLICE_COLS, _extract_orgs_slice


def _raw_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a raw GKG frame with all 27 columns, filling given fields."""
    base = {col: "" for col in _GKG_COLS}
    return pd.DataFrame([{**base, **row} for row in rows])


def test_extract_orgs_slice_keeps_org_articles_only():
    raw = _raw_frame(
        [
            {
                "SourceCommonName": "reuters.com",
                "V2Themes": "ECON_STOCKMARKET",
                "V2Organizations": "Apple Inc,3",
                "V2Tone": "2.5,3.0,-0.5,1.0,10,5,3",
            },
            {"V2Organizations": "", "V2Tone": "1.0,2.0,-1.0,1.0,10,5,3"},  # no orgs → dropped
            {"V2Organizations": "Acme Corp,1", "V2Tone": ""},  # no tone → dropped
        ]
    )
    slim = _extract_orgs_slice(raw, datetime(2026, 7, 7))
    assert list(slim.columns) == _ORGS_SLICE_COLS
    assert len(slim) == 1
    assert slim.iloc[0]["date"] == "2026-07-07"
    assert slim.iloc[0]["orgs"] == "Apple Inc,3"
    assert slim.iloc[0]["source"] == "reuters.com"


def test_extract_orgs_slice_empty_input():
    slim = _extract_orgs_slice(pd.DataFrame(), datetime(2026, 7, 7))
    assert slim.empty
    assert list(slim.columns) == _ORGS_SLICE_COLS
