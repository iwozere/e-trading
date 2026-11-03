"""Tests that GKG parsing keeps only the columns consumed downstream (memory guard)."""

import io
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.gdelt_downloader import (
    _GKG_COLS,
    _GKG_USED_COLS,
    GdeltDownloader,
    _aggregate_gkg,
    _extract_orgs_slice,
)


def _gkg_zip(rows: list[dict[str, str]]) -> bytes:
    """Build an in-memory GKG zip with the full 27-column tab-separated layout."""
    lines = ["\t".join(row.get(col, "x" * 50) for col in _GKG_COLS) for row in rows]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("20261030000000.gkg.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


def test_download_gkg_file_reads_only_used_columns(tmp_path: Path) -> None:
    payload = _gkg_zip(
        [
            {"SourceCommonName": "a.com", "V2Themes": "ECON_X,1;TAX_Y,5", "V2Organizations": "Apple,3", "V2Tone": "1.5,3,1.5,4.5,10,0,100"},
            {"SourceCommonName": "b.com", "V2Themes": "ECON_X,2", "V2Organizations": "", "V2Tone": "-2,1,3,4,10,0,100"},
        ]
    )
    dl = GdeltDownloader(cache_dir=tmp_path)
    with patch.object(dl, "_fetch_zip", return_value=payload):
        df = dl._download_gkg_file("http://example/20261030000000.gkg.csv.zip")

    assert df is not None
    assert list(df.columns) == _GKG_USED_COLS
    assert len(df) == 2


def test_aggregate_and_orgs_slice_work_on_slim_frame(tmp_path: Path) -> None:
    payload = _gkg_zip(
        [
            {"SourceCommonName": "a.com", "V2Themes": "ECON_X,1;TAX_Y,5", "V2Organizations": "Apple,3", "V2Tone": "1.5,3,1.5,4.5,10,0,100"},
            {"SourceCommonName": "b.com", "V2Themes": "ECON_X,2", "V2Organizations": "", "V2Tone": "-2,1,3,4,10,0,100"},
        ]
    )
    dl = GdeltDownloader(cache_dir=tmp_path)
    with patch.object(dl, "_fetch_zip", return_value=payload):
        raw = dl._download_gkg_file("http://example/20261030000000.gkg.csv.zip")
    assert raw is not None

    agg = _aggregate_gkg(raw, datetime(2026, 10, 30))
    counts = dict(zip(agg["theme"], agg["article_count"]))
    assert counts == {"ECON_X": 2, "TAX_Y": 1}

    slim = _extract_orgs_slice(raw, datetime(2026, 10, 30))
    assert list(slim["source"]) == ["a.com"]
