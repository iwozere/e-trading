"""Tests for ingest/sec_universe_ingest.py — mocked HTTP, no network calls."""

import io
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.sec_universe_ingest import (
    discover_quarterly_archive_urls,
    fetch_quarter_submissions,
)


def _mock_response(text=None, content=None, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    if text is not None:
        resp.text = text
    if content is not None:
        resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


def test_discover_quarterly_archive_urls_parses_landing_page():
    html = """
    <html><body>
    <a href="/files/dera/data/financial-statement-data-sets/2019q3.zip">2019 Q3</a>
    <a href="/files/dera/data/financial-statement-data-sets/2019q4.zip">2019 Q4</a>
    <a href="/some/other/link.html">Not an archive</a>
    </body></html>
    """
    client = MagicMock()
    client.get.return_value = _mock_response(text=html)

    urls = discover_quarterly_archive_urls(client)

    assert set(urls.keys()) == {"2019q3", "2019q4"}
    assert urls["2019q3"] == "https://www.sec.gov/files/dera/data/financial-statement-data-sets/2019q3.zip"


def test_fetch_quarter_submissions_filters_to_biotech_sic():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "sub.txt",
            "adsh\tcik\tname\tsic\tform\tfiled\n"
            "0001\t100\tBiotech Co\t2836\t10-K\t2024-01-01\n"
            "0002\t200\tBank Co\t6022\t10-K\t2024-01-01\n",
        )
    client = MagicMock()
    client.get.return_value = _mock_response(content=buf.getvalue())

    rows = fetch_quarter_submissions("2024q1", "https://example.com/2024q1.zip", client)

    assert len(rows) == 1
    assert rows[0]["name"] == "Biotech Co"


def test_fetch_quarter_submissions_handles_bad_zip():
    client = MagicMock()
    client.get.return_value = _mock_response(content=b"not a zip")

    rows = fetch_quarter_submissions("2024q1", "https://example.com/2024q1.zip", client)

    assert rows == []
