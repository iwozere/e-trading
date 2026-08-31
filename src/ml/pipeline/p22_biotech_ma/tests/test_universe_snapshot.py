"""Tests for ingest/universe_snapshot.py."""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.ingest.universe_snapshot import all_landed_quarters, latest_universe_rows


def test_returns_empty_when_nothing_landed(tmp_path):
    assert latest_universe_rows(root=tmp_path) == []


def test_dedups_by_cik_keeping_latest_filed(tmp_path):
    raw_zone.write(
        source="sec_dera_universe",
        entity="2024q1",
        as_of_date=date(2024, 6, 1),
        payload=[
            {"cik": "100", "name": "Old Name Co", "filed": "2024-01-01"},
            {"cik": "200", "name": "Other Co", "filed": "2024-01-01"},
        ],
        root=tmp_path,
    )
    raw_zone.write(
        source="sec_dera_universe",
        entity="2024q2",
        as_of_date=date(2024, 6, 1),
        payload=[
            {"cik": "100", "name": "New Name Co", "filed": "2024-04-01"},
        ],
        root=tmp_path,
    )

    rows = latest_universe_rows(root=tmp_path)
    by_cik = {r["cik"]: r for r in rows}

    assert len(rows) == 2
    assert by_cik["100"]["name"] == "New Name Co"
    assert by_cik["200"]["name"] == "Other Co"


def test_only_reads_most_recent_date_partition(tmp_path):
    raw_zone.write(
        source="sec_dera_universe",
        entity="2024q1",
        as_of_date=date(2024, 1, 1),
        payload=[{"cik": "999", "name": "Stale Co", "filed": "2024-01-01"}],
        root=tmp_path,
    )
    raw_zone.write(
        source="sec_dera_universe",
        entity="2024q2",
        as_of_date=date(2024, 6, 1),
        payload=[{"cik": "100", "name": "Fresh Co", "filed": "2024-04-01"}],
        root=tmp_path,
    )

    rows = latest_universe_rows(root=tmp_path)

    assert len(rows) == 1
    assert rows[0]["cik"] == "100"


def test_all_landed_quarters_returns_empty_when_nothing_landed(tmp_path):
    assert all_landed_quarters(root=tmp_path) == {}


def test_all_landed_quarters_spans_multiple_ingest_dates(tmp_path):
    """Unlike latest_universe_rows, this must NOT be limited to the most recent ingest date —
    land_all_quarters lands 15+ years of history dated by ingest day, not by quarter."""
    raw_zone.write(
        source="sec_dera_universe",
        entity="2019q1",
        as_of_date=date(2019, 4, 1),  # landed years ago
        payload=[{"cik": "100", "name": "Old Quarter Co", "filed": "2019-01-15"}],
        root=tmp_path,
    )
    raw_zone.write(
        source="sec_dera_universe",
        entity="2024q2",
        as_of_date=date(2024, 6, 1),  # landed recently
        payload=[{"cik": "200", "name": "Recent Quarter Co", "filed": "2024-04-15"}],
        root=tmp_path,
    )

    quarters = all_landed_quarters(root=tmp_path)

    assert set(quarters.keys()) == {"2019q1", "2024q2"}
    assert quarters["2019q1"][0]["cik"] == "100"
    assert quarters["2024q2"][0]["cik"] == "200"


def test_all_landed_quarters_later_ingest_date_wins_for_same_quarter(tmp_path):
    raw_zone.write(
        source="sec_dera_universe",
        entity="2024q1",
        as_of_date=date(2024, 4, 1),
        payload=[{"cik": "100", "name": "Stale Copy", "filed": "2024-01-15"}],
        root=tmp_path,
    )
    raw_zone.write(
        source="sec_dera_universe",
        entity="2024q1",
        as_of_date=date(2024, 5, 1),  # later re-ingest of the same quarter
        payload=[{"cik": "100", "name": "Corrected Copy", "filed": "2024-01-15"}],
        root=tmp_path,
    )

    quarters = all_landed_quarters(root=tmp_path)

    assert len(quarters["2024q1"]) == 1
    assert quarters["2024q1"][0]["name"] == "Corrected Copy"
