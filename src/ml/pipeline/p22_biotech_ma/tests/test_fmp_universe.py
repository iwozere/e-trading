"""Tests for ingest/fmp_universe.py. No live DB — repo is a MagicMock; raw zone uses tmp_path."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.ingest.fmp_universe import build_known_universe, build_unresolved_universe


def _repo(companies):
    repo = MagicMock()
    repo.list_companies_full.return_value = companies
    return repo


def test_build_known_universe_only_includes_companies_with_ticker():
    repo = _repo([
        {"company_id": 1, "cik": "0001", "ticker": "ABCD", "name": "Has Ticker Inc"},
        {"company_id": 2, "cik": "0002", "ticker": None, "name": "No Ticker Inc"},
    ])

    targets = build_known_universe(repo)

    assert len(targets) == 1
    assert targets[0].ticker == "ABCD"
    assert targets[0].company_id == 1


def test_build_known_universe_empty_when_no_companies():
    assert build_known_universe(_repo([])) == []


def test_build_unresolved_universe_excludes_known_ciks(tmp_path):
    repo = _repo([{"company_id": 1, "cik": "0000000100", "ticker": "ABCD", "name": "Known Co"}])
    raw_zone.write(
        source="sec_dera_universe",
        entity="2015q1",
        as_of_date=date(2015, 6, 1),
        payload=[
            {"cik": "0000000100", "name": "Known Co", "filed": "2015-02-01"},
            {"cik": "0000000200", "name": "Delisted Before Resolution Inc", "filed": "2015-02-01"},
        ],
        root=tmp_path,
    )

    unresolved = build_unresolved_universe(repo, root=tmp_path)

    ciks = {u.cik for u in unresolved}
    assert "0000000100" not in ciks  # already known, excluded
    assert "0000000200" in ciks


def test_build_unresolved_universe_dedups_by_cik_keeping_latest_filed_name(tmp_path):
    repo = _repo([])
    raw_zone.write(
        source="sec_dera_universe", entity="2015q1", as_of_date=date(2015, 6, 1),
        payload=[{"cik": "0000000300", "name": "Old Name Corp", "filed": "2015-02-01"}],
        root=tmp_path,
    )
    raw_zone.write(
        source="sec_dera_universe", entity="2016q1", as_of_date=date(2016, 6, 1),
        payload=[{"cik": "0000000300", "name": "Renamed Corp", "filed": "2016-02-01"}],
        root=tmp_path,
    )

    unresolved = build_unresolved_universe(repo, root=tmp_path)

    assert len(unresolved) == 1
    assert unresolved[0].name == "Renamed Corp"


def test_build_unresolved_universe_empty_when_nothing_landed(tmp_path):
    assert build_unresolved_universe(_repo([]), root=tmp_path) == []
