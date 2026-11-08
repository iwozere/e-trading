"""Tests for ingest/domicile_history.py — fake EFTS and repo, no network or DB."""

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest import domicile_history
from src.ml.pipeline.p22_biotech_ma.ingest.domicile_history import changes_only, observations_from_hits

_HORIZON = "0001492426"
_SERES = "0001609809"


def _hit(hit_id: str, ciks: List[str], states: List[str], filed: str, adsh: str = "0000000000-15-000001") -> Dict[str, Any]:
    return {"_id": hit_id, "_source": {"ciks": ciks, "biz_states": states, "file_date": filed, "adsh": adsh}}


def test_observations_map_states_by_cik_position_and_flag_foreign_codes():
    hits = [
        _hit("a", [_HORIZON], ["IL"], "2014-05-23"),
        _hit("b", [_HORIZON], ["L2"], "2015-02-27"),
        _hit("c", [_SERES, "0000000099"], ["MA", "X0"], "2016-03-14"),  # co-registrant outside the universe
    ]
    observed = observations_from_hits(hits, {_HORIZON: 1, _SERES: 2})
    assert [(d, f) for d, f, _ in observed[1]] == [(date(2014, 5, 23), False), (date(2015, 2, 27), True)]
    assert [(d, f) for d, f, _ in observed[2]] == [(date(2016, 3, 14), False)]


def test_observations_skip_hits_whose_states_do_not_line_up_with_ciks():
    hits = [_hit("a", [_HORIZON, _SERES], ["IL"], "2014-05-23")]
    assert observations_from_hits(hits, {_HORIZON: 1, _SERES: 2}) == {}


def test_changes_only_keeps_first_and_each_switch():
    rows = [(date(2013, 3, 1), False, "a"), (date(2014, 3, 1), False, "b"), (date(2015, 3, 1), True, "c"), (date(2016, 3, 1), True, "d")]
    assert changes_only(rows) == [rows[0], rows[2]]


class _FakeEdgar:
    def __init__(self, hits: List[Dict[str, Any]]):
        self.hits = hits
        self.calls: List[tuple] = []

    def efts_filings_search(self, *, ciks, forms, start_dt, end_dt):
        del ciks
        self.calls.append((forms, start_dt, end_dt))
        return [h for h in self.hits if start_dt <= h["_source"]["file_date"] <= end_dt and forms == "10-K"]


class _FakeRepo:
    def __init__(self) -> None:
        self.inserted: Dict[int, List[Dict[str, Any]]] = {}

    def list_companies_full(self) -> List[Dict[str, Any]]:
        return [{"company_id": 1, "cik": _HORIZON}, {"company_id": 2, "cik": None}]

    def insert_financial_fact_history(self, *, company_id: int, metric: str, facts):
        assert metric == "is_foreign_domiciled"
        self.inserted[company_id] = list(facts)
        return len(facts)


def test_run_scans_each_year_and_writes_one_fact_per_change_known_from_filing_date():
    edgar = _FakeEdgar([
        _hit("a", [_HORIZON], ["IL"], "2013-03-01", "0001492426-13-000005"),
        _hit("b", [_HORIZON], ["IL"], "2014-05-23"),
        _hit("c", [_HORIZON], ["L2"], "2015-02-27"),
    ])
    repo = _FakeRepo()
    summary = domicile_history.run(repo, edgar, "2013-01-01", "2015-06-30")

    assert summary == {"filings": 3, "companies_observed": 1, "facts_inserted": 2, "companies_changed_domicile": 1}
    assert {(start, end) for _, start, end in edgar.calls} == {
        ("2013-01-01", "2013-12-31"), ("2014-01-01", "2014-12-31"), ("2015-01-01", "2015-06-30"),
    }
    first, second = repo.inserted[1]
    assert (first["value"], first["known_from"]) == (0.0, datetime(2013, 3, 1, tzinfo=timezone.utc))
    assert first["source_url"] == "https://www.sec.gov/Archives/edgar/data/1492426/000149242613000005/"
    assert (second["value"], second["valid_from"]) == (1.0, date(2015, 2, 27))
