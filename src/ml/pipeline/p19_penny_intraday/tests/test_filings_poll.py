"""Tests for the P19 intraday EDGAR filings poll (spec v2 §9)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday.filings_poll import FilingsPoll

_DATE = "2026-08-18"


def _watchlist(tmp_path, entries):
    d = tmp_path / _DATE
    d.mkdir(parents=True)
    (d / "watchlist.json").write_text(json.dumps({"date": _DATE, "entries": entries}))


def _entry(ticker):
    return {"ticker": ticker, "source": "p17", "tier": "B", "avg_volume_30d": 1_000_000, "prior_close": 2.0}


def _hit(cik, adsh, doc="doc.htm", items=None):
    # EFTS returns ciks zero-padded to 10 digits; company_tickers.json's
    # cik_str does not -- both real shapes, exercising the same
    # zero-stripping normalisation _hit_cik/_build_cik_map do in production.
    src = {"ciks": [f"{int(cik):010d}"], "adsh": adsh, "file_date": _DATE}
    if items is not None:
        src["items"] = items
    return {"_id": f"{adsh}:{doc}", "_source": src}


def _edgar_mock(tickers_to_ciks, search_results=None):
    edgar = MagicMock()
    edgar.load_company_tickers.return_value = {
        str(i): {"ticker": t, "cik_str": c} for i, (t, c) in enumerate(tickers_to_ciks.items())
    }
    edgar.efts_filings_search.side_effect = lambda ciks, forms, start_dt, end_dt: (search_results or {}).get(forms, [])
    return edgar


def _poll(tmp_path, edgar):
    return FilingsPoll(output_dir=str(tmp_path), target_date=_DATE, edgar=edgar, db_path=str(tmp_path / "events.sqlite"))


def test_no_watchlist_is_a_clean_no_op(tmp_path):
    edgar = _edgar_mock({})
    poll = _poll(tmp_path, edgar)
    res = poll.run()
    assert res["reason"] == "no watchlist"
    edgar.efts_filings_search.assert_not_called()


def test_dilution_form_hit_is_recorded(tmp_path):
    _watchlist(tmp_path, [_entry("AAA")])
    edgar = _edgar_mock({"AAA": "123"}, {"424B5": [_hit("123", "0001-26-000001")]})
    poll = _poll(tmp_path, edgar)
    res = poll.run()
    assert res["new_hits"] == 1
    events = poll.events_for_date()
    assert len(events) == 1
    assert events[0]["ticker"] == "AAA"
    assert events[0]["form_type"] == "424B5"
    assert events[0]["is_dilution"] == 1


def test_8k_item_301_and_302_are_recorded_others_are_not(tmp_path):
    _watchlist(tmp_path, [_entry("AAA")])
    edgar = _edgar_mock(
        {"AAA": "123"},
        {"8-K": [_hit("123", "0001-26-000002", items=["1.01", "3.01", "3.02"])]},
    )
    poll = _poll(tmp_path, edgar)
    res = poll.run()
    assert res["new_hits"] == 2  # 3.01 and 3.02, not 1.01
    events = poll.events_for_date()
    items = {e["item"] for e in events}
    assert items == {"3.01", "3.02"}
    dilution_flags = {e["item"]: e["is_dilution"] for e in events}
    assert dilution_flags["3.02"] == 1  # unregistered equity sale -> dilution
    assert dilution_flags["3.01"] == 0  # deficiency notice -> not itself dilution


def test_ticker_with_unresolvable_cik_is_skipped_not_crashed(tmp_path):
    _watchlist(tmp_path, [_entry("ZZZ")])  # not in the ticker->CIK map
    edgar = _edgar_mock({})
    poll = _poll(tmp_path, edgar)
    res = poll.run()
    assert res["reason"] == "no CIKs resolved"


def test_second_run_does_not_duplicate_the_same_filing(tmp_path):
    _watchlist(tmp_path, [_entry("AAA")])
    edgar = _edgar_mock({"AAA": "123"}, {"424B5": [_hit("123", "0001-26-000001")]})
    poll = _poll(tmp_path, edgar)
    poll.run()
    res2 = poll.run()  # same hit again (e.g. re-polled 30 min later)
    assert res2["new_hits"] == 0
    assert len(poll.events_for_date()) == 1


def test_efts_failure_for_one_form_does_not_abort_the_others(tmp_path):
    _watchlist(tmp_path, [_entry("AAA")])
    edgar = _edgar_mock({"AAA": "123"}, {"S-1": [_hit("123", "0001-26-000003")]})
    edgar.efts_filings_search.side_effect = [
        Exception("EFTS down"),  # 424B5
        [_hit("123", "0001-26-000003")],  # S-1
        Exception("EFTS down"),  # S-3
        Exception("EFTS down"),  # 8-K
    ]
    poll = _poll(tmp_path, edgar)
    res = poll.run()  # must not raise
    assert res["new_hits"] == 1
