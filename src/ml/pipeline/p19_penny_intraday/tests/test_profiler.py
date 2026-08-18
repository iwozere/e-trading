"""
Tests for the P19 Layer 0 StructuralProfiler orchestrator — wiring only (the
grading logic itself is covered by test_grading.py). EdgarDownloader and
yfinance are mocked; no network calls.
"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p19_penny_intraday.config import P19Config
from src.ml.pipeline.p19_penny_intraday.models.watchlist_entry import WatchlistEntry
from src.ml.pipeline.p19_penny_intraday.structural.cache import StructuralProfileCache
from src.ml.pipeline.p19_penny_intraday.structural.profiler import StructuralProfiler

_AS_OF = date(2026, 8, 18)


def _make_edgar_mock():
    edgar = MagicMock()
    edgar.load_company_tickers.return_value = {"0": {"ticker": "AAA", "cik_str": "123"}}
    edgar.load_company_facts.return_value = {"facts": {}}
    edgar.get_recent_filings.return_value = []
    edgar.download_form4_filings.return_value = pd.DataFrame()
    return edgar


def _profiler(tmp_path, edgar=None):
    cfg = P19Config.create_default()
    cache = StructuralProfileCache(str(tmp_path / "structural_cache"), ttl_days=cfg.structural_config.profile_cache_ttl_days)
    return StructuralProfiler(config=cfg, edgar=edgar or _make_edgar_mock(), cache=cache), cache


def _entry(ticker="AAA"):
    return WatchlistEntry(ticker=ticker, source="p17", prior_close=1.0, market_cap=10_000_000, float_shares=8_000_000)


def test_refresh_watchlist_produces_a_profile_per_entry(tmp_path):
    profiler, _cache = _profiler(tmp_path)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        profiles = profiler.refresh_watchlist([_entry("AAA")], as_of=_AS_OF)
    assert "AAA" in profiles
    assert profiles["AAA"].ticker == "AAA"
    assert profiles["AAA"].cik == "123"


def test_unresolvable_cik_still_produces_a_profile(tmp_path):
    edgar = _make_edgar_mock()
    edgar.load_company_tickers.return_value = {}  # ticker not found -> no CIK
    profiler, _cache = _profiler(tmp_path, edgar=edgar)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        profiles = profiler.refresh_watchlist([_entry("ZZZ")], as_of=_AS_OF)
    assert profiles["ZZZ"].cik is None
    assert profiles["ZZZ"].grade in ("A", "B", "C", "D")  # never raises


def test_second_run_within_ttl_serves_from_cache_without_refetching(tmp_path):
    edgar = _make_edgar_mock()
    profiler, cache = _profiler(tmp_path, edgar=edgar)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        profiler.refresh_watchlist([_entry("AAA")], as_of=_AS_OF)
        calls_after_first = edgar.load_company_facts.call_count
        profiler.refresh_watchlist([_entry("AAA")], as_of=_AS_OF)  # same day -> cache hit
    assert edgar.load_company_facts.call_count == calls_after_first  # no re-fetch
    assert cache.load("AAA") is not None


def test_force_bypasses_cache(tmp_path):
    edgar = _make_edgar_mock()
    profiler, _cache = _profiler(tmp_path, edgar=edgar)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        profiler.refresh_watchlist([_entry("AAA")], as_of=_AS_OF)
        calls_after_first = edgar.load_company_facts.call_count
        profiler.refresh_watchlist([_entry("AAA")], as_of=_AS_OF, force=True)
    assert edgar.load_company_facts.call_count > calls_after_first


def test_form4_read_failure_for_one_day_does_not_abort_the_window(tmp_path):
    edgar = _make_edgar_mock()
    edgar.download_form4_filings.side_effect = Exception("boom")
    profiler, _cache = _profiler(tmp_path, edgar=edgar)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        profiles = profiler.refresh_watchlist([_entry("AAA")], as_of=_AS_OF)  # must not raise
    assert profiles["AAA"].grade in ("A", "B", "C", "D")


def test_splits_fetch_failure_leaves_reverse_splits_unresolved(tmp_path):
    edgar = _make_edgar_mock()
    profiler, _cache = _profiler(tmp_path, edgar=edgar)
    with patch("yfinance.Ticker", side_effect=Exception("network down")):
        profiles = profiler.refresh_watchlist([_entry("AAA")], as_of=_AS_OF)
    # Unresolved splits pulls coverage down but must not crash or grade A.
    assert profiles["AAA"].grade in ("B", "C", "D")


# ── Phase 3 wiring: N5/N6/N16 (EFTS), P8 (13D/G), P11 (yfinance short interest) ──


def _filing(form, filed_date):
    return {"form": form, "items": "", "filingDate": filed_date}


def test_going_concern_hit_flows_through_to_the_profile(tmp_path):
    edgar = _make_edgar_mock()
    edgar.get_recent_filings.return_value = [_filing("10-K", "2026-06-01")]
    edgar.efts_text_search.return_value = [{"_id": "0001:doc.htm", "_source": {"ciks": ["123"], "file_date": "2026-06-01"}}]
    edgar.get_auditor_name.return_value = None
    edgar.download_13dg_filings.return_value = pd.DataFrame()
    profiler, _cache = _profiler(tmp_path, edgar=edgar)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        mock_ticker.return_value.info = {}
        profiles = profiler.refresh_watchlist([_entry("AAA")], as_of=_AS_OF)
    assert profiles["AAA"].going_concern_flag is True
    assert profiles["AAA"].floating_convert_flag is True  # same mocked hit-everything EFTS call
    assert profiles["AAA"].grade == "D"


def test_no_efts_hits_resolve_false_not_none(tmp_path):
    edgar = _make_edgar_mock()
    edgar.get_recent_filings.return_value = [_filing("10-K", "2026-06-01")]
    edgar.efts_text_search.return_value = []
    edgar.get_auditor_name.return_value = "Marcum LLP"
    edgar.download_13dg_filings.return_value = pd.DataFrame()
    profiler, _cache = _profiler(tmp_path, edgar=edgar)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        mock_ticker.return_value.info = {}
        profiles = profiler.refresh_watchlist([_entry("AAA")], as_of=_AS_OF)
    assert profiles["AAA"].going_concern_flag is False
    assert profiles["AAA"].floating_convert_flag is False
    assert profiles["AAA"].auditor_name == "Marcum LLP"
    assert profiles["AAA"].auditor_whitelisted is True


def test_no_annual_or_interim_filing_leaves_n5_n6_unresolved(tmp_path):
    edgar = _make_edgar_mock()
    edgar.get_recent_filings.return_value = []  # no filings at all -> no date window
    profiler, _cache = _profiler(tmp_path, edgar=edgar)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        mock_ticker.return_value.info = {}
        profiles = profiler.refresh_watchlist([_entry("AAA")], as_of=_AS_OF)
    assert profiles["AAA"].going_concern_flag is None
    assert profiles["AAA"].floating_convert_flag is None
    edgar.efts_text_search.assert_not_called()


def test_dg_activity_proxy_flows_through_to_the_profile(tmp_path):
    edgar = _make_edgar_mock()
    edgar.get_recent_filings.return_value = []
    edgar.download_13dg_filings.return_value = pd.DataFrame(
        [{"cik": "123", "entity_name": "Some Fund", "accession_number": "x", "filed_date": "2026-07-01", "form_type": "SC 13G"}]
    )
    profiler, _cache = _profiler(tmp_path, edgar=edgar)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        mock_ticker.return_value.info = {}
        profiles = profiler.refresh_watchlist([_entry("AAA")], as_of=_AS_OF)
    assert profiles["AAA"].inst_13dg_activity_2q is True


def test_dg_no_activity_for_this_cik_resolves_false(tmp_path):
    edgar = _make_edgar_mock()
    edgar.get_recent_filings.return_value = []
    edgar.download_13dg_filings.return_value = pd.DataFrame(
        [{"cik": "999", "entity_name": "Unrelated Fund", "accession_number": "x", "filed_date": "2026-07-01", "form_type": "SC 13G"}]
    )
    profiler, _cache = _profiler(tmp_path, edgar=edgar)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        mock_ticker.return_value.info = {}
        profiles = profiler.refresh_watchlist([_entry("AAA")], as_of=_AS_OF)
    assert profiles["AAA"].inst_13dg_activity_2q is False


def test_short_interest_flows_through_to_the_profile(tmp_path):
    edgar = _make_edgar_mock()
    edgar.get_recent_filings.return_value = []
    edgar.download_13dg_filings.return_value = pd.DataFrame()
    profiler, _cache = _profiler(tmp_path, edgar=edgar)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        mock_ticker.return_value.info = {"shortPercentOfFloat": 0.35, "shortRatio": 6.0}
        profiles = profiler.refresh_watchlist([_entry("AAA")], as_of=_AS_OF)
    assert profiles["AAA"].short_interest_pct_float == 0.35
    assert profiles["AAA"].days_to_cover == 6.0


def test_short_interest_fetch_failure_leaves_it_unresolved(tmp_path):
    edgar = _make_edgar_mock()
    edgar.get_recent_filings.return_value = []
    edgar.download_13dg_filings.return_value = pd.DataFrame()
    profiler, _cache = _profiler(tmp_path, edgar=edgar)
    with patch("yfinance.Ticker", side_effect=Exception("network down")):
        profiles = profiler.refresh_watchlist([_entry("AAA")], as_of=_AS_OF)
    assert profiles["AAA"].short_interest_pct_float is None
    assert profiles["AAA"].days_to_cover is None
