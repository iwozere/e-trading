"""Tests for P19 T+10 label backfill — EdgarDownloader/DataManager/yfinance mocked."""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p19_penny_intraday.label_backfill import LabelBackfill
from src.ml.pipeline.p19_penny_intraday.models.intraday_signal import IntradaySignal
from src.ml.pipeline.p19_penny_intraday.shadow_store import ShadowStore

_OLD_DATE = (datetime.now(UTC).date() - timedelta(days=30)).isoformat()
_RECENT_DATE = (datetime.now(UTC).date() - timedelta(days=2)).isoformat()


def _store_with_eod(tmp_path, date_str, ticker="AAA"):
    store = ShadowStore(str(tmp_path / "s.sqlite"))
    store.append(date_str, IntradaySignal(ticker=ticker, ts=datetime(2026, 1, 1, tzinfo=UTC)))
    store.update_eod(date_str, ticker, {"open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1})
    return store


def _closes_df(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"close": values}, index=idx)


def test_recent_date_is_skipped_not_enough_age(tmp_path):
    store = _store_with_eod(tmp_path, _RECENT_DATE)
    edgar = MagicMock()
    lb = LabelBackfill(store=store, edgar=edgar, data_manager=MagicMock())
    result = lb.run()
    assert result == {"dates": 0, "tickers": 0}
    assert store.tickers_needing_label_backfill(_RECENT_DATE) == ["AAA"]  # untouched


def test_old_enough_date_gets_forward_returns(tmp_path):
    store = _store_with_eod(tmp_path, _OLD_DATE)
    edgar = MagicMock()
    edgar.get_recent_filings.return_value = []
    edgar.resolve_tickers_to_ciks.return_value = ["123"]
    dm = MagicMock()
    dm.get_ohlcv.return_value = _closes_df([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1])

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        lb = LabelBackfill(store=store, edgar=edgar, data_manager=dm)
        result = lb.run()

    assert result == {"dates": 1, "tickers": 1}
    row = store._conn.execute("SELECT ret_t1, ret_t5, ret_t10 FROM shadow_log WHERE ticker='AAA'").fetchone()
    ret_t1, ret_t5, ret_t10 = row
    assert abs(ret_t1 - (1.2 / 1.1 - 1.0)) < 1e-9
    assert abs(ret_t5 - (1.6 / 1.1 - 1.0)) < 1e-9
    assert abs(ret_t10 - (2.1 / 1.1 - 1.0)) < 1e-9


def test_dilution_event_detected_from_offering_form(tmp_path):
    store = _store_with_eod(tmp_path, _OLD_DATE)
    as_of = datetime.strptime(_OLD_DATE, "%Y-%m-%d").date()
    edgar = MagicMock()
    edgar.resolve_tickers_to_ciks.return_value = ["123"]
    edgar.get_recent_filings.return_value = [
        {"form": "424B5", "filingDate": (as_of + timedelta(days=2)).isoformat(), "items": ""}
    ]
    dm = MagicMock()
    dm.get_ohlcv.return_value = _closes_df([1.1, 1.2])

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        LabelBackfill(store=store, edgar=edgar, data_manager=dm).run()

    row = store._conn.execute("SELECT dilution_event_within_5d FROM shadow_log WHERE ticker='AAA'").fetchone()
    assert row[0] == 1


def test_dilution_event_false_when_no_offering(tmp_path):
    store = _store_with_eod(tmp_path, _OLD_DATE)
    edgar = MagicMock()
    edgar.resolve_tickers_to_ciks.return_value = ["123"]
    edgar.get_recent_filings.return_value = []
    dm = MagicMock()
    dm.get_ohlcv.return_value = _closes_df([1.1, 1.2])

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        LabelBackfill(store=store, edgar=edgar, data_manager=dm).run()

    row = store._conn.execute("SELECT dilution_event_within_5d FROM shadow_log WHERE ticker='AAA'").fetchone()
    assert row[0] == 0


def test_reverse_split_within_180d_detected():
    store = MagicMock()
    edgar = MagicMock()
    lb = LabelBackfill(store=store, edgar=edgar, data_manager=MagicMock())
    as_of = datetime.strptime(_OLD_DATE, "%Y-%m-%d").date()
    split_series = pd.Series([0.1], index=pd.to_datetime([as_of + timedelta(days=30)]))
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = split_series
        result = lb._reverse_split_within("AAA", as_of, 180)
    assert result is True


def test_unresolvable_cik_returns_none_for_dilution_label():
    store = MagicMock()
    edgar = MagicMock()
    edgar.resolve_tickers_to_ciks.return_value = []
    lb = LabelBackfill(store=store, edgar=edgar, data_manager=MagicMock())
    as_of = datetime.strptime(_OLD_DATE, "%Y-%m-%d").date()
    assert lb._dilution_event_within("ZZZ", as_of, 5) is None


def test_no_forward_data_skips_ticker_without_crashing(tmp_path):
    store = _store_with_eod(tmp_path, _OLD_DATE)
    edgar = MagicMock()
    edgar.resolve_tickers_to_ciks.return_value = ["123"]
    edgar.get_recent_filings.return_value = []
    dm = MagicMock()
    dm.get_ohlcv.return_value = pd.DataFrame()  # empty -- provider has nothing

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.splits = pd.Series(dtype=float)
        result = LabelBackfill(store=store, edgar=edgar, data_manager=dm).run()

    assert result == {"dates": 0, "tickers": 0}
    row = store._conn.execute("SELECT ret_t1 FROM shadow_log WHERE ticker='AAA'").fetchone()
    assert row[0] is None
