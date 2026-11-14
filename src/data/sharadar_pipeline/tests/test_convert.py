"""Tests for the pure Sharadar conversions. Expected values are hand-computed from the synthetic fixtures."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data.sharadar_pipeline.convert import (
    delisting_return,
    delistings_frame,
    dividends_frame,
    extract_cik,
    fundamentals_ar,
    instrument_type,
    plain_symbol,
    raw_prices,
    securities_frame,
    splits_frame,
    symbol_changes_frame,
)
from src.data.sharadar_pipeline.tests.helpers import TICKERS, actions_frame, fundamentals_frame, stocks_frame


def test_extract_cik_reads_the_number_from_the_filings_url():
    assert extract_cik("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001090872") == 1090872
    assert extract_cik("") is None
    assert extract_cik(None) is None


def test_plain_symbol_removes_the_reuse_number_only():
    assert plain_symbol("XYZ1") == "XYZ"
    assert plain_symbol("WOLF2") == "WOLF"
    assert plain_symbol("ABC") == "ABC"
    assert plain_symbol("ABC-PA") == "ABC-PA"


@pytest.mark.parametrize("category, expected", [
    ("Domestic Common Stock", "stock"),
    ("Domestic Common Stock Primary Class", "stock"),
    ("Canadian Common Stock", "stock"),
    ("Domestic Common Stock Secondary Class", "stock_secondary"),
    ("ADR Common Stock", "adr"),
    ("ADR Common Stock Primary Class", "adr"),
    ("Domestic Preferred Stock", "preferred"),
    ("ADR Preferred Stock", "preferred"),
    (None, "other"),
])
def test_instrument_type_maps_categories(category, expected):
    assert instrument_type(category) == expected


def test_securities_frame_has_one_row_per_permaticker_with_vintage_columns():
    out = securities_frame(TICKERS).set_index("security_id")
    assert list(out.index) == [101, 102, 103]
    assert out.loc[102, "symbol"] == "XYZ" and out.loc[102, "sharadar_ticker"] == "XYZ1"
    assert out.loc[102, "delisted_date"] == date(2024, 1, 8) and not out.loc[102, "is_actively_trading"]
    assert out.loc[101, "delisted_date"] is None and out.loc[101, "is_actively_trading"]
    assert out.loc[101, "cik"] == 111 and out.loc[103, "instrument_type"] == "preferred"
    assert out["figi"].isna().tolist() == [False, True, True] and out.loc[101, "figi"] == "BBG000000101"


def test_securities_frame_drops_duplicate_permatickers():
    doubled = pd.concat([TICKERS, TICKERS[TICKERS["permaticker"] == "101"]], ignore_index=True)
    assert len(securities_frame(doubled)) == 3


def test_raw_prices_undoes_the_split_adjustment():
    abc = raw_prices(stocks_frame().query("ticker == 'ABC'")).set_index("date")
    # before the 2-for-1 split: adjusted close 50 -> raw 100, volume 2000 -> 1000, open 49 -> 98
    assert abc.loc["2024-01-02", "close"] == 100.0
    assert abc.loc["2024-01-02", "open"] == 98.0
    assert abc.loc["2024-01-02", "high"] == 102.0
    assert abc.loc["2024-01-02", "volume"] == 1000.0
    # after the split nothing changes
    assert abc.loc["2024-01-05", "close"] == 53.0 and abc.loc["2024-01-05", "volume"] == 1000.0
    assert abc.loc["2024-01-02", "closeadj_vendor"] == 49.5
    assert "closeunadj" not in abc.columns


def test_raw_prices_drops_rows_without_a_positive_close():
    frame = stocks_frame().query("ticker == 'XYZ1'").copy()
    frame.loc[frame.index[0], "close"] = 0.0
    frame.loc[frame.index[1], "closeunadj"] = None
    assert len(raw_prices(frame)) == len(frame) - 2


def test_splits_frame_is_in_adjust_form():
    out = splits_frame(actions_frame())
    assert out.to_dict("records") == [{"ticker": "ABC", "date": date(2024, 1, 5), "numerator": 2.0, "denominator": 1.0}]


def test_dividends_are_rebased_by_later_splits_only():
    actions = actions_frame()
    out = dividends_frame(actions, splits_frame(actions))
    # ex-date 2024-01-03 is before the 2-for-1 split of 2024-01-05: 0.50 adjusted = 1.00 raw
    assert out.to_dict("records") == [{"ticker": "ABC", "date": date(2024, 1, 3), "kind": "cash", "amount": 1.0, "amount_adj": 0.5}]
    late = actions.copy()
    late.loc[late["action"] == "dividend", "date"] = "2024-01-05"  # on the split's ex-date: already post-split
    assert dividends_frame(late, splits_frame(late))["amount"].tolist() == [0.5]


def test_dividends_frame_keeps_spinoffs_apart_and_handles_no_rows():
    actions = actions_frame()
    spin = pd.DataFrame([("2024-01-04", "spinoffdividend", "ABC", "ABC CORP", 3.0, "NEWCO", "NEWCO INC")], columns=actions.columns)
    out = dividends_frame(pd.concat([actions, spin]), splits_frame(actions))
    assert out.set_index("kind")["amount"].to_dict() == {"cash": 1.0, "spinoff": 6.0}
    assert dividends_frame(actions[actions["action"] == "split"], splits_frame(actions)).empty


def test_delistings_frame_takes_the_specific_reason_and_the_consideration():
    out = delistings_frame(actions_frame())
    assert len(out) == 1
    row = out.iloc[0]
    assert (row["ticker"], row["date"], row["reason"], row["acquirer"]) == ("XYZ1", date(2024, 1, 8), "acquired", "ABC")
    assert row["cash_per_share"] == 10.0 and row["stock_ratio"] == 0.1 and row["stock_ticker"] == "ABC"
    assert row["final_market_cap_musd"] == 140.0 and not row["other_consideration"]


def test_delistings_frame_flags_consideration_it_does_not_interpret():
    actions = actions_frame()
    extra = pd.DataFrame([("2024-01-08", "acquisitioncasheur", "XYZ1", "XYZ INC", 9.0, "N/A", "N/A")], columns=actions.columns)
    assert bool(delistings_frame(pd.concat([actions, extra])).iloc[0]["other_consideration"])
    assert delistings_frame(actions[actions["action"] == "split"]).empty


def test_symbol_changes_frame_lists_former_symbols():
    assert symbol_changes_frame(actions_frame()).to_dict("records") == [{"ticker": "ABC", "old_symbol": "OLDA", "date": date(2024, 1, 3)}]


def test_fundamentals_ar_drops_restated_dimensions_and_sets_availability():
    out = fundamentals_ar(fundamentals_frame())
    assert sorted(out["dimension"]) == ["ARQ", "ART"]
    arq = out[out["dimension"] == "ARQ"].iloc[0]
    assert arq["available_from"] == pd.Timestamp("2024-01-04") and arq["revenue"] == 1000.0
    assert arq["fiscalperiod"] == "2023-Q3"


def test_delisting_return_adds_cash_and_stock():
    assert delisting_return(14.0, 10.0, 0.1, 54.0) == pytest.approx(0.1)  # (10 + 5.4) / 14 - 1
    assert delisting_return(10.0, 12.0) == pytest.approx(0.2)
    assert delisting_return(10.0, None, 0.5, 30.0) == pytest.approx(0.5)


def test_delisting_return_is_none_when_the_consideration_is_unknown():
    assert delisting_return(10.0, None) is None
    assert delisting_return(10.0, float("nan"), float("nan")) is None
    assert delisting_return(10.0, 5.0, 0.1, None) is None
    assert delisting_return(0.0, 5.0) is None
