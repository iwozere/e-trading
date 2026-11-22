"""Tests for the EODHD client and the pure conversions. No network: httpx mock transport only."""

from __future__ import annotations

from datetime import date

import httpx
import pandas as pd
import pytest

from src.data.eodhd_pipeline.client import EodhdClient, EodhdError, PlanRestricted, QuotaExhausted, _Pacer
from src.data.eodhd_pipeline.convert import bars_frame, dividends_frame, later_split_factor, parse_split, splits_frame
from src.data.eodhd_pipeline.tests.helpers import BARS, DIVIDENDS, SPLITS, FakeEodhd, make_client


def test_retries_server_errors_then_succeeds_and_counts_every_attempt():
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json=[{"Code": "A"}])

    client = make_client(flaky)
    assert client.symbols()["Code"].tolist() == ["A"] and client.calls == 3
    with pytest.raises(EodhdError):
        make_client(lambda r: httpx.Response(503)).symbols()


@pytest.mark.parametrize(("status", "error"), [(402, QuotaExhausted), (429, QuotaExhausted), (403, PlanRestricted), (401, EodhdError)])
def test_error_statuses_map_to_exceptions(status: int, error: type):
    with pytest.raises(error):
        make_client(lambda r: httpx.Response(status, text="x")).splits("ABC")


def test_unknown_symbol_is_none_and_free_plan_is_flagged():
    client = make_client(FakeEodhd(free=True))
    assert client.eod_rows("NOPE") == (None, False)
    rows, limited = client.eod_rows("ABC")
    assert limited and rows is not None and len(rows) == 6 and all("date" in row for row in rows)
    history = client.eod("ABC", date(2024, 1, 1), date(2024, 12, 31))
    assert history.plan_limited and history.bars["date"].iloc[0] == date(2024, 1, 2)
    assert client.eod("NOPE").bars.empty
    with pytest.raises(PlanRestricted):
        client.dividends("ABC")


def test_user_is_free_of_charge_and_gives_calls_left():
    fake = FakeEodhd(used=1_000, daily=100_000)
    client = make_client(fake)
    assert client.calls_left() == 99_000 and client.calls == 0
    assert make_client(lambda r: httpx.Response(200, json={})).calls_left() is None
    assert client.splits("ABC") == SPLITS["ABC"] and client.dividends("XYZ_old") == []
    assert client.calls == 2 and client.delisted_symbols()["Code"].tolist() == ["XYZ_old", "GONE"]


def test_eod_sends_the_date_range_only_when_given():
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=[])

    client = make_client(handler)
    client.eod_rows("ABC")
    client.eod_rows("ABC", date(2020, 1, 1), date(2020, 2, 1))
    assert "from" not in seen[0] and seen[1]["from"] == "2020-01-01" and seen[1]["to"] == "2020-02-01"


def test_pacer_waits_when_the_minute_is_full():
    now = {"t": 0.0}
    waits: list[float] = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        now["t"] += seconds

    pacer = _Pacer(2, sleep, lambda: now["t"])
    pacer.wait()
    now["t"] = 10.0
    pacer.wait()
    pacer.wait()  # third call within the minute waits until the first stamp is 60 s old
    assert waits == [50.0] and now["t"] == 60.0
    _Pacer(None, sleep, lambda: 0.0).wait()
    client = EodhdClient("k", client=httpx.Client(transport=httpx.MockTransport(FakeEodhd())), sleep=sleep, calls_per_minute=1,
                         clock=lambda: now["t"])
    client.splits("ABC")
    client.splits("ABC")
    assert len(waits) == 2


def test_parse_split():
    assert parse_split("2.000000/1.000000") == (2.0, 1.0)
    assert parse_split("104.000000/100.000000") == (104.0, 100.0)
    assert parse_split("1/10") == (1.0, 10.0)
    for bad in ("1/1", "0/1", "2/0", "x/1", "2", None, 3):
        assert parse_split(bad) is None


def test_splits_frame_drops_bad_rows_and_duplicates():
    rows = [{"date": "2024-01-05", "split": "2/1"}, {"date": "2024-01-05", "split": "3/1"}, {"date": "bad", "split": "2/1"},
            {"date": "2020-06-01", "split": "1/1"}, {"date": "2010-01-04", "split": "1/4"}]
    frame = splits_frame(rows)
    assert frame.to_dict("records") == [{"date": date(2010, 1, 4), "numerator": 1.0, "denominator": 4.0},
                                        {"date": date(2024, 1, 5), "numerator": 3.0, "denominator": 1.0}]
    assert splits_frame([]).empty


def test_later_split_factor():
    splits = splits_frame([{"date": "2024-01-05", "split": "2/1"}, {"date": "2024-01-09", "split": "3/1"}])
    dates = pd.Series([date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9)])
    assert later_split_factor(dates, splits).tolist() == [6.0, 3.0, 3.0, 1.0]
    assert later_split_factor(dates, splits_frame([])).tolist() == [1.0] * 4


def test_dividends_use_the_unadjusted_value_or_undo_the_split():
    splits = splits_frame(SPLITS["ABC"])
    assert dividends_frame(DIVIDENDS["ABC"], splits).to_dict("records") == [{"date": date(2024, 1, 3), "amount": 1.0, "kind": "cash"}]
    fallback = [{"date": "2024-01-03", "value": 0.5}, {"date": "2024-01-08", "value": 0.25, "unadjustedValue": 0},
                {"date": "2024-01-09", "value": 9.0, "currency": "CAD"}, {"date": "x", "value": 1.0}, {"date": "2024-01-02", "value": None}]
    out = dividends_frame(fallback, splits)
    assert out["amount"].tolist() == [1.0, 0.25] and out["date"].tolist() == [date(2024, 1, 3), date(2024, 1, 8)]
    assert dividends_frame([], splits).empty


def test_bars_frame_is_raw_with_as_traded_volume():
    bars = bars_frame(BARS["ABC"], splits_frame(SPLITS["ABC"]))
    assert bars["close"].tolist() == [100.0, 102.0, 104.0, 53.0, 54.0, 55.0]
    assert bars["volume"].tolist() == [1000.0] * 6
    assert bars["closeadj_vendor"].iloc[0] == 49.5 and bars["date"].iloc[0] == date(2024, 1, 2)
    kept = bars_frame(BARS["ABC"], splits_frame(SPLITS["ABC"]), volume_split_adjusted=False)
    assert kept["volume"].tolist()[:3] == [2000.0] * 3


def test_bars_frame_cleans_rows():
    rows = [{"date": "2024-01-03", "close": 5, "volume": 1}, {"date": "2024-01-02", "close": 0, "volume": 1},
            {"date": "2024-01-03", "close": 6, "volume": 2}, {"date": "bad", "close": 7}]
    out = bars_frame(rows, splits_frame([]))
    assert out["close"].tolist() == [6.0] and out["open"].isna().all() and out["closeadj_vendor"].isna().all()
    assert bars_frame([], splits_frame([])).empty and bars_frame([{"x": 1}], splits_frame([])).empty
