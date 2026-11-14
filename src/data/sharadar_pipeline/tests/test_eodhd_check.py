"""Tests for the EODHD delisted-history check. No network: httpx mock transport and synthetic rows only."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd
import pytest

from src.data.sharadar_pipeline import cli, config
from src.data.sharadar_pipeline.eodhd_check import (
    EodhdClient,
    EodhdError,
    delisted_universe,
    find_candidates,
    grade,
    name_similarity,
    run_check,
    sample_universe,
    summarize,
)
from src.data.sharadar_pipeline.tests.helpers import TICKERS, write_raw

URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000000{cik}"


def _ticker(permaticker: int, ticker: str, name: str, first: str, last: str, delisted: str = "Y") -> dict[str, str]:
    return {"table": "stocks", "permaticker": str(permaticker), "ticker": ticker, "name": name, "exchange": "NYSE",
            "isdelisted": delisted, "category": "Domestic Common Stock", "firstpricedate": first, "lastpricedate": last,
            "secfilings": URL.format(cik=permaticker)}


#: FULL: complete history. LATE: EODHD starts a year late. GONE: not in EODHD. REUSE1: ticker reused by another company.
SHARADAR = pd.DataFrame([
    _ticker(201, "FULL", "FULL HISTORY CORP", "2010-01-04", "2010-12-31"),
    _ticker(202, "LATE", "LATE START INC", "2009-01-02", "2010-12-31"),
    _ticker(203, "GONE", "GONE CO", "2010-01-04", "2010-12-31"),
    _ticker(204, "REUSE1", "OLD REUSE HOLDINGS", "2019-01-02", "2019-12-31"),
    _ticker(205, "LIVE", "STILL TRADING INC", "2010-01-04", "2026-11-13", delisted="N"),
])
LISTED = [
    {"Code": "FULL", "Name": "Full History Corporation", "Exchange": "NYSE", "Type": "Common Stock"},
    {"Code": "LATE", "Name": "Late Start Inc", "Exchange": "NYSE", "Type": "Common Stock"},
    {"Code": "REUSE", "Name": "Totally Different Fund Manager", "Exchange": "NYSE", "Type": "Common Stock"},
    {"Code": "REUSE_old", "Name": "Old Reuse Holdings Inc", "Exchange": "NYSE", "Type": "Common Stock"},
    {"Code": "GONE", "Name": "Gone ETF", "Exchange": "NYSE ARCA", "Type": "ETF"},
]


def _bars(first: str, last: str, close: float = 10.0) -> list[dict[str, Any]]:
    days = pd.bdate_range(first, last)
    return [{"date": d.strftime("%Y-%m-%d"), "open": close, "high": close, "low": close, "close": close, "adjusted_close": close,
             "volume": 100} for d in days]


def _handler(warning: bool = False):
    histories = {"FULL.US": _bars("2010-01-04", "2010-12-31"), "LATE.US": _bars("2010-01-04", "2010-12-31"),
                 "REUSE.US": _bars("2021-01-04", "2021-06-30"), "REUSE_old.US": _bars("2019-01-02", "2019-12-31")}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "exchange-symbol-list" in path:
            assert request.url.params["delisted"] == "1" and request.url.params["api_token"] == "k"
            return httpx.Response(200, json=LISTED)
        code = path.rsplit("/", 1)[-1]
        if code not in histories:
            return httpx.Response(404, text="Ticker Not Found.")
        rows = histories[code]
        return httpx.Response(200, json=rows + ([{"warning": "Data is limited by one year as you have free subscription"}] if warning else []))

    return handler


def _client(handler) -> EodhdClient:
    return EodhdClient("k", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_name_similarity_ignores_legal_forms():
    assert name_similarity("LEHMAN BROTHERS HOLDINGS INC", "Lehman Brothers Holdings Inc") == 1.0
    assert name_similarity("KITE PHARMA INC", "Kite Pharma, Inc.") == 1.0
    assert name_similarity("BEAR STEARNS COMPANIES INC", "ELEMENTS BG Small Cap ETN") == 0.0
    assert name_similarity("", "X") == 0.0 and name_similarity(None, None) == 0.0


def test_delisted_universe_keeps_delisted_common_stocks_and_buckets_them():
    universe = delisted_universe(pd.concat([SHARADAR, TICKERS], ignore_index=True))
    assert sorted(universe["symbol"]) == ["FULL", "GONE", "LATE", "REUSE", "XYZ"]  # LIVE and the preferred are out
    expected = {"FULL": "2008-2011", "LATE": "2008-2011", "GONE": "2008-2011", "REUSE": "2017-2020", "XYZ": "2021+"}
    assert universe.set_index("symbol")["bucket"].to_dict() == expected


def test_sample_universe_is_stratified_and_reproducible():
    universe = delisted_universe(SHARADAR)
    one = sample_universe(universe, 1, seed=3)
    assert one["bucket"].tolist() == ["2008-2011", "2017-2020"]
    assert one.equals(sample_universe(universe, 1, seed=3))
    assert len(sample_universe(universe, None)) == 4 and len(sample_universe(universe, 50)) == 4


def test_find_candidates_ranks_by_name_and_caps_the_list():
    listed = pd.DataFrame(LISTED)
    listed["base"] = listed["Code"].str.split("_").str[0].str.upper()
    assert [code for code, _ in find_candidates("REUSE", "OLD REUSE HOLDINGS", listed)] == ["REUSE_old", "REUSE"]
    assert find_candidates("NOPE", "X", listed) == []
    assert find_candidates("full", "FULL HISTORY CORP", listed)[0] == ("FULL", 1.0)


def test_grade_full_partial_and_none():
    full = pd.DataFrame(_bars("2010-01-04", "2010-12-31"))
    full["date"] = pd.to_datetime(full["date"]).dt.date
    assert grade(date(2010, 1, 4), date(2010, 12, 31), full)["status"] == "full"
    late = grade(date(2009, 1, 2), date(2010, 12, 31), full)
    assert late["status"] == "partial" and late["start_gap_days"] == 367 and late["end_gap_days"] == 0
    early_end = grade(date(2010, 1, 4), date(2011, 6, 30), full)
    assert early_end["status"] == "partial" and early_end["end_gap_days"] == 181
    assert grade(date(2010, 1, 4), date(2010, 12, 31), pd.DataFrame())["status"] == "none"
    assert grade(date(2012, 1, 3), date(2012, 12, 31), full)["status"] == "none"  # bars exist, none in the range
    # a stock listed before Sharadar's own start is compared from 1997-12-31 on
    old = pd.DataFrame(_bars("1997-12-31", "1999-12-31"))
    old["date"] = pd.to_datetime(old["date"]).dt.date
    assert grade(date(1990, 1, 2), date(1999, 12, 31), old)["status"] == "full"


def test_run_check_grades_each_stock_and_prefers_the_matching_history():
    client = _client(_handler())
    result = run_check(client, SHARADAR, per_bucket=None).set_index("symbol")
    assert result["status"].to_dict() == {"FULL": "full", "LATE": "partial", "GONE": "none", "REUSE": "full"}
    assert result.loc["REUSE", "eodhd_code"] == "REUSE_old" and bool(result.loc["REUSE", "name_ok"])
    assert result["eodhd_code"].isna().to_dict()["GONE"] and not result.loc["GONE", "name_ok"]  # the ETF is not a candidate
    assert client.calls == 4  # list + FULL + LATE + REUSE_old (stops at the first full history)
    assert result.attrs["plan_limited"] is False


def test_a_listed_symbol_without_bars_counts_as_found_but_none():
    sharadar = pd.DataFrame([_ticker(301, "EMPTY", "EMPTY SHELL INC", "2005-01-03", "2005-12-30")])

    def handler(request: httpx.Request) -> httpx.Response:
        if "exchange-symbol-list" in request.url.path:
            return httpx.Response(200, json=[{"Code": "EMPTY", "Name": "Empty Shell Inc", "Exchange": "NYSE", "Type": "Common Stock"}])
        return httpx.Response(200, json=[])

    row = run_check(_client(handler), sharadar, per_bucket=None).iloc[0]
    assert row["eodhd_code"] == "EMPTY" and row["status"] == "none" and row["bars"] == 0 and bool(row["name_ok"])
    assert summarize(pd.DataFrame([row])).set_index("bucket").loc["all", "found"] == 1.0


def test_run_check_flags_a_limited_plan_and_respects_max_calls():
    assert run_check(_client(_handler(warning=True)), SHARADAR, per_bucket=None).attrs["plan_limited"] is True
    capped = run_check(_client(_handler()), SHARADAR, per_bucket=None, max_calls=2)
    assert len(capped) == 1
    with pytest.raises(EodhdError):
        run_check(_client(lambda r: httpx.Response(200, json=[])), SHARADAR)
    with pytest.raises(EodhdError):
        run_check(_client(lambda r: httpx.Response(401, text="Unauthenticated")), SHARADAR)


def test_run_check_compares_price_levels_with_fmp_when_available():
    index = pd.bdate_range("2010-01-04", "2010-12-31")

    def fmp_closes(security_id: int, start: date, end: date) -> Optional[pd.Series]:
        if security_id == 201:
            return pd.Series(10.0, index=index)
        if security_id == 202:
            return pd.Series(20.0, index=index)  # EODHD shows 10: a different price basis
        return None

    result = run_check(_client(_handler()), SHARADAR, per_bucket=None, fmp_closes=fmp_closes).set_index("symbol")
    assert result.loc["FULL", "fmp_close_ok"] == 1.0 and result.loc["FULL", "fmp_days"] == len(index)
    assert result.loc["LATE", "fmp_close_ok"] == 0.0 and result["fmp_close_ok"].isna().to_dict()["REUSE"]
    summary = summarize(result.reset_index()).set_index("bucket")
    assert summary.loc["2008-2011", "fmp_pairs"] == 2 and summary.loc["2008-2011", "fmp_close_ok"] == 0.5


def test_summarize_reports_shares_per_bucket_and_overall():
    summary = summarize(run_check(_client(_handler()), SHARADAR, per_bucket=None)).set_index("bucket")
    assert list(summary.index) == ["2008-2011", "2017-2020", "all"]
    first = summary.loc["2008-2011"].to_dict()
    assert first["stocks"] == 3 and first["found"] == pytest.approx(0.667) and first["full"] == pytest.approx(0.333)
    assert first["partial"] == pytest.approx(0.333) and first["none"] == pytest.approx(0.333)
    assert summary.loc["all", "full_name_ok"] == 0.5 and summary.loc["2017-2020", "full"] == 1.0
    assert summarize(pd.DataFrame()).empty


def test_retries_then_raises_on_server_errors():
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json=LISTED)

    assert len(_client(flaky).delisted_symbols()) == len(LISTED) and calls["n"] == 3
    with pytest.raises(EodhdError):
        _client(lambda r: httpx.Response(503)).delisted_symbols()


def test_cli_writes_detail_into_the_data_root_and_the_summary_beside_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = write_raw(tmp_path / config.DATA_DIR_NAME)
    SHARADAR.to_csv(root / config.RAW_DIR / "tickers.csv", index=False)
    monkeypatch.setattr(cli, "EodhdClient", lambda key: _client(_handler()))
    monkeypatch.setattr("src.config.provider_config.get_api_key", lambda name: "k")
    assert cli.main(["--data-root", str(root), "eodhd-check", "--all", "--no-fmp"]) == 0
    assert (root / config.REPORTS_DIR / "eodhd_check.csv").exists()
    summary = pd.read_csv(tmp_path / "eodhd-check" / "summary.csv")
    assert summary["bucket"].tolist() == ["2008-2011", "2017-2020", "all"]
    monkeypatch.setattr(cli, "EodhdClient", lambda key: _client(_handler(warning=True)))
    assert cli.main(["--data-root", str(root), "eodhd-check", "--per-bucket", "5", "--no-fmp"]) == 2
    monkeypatch.setattr("src.config.provider_config.get_api_key", lambda name: None)
    with pytest.raises(SystemExit):
        cli.main(["--data-root", str(root), "eodhd-check"])
