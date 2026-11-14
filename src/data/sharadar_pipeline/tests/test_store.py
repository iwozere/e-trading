"""Build the store from synthetic raw files and read it back; also the mapping, cross-check and P24 compatibility."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Optional, cast

import pandas as pd
import pytest

from src.data.fmp_pipeline.adjust.adjust import adjust
from src.data.sharadar_pipeline import config
from src.data.sharadar_pipeline.build import build_store, raw_path, read_raw
from src.data.sharadar_pipeline.crosscheck import adjustment_check, run_crosscheck, summarize, vendor_check
from src.data.sharadar_pipeline.mapping import map_to_fmp
from src.data.sharadar_pipeline.store import AmbiguousSymbol, SharadarStore
from src.data.sharadar_pipeline.tests.helpers import TICKERS, fmp_securities, write_raw
from src.ml.pipeline.p24_falling_knife.data import load_daily_batch
from src.ml.pipeline.p24_falling_knife.universe import is_common_stock, load_securities

START, END = date(2024, 1, 1), date(2024, 12, 31)


def _sid(store: SharadarStore, symbol: str, as_of: Optional[date] = None) -> int:
    found = store.resolve(symbol, as_of)
    assert found is not None
    return found.security_id


@pytest.fixture(params=[False, True], ids=["csv", "zip"])
def store(request, tmp_path: Path) -> SharadarStore:
    root = write_raw(tmp_path / config.DATA_DIR_NAME, zipped=request.param)
    build_store(root, fmp_securities())
    return SharadarStore(root)


def test_build_summary_counts(tmp_path: Path):
    root = write_raw(tmp_path / config.DATA_DIR_NAME)
    info = build_store(root, fmp_securities())
    assert info["securities"] == 3 and info["delisted"] == 1 and info["mapped_to_fmp"] == 1
    assert info["price_rows"] == 17 and info["price_securities"] == 3 and info["price_rows_unknown_ticker"] == 0
    assert info["tables"][config.FUNDAMENTALS] == 2
    assert info["tables"]["tables/sp500.parquet"] == 1
    assert (root / config.STORE_DIR / config.BUILD_INFO).exists()


def test_build_is_repeatable_and_leaves_raw_alone(tmp_path: Path):
    root = write_raw(tmp_path / config.DATA_DIR_NAME)
    before = sorted(p.name for p in (root / config.RAW_DIR).iterdir())
    first = build_store(root)
    second = build_store(root)
    assert first["price_rows"] == second["price_rows"]
    assert sorted(p.name for p in (root / config.RAW_DIR).iterdir()) == before
    assert not (root / config.STORE_DIR / "_staging_prices").exists()


def test_build_fails_without_core_tables(tmp_path: Path):
    root = write_raw(tmp_path / config.DATA_DIR_NAME)
    (root / config.RAW_DIR / "actions.csv").unlink()
    with pytest.raises(FileNotFoundError):
        build_store(root)
    assert raw_path(root, "actions") is None
    with pytest.raises(FileNotFoundError):
        read_raw(root, "actions")


def test_missing_store_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        SharadarStore(tmp_path / config.DATA_DIR_NAME)


def test_bulk_daily_returns_raw_bars_in_fmp_shape(store: SharadarStore):
    bars = store.bulk_daily([101, 102, 999], START, END)
    assert sorted(bars) == [101, 102]
    abc = bars[101]
    assert list(abc.columns) == ["open", "high", "low", "close", "volume", "flags", "source"]
    assert abc.index.name == "timestamp" and abc.index.is_monotonic_increasing
    assert abc["close"].tolist() == [100.0, 102.0, 104.0, 53.0, 54.0, 55.0]
    assert abc["volume"].tolist() == [1000.0] * 6
    assert (abc["flags"] == 0).all() and (abc["source"] == "sharadar").all()
    assert len(store.bulk_daily([101], date(2024, 1, 4), date(2024, 1, 5))[101]) == 2
    assert store.bulk_daily([], START, END) == {}
    full = store.prices(102)
    assert store.prices(101, "5m") is None and full is not None and len(full) == 5


def test_events_feed_adjust_and_reproduce_the_split_adjusted_close(store: SharadarStore):
    splits, dividends = store.events(101)
    assert splits.to_dict("records") == [{"date": date(2024, 1, 5), "numerator": 2.0, "denominator": 1.0}]
    assert dividends.to_dict("records") == [{"date": date(2024, 1, 3), "amount": 1.0}]
    adjusted = adjust(store.bulk_daily([101], START, END)[101][["open", "high", "low", "close", "volume"]], splits, dividends, mode="split")
    assert adjusted["close"].tolist() == [50.0, 51.0, 52.0, 53.0, 54.0, 55.0]
    assert store.events(102)[0].empty and store.events(102)[1].empty
    assert store.events_many([]) == {}
    assert set(store.events_many([101, 102])) == {101}


def test_resolve_by_traded_symbol_sharadar_ticker_and_former_symbol(store: SharadarStore):
    assert _sid(store, "XYZ") == 102
    assert _sid(store, "XYZ1") == 102
    assert _sid(store, "OLDA") == 101
    assert store.resolve("NOPE") is None
    assert store.security(101).fmp_security_id == 7 and store.security(102).fmp_security_id is None
    assert store.as_of == date(2024, 1, 9)
    assert [s.security_id for s in store.securities()] == [101, 102, 103]
    assert store.security(102).alive is False and store.security(101).covers(date(2024, 1, 5))


def test_resolve_needs_a_date_when_a_symbol_was_reused(tmp_path: Path):
    second = {"permaticker": "104", "ticker": "XYZ", "isdelisted": "N", "firstpricedate": "2025-01-02", "lastpricedate": "2025-06-30"}
    reused = TICKERS[TICKERS["permaticker"] == "102"].assign(**second)
    root = write_raw(tmp_path / config.DATA_DIR_NAME)
    pd.concat([TICKERS, reused]).to_csv(root / config.RAW_DIR / "tickers.csv", index=False)
    build_store(root)
    opened = SharadarStore(root)
    with pytest.raises(AmbiguousSymbol):
        opened.resolve("XYZ")
    assert _sid(opened, "XYZ", date(2024, 1, 4)) == 102
    assert _sid(opened, "XYZ", date(2025, 3, 3)) == 104


def test_fundamentals_are_as_reported_only(store: SharadarStore):
    arq = store.fundamentals([101], "ARQ", ["revenue"])
    assert arq["revenue"].tolist() == [1000.0]
    assert arq["available_from"].iloc[0] == pd.Timestamp("2024-01-04")
    assert store.fundamentals(dimension="ART")["revenue"].tolist() == [4000.0]
    assert store.fundamentals(dimension="MRQ").empty


def test_delisting_and_its_return(store: SharadarStore):
    event = store.delisting(102)
    assert event is not None and event.reason == "acquired" and event.date == date(2024, 1, 8) and event.acquirer == "ABC"
    # 10.00 cash + 0.1 ABC shares at 54.00 against a last close of 14.00
    assert store.delisting_return(102) == pytest.approx(0.1)
    assert store.delisting(101) is None and store.delisting_return(101) is None


def test_p24_helpers_run_on_the_store(store: SharadarStore):
    securities = load_securities(store.dir)
    common = [s for s in securities if is_common_stock(s)]
    assert sorted(s.symbol for s in common) == ["ABC", "XYZ"]
    assert [s.delisted for s in sorted(common, key=lambda s: s.symbol)] == [False, True]
    daily = load_daily_batch(cast(Any, store), [s.security_id for s in common], START, END)  # same reader shape as FmpStore
    assert daily[101]["close"].tolist() == [50.0, 51.0, 52.0, 53.0, 54.0, 55.0]
    assert daily[101]["close_raw"].iloc[0] == 100.0


def test_map_to_fmp_prefers_the_same_symbol_and_is_one_to_one(store: SharadarStore):
    import pyarrow.parquet as pq

    securities = pq.read_table(store.dir / config.SECURITIES).to_pandas()
    mapped = dict(zip(securities["security_id"], map_to_fmp(securities, fmp_securities())))
    # ABCB (id 8) overlaps longer, but ABC must land on ABC (id 7); the preferred is never mapped
    assert mapped[101] == 7 and pd.isna(mapped[102]) and pd.isna(mapped[103])
    shorter = fmp_securities().assign(symbol=["AAA", "BBB", "CCC"])
    shorter.loc[0, "last_price_date"] = date(2024, 1, 5)
    assert map_to_fmp(securities, shorter).dropna().tolist() == [8]  # no symbol match: the longer overlap wins


def test_adjustment_check_matches_the_vendor_adjusted_close(store: SharadarStore):
    out = adjustment_check(store).set_index("security_id")
    assert float(out["max_abs_dev"].loc[101]) < 1e-9 and out.loc[101, "splits"] == 1 and out.loc[101, "dividends"] == 1
    assert out.loc[102, "share_ok"] == 1.0


class _FakeFmp:
    """FMP stand-in: ABC (id 7) with the same raw closes, one of them off by 10%."""

    def bulk_daily(self, security_ids, start, end):
        index = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]), name="timestamp")
        frame = pd.DataFrame({"close": [100.0, 102.0, 104.0, 53.0, 59.4], "volume": [1000.0] * 5}, index=index)
        return {7: frame} if 7 in list(security_ids) else {}

    def events_many(self, security_ids):
        splits = pd.DataFrame({"date": [date(2024, 1, 5)], "numerator": [2.0], "denominator": [1.0]})
        return {7: (splits, pd.DataFrame({"date": [], "amount": []}))}


def test_vendor_check_compares_mapped_pairs(store: SharadarStore):
    out = vendor_check(store, _FakeFmp())
    assert len(out) == 1
    row = out.iloc[0]
    assert row["common_days"] == 5 and row["median_close_ratio"] == 1.0 and row["share_close_ok"] == pytest.approx(0.8)
    assert (row["splits_sharadar"], row["splits_fmp"], row["splits_matched"]) == (1, 1, 1)


def test_run_crosscheck_writes_reports_inside_the_data_root(store: SharadarStore):
    summary = run_crosscheck(store, _FakeFmp())
    reports = store.data_root / config.REPORTS_DIR
    assert {p.name for p in reports.iterdir()} == {"adjustment_check.csv", "vendor_check.csv", "crosscheck_summary.json"}
    assert summary["adjustment"]["share_fully_ok"] == 1.0 and summary["vendor"]["splits_matched"] == 1
    assert "vendor" not in run_crosscheck(store)
    assert summarize(pd.DataFrame(), pd.DataFrame()) == {"adjustment": {"securities": 0}, "vendor": {"pairs": 0}}
