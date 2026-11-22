"""Build the store from mock downloads and read it back; also the FMP mapping, cross-check, CLI and P24 compatibility."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from src.data.eodhd_pipeline import cli, config
from src.data.eodhd_pipeline.build import build_store, coverage_by_year
from src.data.eodhd_pipeline.crosscheck import adjustment_check, run_crosscheck, summarize, vendor_check
from src.data.eodhd_pipeline.download import download_histories
from src.data.eodhd_pipeline.mapping import map_to_fmp, name_similarity
from src.data.eodhd_pipeline.store import AmbiguousSymbol, EodhdStore
from src.data.eodhd_pipeline.tests.helpers import FakeEodhd, make_client
from src.data.eodhd_pipeline.universe import build_universe
from src.ml.pipeline.p24_falling_knife.data import load_daily_batch
from src.ml.pipeline.p24_falling_knife.universe import is_common_stock, load_securities

START, END = date(2024, 1, 1), date(2024, 12, 31)


def fmp_securities() -> pd.DataFrame:
    """FMP vintage stand-in: ABC (id 7) overlaps; XYZ (id 9) is a later company on the reused ticker; an ETF named ABC."""
    return pd.DataFrame([
        {"security_id": 7, "symbol": "ABC", "name": "ABC Corporation", "instrument_type": "stock",
         "first_price_date": date(2020, 1, 2), "last_price_date": date(2026, 10, 30)},
        {"security_id": 9, "symbol": "XYZ", "name": "XYZ Acquisition Corp", "instrument_type": "stock",
         "first_price_date": date(2025, 3, 3), "last_price_date": date(2026, 10, 30)},
        {"security_id": 11, "symbol": "ABC", "name": "ABC Fund", "instrument_type": "etf",
         "first_price_date": date(2020, 1, 2), "last_price_date": date(2026, 10, 30)},
    ])


def _download(root: Path, fake: FakeEodhd | None = None, allow_limited: bool = False) -> pd.DataFrame:
    universe = build_universe(make_client(FakeEodhd()), root)
    download_histories(make_client(fake or FakeEodhd()), root, universe, allow_limited=allow_limited)
    return universe


@pytest.fixture
def store(tmp_path: Path) -> EodhdStore:
    _download(tmp_path)
    build_store(tmp_path, fmp_securities())
    return EodhdStore(tmp_path)


def _sid(store: EodhdStore, symbol: str) -> int:
    found = store.resolve(symbol)
    assert found is not None
    return found.security_id


def test_build_summary(tmp_path: Path):
    _download(tmp_path)
    info = build_store(tmp_path, fmp_securities())
    assert info["universe"] == 3 and info["securities_with_bars"] == 2 and info["price_rows"] == 11
    assert info["splits"] == 1 and info["dividends"] == 1 and info["mapped_to_fmp"] == 1 and info["delisted_with_bars"] == 1
    assert info["coverage_by_last_trade_year"] == [{"bucket": "2021+", "delisted_with_bars": 1, "also_in_fmp": 0, "new_vs_fmp": 1}]
    assert not (tmp_path / config.STORE_DIR / "_staging_prices").exists()
    assert build_store(tmp_path)["price_rows"] == 11  # repeatable, no FMP


def test_bulk_daily_returns_raw_bars_in_fmp_shape(store: EodhdStore):
    abc, xyz = _sid(store, "ABC"), _sid(store, "XYZ")
    bars = store.bulk_daily([abc, xyz, 5], START, END)
    assert sorted(bars) == sorted([abc, xyz])
    frame = bars[abc]
    assert list(frame.columns) == ["open", "high", "low", "close", "volume", "flags", "source"]
    assert frame.index.name == "timestamp" and frame["close"].tolist() == [100.0, 102.0, 104.0, 53.0, 54.0, 55.0]
    assert frame["volume"].tolist() == [1000.0] * 6 and (frame["source"] == "eodhd").all()
    assert "closeadj_vendor" in store.bulk_daily([abc], START, END, vendor_adjusted=True)[abc].columns
    assert store.bulk_daily([], START, END) == {}
    full = store.prices(abc)
    assert store.prices(abc, "5m") is None and full is not None and len(full) == 6


def test_events_and_securities(store: EodhdStore):
    abc, xyz = _sid(store, "ABC"), _sid(store, "XYZ_old")
    splits, dividends = store.events(abc)
    assert splits.to_dict("records") == [{"date": date(2024, 1, 5), "numerator": 2.0, "denominator": 1.0}]
    assert dividends.to_dict("records") == [{"date": date(2024, 1, 3), "amount": 1.0}]
    assert store.events(xyz)[0].empty and store.events_many([]) == {}
    info = store.security(abc)
    assert info.fmp_security_id == 7 and info.alive and info.isin == "US0000000101" and info.covers(date(2024, 1, 5))
    old = store.security(xyz)
    assert not old.alive and old.symbol == "XYZ" and old.last_price_date == date(2024, 1, 8) and old.fmp_security_id is None
    gone = store.security(_sid(store, "GONE"))
    assert gone.first_price_date is None and gone.last_price_date is None
    assert store.as_of == date(2024, 1, 9) and store.resolve("NOPE") is None
    assert [s.security_id for s in store.securities()] == sorted(s.security_id for s in store.securities())


def test_p24_helpers_run_on_the_store(store: EodhdStore):
    common = [s for s in load_securities(store.dir) if is_common_stock(s)]
    assert sorted(s.symbol for s in common) == ["ABC", "XYZ"]  # GONE has no bars
    assert {s.symbol: s.delisted for s in common} == {"ABC": False, "XYZ": True}
    daily = load_daily_batch(cast(Any, store), [s.security_id for s in common], START, END)  # same reader shape as FmpStore
    abc = _sid(store, "ABC")
    assert daily[abc]["close"].tolist() == [50.0, 51.0, 52.0, 53.0, 54.0, 55.0] and daily[abc]["close_raw"].iloc[0] == 100.0


def test_resolve_needs_a_date_for_a_reused_symbol(tmp_path: Path):
    _download(tmp_path)
    build_store(tmp_path)
    universe = pd.read_parquet(tmp_path / config.UNIVERSE_FILE)
    store = EodhdStore(tmp_path)
    store._by_symbol.setdefault("XYZ", []).append(int(universe.loc[universe["eodhd_code"] == "ABC", "security_id"].iloc[0]))
    with pytest.raises(AmbiguousSymbol):
        store.resolve("XYZ")
    assert store.resolve("XYZ", date(2024, 1, 9)) is not None  # only ABC trades on day 6


def test_missing_store_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        EodhdStore(tmp_path)


def test_mapping_needs_overlap_and_a_name_or_a_single_long_candidate():
    ours = pd.DataFrame([
        {"security_id": 1, "symbol": "ABC", "name": "ABC Corp", "first_price_date": date(2024, 1, 2), "last_price_date": date(2024, 1, 9)},
        {"security_id": 2, "symbol": "XYZ", "name": "XYZ Inc", "first_price_date": date(2010, 1, 4), "last_price_date": date(2024, 1, 8)},
        {"security_id": 3, "symbol": "BRK.B", "name": "Berkshire", "first_price_date": date(2010, 1, 4), "last_price_date": date(2024, 1, 8)},
        {"security_id": 4, "symbol": "NEW", "name": "Renamed Co", "first_price_date": date(2010, 1, 4), "last_price_date": date(2024, 1, 8)},
    ])
    fmp = pd.DataFrame([
        {"security_id": 7, "symbol": "ABC", "name": "ABC Corporation", "instrument_type": "stock",
         "first_price_date": date(2020, 1, 2), "last_price_date": date(2026, 1, 2)},
        {"security_id": 9, "symbol": "XYZ", "name": "Other Name", "instrument_type": "stock",
         "first_price_date": date(2023, 12, 1), "last_price_date": date(2026, 1, 2)},
        {"security_id": 10, "symbol": "BRK-B", "name": "Berkshire Hathaway", "instrument_type": "stock",
         "first_price_date": date(2000, 1, 3), "last_price_date": date(2026, 1, 2)},
        {"security_id": 12, "symbol": "NEW", "name": "Old Name Inc", "instrument_type": "stock",
         "first_price_date": date(2000, 1, 3), "last_price_date": date(2026, 1, 2)},
    ])
    mapped = map_to_fmp(ours, fmp).tolist()
    # ABC: name; XYZ: other name, overlap too short; BRK.B ~ BRK-B; NEW: only candidate, long overlap
    assert mapped[0] == 7 and pd.isna(mapped[1]) and mapped[2] == 10 and mapped[3] == 12
    assert map_to_fmp(ours, fmp.assign(symbol="ZZZ")).isna().all()
    assert name_similarity("Kite Pharma, Inc.", "KITE PHARMA INC") == 1.0 and name_similarity(None, "x") == 0.0


def test_coverage_by_year_counts_delisted_only():
    securities = pd.DataFrame({"is_actively_trading": [False, False, True, False],
                               "last_price_date": [date(2005, 3, 1), date(2005, 6, 1), date(2024, 1, 1), None],
                               "fmp_security_id": pd.array([7, pd.NA, pd.NA, pd.NA], dtype="Int64")})
    assert coverage_by_year(securities) == [{"bucket": "2003-2007", "delisted_with_bars": 2, "also_in_fmp": 1, "new_vs_fmp": 1}]


def test_adjustment_check_matches_eodhd_adjusted_close(store: EodhdStore):
    out = adjustment_check(store).set_index("symbol")
    assert out.loc["ABC", "max_abs_dev"] == pytest.approx(0.0, abs=1e-9) and out.loc["ABC", "splits"] == 1 and out.loc["ABC", "dividends"] == 1
    assert out.loc["XYZ", "share_ok"] == 1.0


def test_a_missing_dividend_shows_in_the_adjustment_check(tmp_path: Path):
    class NoDividends(FakeEodhd):
        def __call__(self, request):
            if "/div/" in request.url.path:
                self.requests.append(request.url.path)
                return __import__("httpx").Response(200, json=[])
            return super().__call__(request)

    _download(tmp_path, NoDividends())
    build_store(tmp_path)
    out = adjustment_check(EodhdStore(tmp_path)).set_index("symbol")
    assert out.loc["ABC", "max_abs_dev"] == pytest.approx(0.0101, abs=1e-4) and out.loc["ABC", "worst_date"] == date(2024, 1, 2)


class _FakeFmp:
    """FMP stand-in for ABC (id 7): same raw closes but one 10% off, same split, the dividend one day later."""

    def bulk_daily(self, security_ids, start, end):
        index = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]), name="timestamp")
        frame = pd.DataFrame({"close": [100.0, 102.0, 104.0, 53.0, 59.4], "volume": [1000.0] * 5}, index=index)
        return {7: frame} if 7 in list(security_ids) else {}

    def events_many(self, security_ids):
        splits = pd.DataFrame({"date": [date(2024, 1, 5)], "numerator": [2.0], "denominator": [1.0]})
        return {7: (splits, pd.DataFrame({"date": [date(2024, 1, 4)], "amount": [1.0]}))}


def test_vendor_check_and_reports(store: EodhdStore):
    row = vendor_check(store, _FakeFmp()).iloc[0]
    assert row["common_days"] == 5 and row["share_close_ok"] == pytest.approx(0.8) and row["median_volume_ratio"] == 1.0
    assert (row["splits_eodhd"], row["splits_fmp"], row["splits_matched"]) == (1, 1, 1)
    assert (row["dividends_eodhd"], row["dividends_fmp"], row["dividends_matched"]) == (1, 1, 0)
    summary = run_crosscheck(store, _FakeFmp())
    reports = store.data_root / config.REPORTS_DIR
    assert {p.name for p in reports.iterdir()} == {"adjustment_check.csv", "vendor_check.csv", "crosscheck_summary.json"}
    assert summary["adjustment"]["share_fully_ok"] == 1.0 and summary["vendor"]["splits_matched"] == 1
    assert "vendor" not in run_crosscheck(store)
    assert summarize(pd.DataFrame(), pd.DataFrame()) == {"adjustment": {"securities": 0}, "vendor": {"pairs": 0}}


def test_cli_end_to_end_and_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / config.DATA_DIR_NAME
    fake = {"handler": FakeEodhd()}
    monkeypatch.setattr(cli, "_client", lambda: make_client(fake["handler"]))
    base = ["--data-root", str(root)]
    assert cli.main(base + ["status"]) == 0  # no universe yet
    assert cli.main(base + ["quota"]) == 0
    assert cli.main(base + ["universe"]) == 0
    assert cli.main(base + ["download", "--max-calls", "4", "--workers", "1"]) == cli.EXIT_STOPPED_EARLY
    assert cli.main(base + ["download"]) == 0
    assert cli.main(base + ["build", "--no-fmp"]) == 0
    assert cli.main(base + ["crosscheck", "--no-fmp"]) == 0
    assert cli.main(base + ["status"]) == 0
    fake["handler"] = FakeEodhd(free=True)
    assert cli.main(base + ["download", "--force"]) == cli.EXIT_PLAN_LIMITED
    assert cli.main(base + ["download", "--force", "--allow-limited"]) == cli.EXIT_PLAN_LIMITED
    assert cli.main(base + ["status"]) == 0
    monkeypatch.setattr(cli, "_open_fmp", lambda root: None)
    assert cli.main(base + ["build"]) == 0


def test_cli_without_a_key_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(config, "api_key", lambda: None)
    with pytest.raises(SystemExit):
        cli.main(["--data-root", str(tmp_path), "quota"])
    monkeypatch.setattr(config, "api_key", lambda: "k")
    assert isinstance(cli._client(), cli.EodhdClient)
