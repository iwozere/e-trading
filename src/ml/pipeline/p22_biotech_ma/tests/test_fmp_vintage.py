"""Tests for ingest/fmp_vintage.py — vintage matching, stitching and row building."""

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fmp_pipeline.builders.flags import OHLC_VIOLATION, SPIKE_REVERTED, STALE, ZERO_VOLUME
from src.ml.pipeline.p22_biotech_ma.ingest import fmp_vintage
from src.ml.pipeline.p22_biotech_ma.ingest.fmp_vintage import (
    CompanyMatch,
    VintageSecurity,
    build_corporate_action_rows,
    build_price_rows,
    is_common_listing,
    load_company,
    match_companies,
    name_key,
    price_known_from,
    stitch_daily_bars,
)


def _sec(security_id: int, symbol: str, *, cik: Optional[str] = "0000000001", name: str = "Acme Bio, Inc.",
         instrument_type: str = "stock", first: Optional[date] = date(2015, 1, 2), last: date = date(2026, 10, 16),
         uncertain: bool = False) -> VintageSecurity:
    return VintageSecurity(security_id, symbol, name, cik, instrument_type, first, last, uncertain)


def _bars(days: List[str], close: float = 10.0, flags: int = 0) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(days)).rename("timestamp")
    n = len(days)
    return pd.DataFrame({"open": [close] * n, "high": [close + 1] * n, "low": [close - 1] * n, "close": [close] * n,
                         "volume": [1000] * n, "flags": [flags] * n, "source": ["fmp"] * n}, index=index)


@pytest.mark.parametrize(
    "security,expected",
    [
        (_sec(1, "ACME"), True),
        (_sec(1, "TSM", instrument_type="adr"), True),
        (_sec(1, "AIG-PA"), False),
        (_sec(1, "AP-WT"), False),
        (_sec(1, "ACMEW"), False),
        (_sec(1, "ACMEU"), False),
        (_sec(1, "BGMSP"), False),
        (_sec(1, "ACMP"), True),
        (_sec(1, "ACME", instrument_type="warrant"), False),
        (_sec(1, "ACME", instrument_type="etf"), False),
        (_sec(1, "ACME", uncertain=True), False),
        (_sec(1, "ACME", first=None), False),
    ],
)
def test_is_common_listing(security, expected):
    """Only plain common/ADR listings with price history count; preferreds, warrants and units do not."""
    assert is_common_listing(security) is expected


def test_match_by_cik_orders_latest_listing_first():
    """A CIK with an old and a new ticker (FMP keeps a ticker change as two securities) is primary-first."""
    securities = [
        _sec(10, "OLDT", last=date(2022, 11, 4)),
        _sec(11, "NEWT", first=date(2022, 11, 7)),
        _sec(12, "NEWT-P"),
    ]
    matches, unmatched = match_companies([{"company_id": 1, "cik": "1", "name": "Whatever"}], securities)

    assert unmatched == []
    assert matches[1] == CompanyMatch(1, (11, 10), "cik")


def test_match_falls_back_to_exact_name_without_conflicting_cik():
    """
    Celgene has no CIK in the vintage, so its CIK lookup misses; the exact normalized name still finds it —
    but never a same-named listing that belongs to a different CIK.
    """
    securities = [
        _sec(20, "CELG", cik=None, name="Celgene Corporation"),
        _sec(21, "CELX", cik="0000999999", name="Celgene Corp"),
    ]
    matches, unmatched = match_companies([{"company_id": 2, "cik": "0000816284", "name": "CELGENE CORP"}], securities)

    assert unmatched == []
    assert matches[2] == CompanyMatch(2, (20,), "name")


def test_name_key_strips_legal_form_but_not_business_words():
    """Corp/Corporation/Inc collapse; Therapeutics vs Pharmaceuticals stay distinct companies."""
    assert name_key("Celgene Corporation") == name_key("CELGENE CORP") == "celgene"
    assert name_key("Acme Therapeutics, Inc.") != name_key("Acme Pharmaceuticals, Inc.")


def test_match_leaves_unknown_company_unmatched():
    """No CIK and no exact name match means unmatched — no fuzzy guess."""
    matches, unmatched = match_companies([{"company_id": 3, "cik": "42", "name": "Kite Pharma, Inc."}],
                                         [_sec(30, "KITX", cik="0000000043", name="Kite Realty")])

    assert matches == {}
    assert unmatched == [3]


def test_stitch_takes_older_listing_only_before_primary():
    """The older listing fills dates before the primary's first bar and never interleaves inside it."""
    primary = _bars(["2022-11-07", "2022-11-08", "2022-11-10"], close=20.0)
    older = _bars(["2022-11-03", "2022-11-04", "2022-11-09"], close=5.0)

    bars = stitch_daily_bars([(11, primary), (10, older), (12, None)])

    assert [d.strftime("%Y-%m-%d") for d in bars.index] == ["2022-11-03", "2022-11-04", "2022-11-07", "2022-11-08",
                                                            "2022-11-10"]
    assert bars["security_id"].tolist() == [10, 10, 11, 11, 11]


def test_stitch_with_no_frames_is_empty():
    """No price files at all gives an empty frame, not an error."""
    assert stitch_daily_bars([(1, None)]).empty


def test_build_price_rows_skips_bad_prints_and_keeps_known_from_on_trade_date():
    """
    Bad prints (OHLC violation, reverted spike) are never written — p22_price_daily is never rewritten;
    stale and zero-volume days are real trading days and are kept.
    """
    bars = pd.concat([
        _bars(["2024-06-06"], flags=OHLC_VIOLATION),
        _bars(["2024-06-07"], flags=SPIKE_REVERTED | STALE),
        _bars(["2024-06-10"], flags=STALE | ZERO_VOLUME),
        _bars(["2024-06-11"]),
    ]).assign(security_id=7)

    rows, skipped = build_price_rows(5, bars)

    assert skipped == 2
    assert [r["trade_date"] for r in rows] == [date(2024, 6, 10), date(2024, 6, 11)]
    assert rows[0]["vendor"] == "fmp"
    assert rows[0]["known_from"] == datetime(2024, 6, 10, 21, 0, tzinfo=timezone.utc)


def test_price_known_from_is_trade_date_not_load_date():
    """
    Regression guard: stamping the load date would hide every backfilled bar from a backtest whose as_of is
    before today (P22Repo.get_raw_close filters on known_from <= as_of).
    """
    assert price_known_from(date(2019, 3, 1)).date() == date(2019, 3, 1)


def _events(splits: List[Tuple[str, float, float]], dividends: List[Tuple[str, float]]):
    return (pd.DataFrame({"date": [pd.Timestamp(d).date() for d, _, _ in splits],
                          "numerator": [n for _, n, _ in splits], "denominator": [m for _, _, m in splits]}),
            pd.DataFrame({"date": [pd.Timestamp(d).date() for d, _ in dividends],
                          "amount": [a for _, a in dividends]}))


def test_build_corporate_action_rows_converts_ratios_and_skips_existing():
    """Forward and reverse splits keep P22's ratio convention; rows the company already has are left alone."""
    bars = _bars(["2020-01-02", "2024-06-12"]).assign(security_id=7)
    events = {7: _events([("2021-07-20", 4.0, 1.0), ("2023-05-01", 1.0, 20.0), ("2022-01-03", 1.0, 1.0)],
                         [("2022-03-01", 0.5), ("2023-03-01", 0.6)])}
    existing = {(date(2023, 3, 1), "dividend")}

    rows = build_corporate_action_rows(5, events, bars, existing)

    by_key = {(r["ex_date"], r["action_type"]): r for r in rows}
    assert set(by_key) == {(date(2021, 7, 20), "split"), (date(2023, 5, 1), "reverse_split"),
                           (date(2022, 3, 1), "dividend")}
    assert by_key[(date(2021, 7, 20), "split")]["ratio"] == 4.0
    assert by_key[(date(2023, 5, 1), "reverse_split")]["ratio"] == pytest.approx(0.05)
    assert by_key[(date(2022, 3, 1), "dividend")]["cash_amount"] == 0.5
    assert all(r["source"] == "fmp" and r["is_verified"] is False for r in rows)
    assert by_key[(date(2021, 7, 20), "split")]["known_from"] == datetime(2021, 7, 20, tzinfo=timezone.utc)


def test_build_corporate_action_rows_ignores_events_outside_contributed_range():
    """An older listing's events after it stopped contributing bars do not leak into the company."""
    bars = pd.concat([_bars(["2019-01-02", "2019-06-28"]).assign(security_id=10),
                      _bars(["2019-07-01", "2024-01-02"]).assign(security_id=11)])
    events = {10: _events([("2019-06-29", 2.0, 1.0), ("2022-02-01", 3.0, 1.0)], []),
              11: _events([("2023-01-03", 5.0, 1.0)], [])}

    rows = build_corporate_action_rows(5, events, bars, set())

    assert sorted(r["ex_date"] for r in rows) == [date(2019, 6, 29), date(2023, 1, 3)]


class _FakeStore:
    def __init__(self, prices: Dict[int, pd.DataFrame], events: Dict[int, Any]) -> None:
        self._prices = prices
        self._events = events
        self.event_requests: List[List[int]] = []

    def prices(self, security_id: int, timeframe: str) -> Optional[pd.DataFrame]:
        assert timeframe == "1d"
        return self._prices.get(security_id)

    def events_many(self, security_ids):
        self.event_requests.append(list(security_ids))
        return {sid: self._events[sid] for sid in security_ids if sid in self._events}


class _FakeRepo:
    def __init__(self) -> None:
        self.price_rows: List[Dict[str, Any]] = []
        self.actions: List[Dict[str, Any]] = []
        self.deleted: List[tuple] = []

    def delete_price_daily(self, company_id, vendor):
        self.deleted.append((company_id, vendor))
        return 7

    def insert_price_daily_bulk(self, rows):
        self.price_rows.extend(rows)
        return len(rows)

    def get_corporate_action_keys(self, company_id):
        del company_id
        return set()

    def upsert_corporate_action(self, **row):
        self.actions.append(row)


def test_load_company_writes_stitched_history():
    """End to end over fakes: bars from both listings, events read only for listings that contributed."""
    store = _FakeStore(
        prices={11: _bars(["2023-01-03", "2023-01-04"]), 10: _bars(["2022-12-30"]), 12: _bars(["2023-01-05"])},
        events={11: _events([("2023-01-04", 2.0, 1.0)], [])},
    )
    repo = _FakeRepo()

    result = load_company(store, repo, CompanyMatch(5, (11, 10, 12), "cik"))

    assert result == {"prices_deleted": 0, "prices_written": 3, "bars_skipped": 0, "actions_written": 1,
                      "listings_used": 2}
    assert store.event_requests == [[10, 11]]
    assert repo.actions[0]["action_type"] == "split"
    assert repo.deleted == []


def test_load_company_replace_prices_deletes_fmp_bars_first():
    """replace_prices deletes the company's FMP rows, then writes the (corrected) bars."""
    store = _FakeStore(prices={11: _bars(["2023-01-03"])}, events={})
    repo = _FakeRepo()

    result = load_company(store, repo, CompanyMatch(5, (11,), "cik"), replace_prices=True)

    assert repo.deleted == [(5, "fmp")]
    assert result["prices_deleted"] == 7 and result["prices_written"] == 1


def test_cli_replace_prices_requires_company_ids(tmp_path):
    """A whole-universe delete-and-rewrite is refused before anything is opened."""
    from src.ml.pipeline.p22_biotech_ma.cli import fmp_vintage_load_cli

    assert fmp_vintage_load_cli.run(tmp_path, dry_run=False, limit=None, reload=False, company_ids=[],
                                    replace_prices=True) == 2


def test_vendor_is_fmp():
    """p22_price_daily.vendor / p22_corporate_action.source use the existing 'fmp' value."""
    assert fmp_vintage.VENDOR == "fmp"
