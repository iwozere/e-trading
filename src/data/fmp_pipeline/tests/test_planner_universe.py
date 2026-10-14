"""Tests for universe assembly and tiering (spec §7.2, §6.3.3, plan WP-4). All symbols are synthetic."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Optional

from src.data.fmp_pipeline.planner.config import UniverseConfig
from src.data.fmp_pipeline.planner.universe import assemble_universe, normalize_exchange

AS_OF = date(2026, 10, 14)
CONFIG = UniverseConfig(tier_a=("SPY", "^GSPC"), indices=("^GSPC",))


def _profile(exchange: str = "NASDAQ", **flags: Any) -> dict[str, Any]:
    return {"exchange": exchange, "isEtf": False, "isFund": False, "isAdr": False, "isActivelyTrading": True,
            "ipoDate": "2001-02-03", **flags}


def _build(sources: Mapping[str, Any], profiles: Optional[dict[str, Any]] = None,
           config: UniverseConfig = CONFIG):
    return assemble_universe(sources, profiles or {}, config, AS_OF)


def test_normalize_exchange():
    assert normalize_exchange("NYSE American") == "NYSEAMERICAN"
    assert normalize_exchange("nyse-arca") == "NYSEARCA"
    assert normalize_exchange("Cboe_BZX.") == "CBOEBZX"
    assert normalize_exchange("") is None and normalize_exchange(None) is None and normalize_exchange(3) is None


def test_delisted_companies_filtered_by_exchange_and_otc():
    universe = _build({"delisted-companies": [
        {"symbol": "DUSA", "exchange": "NASDAQ", "ipoDate": "1999-01-04", "delistedDate": "2010-05-03"},
        {"symbol": "DLSE", "exchange": "LSE", "delistedDate": "2012-01-02"},
        {"symbol": "DOTC", "exchange": "OTC", "delistedDate": "2015-01-02"},
        {"symbol": "DSOON", "exchange": "NYSE", "delistedDate": "2026-12-30"},  # scheduled, not yet delisted
    ]})
    assert set(universe.securities) == {"DUSA", "DSOON", "SPY", "^GSPC"}
    assert universe.excluded == {"exchange": 1, "otc": 1}
    dusa = universe.securities["DUSA"]
    assert dusa.tier == "D" and dusa.is_delisted(AS_OF) and dusa.ipo_date == date(1999, 1, 4)
    soon = universe.securities["DSOON"]
    assert soon.delisted_date is None and not soon.is_delisted(AS_OF) and soon.tier == "E"
    assert universe.exchanges == {"NASDAQ": 1, "LSE": 1, "OTC": 1, "NYSE": 1}


def test_include_otc_keeps_otc_symbols():
    config = UniverseConfig(tier_a=(), indices=(), include_otc=True)
    universe = _build({"delisted-companies": [{"symbol": "DOTC", "exchange": "PNK", "delistedDate": "2015-01-02"}]},
                      config=config)
    assert "DOTC" in universe.securities and not universe.excluded


def test_stock_list_without_exchange_waits_for_profile():
    sources = {"stock-list": [{"symbol": "AAA", "companyName": "A"}, {"symbol": "BBB", "companyName": "B"},
                              {"symbol": "CCC", "companyName": "C"}, {"symbol": "DDD", "companyName": "D"}]}
    first = _build(sources)
    assert {s.symbol for s in first.unresolved} == {"AAA", "BBB", "CCC", "DDD"}
    assert {s.symbol for s in first.resolved} == {"SPY", "^GSPC"}

    profiles = {"AAA": _profile(), "BBB": _profile("XETRA"), "CCC": None}
    second = _build(sources, profiles)
    assert second.securities["AAA"].resolved and second.securities["AAA"].tier == "C"
    assert "BBB" not in second.securities and "CCC" not in second.securities
    assert second.excluded == {"exchange": 1, "no_profile": 1}
    assert [s.symbol for s in second.unresolved] == ["DDD"]

    kept = _build(sources, profiles, UniverseConfig(tier_a=(), indices=(), keep_unresolved=True))
    assert kept.securities["CCC"].resolved and kept.securities["CCC"].tier == "E"


def test_list_exchange_and_type_fields_resolve_without_profile():
    universe = _build({"stock-list": [{"symbol": "AAA", "exchangeShortName": "NYSE", "type": "stock"},
                                      {"symbol": "FFF", "exchangeShortName": "NASDAQ", "type": "fund"},
                                      {"symbol": "TTT", "exchangeShortName": "NYSE", "type": "trust"}]})
    assert universe.securities["AAA"].resolved and universe.securities["AAA"].tier == "C"
    assert "FFF" not in universe.securities and universe.excluded == {"fund": 1}
    assert universe.securities["TTT"].instrument == "trust"


def test_profile_overrides_list_attributes():
    universe = _build({"stock-list": [{"symbol": "AAA", "exchangeShortName": "NYSE"}]},
                      {"AAA": _profile("NASDAQ", isAdr=True, ipoDate="2010-01-04")})
    sec = universe.securities["AAA"]
    assert sec.exchange == "NASDAQ" and sec.is_adr and sec.ipo_date == date(2010, 1, 4) and sec.profile == "ok"


def test_profile_fund_is_excluded_but_etf_is_kept():
    universe = _build({"etf-list": [{"symbol": "EEE"}], "stock-list": [{"symbol": "MMM"}]},
                      {"EEE": _profile("NYSEARCA", isEtf=True, isFund=True), "MMM": _profile(isFund=True)})
    assert universe.securities["EEE"].is_etf and universe.securities["EEE"].tier == "E"
    assert "MMM" not in universe.securities and universe.excluded == {"fund": 1}


def test_actively_trading_list_marks_inactive_symbols_delisted():
    sources = {"stock-list": [{"symbol": "AAA", "exchangeShortName": "NYSE", "type": "stock"},
                              {"symbol": "ZZZ", "exchangeShortName": "NYSE", "type": "stock"}],
               "actively-trading-list": [{"symbol": "AAA"}]}
    universe = _build(sources)
    assert universe.securities["AAA"].tier == "C"
    assert universe.securities["ZZZ"].tier == "D"
    # A profile saying the symbol trades wins over the list.
    assert _build(sources, {"ZZZ": _profile("NYSE")}).securities["ZZZ"].tier == "C"


def test_symbol_change_adds_both_sides_and_constituents_make_tier_b():
    universe = _build({
        "symbol-change": [{"oldSymbol": "OLD1", "newSymbol": "NEW1", "date": "2020-01-02"}],
        "historical-sp500-constituent": [{"symbol": "IN1", "removedTicker": "OUT1"}],
        "nasdaq-constituent": [{"symbol": "IN2"}],
        "delisted-companies": [{"symbol": "OUT1", "exchange": "NYSE", "delistedDate": "2019-03-04"}],
    })
    assert {"OLD1", "NEW1"} <= {s.symbol for s in universe.unresolved}
    tiers = {s.symbol: s.tier for s in universe.securities.values()}
    assert tiers["OUT1"] == "B"  # a former constituent is B even though delisted
    assert universe.securities["IN1"].index_member and universe.securities["IN2"].index_member
    assert universe.securities["OUT1"].sources == {"historical-sp500-constituent", "delisted-companies"}


def test_explicit_tier_a_and_indices_are_always_included():
    universe = _build({"delisted-companies": [{"symbol": "SPY", "exchange": "LSE"}]})
    assert universe.securities["SPY"].tier == "A" and universe.securities["SPY"].explicit
    assert universe.securities["^GSPC"].is_index and not universe.securities["^GSPC"].is_stock
    assert not universe.excluded


def test_garbage_rows_are_ignored():
    universe = _build({"stock-list": [{"symbol": ""}, {"symbol": None}, "x", {"name": "no symbol"}, {"symbol": " A "}],
                       "delisted-companies": None, "etf-list": {"symbol": "ONE", "exchange": "NYSE"}})
    assert "A" in universe.securities and "ONE" in universe.securities
    assert universe.source_rows["stock-list"] == 4 and universe.source_rows["delisted-companies"] == 0


def test_statement_symbols_and_sample():
    stocks = [{"symbol": f"S{i:03d}", "exchange": "NYSE", "delistedDate": "2010-01-04"} for i in range(50)]
    universe = _build({"delisted-companies": stocks, "financial-statement-symbol-list": [{"symbol": "S001"}]})
    assert universe.statement_symbols == {"S001"}
    sample = universe.sample(["S007", "NOPE", "SPY"], 10)
    assert sample[:2] == ["S007", "SPY"] and len(sample) == 10 and len(set(sample)) == 10
    assert sample == universe.sample(["S007", "NOPE", "SPY"], 10)
    assert _build({"delisted-companies": stocks}).statement_symbols is None
    assert universe.tier_counts() == {"D": 50, "A": 2}
