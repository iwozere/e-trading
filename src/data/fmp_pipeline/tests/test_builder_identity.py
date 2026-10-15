"""Tests for security identity (spec §7.3): stable ids, ticker-reuse segmentation, symbol history, resolution."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pytest

from src.data.fmp_pipeline.builders.identity import (
    IdentityBuilder,
    IdentityResult,
    IdRegistry,
    instrument_type,
    normalize_name,
    yf_symbol,
)
from src.data.fmp_pipeline.planner.config import UniverseConfig
from src.data.fmp_pipeline.planner.universe import Security, assemble_universe

AS_OF = date(2026, 10, 14)
CONFIG = UniverseConfig(tier_a=(), indices=())


def _profile(symbol: str, name: str, active: bool = True, ipo: str = "2010-01-04") -> dict[str, Any]:
    return {"symbol": symbol, "companyName": name, "exchange": "NASDAQ", "isEtf": False, "isFund": False,
            "isAdr": False, "isActivelyTrading": active, "ipoDate": ipo, "cik": "0000000001", "isin": "US0000000001"}


def _build(tmp_path: Path, sources: dict[str, Any], profiles: dict[str, Optional[dict[str, Any]]],
           segment: bool = False, registry: Optional[IdRegistry] = None) -> tuple[IdentityResult, IdRegistry]:
    universe = assemble_universe(sources, profiles, CONFIG, AS_OF)
    registry = registry or IdRegistry(tmp_path / "ids.json")
    return IdentityBuilder(registry, segment).build(universe, sources, profiles), registry


def test_registry_is_stable_and_never_reuses(tmp_path: Path):
    registry = IdRegistry(tmp_path / "ids.json")
    assert registry.allocate(["MSFT", "AAPL"]) == {"MSFT": 2, "AAPL": 1}  # sorted order
    registry.save()
    again = IdRegistry(tmp_path / "ids.json")
    assert again.allocate(["ZZZ", "AAPL"]) == {"ZZZ": 3, "AAPL": 1} and again.added == ["ZZZ"]
    again.save()
    third = IdRegistry(tmp_path / "ids.json")
    assert third.allocate(["MSFT"]) == {"MSFT": 2} and third.next_id == 4  # dropped keys keep their ids
    third.save()  # nothing added: no write
    assert json.loads((tmp_path / "ids.json").read_text(encoding="utf-8"))["next_id"] == 4


def test_registry_rejects_bad_files(tmp_path: Path):
    path = tmp_path / "ids.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a security id registry"):
        IdRegistry(path)
    path.write_text(json.dumps({"ids": {"A": 5}, "next_id": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="reuse"):
        IdRegistry(path)


def test_plain_securities_and_attributes(tmp_path: Path):
    sources = {"stock-list": [{"symbol": "AAA", "exchange": "NASDAQ"}, {"symbol": "BRK.B", "exchange": "NYSE"}],
               "actively-trading-list": [{"symbol": "AAA"}, {"symbol": "BRK.B"}]}
    result, _ = _build(tmp_path, sources, {"AAA": _profile("AAA", "Alpha Inc")})
    aaa, brk = result.securities
    assert (aaa.symbol, aaa.security_id, aaa.name) == ("AAA", 1, "Alpha Inc")
    assert (aaa.cik, aaa.ipo_date) == ("0000000001", date(2010, 1, 4))
    assert aaa.instrument_type == "stock" and not aaa.identity_uncertain and "stock-list" in aaa.source_flags
    assert brk.yf_symbol == "BRK-B" and brk.name is None
    assert result.resolve("AAA") == 1 and result.resolve("NOPE") is None and result.resolve(None) is None
    assert [(h.symbol, h.valid_from, h.reason) for h in result.history] == [("AAA", None, "current"),
                                                                            ("BRK.B", None, "current")]


REUSED = {
    "delisted-companies": [
        {"symbol": "XYZ", "companyName": "Old Widgets Corp", "exchange": "NYSE", "ipoDate": "1999-03-01",
         "delistedDate": "2008-06-30"},
        {"symbol": "XYZ", "companyName": "Future Delist Inc", "exchange": "NYSE", "ipoDate": "2009-01-01",
         "delistedDate": "2026-12-30"},  # scheduled, ignored (probe F13)
    ],
    "actively-trading-list": [{"symbol": "XYZ"}],
}


def test_ticker_reuse_is_uncertain_while_segmentation_is_off(tmp_path: Path):
    result, _ = _build(tmp_path, REUSED, {"XYZ": _profile("XYZ", "New Robotics Inc", ipo="2012-05-01")})
    [xyz] = result.securities
    assert xyz.identity_uncertain and "ticker reuse suspected" in (xyz.uncertain_reason or "")
    assert "segmentation off" in (xyz.uncertain_reason or "") and xyz.key == "XYZ"


def test_ticker_reuse_segments_when_enabled(tmp_path: Path):
    result, registry = _build(tmp_path, REUSED, {"XYZ": _profile("XYZ", "New Robotics Inc", ipo="2012-05-01")},
                              segment=True)
    old, new = result.securities
    assert (old.key, old.valid_from, old.valid_to, old.name) == ("XYZ|2008-06-30", None, date(2008, 6, 30),
                                                                 "Old Widgets Corp")
    assert (new.key, new.valid_from, new.valid_to) == ("XYZ|active", date(2008, 7, 1), None)
    assert old.cik is None and new.cik == "0000000001" and not old.identity_uncertain
    assert result.resolve("XYZ", date(2005, 1, 3)) == old.security_id
    assert result.resolve("XYZ", date(2020, 1, 3)) == new.security_id
    assert result.resolve("XYZ") is None  # ambiguous without a date
    assert {k for k in registry.ids} == {"XYZ|2008-06-30", "XYZ|active"}


def test_contradictory_reuse_is_not_segmented(tmp_path: Path):
    sources = {"delisted-companies": [
        {"symbol": "Q", "companyName": "Alpha", "ipoDate": "2000-01-01", "delistedDate": "2010-01-01"},
        {"symbol": "Q", "companyName": "Beta", "ipoDate": "2005-01-01", "delistedDate": "2015-01-01"},
    ]}
    result, _ = _build(tmp_path, sources, {"Q": _profile("Q", "Beta", active=False)}, segment=True)
    [q] = result.securities
    assert q.identity_uncertain and "overlap" in (q.uncertain_reason or "")
    sources["delisted-companies"][1]["ipoDate"] = ""
    result, _ = _build(tmp_path, sources, {"Q": _profile("Q", "Beta", active=False)}, segment=True)
    assert "no IPO date" in (result.securities[0].uncertain_reason or "")


def test_same_company_relisted_is_one_security(tmp_path: Path):
    sources = {"delisted-companies": [
        {"symbol": "R", "companyName": "Rho Holdings, Inc.", "ipoDate": "2000-01-01", "delistedDate": "2010-01-01"}],
        "actively-trading-list": [{"symbol": "R"}]}
    result, _ = _build(tmp_path, sources, {"R": _profile("R", "RHO HOLDINGS INC")}, segment=True)
    [r] = result.securities
    assert not r.identity_uncertain and r.valid_to is None


def test_symbol_history_follows_rename_chains(tmp_path: Path):
    sources = {
        "stock-list": [{"symbol": "CCC", "exchange": "NASDAQ"}, {"symbol": "AAA", "exchange": "NASDAQ"}],
        "symbol-change": [{"date": "2015-03-02", "oldSymbol": "AAA", "newSymbol": "BBB"},
                          {"date": "2020-06-01", "oldSymbol": "BBB", "newSymbol": "CCC"},
                          {"date": "bad", "oldSymbol": "X", "newSymbol": "Y"}],
    }
    result, _ = _build(tmp_path, sources, {"BBB": _profile("BBB", "Bee")})  # BBB resolved by its profile only
    ccc = result.by_symbol["CCC"][0]
    rows = [(h.symbol, h.valid_from, h.valid_to, h.reason) for h in result.history if h.security_id == ccc.security_id]
    assert rows == [("AAA", None, date(2015, 3, 1), "symbol-change"),
                    ("BBB", date(2015, 3, 2), date(2020, 5, 31), "symbol-change"),
                    ("CCC", date(2020, 6, 1), None, "current")]
    aaa = result.by_symbol["AAA"][0]  # AAA also listed on its own: possible duplicate history
    assert aaa.identity_uncertain and "renamed to CCC" in (aaa.uncertain_reason or "")
    bbb = result.by_symbol["BBB"][0]
    assert bbb.rename_only and result.resolve("BBB", date(2018, 1, 2)) == bbb.security_id  # own symbol wins
    result.drop({bbb.security_id})
    assert "BBB" not in result.by_symbol and all(h.security_id != bbb.security_id for h in result.history)
    assert result.resolve("BBB", date(2018, 1, 2)) == ccc.security_id  # now through the history
    assert result.resolve("BBB", date(2021, 1, 4)) is None and result.resolve("BBB") is None


def test_helpers():
    assert normalize_name("The Coca-Cola Company, Inc.") == "coca cola"
    assert yf_symbol("BRK.B") == "BRK-B" and yf_symbol("^GSPC") == "^GSPC" and yf_symbol("AAPL") == "AAPL"
    assert instrument_type(Security("^VIX"), None) == "index"
    assert instrument_type(Security("SPY", is_etf=True), None) == "etf"
    assert instrument_type(Security("F", is_fund=True), None) == "fund"
    assert instrument_type(Security("A"), "Acme Corp Warrants") == "warrant"
    assert instrument_type(Security("B"), "Blank Check Units") == "unit"
    assert instrument_type(Security("C"), "Cee Rights") == "right"
    assert instrument_type(Security("D"), "Dee 6% Preferred Series A") == "preferred"
    assert instrument_type(Security("E", is_adr=True), "Eee plc") == "adr"
