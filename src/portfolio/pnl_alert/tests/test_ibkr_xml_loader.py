"""
Unit tests for ibkr_xml_loader.
"""

import textwrap
from pathlib import Path

import pytest

from src.portfolio.pnl_alert.ibkr_xml_loader import load_ibkr_xml, resolve_xml_path
from src.portfolio.pnl_alert.position_aggregator import RawIbkrPosition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SINGLE_ACCOUNT_XML = textwrap.dedent("""\
    <FlexQueryResponse queryName="Open Positions" type="AF">
    <FlexStatements count="1">
    <FlexStatement accountId="U123" fromDate="2026-06-11" toDate="2026-06-11">
    <OpenPositions>
    <OpenPosition symbol="NVDA" position="20" markPrice="204.87"
        costBasisPrice="179.19" costBasisMoney="3583.80" />
    <OpenPosition symbol="ORCL" position="125" markPrice="184.10"
        costBasisPrice="196.25" costBasisMoney="24531.25" />
    <OpenPosition symbol="ZERO" position="0" markPrice="5.00"
        costBasisPrice="10.00" costBasisMoney="0" />
    </OpenPositions>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>
""")

_TWO_ACCOUNT_XML = textwrap.dedent("""\
    <FlexQueryResponse queryName="Open Positions" type="AF">
    <FlexStatements count="2">
    <FlexStatement accountId="U111">
    <OpenPositions>
    <OpenPosition symbol="VT" position="1216" markPrice="155.61"
        costBasisPrice="113.718751658" costBasisMoney="138282.002016" />
    </OpenPositions>
    </FlexStatement>
    <FlexStatement accountId="U222">
    <OpenPositions>
    <OpenPosition symbol="VT" position="74" markPrice="155.61"
        costBasisPrice="93.975441649" costBasisMoney="6954.182682" />
    <OpenPosition symbol="NVDA" position="10" markPrice="204.87"
        costBasisPrice="179.19" costBasisMoney="1791.90" />
    </OpenPositions>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>
""")


# ---------------------------------------------------------------------------
# resolve_xml_path
# ---------------------------------------------------------------------------

def test_resolve_exact_path(tmp_path: Path) -> None:
    xml_file = tmp_path / "positions.xml"
    xml_file.write_text("<root/>")
    resolved = resolve_xml_path(str(xml_file))
    assert resolved == xml_file


def test_resolve_exact_path_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_xml_path(str(tmp_path / "missing.xml"))


def test_resolve_glob_picks_latest(tmp_path: Path) -> None:
    (tmp_path / "Open_Positions-2026-06-09.xml").write_text("<root/>")
    (tmp_path / "Open_Positions-2026-06-10.xml").write_text("<root/>")
    (tmp_path / "Open_Positions-2026-06-11.xml").write_text("<root/>")
    pattern = str(tmp_path / "Open_Positions-*.xml")
    resolved = resolve_xml_path(pattern)
    assert resolved.name == "Open_Positions-2026-06-11.xml"


def test_resolve_glob_no_matches(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No IBKR XML files matched"):
        resolve_xml_path(str(tmp_path / "Open_Positions-*.xml"))


# ---------------------------------------------------------------------------
# load_ibkr_xml — single account
# ---------------------------------------------------------------------------

def test_single_account_loads_positions(tmp_path: Path) -> None:
    xml_file = tmp_path / "positions.xml"
    xml_file.write_text(_SINGLE_ACCOUNT_XML)

    positions = load_ibkr_xml(str(xml_file))

    symbols = {p.symbol for p in positions}
    assert symbols == {"NVDA", "ORCL"}  # ZERO position=0 filtered out


def test_single_account_avg_price(tmp_path: Path) -> None:
    xml_file = tmp_path / "positions.xml"
    xml_file.write_text(_SINGLE_ACCOUNT_XML)

    positions = {p.symbol: p for p in load_ibkr_xml(str(xml_file))}

    # costBasisMoney / position = 3583.80 / 20
    assert positions["NVDA"].avg_price == pytest.approx(179.19, rel=1e-4)
    assert positions["NVDA"].quantity == pytest.approx(20.0)
    assert positions["NVDA"].sec_type == "STK"


def test_zero_position_filtered(tmp_path: Path) -> None:
    xml_file = tmp_path / "positions.xml"
    xml_file.write_text(_SINGLE_ACCOUNT_XML)

    positions = load_ibkr_xml(str(xml_file))
    assert not any(p.symbol == "ZERO" for p in positions)


# ---------------------------------------------------------------------------
# load_ibkr_xml — multi-account position merging
# ---------------------------------------------------------------------------

def test_multi_account_quantities_summed(tmp_path: Path) -> None:
    xml_file = tmp_path / "positions.xml"
    xml_file.write_text(_TWO_ACCOUNT_XML)

    positions = {p.symbol: p for p in load_ibkr_xml(str(xml_file))}

    assert positions["VT"].quantity == pytest.approx(1216 + 74)


def test_multi_account_weighted_avg_price(tmp_path: Path) -> None:
    xml_file = tmp_path / "positions.xml"
    xml_file.write_text(_TWO_ACCOUNT_XML)

    positions = {p.symbol: p for p in load_ibkr_xml(str(xml_file))}

    # weighted avg = (138282.002016 + 6954.182682) / (1216 + 74)
    expected_avg = (138282.002016 + 6954.182682) / 1290
    assert positions["VT"].avg_price == pytest.approx(expected_avg, rel=1e-6)


def test_multi_account_single_symbol_unaffected(tmp_path: Path) -> None:
    xml_file = tmp_path / "positions.xml"
    xml_file.write_text(_TWO_ACCOUNT_XML)

    positions = {p.symbol: p for p in load_ibkr_xml(str(xml_file))}

    assert positions["NVDA"].quantity == pytest.approx(10.0)
    assert positions["NVDA"].avg_price == pytest.approx(179.19, rel=1e-4)


def test_returns_sorted_by_symbol(tmp_path: Path) -> None:
    xml_file = tmp_path / "positions.xml"
    xml_file.write_text(_TWO_ACCOUNT_XML)

    positions = load_ibkr_xml(str(xml_file))
    symbols = [p.symbol for p in positions]
    assert symbols == sorted(symbols)


# ---------------------------------------------------------------------------
# load_ibkr_xml — glob resolution
# ---------------------------------------------------------------------------

def test_load_via_glob(tmp_path: Path) -> None:
    old = tmp_path / "Open_Positions-2026-06-09.xml"
    new = tmp_path / "Open_Positions-2026-06-11.xml"
    old.write_text(_SINGLE_ACCOUNT_XML)
    new.write_text(_SINGLE_ACCOUNT_XML)

    pattern = str(tmp_path / "Open_Positions-*.xml")
    positions = load_ibkr_xml(pattern)
    assert len(positions) == 2  # NVDA + ORCL from the latest file


# ---------------------------------------------------------------------------
# load_ibkr_xml — error handling
# ---------------------------------------------------------------------------

def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ibkr_xml(str(tmp_path / "missing.xml"))


def test_malformed_xml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.xml"
    bad.write_text("this is not xml <<<")
    with pytest.raises(ValueError, match="Failed to parse"):
        load_ibkr_xml(str(bad))


def test_empty_positions_list(tmp_path: Path) -> None:
    empty_xml = textwrap.dedent("""\
        <FlexQueryResponse>
        <FlexStatements count="1">
        <FlexStatement accountId="U999">
        <OpenPositions />
        </FlexStatement>
        </FlexStatements>
        </FlexQueryResponse>
    """)
    xml_file = tmp_path / "empty.xml"
    xml_file.write_text(empty_xml)
    assert load_ibkr_xml(str(xml_file)) == []


def test_all_positions_return_rawibkrposition_type(tmp_path: Path) -> None:
    xml_file = tmp_path / "positions.xml"
    xml_file.write_text(_SINGLE_ACCOUNT_XML)

    for pos in load_ibkr_xml(str(xml_file)):
        assert isinstance(pos, RawIbkrPosition)
