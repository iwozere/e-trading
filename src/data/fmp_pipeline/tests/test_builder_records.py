"""Tests for generic record tables (spec §7.4): newest response wins, lossless ``_extra``, windows, determinism."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq
import pytest

from src.data.fmp_pipeline.archive.archive import RawArchive
from src.data.fmp_pipeline.builders.parquet_io import ParquetSettings, file_sha256
from src.data.fmp_pipeline.builders.records import (
    Field,
    RecordTableBuilder,
    TableSpec,
    _convert,
    build_tables,
    global_records,
    parse_date,
)
from src.data.fmp_pipeline.builders.tables import TABLES, covered_endpoints
from src.data.fmp_pipeline.tests.archive_fixtures import put

SETTINGS = ParquetSettings()
STATEMENT = TableSpec("fundamentals/income_statement", ("income-statement",), per_symbol=True,
                      fields=(Field("symbol", "symbol"), Field("date", "date", "date"), Field("period", "period"),
                              Field("filingDate", "filing_date", "date")),
                      dedup=("symbol", "date", "period"), numeric=True)


def _resolver(symbol: Optional[str], when: Optional[date]) -> Optional[int]:
    return {"AAA": 1, "BBB": 2}.get(symbol or "")


@pytest.fixture
def archive(tmp_path: Path) -> RawArchive:
    return RawArchive(tmp_path / "raw", "d")


def _read(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def test_statement_table_typed_numeric_and_extra(archive: RawArchive, tmp_path: Path):
    rows = [{"symbol": "AAA", "date": "2025-12-31", "period": "FY", "filingDate": "2026-02-01", "revenue": 100,
             "eps": 1.5, "note": "text", "link": None},
            {"symbol": "AAA", "date": "2024-12-31", "period": "FY", "filingDate": "", "revenue": "n/a", "eps": 1.2}]
    put(archive, "income-statement", "AAA", rows, {"period": "annual", "limit": 5})
    put(archive, "income-statement", "ZZZ", [{"symbol": "ZZZ", "date": "2025-12-31", "period": "FY", "revenue": 7}],
        {"period": "annual", "limit": 5})
    builder = RecordTableBuilder(STATEMENT, archive.dump_dir, _resolver, SETTINGS)
    stats = builder.build(tmp_path / "out")
    table = pq.read_table(tmp_path / "out" / STATEMENT.path)
    assert table.schema.names == ["security_id", "symbol", "date", "period", "filing_date", "eps", "_extra"]
    out = table.to_pylist()
    assert [(r["symbol"], r["date"], r["security_id"]) for r in out] == [
        ("AAA", date(2024, 12, 31), 1), ("AAA", date(2025, 12, 31), 1), ("ZZZ", date(2025, 12, 31), None)]
    assert out[0]["filing_date"] is None and out[0]["eps"] == 1.2
    assert json.loads(out[0]["_extra"]) == {"revenue": "n/a"}  # a non-number kept losslessly, not as NaN
    assert json.loads(out[1]["_extra"]) == {"link": None, "note": "text", "revenue": 100}  # revenue not always numeric
    assert stats.rows == 3 and stats.responses == 2 and stats.unresolved == 1 and len(stats.input_digest) == 64


def test_newest_response_and_newest_row_win(archive: RawArchive, tmp_path: Path):
    spec = TableSpec("events/earnings", ("earnings",), per_symbol=True, dedup=("symbol", "date"), numeric=True)
    put(archive, "earnings", "AAA", [{"symbol": "AAA", "date": "2025-01-30", "epsActual": 1.0}],
        fetched_at="2026-10-14T10:00:00+00:00")
    put(archive, "earnings", "AAA", [{"symbol": "AAA", "date": "2025-01-30", "epsActual": 2.0}],
        fetched_at="2026-10-15T10:00:00+00:00")  # same request key, re-fetched later
    put(archive, "earnings", "AAA", [{"symbol": "AAA", "date": "2025-01-30", "epsActual": 3.0}],
        fetched_at="2026-10-16T10:00:00+00:00", classification="server_error", status=503)
    put(archive, "earnings", "AAA", [{"symbol": "AAA", "date": "2025-01-30", "epsActual": 4.0},
                                     {"symbol": "AAA", "date": "2025-04-30", "epsActual": 5.0}],
        {"limit": 10}, fetched_at="2026-10-15T11:00:00+00:00")  # other request, newer: its row wins the date
    builder = RecordTableBuilder(spec, archive.dump_dir, _resolver, SETTINGS)
    builder.build(tmp_path / "out")
    out = _read(tmp_path / "out" / spec.path)
    assert [(r["date"], r["epsActual"]) for r in out] == [(date(2025, 1, 30), 4.0), (date(2025, 4, 30), 5.0)]
    assert builder.stats.duplicates == 1 and builder.stats.responses == 2


def test_rows_outside_their_window_are_dropped(archive: RawArchive, tmp_path: Path):
    spec = TableSpec("fundamentals/market_cap", ("historical-market-capitalization",), per_symbol=True,
                     dedup=("symbol", "date"), numeric=True)
    put(archive, spec.endpoints[0], "AAA", [{"symbol": "AAA", "date": "2021-06-01", "marketCap": 1},
                                            {"symbol": "AAA", "date": "2019-06-03", "marketCap": 2}],
        {"from": "2020-01-01", "to": "2024-12-31"})
    builder = RecordTableBuilder(spec, archive.dump_dir, _resolver, SETTINGS)
    builder.build(tmp_path / "out")
    assert [r["date"] for r in _read(tmp_path / "out" / spec.path)] == [date(2021, 6, 1)]
    assert builder.stats.out_of_window == 1


def test_global_tables_labels_params_and_repeated_pages(archive: RawArchive, tmp_path: Path):
    page = [{"symbol": "AAA", "companyName": "A", "exchange": "NYSE", "ipoDate": "2000-01-01",
             "delistedDate": "2020-01-02"}]
    put(archive, "delisted-companies", None, page, {"page": 0, "limit": 100})
    put(archive, "delisted-companies", None, page, {"page": 1, "limit": 100})  # endpoint ignored `page`
    put(archive, "delisted-companies", None, [], {"page": 2, "limit": 100})
    assert len(global_records(archive.path_for("delisted-companies", None))) == 2
    put(archive, "economic-indicators", None, [{"date": "2025-01-01", "value": 3.1}], {"name": "CPI"})
    put(archive, "historical-sp500-constituent", None, [{"date": "2020-01-02", "symbol": "AAA", "reason": "x"}])
    put(archive, "historical-nasdaq-constituent", None, [{"date": "2021-01-04", "symbol": "BBB"}])
    put(archive, "splits-calendar", None, [{"symbol": "AAA", "date": "2025-02-03", "numerator": 2}],
        {"from": "2025-01-01", "to": "2025-03-31"})
    results = build_tables(TABLES, archive.dump_dir, tmp_path / "out", _resolver, SETTINGS, 2, 500)
    assert set(results) == {"reference/delisted.parquet", "macro/economic_indicators.parquet",
                            "index_membership/index_changes.parquet", "macro/calendars.parquet"}
    delisted = _read(tmp_path / "out" / "reference/delisted.parquet")
    assert [(r["symbol"], r["delisted_date"]) for r in delisted] == [("AAA", date(2020, 1, 2))]
    [cpi] = _read(tmp_path / "out" / "macro/economic_indicators.parquet")
    assert (cpi["name"], cpi["value"]) == ("CPI", 3.1)
    changes = _read(tmp_path / "out" / "index_membership/index_changes.parquet")
    assert [(r["index"], r["symbol"], r["security_id"]) for r in changes] == [
        ("sp500", "AAA", 1), ("nasdaq", "BBB", 2)]
    [cal] = _read(tmp_path / "out" / "macro/calendars.parquet")
    assert cal["calendar"] == "splits" and json.loads(cal["_extra"]) == {"numerator": 2}


def test_output_is_byte_identical_on_rebuild(archive: RawArchive, tmp_path: Path):
    rows = [{"symbol": "AAA", "date": f"2025-0{m}-28", "period": "Q", "revenue": m * 1.5} for m in range(1, 10)]
    put(archive, "income-statement", "AAA", rows, {"period": "quarter", "limit": 40})
    for out in ("a", "b"):
        RecordTableBuilder(STATEMENT, archive.dump_dir, _resolver, SETTINGS, batch_rows=4).build(tmp_path / out)
    assert file_sha256(tmp_path / "a" / STATEMENT.path) == file_sha256(tmp_path / "b" / STATEMENT.path)


def test_convert_and_parse_date():
    assert parse_date("2025-01-02 10:00:00") == date(2025, 1, 2)
    assert parse_date("") is None and parse_date("2025-13-01") is None and parse_date(5) is None
    assert _convert(None, "float") == (None, True) and _convert("", "date") == (None, True)
    assert _convert(5, "str") == ("5", True) and _convert("x", "str") == ("x", True)
    assert _convert(True, "bool") == (True, True) and _convert("yes", "bool") == (None, False)
    assert _convert(3.0, "int") == (3, True) and _convert(3.5, "int") == (None, False)
    assert _convert("4", "int") == (4, True) and _convert("4.5", "int") == (None, False)
    assert _convert("4.5", "float") == (4.5, True) and _convert("abc", "float") == (None, False)
    assert _convert(True, "float") == (None, False) and _convert([1], "float") == (None, False)


def test_every_catalogue_endpoint_has_a_table():
    from src.data.fmp_pipeline.planner.catalog import CATALOG

    covered = covered_endpoints()
    missing = [e.endpoint for e in CATALOG if "{interval}" not in e.endpoint and e.endpoint not in covered]
    assert missing == []
