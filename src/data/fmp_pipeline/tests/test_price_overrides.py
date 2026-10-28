"""Tests for the reviewed price-basis overrides (``price_overrides``) and their detection in ``V-SPLIT-CONSISTENCY``."""

from __future__ import annotations

import csv
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.fmp_pipeline.adjust.adjust import adjust
from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore
from src.data.fmp_pipeline.datamanager_adapter.price_overrides import (
    COLUMNS,
    OVERRIDES_FILE,
    PriceBasisOverride,
    add_override_splits,
    apply_price_overrides,
    load_overrides,
)
from src.data.fmp_pipeline.planner.config import PlanConfig
from src.data.fmp_pipeline.tests.vintage_fixtures import SPLIT_DAY, make_vintage
from src.data.fmp_pipeline.validators.checks_events import SPLIT_CANDIDATES
from src.data.fmp_pipeline.validators.framework import INFO, PASS, WARN, CheckResult, ValidationContext, run_checks

EX = date.fromisoformat(SPLIT_DAY)


@pytest.fixture(scope="module")
def vintage_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("vintage")
    make_vintage(root)
    return root


@pytest.fixture
def vintage_root(vintage_template: Path, tmp_path: Path) -> Path:
    shutil.copytree(vintage_template, tmp_path / "root")
    return tmp_path / "root"


def _write_overrides(root: Path, rows: list[dict[str, object]]) -> None:
    path = root / OVERRIDES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames: list[str] = list(COLUMNS)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{"from_date": "", "add_split": "false", "evidence": "10-Q share count", **r} for r in rows])


def _row(store: FmpStore, name: str, **extra: object) -> dict[str, object]:
    info = store.resolve(name)
    assert info is not None
    return {"vintage": store.vintage, "security_id": info.security_id, "symbol": info.symbol, "ex_date": SPLIT_DAY,
            "ratio": 4.0, **extra}


def _flatten_split(store: FmpStore, symbol: str) -> None:
    """Make the bars before the split already split-adjusted: the vintage-2026-10 error."""
    info = store.resolve(symbol)
    assert info is not None
    path = store.price_path(info.security_id, "1d")
    assert path is not None
    table = pq.read_table(path)
    df = table.to_pandas()
    before = pd.to_datetime(df["date"]) < pd.Timestamp(EX)
    for column in ("open", "high", "low", "close"):
        df.loc[before, column] = df.loc[before, column] / 4.0
    pq.write_table(pa.Table.from_pandas(df, schema=table.schema, preserve_index=False), path)


def _run(root: Path, *ids: str) -> dict[str, CheckResult]:
    return {r.id: r for r in run_checks(ValidationContext(root, PlanConfig(dump_id="v")), list(ids)).results}


def _override(**extra: object) -> PriceBasisOverride:
    values: dict[str, object] = {"security_id": 1, "symbol": "X", "ex_date": date(2025, 1, 3), "ratio": 0.04,
                                 "from_date": None, "add_split": False, "evidence": "e", **extra}
    return PriceBasisOverride(**values)  # type: ignore[arg-type]


# --- pure functions -----------------------------------------------------------------------------------------------


def test_apply_price_overrides_rebases_only_the_stretch():
    index = pd.DatetimeIndex(pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]), name="timestamp")
    frame = pd.DataFrame({"open": 25.0, "high": 25.0, "low": 25.0, "close": [25.0, 25.0, 20.0],
                          "volume": pd.Series([40, 80, 3], index=index, dtype="int64"), "flags": 0}, index=index)
    out = apply_price_overrides(frame, [_override(from_date=date(2025, 1, 2))])
    assert out["close"].tolist() == [25.0, 1.0, 20.0] and out["open"].tolist() == [25.0, 1.0, 25.0]
    assert out["volume"].tolist() == [40, 2000, 3] and out["volume"].dtype == "int64"
    assert frame["close"].iloc[1] == 25.0  # the input is not modified
    assert apply_price_overrides(frame, []) is frame


def test_add_override_splits_keeps_an_existing_vintage_row():
    splits = pd.DataFrame({"date": [date(2020, 1, 2)], "numerator": [2.0], "denominator": [1.0]})
    out = add_override_splits(splits, [_override(add_split=True), _override(add_split=False, ex_date=date(2025, 6, 2))])
    assert out["date"].tolist() == [date(2020, 1, 2), date(2025, 1, 3)] and out["numerator"].tolist() == [2.0, 0.04]
    assert add_override_splits(out, [_override(add_split=True)]) is out  # same date already there


def test_load_overrides_skips_bad_rows_and_other_vintages(tmp_path: Path):
    good = {"vintage": "v", "security_id": 7, "symbol": "X", "ex_date": "2025-01-03", "ratio": 0.04}
    _write_overrides(tmp_path, [
        good,
        {**good, "vintage": "old"},
        {**good, "symbol": "Y"},
        {**good, "ratio": 1.0},
        {**good, "evidence": " "},
        {**good, "from_date": "2025-02-01"},
        {**good, "add_split": "maybe"},
        {**good, "ex_date": "not a date"},
    ])
    found = load_overrides(tmp_path / OVERRIDES_FILE, "v", {7: "X"})
    assert len(found) == 1 and found.for_security(7)[0].ratio == 0.04 and found.other_vintages == 1
    assert len(found.problems) == 6 and "line 4" in found.problems[0] and "'Y'" in found.problems[0]
    assert found.digest and load_overrides(tmp_path / "missing.csv", "v", {}).digest == ""


# --- FmpStore -----------------------------------------------------------------------------------------------------


def test_store_applies_overrides_to_every_daily_read(vintage_root: Path):
    plain = FmpStore(vintage_root)
    sid = plain.resolve("BRK.B").security_id  # type: ignore[union-attr]
    raw = plain.prices(sid, "1d")
    assert raw is not None and sid not in plain.events_many([sid])
    _write_overrides(vintage_root, [_row(plain, "BRK.B", ratio=0.25, add_split="true")])
    store = FmpStore(vintage_root)
    fixed = store.prices(sid, "1d")
    assert fixed is not None
    before = fixed.index < pd.Timestamp(EX)
    assert (fixed.loc[before, "close"] == raw.loc[before, "close"] * 0.25).all()
    assert (fixed.loc[~before, "close"] == raw.loc[~before, "close"]).all()
    bulk = store.bulk_daily([sid], date(2025, 1, 1), date(2025, 12, 31))[sid]
    assert bulk["close"].tolist() == fixed["close"].tolist()
    splits, _ = store.events(sid)
    assert splits["date"].tolist() == [EX] and splits["numerator"].tolist() == [0.25]
    assert store.events_hash(splits, splits) != plain.events_hash(splits, splits)  # adjusted-cache key changes


def test_override_restores_a_flat_split_so_adjust_is_right(vintage_root: Path):
    store = FmpStore(vintage_root)
    sid = store.resolve("AAA").security_id  # type: ignore[union-attr]
    expected = adjust(store.prices(sid, "1d"), *store.events(sid), "split")["close"]  # type: ignore[arg-type]
    _flatten_split(store, "AAA")
    broken = FmpStore(vintage_root)
    double = adjust(broken.prices(sid, "1d"), *broken.events(sid), "split")["close"]  # type: ignore[arg-type]
    assert not double.equals(expected)  # adjusted twice before the split
    _write_overrides(vintage_root, [_row(broken, "AAA")])
    fixed = FmpStore(vintage_root)
    pd.testing.assert_series_equal(adjust(fixed.prices(sid, "1d"), *fixed.events(sid), "split")["close"],  # type: ignore[arg-type]
                                   expected)


# --- detection and the override check ------------------------------------------------------------------------------


def test_flat_split_is_reported_then_passes_once_overridden(vintage_root: Path):
    store = FmpStore(vintage_root)
    _flatten_split(store, "AAA")
    results = _run(vintage_root, "V-SPLIT-CONSISTENCY", "V-OVERRIDES")
    assert results["V-SPLIT-CONSISTENCY"].status == WARN and results["V-SPLIT-CONSISTENCY"].details["flat"] == 1
    assert results["V-OVERRIDES"].status == INFO
    rows = list(csv.DictReader((vintage_root / "reports" / "v" / SPLIT_CANDIDATES).open(encoding="utf-8")))
    assert [(r["symbol"], r["date"], r["verdict"]) for r in rows] == [("AAA", SPLIT_DAY, "flat")]
    _write_overrides(vintage_root, [_row(store, "AAA")])
    results = _run(vintage_root, "V-SPLIT-CONSISTENCY", "V-OVERRIDES")
    assert results["V-SPLIT-CONSISTENCY"].status == PASS and results["V-SPLIT-CONSISTENCY"].details["flat"] == 0
    assert results["V-OVERRIDES"].status == PASS and "AAA" in results["V-OVERRIDES"].details["applied"][0]
    assert len((vintage_root / "reports" / "v" / SPLIT_CANDIDATES).read_text(encoding="utf-8").splitlines()) == 1


def test_override_check_warns_on_bad_and_stale_rows(vintage_root: Path):
    store = FmpStore(vintage_root)
    _write_overrides(vintage_root, [_row(store, "AAA", symbol="ZZZ"), _row(store, "AAA", vintage="2025-01")])
    result = _run(vintage_root, "V-OVERRIDES")["V-OVERRIDES"]
    assert result.status == WARN and result.violations == 2 and result.examined == 2
    assert "'ZZZ'" in result.examples[0] and "another vintage" in result.examples[1]
