"""Tests for the same-CIK listing check ``V-SIBLING-BASIS`` (``checks_siblings``)."""

from __future__ import annotations

import csv
import math
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.fmp_pipeline.builders.validate import SECURITIES
from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore
from src.data.fmp_pipeline.datamanager_adapter.price_overrides import COLUMNS, OVERRIDES_FILE
from src.data.fmp_pipeline.planner.config import PlanConfig
from src.data.fmp_pipeline.tests.vintage_fixtures import SPLIT_DAY, make_vintage
from src.data.fmp_pipeline.validators.checks_siblings import (
    SIBLING_CANDIDATES,
    FactorRun,
    Listing,
    factor_runs,
    propose_rows,
    split_factor,
)
from src.data.fmp_pipeline.validators.framework import PASS, WARN, CheckResult, ValidationContext, run_checks

EX = pd.Timestamp(SPLIT_DAY)


@pytest.fixture(scope="module")
def vintage_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("vintage")
    make_vintage(root)
    return root


@pytest.fixture
def vintage_root(vintage_template: Path, tmp_path: Path) -> Path:
    shutil.copytree(vintage_template, tmp_path / "root")
    return tmp_path / "root"


def _closes(values: list[float], start: str = "2024-01-02") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def _make_siblings(root: Path) -> FmpStore:
    """AAA (raw, 4:1 split row) and NEWC share a CIK; NEWC's bars before the split are already adjusted."""
    store = FmpStore(root)
    path = store.dir / SECURITIES
    table = pq.read_table(path)
    ref = table.to_pandas()
    ref.loc[ref["symbol"].isin(["AAA", "NEWC"]), "cik"] = "0000000042"
    aaa = ref["symbol"] == "AAA"
    ref.loc[aaa, "last_price_date"] = ref.loc[aaa, "last_price_date"] - pd.Timedelta(days=1)  # NEWC continues
    pq.write_table(pa.Table.from_pandas(ref, schema=table.schema, preserve_index=False), path)
    newc = store.resolve("NEWC")
    assert newc is not None
    price_path = store.price_path(newc.security_id, "1d")
    assert price_path is not None
    prices = pq.read_table(price_path)
    df = prices.to_pandas()
    before = pd.to_datetime(df["date"]) < EX
    for column in ("open", "high", "low", "close"):
        df.loc[before, column] = df.loc[before, column] / 4.0
    pq.write_table(pa.Table.from_pandas(df, schema=prices.schema, preserve_index=False), price_path)
    return FmpStore(root)


def _run(root: Path) -> CheckResult:
    return run_checks(ValidationContext(root, PlanConfig(dump_id="v")), ["V-SIBLING-BASIS"]).results[0]


def _candidates(root: Path) -> list[dict[str, str]]:
    with (root / "reports" / "v" / SIBLING_CANDIDATES).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- pure functions -----------------------------------------------------------------------------------------------


def test_split_factor_accepts_whole_factors_only():
    assert split_factor(1.1) == 1.0 and split_factor(25.4) == 25.0 and split_factor(0.0401) == pytest.approx(1 / 25)
    assert math.isnan(split_factor(3.58)) and math.isnan(split_factor(1.5)) and math.isnan(split_factor(0.0))
    assert math.isnan(split_factor(float("nan")))


def test_factor_runs_finds_a_constant_factor_then_agreement():
    right = _closes([1.0 + 0.01 * i for i in range(40)])
    wrong = right.copy()
    wrong.iloc[:25] *= 25.0
    runs, shared = factor_runs(wrong, right)
    assert shared == 40 and [(r.factor, r.days) for r in runs] == [(25.0, 25), (1.0, 15)]
    assert runs[1].start == pd.Timestamp(right.index[25]).date()


def test_factor_runs_ignores_drifting_instruments():
    rng = np.random.default_rng(0)
    right = _closes(list(10 * np.exp(np.cumsum(rng.normal(0, 0.05, 60)))))
    wrong = _closes(list(250 * np.exp(np.cumsum(rng.normal(0, 0.05, 60)))))
    runs, shared = factor_runs(wrong, right)
    assert shared == 60 and sum(r.days for r in runs) < 0.8 * shared


def test_propose_rows_stacks_factor_changes_and_closes_at_overlap_end():
    """Factor 750 (1-for-30 then 1-for-25) then 25: rows x(25/750) at the change and x(1/25) after the overlap."""
    wrong_closes = _closes([1.0] * 30)
    runs = [FactorRun(date(2024, 1, 2), date(2024, 1, 15), 750.0, 10), FactorRun(date(2024, 1, 16), date(2024, 1, 31), 25.0, 12)]
    splits = pd.DataFrame({"security_id": [2], "date": [date(2024, 1, 16)], "ratio": [1 / 30]})
    rows = propose_rows("v", "1", Listing(1, "NEW", date(2026, 1, 1)), Listing(2, "OLD", date(2024, 1, 31)),
                        wrong_closes, runs, 30, "continues after the sibling", splits)
    assert [(r["ex_date"], r["ratio"], r["ex_date_basis"], r["confidence"]) for r in rows] == [
        ("2024-01-16", round(1 / 30, 6), "factor change", "high"),
        ("2024-02-01", 0.04, "overlap end", "medium"),
    ]
    assert rows[0]["confirmed_by"] == "sibling split row" and rows[0]["add_split"] == "true"
    assert all(r["evidence"] == "" for r in rows)  # unreviewed: the store rejects the row until evidence is added


# --- the check on a built vintage ---------------------------------------------------------------------------------


def test_sibling_check_proposes_the_override_then_passes_once_applied(vintage_root: Path):
    assert _run(vintage_root).status == PASS  # no shared CIK in the plain fixture
    _make_siblings(vintage_root)
    result = _run(vintage_root)
    assert result.status == WARN and result.violations == 1 and "NEWC" in result.examples[0]
    rows = _candidates(vintage_root)
    assert [(r["symbol"], r["ex_date"], float(r["ratio"]), r["sibling_symbol"], r["add_split"], r["confidence"])
            for r in rows] == [("NEWC", SPLIT_DAY, 4.0, "AAA", "true", "high")]
    reviewed = {**{c: rows[0][c] for c in COLUMNS}, "evidence": "fixture: AAA 4:1 split row"}
    (vintage_root / OVERRIDES_FILE).parent.mkdir(parents=True, exist_ok=True)
    with (vintage_root / OVERRIDES_FILE).open("w", encoding="utf-8", newline="") as handle:
        fieldnames: list[str] = list(COLUMNS)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(reviewed)
    assert len(FmpStore(vintage_root).overrides) == 1
    result = _run(vintage_root)
    assert result.status == PASS and _candidates(vintage_root) == []
