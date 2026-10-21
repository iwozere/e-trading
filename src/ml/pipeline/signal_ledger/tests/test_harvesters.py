"""Tests for the result-folder harvesters (temp results tree)."""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Dict, List

from src.ml.pipeline.signal_ledger.harvesters import clean_ticker, harvest_all, harvest_p17, harvest_p19
from src.ml.pipeline.signal_ledger.schemas import SignalRecord


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tree(root: Path) -> None:
    _write(root / "p05_ai_selector/2026-10-20/top_picks.csv",
           "rank,ticker,confidence,bias,time_horizon\n1,eog,8,long,3-6 months\n2,SCHW,6,short,1 month\n3,,5,long,x\n")
    _write(root / "p05_ai_selector/2026-10-01/top_picks.csv", "rank,ticker,confidence,bias\n1,OLD,7,long\n")
    _write(root / "p06_emps2/2026-10-20/08_phase2_alerts.csv", "ticker,vol_zscore,alert_priority\nSIBN,4.1,H\n")
    _write(root / "p06_emps2/2026-10-20/09_final_universe.csv", "ticker,vol_zscore\nAMPH,2.0\nBRK.B,1.0\n")
    _write(root / "p10_emps3/2026-10-20/09_final_universe.csv",
           "ticker,prebreakout_score,status\nLOW,10,PASSED\nHIGH,30,PASSED\nFAIL,99,FAILED\n")
    _write(root / "p21_momentum/2026-10-20/targets.json",
           json.dumps({"as_of": "2026-10-20", "targets": [{"rank": 1, "ticker": "SNDK", "sector": "IT"}]}))
    _write(root / "p06_emps2/not-a-date/09_final_universe.csv", "ticker\nNOPE\n")


def test_clean_ticker() -> None:
    assert clean_ticker(" brk.b ") == "BRK.B"
    assert clean_ticker("123") is None
    assert clean_ticker(float("nan")) is None
    assert clean_ticker("TOO LONG NAME") is None


def test_harvest_all_reads_every_pipeline(tmp_path: Path) -> None:
    _tree(tmp_path)

    records = harvest_all(tmp_path, since=date(2026, 10, 10))
    by_source: Dict[str, List[SignalRecord]] = {}
    for r in records:
        by_source.setdefault(r.source, []).append(r)

    assert [(r.ticker, r.side, r.rank, r.score) for r in by_source["p05_top_picks"]] == [("EOG", "long", 1, 8.0), ("SCHW", "short", 2, 6.0)]
    assert [r.ticker for r in by_source["p06_final_universe"]] == ["AMPH", "BRK.B"]
    assert by_source["p06_phase2_alerts"][0].meta == {"alert_priority": "H"}
    assert [(r.ticker, r.rank) for r in by_source["p10_final_passed"]] == [("HIGH", 1), ("LOW", 2)]
    assert by_source["p21_targets"][0].horizon_days == 21
    assert all(r.signal_date == date(2026, 10, 20) for r in records)


def test_full_harvest_includes_old_folders(tmp_path: Path) -> None:
    _tree(tmp_path)
    assert "OLD" in {r.ticker for r in harvest_all(tmp_path, since=None)}


def test_p17_ranked_by_final_score_with_tier(tmp_path: Path) -> None:
    _write(tmp_path / "p17_penny_stocks/2026-10-20/2026-10-20_candidates.csv",
           "ticker,final_score,tier\nAAA,50,C\nBBB,80,A\n")
    records = harvest_p17(tmp_path, None)
    assert [(r.ticker, r.rank, r.meta) for r in records] == [("BBB", 1, {"tier": "A"}), ("AAA", 2, {"tier": "C"})]


def test_p19_first_alert_per_ticker_day_for_momentum_tiers(tmp_path: Path) -> None:
    db = tmp_path / "p19_penny_intraday/shadow.sqlite"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE shadow_log (date TEXT, ts TEXT, ticker TEXT, momentum_tier TEXT, momentum_score REAL, price REAL)")
        conn.executemany("INSERT INTO shadow_log VALUES (?,?,?,?,?,?)", [
            ("2026-10-20", "2026-10-20T14:00", "ZZZ", "T3", 70.0, 2.0),
            ("2026-10-20", "2026-10-20T15:00", "ZZZ", "T2", 80.0, 2.5),
            ("2026-10-20", "2026-10-20T14:00", "QQQQ", "T0", 10.0, 1.0),
        ])

    records = harvest_p19(tmp_path, None)

    assert len(records) == 1
    assert records[0].ticker == "ZZZ"
    assert records[0].horizon_days == 5
    assert records[0].meta["alert_ts"] == "2026-10-20T14:00"


def test_missing_pipelines_yield_nothing(tmp_path: Path) -> None:
    assert harvest_all(tmp_path) == []
