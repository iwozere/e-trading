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


def test_p17_setup_flag_kept_in_meta(tmp_path: Path) -> None:
    _write(tmp_path / "p17_penny_stocks/2026-10-20/2026-10-20_candidates.csv",
           "ticker,final_score,tier,setup_candidate\nAAA,50,C,True\nBBB,80,A,False\n")
    records = harvest_p17(tmp_path, None)
    assert {r.ticker: r.meta for r in records} == {"AAA": {"tier": "C", "setup": True}, "BBB": {"tier": "A"}}


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


def test_harvest_p22_emits_only_tier_entries(tmp_path: Path) -> None:
    from src.ml.pipeline.signal_ledger.harvesters import harvest_p22

    header = "rank_by_composite,ticker,composite,tier,tier_reason\n"
    base = tmp_path / "p22_biotech_ma"
    _write(base / "2026-11-03/watchlist.csv", header + "1,AAA,0.9,0,fundamentals_only\n2,BBB,0.8,0,fundamentals_only\n")
    _write(base / "2026-11-04/watchlist.csv", header + "1,AAA,3.5,3,disclosed_sale_process\n2,BBB,0.8,0,fundamentals_only\n")
    _write(base / "2026-11-05/pipeline.log", "no watchlist this day\n")
    _write(base / "2026-11-06/watchlist.csv", header + "1,AAA,3.5,3,disclosed_sale_process\n2,BBB,2.7,2,strategic_toehold\n")
    _write(base / "2026-11-07/watchlist.csv", header + "1,BBB,2.7,2,strategic_toehold\n2,AAA,0.9,0,fundamentals_only\n")
    _write(base / "2026-11-08/watchlist.csv", header + "1,AAA,3.4,3,disclosed_sale_process\n")

    records = harvest_p22(tmp_path, since=None)

    assert [(r.signal_date, r.ticker) for r in records] == [
        (date(2026, 11, 4), "AAA"), (date(2026, 11, 6), "BBB"), (date(2026, 11, 8), "AAA"),
    ]
    first = records[0]
    assert (first.source, first.rank, first.score, first.horizon_days) == ("p22_tier_entry", 1, 3.5, 250)
    # AAA's first spell ends on the 11-07 watchlist (dropped to tier 0): that is its native exit signal.
    assert first.meta == {"tier": 3, "tier_reason": "disclosed_sale_process", "exit_signal_date": "2026-11-07"}
    assert records[1].meta["exit_signal_date"] == "2026-11-08"  # BBB is missing from the 11-08 watchlist
    assert "exit_signal_date" not in records[2].meta  # AAA's second spell is still open
    # Exit dates arrive long after the entry, so the whole archive is always re-emitted.
    assert len(harvest_p22(tmp_path, since=date(2026, 11, 8))) == 3


def test_harvest_p24_attaches_the_paper_exit_signal(tmp_path: Path) -> None:
    from src.ml.pipeline.signal_ledger.harvesters import harvest_p24

    base = tmp_path / "p24_falling_knife"
    _write(base / "2026-11-02/signals_daily.csv",
           "timeframe,ticker,signal_date,close,drop_pct,score\ndaily,KNF,2026-10-30,5,0.6,0.9\ndaily,OPN,2026-10-30,5,0.6,0.5\n")
    _write(base / "_state/paper_positions.json", json.dumps({"positions": [
        {"timeframe": "daily", "ticker": "KNF", "signal_date": "2026-10-30", "exit_signal_date": "2026-11-20"},
        {"timeframe": "daily", "ticker": "OPN", "signal_date": "2026-10-30", "exit_signal_date": None},
        {"timeframe": "weekly", "ticker": "KNF", "signal_date": "2026-10-30", "exit_signal_date": "2027-01-01"},
    ]}))

    records = {r.ticker: r for r in harvest_p24(tmp_path, since=date(2026, 11, 5))}

    assert records["KNF"].meta["exit_signal_date"] == "2026-11-20"
    assert "exit_signal_date" not in records["OPN"].meta


def test_harvest_p24_reads_both_timeframes_and_skips_suspect(tmp_path: Path) -> None:
    from src.ml.pipeline.signal_ledger.harvesters import harvest_p24

    _write(tmp_path / "p24_falling_knife/2026-11-06/signals_daily.csv",
           "timeframe,ticker,close,drop_pct,score,suspect_gap\ndaily,low,5.0,0.45,0.2,False\n"
           "daily,HIGH,9.0,0.6,0.9,False\ndaily,BAD,3.0,0.5,0.99,True\n")
    _write(tmp_path / "p24_falling_knife/2026-11-06/signals_weekly.csv", "timeframe,ticker,close,drop_pct,score\nweekly,WK,20,0.5,0.4\n")
    _write(tmp_path / "p24_falling_knife/_state/signals_daily.csv", "ticker\nNOPE\n")

    records = harvest_p24(tmp_path, since=None)
    daily = [r for r in records if r.source == "p24_daily"]
    assert [(r.ticker, r.rank) for r in daily] == [("HIGH", 1), ("LOW", 2)]
    assert daily[0].horizon_days == 40 and daily[0].meta["drop_pct"] == 0.6
    weekly = [r for r in records if r.source == "p24_weekly"]
    assert [r.ticker for r in weekly] == ["WK"] and weekly[0].horizon_days == 130
    assert all(r.pipeline == "p24" and r.side == "long" and r.signal_date == date(2026, 11, 6) for r in records)


def test_p17_fresh_13g_flag_kept_in_meta(tmp_path: Path) -> None:
    _write(tmp_path / "p17_penny_stocks/2026-11-09/2026-11-09_candidates.csv",
           "ticker,final_score,tier,fresh_13g\nAAA,50,C,True\nBBB,80,A,False\n")
    assert {r.ticker: r.meta for r in harvest_p17(tmp_path, None)} == {"AAA": {"tier": "C", "fresh_13g": True}, "BBB": {"tier": "A"}}
