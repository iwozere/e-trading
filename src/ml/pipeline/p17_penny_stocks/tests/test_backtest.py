"""Tests for the P17 best-case backtest helpers."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p17_penny_stocks import backtest as bt


def _write_day(root: Path, day: str, rows: list) -> None:
    d = root / day
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(d / f"{day}_candidates.csv", index=False)


# ── collect_first_detections ────────────────────────────────────────────────


def test_keeps_earliest_detection(tmp_path):
    _write_day(
        tmp_path, "2026-06-01", [{"ticker": "AAA", "price": 2.0, "tier": "B", "company_name": "A", "final_score": 60}]
    )
    _write_day(
        tmp_path, "2026-06-02", [{"ticker": "AAA", "price": 9.0, "tier": "A", "company_name": "A", "final_score": 80}]
    )
    recs = bt.collect_first_detections(str(tmp_path))
    assert set(recs) == {"AAA"}
    assert recs["AAA"]["detection_date"] == "2026-06-01"
    assert recs["AAA"]["detection_price"] == 2.0
    assert recs["AAA"]["tier"] == "B"  # tier from first detection


def test_since_and_tier_filters_and_bad_price(tmp_path):
    _write_day(tmp_path, "2026-06-01", [{"ticker": "OLD", "price": 1.0, "tier": "B"}])
    _write_day(
        tmp_path,
        "2026-06-05",
        [
            {"ticker": "AAA", "price": 3.0, "tier": "B"},
            {"ticker": "BBB", "price": 4.0, "tier": "C"},
            {"ticker": "ZERO", "price": 0.0, "tier": "B"},  # skipped: bad price
        ],
    )
    recs = bt.collect_first_detections(str(tmp_path), since="2026-06-02", tiers=["B"])
    assert set(recs) == {"AAA"}  # OLD before since; BBB wrong tier; ZERO bad price


# ── backtest_ticker ─────────────────────────────────────────────────────────


class _FakeDownloader:
    def __init__(self, df):
        self._df = df

    def get_ohlcv(self, symbol, interval, start, end):
        return self._df


def test_backtest_ticker_math():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"]),
            "high": [2.5, 6.0, 4.0],
        }
    )
    rec = {
        "ticker": "AAA",
        "detection_date": "2026-06-01",
        "detection_price": 2.0,
        "tier": "B",
        "company_name": "A",
        "final_score": 60,
    }
    out = bt.backtest_ticker(
        _FakeDownloader(df), rec, invest=1000.0, end_date=pd.Timestamp("2026-06-10").to_pydatetime()
    )
    assert out["status"] == "ok"
    assert out["max_high"] == 6.0
    assert out["peak_date"] == "2026-06-02"
    assert out["shares"] == 500.0  # 1000 / 2.0
    assert out["peak_value"] == 3000.0  # 500 * 6.0
    assert out["profit"] == 2000.0
    assert out["return_pct"] == 200.0  # (6/2 - 1)*100


def test_backtest_ticker_no_data():
    rec = {
        "ticker": "AAA",
        "detection_date": "2026-06-01",
        "detection_price": 2.0,
        "tier": "B",
        "company_name": "A",
        "final_score": 60,
    }
    out = bt.backtest_ticker(_FakeDownloader(pd.DataFrame()), rec, 1000.0, pd.Timestamp("2026-06-10").to_pydatetime())
    assert out["status"] == "no_data" and out["profit"] is None


# ── summarize ───────────────────────────────────────────────────────────────


def test_summarize_overall_and_per_tier():
    results = [
        {"tier": "B", "status": "ok", "peak_value": 3000.0, "profit": 2000.0, "return_pct": 200.0},
        {"tier": "C", "status": "ok", "peak_value": 1500.0, "profit": 500.0, "return_pct": 50.0},
        {"tier": "B", "status": "no_data", "profit": None},  # excluded
    ]
    s = bt.summarize(results, invest=1000.0).set_index("group")
    assert s.loc["ALL", "tickers"] == 2
    assert s.loc["ALL", "profit"] == 2500.0
    assert s.loc["ALL", "invested"] == 2000.0
    assert s.loc["ALL", "roi_pct"] == 125.0
    assert s.loc["Tier B", "tickers"] == 1
    assert s.loc["Tier B", "win_rate_pct"] == 100.0
    assert "Tier W" not in s.index  # no W rows present
