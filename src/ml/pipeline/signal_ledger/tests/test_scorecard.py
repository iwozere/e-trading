"""Tests for the per-stream scorecard and the Newey-West t-statistic."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from src.ml.pipeline.signal_ledger.scorecard import MIN_COHORTS, build_scorecard, newey_west_t, verdict


def _row(source: str, day: int, ret: float, bench: float = 0.0, side: str = "long", meta=None) -> dict:
    row = {"source": source, "signal_date": date(2026, 1, 1) + timedelta(days=day), "side": side, "meta": meta}
    for h in (1, 5, 10, 20):
        row[f"ret_{h}d"], row[f"vt_{h}d"] = ret, bench
    return row


def test_newey_west_t_without_lags_is_plain_t() -> None:
    values = [0.01, 0.03, -0.01, 0.02, 0.04, 0.0]
    n = len(values)
    mean = sum(values) / n
    plain = mean / math.sqrt(sum((v - mean) ** 2 for v in values) / n / n)
    assert newey_west_t(values, 0) == pytest.approx(plain)


def test_newey_west_t_handles_degenerate_input() -> None:
    assert math.isnan(newey_west_t([0.1, 0.2], 1))
    assert math.isnan(newey_west_t([0.1, 0.1, 0.1], 1))


def test_verdicts() -> None:
    assert verdict(3.0, 0.02, MIN_COHORTS) == "edge"
    assert verdict(-3.0, -0.02, MIN_COHORTS) == "negative"
    assert verdict(1.0, 0.02, MIN_COHORTS) == "no_edge"
    assert verdict(5.0, 0.02, MIN_COHORTS - 1) == "insufficient"


def test_scorecard_applies_cost_side_and_cohorts() -> None:
    rows = [_row("p05_top_picks", d, 0.05 + 0.001 * (d % 3), bench=0.01) for d in range(30)]
    rows += [_row("p05_top_picks", d, -0.05, bench=0.01, side="short") for d in range(30)]

    card = build_scorecard(rows)

    assert len(card) == 1
    h20 = next(h for h in card[0]["horizons"] if h["horizon"] == 20)
    assert h20["signals"] == 60
    assert h20["cohorts"] == 30
    # long: +4.0..4.2% minus 0.1% cost; short of -5% vs +1% bench: +6% minus 0.1%
    assert h20["mean_excess"] == pytest.approx((0.041 + 0.059) / 2, abs=1e-3)
    assert h20["verdict"] == "edge"


def test_sub_stream_tier_ab_is_carved_out_of_p17() -> None:
    rows = [_row("p17_candidates", d, 0.0, meta={"tier": "A" if d % 2 else "C"}) for d in range(10)]
    streams = {s["stream"]: s["signals"] for s in build_scorecard(rows)}
    assert streams == {"p17_candidates": 10, "p17_tier_ab": 5}


def test_flagged_rows_are_excluded() -> None:
    rows = [_row("p19_momentum_alerts", d, 0.0) for d in range(5)] + [{**_row("p19_momentum_alerts", 9, 50.0), "data_flag": "price_jump"}]
    h20 = next(h for h in build_scorecard(rows)[0]["horizons"] if h["horizon"] == 20)
    assert h20["signals"] == 5


def test_unmatured_rows_are_ignored() -> None:
    rows = [{"source": "p10_final_passed", "signal_date": date(2026, 1, 1), "side": "long", "meta": None}]
    card = build_scorecard(rows)
    assert card[0]["horizons"] == []
