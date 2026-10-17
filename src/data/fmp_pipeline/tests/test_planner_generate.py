"""Tests for task generation and the byte projection (spec §6.3, §6.5, plan WP-4). Synthetic universes only."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import date
from typing import Any

import pytest

from src.data.fmp_pipeline.planner.catalog import (
    BOOTSTRAP,
    CATALOG,
    DAILY_RAW,
    EndpointSpec,
    priority_of,
    selected,
)
from src.data.fmp_pipeline.planner.config import IntradayConfig, PlanConfig, UniverseConfig
from src.data.fmp_pipeline.planner.generate import PlannedTask, bootstrap_specs, generate, intraday_tier_priorities
from src.data.fmp_pipeline.planner.projection import GB, Bucket, Projection, disk_need
from src.data.fmp_pipeline.planner.universe import Universe, assemble_universe
from src.data.fmp_pipeline.planner.windowing import weekdays

AS_OF = date(2026, 10, 14)
INTRADAY = "historical-chart/1hour"


def _config(**changes: Any) -> PlanConfig:
    base = PlanConfig(dump_id="t", priorities=("P0",), daily_from=date(2016, 1, 1),
                      universe=UniverseConfig(tier_a=("SPY",), indices=()),
                      intraday=IntradayConfig(max_rows=1000))
    return replace(base, **changes)


def _universe(config: PlanConfig) -> Universe:
    sources = {
        "delisted-companies": [{"symbol": "DEAD", "exchange": "NYSE", "delistedDate": "2018-06-01"}],
        "stock-list": [{"symbol": "LIVE", "exchangeShortName": "NASDAQ", "type": "stock"},
                       {"symbol": "WAIT"}],
        "sp500-constituent": [{"symbol": "IDX", "exchange": "NYSE"}],
        "etf-list": [{"symbol": "ETFX", "exchange": "NYSE"}],
    }
    universe = assemble_universe(sources, {"IDX": {"exchange": "NYSE", "isActivelyTrading": True}},
                                 config.universe, AS_OF)
    return universe


def _tasks(config: PlanConfig, **kwargs: Any) -> list[PlannedTask]:
    return list(generate(_universe(config), config, **kwargs))


def test_catalog_entries_are_valid_and_unique():
    names = [e.endpoint for e in CATALOG]
    assert len(names) == len(set(names))
    expected = {"stock-list", "delisted-companies", "etf-list", "symbol-change", "historical-sp500-constituent",
                "sp500-constituent"}
    assert {e.endpoint for e in BOOTSTRAP} >= expected
    with pytest.raises(ValueError):
        EndpointSpec("x", 0, "P0", "nope")
    with pytest.raises(ValueError):
        EndpointSpec("x", 0, "P0", "symbol", "nobody")
    assert priority_of(3, 2) == 302


def test_selected_filters_priorities_but_keeps_bootstrap():
    p0 = {e.endpoint for e in selected(("P0",), "1hour")}
    assert "income-statement" not in p0 and "available-exchanges" in p0 and INTRADAY in p0
    assert "historical-chart/30min" in {e.endpoint for e in selected(("P0",), "30min")}
    assert "employee-count" in {e.endpoint for e in selected(("P0", "P1", "P2"), "1hour")}


def test_bootstrap_specs_page_from_zero_at_priority_zero():
    specs = {s.endpoint: s for s in bootstrap_specs(_config(page_size=250))}
    assert specs["delisted-companies"].params == {"page": 0, "limit": 250}
    assert specs["stock-list"].params == {}
    assert {s.priority for s in specs.values()} == {0}


def test_blocks_come_in_spec_order_and_delisted_first():
    tasks = _tasks(_config(), bounds={"SPY": (date(2016, 1, 4), AS_OF)}, daily_done={"SPY"})
    blocks = [t.block for t in tasks]
    assert blocks == sorted(blocks)
    daily = [t for t in tasks if t.spec.endpoint == DAILY_RAW]
    by_symbol = {t.spec.symbol: t.spec.priority for t in daily}
    assert by_symbol["DEAD"] == 301 and by_symbol["LIVE"] == 302
    assert all(t.spec.priority // 100 == t.block for t in tasks)


def test_unresolved_candidates_get_only_a_profile():
    tasks = _tasks(_config())
    wait = [t for t in tasks if t.spec.symbol == "WAIT"]
    profile = [t for t in wait if t.spec.endpoint == "profile"]
    assert profile and profile[0].deferred is None
    assert all(t.deferred == "exchange unknown until profile" for t in wait if t.spec.endpoint != "profile")


def test_daily_windows_skip_years_after_delisting():
    tasks = [t for t in _tasks(_config()) if t.spec.endpoint == DAILY_RAW]
    dead = sorted(t.spec.params["from"] for t in tasks if t.spec.symbol == "DEAD")
    live = sorted(t.spec.params["from"] for t in tasks if t.spec.symbol == "LIVE")
    assert dead == ["2016-01-01"] and live == ["2016-01-01", "2021-01-01", "2026-01-01"]
    dead_task = next(t for t in tasks if t.spec.symbol == "DEAD")
    assert dead_task.rows == pytest.approx(0.9655 * 627, rel=0.02)  # 2016-01-01 .. 2018-06-01 only
    assert dead_task.bytes_wire == pytest.approx(dead_task.rows * 25) and dead_task.measured


def test_intraday_waits_for_daily_and_row_cap():
    config = _config()
    tasks = [t for t in _tasks(config) if t.spec.endpoint == INTRADAY]
    assert tasks and {t.deferred for t in tasks} == {"awaiting daily raw", "exchange unknown until profile"}

    bounds = {"LIVE": (date(2024, 1, 2), AS_OF), "DEAD": (date(2016, 1, 4), date(2018, 5, 31))}
    done = {"LIVE", "DEAD", "IDX"}  # IDX finished with no bars: no intraday at all
    tasks = [t for t in _tasks(config, bounds=bounds, daily_done=done) if t.spec.endpoint == INTRADAY]
    ready = [t for t in tasks if t.deferred is None]
    assert {t.spec.symbol for t in ready} == {"LIVE", "DEAD"}
    assert not [t for t in tasks if t.spec.symbol == "IDX"]
    assert min(t.spec.params["from"] for t in ready if t.spec.symbol == "LIVE") <= "2024-01-02"
    assert max(t.spec.params["to"] for t in ready if t.spec.symbol == "DEAD") >= "2018-05-31"
    assert max(t.spec.params["from"] for t in ready if t.spec.symbol == "DEAD") <= "2018-07-01"  # + 30 day margin
    tiers = {t.spec.symbol: (t.spec.tier, t.spec.priority) for t in ready}
    assert tiers["DEAD"] == ("D", 600 + 1 + 3 * 2 + 0) and tiers["LIVE"] == ("C", 600 + 1 + 2 * 2 + 1)
    assert intraday_tier_priorities("D") == (607, 608) and intraday_tier_priorities("A") == (601, 602)
    for t in ready:
        lowest, highest = intraday_tier_priorities(t.spec.tier or "")
        assert lowest <= t.spec.priority <= highest  # `fmp run --tier` claims inside this range

    unknown = replace(config, intraday=IntradayConfig(max_rows=None))
    deferred = {t.deferred for t in _tasks(unknown, bounds=bounds, daily_done=done) if t.spec.endpoint == INTRADAY}
    assert deferred == {"row cap unknown (paid probe)"}


def test_intraday_respects_tiers_and_earliest():
    config = _config(intraday=IntradayConfig(max_rows=1000, tiers=("C",), earliest=date(2025, 1, 1)))
    bounds = {"LIVE": (date(2016, 1, 4), AS_OF), "DEAD": (date(2016, 1, 4), date(2018, 5, 31))}
    tasks = [t for t in _tasks(config, bounds=bounds, daily_done={"LIVE", "DEAD"}) if t.spec.endpoint == INTRADAY]
    assert {t.spec.symbol for t in tasks} == {"LIVE"}
    assert min(t.spec.params["to"] for t in tasks) >= "2025-01-01"


def test_selected_expands_intraday_per_interval():
    names = [e.endpoint for e in selected(("P0",), ("5min", "15min"))]
    assert names.count("historical-chart/5min") == 1 and names.count("historical-chart/15min") == 1


def test_tier_intervals_window_each_tier_at_its_own_interval():
    intraday = IntradayConfig(interval="5min", max_rows=510, bars_per_day=78, tier_intervals={"C": "15min"},
                              measured={"15min": {"max_rows": 500, "bars_per_day": 26}})
    config = _config(intraday=intraday)
    bounds = {"LIVE": (date(2024, 1, 2), AS_OF), "DEAD": (date(2016, 1, 4), date(2018, 5, 31))}
    tasks = [t for t in _tasks(config, bounds=bounds, daily_done={"LIVE", "DEAD"}) if t.block == 6]
    by_symbol = {t.spec.symbol: t.spec.endpoint for t in tasks}
    assert by_symbol["DEAD"] == "historical-chart/5min" and by_symbol["LIVE"] == "historical-chart/15min"
    live = sorted((t for t in tasks if t.spec.symbol == "LIVE"), key=lambda t: t.spec.params["from"])
    window = [date.fromisoformat(live[1].spec.params[k]) for k in ("from", "to")]
    assert weekdays(*window) == 15  # floor(500 / 26 / 1.25), against 5 weekdays at 5min
    assert all(t.deferred is None for t in tasks if t.spec.symbol in ("LIVE", "DEAD"))
    assert live[1].rows == pytest.approx(15 * 26, rel=0.15)

    unmeasured = _config(intraday=replace(intraday, measured={}))
    deferred = {t.deferred for t in _tasks(unmeasured, bounds=bounds, daily_done={"LIVE", "DEAD"})
                if t.spec.endpoint == "historical-chart/15min"}
    assert deferred == {"row cap unknown (paid probe)"}  # 5min's row cap is never borrowed


def test_intervals_and_for_interval():
    intraday = IntradayConfig(interval="5min", max_rows=510, tiers=("A", "B", "C", "D"),
                              tier_intervals={"C": "15min", "E": "30min"}, measured={"15min": {"max_rows": 500}})
    assert intraday.intervals == ("5min", "15min")  # E is not planned, so 30min is unused
    assert intraday.for_interval("5min").tiers == ("A", "B", "D")
    fifteen = intraday.for_interval("15min")
    assert (fifteen.tiers, fifteen.max_rows, fifteen.session_bars, fifteen.tier_intervals) == (("C",), 500, 26, {})
    assert IntradayConfig(tier_intervals={"A": "1hour"}).intervals == ("1hour",)


def test_p1_endpoints_calendars_statements_and_sample():
    config = _config(priorities=("P0", "P1"), reconciliation_fixed=("LIVE",), reconciliation_size=2,
                     calendar_window_days=365, daily_from=date(2024, 1, 1))
    tasks = _tasks(config)
    endpoints = Counter(t.spec.endpoint for t in tasks)
    assert endpoints["treasury-rates"] == 3  # 2024, 2025, 2026 (partial)
    assert endpoints["holidays-by-exchange"] == 2 and endpoints["economic-indicators"] == 10
    income = [t for t in tasks if t.spec.endpoint == "income-statement" and t.spec.symbol == "LIVE"]
    assert {t.spec.params["period"] for t in income} == {"annual", "quarter"}
    assert next(t for t in income if t.spec.params["period"] == "quarter").rows == 80
    sample = {t.spec.symbol for t in tasks if t.spec.endpoint == "historical-price-eod/full"}
    assert "LIVE" in sample and len(sample) == 2  # the fixed symbol plus one hash-ranked stock
    assert "ETFX" not in {t.spec.symbol for t in tasks if t.spec.endpoint == "earnings"}
    partial = [t for t in tasks if t.spec.endpoint == "treasury-rates"][-1]
    assert partial.spec.params["to"] == "2026-10-14" and partial.rows < 63
    assert not [t for t in tasks if t.spec.endpoint == "employee-count"]  # P2


def test_projection_counts_done_planned_and_deferred():
    config = _config()
    projection = Projection()
    for task in _tasks(config):
        projection.add_planned(task)
    projection.add_done(0, None, "ok", 3, 100, 700)
    projection.add_done(301, None, "ok", 2, GB, 7 * GB)
    projection.add_done(605, "C", "ok", 1, 10, 70)
    projection.add_done(302, None, "pending", 5, 0, 0)  # not done
    assert projection.bootstrap.done == 3 and projection.blocks[3].done == 2
    assert projection.blocks[3].done_decoded == 7 * GB and projection.tiers["C"].done == 1
    assert projection.blocks[3].planned > 0 and projection.blocks[6].deferred > 0
    assert projection.deferred_reasons["awaiting daily raw"] > 0
    wire, decoded = projection.remaining_bytes()
    assert 0 < wire < decoded


def test_schedule_and_disk_need():
    projection = Projection()
    projection.blocks[3] = Bucket(planned=1, wire=10 * GB, decoded=70 * GB)
    projection.tiers = {"A": Bucket(planned=1, wire=5 * GB, decoded=35 * GB),
                        "B": Bucket(planned=1, wire=20 * GB, decoded=140 * GB)}
    projection.blocks[6] = Bucket(planned=2, wire=25 * GB, decoded=175 * GB)
    steps = projection.schedule(45 * GB, used_wire=0, used_decoded=40 * GB)
    assert steps == [("daily raw", 1, 3), ("intraday tier A", 1, 4), ("intraday tier B", 1, 7)]  # 285 GB decoded
    assert disk_need(projection, 1.25, 12, 10) == int(35 * GB * 1.25 + 22 * GB)
    assert Projection().schedule(0, 0, 0) == []


def test_p2_etf_endpoints_cover_etfs_only():
    tasks = _tasks(_config(priorities=("P0", "P2")))
    assert {t.spec.symbol for t in tasks if t.spec.endpoint == "etf/holdings"} == {"ETFX"}
