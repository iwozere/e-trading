"""Tests for run-time follow-ups: paging, daily bounds, intraday truncation (spec §5, §6.5, plan WP-4)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pytest

from src.data.fmp_pipeline.archive.archive import RawArchive
from src.data.fmp_pipeline.archive.reader import iter_records, latest_successful
from src.data.fmp_pipeline.client.fmp_client import FetchResult
from src.data.fmp_pipeline.manifest.manifest import Manifest, Task, TaskSpec
from src.data.fmp_pipeline.planner.config import IntradayConfig, PlanConfig
from src.data.fmp_pipeline.planner.follow_ups import FollowUps
from src.data.fmp_pipeline.runner import Downloader
from src.data.fmp_pipeline.tests.helpers import FakeSleep, make_client
from src.data.fmp_pipeline.tests.mock_server import MockConfig, MockFmpServer

INTRADAY = "historical-chart/1hour"
CONFIG = PlanConfig(dump_id="d", max_pages=5, intraday=IntradayConfig(max_rows=100))


def _task(endpoint: str, params: dict[str, Any], symbol: Optional[str] = None) -> Task:
    return Task("tid", "d", endpoint, symbol, params, 7, "C", 1)


def _result(body: Any, classification: str = "ok") -> FetchResult:
    raw = json.dumps(body).encode()
    rows = len(body) if isinstance(body, list) else 1
    return FetchResult("x", {}, 200, classification, raw, len(raw), len(raw), rows, 1, "2026-10-14T00:00:00+00:00")


@pytest.fixture
def manifest(tmp_path: Path):
    with Manifest(tmp_path / "m.sqlite", "d") as m:
        yield m


def test_non_empty_page_plans_the_next_one(manifest: Manifest):
    review = FollowUps(manifest, CONFIG)
    result, specs = review(_task("delisted-companies", {"page": 0, "limit": 100}), _result([{"symbol": "A"}]))
    assert result.classification == "ok"
    assert specs == [TaskSpec("delisted-companies", None, {"page": 1, "limit": 100}, 7, "C")]


def test_empty_or_failed_page_ends_paging(manifest: Manifest):
    review = FollowUps(manifest, CONFIG)
    assert review(_task("delisted-companies", {"page": 3, "limit": 100}), _result([], "ok_empty"))[1] == []
    assert review(_task("cik-list", {"page": 0, "limit": 100}), _result("x", "plan_restricted"))[1] == []


def test_repeated_page_and_max_pages_stop_paging(manifest: Manifest):
    review = FollowUps(manifest, CONFIG)
    body = _result([{"symbol": "SAME"}])
    assert review(_task("symbol-change", {"page": 0, "limit": 100}), body)[1]
    assert review(_task("symbol-change", {"page": 1, "limit": 100}), body)[1] == []  # endpoint ignores `page`
    assert review(_task("cik-list", {"page": 4, "limit": 100}), _result([{"cik": 1}]))[1] == []  # max_pages=5
    assert review(_task("cik-list", {"page": 3, "limit": 100}), _result([{"cik": 2}]))[1]


def test_daily_raw_records_bounds(manifest: Manifest):
    review = FollowUps(manifest, CONFIG)
    task = _task("historical-price-eod/non-split-adjusted", {"from": "2020-01-01", "to": "2024-12-31"}, "AAA")
    review(task, _result([{"date": "2024-12-31"}, {"date": "2020-01-02"}, {"date": "bad"}, {"x": 1}]))
    review(task, _result([{"date": "2025-03-03"}]))
    review(task, _result([], "ok_empty"))
    assert manifest.bounds() == {"AAA": (date(2020, 1, 2), date(2025, 3, 3))}


def test_intraday_window_at_the_cap_is_split(manifest: Manifest):
    review = FollowUps(manifest, CONFIG)
    task = _task(INTRADAY, {"from": "2026-01-05", "to": "2026-01-30"}, "AAA")
    result, specs = review(task, _result([{"date": "x"}] * 95))
    assert result.classification == "possibly_truncated" and "95 rows" in result.error
    assert [s.params for s in specs] == [{"from": "2026-01-05", "to": "2026-01-17"},
                                         {"from": "2026-01-18", "to": "2026-01-30"}]
    assert all(s.symbol == "AAA" and s.priority == 7 and s.tier == "C" for s in specs)
    assert review(task, _result([{"date": "x"}] * 94))[0].classification == "ok"


def test_one_day_window_at_the_cap_cannot_split(manifest: Manifest):
    result, specs = FollowUps(manifest, CONFIG)(_task(INTRADAY, {"from": "2026-01-05", "to": "2026-01-05"}, "AAA"),
                                                _result([{}] * 100))
    assert result.classification == "possibly_truncated" and "cannot split" in result.error and specs == []


def test_intraday_without_a_row_cap_is_not_checked(manifest: Manifest):
    review = FollowUps(manifest, PlanConfig(dump_id="d"))
    result, specs = review(_task(INTRADAY, {"from": "2026-01-05", "to": "2026-01-30"}, "AAA"), _result([{}] * 5000))
    assert result.classification == "ok" and specs == []


@pytest.mark.asyncio
async def test_truncated_windows_are_split_until_complete(tmp_path: Path):
    """A 7-bar/day mock with a 50-row cap: a 4-week window splits until every piece is below 95% of the cap."""
    with MockFmpServer(MockConfig(intraday_max_rows=50)) as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        manifest.add_tasks([TaskSpec(INTRADAY, "AAA", {"from": "2026-02-02", "to": "2026-02-27"}, 601, "C")])
        archive = RawArchive(tmp_path / "raw", "d")
        config = PlanConfig(dump_id="d", intraday=IntradayConfig(max_rows=50))
        async with make_client(fmp, FakeSleep()) as client:
            summary = await Downloader(manifest, client, sink=archive, workers=2,
                                       reviewer=FollowUps(manifest, config)).run()
        assert summary.stop_reason == "done"
        counts = manifest.counts()
        assert counts["possibly_truncated"] >= 1 and set(counts) == {"ok", "possibly_truncated"}
        records = list(iter_records(archive.path_for(INTRADAY, "AAA")))
        kept = latest_successful(records).values()
        assert {r["classification"] for r in records} == {"ok", "possibly_truncated"}
        days = {row["date"][:10] for r in kept for row in r["body"]}
        assert len(days) == 20  # every weekday of February 2026, nothing lost
        assert all(r["rows"] < 48 for r in kept)


@pytest.mark.asyncio
async def test_paging_runs_until_the_empty_page(tmp_path: Path):
    mock = MockConfig(paged_totals={"delisted-companies": 230}, max_page_size=100)
    with MockFmpServer(mock) as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        manifest.add_tasks([TaskSpec("delisted-companies", None, {"page": 0, "limit": 100}, 0)])
        async with make_client(fmp, FakeSleep()) as client:
            await Downloader(manifest, client, workers=1, reviewer=FollowUps(manifest, CONFIG)).run()
        rows = sorted((r["params_json"], r["status"], r["rows"]) for r in manifest.task_rows())
        assert [r[1:] for r in rows] == [("ok", 100), ("ok", 100), ("ok", 30), ("ok_empty", 0)]


def test_each_tier_interval_is_checked_against_its_own_row_cap(manifest: Manifest):
    intraday = IntradayConfig(interval="5min", max_rows=100, tier_intervals={"C": "15min"},
                              measured={"15min": {"max_rows": 400}})
    review = FollowUps(manifest, PlanConfig(dump_id="d", intraday=intraday))
    window = {"from": "2026-01-05", "to": "2026-01-30"}
    assert review(_task("historical-chart/5min", window, "AAA"), _result([{}] * 95))[0].classification == \
        "possibly_truncated"
    assert review(_task("historical-chart/15min", window, "AAA"), _result([{}] * 95))[0].classification == "ok"
    assert review(_task("historical-chart/15min", window, "AAA"), _result([{}] * 380))[0].classification == \
        "possibly_truncated"
