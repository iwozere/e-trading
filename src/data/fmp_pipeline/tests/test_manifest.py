"""Tests for the SQLite task manifest (spec §6.2)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.data.fmp_pipeline.client.fmp_client import FetchResult
from src.data.fmp_pipeline.manifest.manifest import Manifest, TaskSpec, canonical_params, task_id


def _result(classification: str, status: int = 200, error: str = "") -> FetchResult:
    return FetchResult("e", {}, status, classification, b"[]", 10, 20, 3, 1, "2026-10-14T00:00:00+00:00", error)


@pytest.fixture
def manifest(tmp_path: Path):
    with Manifest(tmp_path / "manifest.sqlite", "2026-10") as m:
        yield m


def test_task_id_is_deterministic_and_param_order_independent():
    a = task_id("2026-10", "splits", "AAPL", {"from": "2000-01-01", "to": "2026-10-14"})
    b = task_id("2026-10", "splits", "AAPL", {"to": "2026-10-14", "from": "2000-01-01"})
    assert a == b and len(a) == 40
    assert a != task_id("2026-11", "splits", "AAPL", {"from": "2000-01-01", "to": "2026-10-14"})
    assert task_id("d", "stock-list", None, {}) == task_id("d", "stock-list", None, {})
    assert canonical_params({"b": 1, "a": "x"}) == '{"a":"x","b":1}'


def test_add_tasks_is_idempotent(manifest: Manifest):
    specs = [TaskSpec("splits", "AAPL"), TaskSpec("splits", "MSFT"), TaskSpec("stock-list")]
    assert manifest.add_tasks(specs) == 3
    assert manifest.add_tasks(specs + [TaskSpec("splits", "NVDA")]) == 1
    assert manifest.counts() == {"pending": 4}


def test_claim_orders_by_priority_and_marks_running(manifest: Manifest):
    manifest.add_tasks([TaskSpec("b", "X", priority=20), TaskSpec("a", "Y", priority=10),
                        TaskSpec("c", "Z", {"from": "2025-01-01"}, priority=30, tier="A")])
    first = manifest.claim(2)
    assert [t.endpoint for t in first] == ["a", "b"] and all(t.attempts == 1 for t in first)
    third = manifest.claim(5)
    assert len(third) == 1 and third[0].tier == "A"
    assert third[0].request_params == {"symbol": "Z", "from": "2025-01-01"}
    assert manifest.claim(1) == []
    assert manifest.counts() == {"running": 3}


def test_request_params_without_symbol(manifest: Manifest):
    manifest.add_tasks([TaskSpec("delisted-companies", None, {"page": 0, "limit": 100})])
    assert manifest.claim(1)[0].request_params == {"page": 0, "limit": 100}


def test_reopen_reverts_running_to_pending(tmp_path: Path):
    path = tmp_path / "m.sqlite"
    with Manifest(path, "d") as m:
        m.add_tasks([TaskSpec("a", "X"), TaskSpec("b", "Y")])
        m.claim(2)
    with Manifest(path, "d") as m:
        assert m.recovered == 2 and m.counts() == {"pending": 2}
        assert m.claim(1)[0].attempts == 2  # the interrupted attempt still counts


@pytest.mark.parametrize("classification,status,expected", [
    ("ok", 200, "ok"), ("ok_empty", 200, "ok_empty"), ("plan_restricted", 402, "plan_restricted"),
    ("api_error", 400, "api_error"), ("not_found", 404, "api_error"), ("unauthorized", 401, "failed"),
    ("rate_limited", 429, "failed"), ("server_error", 503, "failed"), ("network_error", 0, "failed"),
    ("something_new", 200, "failed"),
])
def test_complete_maps_classification_to_status(manifest: Manifest, classification: str, status: int, expected: str):
    manifest.add_tasks([TaskSpec("e", "X")])
    task = manifest.claim(1)[0]
    assert manifest.complete(task, _result(classification, status), sha256="abc") == expected
    row = manifest.task(task.task_id)
    assert row is not None
    assert (row["status"], row["http_status"], row["rows"], row["bytes_wire"], row["bytes_decoded"]) == (
        expected, status, 3, 10, 20)
    assert row["sha256"] == "abc" and row["fetched_at"].startswith("2026-10-14")
    assert (row["error"] is None) == (expected in ("ok", "ok_empty"))


def test_fail_release_and_requeue(manifest: Manifest):
    manifest.add_tasks([TaskSpec("a", "X"), TaskSpec("b", "Y")])
    a, b = manifest.claim(2)
    manifest.fail(a, "sink error: disk full")
    manifest.release(b)
    assert manifest.counts() == {"failed": 1, "pending": 1}
    row_b = manifest.task(b.task_id)
    assert row_b is not None and row_b["attempts"] == 0
    assert manifest.requeue_failed() == 1
    assert manifest.counts() == {"pending": 2}


def test_bandwidth_window(manifest: Manifest):
    now = datetime.now(timezone.utc)
    manifest.record_bandwidth(100, 700, when=now - timedelta(days=31))
    manifest.record_bandwidth(10, 70, when=now - timedelta(days=29))
    manifest.record_bandwidth(1, 7)
    assert manifest.bandwidth_used(30) == (11, 77)
    assert manifest.bandwidth_used(60) == (111, 777)


def test_dumps_are_isolated(tmp_path: Path):
    path = tmp_path / "m.sqlite"
    with Manifest(path, "2026-10") as old:
        old.add_tasks([TaskSpec("a", "X")])
        old.record_bandwidth(5, 50)
    with Manifest(path, "2029-01") as new:
        assert new.counts() == {} and new.claim(1) == []
        assert new.bandwidth_used() == (5, 50)  # the cap is per account, not per dump


def test_events(manifest: Manifest):
    manifest.log_event("run_start", "workers=4")
    manifest.log_event("stop_quota", "quota at splits")
    assert [(e[1], e[2]) for e in manifest.events()] == [("run_start", "workers=4"), ("stop_quota", "quota at splits")]


def test_failed_transaction_rolls_back(manifest: Manifest):
    manifest.add_tasks([TaskSpec("a", "X")])
    with pytest.raises(RuntimeError):
        with manifest._tx() as db:  # noqa: SLF001 - verifying rollback behaviour
            db.execute("UPDATE tasks SET status='ok'")
            raise RuntimeError("crash mid-transaction")
    assert manifest.counts() == {"pending": 1}


def test_api_key_never_stored(manifest: Manifest, tmp_path: Path):
    manifest.add_tasks([TaskSpec("splits", "AAPL", {"limit": 5})])
    task = manifest.claim(1)[0]
    manifest.complete(task, _result("ok"))
    manifest.close()
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir())
    assert b"apikey" not in raw


def test_claim_respects_max_priority(manifest: Manifest):
    manifest.add_tasks([TaskSpec("a", None, {}, 0), TaskSpec("b", None, {}, 300)])
    assert [t.endpoint for t in manifest.claim(5, max_priority=0)] == ["a"]
    assert manifest.claim(5, max_priority=0) == []
    assert [t.endpoint for t in manifest.claim(5)] == ["b"]


def test_meta_is_set_once_per_dump(manifest: Manifest, tmp_path: Path):
    assert manifest.meta("as_of") is None
    assert manifest.meta_setdefault("as_of", "2026-10-14") == "2026-10-14"
    assert manifest.meta_setdefault("as_of", "2027-01-01") == "2026-10-14"
    with Manifest(tmp_path / "manifest.sqlite", "other") as other:
        assert other.meta("as_of") is None


def test_bounds_only_widen(manifest: Manifest):
    from datetime import date

    manifest.update_bounds("AAA", date(2020, 1, 2), date(2020, 12, 31))
    manifest.update_bounds("AAA", date(2021, 1, 4), date(2024, 6, 3))
    manifest.update_bounds("AAA", date(2010, 1, 4), date(2011, 1, 3))
    assert manifest.bounds() == {"AAA": (date(2010, 1, 4), date(2024, 6, 3))}


def test_planner_queries(manifest: Manifest):
    manifest.add_tasks([TaskSpec("p", "A", {"w": 1}, 301), TaskSpec("p", "A", {"w": 2}, 301),
                        TaskSpec("p", "B", {"w": 1}, 302, "C"), TaskSpec("q", None, {}, 0)])
    for task in manifest.claim(10):
        if (task.symbol, task.params) != ("A", {"w": 2}):
            manifest.complete(task, _result("ok"))
        else:
            manifest.release(task)
    assert manifest.symbols_finished("p") == {"B"}
    assert len(manifest.finished_ids()) == 3
    assert manifest.pending_symbols() == {"A"}
    assert sorted(manifest.totals()) == [(0, None, "ok", 1, 10, 20), (301, None, "ok", 1, 10, 20),
                                         (301, None, "pending", 1, 0, 0), (302, "C", "ok", 1, 10, 20)]
    assert {r["endpoint"] for r in manifest.task_rows(["q"])} == {"q"} and len(manifest.task_rows()) == 4
    assert manifest.skip_symbols(["A", "Z"], "dropped") == 1
    assert manifest.counts() == {"ok": 3, "skipped": 1}
    assert manifest.symbols_finished("p") == {"A", "B"}


def test_possibly_truncated_is_a_final_status(manifest: Manifest):
    manifest.add_tasks([TaskSpec("historical-chart/1hour", "A", {"from": "2026-01-05", "to": "2026-01-30"})])
    [task] = manifest.claim()
    assert manifest.complete(task, _result("possibly_truncated", error="possibly truncated")) == "possibly_truncated"
    row = manifest.task(task.task_id)
    assert row is not None and row["error"] == "possibly truncated"
    assert manifest.finished_ids() == {task.task_id}
