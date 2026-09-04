"""
Tests for `src.data.pipeline.dependency_status`.

`require_dependencies_or_defer`'s decision logic is tested here by
monkeypatching `check_dependencies` — it needs no DB. `check_dependency`
itself (the actual `job_schedules`/`job_schedule_runs` query) is exercised
manually against a live-connected DB rather than in this suite, matching how
the rest of this repo's DB-backed tests already require a reachable DB (see
`p22_biotech_ma/tests/db/test_repo_p22_bitemporal.py`, which errors without
one in this environment) — no in-memory DB fixture exists yet to assert
against here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.data.pipeline import dependency_status as ds
from src.data.pipeline.dependency_status import DependencyStatus, require_dependencies_or_defer


def _status(name: str, succeeded: bool, *, ran_today: bool = True, registered: bool = True) -> DependencyStatus:
    now = datetime.now(timezone.utc)
    return DependencyStatus(
        name=name, registered=registered, ran_today=ran_today, succeeded=succeeded,
        status="completed" if succeeded else "failed",
        started_at=now if ran_today else None, finished_at=now if succeeded else None,
    )


def test_plugin_with_no_depends_on_is_always_ready():
    # A real registry entry with an empty depends_on (most plugins).
    ready, statuses = require_dependencies_or_defer("P22 SEC Universe Ingest")
    assert ready is True
    assert statuses == []


def test_unregistered_plugin_name_is_ready_with_warning():
    ready, statuses = require_dependencies_or_defer("this-plugin-name-does-not-exist")
    assert ready is True
    assert statuses == []


def test_all_dependencies_succeeded_is_ready(monkeypatch):
    monkeypatch.setattr(ds, "get_by_name", lambda name: _FakeSpec(["dep-a", "dep-b"]))
    monkeypatch.setattr(ds, "check_dependencies", lambda names, as_of=None: [_status(n, True) for n in names])

    ready, statuses = require_dependencies_or_defer("consumer")
    assert ready is True
    assert len(statuses) == 2
    assert all(s.succeeded for s in statuses)


def test_one_dependency_not_run_defers(monkeypatch):
    monkeypatch.setattr(ds, "get_by_name", lambda name: _FakeSpec(["dep-a", "dep-b"]))
    monkeypatch.setattr(
        ds, "check_dependencies",
        lambda names, as_of=None: [_status("dep-a", True), _status("dep-b", False, ran_today=False)],
    )

    ready, statuses = require_dependencies_or_defer("consumer")
    assert ready is False
    assert len(statuses) == 2


def test_one_dependency_failed_defers(monkeypatch):
    monkeypatch.setattr(ds, "get_by_name", lambda name: _FakeSpec(["dep-a"]))
    monkeypatch.setattr(ds, "check_dependencies", lambda names, as_of=None: [_status("dep-a", False)])

    ready, statuses = require_dependencies_or_defer("consumer")
    assert ready is False
    assert statuses[0].succeeded is False


def test_unregistered_dependency_counts_as_not_ready(monkeypatch):
    monkeypatch.setattr(ds, "get_by_name", lambda name: _FakeSpec(["typo'd-dep-name"]))
    monkeypatch.setattr(
        ds, "check_dependencies",
        lambda names, as_of=None: [_status("typo'd-dep-name", False, ran_today=False, registered=False)],
    )

    ready, statuses = require_dependencies_or_defer("consumer")
    assert ready is False
    assert statuses[0].registered is False


class _FakeSpec:
    def __init__(self, depends_on):
        self.depends_on = depends_on
