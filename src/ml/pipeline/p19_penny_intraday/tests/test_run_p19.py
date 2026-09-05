"""Tests for the P19 CLI entry point (run_p19.py): --date validation and the
dependency-gating added to profile-structural/run-once/filings-poll."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.data.pipeline.dependency_status import DependencyStatus
from src.ml.pipeline.p19_penny_intraday import run_p19

_NOT_READY = [
    DependencyStatus(
        name="P19 Intraday Watchlist Build",
        registered=True,
        ran_today=False,
        succeeded=False,
        status=None,
        started_at=None,
        finished_at=None,
    )
]


def _scheduler_result(capsys) -> dict:
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("__SCHEDULER_RESULT__:"))
    return json.loads(line[len("__SCHEDULER_RESULT__:") :])


# ── --date validation ─────────────────────────────────────────────────────


def test_invalid_date_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_p19.py", "filings-poll", "--date", "not-a-date"])
    with pytest.raises(SystemExit) as exc:
        run_p19.main()
    assert exc.value.code != 0
    assert "--date must be YYYY-MM-DD" in capsys.readouterr().err


def test_valid_date_passes_validation(monkeypatch):
    """A well-formed --date must not raise before dispatch even reaches the
    dependency check (which itself defers here since nothing is mocked ready)."""
    monkeypatch.setattr(sys, "argv", ["run_p19.py", "filings-poll", "--date", "2026-06-29"])
    monkeypatch.setattr(run_p19, "require_dependencies_or_defer", lambda name: (False, _NOT_READY))
    assert run_p19.main() == 0  # no SystemExit from the date parse


# ── dependency gating ──────────────────────────────────────────────────────


def test_profile_structural_defers_when_watchlist_build_not_ready(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_p19.py", "profile-structural"])
    monkeypatch.setattr(run_p19, "require_dependencies_or_defer", lambda name: (False, _NOT_READY))

    def _fail_if_called(*a, **k):
        raise AssertionError("StructuralProfiler must not run while the dependency is deferred")

    monkeypatch.setattr(
        "src.ml.pipeline.p19_penny_intraday.structural.profiler.StructuralProfiler", _fail_if_called
    )

    assert run_p19.main() == 0
    result = _scheduler_result(capsys)
    assert result["deferred"] is True
    assert result["success"] is True
    assert result["dependency_status"][0]["name"] == "P19 Intraday Watchlist Build"


def test_run_once_shadow_defers_when_watchlist_build_not_ready(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_p19.py", "run-once", "--mode", "shadow"])
    monkeypatch.setattr(run_p19, "require_dependencies_or_defer", lambda name: (False, _NOT_READY))

    def _fail_if_called(*a, **k):
        raise AssertionError("ShadowLoop must not run while the dependency is deferred")

    monkeypatch.setattr("src.ml.pipeline.p19_penny_intraday.shadow_loop.ShadowLoop", _fail_if_called)

    assert run_p19.main() == 0
    result = _scheduler_result(capsys)
    assert result["deferred"] is True


def test_filings_poll_defers_when_watchlist_build_not_ready(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_p19.py", "filings-poll"])
    monkeypatch.setattr(run_p19, "require_dependencies_or_defer", lambda name: (False, _NOT_READY))

    def _fail_if_called(*a, **k):
        raise AssertionError("FilingsPoll must not run while the dependency is deferred")

    monkeypatch.setattr("src.ml.pipeline.p19_penny_intraday.filings_poll.FilingsPoll", _fail_if_called)

    assert run_p19.main() == 0
    result = _scheduler_result(capsys)
    assert result["deferred"] is True


def test_profile_structural_proceeds_when_dependency_ready(monkeypatch, capsys, tmp_path):
    """When the dependency check says ready, the subcommand must still run its
    normal path (regression guard against the gate swallowing the happy path)."""
    monkeypatch.setattr(sys, "argv", ["run_p19.py", "profile-structural"])
    monkeypatch.setattr(run_p19, "require_dependencies_or_defer", lambda name: (True, []))
    monkeypatch.setattr(
        "src.ml.pipeline.p19_penny_intraday.watchlist_builder.load_watchlist", lambda output_dir, date: []
    )

    class _FakeProfiler:
        def __init__(self, config):
            pass

        def refresh_watchlist(self, entries, force):
            return {}

    monkeypatch.setattr(
        "src.ml.pipeline.p19_penny_intraday.structural.profiler.StructuralProfiler", _FakeProfiler
    )

    assert run_p19.main() == 0
    result = _scheduler_result(capsys)
    assert result.get("count") == 0
    assert "deferred" not in result
