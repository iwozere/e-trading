"""
Tests for scheduler run-failure detection and notification conditions.

Guards two prod incidents (2026-07-07):
- Scripts exiting nonzero were logged ("Script exit code: 1") but the run was
  marked COMPLETED, hiding failures from run history.
- Event conditions like {"on": "failure", "channel": "telegram"} were rejected
  as "Invalid condition" because only threshold conditions were supported —
  and failure runs never reached the evaluator at all.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduler.scheduler_service import SchedulerService, is_failed_job_result


def _service() -> SchedulerService:
    """Bare instance — _check_condition/_condition_channels use no instance state."""
    return SchedulerService.__new__(SchedulerService)


# ── is_failed_job_result ────────────────────────────────────────────────────


def test_nonzero_exit_result_is_failed():
    assert is_failed_job_result({"success": False, "exit_code": 1})


def test_success_result_is_not_failed():
    assert not is_failed_job_result({"success": True, "exit_code": 0})


def test_results_without_success_key_are_not_failed():
    # Alert/screener/report job results carry no "success" key.
    assert not is_failed_job_result({"alerts_fired": 2})
    assert not is_failed_job_result(None)
    assert not is_failed_job_result("done")


# ── Event conditions ────────────────────────────────────────────────────────


def test_on_failure_fires_only_on_failure():
    svc = _service()
    cond = {"on": "failure", "channel": "telegram"}
    assert svc._check_condition(cond, {}, success=False)
    assert not svc._check_condition(cond, {}, success=True)


def test_on_success_fires_only_on_success():
    svc = _service()
    cond = {"on": "success"}
    assert svc._check_condition(cond, {}, success=True)
    assert not svc._check_condition(cond, {}, success=False)


def test_on_always_fires_either_way():
    svc = _service()
    assert svc._check_condition({"on": "always"}, {}, success=True)
    assert svc._check_condition({"on": "always"}, {}, success=False)


def test_unknown_event_does_not_fire():
    assert not _service()._check_condition({"on": "sometimes"}, {}, success=False)


# ── Threshold conditions (existing behavior preserved) ──────────────────────


def test_threshold_condition_still_works():
    svc = _service()
    cond = {"check_field": "vix_current", "operator": ">=", "threshold": 20}
    assert svc._check_condition(cond, {"vix_current": 25})
    assert not svc._check_condition(cond, {"vix_current": 15})


def test_malformed_condition_is_rejected():
    assert not _service()._check_condition({"channel": "telegram"}, {})


# ── Channel extraction ──────────────────────────────────────────────────────


def test_condition_channels_accepts_plural_and_singular():
    svc = _service()
    assert svc._condition_channels({"channels": ["email", "telegram"]}) == ["email", "telegram"]
    assert svc._condition_channels({"channel": "telegram"}) == ["telegram"]
    assert svc._condition_channels({"on": "failure"}) == []
