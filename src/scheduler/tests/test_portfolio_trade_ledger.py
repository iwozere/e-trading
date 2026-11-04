"""The scheduler's portfolio.trade_ledger branch: success flag + error drive the run status."""

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from src.portfolio.trade_ledger import runner
from src.portfolio.trade_ledger.runner import RunSummary
from src.scheduler.scheduler_service import SchedulerService, is_failed_job_result


def _run(monkeypatch: pytest.MonkeyPatch, summary: RunSummary) -> Dict[str, Any]:
    seen: List[Any] = []

    async def fake_run_once(cfg: Any) -> RunSummary:
        seen.append(cfg)
        return summary

    monkeypatch.setattr(runner, "run_once", fake_run_once)
    service = SchedulerService.__new__(SchedulerService)  # the branch uses no instance state
    schedule = SimpleNamespace(target="portfolio.trade_ledger", task_params={"config_path": "src/portfolio/trade_ledger/config/trade_ledger.yaml"})
    result = asyncio.run(service._execute_portfolio_job(schedule, None))  # type: ignore[arg-type]
    assert seen and seen[0].recipient_id == 2
    return result


def test_successful_run_is_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run(monkeypatch, RunSummary(month="2026-10", started_at="x", email_sent=True))

    assert result["success"] is True and "error" not in result
    assert result["summary"]["month"] == "2026-10"
    assert not is_failed_job_result(result)


def test_failed_run_is_marked_failed_with_error(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run(monkeypatch, RunSummary(month="2026-10", started_at="x", errors=["no statement", "email not queued"]))

    assert is_failed_job_result(result)
    assert result["error"] == "no statement; email not queued"
