"""Unit tests for the trade-ledger schedule seeder (fake JobsService, no DB)."""

from types import SimpleNamespace
from typing import Any, List, Optional

from src.data.db.models.model_jobs import JobType, ScheduleCreate, ScheduleUpdate
from src.portfolio.trade_ledger.config import DEFAULT_CONFIG_PATH
from src.portfolio.trade_ledger.seed_schedule import DEFAULT_TARGET, seed


class _FakeJobs:
    def __init__(self, existing: Optional[List[Any]] = None) -> None:
        self.existing = existing or []
        self.created: List[ScheduleCreate] = []
        self.updated: List[ScheduleUpdate] = []

    def list_schedules(self, user_id: int) -> List[Any]:
        return self.existing

    def create_schedule(self, user_id: int, create: ScheduleCreate) -> Any:
        self.created.append(create)
        return SimpleNamespace(id=41)

    def update_schedule(self, schedule_id: int, update: ScheduleUpdate) -> Any:
        self.updated.append(update)
        return SimpleNamespace(id=schedule_id)


def test_seed_creates_monthly_alert_schedule() -> None:
    jobs = _FakeJobs()

    schedule_id = seed(DEFAULT_CONFIG_PATH, 1, "portfolio_trade_ledger", True, jobs_service=jobs)  # type: ignore[arg-type]

    assert schedule_id == 41
    (create,) = jobs.created
    assert create.target == DEFAULT_TARGET == "portfolio.trade_ledger"
    assert create.job_type == JobType.ALERT
    assert create.cron == "0 6 2 * *"
    assert create.task_params == {"config_path": DEFAULT_CONFIG_PATH}


def test_seed_updates_existing_schedule() -> None:
    jobs = _FakeJobs([SimpleNamespace(id=7, name="other"), SimpleNamespace(id=9, name="portfolio_trade_ledger")])

    schedule_id = seed(DEFAULT_CONFIG_PATH, 1, "portfolio_trade_ledger", False, jobs_service=jobs)  # type: ignore[arg-type]

    assert schedule_id == 9
    assert jobs.created == []
    (update,) = jobs.updated
    assert update.enabled is False and update.cron == "0 6 2 * *"
