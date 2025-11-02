from __future__ import annotations
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from src.data.db.repos.repo_jobs import JobsRepository
from src.data.db.models.model_jobs import JobType, RunStatus
from src.data.db.repos.repo_users import UsersRepo

UTC = timezone.utc


def test_jobs_repo_basic_flow(db_session: Session):
    # need a user FK for schedules
    u = UsersRepo(db_session).ensure_user_for_telegram("5005", defaults_user={"email": "sched@example.com"})

    repo = JobsRepository(db_session)
    now = datetime.now(UTC)
    sched = repo.create_schedule({
        "user_id": u.id,
        "job_type": JobType.SCREENER.value,
        "name": "daily-screener",
        "cron": "0 12 * * *",
        "enabled": True,
        "next_run_at": now - timedelta(minutes=1),
        "metadata": None,
    })
    assert sched.id is not None

    # list pending schedules
    due = repo.get_pending_schedules(current_time=now)
    assert any(s.id == sched.id for s in due)

    # update next run
    next_time = now + timedelta(hours=1)
    assert repo.update_schedule_next_run(sched.id, next_time)

    # create a run
    run = repo.create_run({
        "user_id": u.id,
        "job_type": JobType.SCREENER.value,
        "job_id": sched.id,
        "scheduled_for": now,
        "status": RunStatus.PENDING.value,
        "result": None,
    })
    assert run.id is not None

    # claim the run
    claimed = repo.claim_run(run.id, worker_id="worker-1")
    assert claimed is not None
    assert claimed.status == RunStatus.RUNNING.value
