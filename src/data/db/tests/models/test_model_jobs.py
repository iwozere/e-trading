import pytest
from datetime import datetime, timezone

from src.data.db.models.model_jobs import (
    Schedule,
    ScheduleRun,
    ScheduleCreate,
)


def test_schedule_pydantic_cron_validator():
    with pytest.raises(ValueError):
        ScheduleCreate(name="a", job_type="report", target="t", cron="* * *")


def test_schedule_db_insert_and_query(db_session):
    s = Schedule()
    s.user_id = 1
    s.name = "daily"
    s.job_type = "report"
    s.target = "do_report"
    s.task_params = {}
    s.cron = "0 0 * * *"
    db_session.add(s)
    db_session.flush()

    assert s.id is not None

    sr = ScheduleRun()
    sr.job_type = "report"
    sr.job_id = s.id
    sr.status = "PENDING"
    sr.scheduled_for = datetime.now(timezone.utc)
    db_session.add(sr)
    db_session.flush()

    assert sr.id is not None
