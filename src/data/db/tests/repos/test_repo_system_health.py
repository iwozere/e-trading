from __future__ import annotations
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.data.db.repos.repo_system_health import SystemHealthRepository
from src.data.db.models.model_system_health import SystemHealthStatus


UTC = timezone.utc


def test_system_health_create_update_list(db_session: Session):
    repo = SystemHealthRepository(db_session)

    # create new record
    rec = repo.create_or_update_system_health({
        "system": "notification",
        "component": "telegram",
        "status": SystemHealthStatus.HEALTHY.value,
        "last_success": datetime.now(UTC),
    })
    assert rec.id is not None

    # fetch
    got = repo.get_system_health("notification", "telegram")
    assert got is not None

    # degrade
    updated = repo.update_system_status("notification", "telegram", SystemHealthStatus.DEGRADED, response_time_ms=250)
    assert updated.status == SystemHealthStatus.DEGRADED.value

    # list non-stale only
    recent = repo.list_system_health(include_stale=False, stale_threshold_minutes=60)
    assert any(r.system == "notification" and r.component == "telegram" for r in recent)
